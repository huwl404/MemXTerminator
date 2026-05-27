from __future__ import annotations

import argparse
import json
import os
import socket
import time
import traceback
from datetime import datetime, timezone
import multiprocessing

import cupy as cp
import mrcfile
import numpy as np
import starfile

from memxterminator.mxt_state import (
    compute_params_hash,
    fingerprint_file,
    is_uptodate,
    read_mxt,
    release_lock,
    to_output_stack_path,
    try_acquire_lock,
    validate_output_dirname,
    write_json_atomic,
    write_mrc_atomic,
)

from ..lib._utils import *
from ..lib.mem_subtract import *
from setproctitle import setproctitle

_WORKER_RUNNER = None
_WORKER_DEVICE_ID = None


def _remove_duplicates_preserve_order(seq: list[str]) -> list[str]:
    return list(dict.fromkeys(seq))


def _init_worker(config: dict) -> None:
    """
    Per-process initializer for multiprocessing workers.

    We intentionally build the full runner inside each worker to avoid pickling
    a large in-memory object (CuPy arrays, pandas DataFrames) for every task.
    """
    global _WORKER_RUNNER
    global _WORKER_DEVICE_ID

    # Best-effort: pin each worker to a GPU in round-robin order if multiple
    # CUDA devices are visible. Users can still control visibility via
    # CUDA_VISIBLE_DEVICES.
    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
    except Exception:
        device_count = 0
    if device_count > 0:
        try:
            ident = getattr(multiprocessing.current_process(), "_identity", ())
            worker_rank = int(ident[0]) if ident else 1  # 1-based in multiprocessing pools
            device_id = (worker_rank - 1) % device_count
            cp.cuda.Device(device_id).use()
            _WORKER_DEVICE_ID = device_id
            print(
                f">>> WORKER_INIT pid={os.getpid()} worker_rank={worker_rank} "
                f"cuda_device={device_id} visible_cuda_devices={device_count}",
                flush=True,
            )
        except Exception as exc:
            print(f">>> WARNING: failed to pin worker pid={os.getpid()} to a CUDA device: {exc}", flush=True)
            _WORKER_DEVICE_ID = None

    _WORKER_RUNNER = MembraneSubtract(**config)


def _process_rawimage_stack_worker(stack_path: str) -> str:
    if _WORKER_RUNNER is None:
        raise RuntimeError("Worker runner not initialized")
    return _WORKER_RUNNER.process_rawimage_stack(stack_path)


def _process_stack_paths_with_pool(
    *,
    stack_paths: list[str],
    procs: int,
    worker_config: dict,
    batch_size: int,
) -> list[str]:
    """
    Keep GPU workers fed continuously and use `batch_size` only as a progress window.

    Older code submitted one minibatch and waited for every stack in that minibatch
    before submitting the next one. A single slow stack could leave other GPU
    workers idle. Continuous dispatch matches the worker-pool architecture better:
    `--procs` controls real concurrency; `--batch_size` controls progress cadence.
    """
    batch_size_int = int(batch_size)
    if batch_size_int <= 0:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    procs_int = int(procs)
    if procs_int <= 0:
        raise ValueError(f"procs must be >= 1 after auto-detection, got {procs}")

    outcomes: list[str] = []
    total = len(stack_paths)
    if total == 0:
        return outcomes
    completed = 0
    window_completed = 0
    window_started = time.time()
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=procs_int, initializer=_init_worker, initargs=(worker_config,)) as pool:
        for outcome in pool.imap_unordered(_process_rawimage_stack_worker, stack_paths, chunksize=1):
            outcomes.append(outcome)
            completed += 1
            window_completed += 1
            if completed % batch_size_int == 0 or completed == total:
                now = time.time()
                print(f">>> {completed} / {total} particle stacks finished.")
                print(f">>> Last {window_completed} completed stack(s) took {now - window_started:.4f} seconds.")
                window_completed = 0
                window_started = now
    return outcomes


def _get_raw_particle_stack_paths(particles_star: str) -> list[str]:
    """
    Return unique stack paths (preserving order) from a RELION particles STAR.
    """
    star_tables_particles = read_star_any(particles_star)
    df_star = get_relion_particles_table(star_tables_particles)
    parsed = [parse_relion_image_name_1based(x) for x in df_star["rlnImageName"].tolist()]
    raw_paths = [p for p, _ in parsed]
    return _remove_duplicates_preserve_order(raw_paths)


def _derive_star_output_paths(input_star: str, *, output_dirname: str = "subtracted") -> tuple[str, str]:
    """
    Derive default STAR output paths from an input STAR path.

    Example:
        /path/particles_selected.star ->
            /path/particles_selected_subtracted.star
            /path/particles_selected_subtracted_completed.star

        with output_dirname="run_01":
            /path/particles_selected_run_01.star
            /path/particles_selected_run_01_completed.star
    """
    base, ext = os.path.splitext(input_star)
    if ext.lower() != ".star":
        # Conservative fallback: treat the whole path as the base.
        base = input_star
    out_dirname = validate_output_dirname(output_dirname)
    return f"{base}_{out_dirname}.star", f"{base}_{out_dirname}_completed.star"


def _rewrite_particles_rln_image_name_to_subtracted(df_particles, *, output_dirname: str = "subtracted"):
    """
    Return (df_out, mapped_stack_paths_sub) where `rlnImageName` now points to subtracted stacks.
    """
    if "rlnImageName" not in df_particles.columns:
        raise KeyError("STAR particles table is missing required column: rlnImageName")

    image_names = df_particles["rlnImageName"].astype(str).tolist()
    mapped_paths_sub: list[str] = []
    mapped_rln: list[str] = []
    for image_ref in image_names:
        stack_path, idx_1based = parse_relion_image_name_1based(image_ref)
        stack_path_sub = to_output_stack_path(stack_path, output_dirname=output_dirname)
        mapped_paths_sub.append(stack_path_sub)
        mapped_rln.append(f"{int(idx_1based)}@{stack_path_sub}")

    df_out = df_particles.copy()
    df_out["rlnImageName"] = mapped_rln
    return df_out, mapped_paths_sub


def _stack_path_is_completed_for_hash(*, stack_path_sub: str, expected_params_hash: str) -> bool:
    """
    Check whether a given subtracted stack is "complete for this run's parameters" (Spec v1 §8.3).
    """
    if not os.path.exists(stack_path_sub):
        return False
    mxt_path = stack_path_sub + ".mxt"
    if not os.path.exists(mxt_path):
        return False

    try:
        mxt = read_mxt(mxt_path)
    except Exception:
        return False

    return bool(
        mxt.get("task") == "radonfit_particle_pms"
        and mxt.get("status") == "success"
        and mxt.get("params_hash") == expected_params_hash
    )


def write_radonfit_pms_star_outputs(
    *,
    input_star: str,
    expected_params_hash: str,
    out_star_all: str,
    out_star_completed: str,
    output_dirname: str = "subtracted",
    write_all: bool = True,
    write_completed: bool = True,
) -> None:
    """
    Write:
      - <input>_<output_dirname>.star (all rows, rewritten `rlnImageName`)
      - <input>_<output_dirname>_completed.star (filtered by `.mxt` success + params_hash)

    Preserves optics block (RELION 3.1+) when present. See Spec v1 §8.
    """
    if not write_all and not write_completed:
        return

    star_tables = read_star_any(input_star)
    particles_df = get_relion_particles_table(star_tables)

    out_dirname = validate_output_dirname(output_dirname)
    df_all, mapped_paths_sub = _rewrite_particles_rln_image_name_to_subtracted(
        particles_df, output_dirname=out_dirname
    )

    # Build completion mask (one `.mxt` read per unique stack path).
    unique_paths_sub = sorted(set(mapped_paths_sub))
    completed_paths_sub = {
        p for p in unique_paths_sub if _stack_path_is_completed_for_hash(stack_path_sub=p, expected_params_hash=expected_params_hash)
    }
    completed_mask = [p in completed_paths_sub for p in mapped_paths_sub]
    df_completed = df_all[completed_mask].copy()

    # Preserve block structure (single block vs. dict with optics/particles).
    is_single_block = len(star_tables) == 1 and "__single__" in star_tables
    if is_single_block:
        if write_all:
            parent = os.path.dirname(out_star_all)
            if parent:
                os.makedirs(parent, exist_ok=True)
            starfile.write(df_all, out_star_all, overwrite=True)
        if write_completed:
            parent = os.path.dirname(out_star_completed)
            if parent:
                os.makedirs(parent, exist_ok=True)
            starfile.write(df_completed, out_star_completed, overwrite=True)
        return

    particles_block_name = None
    for block_name, df in star_tables.items():
        if df is particles_df:
            particles_block_name = block_name
            break
    if particles_block_name is None:
        particles_block_name, _df, _col = find_star_column(star_tables, candidates=["rlnImageName"])

    if write_all:
        out_tables_all = dict(star_tables)
        out_tables_all[particles_block_name] = df_all
        parent = os.path.dirname(out_star_all)
        if parent:
            os.makedirs(parent, exist_ok=True)
        starfile.write(out_tables_all, out_star_all, overwrite=True)

    if write_completed:
        out_tables_completed = dict(star_tables)
        out_tables_completed[particles_block_name] = df_completed
        parent = os.path.dirname(out_star_completed)
        if parent:
            os.makedirs(parent, exist_ok=True)
        starfile.write(out_tables_completed, out_star_completed, overwrite=True)


class MembraneSubtract:
    def __init__(
        self,
        particles_selected_filename,
        membrane_analysis_filename,
        bias,
        extra_mem_dist,
        scaling_factor_start,
        scaling_factor_end,
        scaling_factor_step,
        *,
        resume: bool = True,
        force: bool = False,
        adopt_existing_outputs: bool = False,
        skip_failed: bool = False,
        strict_output_check: bool = True,
        output_dirname: str = "subtracted",
    ):
        # Keep paths for `.mxt` input fingerprints.
        self.particles_selected_filename = particles_selected_filename
        self.membrane_analysis_filename = membrane_analysis_filename

        # Resume / cache invalidation controls (Spec v1).
        self.resume = bool(resume)
        self.force = bool(force)
        self.adopt_existing_outputs = bool(adopt_existing_outputs)
        self.skip_failed = bool(skip_failed)
        self.strict_output_check = bool(strict_output_check)
        self.output_dirname = validate_output_dirname(output_dirname)

        # `.mxt` stable params/hash for this invocation (pixel-affecting only).
        self.mxt_task = "radonfit_particle_pms"
        self.mxt_params = {
            "bias": float(bias),
            "extra_mem_dist": float(extra_mem_dist),
            "scaling_factor_start": float(scaling_factor_start),
            "scaling_factor_end": float(scaling_factor_end),
            "scaling_factor_step": float(scaling_factor_step),
        }
        self.mxt_params_hash = compute_params_hash(self.mxt_params)
        self._host = socket.gethostname()
        try:
            import memxterminator  # noqa: WPS433 (local import by design)

            self._software_version = getattr(memxterminator, "__version__", "unknown")
        except Exception:
            self._software_version = "unknown"

        # Input fingerprints (raw_stack is per-item).
        self._particles_star_fp = fingerprint_file(self.particles_selected_filename)
        self._mem_analysis_star_fp = fingerprint_file(self.membrane_analysis_filename)

        star_tables_particles = read_star_any(particles_selected_filename)
        self.df_star = get_relion_particles_table(star_tables_particles)
        self.df_optics = get_relion_optics_table(star_tables_particles)

        # Pre-parse RELION image references and normalise origin shifts to pixels.
        # - rlnImageName: typically `N@path/to/stack.mrcs` (N is 1-based)
        # - rlnOriginX/Y: either in pixels, or in Å as rlnOriginXAngst/YAngst (RELION 3.1+)
        parsed = [parse_relion_image_name_1based(x) for x in self.df_star["rlnImageName"].tolist()]
        self.df_star = self.df_star.copy()
        self.df_star["_memx_stack_path"] = [p for p, _ in parsed]
        self.df_star["_memx_section_1based"] = [i for _, i in parsed]
        dx_pix, dy_pix = get_relion_origin_shifts_pixels(
            self.df_star, star_tables=star_tables_particles, optics_df=self.df_optics
        )
        self.df_star["_memx_origin_x_pix"] = dx_pix.astype(float)
        self.df_star["_memx_origin_y_pix"] = dy_pix.astype(float)

        # Membrane analysis STAR is generated by MemXTerminator and is expected to be a single table.
        df_mem_analysis_tables = read_star_any(membrane_analysis_filename)
        if len(df_mem_analysis_tables) == 1:
            self.df_mem_analysis = next(iter(df_mem_analysis_tables.values()))
        else:
            _, self.df_mem_analysis, _ = find_star_column(
                df_mem_analysis_tables, candidates=["rln2DAverageimageName"]
            )
        self.bias = float(bias)
        self.extra_mem_dist = float(extra_mem_dist)
        self.scaling_factor_start = float(scaling_factor_start)
        self.scaling_factor_end = float(scaling_factor_end)
        self.scaling_factor_step = float(scaling_factor_step)

        self.rawimage_stacks_name_lst = self.get_rawimage_stacks_name_lst()
        self.average2ds_dict, self.average2d_name, self.averaged_mem_name, self.mem_mask_name = self.get_2daverage_averaged_mask_stack_name()
        with mrcfile.open(self.average2d_name, permissive=True) as mrc:
            self.average2ds = cp.asarray(mrc.data)
        with mrcfile.open(self.averaged_mem_name, permissive=True) as mrc:
            self.averaged_membranes = cp.asarray(mrc.data)
        with mrcfile.open(self.mem_mask_name, permissive=True) as mrc:
            self.membrane_masks = cp.asarray(mrc.data)
        # self.average2ds = cp.asarray(mrcfile.open(self.average2d_name).data)
        # self.averaged_membranes = cp.asarray(mrcfile.open(self.averaged_mem_name).data)
        # self.membrane_masks = cp.asarray(mrcfile.open(self.mem_mask_name).data)
        
    def get_rawimage_stacks_name_lst(self):
        rawimage_lst = list(self.df_star["_memx_stack_path"])
        rawimage_name_lst = list(dict.fromkeys(rawimage_lst))
        # rawimage_name_lst = rawimage_name_lst[1:4000]
        return rawimage_name_lst
    def get_2daverage_averaged_mask_stack_name(self):
        average2d_lst = list(map(lambda x: int(x), list(self.df_mem_analysis['rln2DAverageimageName'].apply(lambda x: x.split('@')[0]))))
        average2ds_dict = dict(zip(average2d_lst, [i for i in range(len(average2d_lst))]))
        average2d_name = list(dict.fromkeys(list(self.df_mem_analysis['rln2DAverageimageName'].apply(lambda x: x.split('@')[1]))))[0]
        averaged_mem_name = list(dict.fromkeys(list(self.df_mem_analysis['rlnAveragedMembraneName'].apply(lambda x: x.split('@')[1]))))[0]
        mem_mask_name = list(dict.fromkeys(list(self.df_mem_analysis['rlnMembraneMaskName'].apply(lambda x: x.split('@')[1]))))[0]
        return average2ds_dict, average2d_name, averaged_mem_name, mem_mask_name
    def get_center_posi_theta_memdist_kappa(self, class_number):
        index = self.average2ds_dict[class_number]
        center_posi_theta_memdist_kappa = self.df_mem_analysis['rlnCenterX'][index], self.df_mem_analysis['rlnCenterY'][index], self.df_mem_analysis['rlnAngleTheta'][index], self.df_mem_analysis['rlnMembraneDistance'][index], self.df_mem_analysis['rlnCurveKappa'][index]
        return center_posi_theta_memdist_kappa
    def get_df_temp(self, rawimage_stacks_name):
        df_temp = self.df_star[self.df_star["_memx_stack_path"] == rawimage_stacks_name]
        return df_temp
    def get_class_number_lst(self, df_temp):
        class_number_lst = list(map(lambda x: int(x), list(df_temp['rlnClassNumber'])))
        return class_number_lst
    def get_df_temp_psi_dx_dy_class(self, df_temp):
        rawimage_sections_lst = list(map(int, list(df_temp["_memx_section_1based"])))
        psi_lst = list(df_temp['rlnAnglePsi'])
        dx_lst = list(df_temp["_memx_origin_x_pix"])
        dy_lst = list(df_temp["_memx_origin_y_pix"])
        rawimage_class_lst = self.get_class_number_lst(df_temp)
        zip_lst = list(zip(psi_lst, dx_lst, dy_lst, rawimage_class_lst))
        df_temp_dict_psi_dx_dy_class = dict(zip(rawimage_sections_lst, zip_lst))
        return df_temp_dict_psi_dx_dy_class
    def fill_nan_with_gaussian_noise(self, image):
        image_copy = image.copy()
        mean_val = cp.nanmean(image_copy)
        std_val = cp.nanstd(image_copy)
        nan_mask = cp.isnan(image_copy)
        noise = cp.random.normal(mean_val, std_val, image_copy.shape)
        image_copy[nan_mask] = noise[nan_mask]
        return image_copy

    def _append_event_log(self, level: str, event: str, **fields: object) -> None:
        """
        Append a single human-readable line to `radfit_pms_run_data.log`.

        This file is never read for resume; `.mxt` is authoritative (Spec v1).
        """
        ts = datetime.now(timezone.utc).isoformat()
        parts = [ts, level, event]
        for key, value in fields.items():
            try:
                encoded = json.dumps(value, ensure_ascii=False)
            except Exception:
                encoded = repr(value)
            parts.append(f"{key}={encoded}")

        line = " ".join(parts)
        try:
            with open("radfit_pms_run_data.log", "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            # Best-effort: logging must never crash workers.
            pass

    def _output_passes_sanity_check(self, out_stack: str) -> bool:
        """
        Minimal output sanity check for `--adopt_existing_outputs`.
        """
        try:
            with mrcfile.open(out_stack, permissive=True) as mrc:
                shape = getattr(mrc, "data", None).shape
            return bool(shape) and all(int(x) > 0 for x in shape)
        except Exception:
            return False

    def _build_mxt_object(
        self,
        *,
        status: str,
        out_stack: str,
        expected_inputs: dict,
        started_utc: str,
        finished_utc: str,
        duration_sec: float,
        adopted: bool,
        run_id: str,
        error=None,
    ) -> dict:
        output_obj: dict[str, object] = {"path": out_stack, "output_dirname": self.output_dirname}
        if os.path.exists(out_stack):
            try:
                output_obj["file"] = fingerprint_file(out_stack)
            except Exception:
                pass

        obj = {
            "mxt_schema": 1,
            "task": self.mxt_task,
            "status": status,
            "params": self.mxt_params,
            "params_hash": self.mxt_params_hash,
            "inputs": expected_inputs,
            "output": output_obj,
            "run": {
                "run_id": run_id,
                "started_utc": started_utc,
                "finished_utc": finished_utc,
                "duration_sec": float(duration_sec),
                "pid": int(os.getpid()),
                "host": self._host,
                "software_version": self._software_version,
                "git_commit": None,
            },
            "adopted": bool(adopted),
        }
        if error is not None:
            obj["error"] = error
        return obj
    def process_rawimage_stack(self, rawimage_stacks_name: str) -> str:
        setproctitle('MemXTerminator-radPMS')

        raw_stack = rawimage_stacks_name
        out_stack = to_output_stack_path(raw_stack, output_dirname=self.output_dirname)
        mxt_path = out_stack + ".mxt"
        lock_path = mxt_path + ".lock"

        expected_inputs = {
            "raw_stack": fingerprint_file(raw_stack),
            "particles_star": self._particles_star_fp,
            "mem_analysis_star": self._mem_analysis_star_fp,
        }

        # Fast-path: skip without locking.
        if (not self.force) and self.resume:
            uptodate, reason = is_uptodate(
                out_stack,
                mxt_path,
                self.mxt_task,
                self.mxt_params_hash,
                expected_inputs,
                strict_output_check=self.strict_output_check,
            )
            if uptodate:
                print(f">>> SKIP_UPTODATE {raw_stack} -> {out_stack}")
                self._append_event_log("INFO", "SKIP_UPTODATE", raw_stack=raw_stack, out_stack=out_stack)
                return "SKIP_UPTODATE"

            if self.skip_failed and os.path.exists(mxt_path):
                try:
                    mxt = read_mxt(mxt_path)
                    if (
                        mxt.get("task") == self.mxt_task
                        and mxt.get("status") == "failed"
                        and mxt.get("params_hash") == self.mxt_params_hash
                    ):
                        print(f">>> SKIP_FAILED {raw_stack} -> {out_stack}")
                        self._append_event_log("WARN", "SKIP_FAILED", raw_stack=raw_stack, out_stack=out_stack)
                        return "SKIP_FAILED"
                except Exception:
                    pass

            if (
                self.adopt_existing_outputs
                and reason in {"MISSING_MXT", "INVALID_MXT_JSON"}
                and os.path.exists(out_stack)
                and self._output_passes_sanity_check(out_stack)
            ):
                now = datetime.now(timezone.utc).isoformat()
                adopt_run_id = f"{now}-{os.getpid()}-adopt"
                obj = self._build_mxt_object(
                    status="success",
                    out_stack=out_stack,
                    expected_inputs=expected_inputs,
                    started_utc=now,
                    finished_utc=now,
                    duration_sec=0.0,
                    adopted=True,
                    run_id=adopt_run_id,
                    error=None,
                )
                write_json_atomic(mxt_path, obj)
                print(f">>> ADOPT_EXISTING_OUTPUT {raw_stack} -> {out_stack}")
                self._append_event_log(
                    "INFO",
                    "ADOPT_EXISTING_OUTPUT",
                    raw_stack=raw_stack,
                    out_stack=out_stack,
                    run_id=adopt_run_id,
                )
                return "ADOPT_EXISTING_OUTPUT"

        run_id = f"{datetime.now(timezone.utc).isoformat()}-{os.getpid()}"
        if not try_acquire_lock(lock_path, run_id=run_id):
            print(f">>> LOCKED_SKIP {raw_stack} -> {out_stack}")
            self._append_event_log("INFO", "LOCKED_SKIP", raw_stack=raw_stack, out_stack=out_stack)
            return "LOCKED_SKIP"

        started_wall = time.time()
        started_utc = datetime.now(timezone.utc).isoformat()
        self._append_event_log(
            "INFO",
            "START",
            raw_stack=raw_stack,
            out_stack=out_stack,
            run_id=run_id,
            params_hash=self.mxt_params_hash,
        )
        try:
            # Re-check under lock to avoid duplicate work when two processes race.
            if (not self.force) and self.resume:
                uptodate, _reason2 = is_uptodate(
                    out_stack,
                    mxt_path,
                    self.mxt_task,
                    self.mxt_params_hash,
                    expected_inputs,
                    strict_output_check=self.strict_output_check,
                )
                if uptodate:
                    print(f">>> SKIP_UPTODATE {raw_stack} -> {out_stack}")
                    self._append_event_log(
                        "INFO", "SKIP_UPTODATE", raw_stack=raw_stack, out_stack=out_stack, run_id=run_id
                    )
                    return "SKIP_UPTODATE"

            with mrcfile.open(raw_stack) as mrc:
                rawimages_stacks = cp.asarray(mrc.data)
            rawimages_stacks_subtracted = rawimages_stacks.copy()
            df_rawimage_temp = self.get_df_temp(raw_stack)
            df_rawimage_temp_dict_psi_dx_dy_class = self.get_df_temp_psi_dx_dy_class(df_rawimage_temp)
            for df_rawimage_temp_section_num, df_rawimage_temp_psi_dx_dy_class in df_rawimage_temp_dict_psi_dx_dy_class.items():
                try:
                    if rawimages_stacks.ndim == 2:
                        rawimage_temp = rawimages_stacks
                    else:
                        rawimage_temp = rawimages_stacks[df_rawimage_temp_section_num-1]
                    psi, dx, dy, class_number = df_rawimage_temp_psi_dx_dy_class
                    x0, y0, theta, memdist, kappa = self.get_center_posi_theta_memdist_kappa(class_number)
                    index = self.average2ds_dict[class_number]
                    average2d = self.average2ds[class_number-1]
                    averaged_membrane = self.averaged_membranes[index]
                    membrane_mask = self.membrane_masks[index]
                    get_to_raw = Get2Raw(self.membrane_analysis_filename, average2d, averaged_membrane, rawimage_temp, membrane_mask, x0, y0, theta, memdist, kappa, psi, dx, dy)
                    get_to_raw.rotate_average_to_raw()
                    subtracted_membrane = get_to_raw.raw_membrane_average_subtract(self.bias, self.extra_mem_dist, self.scaling_factor_start, self.scaling_factor_end, self.scaling_factor_step)
                    if cp.isnan(subtracted_membrane).any():
                        subtracted_membrane = self.fill_nan_with_gaussian_noise(subtracted_membrane)
                    # print(f'{df_rawimage_temp_section_num}@{rawimage_stacks_name} membrane subtracted')
                except Exception as e:
                    print(f"Error processing {df_rawimage_temp_section_num}@{raw_stack}: {e}")
                    raise e
                if rawimages_stacks_subtracted.ndim == 2:
                    rawimages_stacks_subtracted = subtracted_membrane
                else:
                    rawimages_stacks_subtracted[df_rawimage_temp_section_num-1] = subtracted_membrane
                del get_to_raw
                del subtracted_membrane
                del rawimage_temp
                del average2d
                del averaged_membrane
                del membrane_mask

            # Atomic output write + `.mxt` sidecar (Spec v1).
            write_mrc_atomic(out_stack, rawimages_stacks_subtracted)
            print(f'>>> {out_stack} saved')

            finished_utc = datetime.now(timezone.utc).isoformat()
            duration = time.time() - started_wall
            obj = self._build_mxt_object(
                status="success",
                out_stack=out_stack,
                expected_inputs=expected_inputs,
                started_utc=started_utc,
                finished_utc=finished_utc,
                duration_sec=duration,
                adopted=False,
                run_id=run_id,
                error=None,
            )
            write_json_atomic(mxt_path, obj)
            self._append_event_log(
                "INFO",
                "SUCCESS",
                raw_stack=raw_stack,
                out_stack=out_stack,
                run_id=run_id,
                duration_sec=duration,
            )

            del rawimages_stacks
            del rawimages_stacks_subtracted
            cp.cuda.Stream.null.synchronize()
            cp.get_default_memory_pool().free_all_blocks()
            return "SUCCESS"
        except Exception as exc:
            finished_utc = datetime.now(timezone.utc).isoformat()
            duration = time.time() - started_wall
            tb = traceback.format_exc()
            if len(tb) > 20_000:
                tb = tb[:20_000] + "\n... (truncated) ..."
            error = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": tb,
            }
            try:
                obj = self._build_mxt_object(
                    status="failed",
                    out_stack=out_stack,
                    expected_inputs=expected_inputs,
                    started_utc=started_utc,
                    finished_utc=finished_utc,
                    duration_sec=duration,
                    adopted=False,
                    run_id=run_id,
                    error=error,
                )
                write_json_atomic(mxt_path, obj)
            except Exception:
                pass
            self._append_event_log(
                "ERROR",
                "FAILED",
                raw_stack=raw_stack,
                out_stack=out_stack,
                run_id=run_id,
                error_type=type(exc).__name__,
            )
            raise
        finally:
            release_lock(lock_path)

    def membrane_subtract_multiprocess(self, num_cpus, batch_size):
        print('>>> Preparing Radonfit Particle Membrane Subtraction dataset...')
        print(f'>>> Found {len(self.rawimage_stacks_name_lst)} raw particle stacks in total.')
        num_cpus_int = int(num_cpus)
        batch_size_int = int(batch_size)
        if batch_size_int <= 0:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        try:
            device_count = int(cp.cuda.runtime.getDeviceCount())
        except Exception:
            device_count = 0
        if num_cpus_int <= 0:
            num_cpus_int = device_count if device_count > 0 else 1
        if device_count > 0 and num_cpus_int > device_count:
            print(
                f">>> WARNING: requested procs={num_cpus_int}, but only {device_count} CUDA device(s) are visible. "
                "Oversubscribing a single GPU with many processes is often slower."
            )

        if num_cpus_int <= 1:
            for stack_path in self.rawimage_stacks_name_lst:
                self.process_rawimage_stack(stack_path)
            return

        worker_config = {
            "particles_selected_filename": self.particles_selected_filename,
            "membrane_analysis_filename": self.membrane_analysis_filename,
            "bias": float(self.bias),
            "extra_mem_dist": float(self.extra_mem_dist),
            "scaling_factor_start": float(self.scaling_factor_start),
            "scaling_factor_end": float(self.scaling_factor_end),
            "scaling_factor_step": float(self.scaling_factor_step),
            "resume": bool(self.resume),
            "force": bool(self.force),
            "adopt_existing_outputs": bool(self.adopt_existing_outputs),
            "skip_failed": bool(self.skip_failed),
            "strict_output_check": bool(self.strict_output_check),
            "output_dirname": self.output_dirname,
        }

        _process_stack_paths_with_pool(
            stack_paths=self.rawimage_stacks_name_lst,
            procs=num_cpus_int,
            worker_config=worker_config,
            batch_size=batch_size_int,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--particles_selected_filename", "-ps", type=str, default="J419/particles_selected.star")
    parser.add_argument("--membrane_analysis_filename", "-ms", type=str, default="J419/mem_analysis.star")
    parser.add_argument("--bias", "-b", type=float, default=0.05)
    parser.add_argument("--extra_mem_dist", type=float, default=20)
    parser.add_argument("--scaling_factor_start", type=float, default=0.1)
    parser.add_argument("--scaling_factor_end", type=float, default=1)
    parser.add_argument("--scaling_factor_step", type=float, default=0.01)
    parser.add_argument(
        "--procs",
        "--cpu",
        dest="cpu",
        type=int,
        default=0,
        help=(
            "Number of GPU worker processes. Default 0 auto-detects visible CUDA devices. "
            "Use 1 for serial/single-GPU runs."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=20,
        help=(
            "Progress/reporting window for processed particle stacks. "
            "Actual parallel GPU workers are controlled by --procs."
        ),
    )
    parser.add_argument(
        "--output_dirname",
        type=str,
        default="subtracted",
        help=(
            "Name of the output directory replacing the input extract/ path component "
            "(default: subtracted). Use a single directory name, not a path."
        ),
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable resume via per-output .mxt sidecars (default: enabled).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force recompute even if .mxt indicates the output is up-to-date.",
    )
    parser.add_argument(
        "--adopt_existing_outputs",
        action="store_true",
        default=False,
        help="If output exists but .mxt is missing/invalid, write an adopted .mxt and skip recompute.",
    )
    parser.add_argument(
        "--skip_failed",
        action="store_true",
        default=False,
        help="Skip items whose existing .mxt has status=failed for the current params_hash.",
    )
    parser.add_argument(
        "--write_star",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write the full-size output STAR after processing (default: enabled).",
    )
    parser.add_argument(
        "--write_star_completed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write the completed-only output STAR after processing (default: enabled).",
    )
    parser.add_argument(
        "--star_out_all",
        type=str,
        default=None,
        help="Override output path for the full-size subtracted STAR (default: derived from input STAR).",
    )
    parser.add_argument(
        "--star_out_completed",
        type=str,
        default=None,
        help="Override output path for the completed-only STAR (default: derived from input STAR).",
    )

    args = parser.parse_args()
    particles_selected_filename = args.particles_selected_filename
    membrane_analysis_filename = args.membrane_analysis_filename
    bias = float(args.bias)
    extra_mem_dist = float(args.extra_mem_dist)
    scaling_factor_start = float(args.scaling_factor_start)
    scaling_factor_end = float(args.scaling_factor_end)
    scaling_factor_step = float(args.scaling_factor_step)
    batch_size = int(args.batch_size)
    if batch_size <= 0:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    output_dirname = validate_output_dirname(args.output_dirname)

    # Match MembraneSubtract.mxt_params_hash (Spec v1 §8.3).
    expected_params_hash = compute_params_hash(
        {
            "bias": float(bias),
            "extra_mem_dist": float(extra_mem_dist),
            "scaling_factor_start": float(scaling_factor_start),
            "scaling_factor_end": float(scaling_factor_end),
            "scaling_factor_step": float(scaling_factor_step),
        }
    )

    # Decide worker count (allow auto-detect with --procs 0).
    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
    except Exception:
        device_count = 0

    cpus_num = int(args.cpu)
    if cpus_num <= 0:
        cpus_num = device_count if device_count > 0 else 1
    if device_count > 0 and cpus_num > device_count:
        print(
            f">>> WARNING: requested --procs={cpus_num}, but only {device_count} CUDA device(s) are visible. "
            "Oversubscribing a single GPU with many processes is often slower."
        )
    print(f">>> CUDA devices visible: {device_count}; using {cpus_num} GPU worker process(es).")
    print(f">>> Output directory name: {output_dirname}")

    print(">>> Preparing Radonfit Particle Membrane Subtraction dataset...")
    stack_paths = _get_raw_particle_stack_paths(particles_selected_filename)
    print(f">>> Found {len(stack_paths)} raw particle stacks in total.")

    processing_failed = False
    outcomes: list[str] = []
    try:
        if cpus_num <= 1:
            runner = MembraneSubtract(
                particles_selected_filename,
                membrane_analysis_filename,
                bias,
                extra_mem_dist,
                scaling_factor_start,
                scaling_factor_end,
                scaling_factor_step,
                resume=args.resume,
                force=args.force,
                adopt_existing_outputs=args.adopt_existing_outputs,
                skip_failed=args.skip_failed,
                output_dirname=output_dirname,
            )
            for stack_path in stack_paths:
                outcomes.append(runner.process_rawimage_stack(stack_path))
        else:
            # Multiprocessing across particle stacks.
            worker_config = {
                "particles_selected_filename": particles_selected_filename,
                "membrane_analysis_filename": membrane_analysis_filename,
                "bias": float(bias),
                "extra_mem_dist": float(extra_mem_dist),
                "scaling_factor_start": float(scaling_factor_start),
                "scaling_factor_end": float(scaling_factor_end),
                "scaling_factor_step": float(scaling_factor_step),
                "resume": bool(args.resume),
                "force": bool(args.force),
                "adopt_existing_outputs": bool(args.adopt_existing_outputs),
                "skip_failed": bool(args.skip_failed),
                "strict_output_check": True,
                "output_dirname": output_dirname,
            }

            outcomes.extend(
                _process_stack_paths_with_pool(
                    stack_paths=stack_paths,
                    procs=int(cpus_num),
                    worker_config=worker_config,
                    batch_size=batch_size,
                )
            )

        succeeded = sum(1 for x in outcomes if x in {"SUCCESS", "SKIP_UPTODATE", "ADOPT_EXISTING_OUTPUT"})
        skipped = sum(1 for x in outcomes if x in {"LOCKED_SKIP", "SKIP_FAILED"})
        failed = len(stack_paths) - succeeded - skipped
        print(f">>> SUMMARY: Succeeded={succeeded} Failed={failed} Skipped={skipped} Total={len(stack_paths)}")
    except Exception:
        processing_failed = True
        raise
    finally:
        if args.write_star or args.write_star_completed:
            out_all_default, out_completed_default = _derive_star_output_paths(
                particles_selected_filename, output_dirname=output_dirname
            )
            out_star_all = args.star_out_all or out_all_default
            out_star_completed = args.star_out_completed or out_completed_default
            print(f">>> STAR outputs (for cryoSPARC import): {out_star_all}")
            print(f">>> STAR outputs (completed-only): {out_star_completed}")
            try:
                write_radonfit_pms_star_outputs(
                    input_star=particles_selected_filename,
                    expected_params_hash=expected_params_hash,
                    out_star_all=out_star_all,
                    out_star_completed=out_star_completed,
                    output_dirname=output_dirname,
                    write_all=args.write_star,
                    write_completed=args.write_star_completed,
                )
            except Exception as exc:
                print(f">>> WARNING: failed to write STAR outputs: {type(exc).__name__}: {exc}")
                if not processing_failed:
                    raise


if __name__ == "__main__":
    main()
