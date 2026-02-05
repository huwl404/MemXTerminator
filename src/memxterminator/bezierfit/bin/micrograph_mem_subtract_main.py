from __future__ import annotations

import argparse
import json
import os
import socket
import time
import traceback
from collections.abc import Mapping
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import mrcfile
import multiprocessing
import starfile
from setproctitle import setproctitle

from memxterminator.mxt_state import (
    compute_params_hash,
    fingerprint_file,
    is_uptodate,
    parse_relion_image_name_1based,
    read_mxt,
    release_lock,
    to_subtracted_micrograph_path,
    to_subtracted_stack_path_in_root,
    try_acquire_lock,
    write_json_atomic,
    write_mrc_atomic,
)
from memxterminator.path_resolve import infer_input_base_dir, normalise_dir, resolve_path

_WORKER_RUNNER = None
_EVENT_LOG_PATH: str | None = None


def _init_worker(config: dict) -> None:
    """
    Per-process initializer for multiprocessing workers.

    We build the runner inside each worker to avoid pickling large state (STAR
    DataFrames, cached weights/masks, etc.) for every task.
    """
    global _WORKER_RUNNER
    global _EVENT_LOG_PATH

    # Best-effort: pin each worker to a GPU in round-robin order if multiple
    # CUDA devices are visible. Users can still control visibility via
    # CUDA_VISIBLE_DEVICES.
    try:
        import cupy as cp

        device_count = int(cp.cuda.runtime.getDeviceCount())
    except Exception:
        device_count = 0
    if device_count > 0:
        try:
            ident = getattr(multiprocessing.current_process(), "_identity", ())
            worker_rank = int(ident[0]) if ident else 1  # 1-based in multiprocessing pools
            device_id = (worker_rank - 1) % device_count
            cp.cuda.Device(device_id).use()
        except Exception:
            pass

    output_root = config.get("output_root")
    if output_root:
        try:
            _EVENT_LOG_PATH = os.fspath(output_root) + os.sep + "mms_run_data.log"
        except Exception:
            _EVENT_LOG_PATH = None
    else:
        _EVENT_LOG_PATH = "mms_run_data.log"

    _WORKER_RUNNER = MicrographMembraneSubtract(**config)


def _process_micrograph_worker(raw_mg_name: tuple[str, str]) -> None:
    if _WORKER_RUNNER is None:
        raise RuntimeError("Worker runner not initialized")
    _WORKER_RUNNER.process_micrograph_mem_subtract(raw_mg_name)


def _append_event_log(level: str, event: str, **fields: object) -> None:
    """
    Append a single human-readable line to `mms_run_data.log`.

    This file is never read for resume; `.mxt` sidecars are authoritative (Spec v1).
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
        path = _EVENT_LOG_PATH or "mms_run_data.log"
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        # Best-effort: logging must never crash workers.
        pass


def _read_particles_table(star_path: str):
    """
    Read a STAR file and return the particle table DataFrame.

    Supports single-block and multi-block STAR files.
    """
    try:
        obj = starfile.read(star_path, always_dict=True)
    except TypeError:
        obj = starfile.read(star_path)

    # Import pandas lazily only for type checks; starfile already depends on it.
    import pandas as pd  # type: ignore

    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, Mapping):
        # Prefer a block that contains per-particle columns.
        for _name, df in obj.items():
            if isinstance(df, pd.DataFrame) and "rlnImageName" in df.columns:
                return df
        for _name, df in obj.items():
            if isinstance(df, pd.DataFrame):
                return df
    raise TypeError(f"Unsupported STAR content for {star_path!r}: {type(obj).__name__}")

class MicrographMembraneSubtract:
    def __init__(
        self,
        particles_selected_filename: str,
        *,
        input_base_dir: str | None = None,
        resume: bool = True,
        force: bool = False,
        adopt_existing_outputs: bool = False,
        skip_failed: bool = False,
        require_particle_mxt: bool = True,
        strict_output_check: bool = True,
        output_root: str | None = None,
    ):
        self.particles_selected_filename = particles_selected_filename

        if input_base_dir is not None and str(input_base_dir).strip() != "":
            self.input_base_dir = normalise_dir(input_base_dir)
        else:
            self.input_base_dir = infer_input_base_dir(self.particles_selected_filename)

        self.resume = bool(resume)
        self.force = bool(force)
        self.adopt_existing_outputs = bool(adopt_existing_outputs)
        self.skip_failed = bool(skip_failed)
        self.require_particle_mxt = bool(require_particle_mxt)
        self.strict_output_check = bool(strict_output_check)
        self.output_root = output_root

        # `.mxt` stable params/hash for this invocation (pixel-affecting only).
        self.mxt_task = "bezierfit_micrograph_mms"
        self.mxt_params = {
            "weighting": "gaussian",
            "gaussian_sigma": 0.6,
            "epsilon": 1e-10,
        }
        self.mxt_params_hash = compute_params_hash(self.mxt_params)

        self._host = socket.gethostname()
        try:
            import memxterminator  # noqa: WPS433 (local import by design)

            self._software_version = getattr(memxterminator, "__version__", "unknown")
        except Exception:
            self._software_version = "unknown"

        self._particles_star_fp = fingerprint_file(self.particles_selected_filename)

        self.df_star = _read_particles_table(self.particles_selected_filename)

        parsed = [parse_relion_image_name_1based(x) for x in self.df_star["rlnImageName"].tolist()]
        self.df_star = self.df_star.copy()
        self.df_star["_memx_stack_path"] = [resolve_path(p, base_dir=self.input_base_dir) for p, _ in parsed]
        self.df_star["_memx_section_1based"] = [i for _, i in parsed]

        self.rawimage_stacks_name_lst = self.get_rawimage_stacks_name_lst()
        self.raw_mg_name_lst = self.get_raw_stack_micrograph_pairs()

        # Fail-fast when the base dir is clearly wrong (common batch misconfiguration).
        if self.rawimage_stacks_name_lst and not any(os.path.exists(p) for p in self.rawimage_stacks_name_lst):
            raise SystemExit(
                "ERROR: None of the particle stacks referenced in the STAR file could be found on disk.\n"
                f"  particles_selected_filename: {self.particles_selected_filename}\n"
                f"  inferred input_base_dir: {self.input_base_dir}\n"
                "This usually means the STAR contains relative paths (e.g. 'J220/extract/...') and the base directory is wrong.\n\n"
                "Fix: re-run with an explicit base directory, for example:\n"
                "  --input_base_dir /path/to/cryosparc_project_root\n"
            )
        if self.raw_mg_name_lst and not any(os.path.exists(mg) for _stack, mg in self.raw_mg_name_lst):
            raise SystemExit(
                "ERROR: None of the micrographs referenced in the STAR file could be found on disk.\n"
                f"  particles_selected_filename: {self.particles_selected_filename}\n"
                f"  inferred input_base_dir: {self.input_base_dir}\n"
                "Fix: re-run with an explicit base directory, for example:\n"
                "  --input_base_dir /path/to/cryosparc_project_root\n"
            )

        self.rawimage_size: int | None = None
        self.weights = None
        self.mask = None

    def get_rawimage_stacks_name_lst(self):
        rawimage_lst = list(self.df_star["_memx_stack_path"])
        rawimage_name_lst = list(dict.fromkeys(rawimage_lst))
        return rawimage_name_lst

    def get_raw_stack_micrograph_pairs(self) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        for stack_path in self.rawimage_stacks_name_lst:
            df_temp = self.get_df_temp(stack_path)
            micrographs = list(dict.fromkeys(df_temp["rlnMicrographName"].astype(str).tolist()))
            if not micrographs:
                raise ValueError(f"Missing rlnMicrographName entries for stack: {stack_path}")
            if len(micrographs) != 1:
                raise ValueError(
                    f"Expected exactly 1 micrograph per particle stack, got {len(micrographs)} for {stack_path}: {micrographs[:3]}"
                )
            mg = str(micrographs[0]).strip()
            if mg == "":
                raise ValueError(f"Empty rlnMicrographName for stack: {stack_path}")
            pairs.append((stack_path, resolve_path(mg, base_dir=self.input_base_dir)))
        return pairs

    def get_df_temp(self, rawimage_stacks_name):
        df_temp = self.df_star[self.df_star["_memx_stack_path"] == rawimage_stacks_name]
        return df_temp
    def get_df_temp_X_Y(self, df_temp):
        rawimage_sections_lst = list(map(int, list(df_temp["_memx_section_1based"])))
        CoordinateX_lst = list(df_temp['rlnCoordinateX'])
        CoordinateY_lst = list(df_temp['rlnCoordinateY'])
        zip_lst = list(zip(CoordinateX_lst, CoordinateY_lst))
        df_temp_dict_X_Y = dict(zip(rawimage_sections_lst, zip_lst))
        return df_temp_dict_X_Y
    
    def fill_nan_with_gaussian_noise(self, image):
        import cupy as cp

        image_copy = image.copy()
        mean_val = cp.nanmean(image_copy)
        std_val = cp.nanstd(image_copy)
        nan_mask = cp.isnan(image_copy)
        noise = cp.random.normal(mean_val, std_val, image_copy.shape)
        image_copy[nan_mask] = noise[nan_mask]
        return image_copy

    def _ensure_weights_for_box(self, rawimage_size: int) -> None:
        import cupy as cp

        if self.rawimage_size == int(rawimage_size) and self.weights is not None and self.mask is not None:
            return
        self.rawimage_size = int(rawimage_size)
        sigma = float(self.mxt_params["gaussian_sigma"])
        x, y = cp.meshgrid(
            cp.linspace(-1, 1, self.rawimage_size),
            cp.linspace(-1, 1, self.rawimage_size),
        )
        d = cp.sqrt(x * x + y * y)
        self.weights = cp.exp(-(d**2) / (2.0 * sigma**2))
        self.mask = cp.ones((self.rawimage_size, self.rawimage_size))

    def _output_passes_sanity_check(self, out_path: str) -> bool:
        try:
            with mrcfile.open(out_path, permissive=True) as mrc:
                shape = getattr(mrc, "data", None).shape
            return bool(shape) and all(int(x) > 0 for x in shape)
        except Exception:
            return False

    def _build_mxt_object(
        self,
        *,
        status: str,
        out_micrograph: str,
        expected_inputs: dict,
        started_utc: str,
        finished_utc: str,
        duration_sec: float,
        adopted: bool,
        run_id: str,
        error=None,
    ) -> dict:
        output_obj: dict[str, object] = {"path": out_micrograph}
        if os.path.exists(out_micrograph):
            try:
                output_obj["file"] = fingerprint_file(out_micrograph)
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
    
    def process_micrograph_mem_subtract(self, raw_mg_name):
        setproctitle("MemXTerminator-MMS")
        rawimage_stacks_name, micrograph_name = raw_mg_name

        particle_stack_subtracted_path = to_subtracted_stack_path_in_root(
            rawimage_stacks_name, output_root=self.output_root
        )
        particle_stack_mxt_path = particle_stack_subtracted_path + ".mxt"

        if not os.path.exists(particle_stack_subtracted_path):
            print(f">>> BLOCKED_DEPENDENCY missing_stack={particle_stack_subtracted_path} micrograph={micrograph_name}")
            _append_event_log(
                "WARN",
                "BLOCKED_DEPENDENCY",
                raw_stack=rawimage_stacks_name,
                particle_stack_subtracted=particle_stack_subtracted_path,
                micrograph=micrograph_name,
                reason="MISSING_PARTICLE_STACK",
            )
            return

        if not os.path.exists(micrograph_name):
            print(f">>> BLOCKED_DEPENDENCY missing_micrograph={micrograph_name} stack={particle_stack_subtracted_path}")
            _append_event_log(
                "WARN",
                "BLOCKED_DEPENDENCY",
                raw_stack=rawimage_stacks_name,
                particle_stack_subtracted=particle_stack_subtracted_path,
                micrograph=micrograph_name,
                reason="MISSING_MICROGRAPH",
            )
            return

        particle_pms_params_hash = None
        if self.require_particle_mxt:
            if not os.path.exists(particle_stack_mxt_path):
                print(f">>> BLOCKED_DEPENDENCY missing_mxt={particle_stack_mxt_path} micrograph={micrograph_name}")
                _append_event_log(
                    "WARN",
                    "BLOCKED_DEPENDENCY",
                    raw_stack=rawimage_stacks_name,
                    particle_stack_subtracted=particle_stack_subtracted_path,
                    micrograph=micrograph_name,
                    reason="MISSING_PARTICLE_MXT",
                    particle_stack_mxt=particle_stack_mxt_path,
                )
                return
            try:
                dep = read_mxt(particle_stack_mxt_path)
            except Exception:
                print(f">>> BLOCKED_DEPENDENCY invalid_mxt={particle_stack_mxt_path} micrograph={micrograph_name}")
                _append_event_log(
                    "WARN",
                    "BLOCKED_DEPENDENCY",
                    raw_stack=rawimage_stacks_name,
                    particle_stack_subtracted=particle_stack_subtracted_path,
                    micrograph=micrograph_name,
                    reason="INVALID_PARTICLE_MXT",
                    particle_stack_mxt=particle_stack_mxt_path,
                )
                return

            if dep.get("task") != "bezierfit_particle_pms" or dep.get("status") != "success":
                print(f">>> BLOCKED_DEPENDENCY particle_mxt_not_success={particle_stack_mxt_path} micrograph={micrograph_name}")
                _append_event_log(
                    "WARN",
                    "BLOCKED_DEPENDENCY",
                    raw_stack=rawimage_stacks_name,
                    particle_stack_subtracted=particle_stack_subtracted_path,
                    micrograph=micrograph_name,
                    reason="PARTICLE_MXT_NOT_SUCCESS",
                    particle_stack_mxt=particle_stack_mxt_path,
                    particle_mxt_status=dep.get("status"),
                    particle_mxt_task=dep.get("task"),
                )
                return

            particle_pms_params_hash = dep.get("params_hash")
            if not isinstance(particle_pms_params_hash, str) or particle_pms_params_hash == "":
                print(f">>> BLOCKED_DEPENDENCY particle_mxt_missing_hash={particle_stack_mxt_path} micrograph={micrograph_name}")
                _append_event_log(
                    "WARN",
                    "BLOCKED_DEPENDENCY",
                    raw_stack=rawimage_stacks_name,
                    particle_stack_subtracted=particle_stack_subtracted_path,
                    micrograph=micrograph_name,
                    reason="PARTICLE_MXT_MISSING_HASH",
                    particle_stack_mxt=particle_stack_mxt_path,
                )
                return
        else:
            if os.path.exists(particle_stack_mxt_path):
                try:
                    dep = read_mxt(particle_stack_mxt_path)
                    if dep.get("status") == "success":
                        particle_pms_params_hash = dep.get("params_hash")
                except Exception:
                    particle_pms_params_hash = None

        out_micrograph = to_subtracted_micrograph_path(micrograph_name, output_root=self.output_root)
        mxt_path = out_micrograph + ".mxt"
        lock_path = mxt_path + ".lock"

        expected_inputs = {
            "micrograph_raw": fingerprint_file(micrograph_name),
            "particles_star": self._particles_star_fp,
            "particle_stack_subtracted": fingerprint_file(particle_stack_subtracted_path),
            "deps": {
                "particle_pms_params_hash": particle_pms_params_hash,
                "particle_stack_mxt_path": particle_stack_mxt_path,
            },
        }

        if (not self.force) and self.resume:
            uptodate, reason = is_uptodate(
                out_micrograph,
                mxt_path,
                self.mxt_task,
                self.mxt_params_hash,
                expected_inputs,
                strict_output_check=self.strict_output_check,
            )
            if uptodate:
                print(f">>> SKIP_UPTODATE {micrograph_name} -> {out_micrograph}")
                _append_event_log(
                    "INFO",
                    "SKIP_UPTODATE",
                    micrograph=micrograph_name,
                    out_micrograph=out_micrograph,
                    particle_stack_subtracted=particle_stack_subtracted_path,
                )
                return

            if self.skip_failed and os.path.exists(mxt_path):
                try:
                    mxt = read_mxt(mxt_path)
                    if (
                        mxt.get("task") == self.mxt_task
                        and mxt.get("status") == "failed"
                        and mxt.get("params_hash") == self.mxt_params_hash
                    ):
                        print(f">>> SKIP_FAILED {micrograph_name} -> {out_micrograph}")
                        _append_event_log(
                            "WARN",
                            "SKIP_FAILED",
                            micrograph=micrograph_name,
                            out_micrograph=out_micrograph,
                        )
                        return
                except Exception:
                    pass

            if (
                self.adopt_existing_outputs
                and reason in {"MISSING_MXT", "INVALID_MXT_JSON"}
                and os.path.exists(out_micrograph)
                and self._output_passes_sanity_check(out_micrograph)
            ):
                now = datetime.now(timezone.utc).isoformat()
                adopt_run_id = f"{now}-{os.getpid()}-adopt"
                obj = self._build_mxt_object(
                    status="success",
                    out_micrograph=out_micrograph,
                    expected_inputs=expected_inputs,
                    started_utc=now,
                    finished_utc=now,
                    duration_sec=0.0,
                    adopted=True,
                    run_id=adopt_run_id,
                    error=None,
                )
                write_json_atomic(mxt_path, obj)
                print(f">>> ADOPT_EXISTING_OUTPUT {micrograph_name} -> {out_micrograph}")
                _append_event_log(
                    "INFO",
                    "ADOPT_EXISTING_OUTPUT",
                    micrograph=micrograph_name,
                    out_micrograph=out_micrograph,
                    run_id=adopt_run_id,
                )
                return

        run_id = f"{datetime.now(timezone.utc).isoformat()}-{os.getpid()}"
        if not try_acquire_lock(lock_path, run_id=run_id):
            print(f">>> LOCKED_SKIP {micrograph_name} -> {out_micrograph}")
            _append_event_log(
                "INFO",
                "LOCKED_SKIP",
                micrograph=micrograph_name,
                out_micrograph=out_micrograph,
            )
            return

        started_wall = time.time()
        started_utc = datetime.now(timezone.utc).isoformat()
        _append_event_log(
            "INFO",
            "START",
            micrograph=micrograph_name,
            out_micrograph=out_micrograph,
            run_id=run_id,
            params_hash=self.mxt_params_hash,
            particle_stack_subtracted=particle_stack_subtracted_path,
        )

        try:
            import cupy as cp

            if (not self.force) and self.resume:
                uptodate, _reason2 = is_uptodate(
                    out_micrograph,
                    mxt_path,
                    self.mxt_task,
                    self.mxt_params_hash,
                    expected_inputs,
                    strict_output_check=self.strict_output_check,
                )
                if uptodate:
                    print(f">>> SKIP_UPTODATE {micrograph_name} -> {out_micrograph}")
                    _append_event_log(
                        "INFO",
                        "SKIP_UPTODATE",
                        micrograph=micrograph_name,
                        out_micrograph=out_micrograph,
                        run_id=run_id,
                    )
                    return

            with mrcfile.open(particle_stack_subtracted_path, permissive=True) as f:
                subtracted_images_stacks = cp.asarray(f.data)
            if subtracted_images_stacks.ndim == 2:
                subtracted_images_stacks = cp.expand_dims(subtracted_images_stacks, axis=0)

            self._ensure_weights_for_box(int(subtracted_images_stacks.shape[1]))

            for i in range(subtracted_images_stacks.shape[0]):
                subtracted_image = subtracted_images_stacks[i]
                if cp.isnan(subtracted_image).any():
                    subtracted_images_stacks[i] = self.fill_nan_with_gaussian_noise(subtracted_image)

            with mrcfile.open(micrograph_name, permissive=True) as f:
                micrograph = cp.asarray(f.data)

            micrograph_subtracted = micrograph.copy()
            mem_mosaic_image = cp.zeros_like(micrograph)
            weight_sum_image = cp.zeros_like(micrograph)
            mem_mosaic_image_mask = cp.zeros_like(micrograph)

            df_rawimage_temp = self.get_df_temp(rawimage_stacks_name)
            for df_rawimage_temp_section_num, df_rawimage_temp_X_Y in self.get_df_temp_X_Y(df_rawimage_temp).items():
                # STAR provides 1-based section indices.
                section_1based = int(df_rawimage_temp_section_num)
                idx0 = section_1based - 1
                if idx0 < 0 or idx0 >= int(subtracted_images_stacks.shape[0]):
                    raise ValueError(
                        f"STAR section index out of bounds: section_1based={section_1based} "
                        f"stack_len={int(subtracted_images_stacks.shape[0])} stack={rawimage_stacks_name}"
                    )

                subtracted_image_temp_full = subtracted_images_stacks[idx0]
                x_center, y_center = df_rawimage_temp_X_Y

                # Coordinates should be integer pixel indices; enforce int conversion to
                # avoid implicit float slicing errors.
                x_center = int(x_center)
                y_center = int(y_center)

                half = int(self.rawimage_size) // 2
                row0 = y_center - half
                col0 = x_center - half
                row1 = row0 + int(self.rawimage_size)
                col1 = col0 + int(self.rawimage_size)

                # Clamp box to micrograph bounds to avoid negative-index wraparound.
                mg_h, mg_w = int(micrograph.shape[0]), int(micrograph.shape[1])
                row0_c = max(row0, 0)
                col0_c = max(col0, 0)
                row1_c = min(row1, mg_h)
                col1_c = min(col1, mg_w)

                if row0_c >= row1_c or col0_c >= col1_c:
                    raise ValueError(
                        f"Zero-area micrograph box after clamping: center=({x_center},{y_center}) "
                        f"box={int(self.rawimage_size)} micrograph_shape={micrograph.shape} "
                        f"clamped_rows=({row0_c},{row1_c}) clamped_cols=({col0_c},{col1_c})"
                    )

                # Corresponding slice in the particle/subtracted-image box.
                pr0 = row0_c - row0
                pc0 = col0_c - col0
                pr1 = pr0 + (row1_c - row0_c)
                pc1 = pc0 + (col1_c - col0_c)

                particle_be_replaced = micrograph[row0_c:row1_c, col0_c:col1_c]
                sub_patch = -subtracted_image_temp_full[pr0:pr1, pc0:pc1]

                # Match per-particle contrast/scale to the micrograph crop (baseline behavior).
                sub_patch = (
                    (sub_patch - cp.mean(sub_patch))
                    / cp.std(sub_patch)
                    * cp.std(particle_be_replaced)
                    + cp.mean(particle_be_replaced)
                )

                weights_patch = self.weights[pr0:pr1, pc0:pc1]
                mask_patch = self.mask[pr0:pr1, pc0:pc1]

                mem_mosaic_image[row0_c:row1_c, col0_c:col1_c] += sub_patch * weights_patch
                weight_sum_image[row0_c:row1_c, col0_c:col1_c] += weights_patch
                mem_mosaic_image_mask[row0_c:row1_c, col0_c:col1_c] = mask_patch

            epsilon = float(self.mxt_params["epsilon"])
            weight_sum_image = cp.where(weight_sum_image == 0, epsilon, weight_sum_image)
            mem_mosaic_image /= weight_sum_image
            micrograph_subtracted = micrograph_subtracted * (1 - mem_mosaic_image_mask) + mem_mosaic_image * mem_mosaic_image_mask

            write_mrc_atomic(out_micrograph, micrograph_subtracted)
            print(f">>> {out_micrograph} finished")

            finished_utc = datetime.now(timezone.utc).isoformat()
            duration = time.time() - started_wall
            obj = self._build_mxt_object(
                status="success",
                out_micrograph=out_micrograph,
                expected_inputs=expected_inputs,
                started_utc=started_utc,
                finished_utc=finished_utc,
                duration_sec=duration,
                adopted=False,
                run_id=run_id,
                error=None,
            )
            write_json_atomic(mxt_path, obj)
            _append_event_log(
                "INFO",
                "SUCCESS",
                micrograph=micrograph_name,
                out_micrograph=out_micrograph,
                run_id=run_id,
                duration_sec=duration,
            )

            del micrograph_subtracted
            del subtracted_images_stacks
            del mem_mosaic_image
            del weight_sum_image
            del mem_mosaic_image_mask
            del df_rawimage_temp
            cp.cuda.Stream.null.synchronize()
            cp.get_default_memory_pool().free_all_blocks()
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
                    out_micrograph=out_micrograph,
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
            _append_event_log(
                "ERROR",
                "FAILED",
                micrograph=micrograph_name,
                out_micrograph=out_micrograph,
                run_id=run_id,
                error_type=type(exc).__name__,
            )
            raise
        finally:
            release_lock(lock_path)
    
    def micrograph_mem_subtract_multiprocessing(self, num_cpus, batch_size):
        try:
            multiprocessing.set_start_method("spawn")
        except RuntimeError:
            # Start method can only be set once per interpreter.
            pass

        def chunks(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i : i + n]

        print(">>> Preparing Micrograph Membrane Subtraction dataset...")
        print(f">>> input_base_dir: {self.input_base_dir}")
        print(f">>> Found {len(self.raw_mg_name_lst)} raw micrographs in total.")

        worker_config = {
            "particles_selected_filename": self.particles_selected_filename,
            "input_base_dir": self.input_base_dir,
            "resume": bool(self.resume),
            "force": bool(self.force),
            "adopt_existing_outputs": bool(self.adopt_existing_outputs),
            "skip_failed": bool(self.skip_failed),
            "require_particle_mxt": bool(self.require_particle_mxt),
            "strict_output_check": bool(self.strict_output_check),
            "output_root": self.output_root,
        }

        minibatches = list(chunks(self.raw_mg_name_lst, int(batch_size)))
        total_len = len(minibatches)
        with Pool(processes=int(num_cpus), initializer=_init_worker, initargs=(worker_config,)) as p:
            for i, minibatch in enumerate(minibatches, start=1):
                start_time = time.time()
                p.map(_process_micrograph_worker, minibatch, chunksize=1)
                end_time = time.time()
                print(f">>> {i} / {total_len} minibatch finished.")
                print(f">>> {len(minibatch)} micrograph stacks took {end_time - start_time:.4f} seconds.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Micrograph membrane subtraction')
    parser.add_argument(
        '--particles_selected_filename',
        '--particle',
        '-ps',
        type=str,
        help='particles_selected.star',
    )
    parser.add_argument(
        "--procs",
        type=int,
        default=0,
        help=(
            "Number of worker processes to run micrograph stacks in parallel. "
            "Use 0 to auto-detect from visible GPUs. (default: 0)"
        ),
    )
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument(
        "--input_base_dir",
        type=str,
        default=None,
        help=(
            "Base directory used to resolve relative paths stored inside STAR files "
            "(e.g. 'J220/extract/...'). If omitted, auto-infer from the STAR path "
            "(CryoSPARC layout aware)."
        ),
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="If set, read/write all outputs under <output_root>/subtracted/... (isolates sweeps/batches).",
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
        "--require_particle_mxt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require dependency particle-stack .mxt sidecars to exist and be success (default: enabled).",
    )
    args = parser.parse_args()
    if args.output_root:
        try:
            _EVENT_LOG_PATH = os.fspath(args.output_root) + os.sep + "mms_run_data.log"
        except Exception:
            _EVENT_LOG_PATH = None
    else:
        _EVENT_LOG_PATH = "mms_run_data.log"

    mms = MicrographMembraneSubtract(
        args.particles_selected_filename,
        input_base_dir=args.input_base_dir,
        resume=args.resume,
        force=args.force,
        adopt_existing_outputs=args.adopt_existing_outputs,
        skip_failed=args.skip_failed,
        require_particle_mxt=args.require_particle_mxt,
        output_root=args.output_root,
    )

    procs = int(args.procs)
    if procs <= 0:
        try:
            import cupy as _cp

            procs = int(_cp.cuda.runtime.getDeviceCount())
        except Exception:
            procs = 1

    mms.micrograph_mem_subtract_multiprocessing(procs, args.batch_size)
