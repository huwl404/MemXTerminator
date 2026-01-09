import argparse
import json
import os
import socket
import time
import traceback
from datetime import datetime, timezone
from multiprocessing import Pool
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
    to_subtracted_stack_path,
    try_acquire_lock,
    write_json_atomic,
    write_mrc_atomic,
)

from ..lib._utils import *
from ..lib.mem_subtract import *
from setproctitle import setproctitle

def _derive_star_output_paths(input_star: str) -> tuple[str, str]:
    """
    Derive default STAR output paths from an input STAR path.

    Example:
        /path/particles_selected.star ->
            /path/particles_selected_subtracted.star
            /path/particles_selected_subtracted_completed.star
    """
    base, ext = os.path.splitext(input_star)
    if ext.lower() != ".star":
        # Conservative fallback: treat the whole path as the base.
        base = input_star
    return f"{base}_subtracted.star", f"{base}_subtracted_completed.star"


def _rewrite_particles_rln_image_name_to_subtracted(df_particles):
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
        stack_path_sub = to_subtracted_stack_path(stack_path)
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
    write_all: bool = True,
    write_completed: bool = True,
) -> None:
    """
    Write:
      - <input>_subtracted.star (all rows, rewritten `rlnImageName`)
      - <input>_subtracted_completed.star (filtered by `.mxt` success + params_hash)

    Preserves optics block (RELION 3.1+) when present. See Spec v1 §8.
    """
    if not write_all and not write_completed:
        return

    star_tables = read_star_any(input_star)
    particles_df = get_relion_particles_table(star_tables)

    df_all, mapped_paths_sub = _rewrite_particles_rln_image_name_to_subtracted(particles_df)

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
        with mrcfile.open(self.average2d_name) as mrc:
            self.average2ds = cp.asarray(mrc.data)
        with mrcfile.open(self.averaged_mem_name) as mrc:
            self.averaged_membranes = cp.asarray(mrc.data)
        with mrcfile.open(self.mem_mask_name) as mrc:
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
        output_obj: dict[str, object] = {"path": out_stack}
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
    def process_rawimage_stack(self, rawimage_stacks_name):
        setproctitle('MemXTerminator-radPMS')

        raw_stack = rawimage_stacks_name
        out_stack = to_subtracted_stack_path(raw_stack)
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
                return

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
                        return
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
                return

        run_id = f"{datetime.now(timezone.utc).isoformat()}-{os.getpid()}"
        if not try_acquire_lock(lock_path, run_id=run_id):
            print(f">>> LOCKED_SKIP {raw_stack} -> {out_stack}")
            self._append_event_log("INFO", "LOCKED_SKIP", raw_stack=raw_stack, out_stack=out_stack)
            return

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
                    return

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
                cp.cuda.Stream.null.synchronize()
                cp.cuda.MemoryPool().free_all_blocks()

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
            cp.cuda.MemoryPool().free_all_blocks()
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
        multiprocessing.set_start_method('spawn')

        def chunks(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i:i + n]
        
        print('>>> Preparing Radonfit Particle Membrane Subtraction dataset...')
        print(f'>>> Found {len(self.rawimage_stacks_name_lst)} raw particle stacks in total.')

        minibatches = list(chunks(self.rawimage_stacks_name_lst, batch_size))
        i = 1
        total_len = len(minibatches)
        for minibatch in minibatches:
            with Pool(num_cpus) as p:
                start_time = time.time()
                p.map(self.process_rawimage_stack, minibatch)
                end_time = time.time()
                print(f'>>> {i} / {total_len} minibatch finished.')
                print(f">>> {len(minibatch)} particle stacks took {end_time - start_time:.4f} seconds.")
                i += 1

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--particles_selected_filename', '-ps', type=str, default='J419/particles_selected.star')
    parser.add_argument('--membrane_analysis_filename', '-ms', type=str, default='J419/mem_analysis.star')
    parser.add_argument('--bias', '-b', type=float, default=0.05)
    parser.add_argument('--extra_mem_dist', type=float, default=20)
    parser.add_argument('--scaling_factor_start', type=float, default=0.1)
    parser.add_argument('--scaling_factor_end', type=float, default=1)
    parser.add_argument('--scaling_factor_step', type=float, default=0.01)
    parser.add_argument('--cpu', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=20)
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
        help="Write <input>_subtracted.star after processing (default: enabled).",
    )
    parser.add_argument(
        "--write_star_completed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write <input>_subtracted_completed.star after processing (default: enabled).",
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
    bias = args.bias
    extra_mem_dist = args.extra_mem_dist
    scaling_factor_start = args.scaling_factor_start
    scaling_factor_end = args.scaling_factor_end
    scaling_factor_step = args.scaling_factor_step
    cpus_num = args.cpu
    batch_size = args.batch_size

    membrane_subtract = MembraneSubtract(
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
    )
    membrane_subtract.membrane_subtract_multiprocess(num_cpus=cpus_num, batch_size=batch_size)

    if args.write_star or args.write_star_completed:
        out_all_default, out_completed_default = _derive_star_output_paths(particles_selected_filename)
        out_star_all = args.star_out_all or out_all_default
        out_star_completed = args.star_out_completed or out_completed_default
        write_radonfit_pms_star_outputs(
            input_star=particles_selected_filename,
            expected_params_hash=membrane_subtract.mxt_params_hash,
            out_star_all=out_star_all,
            out_star_completed=out_star_completed,
            write_all=args.write_star,
            write_completed=args.write_star_completed,
        )
