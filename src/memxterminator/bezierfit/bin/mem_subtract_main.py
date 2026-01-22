from __future__ import annotations

import argparse
import json
import os
import socket
import time
import traceback
from datetime import datetime, timezone
from multiprocessing import Pool
import multiprocessing

import mrcfile
import numpy as np
from cryosparc.dataset import Dataset
from setproctitle import setproctitle

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


_WORKER_RUNNER = None


def _chunks(lst: list[str], n: int):
    if n <= 0:
        raise ValueError(f"batch_size must be >= 1, got {n}")
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def _init_worker(config: dict) -> None:
    """
    Per-process initializer for multiprocessing workers.

    We intentionally build the full runner inside each worker to avoid pickling
    a large in-memory object (CryoSPARC dataset arrays) for every task.
    """
    global _WORKER_RUNNER
    _WORKER_RUNNER = BezierfitParticleMembraneSubtract(**config)

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


def _process_particle_stack_worker(stack_path: str) -> None:
    if _WORKER_RUNNER is None:
        raise RuntimeError("Worker runner not initialized")
    _WORKER_RUNNER.process_particle_stack(stack_path)


def _append_event_log(level: str, event: str, **fields: object) -> None:
    """
    Append a single human-readable line to `bezfit_pms_run_data.log`.

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
        with open("bezfit_pms_run_data.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        # Best-effort: logging must never crash workers.
        pass


def fill_nan_with_gaussian_noise(image):
    import cupy as cp

    image_copy = image.copy()
    mean_val = cp.nanmean(image_copy)
    std_val = cp.nanstd(image_copy)
    nan_mask = cp.isnan(image_copy)
    noise = cp.random.normal(mean_val, std_val, image_copy.shape)
    image_copy[nan_mask] = noise[nan_mask]
    return image_copy


def _remove_duplicates_preserve_order(seq: list[str]) -> list[str]:
    return list(dict.fromkeys(seq))


class BezierfitParticleMembraneSubtract:
    def __init__(
        self,
        *,
        particle_cs: str,
        template_cs: str,
        control_points_json: str,
        points_step: float,
        physical_membrane_dist: int,
        resume: bool = True,
        force: bool = False,
        adopt_existing_outputs: bool = False,
        skip_failed: bool = False,
        strict_output_check: bool = True,
    ):
        self.particle_cs = particle_cs
        self.template_cs = template_cs
        self.control_points_json = control_points_json

        self.resume = bool(resume)
        self.force = bool(force)
        self.adopt_existing_outputs = bool(adopt_existing_outputs)
        self.skip_failed = bool(skip_failed)
        self.strict_output_check = bool(strict_output_check)

        self.mxt_task = "bezierfit_particle_pms"
        self.mxt_params = {
            "points_step": float(points_step),
            "physical_membrane_dist": int(physical_membrane_dist),
        }
        self.mxt_params_hash = compute_params_hash(self.mxt_params)

        self._host = socket.gethostname()
        try:
            import memxterminator  # noqa: WPS433 (local import by design)

            self._software_version = getattr(memxterminator, "__version__", "unknown")
        except Exception:
            self._software_version = "unknown"

        self._particles_cs_fp = fingerprint_file(self.particle_cs)
        self._templates_cs_fp = fingerprint_file(self.template_cs)
        self._control_points_fp = fingerprint_file(self.control_points_json)

        particle_dset = Dataset.load(self.particle_cs)
        # Ensure stable string dtype even if CryoSPARC returns bytes-like objects.
        self._particle_filenames = np.asarray(particle_dset["blob/path"]).astype(str)
        self._particle_idx = np.asarray(particle_dset["blob/idx"])
        self._psi = np.asarray(particle_dset["alignments2D/pose"])
        self._pixel_size = np.asarray(particle_dset["blob/psize_A"])
        self._shift = np.asarray(particle_dset["alignments2D/shift"])
        self._class = np.asarray(particle_dset["alignments2D/class"])

        with open(self.control_points_json, "r", encoding="utf-8") as f:
            self._control_points_dict = json.load(f)

        unique_stacks = _remove_duplicates_preserve_order(self._particle_filenames.tolist())
        self.particle_stack_paths = unique_stacks

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

    def process_particle_stack(self, particle_filename: str) -> None:
        setproctitle("MemXTerminator-bezPMS")

        raw_stack = particle_filename
        out_stack = to_subtracted_stack_path(raw_stack)
        mxt_path = out_stack + ".mxt"
        lock_path = mxt_path + ".lock"

        expected_inputs = {
            "raw_stack": fingerprint_file(raw_stack),
            "particles_cs": self._particles_cs_fp,
            "templates_cs": self._templates_cs_fp,
            "control_points_json": self._control_points_fp,
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
                _append_event_log("INFO", "SKIP_UPTODATE", raw_stack=raw_stack, out_stack=out_stack)
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
                        _append_event_log("WARN", "SKIP_FAILED", raw_stack=raw_stack, out_stack=out_stack)
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
                _append_event_log(
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
            _append_event_log("INFO", "LOCKED_SKIP", raw_stack=raw_stack, out_stack=out_stack)
            return

        started_wall = time.time()
        started_utc = datetime.now(timezone.utc).isoformat()
        _append_event_log(
            "INFO",
            "START",
            raw_stack=raw_stack,
            out_stack=out_stack,
            run_id=run_id,
            params_hash=self.mxt_params_hash,
        )
        try:
            import cupy as cp
            from ..lib.subtraction import MembraneSubtract

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
                    _append_event_log("INFO", "SKIP_UPTODATE", raw_stack=raw_stack, out_stack=out_stack, run_id=run_id)
                    return

            with mrcfile.open(raw_stack, permissive=True) as mrc:
                particle_stack = cp.asarray(mrc.data)

            subtracted_particle_stack = particle_stack.copy()
            mask = self._particle_filenames == raw_stack
            particle_idxes = self._particle_idx[mask]
            psis = self._psi[mask]
            pixel_sizes = self._pixel_size[mask]
            shifts = self._shift[mask]
            classes = self._class[mask]

            for particle_idx, psi, pixel_size, shift, class_ in zip(
                particle_idxes, psis, pixel_sizes, shifts, classes
            ):
                class_key = str(int(class_))
                control_points = np.array(self._control_points_dict[class_key])

                if particle_stack.ndim == 2:
                    subtractor = MembraneSubtract(
                        control_points,
                        particle_stack,
                        psi,
                        shift[0],
                        shift[1],
                        pixel_size,
                        self.mxt_params["points_step"],
                        self.mxt_params["physical_membrane_dist"],
                    )
                    subtracted_particle_stack = subtractor.mem_subtract()
                    if cp.isnan(subtracted_particle_stack).any():
                        subtracted_particle_stack = fill_nan_with_gaussian_noise(subtracted_particle_stack)
                elif particle_stack.ndim == 3:
                    idx = int(particle_idx)
                    subtractor = MembraneSubtract(
                        control_points,
                        particle_stack[idx],
                        psi,
                        shift[0],
                        shift[1],
                        pixel_size,
                        self.mxt_params["points_step"],
                        self.mxt_params["physical_membrane_dist"],
                    )
                    subtracted_particle = subtractor.mem_subtract()
                    if cp.isnan(subtracted_particle).any():
                        subtracted_particle = fill_nan_with_gaussian_noise(subtracted_particle)
                    subtracted_particle_stack[idx] = subtracted_particle
                else:
                    raise ValueError(f"Unsupported particle stack ndim={particle_stack.ndim} for {raw_stack}")

                del subtractor
                cp.cuda.Stream.null.synchronize()
                cp.get_default_memory_pool().free_all_blocks()

            # Atomic output write + `.mxt` sidecar (Spec v1).
            write_mrc_atomic(out_stack, subtracted_particle_stack)
            print(f">>> {out_stack} saved")

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
            _append_event_log(
                "INFO",
                "SUCCESS",
                raw_stack=raw_stack,
                out_stack=out_stack,
                run_id=run_id,
                duration_sec=duration,
            )

            del subtracted_particle_stack
            del particle_stack
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
            _append_event_log(
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--particle", type=str, default="./particles_selected.cs")
    parser.add_argument("--template", type=str, default="./templates_selected.cs")
    parser.add_argument("--control_points", type=str, default="./control_points.json")
    parser.add_argument("--points_step", type=float, default=0.005)
    parser.add_argument("--physical_membrane_dist", type=int, default=35)
    parser.add_argument(
        "--procs",
        type=int,
        default=1,
        help=(
            "Number of worker processes to run particle stacks in parallel. "
            "Use 0 to auto-detect from visible GPUs. "
            "(default: 1 = serial)"
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=20,
        help="How many particle stacks to process per minibatch when --procs > 1. (default: 20)",
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

    args = parser.parse_args()

    runner = BezierfitParticleMembraneSubtract(
        particle_cs=args.particle,
        template_cs=args.template,
        control_points_json=args.control_points,
        points_step=args.points_step,
        physical_membrane_dist=args.physical_membrane_dist,
        resume=args.resume,
        force=args.force,
        adopt_existing_outputs=args.adopt_existing_outputs,
        skip_failed=args.skip_failed,
    )

    print(">>> Preparing Bezierfit Particle Membrane Subtraction dataset...")
    print(f">>> Found {len(runner.particle_stack_paths)} raw particle stacks in total.")
    procs = int(args.procs)
    if procs <= 0:
        try:
            import cupy as _cp

            procs = int(_cp.cuda.runtime.getDeviceCount())
        except Exception:
            procs = 1

    if procs <= 1:
        for stack_path in runner.particle_stack_paths:
            runner.process_particle_stack(stack_path)
        return

    # Multiprocessing across particle stacks.
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        # Start method can only be set once per interpreter.
        pass

    num_cpus = int(procs)
    batch_size = int(args.batch_size)
    stack_paths = runner.particle_stack_paths

    worker_config = {
        "particle_cs": args.particle,
        "template_cs": args.template,
        "control_points_json": args.control_points,
        "points_step": float(args.points_step),
        "physical_membrane_dist": int(args.physical_membrane_dist),
        "resume": bool(args.resume),
        "force": bool(args.force),
        "adopt_existing_outputs": bool(args.adopt_existing_outputs),
        "skip_failed": bool(args.skip_failed),
    }

    minibatches = list(_chunks(stack_paths, batch_size))
    total_batches = len(minibatches)
    with Pool(processes=num_cpus, initializer=_init_worker, initargs=(worker_config,)) as pool:
        for i, minibatch in enumerate(minibatches, start=1):
            start_time = time.time()
            pool.map(_process_particle_stack_worker, minibatch, chunksize=1)
            end_time = time.time()
            print(f">>> {i} / {total_batches} minibatch finished.")
            print(f">>> {len(minibatch)} particle stacks took {end_time - start_time:.4f} seconds.")


if __name__ == "__main__":
    main()
