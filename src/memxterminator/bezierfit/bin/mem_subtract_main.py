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
    to_subtracted_stack_path_in_root,
    try_acquire_lock,
    write_json_atomic,
    write_mrc_atomic,
)
from memxterminator.path_resolve import infer_input_base_dir, normalise_dir, resolve_path


_WORKER_RUNNER = None
_EVENT_LOG_PATH: str | None = None


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
    global _EVENT_LOG_PATH
    output_root = config.get("output_root")
    if output_root:
        try:
            _EVENT_LOG_PATH = os.fspath(output_root) + os.sep + "bezfit_pms_run_data.log"
        except Exception:
            _EVENT_LOG_PATH = None
    else:
        _EVENT_LOG_PATH = "bezfit_pms_run_data.log"

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
        path = _EVENT_LOG_PATH or "bezfit_pms_run_data.log"
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
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
        input_base_dir: str | None = None,
        resume: bool = True,
        force: bool = False,
        adopt_existing_outputs: bool = False,
        skip_failed: bool = False,
        strict_output_check: bool = True,
        output_root: str | None = None,
    ):
        self.particle_cs = particle_cs
        self.template_cs = template_cs
        self.control_points_json = control_points_json

        self.resume = bool(resume)
        self.force = bool(force)
        self.adopt_existing_outputs = bool(adopt_existing_outputs)
        self.skip_failed = bool(skip_failed)
        self.strict_output_check = bool(strict_output_check)
        self.output_root = output_root

        if input_base_dir is not None and str(input_base_dir).strip() != "":
            self.input_base_dir = normalise_dir(input_base_dir)
        else:
            self.input_base_dir = infer_input_base_dir(self.particle_cs)

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

        raw_stacks = [str(x) for x in self._particle_filenames.tolist()]
        if not raw_stacks:
            raise ValueError(f"No particle stacks found in CryoSPARC dataset: {self.particle_cs!r} (blob/path is empty)")

        resolved_stacks: list[str] = []
        for raw in raw_stacks:
            raw_s = str(raw).strip()
            if raw_s == "":
                raise ValueError(f"Empty stack path found in CryoSPARC dataset: {self.particle_cs!r} (blob/path contains empty)")
            resolved_stacks.append(resolve_path(raw_s, base_dir=self.input_base_dir))

        # Keep a resolved, row-aligned copy for correct per-stack masking.
        self._particle_filenames_resolved = np.asarray(resolved_stacks).astype(str)

        unique_stacks = _remove_duplicates_preserve_order(resolved_stacks)
        self.particle_stack_paths = unique_stacks

        # Fail-fast on common misconfiguration: wrong base directory causing all relative paths to break.
        sample_n = min(3, len(self.particle_stack_paths))
        sample = self.particle_stack_paths[:sample_n]
        missing = [p for p in sample if not os.path.exists(p)]
        if missing:
            raise SystemExit(
                "ERROR: Cannot find particle stack file(s) referenced in CryoSPARC .cs.\n"
                f"  particle_cs: {self.particle_cs}\n"
                f"  inferred input_base_dir: {self.input_base_dir}\n"
                f"  example missing resolved path: {missing[0]}\n"
                "This usually means the paths stored inside the .cs file are relative (e.g. 'J220/extract/...')\n"
                "but the current working directory is not the CryoSPARC project root.\n\n"
                "Fix: re-run with an explicit base directory, for example:\n"
                "  --input_base_dir /path/to/cryosparc_project_root\n"
            )

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
        out_stack = to_subtracted_stack_path_in_root(raw_stack, output_root=self.output_root)
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
            mask = self._particle_filenames_resolved == raw_stack
            particle_idxes = self._particle_idx[mask]
            psis = self._psi[mask]
            pixel_sizes = self._pixel_size[mask]
            shifts = self._shift[mask]
            classes = self._class[mask]

            for particle_idx, psi, pixel_size, shift, class_ in zip(
                particle_idxes, psis, pixel_sizes, shifts, classes
            ):
                class_key = str(int(class_))
                if class_key not in self._control_points_dict:
                    raise ValueError(
                        f"Missing control points for class_id={class_key}. "
                        f"Available keys: {sorted(list(self._control_points_dict.keys()))[:10]}"
                    )
                control_points = np.asarray(self._control_points_dict[class_key], dtype=np.float32)
                if control_points.ndim != 2 or control_points.shape[1] != 2 or control_points.shape[0] < 4:
                    raise ValueError(
                        f"Invalid control points for class_id={class_key}: expected shape (N,2) with N>=4, "
                        f"got shape={control_points.shape}"
                    )
                if not np.isfinite(control_points).all():
                    raise ValueError(f"Control points contain NaN/Inf for class_id={class_key}")

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

            # Fail-fast: surface any pending async CUDA errors before writing to disk.
            cp.cuda.Stream.null.synchronize()

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
        "--input_base_dir",
        type=str,
        default=None,
        help=(
            "Base directory used to resolve relative paths stored inside CryoSPARC .cs files "
            "(e.g. 'J220/extract/...'). If omitted, auto-infer from the particle .cs path "
            "(CryoSPARC layout aware)."
        ),
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="If set, write all outputs under <output_root>/subtracted/... (isolates sweeps/batches).",
    )
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
    global _EVENT_LOG_PATH
    if args.output_root:
        try:
            _EVENT_LOG_PATH = os.fspath(args.output_root) + os.sep + "bezfit_pms_run_data.log"
        except Exception:
            _EVENT_LOG_PATH = None
    else:
        _EVENT_LOG_PATH = "bezfit_pms_run_data.log"

    runner = BezierfitParticleMembraneSubtract(
        particle_cs=args.particle,
        template_cs=args.template,
        control_points_json=args.control_points,
        points_step=args.points_step,
        physical_membrane_dist=args.physical_membrane_dist,
        input_base_dir=args.input_base_dir,
        resume=args.resume,
        force=args.force,
        adopt_existing_outputs=args.adopt_existing_outputs,
        skip_failed=args.skip_failed,
        output_root=args.output_root,
    )

    print(">>> Preparing Bezierfit Particle Membrane Subtraction dataset...")
    print(f">>> input_base_dir: {runner.input_base_dir}")
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
        "input_base_dir": runner.input_base_dir,
        "resume": bool(args.resume),
        "force": bool(args.force),
        "adopt_existing_outputs": bool(args.adopt_existing_outputs),
        "skip_failed": bool(args.skip_failed),
        "output_root": args.output_root,
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
