from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .local_scheduler import run_jobs
from .spec import JobSpecFile, SchedulerSpec, load_spec_file, parse_gpu_list


def _detect_gpus_auto() -> list[int]:
    try:
        import cupy as cp
    except Exception as exc:
        raise SystemExit(f"GPU auto-detect requested but CuPy import failed: {type(exc).__name__}: {exc}")

    try:
        count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:
        raise SystemExit(f"GPU auto-detect requested but CUDA runtime failed: {type(exc).__name__}: {exc}")

    if count <= 0:
        raise SystemExit("GPU auto-detect requested but no CUDA devices are visible.")

    return list(range(count))


def _apply_overrides(spec: SchedulerSpec, *, gpus: list[int] | None, policy: str | None, max_running_jobs: int | None) -> SchedulerSpec:
    return SchedulerSpec(
        gpus=list(gpus if gpus is not None else spec.gpus),
        policy=str(policy or spec.policy),  # type: ignore[arg-type]
        max_running_jobs=int(max_running_jobs if max_running_jobs is not None else spec.max_running_jobs),
        fail_fast=True,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="MemXTerminator Bezierfit batch scheduler (single-node).")
    parser.add_argument("--spec", type=str, required=True, help="Path to a batch spec JSON file.")
    parser.add_argument(
        "--state",
        type=str,
        default=None,
        help="Path to write scheduler_state.json (default: <spec_dir>/scheduler_state.json).",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Override GPU list, e.g. '0,1,2' or 'auto'. (default: from spec)",
    )
    parser.add_argument(
        "--policy",
        type=str,
        choices=["fill_first", "round_robin"],
        default=None,
        help="Override scheduling policy (default: from spec).",
    )
    parser.add_argument(
        "--max_running_jobs",
        type=int,
        default=None,
        help="Override max concurrent running jobs (default: from spec).",
    )

    args = parser.parse_args(argv)

    spec_path = Path(args.spec).resolve()
    if not spec_path.exists():
        raise SystemExit(f"Spec file not found: {spec_path}")

    job_spec: JobSpecFile = load_spec_file(spec_path)

    gpus_override: list[int] | None = None
    if args.gpus is not None:
        g = str(args.gpus).strip()
        if g.lower() == "auto":
            gpus_override = _detect_gpus_auto()
        else:
            gpus_override = parse_gpu_list(g)

    scheduler = _apply_overrides(
        job_spec.scheduler,
        gpus=gpus_override,
        policy=args.policy,
        max_running_jobs=args.max_running_jobs,
    )

    state_path = Path(args.state).resolve() if args.state else spec_path.parent / "scheduler_state.json"

    results = run_jobs(spec=scheduler, jobs=job_spec.jobs, state_path=state_path)

    # Print a concise summary.
    summary = {k: {"status": v.status, "returncode": v.returncode, "gpus": v.gpu_ids, "output_root": v.output_root} for k, v in results.items()}
    print(">>> Batch results:", json.dumps(summary, indent=2, sort_keys=True), flush=True)

    # Exit code: success only if all executed jobs succeeded.
    failed = [r for r in results.values() if r.status != "success"]
    raise SystemExit(0 if not failed else 1)


if __name__ == "__main__":
    main()

