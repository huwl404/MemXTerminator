from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping

JobKind = Literal[
    "bezierfit_mem_analyze",
    "bezierfit_particle_pms",
    "bezierfit_micrograph_mms",
]

JobStatus = Literal["queued", "running", "success", "failed", "canceled"]

SchedulerPolicy = Literal["fill_first", "round_robin"]

_JOB_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True)
class JobResources:
    gpus: int
    procs: int | None = None


@dataclass(frozen=True)
class BezierfitJob:
    job_id: str
    kind: JobKind
    args: dict[str, Any]
    output_root: str
    resources: JobResources
    enabled: bool = True


@dataclass(frozen=True)
class SchedulerSpec:
    gpus: list[int]
    policy: SchedulerPolicy = "fill_first"
    max_running_jobs: int = 1
    fail_fast: bool = True


@dataclass(frozen=True)
class JobSpecFile:
    scheduler: SchedulerSpec
    jobs: list[BezierfitJob]


@dataclass(frozen=True)
class JobResult:
    job_id: str
    status: JobStatus
    returncode: int | None
    gpu_ids: list[int]
    output_root: str
    pid: int | None
    started_utc: str
    finished_utc: str | None
    cmd_argv: list[str]
    stdout_path: str
    stderr_path: str


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_job_id(value: object) -> str:
    s = str(value).strip()
    if s == "":
        raise ValueError("job_id must be non-empty")
    if not _JOB_ID_RE.match(s):
        raise ValueError(
            f"Invalid job_id {s!r}. Allowed pattern: {_JOB_ID_RE.pattern} "
            "(use letters, digits, '.', '_', '-')."
        )
    return s


def parse_gpu_list(text: str) -> list[int]:
    parts = [p.strip() for p in str(text).split(",") if p.strip() != ""]
    if not parts:
        raise ValueError("GPU list is empty")
    gpus: list[int] = []
    for p in parts:
        try:
            gpus.append(int(p))
        except Exception as exc:
            raise ValueError(f"Invalid GPU id {p!r}: {exc}") from exc
    if len(set(gpus)) != len(gpus):
        raise ValueError(f"GPU list has duplicates: {gpus}")
    if any(x < 0 for x in gpus):
        raise ValueError(f"GPU ids must be >= 0, got {gpus}")
    return gpus


def _require_mapping(obj: Any, *, where: str) -> Mapping[str, Any]:
    if not isinstance(obj, Mapping):
        raise ValueError(f"Expected object at {where} to be a mapping, got {type(obj).__name__}")
    return obj


def _require_bool(value: Any, *, where: str) -> bool:
    if isinstance(value, bool):
        return bool(value)
    raise ValueError(f"Expected boolean at {where}, got {type(value).__name__}")


def _require_int(value: Any, *, where: str) -> int:
    try:
        i = int(value)
    except Exception as exc:
        raise ValueError(f"Expected int at {where}, got {value!r}") from exc
    return i


def _require_str(value: Any, *, where: str) -> str:
    s = str(value)
    if s == "":
        raise ValueError(f"Expected non-empty string at {where}")
    return s


def load_spec_file(path: os.PathLike[str] | str) -> JobSpecFile:
    p = Path(os.fspath(path))
    with open(p, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return parse_spec_dict(obj, base_dir=p.parent)


def parse_spec_dict(obj: Any, *, base_dir: Path) -> JobSpecFile:
    root = _require_mapping(obj, where="spec")
    scheduler_raw = _require_mapping(root.get("scheduler"), where="spec.scheduler")

    policy = str(scheduler_raw.get("policy", "fill_first")).strip()
    if policy not in {"fill_first", "round_robin"}:
        raise ValueError(f"Unsupported scheduler.policy={policy!r} (expected fill_first or round_robin)")

    gpus_raw = scheduler_raw.get("gpus")
    if not isinstance(gpus_raw, list) or not gpus_raw:
        raise ValueError("spec.scheduler.gpus must be a non-empty list of integers")
    gpus = [_require_int(x, where="spec.scheduler.gpus[]") for x in gpus_raw]
    if len(set(gpus)) != len(gpus):
        raise ValueError(f"spec.scheduler.gpus contains duplicates: {gpus}")
    if any(x < 0 for x in gpus):
        raise ValueError(f"spec.scheduler.gpus must be >=0, got {gpus}")

    max_running_jobs = int(scheduler_raw.get("max_running_jobs", len(gpus)))
    if max_running_jobs < 1:
        raise ValueError(f"spec.scheduler.max_running_jobs must be >= 1, got {max_running_jobs}")

    if "fail_fast" in scheduler_raw:
        fail_fast = _require_bool(scheduler_raw.get("fail_fast"), where="spec.scheduler.fail_fast")
    else:
        fail_fast = True
    if fail_fast is not True:
        raise ValueError("This version only supports fail_fast=true (fail-fast mode is mandatory).")

    scheduler = SchedulerSpec(gpus=gpus, policy=policy, max_running_jobs=max_running_jobs, fail_fast=True)

    jobs_raw = root.get("jobs")
    if not isinstance(jobs_raw, list) or not jobs_raw:
        raise ValueError("spec.jobs must be a non-empty list")

    jobs: list[BezierfitJob] = []
    for i, j in enumerate(jobs_raw):
        j_map = _require_mapping(j, where=f"spec.jobs[{i}]")
        if "enabled" in j_map:
            enabled = _require_bool(j_map.get("enabled"), where=f"spec.jobs[{i}].enabled")
        else:
            enabled = True
        job_id = parse_job_id(j_map.get("job_id", f"job_{i:03d}"))
        kind = str(j_map.get("kind", "")).strip()
        if kind not in {"bezierfit_mem_analyze", "bezierfit_particle_pms", "bezierfit_micrograph_mms"}:
            raise ValueError(f"Unsupported job kind {kind!r} for job_id={job_id!r}")

        args_raw = _require_mapping(j_map.get("args", {}), where=f"spec.jobs[{i}].args")
        args = dict(args_raw)

        output_root_raw = j_map.get("output_root")
        if output_root_raw is None:
            raise ValueError(f"Missing output_root for job_id={job_id!r}")
        output_root_s = _require_str(output_root_raw, where=f"spec.jobs[{i}].output_root")
        output_root_path = Path(output_root_s)
        if not output_root_path.is_absolute():
            output_root_path = (base_dir / output_root_path).resolve()

        resources_raw = _require_mapping(j_map.get("resources", {}), where=f"spec.jobs[{i}].resources")
        gpus_per_job = _require_int(resources_raw.get("gpus", 1), where=f"spec.jobs[{i}].resources.gpus")
        if gpus_per_job < 1:
            raise ValueError(f"resources.gpus must be >=1 for job_id={job_id!r}, got {gpus_per_job}")
        procs_val = resources_raw.get("procs", None)
        procs = None if procs_val is None else _require_int(procs_val, where=f"spec.jobs[{i}].resources.procs")

        _validate_job_args(kind=kind, args=args, job_id=job_id)

        jobs.append(
            BezierfitJob(
                job_id=job_id,
                kind=kind,  # type: ignore[arg-type]
                args=args,
                output_root=str(output_root_path),
                resources=JobResources(gpus=gpus_per_job, procs=procs),
                enabled=bool(enabled),
            )
        )

    # Ensure unique job IDs (after defaulting).
    ids = [j.job_id for j in jobs]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate job_id values found: {ids}")

    return JobSpecFile(scheduler=scheduler, jobs=jobs)


def _validate_job_args(*, kind: str, args: dict[str, Any], job_id: str) -> None:
    def require(key: str) -> None:
        if key not in args or args[key] in (None, ""):
            raise ValueError(f"Missing required arg {key!r} for job_id={job_id!r} kind={kind!r}")

    if kind == "bezierfit_particle_pms":
        require("particle")
        require("template")
        require("control_points")
        require("points_step")
        require("physical_membrane_dist")
        return

    if kind == "bezierfit_micrograph_mms":
        require("particle")
        return

    if kind == "bezierfit_mem_analyze":
        require("template")
        require("particle")
        require("output")
        return

    raise ValueError(f"Unexpected kind {kind!r}")


def build_job_argv(job: BezierfitJob) -> list[str]:
    """
    Build the exact `argv` to run this job as a subprocess.

    Notes:
    - For Bezierfit PMS/MMS, always inject `--output_root <job.output_root>` for isolation.
    - `job.args` keys are interpreted as CLI flags without leading '--'.
    """
    if job.kind == "bezierfit_particle_pms":
        module = "memxterminator.bezierfit.bin.mem_subtract_main"
    elif job.kind == "bezierfit_micrograph_mms":
        module = "memxterminator.bezierfit.bin.micrograph_mem_subtract_main"
    elif job.kind == "bezierfit_mem_analyze":
        module = "memxterminator.bezierfit.bin.mem_analyze_main"
    else:  # pragma: no cover
        raise ValueError(f"Unsupported job kind: {job.kind!r}")

    argv: list[str] = [sys.executable, "-u", "-m", module]

    args = dict(job.args)
    if job.kind in {"bezierfit_particle_pms", "bezierfit_micrograph_mms"}:
        args["output_root"] = str(job.output_root)

    argv.extend(_args_dict_to_argv(args, kind=job.kind))
    return argv


def _args_dict_to_argv(args: dict[str, Any], *, kind: JobKind) -> list[str]:
    """
    Convert a key/value dict into CLI argv.

    Conventions:
    - Keys correspond to flag names without leading '--' (e.g. 'points_step' -> '--points_step').
    - Booleans:
      - BooleanOptionalAction style: include '--foo' or '--no-foo' when value is bool.
      - store_true flags: include '--flag' only when True.
    """
    argv: list[str] = []

    bool_optional = set()
    store_true = set()
    if kind == "bezierfit_particle_pms":
        bool_optional = {"resume"}
        store_true = {"force", "adopt_existing_outputs", "skip_failed"}
    elif kind == "bezierfit_micrograph_mms":
        bool_optional = {"resume", "require_particle_mxt"}
        store_true = {"force", "adopt_existing_outputs", "skip_failed"}
    elif kind == "bezierfit_mem_analyze":
        bool_optional = set()
        store_true = set()

    for key in sorted(args.keys()):
        value = args[key]
        flag = f"--{key}"

        if key in store_true:
            if bool(value):
                argv.append(flag)
            continue

        if key in bool_optional:
            if isinstance(value, bool):
                argv.append(flag if value else f"--no-{key}")
                continue
            # If not bool, fall through and pass as value.

        if value is None:
            continue

        argv.extend([flag, str(value)])

    return argv
