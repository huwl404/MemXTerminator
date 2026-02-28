from __future__ import annotations

import json
import os
import re
import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping

from memxterminator.mxt_state import validate_output_dirname

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
    depends_on: tuple[str, ...] = ()


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
    spec_schema_version: int = 1


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
    if "spec_schema_version" in root:
        spec_schema_version = _require_int(root.get("spec_schema_version"), where="spec.spec_schema_version")
    else:
        spec_schema_version = 1
    if spec_schema_version < 1:
        raise ValueError(f"spec.spec_schema_version must be >= 1, got {spec_schema_version}")
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
        args = _normalise_job_args_paths(kind=kind, args=dict(args_raw), base_dir=base_dir)
        if kind in {"bezierfit_particle_pms", "bezierfit_micrograph_mms"} and "output_dirname" in args:
            args["output_dirname"] = validate_output_dirname(_require_str(args.get("output_dirname"), where=f"spec.jobs[{i}].args.output_dirname"))

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

        depends_raw = j_map.get("depends_on", [])
        if depends_raw is None:
            depends_raw = []
        if not isinstance(depends_raw, list):
            raise ValueError(f"spec.jobs[{i}].depends_on must be a list[str], got {type(depends_raw).__name__}")
        depends_on: list[str] = []
        for dep_i, dep in enumerate(depends_raw):
            dep_id = parse_job_id(dep)
            if dep_id in depends_on:
                raise ValueError(
                    f"Duplicate dependency id {dep_id!r} in spec.jobs[{i}].depends_on[{dep_i}] for job_id={job_id!r}"
                )
            depends_on.append(dep_id)

        _validate_job_args(kind=kind, args=args, job_id=job_id)

        jobs.append(
            BezierfitJob(
                job_id=job_id,
                kind=kind,  # type: ignore[arg-type]
                args=args,
                output_root=str(output_root_path),
                resources=JobResources(gpus=gpus_per_job, procs=procs),
                enabled=bool(enabled),
                depends_on=tuple(depends_on),
            )
        )

    # Ensure unique job IDs (after defaulting).
    ids = [j.job_id for j in jobs]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate job_id values found: {ids}")

    jobs = _validate_and_inject_dependencies(jobs=jobs)

    return JobSpecFile(scheduler=scheduler, jobs=jobs, spec_schema_version=int(spec_schema_version))


def _normalise_job_args_paths(*, kind: str, args: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    """
    Normalize known path-like job args relative to `base_dir` (spec directory).

    This prevents surprises when the scheduler changes `cwd` to the per-job
    output directory, which would otherwise break relative input paths.
    """
    keys: set[str]
    if kind == "bezierfit_particle_pms":
        keys = {"particle", "template", "control_points", "input_base_dir"}
    elif kind == "bezierfit_micrograph_mms":
        keys = {"particle", "input_base_dir", "particle_output_root"}
    elif kind == "bezierfit_mem_analyze":
        # Note: `output` is intentionally not normalized so relative outputs
        # remain relative to the job's `cwd` (job output_root) under the scheduler.
        keys = {"template", "particle", "input_base_dir"}
    else:  # pragma: no cover
        keys = set()

    for k in keys:
        v = args.get(k)
        if not isinstance(v, str):
            continue
        raw = v.strip()
        if raw == "":
            continue
        expanded = os.path.expanduser(os.path.expandvars(raw))
        p = Path(expanded)
        if p.is_absolute():
            args[k] = str(p)
            continue
        args[k] = str((base_dir / p).resolve())

    return args


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
        if "output_dirname" in args and _value_is_nonempty(args.get("output_dirname")):
            args["output_dirname"] = validate_output_dirname(_require_str(args.get("output_dirname"), where=f"job[{job_id}].args.output_dirname"))
        for key in {"resume"}:
            if key in args:
                _require_bool(args.get(key), where=f"job[{job_id}].args.{key}")
        for key in {"force", "adopt_existing_outputs", "skip_failed"}:
            if key in args:
                _require_bool(args.get(key), where=f"job[{job_id}].args.{key}")
        return

    if kind == "bezierfit_micrograph_mms":
        require("particle")
        if "output_dirname" in args and _value_is_nonempty(args.get("output_dirname")):
            args["output_dirname"] = validate_output_dirname(_require_str(args.get("output_dirname"), where=f"job[{job_id}].args.output_dirname"))
        for key in {"resume", "require_particle_mxt", "strict_dependencies", "write_output_star"}:
            if key in args:
                _require_bool(args.get(key), where=f"job[{job_id}].args.{key}")
        for key in {"force", "adopt_existing_outputs", "skip_failed"}:
            if key in args:
                _require_bool(args.get(key), where=f"job[{job_id}].args.{key}")
        return

    if kind == "bezierfit_mem_analyze":
        require("template")
        require("particle")
        require("output")
        return

    raise ValueError(f"Unexpected kind {kind!r}")


def _value_is_nonempty(value: Any) -> bool:
    if value is None:
        return False
    return str(value).strip() != ""


def _require_bool_with_default(args: Mapping[str, Any], *, key: str, default: bool, where: str) -> bool:
    if key not in args:
        return bool(default)
    return _require_bool(args.get(key), where=where)


def _validate_dependency_graph_acyclic(jobs: list[BezierfitJob]) -> None:
    by_id = {job.job_id: job for job in jobs}
    indegree: dict[str, int] = {job.job_id: 0 for job in jobs}
    edges: dict[str, list[str]] = {job.job_id: [] for job in jobs}

    for job in jobs:
        for dep_id in job.depends_on:
            # Unknown dependency IDs are validated earlier.
            if dep_id not in by_id:  # pragma: no cover
                continue
            edges[dep_id].append(job.job_id)
            indegree[job.job_id] += 1

    q: deque[str] = deque([job_id for job_id, deg in indegree.items() if deg == 0])
    visited: list[str] = []
    while q:
        node = q.popleft()
        visited.append(node)
        for nxt in edges.get(node, []):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                q.append(nxt)

    if len(visited) != len(jobs):
        cycle_nodes = sorted([job_id for job_id, deg in indegree.items() if deg > 0])
        raise ValueError(
            "Dependency cycle detected in spec.jobs[].depends_on.\n"
            f"  cycle_nodes={cycle_nodes}"
        )


def _validate_and_inject_dependencies(*, jobs: list[BezierfitJob]) -> list[BezierfitJob]:
    by_id = {job.job_id: job for job in jobs}
    args_by_id: dict[str, dict[str, Any]] = {job.job_id: dict(job.args) for job in jobs}

    # Validation order matters for clear diagnostics:
    # (1) unknown IDs, (2) kind constraints, (3) cycle detection.
    for job in jobs:
        for dep_id in job.depends_on:
            if dep_id not in by_id:
                raise ValueError(
                    f"Unknown dependency job_id={dep_id!r} in depends_on for job_id={job.job_id!r}. "
                    "Fix the ID or remove it."
                )

    for job in jobs:
        if not job.depends_on:
            continue
        if job.kind != "bezierfit_micrograph_mms":
            raise ValueError(
                f"Only bezierfit_micrograph_mms jobs may declare depends_on; "
                f"job_id={job.job_id!r} kind={job.kind!r} must use depends_on=[]."
            )
        for dep_id in job.depends_on:
            dep_job = by_id[dep_id]
            if dep_job.kind != "bezierfit_particle_pms":
                raise ValueError(
                    f"MMS job_id={job.job_id!r} can only depend on PMS jobs, "
                    f"but dependency {dep_id!r} has kind={dep_job.kind!r}."
                )
        if bool(job.enabled):
            for dep_id in job.depends_on:
                dep_job = by_id[dep_id]
                if not bool(dep_job.enabled):
                    raise ValueError(
                        f"Enabled MMS job_id={job.job_id!r} depends on disabled dependency "
                        f"job_id={dep_id!r}. Enable the dependency or remove depends_on."
                    )

    _validate_dependency_graph_acyclic(jobs)

    # MMS dependency-root and output_dirname injection / consistency checks.
    for job in jobs:
        if job.kind != "bezierfit_micrograph_mms":
            continue
        mms_args = args_by_id[job.job_id]
        explicit_particle_root = _value_is_nonempty(mms_args.get("particle_output_root"))
        dep_count = len(job.depends_on)

        if dep_count == 0:
            # Legacy standalone MMS behavior.
            continue

        if dep_count > 1 and not explicit_particle_root:
            raise ValueError(
                f"Ambiguous MMS dependencies for job_id={job.job_id!r}: depends_on has multiple PMS jobs "
                f"{list(job.depends_on)} but args.particle_output_root is not set."
            )

        if dep_count == 1:
            upstream = by_id[job.depends_on[0]]
            upstream_args = args_by_id[upstream.job_id]
            upstream_output_dirname = validate_output_dirname(
                _require_str(
                    upstream_args.get("output_dirname", "subtracted"),
                    where=f"spec.jobs[{upstream.job_id}].args.output_dirname",
                )
            )

            if not explicit_particle_root:
                mms_args["particle_output_root"] = upstream.output_root

            if _value_is_nonempty(mms_args.get("output_dirname")):
                mms_output_dirname = validate_output_dirname(
                    _require_str(
                        mms_args.get("output_dirname"),
                        where=f"spec.jobs[{job.job_id}].args.output_dirname",
                    )
                )
                if mms_output_dirname != upstream_output_dirname:
                    raise ValueError(
                        f"MMS job_id={job.job_id!r} output_dirname={mms_output_dirname!r} does not match "
                        f"upstream PMS job_id={upstream.job_id!r} output_dirname={upstream_output_dirname!r}. "
                        "Use the same output_dirname to keep dependency lookup consistent."
                    )
                mms_args["output_dirname"] = mms_output_dirname
            else:
                mms_args["output_dirname"] = upstream_output_dirname

    enabled_jobs = [job for job in jobs if bool(job.enabled)]
    has_enabled_pms = any(job.kind == "bezierfit_particle_pms" for job in enabled_jobs)
    has_enabled_mms = any(job.kind == "bezierfit_micrograph_mms" for job in enabled_jobs)
    if not (has_enabled_pms and has_enabled_mms):
        resolved_jobs: list[BezierfitJob] = []
        for job in jobs:
            resolved_jobs.append(
                BezierfitJob(
                    job_id=job.job_id,
                    kind=job.kind,
                    args=dict(args_by_id[job.job_id]),
                    output_root=job.output_root,
                    resources=job.resources,
                    enabled=job.enabled,
                    depends_on=job.depends_on,
                )
            )
        return resolved_jobs

    migration_hint = (
        "Migration required for PMS+MMS specs: each MMS job must either set depends_on to exactly one PMS job "
        "or set args.particle_output_root explicitly; and must keep args.require_particle_mxt=true plus "
        "args.strict_dependencies=true."
    )

    for job in enabled_jobs:
        if job.kind != "bezierfit_micrograph_mms":
            continue

        mms_args = args_by_id[job.job_id]
        explicit_particle_root = _value_is_nonempty(mms_args.get("particle_output_root"))
        dep_ok = len(job.depends_on) == 1 and by_id[job.depends_on[0]].kind == "bezierfit_particle_pms"
        if not (dep_ok or explicit_particle_root):
            raise ValueError(
                f"{migration_hint}\n"
                f"  offending_job={job.job_id!r} depends_on={list(job.depends_on)} "
                f"particle_output_root={mms_args.get('particle_output_root')!r}"
            )

        require_particle_mxt = _require_bool_with_default(
            mms_args,
            key="require_particle_mxt",
            default=True,
            where=f"spec.jobs[{job.job_id}].args.require_particle_mxt",
        )
        if require_particle_mxt is not True:
            raise ValueError(
                f"{migration_hint}\n"
                f"  offending_job={job.job_id!r} has require_particle_mxt={require_particle_mxt!r}"
            )

        strict_dependencies = _require_bool_with_default(
            mms_args,
            key="strict_dependencies",
            default=True,
            where=f"spec.jobs[{job.job_id}].args.strict_dependencies",
        )
        if strict_dependencies is not True:
            raise ValueError(
                f"{migration_hint}\n"
                f"  offending_job={job.job_id!r} has strict_dependencies={strict_dependencies!r}"
            )

    resolved_jobs: list[BezierfitJob] = []
    for job in jobs:
        resolved_jobs.append(
            BezierfitJob(
                job_id=job.job_id,
                kind=job.kind,
                args=dict(args_by_id[job.job_id]),
                output_root=job.output_root,
                resources=job.resources,
                enabled=job.enabled,
                depends_on=job.depends_on,
            )
        )
    return resolved_jobs


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
        bool_optional = {"resume", "require_particle_mxt", "strict_dependencies", "write_output_star"}
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
