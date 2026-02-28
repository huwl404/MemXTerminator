from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


def _expect_value_error(fn, contains: str) -> None:
    try:
        fn()
        raise AssertionError(f"Expected ValueError containing: {contains!r}")
    except ValueError as exc:
        msg = str(exc)
        if contains not in msg:
            raise AssertionError(f"Expected ValueError containing {contains!r}, got: {msg!r}") from exc


def _base_scheduler() -> dict:
    return {"scheduler": {"gpus": [0, 1], "policy": "fill_first", "max_running_jobs": 2, "fail_fast": True}}


def _mk_pms(job_id: str, output_root: str, *, output_dirname: str = "subtracted") -> dict:
    return {
        "job_id": job_id,
        "kind": "bezierfit_particle_pms",
        "output_root": output_root,
        "resources": {"gpus": 1},
        "args": {
            "particle": "/data/particles_selected.cs",
            "template": "/data/templates_selected.cs",
            "control_points": "/data/control_points.json",
            "points_step": 0.005,
            "physical_membrane_dist": 35,
            "output_dirname": output_dirname,
            "resume": True,
            "force": False,
        },
    }


def _mk_mms(job_id: str, output_root: str, *, depends_on: list[str] | None = None, args_extra: dict | None = None) -> dict:
    args = {
        "particle": "/data/particles_selected.star",
        "require_particle_mxt": True,
        "strict_dependencies": True,
    }
    if args_extra:
        args.update(args_extra)
    return {
        "job_id": job_id,
        "kind": "bezierfit_micrograph_mms",
        "output_root": output_root,
        "depends_on": [] if depends_on is None else list(depends_on),
        "resources": {"gpus": 1},
        "args": args,
    }


def test_dependency_parse_and_validation() -> None:
    from memxterminator.bezierfit.scheduler.spec import (
        BezierfitJob,
        JobResources,
        _validate_dependency_graph_acyclic,
        parse_spec_dict,
    )

    tmp = Path(tempfile.mkdtemp(prefix="mxt_sched_deps_parse_"))

    def parse(obj: dict):
        return parse_spec_dict(obj, base_dir=tmp)

    # Unknown dependency ID: should fail before cycle checks.
    obj_unknown = dict(_base_scheduler())
    obj_unknown["jobs"] = [_mk_pms("pms_001", "runs/pms_001"), _mk_mms("mms_001", "runs/mms_001", depends_on=["nope"])]
    _expect_value_error(lambda: parse(obj_unknown), "Unknown dependency")

    # Non-MMS job may not depend on others.
    bad_non_mms = dict(_base_scheduler())
    pms = _mk_pms("pms_001", "runs/pms_001")
    pms["depends_on"] = ["mms_001"]
    bad_non_mms["jobs"] = [pms, _mk_mms("mms_001", "runs/mms_001", depends_on=["pms_001"])]
    _expect_value_error(lambda: parse(bad_non_mms), "Only bezierfit_micrograph_mms jobs may declare depends_on")

    # MMS can only depend on PMS.
    bad_kind = dict(_base_scheduler())
    bad_kind["jobs"] = [
        {
            "job_id": "an_001",
            "kind": "bezierfit_mem_analyze",
            "output_root": "runs/an_001",
            "resources": {"gpus": 1},
            "args": {
                "template": "/data/t.cs",
                "particle": "/data/p.cs",
                "output": "/data/o.json",
            },
        },
        _mk_mms("mms_001", "runs/mms_001", depends_on=["an_001"]),
    ]
    _expect_value_error(lambda: parse(bad_kind), "can only depend on PMS")

    # Enabled MMS cannot depend on disabled PMS.
    bad_disabled_dep = dict(_base_scheduler())
    pms_disabled = _mk_pms("pms_disabled", "runs/pms_disabled")
    pms_disabled["enabled"] = False
    bad_disabled_dep["jobs"] = [
        pms_disabled,
        _mk_mms("mms_001", "runs/mms_001", depends_on=["pms_disabled"]),
    ]
    _expect_value_error(lambda: parse(bad_disabled_dep), "depends on disabled dependency")

    # Explicit cycle check (internal helper): A -> B -> A.
    cycle_jobs = [
        BezierfitJob(
            job_id="a",
            kind="bezierfit_particle_pms",
            args={},
            output_root="/tmp/a",
            resources=JobResources(gpus=1),
            depends_on=("b",),
        ),
        BezierfitJob(
            job_id="b",
            kind="bezierfit_micrograph_mms",
            args={},
            output_root="/tmp/b",
            resources=JobResources(gpus=1),
            depends_on=("a",),
        ),
    ]
    _expect_value_error(lambda: _validate_dependency_graph_acyclic(cycle_jobs), "cycle")


def test_mms_injection_and_batch_safety() -> None:
    from memxterminator.bezierfit.scheduler.spec import parse_spec_dict

    tmp = Path(tempfile.mkdtemp(prefix="mxt_sched_deps_inject_"))

    spec_ok = dict(_base_scheduler())
    spec_ok["jobs"] = [
        _mk_pms("pms_001", "runs/pms_001", output_dirname="class_01"),
        _mk_mms("mms_001", "runs/mms_001", depends_on=["pms_001"]),
    ]
    parsed = parse_spec_dict(spec_ok, base_dir=tmp)
    by_id = {j.job_id: j for j in parsed.jobs}
    mms = by_id["mms_001"]
    pms = by_id["pms_001"]
    assert mms.depends_on == ("pms_001",)
    assert mms.args["particle_output_root"] == pms.output_root
    assert mms.args["output_dirname"] == "class_01"

    # Do not override explicit particle_output_root.
    explicit_root = str((tmp / "manual_root").resolve())
    spec_explicit = dict(_base_scheduler())
    spec_explicit["jobs"] = [
        _mk_pms("pms_001", "runs/pms_001", output_dirname="class_01"),
        _mk_mms(
            "mms_001",
            "runs/mms_001",
            depends_on=["pms_001"],
            args_extra={"particle_output_root": explicit_root},
        ),
    ]
    parsed2 = parse_spec_dict(spec_explicit, base_dir=tmp)
    mms2 = {j.job_id: j for j in parsed2.jobs}["mms_001"]
    assert mms2.args["particle_output_root"] == explicit_root
    assert mms2.args["output_dirname"] == "class_01"

    # Multiple PMS dependencies without explicit root is ambiguous.
    spec_ambiguous = dict(_base_scheduler())
    spec_ambiguous["jobs"] = [
        _mk_pms("pms_a", "runs/pms_a"),
        _mk_pms("pms_b", "runs/pms_b"),
        _mk_mms("mms_001", "runs/mms_001", depends_on=["pms_a", "pms_b"]),
    ]
    _expect_value_error(
        lambda: parse_spec_dict(spec_ambiguous, base_dir=tmp),
        "Ambiguous MMS dependencies",
    )

    # PMS+MMS hard-fail migration rule.
    spec_missing = dict(_base_scheduler())
    spec_missing["jobs"] = [_mk_pms("pms_001", "runs/pms_001"), _mk_mms("mms_001", "runs/mms_001", depends_on=[])]
    _expect_value_error(
        lambda: parse_spec_dict(spec_missing, base_dir=tmp),
        "Migration required for PMS+MMS specs",
    )

    # PMS+MMS requires strict_dependencies=true.
    spec_strict_off = dict(_base_scheduler())
    spec_strict_off["jobs"] = [
        _mk_pms("pms_001", "runs/pms_001"),
        _mk_mms("mms_001", "runs/mms_001", depends_on=["pms_001"], args_extra={"strict_dependencies": False}),
    ]
    _expect_value_error(lambda: parse_spec_dict(spec_strict_off, base_dir=tmp), "strict_dependencies")


def test_scheduler_ordering_and_failfast_reasons() -> None:
    from memxterminator.bezierfit.scheduler.local_scheduler import run_jobs
    from memxterminator.bezierfit.scheduler.spec import BezierfitJob, JobResources, SchedulerSpec

    tmp = Path(tempfile.mkdtemp(prefix="mxt_sched_deps_runtime_"))

    # Ordering test: MMS must not start before PMS success even with spare GPUs.
    pms_root = tmp / "run_order" / "pms"
    mms_root = tmp / "run_order" / "mms"
    pms_root.mkdir(parents=True, exist_ok=True)
    mms_root.mkdir(parents=True, exist_ok=True)
    marker_dir = tmp / "markers_order"
    marker_dir.mkdir(parents=True, exist_ok=True)

    pms_end = marker_dir / "pms_end.txt"
    mms_start = marker_dir / "mms_start.txt"

    jobs_order = [
        BezierfitJob(
            job_id="pms_001",
            kind="bezierfit_particle_pms",
            args={"particle": "/data/p.cs", "input_base_dir": str(tmp)},
            output_root=str(pms_root),
            resources=JobResources(gpus=1, procs=1),
            enabled=True,
            depends_on=(),
        ),
        BezierfitJob(
            job_id="mms_001",
            kind="bezierfit_micrograph_mms",
            args={"particle": "/data/p.star", "input_base_dir": str(tmp)},
            output_root=str(mms_root),
            resources=JobResources(gpus=1, procs=1),
            enabled=True,
            depends_on=("pms_001",),
        ),
    ]

    import memxterminator.bezierfit.scheduler.local_scheduler as ls

    orig_build = ls.build_job_argv

    def fake_build(job: BezierfitJob) -> list[str]:
        if job.job_id == "pms_001":
            code = (
                "import time, pathlib; "
                f"time.sleep(0.4); pathlib.Path({str(pms_end)!r}).write_text(str(time.time()), encoding='utf-8')"
            )
        else:
            code = f"import time, pathlib; pathlib.Path({str(mms_start)!r}).write_text(str(time.time()), encoding='utf-8')"
        return [sys.executable, "-c", code]

    ls.build_job_argv = fake_build
    try:
        results = run_jobs(
            spec=SchedulerSpec(gpus=[0, 1], policy="fill_first", max_running_jobs=2, fail_fast=True),
            jobs=jobs_order,
            state_path=tmp / "scheduler_state_order.json",
            poll_interval_s=0.05,
        )
    finally:
        ls.build_job_argv = orig_build

    assert results["pms_001"].status == "success"
    assert results["mms_001"].status == "success"
    pms_end_ts = float(pms_end.read_text(encoding="utf-8").strip())
    mms_start_ts = float(mms_start.read_text(encoding="utf-8").strip())
    assert mms_start_ts >= pms_end_ts

    # Fail-fast reason propagation: dependent pending jobs should be marked upstream_failed.
    fail_root = tmp / "run_fail"
    pms_fail_root = fail_root / "pms_fail"
    mms_dep_root = fail_root / "mms_dep"
    pms_fail_root.mkdir(parents=True, exist_ok=True)
    mms_dep_root.mkdir(parents=True, exist_ok=True)

    jobs_fail = [
        BezierfitJob(
            job_id="pms_fail",
            kind="bezierfit_particle_pms",
            args={"particle": "/data/p.cs", "input_base_dir": str(tmp)},
            output_root=str(pms_fail_root),
            resources=JobResources(gpus=1, procs=1),
            enabled=True,
            depends_on=(),
        ),
        BezierfitJob(
            job_id="mms_dep",
            kind="bezierfit_micrograph_mms",
            args={"particle": "/data/p.star", "input_base_dir": str(tmp)},
            output_root=str(mms_dep_root),
            resources=JobResources(gpus=1, procs=1),
            enabled=True,
            depends_on=("pms_fail",),
        ),
    ]

    def fake_build_fail(job: BezierfitJob) -> list[str]:
        if job.job_id == "pms_fail":
            code = "import sys; sys.exit(5)"
        else:
            code = "import sys; sys.exit(0)"
        return [sys.executable, "-c", code]

    ls.build_job_argv = fake_build_fail
    state_fail = tmp / "scheduler_state_fail.json"
    try:
        results_fail = run_jobs(
            spec=SchedulerSpec(gpus=[0], policy="fill_first", max_running_jobs=1, fail_fast=True),
            jobs=jobs_fail,
            state_path=state_fail,
            poll_interval_s=0.05,
        )
    finally:
        ls.build_job_argv = orig_build

    assert results_fail["pms_fail"].status == "failed"
    assert results_fail["mms_dep"].status == "canceled"

    state_obj = json.loads(state_fail.read_text(encoding="utf-8"))
    by_id = {rec["job_id"]: rec for rec in state_obj["jobs"]}
    assert by_id["mms_dep"]["reason"] == "upstream_failed"

    # Deadlock path: pending enabled MMS blocked by disabled upstream dependency.
    deadlock_root = tmp / "run_deadlock"
    pms_disabled_root = deadlock_root / "pms_disabled"
    mms_blocked_root = deadlock_root / "mms_blocked"
    pms_disabled_root.mkdir(parents=True, exist_ok=True)
    mms_blocked_root.mkdir(parents=True, exist_ok=True)

    jobs_deadlock = [
        BezierfitJob(
            job_id="pms_disabled",
            kind="bezierfit_particle_pms",
            args={"particle": "/data/p.cs", "input_base_dir": str(tmp)},
            output_root=str(pms_disabled_root),
            resources=JobResources(gpus=1, procs=1),
            enabled=False,
            depends_on=(),
        ),
        BezierfitJob(
            job_id="mms_blocked",
            kind="bezierfit_micrograph_mms",
            args={"particle": "/data/p.star", "input_base_dir": str(tmp)},
            output_root=str(mms_blocked_root),
            resources=JobResources(gpus=1, procs=1),
            enabled=True,
            depends_on=("pms_disabled",),
        ),
    ]
    ls.build_job_argv = fake_build_fail
    state_deadlock = tmp / "scheduler_state_deadlock.json"
    try:
        results_deadlock = run_jobs(
            spec=SchedulerSpec(gpus=[0], policy="fill_first", max_running_jobs=1, fail_fast=True),
            jobs=jobs_deadlock,
            state_path=state_deadlock,
            poll_interval_s=0.05,
        )
    finally:
        ls.build_job_argv = orig_build

    assert results_deadlock["mms_blocked"].status == "canceled"
    dead_obj = json.loads(state_deadlock.read_text(encoding="utf-8"))
    dead_by_id = {rec["job_id"]: rec for rec in dead_obj["jobs"]}
    assert dead_by_id["mms_blocked"]["reason"] == "upstream_failed"


def main() -> None:
    test_dependency_parse_and_validation()
    test_mms_injection_and_batch_safety()
    test_scheduler_ordering_and_failfast_reasons()
    print(">>> OK: scheduler dependencies.")


if __name__ == "__main__":
    main()
