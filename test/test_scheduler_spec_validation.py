from __future__ import annotations

import tempfile
from pathlib import Path


def main() -> None:
    from memxterminator.bezierfit.scheduler.spec import parse_gpu_list, parse_spec_dict

    tmp = Path(tempfile.mkdtemp(prefix="mxt_sched_spec_"))

    spec_ok = {
        "scheduler": {"gpus": [0, 1], "policy": "fill_first", "max_running_jobs": 2, "fail_fast": True},
        "jobs": [
            {
                "job_id": "pms_a",
                "kind": "bezierfit_particle_pms",
                "output_root": "runs/pms_a",
                "resources": {"gpus": 1, "procs": None},
                "args": {
                    "particle": "/data/particles_selected.cs",
                    "template": "/data/templates_selected.cs",
                    "control_points": "/data/control_points.json",
                    "input_base_dir": "cryosparc_project_root",
                    "points_step": 0.005,
                    "physical_membrane_dist": 35,
                    "resume": True,
                    "force": False,
                },
            }
        ],
    }

    parsed = parse_spec_dict(spec_ok, base_dir=tmp)
    assert parsed.scheduler.gpus == [0, 1]
    assert parsed.scheduler.policy == "fill_first"
    assert parsed.scheduler.fail_fast is True
    assert len(parsed.jobs) == 1
    assert Path(parsed.jobs[0].output_root).is_absolute()
    assert str(tmp / "runs" / "pms_a") == parsed.jobs[0].output_root
    assert parsed.jobs[0].args["input_base_dir"] == str((tmp / "cryosparc_project_root").resolve())

    # parse_gpu_list
    assert parse_gpu_list("0,1,2") == [0, 1, 2]

    # Missing required args.
    spec_missing = {
        "scheduler": {"gpus": [0], "policy": "fill_first", "max_running_jobs": 1, "fail_fast": True},
        "jobs": [{"job_id": "bad", "kind": "bezierfit_particle_pms", "output_root": "x", "resources": {"gpus": 1}, "args": {}}],
    }
    try:
        parse_spec_dict(spec_missing, base_dir=tmp)
        raise AssertionError("Expected parse_spec_dict to fail for missing args")
    except ValueError as exc:
        assert "Missing required arg" in str(exc)

    # fail_fast must be true.
    spec_ff = {
        "scheduler": {"gpus": [0], "policy": "fill_first", "max_running_jobs": 1, "fail_fast": False},
        "jobs": [
            {
                "job_id": "pms",
                "kind": "bezierfit_particle_pms",
                "output_root": "x",
                "resources": {"gpus": 1},
                "args": {
                    "particle": "a.cs",
                    "template": "b.cs",
                    "control_points": "c.json",
                    "points_step": 0.01,
                    "physical_membrane_dist": 35,
                },
            }
        ],
    }
    try:
        parse_spec_dict(spec_ff, base_dir=tmp)
        raise AssertionError("Expected parse_spec_dict to reject fail_fast=false")
    except ValueError as exc:
        assert "fail_fast=true" in str(exc)

    # Duplicate job_id.
    spec_dup = {
        "scheduler": {"gpus": [0], "policy": "fill_first", "max_running_jobs": 1, "fail_fast": True},
        "jobs": [
            {
                "job_id": "dup",
                "kind": "bezierfit_micrograph_mms",
                "output_root": "x1",
                "resources": {"gpus": 1},
                "args": {"particle": "particles_selected.star"},
            },
            {
                "job_id": "dup",
                "kind": "bezierfit_micrograph_mms",
                "output_root": "x2",
                "resources": {"gpus": 1},
                "args": {"particle": "particles_selected.star"},
            },
        ],
    }
    try:
        parse_spec_dict(spec_dup, base_dir=tmp)
        raise AssertionError("Expected parse_spec_dict to reject duplicate job_id")
    except ValueError as exc:
        assert "Duplicate job_id" in str(exc)

    print(">>> OK: scheduler spec validation.")


if __name__ == "__main__":
    main()
