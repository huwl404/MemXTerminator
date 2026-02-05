from __future__ import annotations

import tempfile
from pathlib import Path


def main() -> None:
    from memxterminator.bezierfit.scheduler.local_scheduler import _ensure_input_base_dir

    tmp = Path(tempfile.mkdtemp(prefix="mxt_sched_inject_"))
    proj = tmp / "P1"
    j220 = proj / "J220"
    j220.mkdir(parents=True, exist_ok=True)

    particle_cs = j220 / "particles_selected.cs"
    out = _ensure_input_base_dir("bezierfit_particle_pms", {"particle": str(particle_cs)})
    assert out["input_base_dir"] == str(proj)

    # Do not overwrite an explicit user value.
    out2 = _ensure_input_base_dir(
        "bezierfit_particle_pms",
        {"particle": str(particle_cs), "input_base_dir": "/explicit/base"},
    )
    assert out2["input_base_dir"] == "/explicit/base"

    # mem_analyze uses template (preferred) for inference.
    template_cs = j220 / "templates_selected.cs"
    out3 = _ensure_input_base_dir("bezierfit_mem_analyze", {"template": str(template_cs), "particle": "ignored.cs"})
    assert out3["input_base_dir"] == str(proj)

    print(">>> OK: scheduler input_base_dir injection.")


if __name__ == "__main__":
    main()

