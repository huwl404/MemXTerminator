from __future__ import annotations

import tempfile
from pathlib import Path


def _write_mrc(path: Path) -> None:
    import mrcfile  # type: ignore
    import numpy as np  # type: ignore

    path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(str(path), overwrite=True) as mrc:
        mrc.set_data(np.zeros((2, 2), dtype=np.float32))


def _write_star(path: Path, stack_path: Path, micrograph_path: Path) -> None:
    import pandas as pd  # type: ignore
    import starfile  # type: ignore

    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "rlnImageName": [f"1@{stack_path}"],
            "rlnMicrographName": [str(micrograph_path)],
            "rlnCoordinateX": [10.0],
            "rlnCoordinateY": [20.0],
        }
    )
    starfile.write(df, str(path), overwrite=True)


def _expect_preflight_error(fn, contains: str) -> None:
    from memxterminator.bezierfit.bin.micrograph_mem_subtract_main import DependencyPreflightError

    try:
        fn()
        raise AssertionError(f"Expected DependencyPreflightError containing {contains!r}")
    except DependencyPreflightError as exc:
        msg = str(exc)
        if contains not in msg:
            raise AssertionError(f"Expected error to contain {contains!r}, got: {msg!r}") from exc


def main() -> None:
    from memxterminator.bezierfit.bin.micrograph_mem_subtract_main import MicrographMembraneSubtract
    from memxterminator.mxt_state import to_output_stack_path_in_root, write_json_atomic

    tmp = Path(tempfile.mkdtemp(prefix="mxt_mms_preflight_"))
    raw_root = tmp / "raw"
    pms_root = tmp / "pms_root"
    mms_root = tmp / "mms_root"

    raw_stack = raw_root / "extract" / "stack_001.mrcs"
    raw_micrograph = raw_root / "micrographs" / "extract" / "mg_001.mrc"
    star_path = tmp / "particles.star"
    _write_mrc(raw_stack)
    _write_mrc(raw_micrograph)
    _write_star(star_path, raw_stack, raw_micrograph)

    output_dirname = "class_01"
    dep_stack = Path(
        to_output_stack_path_in_root(
            str(raw_stack),
            output_root=str(pms_root),
            output_dirname=output_dirname,
        )
    )
    _write_mrc(dep_stack)
    write_json_atomic(
        str(dep_stack) + ".mxt",
        {
            "mxt_schema": 1,
            "task": "bezierfit_particle_pms",
            "status": "success",
            "params_hash": "abc123",
        },
    )

    # Without particle_output_root, strict preflight should fail (legacy wrong root).
    mms_wrong_root = MicrographMembraneSubtract(
        str(star_path),
        output_root=str(mms_root),
        output_dirname=output_dirname,
        require_particle_mxt=True,
        strict_dependencies=True,
    )
    _expect_preflight_error(mms_wrong_root.preflight_check_dependencies, "missing_stack")

    # With explicit particle_output_root, preflight passes.
    mms_ok = MicrographMembraneSubtract(
        str(star_path),
        output_root=str(mms_root),
        particle_output_root=str(pms_root),
        output_dirname=output_dirname,
        require_particle_mxt=True,
        strict_dependencies=True,
    )
    mms_ok.preflight_check_dependencies()

    # Explicit invalid root should fail-fast.
    mms_bad_explicit = MicrographMembraneSubtract(
        str(star_path),
        output_root=str(mms_root),
        particle_output_root=str(tmp / "missing_root"),
        output_dirname=output_dirname,
        require_particle_mxt=True,
        strict_dependencies=True,
    )
    _expect_preflight_error(mms_bad_explicit.preflight_check_dependencies, "particle_output_root does not exist")

    # Non-strict mode keeps legacy best-effort behavior (no preflight exception).
    mms_legacy = MicrographMembraneSubtract(
        str(star_path),
        output_root=str(mms_root),
        output_dirname=output_dirname,
        require_particle_mxt=True,
        strict_dependencies=False,
    )
    mms_legacy.preflight_check_dependencies()

    # Strict mode checks output directory writability.
    readonly_root = tmp / "readonly"
    readonly_root.mkdir(parents=True, exist_ok=True)
    readonly_root.chmod(0o500)
    try:
        mms_unwritable = MicrographMembraneSubtract(
            str(star_path),
            output_root=str(readonly_root / "mms"),
            particle_output_root=str(pms_root),
            output_dirname=output_dirname,
            require_particle_mxt=True,
            strict_dependencies=True,
        )
        _expect_preflight_error(mms_unwritable.preflight_check_dependencies, "output directory is not writable")
    finally:
        # Best-effort restore permissions for cleanup.
        try:
            readonly_root.chmod(0o700)
        except Exception:
            pass

    print(">>> OK: mms dependency preflight.")


if __name__ == "__main__":
    main()
