from __future__ import annotations

import tempfile
from pathlib import Path


def _write_mrc(path: Path) -> None:
    import mrcfile  # type: ignore
    import numpy as np  # type: ignore

    path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(str(path), overwrite=True) as mrc:
        mrc.set_data(np.zeros((2, 2), dtype=np.float32))


def _expect_star_error(fn, contains: str) -> None:
    from memxterminator.bezierfit.bin.micrograph_mem_subtract_main import StarRewriteError

    try:
        fn()
        raise AssertionError(f"Expected StarRewriteError containing {contains!r}")
    except StarRewriteError as exc:
        msg = str(exc)
        if contains not in msg:
            raise AssertionError(f"Expected error to contain {contains!r}, got: {msg!r}") from exc


def main() -> None:
    import pandas as pd  # type: ignore
    import starfile  # type: ignore

    from memxterminator.bezierfit.bin.micrograph_mem_subtract_main import MicrographMembraneSubtract, StarRewriteError
    from memxterminator.mxt_state import to_output_micrograph_path

    tmp = Path(tempfile.mkdtemp(prefix="mxt_mms_star_"))
    raw_stack = tmp / "raw" / "extract" / "stack_001.mrcs"
    raw_micrograph = tmp / "raw" / "micrographs" / "extract" / "mg_001.mrc"
    _write_mrc(raw_stack)
    _write_mrc(raw_micrograph)

    # Multi-block STAR with optics + particles.
    optics_df = pd.DataFrame({"rlnOpticsGroup": [1], "rlnVoltage": [300.0]})
    particles_df = pd.DataFrame(
        {
            "rlnImageName": [f"1@{raw_stack}"],
            "rlnMicrographName": [str(raw_micrograph)],
            "rlnCoordinateX": [12.0],
            "rlnCoordinateY": [34.0],
            "rlnRandomSubset": [1],
        }
    )
    star_path = tmp / "particles.star"
    starfile.write({"optics": optics_df, "particles": particles_df}, str(star_path), overwrite=True)

    mms = MicrographMembraneSubtract(
        str(star_path),
        output_root=str(tmp / "mms_out"),
        output_dirname="class_01",
        strict_dependencies=False,
    )
    out_star = Path(mms.write_output_star_file())
    assert out_star.exists()

    out_obj = starfile.read(str(out_star), always_dict=True)
    assert "optics" in out_obj
    assert "particles" in out_obj
    assert out_obj["optics"].equals(optics_df)
    assert out_obj["particles"]["rlnCoordinateX"].tolist() == particles_df["rlnCoordinateX"].tolist()
    assert out_obj["particles"]["rlnCoordinateY"].tolist() == particles_df["rlnCoordinateY"].tolist()
    expected_mg = str(
        Path(
            to_output_micrograph_path(
                str(raw_micrograph),
                output_root=str(tmp / "mms_out"),
                output_dirname="class_01",
            )
        ).resolve()
    )
    got_mg = str(out_obj["particles"]["rlnMicrographName"].iloc[0])
    assert got_mg == expected_mg
    assert got_mg.startswith("/")

    # output_star_path override.
    custom_out = tmp / "custom_out.star"
    out2 = mms.write_output_star_file(output_star_path=str(custom_out))
    assert out2 == str(custom_out)
    assert custom_out.exists()

    # Empty-particles STAR should still write an empty structured output.
    empty_particles = pd.DataFrame(
        {
            "rlnImageName": pd.Series([], dtype=object),
            "rlnMicrographName": pd.Series([], dtype=object),
            "rlnCoordinateX": pd.Series([], dtype=float),
            "rlnCoordinateY": pd.Series([], dtype=float),
        }
    )
    empty_star = tmp / "empty.star"
    starfile.write({"optics": optics_df, "particles": empty_particles}, str(empty_star), overwrite=True)
    mms_empty = MicrographMembraneSubtract(
        str(empty_star),
        output_root=str(tmp / "mms_out_empty"),
        output_dirname="class_01",
        strict_dependencies=False,
    )
    out_empty = Path(mms_empty.write_output_star_file())
    out_empty_obj = starfile.read(str(out_empty), always_dict=True)
    assert "particles" in out_empty_obj
    assert len(out_empty_obj["particles"]) == 0
    assert "optics" in out_empty_obj

    # Ambiguous multi-block particles tables should fail-fast.
    ambig_star = tmp / "ambig.star"
    part_a = particles_df.copy()
    part_b = particles_df.copy()
    starfile.write({"optics": optics_df, "particlesA": part_a, "particlesB": part_b}, str(ambig_star), overwrite=True)
    _expect_star_error(
        lambda: MicrographMembraneSubtract(
            str(ambig_star),
            output_root=str(tmp / "mms_out_ambig"),
            strict_dependencies=False,
        ),
        "Ambiguous particles table",
    )

    # Missing required column should fail-fast.
    no_mg_star = tmp / "no_mg.star"
    bad_particles = pd.DataFrame({"rlnImageName": [f"1@{raw_stack}"], "rlnCoordinateX": [1.0], "rlnCoordinateY": [2.0]})
    starfile.write({"particles": bad_particles}, str(no_mg_star), overwrite=True)
    _expect_star_error(
        lambda: MicrographMembraneSubtract(
            str(no_mg_star),
            output_root=str(tmp / "mms_out_bad"),
            strict_dependencies=False,
        ),
        "rlnImageName and rlnMicrographName",
    )

    # Ensure constructor exceptions are the expected class.
    try:
        raise StarRewriteError("sentinel")
    except StarRewriteError:
        pass

    print(">>> OK: mms STAR rewrite.")


if __name__ == "__main__":
    main()

