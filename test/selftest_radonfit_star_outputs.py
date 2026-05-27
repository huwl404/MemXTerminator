from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def _write_mxt(path: Path, *, params_hash: str) -> None:
    obj = {
        "mxt_schema": 1,
        "task": "radonfit_particle_pms",
        "status": "success",
        "params_hash": params_hash,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def _stack_path_from_image_ref(image_ref: str) -> Path:
    return Path(str(image_ref).split("@", 1)[1])


def main() -> None:
    # Ensure the repo checkout is used (not an older installed version).
    repo_root = Path(__file__).resolve().parents[1]
    import sys

    sys.path.insert(0, str(repo_root / "src"))

    import starfile

    try:
        import cupy  # noqa: F401
    except Exception as exc:
        print(f">>> SKIP: Radonfit PMS module imports CuPy in this version. Error: {exc}")
        return

    from memxterminator.mxt_state import compute_params_hash, to_output_stack_path, to_subtracted_stack_path
    from memxterminator.radonfit.lib._utils import read_star_any
    import importlib

    # The Radonfit PMS entrypoint uses a hyphenated module name.
    pms_mod = importlib.import_module("memxterminator.radonfit.bin.membrane_subtract-main")

    tmp_root = Path(tempfile.mkdtemp(prefix="mxt_radonfit_star_selftest_"))
    extract_dir = tmp_root / "extract"
    sub_dir = tmp_root / "subtracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    sub_dir.mkdir(parents=True, exist_ok=True)

    raw_a = extract_dir / "stack_a.mrcs"
    raw_b = extract_dir / "stack_b.mrcs"
    _touch(raw_a)
    _touch(raw_b)

    # Input STAR (single-block).
    df_particles = pd.DataFrame(
        {
            "rlnImageName": [
                f"1@{raw_a}",
                f"2@{raw_a}",
                f"1@{raw_b}",
            ],
            "rlnMicrographName": ["mg_a.mrc"] * 3,
        }
    )
    in_star = tmp_root / "particles_selected.star"
    starfile.write(df_particles, str(in_star), overwrite=True)

    params_hash = compute_params_hash({"bias": 0.05})

    out_a = Path(to_subtracted_stack_path(str(raw_a)))
    out_b = Path(to_subtracted_stack_path(str(raw_b)))
    _touch(out_a)
    _touch(out_b)

    # Mark only stack_a as completed for this params_hash.
    _write_mxt(Path(str(out_a) + ".mxt"), params_hash=params_hash)

    out_all = tmp_root / "particles_selected_subtracted.star"
    out_completed = tmp_root / "particles_selected_subtracted_completed.star"
    pms_mod.write_radonfit_pms_star_outputs(
        input_star=str(in_star),
        expected_params_hash=params_hash,
        out_star_all=str(out_all),
        out_star_completed=str(out_completed),
        write_all=True,
        write_completed=True,
    )

    out_all_tables = read_star_any(str(out_all))
    out_all_df = next(iter(out_all_tables.values()))
    assert out_all_df.shape[0] == 3
    assert all("subtracted" in _stack_path_from_image_ref(s).parts for s in out_all_df["rlnImageName"].astype(str).tolist())
    assert out_all_df["rlnImageName"].iloc[0].endswith(f"@{out_a}")

    out_completed_tables = read_star_any(str(out_completed))
    out_completed_df = next(iter(out_completed_tables.values()))
    assert out_completed_df.shape[0] == 2
    assert all(str(out_a) in s for s in out_completed_df["rlnImageName"].astype(str).tolist())

    # Input STAR (multi-block optics+particles).
    df_optics = pd.DataFrame({"rlnOpticsGroup": [1], "rlnImagePixelSize": [1.0]})
    in_star_mb = tmp_root / "particles_selected_multiblock.star"
    starfile.write({"data_optics": df_optics, "data_particles": df_particles}, str(in_star_mb), overwrite=True)

    out_all_mb = tmp_root / "particles_selected_multiblock_subtracted.star"
    out_completed_mb = tmp_root / "particles_selected_multiblock_subtracted_completed.star"
    pms_mod.write_radonfit_pms_star_outputs(
        input_star=str(in_star_mb),
        expected_params_hash=params_hash,
        out_star_all=str(out_all_mb),
        out_star_completed=str(out_completed_mb),
        write_all=True,
        write_completed=True,
    )

    out_mb_tables = read_star_any(str(out_all_mb))
    assert any("optics" in name.lower() for name in out_mb_tables.keys()), f"Missing optics block: {list(out_mb_tables.keys())}"
    particles_block = None
    for name, df in out_mb_tables.items():
        if "rlnImageName" in df.columns:
            particles_block = df
            break
    assert particles_block is not None, "Missing particles block in output STAR"
    assert all("subtracted" in _stack_path_from_image_ref(s).parts for s in particles_block["rlnImageName"].astype(str).tolist())

    # Custom output_dirname: STAR paths must point at the requested output folder,
    # and completed-only filtering must check the matching .mxt sidecar there.
    custom_dirname = "subtracted_b005"
    out_a_custom = Path(to_output_stack_path(str(raw_a), output_dirname=custom_dirname))
    out_b_custom = Path(to_output_stack_path(str(raw_b), output_dirname=custom_dirname))
    _touch(out_a_custom)
    _touch(out_b_custom)
    _write_mxt(Path(str(out_b_custom) + ".mxt"), params_hash=params_hash)

    out_all_custom = tmp_root / f"particles_selected_{custom_dirname}.star"
    out_completed_custom = tmp_root / f"particles_selected_{custom_dirname}_completed.star"
    pms_mod.write_radonfit_pms_star_outputs(
        input_star=str(in_star),
        expected_params_hash=params_hash,
        out_star_all=str(out_all_custom),
        out_star_completed=str(out_completed_custom),
        output_dirname=custom_dirname,
        write_all=True,
        write_completed=True,
    )
    custom_all_df = next(iter(read_star_any(str(out_all_custom)).values()))
    assert all(
        custom_dirname in _stack_path_from_image_ref(s).parts
        for s in custom_all_df["rlnImageName"].astype(str).tolist()
    )
    assert custom_all_df["rlnImageName"].iloc[0].endswith(f"@{out_a_custom}")
    custom_completed_df = next(iter(read_star_any(str(out_completed_custom)).values()))
    assert custom_completed_df.shape[0] == 1
    assert custom_completed_df["rlnImageName"].iloc[0].endswith(f"@{out_b_custom}")

    print(f">>> Self-test OK. Outputs are under: {tmp_root}")


if __name__ == "__main__":
    main()
