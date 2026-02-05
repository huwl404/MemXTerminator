from __future__ import annotations

import tempfile
from pathlib import Path


def main() -> None:
    from memxterminator.path_resolve import infer_input_base_dir, resolve_path

    tmp = Path(tempfile.mkdtemp(prefix="mxt_path_resolve_"))
    proj = tmp / "P1"
    j220 = proj / "J220"
    (j220 / "extract").mkdir(parents=True, exist_ok=True)

    # Common CryoSPARC layout: primary input file under `.../J###/`.
    primary = j220 / "particles_selected.cs"
    assert infer_input_base_dir(str(primary)) == str(proj)

    # Non-job-dir: base dir is the file's directory.
    other = proj / "exports" / "particles_selected.cs"
    other.parent.mkdir(parents=True, exist_ok=True)
    assert infer_input_base_dir(str(other)) == str(other.parent)

    # resolve_path: relative uses base_dir; absolute remains unchanged.
    f = j220 / "extract" / "a.mrc"
    f.write_bytes(b"abc")
    resolved = resolve_path("J220/extract/a.mrc", base_dir=str(proj))
    assert resolved == str(f.resolve())
    assert Path(resolved).exists()

    resolved2 = resolve_path("J220/../J220/extract/a.mrc", base_dir=str(proj))
    assert resolved2 == str(f.resolve())

    abs_path = str(proj / "abs.mrc")
    assert resolve_path(abs_path, base_dir=str(proj)) == str(Path(abs_path).resolve())

    print(">>> OK: path base-dir inference/resolution.")


if __name__ == "__main__":
    main()
