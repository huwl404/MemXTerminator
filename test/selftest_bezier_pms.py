from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def _write_mrc_stack(path: Path, *, n_images: int, box: int, seed: int) -> None:
    import mrcfile  # imported lazily so the script can print a nicer error early

    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_images, box, box), dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(str(path), overwrite=True) as mrc:
        mrc.set_data(data)


def _write_dummy_cs(path: Path, *, n_rows: int) -> None:
    """
    Write a minimal `.cs` file just so MemXTerminator can fingerprint it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"dummy cs file for selftest (rows={n_rows})\n", encoding="utf-8")


def _write_particles_cs(
    path: Path,
    *,
    stack_paths: list[Path],
    n_images_per_stack: int,
    pixel_size_a: float,
    class_ids: list[int],
) -> None:
    """
    Create a small CryoSPARC Dataset `.cs` with required columns for Bezierfit PMS.
    """
    from cryosparc.dataset import Dataset

    rows = []
    for stack_i, stack_path in enumerate(stack_paths):
        for idx in range(n_images_per_stack):
            class_id = class_ids[(stack_i * n_images_per_stack + idx) % len(class_ids)]
            rows.append((str(stack_path), idx, class_id))

    blob_path = np.asarray([r[0] for r in rows]).astype(str)
    blob_idx = np.asarray([r[1] for r in rows], dtype=np.int32)
    blob_psize = np.full((len(rows),), float(pixel_size_a), dtype=np.float32)

    # Use simple values to keep the pipeline stable and fast.
    pose = np.zeros((len(rows),), dtype=np.float32)  # alignments2D/pose
    shift = np.zeros((len(rows), 2), dtype=np.float32)  # alignments2D/shift
    cls = np.asarray([r[2] for r in rows], dtype=np.int32)  # alignments2D/class

    dset = Dataset(
        {
            "blob/path": blob_path,
            "blob/idx": blob_idx,
            "alignments2D/pose": pose,
            "blob/psize_A": blob_psize,
            "alignments2D/shift": shift,
            "alignments2D/class": cls,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    dset.save(str(path))


def _write_control_points(path: Path, *, box: int, class_ids: list[int]) -> None:
    # A gentle curve across the image (x,y in pixel coords).
    pts = [
        [float(box * 0.15), float(box * 0.50)],
        [float(box * 0.35), float(box * 0.35)],
        [float(box * 0.65), float(box * 0.65)],
        [float(box * 0.85), float(box * 0.50)],
    ]
    obj = {str(int(cid)): pts for cid in class_ids}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _run_mem_subtract(cmd: list[str], *, cwd: Path) -> None:
    env = os.environ.copy()
    # Ensure the subprocess uses the repo checkout (not an older installed version).
    repo_src = str(cwd / "src")
    env["PYTHONPATH"] = repo_src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    print("\n>>> Running:")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def _check_outputs(stack_paths: list[Path]) -> None:
    import mrcfile

    from memxterminator.mxt_state import to_subtracted_stack_path

    for raw_stack in stack_paths:
        out_stack = Path(to_subtracted_stack_path(str(raw_stack)))
        mxt_path = Path(str(out_stack) + ".mxt")

        assert out_stack.exists(), f"Missing output stack: {out_stack}"
        assert mxt_path.exists(), f"Missing .mxt sidecar: {mxt_path}"

        with mrcfile.open(str(raw_stack), permissive=True) as mrc:
            raw = np.asarray(mrc.data)
        with mrcfile.open(str(out_stack), permissive=True) as mrc:
            out = np.asarray(mrc.data)

        assert raw.shape == out.shape, f"Shape mismatch for {raw_stack}: {raw.shape} vs {out.shape}"
        assert np.isfinite(out).all(), f"NaN/Inf detected in output: {out_stack}"

        with open(mxt_path, "r", encoding="utf-8") as f:
            mxt = json.load(f)
        assert mxt.get("status") == "success", f"Unexpected mxt status for {out_stack}: {mxt.get('status')}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic self-test for Bezierfit particle membrane subtraction.")
    parser.add_argument("--num_stacks", type=int, default=3)
    parser.add_argument("--n_images", type=int, default=8, help="Images per stack (default: 8)")
    parser.add_argument("--box", type=int, default=96, help="Box size (default: 96)")
    parser.add_argument("--pixel_size", type=float, default=1.0)
    parser.add_argument("--procs", type=int, default=2, help="Pass-through to mem_subtract_main --procs (default: 2)")
    parser.add_argument("--batch_size", type=int, default=4, help="Pass-through to mem_subtract_main --batch_size (default: 4)")
    parser.add_argument("--keep", action="store_true", help="Keep the generated temp directory.")
    args = parser.parse_args()

    try:
        import cupy as cp  # noqa: F401
    except Exception as exc:
        raise SystemExit(f"CuPy import failed; run inside the mxt environment. Error: {exc}")

    # Place stacks under .../extract/ so output mapping goes to .../subtracted/.
    tmp_root = Path(tempfile.mkdtemp(prefix="mxt_bezier_pms_selftest_"))
    if args.keep:
        print(f">>> Using workdir: {tmp_root}")
    else:
        print(f">>> Using temp workdir: {tmp_root} (use --keep to preserve)")

    extract_dir = tmp_root / "extract"
    class_ids = [0, 1]

    stack_paths: list[Path] = []
    for i in range(int(args.num_stacks)):
        stack_path = extract_dir / f"stack_{i:02d}.mrc"
        _write_mrc_stack(stack_path, n_images=int(args.n_images), box=int(args.box), seed=1234 + i)
        stack_paths.append(stack_path)

    particle_cs = tmp_root / "particles_selected.cs"
    template_cs = tmp_root / "templates_selected.cs"
    control_points = tmp_root / "control_points.json"

    _write_particles_cs(
        particle_cs,
        stack_paths=stack_paths,
        n_images_per_stack=int(args.n_images),
        pixel_size_a=float(args.pixel_size),
        class_ids=class_ids,
    )
    _write_dummy_cs(template_cs, n_rows=int(args.num_stacks) * int(args.n_images))
    _write_control_points(control_points, box=int(args.box), class_ids=class_ids)

    repo_root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "memxterminator.bezierfit.bin.mem_subtract_main",
        "--particle",
        str(particle_cs),
        "--template",
        str(template_cs),
        "--control_points",
        str(control_points),
        # Keep self-test fast.
        "--points_step",
        "0.05",
        "--physical_membrane_dist",
        "5",
        "--procs",
        str(int(args.procs)),
        "--batch_size",
        str(int(args.batch_size)),
        "--force",
    ]
    _run_mem_subtract(cmd, cwd=repo_root)
    _check_outputs(stack_paths)

    if args.keep:
        print(f">>> Self-test OK. Outputs are under: {tmp_root}")
    else:
        # Best-effort cleanup.
        import shutil

        shutil.rmtree(tmp_root, ignore_errors=True)
        print(">>> Self-test OK. Temp directory removed.")


if __name__ == "__main__":
    main()
