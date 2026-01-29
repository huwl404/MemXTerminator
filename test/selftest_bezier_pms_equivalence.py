from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    # When this script is executed as `python test/...py`, sys.path[0] points to
    # the `test/` directory. Insert the repo root so `import test.*` resolves to
    # this checkout (not the stdlib `test` package).
    sys.path.insert(0, str(_REPO_ROOT))


def _run(cmd: list[str], *, cwd: Path) -> None:
    env = os.environ.copy()
    # Ensure the subprocess uses the repo checkout (not an older installed version).
    repo_src = str(cwd / "src")
    env["PYTHONPATH"] = repo_src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def _compare_outputs(*, out_a: Path, out_b: Path, tol_max: float, tol_mean: float) -> None:
    import mrcfile

    with mrcfile.open(str(out_a), permissive=True) as mrc:
        a = np.asarray(mrc.data, dtype=np.float32)
    with mrcfile.open(str(out_b), permissive=True) as mrc:
        b = np.asarray(mrc.data, dtype=np.float32)

    if a.shape != b.shape:
        raise AssertionError(f"Shape mismatch: {out_a} {a.shape} vs {out_b} {b.shape}")

    diff = np.abs(a - b)
    max_abs = float(diff.max(initial=0.0))
    mean_abs = float(diff.mean(dtype=np.float64))
    if max_abs > tol_max or mean_abs > tol_mean:
        raise AssertionError(
            f"Output mismatch for {out_a.name}: max_abs={max_abs:.6g} mean_abs={mean_abs:.6g} "
            f"(tols: max={tol_max} mean={tol_mean})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Equivalence self-test for Bezierfit PMS: procs A vs procs B.")
    parser.add_argument("--num_stacks", type=int, default=3)
    parser.add_argument("--n_images", type=int, default=8)
    parser.add_argument("--box", type=int, default=96)
    parser.add_argument("--pixel_size", type=float, default=1.0)
    parser.add_argument("--procs_a", type=int, default=1)
    parser.add_argument("--procs_b", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--tol_max", type=float, default=1e-4)
    parser.add_argument("--tol_mean", type=float, default=1e-6)
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()

    try:
        import cupy as cp  # noqa: F401
    except Exception as exc:
        raise SystemExit(f"CuPy import failed; run inside the mxt environment. Error: {exc}")

    # Reuse helper writers from the existing selftest module.
    from test.selftest_bezier_pms import (  # noqa: WPS433 (test-only imports)
        _check_outputs,
        _write_control_points,
        _write_dummy_cs,
        _write_mrc_stack,
        _write_particles_cs,
    )
    from memxterminator.mxt_state import to_subtracted_stack_path  # noqa: WPS433 (local import by design)

    tmp_root = Path(tempfile.mkdtemp(prefix="mxt_bezier_pms_equiv_"))
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
    cmd_base = [
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
        # Keep the self-test fast.
        "--points_step",
        "0.05",
        "--physical_membrane_dist",
        "5",
        "--batch_size",
        str(int(args.batch_size)),
        "--force",
    ]

    # Run A
    print(f">>> Running procs_a={int(args.procs_a)} ...")
    _run(cmd_base + ["--procs", str(int(args.procs_a))], cwd=repo_root)
    _check_outputs(stack_paths)

    subtracted_dir = tmp_root / "subtracted"
    snap_a = tmp_root / "subtracted_procs_a"
    if snap_a.exists():
        shutil.rmtree(snap_a)
    shutil.copytree(subtracted_dir, snap_a)

    # Run B (overwrite outputs in-place)
    print(f">>> Running procs_b={int(args.procs_b)} ...")
    _run(cmd_base + ["--procs", str(int(args.procs_b))], cwd=repo_root)
    _check_outputs(stack_paths)

    # Compare
    for raw_stack in stack_paths:
        out_stack = Path(to_subtracted_stack_path(str(raw_stack)))
        out_a = snap_a / out_stack.name
        out_b = subtracted_dir / out_stack.name
        _compare_outputs(out_a=out_a, out_b=out_b, tol_max=float(args.tol_max), tol_mean=float(args.tol_mean))

    print(">>> Equivalence OK.")

    if not args.keep:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
