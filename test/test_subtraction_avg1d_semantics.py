from __future__ import annotations

import numpy as np


def main() -> None:
    try:
        import cupy as cp
    except Exception as exc:
        raise SystemExit(f"CuPy import failed; run inside the mxt environment. Error: {exc}")

    from memxterminator.bezierfit.lib.subtraction import MembraneSubtract  # noqa: WPS433 (test-only import)

    image = cp.arange(100, dtype=cp.float32).reshape(10, 10)
    dummy_control_points = np.zeros((4, 2), dtype=np.float32)
    ms = MembraneSubtract(
        dummy_control_points,
        image,
        psi=0.0,
        origin_x=0.0,
        origin_y=0.0,
        pixel_size=1.0,
        points_step=0.1,
        physical_membrane_dist=2.0,
    )

    # Case 1: at least one membrane distance has zero in-bounds points.
    fitted_points = cp.asarray([[0.0, 0.0]], dtype=cp.float32)  # (x,y)
    normals = cp.asarray([[1.0, 1.0]], dtype=cp.float32)  # diagonal
    avg = ms.average_1d(image, fitted_points, normals, mem_dist=1).get()
    # Distances: -1, 0, +1
    assert avg.shape == (3,), f"Unexpected avg shape: {avg.shape}"
    assert float(avg[0]) == 0.0, f"Expected empty-mask mean=0.0, got {avg[0]}"
    assert float(avg[1]) == 0.0, f"Expected mean at (0,0)=0.0, got {avg[1]}"
    assert float(avg[2]) == 11.0, f"Expected mean at (1,1)=11.0, got {avg[2]}"

    # Case 2: OOB points must be excluded from the denominator.
    fitted_points = cp.asarray([[0.0, 0.0], [9.0, 9.0]], dtype=cp.float32)
    normals = cp.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=cp.float32)  # +x direction
    avg = ms.average_1d(image, fitted_points, normals, mem_dist=1).get()
    # For distance +1: points are (1,0) in-bounds and (10,9) OOB -> mean must be value(1,0)=1.0
    expected_d_plus_1 = 1.0
    got_d_plus_1 = float(avg[2])
    if abs(got_d_plus_1 - expected_d_plus_1) > 1e-5:
        raise AssertionError(f"OOB exclusion failed: expected {expected_d_plus_1}, got {got_d_plus_1}")

    print(">>> OK: average_1d empty-mask=0.0 and OOB exclusion semantics preserved.")


if __name__ == "__main__":
    main()

