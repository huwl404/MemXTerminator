from __future__ import annotations


def main() -> None:
    try:
        import cupy as cp
    except Exception as exc:
        raise SystemExit(f"CuPy import failed; run inside the mxt environment. Error: {exc}")

    import numpy as np

    from memxterminator.bezierfit.lib.subtraction import _select_best_scaling_factor  # noqa: WPS433 (test-only import)

    # Create a synthetic tie case: mem_mask == 0 makes the objective identically 0 for all factors.
    image_conv = cp.zeros((8, 8), dtype=cp.float64)
    avg_conv = cp.ones((8, 8), dtype=cp.float64)
    mem_mask = cp.zeros((8, 8), dtype=cp.float64)

    scaling_factor_lst = list(np.arange(0.01, 1, 0.02))
    best_factor, best_idx = _select_best_scaling_factor(
        image_conv=image_conv,
        average_conv=avg_conv,
        mem_mask=mem_mask,
        scaling_factor_lst=scaling_factor_lst,
    )

    assert best_idx == 0, f"Expected first-min index 0, got {best_idx}"
    assert float(best_factor) == float(scaling_factor_lst[0]), f"Expected best_factor={scaling_factor_lst[0]}, got {best_factor}"

    print(">>> OK: scaling-factor selection preserves first-min tie-breaking.")


if __name__ == "__main__":
    main()

