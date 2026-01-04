from __future__ import annotations

from typing import Tuple


def check_cupy_cuda_available() -> Tuple[bool, str]:
    """
    Return (ok, details) for whether CuPy + CUDA appear usable.

    This intentionally imports CuPy only when called (i.e. from GPU-dependent UI
    actions), so the main GUI can start on CPU/login nodes.
    """

    try:
        import cupy as cp
    except Exception as exc:  # pragma: no cover - environment-dependent
        return False, f"Failed to import CuPy (`cupy`): {type(exc).__name__}: {exc}"

    try:
        device_count = cp.cuda.runtime.getDeviceCount()
    except Exception as exc:  # pragma: no cover - environment-dependent
        return False, f"CUDA runtime not available: {type(exc).__name__}: {exc}"

    if device_count < 1:
        return False, "No CUDA devices detected (deviceCount == 0)."

    return True, f"Detected {device_count} CUDA device(s)."
