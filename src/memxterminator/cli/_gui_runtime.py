from __future__ import annotations

import os
import sys


def _linux_x11_forwarding() -> bool:
    return sys.platform.startswith("linux") and bool(os.environ.get("DISPLAY"))


def configure_gui_runtime() -> None:
    """
    Apply conservative GUI defaults before importing PyQt or matplotlib.pyplot.

    On Linux/X11 forwarding, Qt/Matplotlib may try to use shared-memory or heavy
    Qt/GTK Matplotlib backends. Those choices are fragile over SSH tunnels.
    """
    if _linux_x11_forwarding():
        os.environ.setdefault("QT_X11_NO_MITSHM", "1")

    if os.environ.get("MPLBACKEND"):
        return

    requested = os.environ.get("MXT_MPL_BACKEND")
    if requested:
        backend = requested
    elif _linux_x11_forwarding():
        backend = "TkAgg"
    elif sys.platform.startswith("linux"):
        backend = "Agg"
    else:
        return

    if backend.lower() == "tkagg":
        try:
            import tkinter  # noqa: F401
        except Exception as exc:
            print(f">>> WARNING: TkAgg is unavailable ({type(exc).__name__}: {exc}); falling back to Agg.")
            backend = "Agg"

    try:
        import matplotlib

        matplotlib.use(backend, force=True)
        os.environ["MPLBACKEND"] = backend
        print(f">>> Matplotlib backend set to {backend} for MemXTerminator GUI.")
    except Exception as exc:
        os.environ["MPLBACKEND"] = "Agg"
        try:
            import matplotlib

            matplotlib.use("Agg", force=True)
        except Exception:
            pass
        print(f">>> WARNING: failed to set Matplotlib backend {backend!r}: {exc}. Falling back to Agg.")
