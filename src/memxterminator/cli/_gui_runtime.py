from __future__ import annotations

import os
import sys


def _linux_x11_forwarding() -> bool:
    return sys.platform.startswith("linux") and bool(os.environ.get("DISPLAY"))


def configure_gui_runtime() -> None:
    """
    Apply conservative GUI defaults before importing PyQt.

    On Linux/X11 forwarding, Qt shared-memory can be fragile over SSH tunnels.
    Matplotlib backend selection is intentionally left untouched so it can follow
    the running Qt GUI framework.
    """
    if _linux_x11_forwarding():
        os.environ.setdefault("QT_X11_NO_MITSHM", "1")
