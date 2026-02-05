from __future__ import annotations

"""
Path resolution helpers for CryoSPARC-style relative paths.

Problem:
Many CryoSPARC datasets (.cs) and RELION STAR files contain relative paths such
as `J220/extract/...`. In batch mode, MemXTerminator's scheduler runs each job
with `cwd=<job_output_root>`, which breaks those relative paths.

This module provides:
- A robust default base-dir inference for common CryoSPARC layouts
- Deterministic resolution of relative paths independent of `cwd`

Design constraints:
- Standard library only (safe to import in worker processes)
- Fail-fast: callers should validate resolved paths exist and raise a clear error
"""

import os
import re
from pathlib import Path

_CRYOSPARC_JOB_DIR_RE = re.compile(r"^J\d+$")


def _expand_path_str(path_str: str) -> str:
    return os.path.expanduser(os.path.expandvars(str(path_str)))


def normalise_dir(path: os.PathLike[str] | str) -> str:
    """
    Normalize a directory path to an absolute resolved string.

    - Expands env vars and '~'
    - If relative: interpret relative to current working directory
    - Returns a resolved absolute path (does not require existence)
    """
    s = str(path).strip()
    if s == "":
        raise ValueError("Directory path must be non-empty")
    p = Path(_expand_path_str(s))
    if not p.is_absolute():
        p = (Path.cwd() / p)
    return str(p.resolve())


def infer_input_base_dir(primary_input_file: os.PathLike[str] | str) -> str:
    """
    Infer a base directory for resolving CryoSPARC/RELION relative paths.

    Heuristic (covers the common CryoSPARC layout):
    - If the primary input file lives in a directory named like `J<digits>`
      (e.g. `/.../P1/J220/particles_selected.cs`), use its parent directory as
      the base dir (e.g. `/.../P1`), because embedded paths are often like
      `J220/extract/...`.
    - Otherwise, use the directory containing the primary input file.
    """
    s = str(primary_input_file).strip()
    if s == "":
        raise ValueError("primary_input_file must be non-empty")

    p = Path(_expand_path_str(s))
    if not p.is_absolute():
        p = (Path.cwd() / p)
    p = p.resolve()

    parent = p.parent
    if _CRYOSPARC_JOB_DIR_RE.match(parent.name or ""):
        return str(parent.parent)
    return str(parent)


def resolve_path(path: os.PathLike[str] | str, *, base_dir: os.PathLike[str] | str) -> str:
    """
    Resolve a possibly-relative path against `base_dir`.

    - Expands env vars and '~' in `path`
    - Absolute paths are returned unchanged (except expansion)
    - Relative paths are interpreted as `<base_dir>/<path>`
    - Does not check existence; callers should validate and fail-fast
    """
    raw = str(path).strip()
    if raw == "":
        raise ValueError("path must be non-empty")
    expanded = _expand_path_str(raw)

    p = Path(expanded)
    if p.is_absolute():
        return str(p.resolve())

    base = Path(_expand_path_str(str(base_dir)))
    return str((base / p).resolve())
