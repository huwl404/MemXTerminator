"""
MemXTerminator `.mxt` sidecar state helpers (Spec v1).

This module provides small, reusable primitives for:

- Stable hashing of pixel-affecting parameters (`params_hash`)
- File fingerprints for cache invalidation
- Atomic writers for JSON and MRC outputs
- Idempotent mapping from `extract/` paths to `subtracted/` paths
- Up-to-date checks for resume logic
- Optional per-item lock files for parallel safety

Design constraints:
- Must be safe to import from CLI/worker processes.
- Must NOT import GUI dependencies (e.g. PyQt).
- Must NOT require a GPU at import time (no unconditional `cupy` import).
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Tuple


def fingerprint_file(path: os.PathLike[str] | str) -> dict[str, Any]:
    """
    Minimal file fingerprint used for cache invalidation.

    Required fields:
    - path: string (normalised)
    - size_bytes: int
    - mtime_ns: int
    """
    path_str = _normalise_path(path)
    st = os.stat(path_str)
    return {
        "path": path_str,
        "size_bytes": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def compute_params_hash(params: Mapping[str, Any]) -> str:
    """
    Compute a stable sha256 hex digest over `params` only.

    - Uses stable JSON serialisation: sort keys, compact separators.
    - Does NOT include MemXTerminator version in the hash by design.
    """
    serialised = json.dumps(
        params,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    )
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


def read_mxt(path: os.PathLike[str] | str) -> dict[str, Any]:
    """
    Read a `.mxt` JSON file (UTF-8) and return the parsed object.

    Callers should catch exceptions and treat failures as "stale".
    """
    path_str = _normalise_path(path)
    with open(path_str, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected .mxt JSON object at {path_str}, got {type(obj).__name__}")
    return obj


def write_json_atomic(path: os.PathLike[str] | str, obj: Any) -> None:
    """
    Atomically write JSON to `path`:

    - Write to a temp file in the same directory
    - flush + fsync
    - os.replace(tmp, path)
    """
    path_str = _normalise_path(path)
    parent = os.path.dirname(path_str)
    if parent:
        os.makedirs(parent, exist_ok=True)

    tmp_path = _tmp_path_for(path_str)
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path_str)
    finally:
        # Best-effort cleanup if something failed before replace.
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def write_mrc_atomic(path: os.PathLike[str] | str, array: Any) -> None:
    """
    Atomically write an MRC/MRCS file to `path`.

    Accepts numpy arrays or CuPy arrays; CuPy arrays must be moved to CPU via `.get()`.
    Writes as float32, matching existing MemXTerminator outputs.
    """
    # Imported lazily so `memxterminator.mxt_state` remains importable in minimal
    # environments (and doesn't require GPU libraries).
    import mrcfile  # type: ignore
    import numpy as np  # type: ignore

    path_str = _normalise_path(path)
    parent = os.path.dirname(path_str)
    if parent:
        os.makedirs(parent, exist_ok=True)

    arr = _to_numpy(array)
    arr = np.asarray(arr, dtype=np.float32)

    tmp_path = _tmp_path_for(path_str)
    try:
        with mrcfile.new(tmp_path, overwrite=True) as mrc:
            mrc.set_data(arr)
        os.replace(tmp_path, path_str)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def parse_relion_image_name_1based(image_name: str) -> Tuple[str, int]:
    """
    Parse a RELION image reference string and return a 1-based index.

    RELION commonly stores images as `N@path/to/stack.mrcs`, where `N` is a 1-based
    index into the stack. This function returns:

        (stack_path, idx_1based)

    Compatibility rule:
    - If no '@' exists, treat the string as a direct path and return (s, 1).
    """
    if image_name is None:
        raise ValueError("RELION image reference is None")
    s = str(image_name).strip()
    if s == "":
        raise ValueError("RELION image reference is empty")

    left, sep, right = s.partition("@")
    if not sep:
        return s, 1

    left = left.strip()
    right = right.strip()
    if left == "" or right == "":
        raise ValueError(f"Invalid RELION image reference (expected N@path): {s!r}")
    idx = int(left)
    if idx < 1:
        raise ValueError(f"RELION image index must be >= 1, got {idx} in {s!r}")
    return right, idx


def validate_output_dirname(output_dirname: os.PathLike[str] | str) -> str:
    """
    Validate a user-provided output directory name.

    Rules:
    - Must be a single path segment.
    - Must be non-empty.
    - Must not be `.` or `..`.
    - Must not contain path separators.
    """
    raw = os.fspath(output_dirname)
    name = str(raw).strip()
    if name == "":
        raise ValueError("output_dirname must be non-empty")
    if name in {".", ".."}:
        raise ValueError(f"Invalid output_dirname {name!r}: '.' and '..' are not allowed")
    if "/" in name or "\\" in name:
        raise ValueError(
            f"Invalid output_dirname {name!r}: expected a single path segment without path separators"
        )
    if Path(name).name != name:
        raise ValueError(
            f"Invalid output_dirname {name!r}: expected a single path segment without path separators"
        )
    return name


def to_output_stack_path(
    stack_path: os.PathLike[str] | str,
    *,
    output_dirname: os.PathLike[str] | str = "subtracted",
) -> str:
    """
    Map a particle stack path to an output stack path (idempotent).

    Rules:
    - If already in `<output_dirname>/` OR already ends with `_subtracted.mrc` /
      `_subtracted.mrcs`, return as-is.
    - Else:
      - Replace directory component `/extract/` -> `/<output_dirname>/`
      - Insert `_subtracted` before extension (`.mrc` or `.mrcs`)
    """
    out_dirname = validate_output_dirname(output_dirname)
    p = _normalise_path(stack_path)
    path_obj = Path(p)

    suffix = path_obj.suffix.lower()
    if suffix not in {".mrc", ".mrcs"}:
        return p

    stem_lower = path_obj.stem.lower()
    if stem_lower.endswith("_subtracted"):
        return p

    if out_dirname.lower() in {part.lower() for part in path_obj.parts}:
        return p

    parts = list(path_obj.parts)
    for i, part in enumerate(parts):
        if part.lower() == "extract":
            parts[i] = out_dirname
            break
    mapped_dir = Path(*parts[:-1])

    mapped_name = f"{path_obj.stem}_subtracted{suffix}"
    return str(mapped_dir / mapped_name)


def to_output_stack_path_in_root(
    stack_path: os.PathLike[str] | str,
    *,
    output_root: os.PathLike[str] | str | None = None,
    output_dirname: os.PathLike[str] | str = "subtracted",
) -> str:
    """
    Like `to_output_stack_path`, but optionally place outputs under `output_root`.

    Rules:
    - If `output_root` is None: return `to_output_stack_path(stack_path, output_dirname=...)`.
    - Else: preserve the relative subtree starting at `<output_dirname>/`
      if present; otherwise, place the mapped filename under
      `<output_root>/<output_dirname>/`.
    """
    out_dirname = validate_output_dirname(output_dirname)
    base = Path(to_output_stack_path(stack_path, output_dirname=out_dirname))
    if output_root is None:
        return str(base)

    out_root = Path(_normalise_path(output_root))
    parts_lower = [p.lower() for p in base.parts]
    try:
        sub_idx = parts_lower.index(out_dirname.lower())
    except ValueError:
        sub_idx = -1

    if sub_idx >= 0:
        return str(out_root / Path(*base.parts[sub_idx:]))
    return str(out_root / out_dirname / base.name)


def to_output_micrograph_path(
    micrograph_path: os.PathLike[str] | str,
    *,
    output_root: os.PathLike[str] | str | None = None,
    output_dirname: os.PathLike[str] | str = "subtracted",
) -> str:
    """
    Derive a micrograph output path for Bezierfit micrograph MMS.

    - Output name: `<stem>_subtracted<suffix>` (idempotent if already suffixed)
    - Default output dir: `<micrograph>/../.. / <output_dirname>`
    - If `output_root` is provided: `<output_root>/<output_dirname>/<name>`
    """
    out_dirname = validate_output_dirname(output_dirname)
    p = Path(_normalise_path(micrograph_path))
    suffix = p.suffix
    stem = p.stem
    if stem.lower().endswith("_subtracted"):
        out_name = f"{stem}{suffix}"
    else:
        out_name = f"{stem}_subtracted{suffix}"

    if output_root is None:
        out_dir = p.parent.parent / out_dirname
    else:
        out_dir = Path(_normalise_path(output_root)) / out_dirname

    return str(out_dir / out_name)


def to_subtracted_stack_path(stack_path: os.PathLike[str] | str) -> str:
    """
    Backward-compatible wrapper for `output_dirname="subtracted"`.
    """
    return to_output_stack_path(stack_path, output_dirname="subtracted")


def to_subtracted_stack_path_in_root(
    stack_path: os.PathLike[str] | str,
    *,
    output_root: os.PathLike[str] | str | None = None,
) -> str:
    """
    Backward-compatible wrapper for `output_dirname="subtracted"`.
    """
    return to_output_stack_path_in_root(stack_path, output_root=output_root, output_dirname="subtracted")


def to_subtracted_micrograph_path(
    micrograph_path: os.PathLike[str] | str,
    *,
    output_root: os.PathLike[str] | str | None = None,
) -> str:
    """
    Backward-compatible wrapper for `output_dirname="subtracted"`.
    """
    return to_output_micrograph_path(micrograph_path, output_root=output_root, output_dirname="subtracted")


def is_uptodate(
    output_path: os.PathLike[str] | str,
    mxt_path: os.PathLike[str] | str,
    expected_task: str,
    expected_params_hash: str,
    expected_inputs: Mapping[str, Any],
    *,
    strict_output_check: bool = True,
) -> tuple[bool, str]:
    """
    Determine whether an output is up-to-date given expected task, params hash, and inputs.

    This function must not throw: any exception results in (False, <reason>).
    """
    try:
        out_path = _normalise_path(output_path)
        sidecar_path = _normalise_path(mxt_path)

        if not os.path.exists(out_path):
            return False, "MISSING_OUTPUT"
        if not os.path.exists(sidecar_path):
            return False, "MISSING_MXT"

        try:
            mxt = read_mxt(sidecar_path)
        except Exception:
            return False, "INVALID_MXT_JSON"

        if mxt.get("task") != expected_task:
            return False, "TASK_MISMATCH"
        if mxt.get("status") != "success":
            return False, "STATUS_NOT_SUCCESS"
        if mxt.get("params_hash") != expected_params_hash:
            return False, "HASH_MISMATCH"

        inputs = mxt.get("inputs")
        if not isinstance(inputs, dict):
            return False, "MISSING_INPUTS"

        for key, expected_value in expected_inputs.items():
            if key not in inputs:
                return False, f"INPUT_MISSING:{key}"
            actual_value = inputs.get(key)
            if not _input_value_matches(expected_value, actual_value):
                return False, f"INPUT_MISMATCH:{key}"

        # Optional: if output fingerprint is recorded, require it to match the current output.
        output_meta = mxt.get("output")
        if isinstance(output_meta, dict):
            recorded_fp = output_meta.get("file")
            if isinstance(recorded_fp, dict) and "size_bytes" in recorded_fp and "mtime_ns" in recorded_fp:
                try:
                    cur_fp = fingerprint_file(out_path)
                except Exception:
                    return False, "OUTPUT_FINGERPRINT_STAT_FAILED"
                if not _fingerprint_matches(cur_fp, recorded_fp):
                    return False, "OUTPUT_FINGERPRINT_MISMATCH"

        if strict_output_check:
            try:
                import mrcfile  # type: ignore

                with mrcfile.open(out_path, permissive=True) as mrc:
                    shape = getattr(mrc, "data", None).shape
                if not shape or any(int(x) <= 0 for x in shape):
                    return False, "OUTPUT_CHECK_BAD_SHAPE"
            except Exception:
                return False, "OUTPUT_CHECK_FAILED"

        return True, "UPTODATE"
    except Exception:
        return False, "UPTODATE_CHECK_EXCEPTION"


def try_acquire_lock(lock_path: os.PathLike[str] | str, *, run_id: str | None = None) -> bool:
    """
    Try to acquire a per-item lock using an exclusive lock file.

    Returns True if the lock was acquired, False if the lock already exists.
    """
    lock_path_str = _normalise_path(lock_path)
    parent = os.path.dirname(lock_path_str)
    if parent:
        os.makedirs(parent, exist_ok=True)

    pid = os.getpid()
    now = datetime.now(timezone.utc).isoformat()
    run_id_val = run_id or f"{now}-{pid}-{secrets.token_hex(4)}"

    payload = {
        "pid": pid,
        "run_id": run_id_val,
        "started_utc": now,
    }

    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(lock_path_str, flags, 0o644)
    except FileExistsError:
        return False
    except OSError:
        # Conservative: treat inability to lock as "locked" to avoid duplicate work.
        return False

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
    finally:
        # fdopen context closes fd.
        pass

    return True


def release_lock(lock_path: os.PathLike[str] | str) -> None:
    """
    Release a lock file (best-effort).
    """
    lock_path_str = _normalise_path(lock_path)
    try:
        os.remove(lock_path_str)
    except FileNotFoundError:
        return
    except OSError:
        return


def _normalise_path(path: os.PathLike[str] | str) -> str:
    # Keep this simple and stable; callers compare fingerprints by size/mtime, not path.
    return os.fspath(path)


def _tmp_path_for(final_path: str) -> str:
    pid = os.getpid()
    token = secrets.token_hex(8)
    return f"{final_path}.tmp.{pid}.{token}"


def _to_numpy(array: Any) -> Any:
    """
    Convert numpy-like/cupy-like arrays to a numpy ndarray without importing cupy.
    """
    import numpy as np  # type: ignore

    if hasattr(array, "get") and callable(getattr(array, "get")):
        try:
            return array.get()
        except Exception:
            # Fall back to numpy conversion attempt below.
            pass
    return np.asarray(array)


def _json_default(obj: Any) -> Any:
    """
    JSON encoder fallback for common scientific-python types.
    """
    if isinstance(obj, Path):
        return str(obj)
    try:
        import numpy as np  # type: ignore

        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
    except Exception:
        # If numpy isn't available (or anything goes wrong), fall through to TypeError.
        pass
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def _fingerprint_matches(actual: Mapping[str, Any], recorded: Mapping[str, Any]) -> bool:
    """
    Compare two fingerprints using the required fields.
    """
    try:
        return (
            int(actual.get("size_bytes")) == int(recorded.get("size_bytes"))
            and int(actual.get("mtime_ns")) == int(recorded.get("mtime_ns"))
        )
    except Exception:
        return False


def _input_value_matches(expected: Any, actual: Any) -> bool:
    """
    Compare a single input entry from expected_inputs to the stored `.mxt.inputs` entry.

    - If expected looks like a file fingerprint (has size_bytes + mtime_ns), compare those fields.
    - Otherwise, compare by deep equality (dict/list/scalar).
    """
    if isinstance(expected, Mapping) and "size_bytes" in expected and "mtime_ns" in expected:
        if not isinstance(actual, Mapping):
            return False
        return _fingerprint_matches(expected, actual)

    return expected == actual


if __name__ == "__main__":
    # Minimal self-test (developer convenience; no external fixtures).
    assert parse_relion_image_name_1based("3@foo.mrcs") == ("foo.mrcs", 3)
    assert parse_relion_image_name_1based("foo.mrc") == ("foo.mrc", 1)
    assert validate_output_dirname("subtracted") == "subtracted"
    assert to_subtracted_stack_path("/a/extract/x.mrc") == "/a/subtracted/x_subtracted.mrc"
    assert to_subtracted_stack_path("/a/subtracted/x_subtracted.mrc") == "/a/subtracted/x_subtracted.mrc"
    assert to_output_stack_path("/a/extract/x.mrc", output_dirname="class_01") == "/a/class_01/x_subtracted.mrc"
    assert (
        to_subtracted_stack_path_in_root("/a/extract/x.mrc", output_root="/tmp/run1")
        == "/tmp/run1/subtracted/x_subtracted.mrc"
    )
    assert (
        to_output_stack_path_in_root("/a/extract/x.mrc", output_root="/tmp/run1", output_dirname="class_01")
        == "/tmp/run1/class_01/x_subtracted.mrc"
    )
    assert (
        to_subtracted_micrograph_path("/a/micrographs/extract/mg_001.mrc", output_root="/tmp/run2")
        == "/tmp/run2/subtracted/mg_001_subtracted.mrc"
    )
    assert (
        to_output_micrograph_path("/a/micrographs/extract/mg_001.mrc", output_root="/tmp/run2", output_dirname="class_01")
        == "/tmp/run2/class_01/mg_001_subtracted.mrc"
    )
    assert compute_params_hash({"b": 1, "a": 2}) == compute_params_hash({"a": 2, "b": 1})
    print("memxterminator.mxt_state: self-test OK")
