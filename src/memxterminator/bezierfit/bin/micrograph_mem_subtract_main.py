from __future__ import annotations

import argparse
import json
import os
import socket
import time
import traceback
from collections.abc import Mapping
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import mrcfile
import multiprocessing
import starfile
from setproctitle import setproctitle

from memxterminator.mxt_state import (
    compute_params_hash,
    fingerprint_file,
    is_uptodate,
    parse_relion_image_name_1based,
    read_mxt,
    release_lock,
    to_output_micrograph_path,
    to_output_stack_path_in_root,
    try_acquire_lock,
    validate_output_dirname,
    write_json_atomic,
    write_mrc_atomic,
)
from memxterminator.path_resolve import infer_input_base_dir, normalise_dir, resolve_path

_WORKER_RUNNER = None
_EVENT_LOG_PATH: str | None = None


class DependencyPreflightError(RuntimeError):
    pass


class DependencyRuntimeError(RuntimeError):
    pass


class StarRewriteError(RuntimeError):
    pass


def _init_worker(config: dict) -> None:
    """
    Per-process initializer for multiprocessing workers.

    We build the runner inside each worker to avoid pickling large state (STAR
    DataFrames, cached weights/masks, etc.) for every task.
    """
    global _WORKER_RUNNER
    global _EVENT_LOG_PATH

    # Best-effort: pin each worker to a GPU in round-robin order if multiple
    # CUDA devices are visible. Users can still control visibility via
    # CUDA_VISIBLE_DEVICES.
    try:
        import cupy as cp

        device_count = int(cp.cuda.runtime.getDeviceCount())
    except Exception:
        device_count = 0
    if device_count > 0:
        try:
            ident = getattr(multiprocessing.current_process(), "_identity", ())
            worker_rank = int(ident[0]) if ident else 1  # 1-based in multiprocessing pools
            device_id = (worker_rank - 1) % device_count
            cp.cuda.Device(device_id).use()
        except Exception:
            pass

    output_root = config.get("output_root")
    if output_root:
        try:
            _EVENT_LOG_PATH = os.fspath(output_root) + os.sep + "mms_run_data.log"
        except Exception:
            _EVENT_LOG_PATH = None
    else:
        _EVENT_LOG_PATH = "mms_run_data.log"

    _WORKER_RUNNER = MicrographMembraneSubtract(**config)


def _process_micrograph_worker(raw_mg_name: tuple[str, str]) -> None:
    if _WORKER_RUNNER is None:
        raise RuntimeError("Worker runner not initialized")
    _WORKER_RUNNER.process_micrograph_mem_subtract(raw_mg_name)


def _append_event_log(level: str, event: str, **fields: object) -> None:
    """
    Append a single human-readable line to `mms_run_data.log`.

    This file is never read for resume; `.mxt` sidecars are authoritative (Spec v1).
    """
    ts = datetime.now(timezone.utc).isoformat()
    parts = [ts, level, event]
    for key, value in fields.items():
        try:
            encoded = json.dumps(value, ensure_ascii=False)
        except Exception:
            encoded = repr(value)
        parts.append(f"{key}={encoded}")

    line = " ".join(parts)
    try:
        path = _EVENT_LOG_PATH or "mms_run_data.log"
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        # Best-effort: logging must never crash workers.
        pass


def _read_star_object(star_path: str):
    """
    Read STAR content and preserve block structure when possible.

    Supports single-block and multi-block STAR files.
    """
    try:
        return starfile.read(star_path, always_dict=True)
    except TypeError:
        return starfile.read(star_path)


def _select_particles_table(star_obj, *, star_path: str):
    """
    Select the unique particles table block containing required columns.

    Candidate rule:
    - DataFrame with both `rlnImageName` and `rlnMicrographName`.
    """
    import pandas as pd  # type: ignore

    if isinstance(star_obj, pd.DataFrame):
        cols = set(str(c) for c in star_obj.columns)
        if {"rlnImageName", "rlnMicrographName"}.issubset(cols):
            return None, star_obj
        if len(star_obj) == 0:
            df_empty = star_obj.copy()
            if "rlnImageName" not in df_empty.columns:
                df_empty["rlnImageName"] = pd.Series([], dtype=object)
            if "rlnMicrographName" not in df_empty.columns:
                df_empty["rlnMicrographName"] = pd.Series([], dtype=object)
            return None, df_empty
        raise StarRewriteError(
            "STAR particles table must contain columns: rlnImageName and rlnMicrographName.\n"
            f"  star_path={star_path}"
        )

    if isinstance(star_obj, Mapping):
        candidates: list[tuple[str, object]] = []
        empty_particle_like: list[tuple[str, object]] = []
        for block_name, block in star_obj.items():
            if not isinstance(block, pd.DataFrame):
                continue
            cols = set(str(c) for c in block.columns)
            if {"rlnImageName", "rlnMicrographName"}.issubset(cols):
                candidates.append((str(block_name), block))
                continue
            if len(block) == 0 and "particle" in str(block_name).lower():
                empty_particle_like.append((str(block_name), block))

        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) == 0:
            if len(empty_particle_like) == 1:
                block_name, block = empty_particle_like[0]
                block_empty = block.copy()
                if "rlnImageName" not in block_empty.columns:
                    block_empty["rlnImageName"] = pd.Series([], dtype=object)
                if "rlnMicrographName" not in block_empty.columns:
                    block_empty["rlnMicrographName"] = pd.Series([], dtype=object)
                return block_name, block_empty
            if len(empty_particle_like) > 1:
                names = [name for name, _ in empty_particle_like]
                raise StarRewriteError(
                    "Ambiguous particles table: multiple empty particle-like STAR blocks found.\n"
                    f"  star_path={star_path}\n"
                    f"  candidate_blocks={names}"
                )
            raise StarRewriteError(
                "Could not find a particles table containing both rlnImageName and rlnMicrographName.\n"
                f"  star_path={star_path}"
            )
        names = [name for name, _ in candidates]
        raise StarRewriteError(
            "Ambiguous particles table: multiple STAR blocks contain both rlnImageName and rlnMicrographName.\n"
            f"  star_path={star_path}\n"
            f"  candidate_blocks={names}"
        )

    raise StarRewriteError(f"Unsupported STAR content for {star_path!r}: {type(star_obj).__name__}")


def _read_particles_table(star_path: str):
    """
    Read a STAR file and return the selected particle table DataFrame.
    """
    star_obj = _read_star_object(star_path)
    _block_name, df = _select_particles_table(star_obj, star_path=star_path)
    return df

class MicrographMembraneSubtract:
    def __init__(
        self,
        particles_selected_filename: str,
        *,
        input_base_dir: str | None = None,
        resume: bool = True,
        force: bool = False,
        adopt_existing_outputs: bool = False,
        skip_failed: bool = False,
        require_particle_mxt: bool = True,
        strict_dependencies: bool = True,
        strict_output_check: bool = True,
        output_root: str | None = None,
        particle_output_root: str | None = None,
        output_dirname: str = "subtracted",
        write_output_star: bool = True,
        output_star_path: str | None = None,
    ):
        self.particles_selected_filename = particles_selected_filename

        if input_base_dir is not None and str(input_base_dir).strip() != "":
            self.input_base_dir = normalise_dir(input_base_dir)
        else:
            self.input_base_dir = infer_input_base_dir(self.particles_selected_filename)

        self.resume = bool(resume)
        self.force = bool(force)
        self.adopt_existing_outputs = bool(adopt_existing_outputs)
        self.skip_failed = bool(skip_failed)
        self.require_particle_mxt = bool(require_particle_mxt)
        self.strict_dependencies = bool(strict_dependencies)
        self.strict_output_check = bool(strict_output_check)
        self.output_root = output_root
        self.particle_output_root = (
            normalise_dir(particle_output_root)
            if (particle_output_root is not None and str(particle_output_root).strip() != "")
            else None
        )
        self.output_dirname = validate_output_dirname(output_dirname)
        self.write_output_star = bool(write_output_star)
        self.output_star_path = output_star_path

        # `.mxt` stable params/hash for this invocation (pixel-affecting only).
        self.mxt_task = "bezierfit_micrograph_mms"
        self.mxt_params = {
            "weighting": "gaussian",
            "gaussian_sigma": 0.6,
            "epsilon": 1e-10,
        }
        self.mxt_params_hash = compute_params_hash(self.mxt_params)

        self._host = socket.gethostname()
        try:
            import memxterminator  # noqa: WPS433 (local import by design)

            self._software_version = getattr(memxterminator, "__version__", "unknown")
        except Exception:
            self._software_version = "unknown"

        self._particles_star_fp = fingerprint_file(self.particles_selected_filename)

        self.df_star = _read_particles_table(self.particles_selected_filename)

        parsed = [parse_relion_image_name_1based(x) for x in self.df_star["rlnImageName"].tolist()]
        self.df_star = self.df_star.copy()
        self.df_star["_memx_stack_path"] = [resolve_path(p, base_dir=self.input_base_dir) for p, _ in parsed]
        self.df_star["_memx_section_1based"] = [i for _, i in parsed]

        self.rawimage_stacks_name_lst = self.get_rawimage_stacks_name_lst()
        self.raw_mg_name_lst = self.get_raw_stack_micrograph_pairs()

        # Fail-fast when the base dir is clearly wrong (common batch misconfiguration).
        if self.rawimage_stacks_name_lst and not any(os.path.exists(p) for p in self.rawimage_stacks_name_lst):
            raise SystemExit(
                "ERROR: None of the particle stacks referenced in the STAR file could be found on disk.\n"
                f"  particles_selected_filename: {self.particles_selected_filename}\n"
                f"  inferred input_base_dir: {self.input_base_dir}\n"
                "This usually means the STAR contains relative paths (e.g. 'J220/extract/...') and the base directory is wrong.\n\n"
                "Fix: re-run with an explicit base directory, for example:\n"
                "  --input_base_dir /path/to/cryosparc_project_root\n"
            )
        if self.raw_mg_name_lst and not any(os.path.exists(mg) for _stack, mg in self.raw_mg_name_lst):
            raise SystemExit(
                "ERROR: None of the micrographs referenced in the STAR file could be found on disk.\n"
                f"  particles_selected_filename: {self.particles_selected_filename}\n"
                f"  inferred input_base_dir: {self.input_base_dir}\n"
                "Fix: re-run with an explicit base directory, for example:\n"
                "  --input_base_dir /path/to/cryosparc_project_root\n"
            )

        self.rawimage_size: int | None = None
        self.weights = None
        self.mask = None

    def get_rawimage_stacks_name_lst(self):
        rawimage_lst = list(self.df_star["_memx_stack_path"])
        rawimage_name_lst = list(dict.fromkeys(rawimage_lst))
        return rawimage_name_lst

    def get_raw_stack_micrograph_pairs(self) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        for stack_path in self.rawimage_stacks_name_lst:
            df_temp = self.get_df_temp(stack_path)
            micrographs = list(dict.fromkeys(df_temp["rlnMicrographName"].astype(str).tolist()))
            if not micrographs:
                raise ValueError(f"Missing rlnMicrographName entries for stack: {stack_path}")
            if len(micrographs) != 1:
                raise ValueError(
                    f"Expected exactly 1 micrograph per particle stack, got {len(micrographs)} for {stack_path}: {micrographs[:3]}"
                )
            mg = str(micrographs[0]).strip()
            if mg == "":
                raise ValueError(f"Empty rlnMicrographName for stack: {stack_path}")
            pairs.append((stack_path, resolve_path(mg, base_dir=self.input_base_dir)))
        return pairs

    def _dependency_output_root(self) -> str | None:
        return self.particle_output_root if self.particle_output_root is not None else self.output_root

    def _dependency_stack_path(self, raw_stack: str) -> str:
        return to_output_stack_path_in_root(
            raw_stack,
            output_root=self._dependency_output_root(),
            output_dirname=self.output_dirname,
        )

    def _raise_dependency_failure(self, issues: list[str], *, heading: str) -> None:
        preview_n = 12
        lines = [
            heading,
            f"output_root={self.output_root!r}",
            f"particle_output_root={self.particle_output_root!r}",
            f"output_dirname={self.output_dirname!r}",
            f"missing_count={len(issues)}",
            "missing_entries:",
        ]
        for issue in issues[:preview_n]:
            lines.append(f"  - {issue}")
        if len(issues) > preview_n:
            lines.append(f"  - ... and {len(issues) - preview_n} more")
        lines.append(
            "Hint: set scheduler depends_on for MMS->PMS, or pass --particle_output_root to the PMS output root."
        )
        raise DependencyPreflightError("\n".join(lines))

    def preflight_check_dependencies(self) -> None:
        """
        Fail-fast dependency validation before multiprocessing starts.
        """
        if not self.strict_dependencies:
            return

        dep_root = self._dependency_output_root()
        if self.particle_output_root is not None:
            if not os.path.exists(self.particle_output_root):
                self._raise_dependency_failure(
                    [f"particle_output_root_missing path={self.particle_output_root}"],
                    heading="Dependency preflight failed: particle_output_root does not exist.",
                )
            if not os.path.isdir(self.particle_output_root):
                self._raise_dependency_failure(
                    [f"particle_output_root_not_directory path={self.particle_output_root}"],
                    heading="Dependency preflight failed: particle_output_root is not a directory.",
                )

        issues: list[str] = []
        seen_stacks: set[str] = set()
        for raw_stack in self.rawimage_stacks_name_lst:
            raw_stack_s = str(raw_stack)
            if raw_stack_s in seen_stacks:
                continue
            seen_stacks.add(raw_stack_s)

            dep_stack = self._dependency_stack_path(raw_stack_s)
            dep_mxt = dep_stack + ".mxt"
            if not os.path.exists(dep_stack):
                issues.append(f"missing_stack path={dep_stack}")
                continue

            if self.require_particle_mxt:
                if not os.path.exists(dep_mxt):
                    issues.append(f"missing_mxt path={dep_mxt}")
                    continue
                try:
                    dep_obj = read_mxt(dep_mxt)
                except Exception as exc:
                    issues.append(f"invalid_mxt path={dep_mxt} error={type(exc).__name__}")
                    continue
                if dep_obj.get("task") != "bezierfit_particle_pms":
                    issues.append(f"mxt_wrong_task path={dep_mxt} task={dep_obj.get('task')!r}")
                    continue
                if dep_obj.get("status") != "success":
                    issues.append(f"mxt_not_success path={dep_mxt} status={dep_obj.get('status')!r}")
                    continue
                dep_hash = dep_obj.get("params_hash")
                if not isinstance(dep_hash, str) or dep_hash.strip() == "":
                    issues.append(f"mxt_missing_hash path={dep_mxt}")

        if issues:
            self._raise_dependency_failure(issues, heading="Dependency preflight failed.")

        # Validate micrograph inputs once up front in strict mode.
        missing_micrographs = [mg for _stack, mg in self.raw_mg_name_lst if not os.path.exists(mg)]
        if missing_micrographs:
            issues = [f"missing_micrograph path={p}" for p in missing_micrographs[:256]]
            self._raise_dependency_failure(issues, heading="Dependency preflight failed: missing input micrograph files.")

        # Validate output directory writability once up front in strict mode.
        output_dirs: set[str] = set()
        for _raw_stack, micrograph_path in self.raw_mg_name_lst:
            out_micrograph = to_output_micrograph_path(
                micrograph_path,
                output_root=self.output_root,
                output_dirname=self.output_dirname,
            )
            output_dirs.add(str(Path(out_micrograph).parent))

        issues = []
        for out_dir_str in sorted(output_dirs):
            out_dir = Path(out_dir_str)
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
                probe = out_dir / f".mms_preflight_write_probe_{os.getpid()}"
                with open(probe, "w", encoding="utf-8") as f:
                    f.write("ok\n")
                try:
                    probe.unlink()
                except Exception:
                    pass
            except Exception as exc:
                issues.append(f"cannot_create_output_dir path={out_dir} error={type(exc).__name__}: {exc}")
        if issues:
            self._raise_dependency_failure(
                issues[:256],
                heading="Dependency preflight failed: output directory is not writable.",
            )

    def _default_output_star_path(self) -> str:
        input_star = Path(self.particles_selected_filename)
        out_name = f"{input_star.stem}_mms_micrograph_{self.output_dirname}.star"
        if self.output_root is not None and str(self.output_root).strip() != "":
            return str(Path(self.output_root) / out_name)
        return str(input_star.parent / out_name)

    def write_output_star_file(self, output_star_path: str | None = None) -> str:
        """
        Rewrite `rlnMicrographName` to the mapped output path and write STAR.
        """
        star_obj = _read_star_object(self.particles_selected_filename)
        block_name, df_particles = _select_particles_table(star_obj, star_path=self.particles_selected_filename)

        if "rlnMicrographName" not in df_particles.columns:
            raise StarRewriteError(
                "Particles table is missing required column rlnMicrographName.\n"
                f"  star_path={self.particles_selected_filename}"
            )

        rewritten = df_particles.copy()
        mapped_micrographs: list[str] = []
        for raw_mg in rewritten["rlnMicrographName"].astype(str).tolist():
            mg = str(raw_mg).strip()
            if mg == "":
                raise StarRewriteError(
                    "Particles table contains empty rlnMicrographName entries.\n"
                    f"  star_path={self.particles_selected_filename}"
                )
            mg_abs = resolve_path(mg, base_dir=self.input_base_dir)
            mapped_path = to_output_micrograph_path(
                mg_abs,
                output_root=self.output_root,
                output_dirname=self.output_dirname,
            )
            mapped_micrographs.append(os.path.abspath(mapped_path))
        rewritten["rlnMicrographName"] = mapped_micrographs

        target = output_star_path if (output_star_path is not None and str(output_star_path).strip() != "") else self.output_star_path
        if target is None or str(target).strip() == "":
            target = self._default_output_star_path()
        target_path = Path(str(target))
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(star_obj, Mapping):
            out_obj = dict(star_obj)
            if block_name is None:
                raise StarRewriteError("Internal error: missing particles table block name for multi-block STAR.")
            out_obj[block_name] = rewritten
            starfile.write(out_obj, str(target_path), overwrite=True)
        else:
            starfile.write(rewritten, str(target_path), overwrite=True)
        return str(target_path)

    def get_df_temp(self, rawimage_stacks_name):
        df_temp = self.df_star[self.df_star["_memx_stack_path"] == rawimage_stacks_name]
        return df_temp
    def get_df_temp_X_Y(self, df_temp):
        rawimage_sections_lst = list(map(int, list(df_temp["_memx_section_1based"])))
        CoordinateX_lst = list(df_temp['rlnCoordinateX'])
        CoordinateY_lst = list(df_temp['rlnCoordinateY'])
        zip_lst = list(zip(CoordinateX_lst, CoordinateY_lst))
        df_temp_dict_X_Y = dict(zip(rawimage_sections_lst, zip_lst))
        return df_temp_dict_X_Y
    
    def fill_nan_with_gaussian_noise(self, image):
        import cupy as cp

        image_copy = image.copy()
        mean_val = cp.nanmean(image_copy)
        std_val = cp.nanstd(image_copy)
        nan_mask = cp.isnan(image_copy)
        noise = cp.random.normal(mean_val, std_val, image_copy.shape)
        image_copy[nan_mask] = noise[nan_mask]
        return image_copy

    def _ensure_weights_for_box(self, rawimage_size: int) -> None:
        import cupy as cp

        if self.rawimage_size == int(rawimage_size) and self.weights is not None and self.mask is not None:
            return
        self.rawimage_size = int(rawimage_size)
        sigma = float(self.mxt_params["gaussian_sigma"])
        x, y = cp.meshgrid(
            cp.linspace(-1, 1, self.rawimage_size),
            cp.linspace(-1, 1, self.rawimage_size),
        )
        d = cp.sqrt(x * x + y * y)
        self.weights = cp.exp(-(d**2) / (2.0 * sigma**2))
        self.mask = cp.ones((self.rawimage_size, self.rawimage_size))

    def _output_passes_sanity_check(self, out_path: str) -> bool:
        try:
            with mrcfile.open(out_path, permissive=True) as mrc:
                shape = getattr(mrc, "data", None).shape
            return bool(shape) and all(int(x) > 0 for x in shape)
        except Exception:
            return False

    def _build_mxt_object(
        self,
        *,
        status: str,
        out_micrograph: str,
        expected_inputs: dict,
        started_utc: str,
        finished_utc: str,
        duration_sec: float,
        adopted: bool,
        run_id: str,
        error=None,
    ) -> dict:
        output_obj: dict[str, object] = {"path": out_micrograph}
        if os.path.exists(out_micrograph):
            try:
                output_obj["file"] = fingerprint_file(out_micrograph)
            except Exception:
                pass

        obj = {
            "mxt_schema": 1,
            "task": self.mxt_task,
            "status": status,
            "params": self.mxt_params,
            "params_hash": self.mxt_params_hash,
            "inputs": expected_inputs,
            "output": output_obj,
            "run": {
                "run_id": run_id,
                "started_utc": started_utc,
                "finished_utc": finished_utc,
                "duration_sec": float(duration_sec),
                "pid": int(os.getpid()),
                "host": self._host,
                "software_version": self._software_version,
                "git_commit": None,
            },
            "adopted": bool(adopted),
        }
        if error is not None:
            obj["error"] = error
        return obj
    
    def process_micrograph_mem_subtract(self, raw_mg_name):
        setproctitle("MemXTerminator-MMS")
        rawimage_stacks_name, micrograph_name = raw_mg_name

        particle_stack_subtracted_path = self._dependency_stack_path(rawimage_stacks_name)
        particle_stack_mxt_path = particle_stack_subtracted_path + ".mxt"

        if not os.path.exists(particle_stack_subtracted_path):
            message = (
                f">>> BLOCKED_DEPENDENCY missing_stack={particle_stack_subtracted_path} "
                f"micrograph={micrograph_name}"
            )
            print(message)
            _append_event_log(
                "WARN",
                "BLOCKED_DEPENDENCY",
                raw_stack=rawimage_stacks_name,
                particle_stack_subtracted=particle_stack_subtracted_path,
                micrograph=micrograph_name,
                reason="MISSING_PARTICLE_STACK",
            )
            if self.strict_dependencies:
                raise DependencyRuntimeError(message)
            return

        if not os.path.exists(micrograph_name):
            message = (
                f">>> BLOCKED_DEPENDENCY missing_micrograph={micrograph_name} "
                f"stack={particle_stack_subtracted_path}"
            )
            print(message)
            _append_event_log(
                "WARN",
                "BLOCKED_DEPENDENCY",
                raw_stack=rawimage_stacks_name,
                particle_stack_subtracted=particle_stack_subtracted_path,
                micrograph=micrograph_name,
                reason="MISSING_MICROGRAPH",
            )
            if self.strict_dependencies:
                raise DependencyRuntimeError(message)
            return

        particle_pms_params_hash = None
        if self.require_particle_mxt:
            if not os.path.exists(particle_stack_mxt_path):
                message = (
                    f">>> BLOCKED_DEPENDENCY missing_mxt={particle_stack_mxt_path} "
                    f"micrograph={micrograph_name}"
                )
                print(message)
                _append_event_log(
                    "WARN",
                    "BLOCKED_DEPENDENCY",
                    raw_stack=rawimage_stacks_name,
                    particle_stack_subtracted=particle_stack_subtracted_path,
                    micrograph=micrograph_name,
                    reason="MISSING_PARTICLE_MXT",
                    particle_stack_mxt=particle_stack_mxt_path,
                )
                if self.strict_dependencies:
                    raise DependencyRuntimeError(message)
                return
            try:
                dep = read_mxt(particle_stack_mxt_path)
            except Exception:
                message = (
                    f">>> BLOCKED_DEPENDENCY invalid_mxt={particle_stack_mxt_path} "
                    f"micrograph={micrograph_name}"
                )
                print(message)
                _append_event_log(
                    "WARN",
                    "BLOCKED_DEPENDENCY",
                    raw_stack=rawimage_stacks_name,
                    particle_stack_subtracted=particle_stack_subtracted_path,
                    micrograph=micrograph_name,
                    reason="INVALID_PARTICLE_MXT",
                    particle_stack_mxt=particle_stack_mxt_path,
                )
                if self.strict_dependencies:
                    raise DependencyRuntimeError(message)
                return

            if dep.get("task") != "bezierfit_particle_pms" or dep.get("status") != "success":
                message = (
                    f">>> BLOCKED_DEPENDENCY particle_mxt_not_success={particle_stack_mxt_path} "
                    f"micrograph={micrograph_name}"
                )
                print(message)
                _append_event_log(
                    "WARN",
                    "BLOCKED_DEPENDENCY",
                    raw_stack=rawimage_stacks_name,
                    particle_stack_subtracted=particle_stack_subtracted_path,
                    micrograph=micrograph_name,
                    reason="PARTICLE_MXT_NOT_SUCCESS",
                    particle_stack_mxt=particle_stack_mxt_path,
                    particle_mxt_status=dep.get("status"),
                    particle_mxt_task=dep.get("task"),
                )
                if self.strict_dependencies:
                    raise DependencyRuntimeError(message)
                return

            particle_pms_params_hash = dep.get("params_hash")
            if not isinstance(particle_pms_params_hash, str) or particle_pms_params_hash == "":
                message = (
                    f">>> BLOCKED_DEPENDENCY particle_mxt_missing_hash={particle_stack_mxt_path} "
                    f"micrograph={micrograph_name}"
                )
                print(message)
                _append_event_log(
                    "WARN",
                    "BLOCKED_DEPENDENCY",
                    raw_stack=rawimage_stacks_name,
                    particle_stack_subtracted=particle_stack_subtracted_path,
                    micrograph=micrograph_name,
                    reason="PARTICLE_MXT_MISSING_HASH",
                    particle_stack_mxt=particle_stack_mxt_path,
                )
                if self.strict_dependencies:
                    raise DependencyRuntimeError(message)
                return
        else:
            if os.path.exists(particle_stack_mxt_path):
                try:
                    dep = read_mxt(particle_stack_mxt_path)
                    if dep.get("status") == "success":
                        particle_pms_params_hash = dep.get("params_hash")
                except Exception:
                    particle_pms_params_hash = None

        out_micrograph = to_output_micrograph_path(
            micrograph_name,
            output_root=self.output_root,
            output_dirname=self.output_dirname,
        )
        mxt_path = out_micrograph + ".mxt"
        lock_path = mxt_path + ".lock"

        expected_inputs = {
            "micrograph_raw": fingerprint_file(micrograph_name),
            "particles_star": self._particles_star_fp,
            "particle_stack_subtracted": fingerprint_file(particle_stack_subtracted_path),
            "deps": {
                "particle_pms_params_hash": particle_pms_params_hash,
                "particle_stack_mxt_path": particle_stack_mxt_path,
            },
        }

        if (not self.force) and self.resume:
            uptodate, reason = is_uptodate(
                out_micrograph,
                mxt_path,
                self.mxt_task,
                self.mxt_params_hash,
                expected_inputs,
                strict_output_check=self.strict_output_check,
            )
            if uptodate:
                print(f">>> SKIP_UPTODATE {micrograph_name} -> {out_micrograph}")
                _append_event_log(
                    "INFO",
                    "SKIP_UPTODATE",
                    micrograph=micrograph_name,
                    out_micrograph=out_micrograph,
                    particle_stack_subtracted=particle_stack_subtracted_path,
                )
                return

            if self.skip_failed and os.path.exists(mxt_path):
                try:
                    mxt = read_mxt(mxt_path)
                    if (
                        mxt.get("task") == self.mxt_task
                        and mxt.get("status") == "failed"
                        and mxt.get("params_hash") == self.mxt_params_hash
                    ):
                        print(f">>> SKIP_FAILED {micrograph_name} -> {out_micrograph}")
                        _append_event_log(
                            "WARN",
                            "SKIP_FAILED",
                            micrograph=micrograph_name,
                            out_micrograph=out_micrograph,
                        )
                        return
                except Exception:
                    pass

            if (
                self.adopt_existing_outputs
                and reason in {"MISSING_MXT", "INVALID_MXT_JSON"}
                and os.path.exists(out_micrograph)
                and self._output_passes_sanity_check(out_micrograph)
            ):
                now = datetime.now(timezone.utc).isoformat()
                adopt_run_id = f"{now}-{os.getpid()}-adopt"
                obj = self._build_mxt_object(
                    status="success",
                    out_micrograph=out_micrograph,
                    expected_inputs=expected_inputs,
                    started_utc=now,
                    finished_utc=now,
                    duration_sec=0.0,
                    adopted=True,
                    run_id=adopt_run_id,
                    error=None,
                )
                write_json_atomic(mxt_path, obj)
                print(f">>> ADOPT_EXISTING_OUTPUT {micrograph_name} -> {out_micrograph}")
                _append_event_log(
                    "INFO",
                    "ADOPT_EXISTING_OUTPUT",
                    micrograph=micrograph_name,
                    out_micrograph=out_micrograph,
                    run_id=adopt_run_id,
                )
                return

        run_id = f"{datetime.now(timezone.utc).isoformat()}-{os.getpid()}"
        if not try_acquire_lock(lock_path, run_id=run_id):
            print(f">>> LOCKED_SKIP {micrograph_name} -> {out_micrograph}")
            _append_event_log(
                "INFO",
                "LOCKED_SKIP",
                micrograph=micrograph_name,
                out_micrograph=out_micrograph,
            )
            return

        started_wall = time.time()
        started_utc = datetime.now(timezone.utc).isoformat()
        _append_event_log(
            "INFO",
            "START",
            micrograph=micrograph_name,
            out_micrograph=out_micrograph,
            run_id=run_id,
            params_hash=self.mxt_params_hash,
            particle_stack_subtracted=particle_stack_subtracted_path,
        )

        try:
            import cupy as cp

            if (not self.force) and self.resume:
                uptodate, _reason2 = is_uptodate(
                    out_micrograph,
                    mxt_path,
                    self.mxt_task,
                    self.mxt_params_hash,
                    expected_inputs,
                    strict_output_check=self.strict_output_check,
                )
                if uptodate:
                    print(f">>> SKIP_UPTODATE {micrograph_name} -> {out_micrograph}")
                    _append_event_log(
                        "INFO",
                        "SKIP_UPTODATE",
                        micrograph=micrograph_name,
                        out_micrograph=out_micrograph,
                        run_id=run_id,
                    )
                    return

            with mrcfile.open(particle_stack_subtracted_path, permissive=True) as f:
                subtracted_images_stacks = cp.asarray(f.data)
            if subtracted_images_stacks.ndim == 2:
                subtracted_images_stacks = cp.expand_dims(subtracted_images_stacks, axis=0)

            self._ensure_weights_for_box(int(subtracted_images_stacks.shape[1]))

            for i in range(subtracted_images_stacks.shape[0]):
                subtracted_image = subtracted_images_stacks[i]
                if cp.isnan(subtracted_image).any():
                    subtracted_images_stacks[i] = self.fill_nan_with_gaussian_noise(subtracted_image)

            with mrcfile.open(micrograph_name, permissive=True) as f:
                micrograph = cp.asarray(f.data)

            micrograph_subtracted = micrograph.copy()
            mem_mosaic_image = cp.zeros_like(micrograph)
            weight_sum_image = cp.zeros_like(micrograph)
            mem_mosaic_image_mask = cp.zeros_like(micrograph)

            df_rawimage_temp = self.get_df_temp(rawimage_stacks_name)
            for df_rawimage_temp_section_num, df_rawimage_temp_X_Y in self.get_df_temp_X_Y(df_rawimage_temp).items():
                # STAR provides 1-based section indices.
                section_1based = int(df_rawimage_temp_section_num)
                idx0 = section_1based - 1
                if idx0 < 0 or idx0 >= int(subtracted_images_stacks.shape[0]):
                    raise ValueError(
                        f"STAR section index out of bounds: section_1based={section_1based} "
                        f"stack_len={int(subtracted_images_stacks.shape[0])} stack={rawimage_stacks_name}"
                    )

                subtracted_image_temp_full = subtracted_images_stacks[idx0]
                x_center, y_center = df_rawimage_temp_X_Y

                # Coordinates should be integer pixel indices; enforce int conversion to
                # avoid implicit float slicing errors.
                x_center = int(x_center)
                y_center = int(y_center)

                half = int(self.rawimage_size) // 2
                row0 = y_center - half
                col0 = x_center - half
                row1 = row0 + int(self.rawimage_size)
                col1 = col0 + int(self.rawimage_size)

                # Clamp box to micrograph bounds to avoid negative-index wraparound.
                mg_h, mg_w = int(micrograph.shape[0]), int(micrograph.shape[1])
                row0_c = max(row0, 0)
                col0_c = max(col0, 0)
                row1_c = min(row1, mg_h)
                col1_c = min(col1, mg_w)

                if row0_c >= row1_c or col0_c >= col1_c:
                    raise ValueError(
                        f"Zero-area micrograph box after clamping: center=({x_center},{y_center}) "
                        f"box={int(self.rawimage_size)} micrograph_shape={micrograph.shape} "
                        f"clamped_rows=({row0_c},{row1_c}) clamped_cols=({col0_c},{col1_c})"
                    )

                # Corresponding slice in the particle/subtracted-image box.
                pr0 = row0_c - row0
                pc0 = col0_c - col0
                pr1 = pr0 + (row1_c - row0_c)
                pc1 = pc0 + (col1_c - col0_c)

                particle_be_replaced = micrograph[row0_c:row1_c, col0_c:col1_c]
                sub_patch = -subtracted_image_temp_full[pr0:pr1, pc0:pc1]

                # Match per-particle contrast/scale to the micrograph crop (baseline behavior).
                sub_patch = (
                    (sub_patch - cp.mean(sub_patch))
                    / cp.std(sub_patch)
                    * cp.std(particle_be_replaced)
                    + cp.mean(particle_be_replaced)
                )

                weights_patch = self.weights[pr0:pr1, pc0:pc1]
                mask_patch = self.mask[pr0:pr1, pc0:pc1]

                mem_mosaic_image[row0_c:row1_c, col0_c:col1_c] += sub_patch * weights_patch
                weight_sum_image[row0_c:row1_c, col0_c:col1_c] += weights_patch
                mem_mosaic_image_mask[row0_c:row1_c, col0_c:col1_c] = mask_patch

            epsilon = float(self.mxt_params["epsilon"])
            weight_sum_image = cp.where(weight_sum_image == 0, epsilon, weight_sum_image)
            mem_mosaic_image /= weight_sum_image
            micrograph_subtracted = micrograph_subtracted * (1 - mem_mosaic_image_mask) + mem_mosaic_image * mem_mosaic_image_mask

            write_mrc_atomic(out_micrograph, micrograph_subtracted)
            print(f">>> {out_micrograph} finished")

            finished_utc = datetime.now(timezone.utc).isoformat()
            duration = time.time() - started_wall
            obj = self._build_mxt_object(
                status="success",
                out_micrograph=out_micrograph,
                expected_inputs=expected_inputs,
                started_utc=started_utc,
                finished_utc=finished_utc,
                duration_sec=duration,
                adopted=False,
                run_id=run_id,
                error=None,
            )
            write_json_atomic(mxt_path, obj)
            _append_event_log(
                "INFO",
                "SUCCESS",
                micrograph=micrograph_name,
                out_micrograph=out_micrograph,
                run_id=run_id,
                duration_sec=duration,
            )

            del micrograph_subtracted
            del subtracted_images_stacks
            del mem_mosaic_image
            del weight_sum_image
            del mem_mosaic_image_mask
            del df_rawimage_temp
            cp.cuda.Stream.null.synchronize()
            cp.get_default_memory_pool().free_all_blocks()
        except Exception as exc:
            finished_utc = datetime.now(timezone.utc).isoformat()
            duration = time.time() - started_wall
            tb = traceback.format_exc()
            if len(tb) > 20_000:
                tb = tb[:20_000] + "\n... (truncated) ..."
            error = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": tb,
            }
            try:
                obj = self._build_mxt_object(
                    status="failed",
                    out_micrograph=out_micrograph,
                    expected_inputs=expected_inputs,
                    started_utc=started_utc,
                    finished_utc=finished_utc,
                    duration_sec=duration,
                    adopted=False,
                    run_id=run_id,
                    error=error,
                )
                write_json_atomic(mxt_path, obj)
            except Exception:
                pass
            _append_event_log(
                "ERROR",
                "FAILED",
                micrograph=micrograph_name,
                out_micrograph=out_micrograph,
                run_id=run_id,
                error_type=type(exc).__name__,
            )
            raise
        finally:
            release_lock(lock_path)
    
    def micrograph_mem_subtract_multiprocessing(self, num_cpus, batch_size):
        try:
            multiprocessing.set_start_method("spawn")
        except RuntimeError:
            # Start method can only be set once per interpreter.
            pass

        def chunks(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i : i + n]

        print(">>> Preparing Micrograph Membrane Subtraction dataset...")
        print(f">>> input_base_dir: {self.input_base_dir}")
        print(f">>> Found {len(self.raw_mg_name_lst)} raw micrographs in total.")

        worker_config = {
            "particles_selected_filename": self.particles_selected_filename,
            "input_base_dir": self.input_base_dir,
            "resume": bool(self.resume),
            "force": bool(self.force),
            "adopt_existing_outputs": bool(self.adopt_existing_outputs),
            "skip_failed": bool(self.skip_failed),
            "require_particle_mxt": bool(self.require_particle_mxt),
            "strict_dependencies": bool(self.strict_dependencies),
            "strict_output_check": bool(self.strict_output_check),
            "output_root": self.output_root,
            "particle_output_root": self.particle_output_root,
            "output_dirname": self.output_dirname,
            "write_output_star": bool(self.write_output_star),
            "output_star_path": self.output_star_path,
        }

        minibatches = list(chunks(self.raw_mg_name_lst, int(batch_size)))
        total_len = len(minibatches)
        with Pool(processes=int(num_cpus), initializer=_init_worker, initargs=(worker_config,)) as p:
            for i, minibatch in enumerate(minibatches, start=1):
                start_time = time.time()
                p.map(_process_micrograph_worker, minibatch, chunksize=1)
                end_time = time.time()
                print(f">>> {i} / {total_len} minibatch finished.")
                print(f">>> {len(minibatch)} micrograph stacks took {end_time - start_time:.4f} seconds.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Micrograph membrane subtraction')
    parser.add_argument(
        '--particles_selected_filename',
        '--particle',
        '-ps',
        type=str,
        help='particles_selected.star',
    )
    parser.add_argument(
        "--procs",
        type=int,
        default=0,
        help=(
            "Number of worker processes to run micrograph stacks in parallel. "
            "Use 0 to auto-detect from visible GPUs. (default: 0)"
        ),
    )
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument(
        "--input_base_dir",
        type=str,
        default=None,
        help=(
            "Base directory used to resolve relative paths stored inside STAR files "
            "(e.g. 'J220/extract/...'). If omitted, auto-infer from the STAR path "
            "(CryoSPARC layout aware)."
        ),
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help=(
            "Where this MMS job writes outputs. "
            "Outputs are written under <output_root>/<output_dirname>/... when set."
        ),
    )
    parser.add_argument(
        "--particle_output_root",
        type=str,
        default=None,
        help=(
            "Where MMS reads upstream PMS particle-stack outputs from. "
            "If omitted, falls back to --output_root (legacy behavior)."
        ),
    )
    parser.add_argument(
        "--output_dirname",
        type=str,
        default="subtracted",
        help="Output folder name (default: subtracted). Must be a single path segment.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable resume via per-output .mxt sidecars (default: enabled).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force recompute even if .mxt indicates the output is up-to-date.",
    )
    parser.add_argument(
        "--adopt_existing_outputs",
        action="store_true",
        default=False,
        help="If output exists but .mxt is missing/invalid, write an adopted .mxt and skip recompute.",
    )
    parser.add_argument(
        "--skip_failed",
        action="store_true",
        default=False,
        help="Skip items whose existing .mxt has status=failed for the current params_hash.",
    )
    parser.add_argument(
        "--require_particle_mxt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require dependency particle-stack .mxt sidecars to exist and be success (default: enabled).",
    )
    parser.add_argument(
        "--strict_dependencies",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Fail-fast preflight for dependencies before multiprocessing "
            "(default: enabled). Disable to keep legacy best-effort behavior."
        ),
    )
    parser.add_argument(
        "--write_output_star",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write output STAR with rewritten absolute rlnMicrographName paths (default: enabled).",
    )
    parser.add_argument(
        "--output_star_path",
        type=str,
        default=None,
        help="Optional explicit path for output STAR; default is deterministic based on input STAR and output_dirname.",
    )
    args = parser.parse_args()

    if args.output_root:
        try:
            _EVENT_LOG_PATH = os.fspath(args.output_root) + os.sep + "mms_run_data.log"
        except Exception:
            _EVENT_LOG_PATH = None
    else:
        _EVENT_LOG_PATH = "mms_run_data.log"

    try:
        mms = MicrographMembraneSubtract(
            args.particles_selected_filename,
            input_base_dir=args.input_base_dir,
            resume=args.resume,
            force=args.force,
            adopt_existing_outputs=args.adopt_existing_outputs,
            skip_failed=args.skip_failed,
            require_particle_mxt=args.require_particle_mxt,
            strict_dependencies=args.strict_dependencies,
            output_root=args.output_root,
            particle_output_root=args.particle_output_root,
            output_dirname=args.output_dirname,
            write_output_star=args.write_output_star,
            output_star_path=args.output_star_path,
        )

        procs = int(args.procs)
        if procs <= 0:
            try:
                import cupy as _cp

                procs = int(_cp.cuda.runtime.getDeviceCount())
            except Exception:
                procs = 1

        if bool(args.strict_dependencies):
            mms.preflight_check_dependencies()

        mms.micrograph_mem_subtract_multiprocessing(procs, args.batch_size)

        if bool(args.write_output_star):
            out_star = mms.write_output_star_file(output_star_path=args.output_star_path)
            print(f">>> OUTPUT_STAR {out_star}")
    except (DependencyPreflightError, DependencyRuntimeError) as exc:
        print(str(exc), flush=True)
        raise SystemExit(3)
    except StarRewriteError as exc:
        print(str(exc), flush=True)
        raise SystemExit(4)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
