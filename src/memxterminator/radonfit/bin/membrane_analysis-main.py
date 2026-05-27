from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Optional

import cupy as cp
import numpy as np
import pandas as pd
import starfile
from cupyx.scipy.ndimage import zoom

from ..lib._utils import (
    find_star_column,
    parse_relion_image_name,
    read_star_any,
    readmrc,
    savemrc,
)
from ..lib.calculate_curve import Curvefitting
from ..lib.generate_membrane_mask import mem_mask
from ..lib.mem_average import average_membrane
from ..lib.radonanalyser import RadonAnalyzer
from ..lib.template_centerfitting import Template_centerfitting


OUTPUT_COLUMNS = [
    "rln2DAverageimageName",
    "rlnAveragedMembraneName",
    "rlnMembraneMaskName",
    "rlnCenterX",
    "rlnCenterY",
    "rlnAngleTheta",
    "rlnMembraneDistance",
    "rlnSigma1",
    "rlnSigma2",
    "rlnCurveKappa",
]

_WORKER_CONFIG: Optional[dict[str, Any]] = None
_WORKER_OUTPUT_DF: Optional[pd.DataFrame] = None
_WORKER_RADON_INFO: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class TemplateTask:
    index: int
    average2d_filename: str
    section0: int


@dataclass
class TemplateResult:
    index: int
    row: dict[str, Any]
    membrane_mask: np.ndarray
    averaged_membrane: np.ndarray
    average2d_filename: str
    section0: int
    elapsed_sec: float


def _derive_masks_filename(average2d_filename: str) -> str:
    return average2d_filename.replace(".mrc", "_masks.mrc")


def _derive_averaged_filename(average2d_filename: str) -> str:
    return average2d_filename.replace(".mrc", "_averaged.mrc")


def _normalise_scalar(value: Any) -> Any:
    try:
        return value.item()
    except Exception:
        return value


def get_zoom_factor(particle_starfile_name: str, templates_starfile_name: str) -> float:
    def get_image_size(starfile_name: str) -> int:
        star_tables = read_star_any(starfile_name)
        _, df_star, image_col = find_star_column(
            star_tables, candidates=["rlnImageName", "rlnReferenceImage"]
        )
        image_ref = df_star[image_col].iloc[0]
        mrc_path, section0 = parse_relion_image_name(image_ref)
        image_sample_array = readmrc(mrc_path, section=section0, mode="gpu")
        return int(image_sample_array.shape[0])

    particle_image_size = get_image_size(particle_starfile_name)
    templates_image_size = get_image_size(templates_starfile_name)
    return particle_image_size / templates_image_size


def get_parameters_for_section(radon_info: dict[str, Any], i: int):
    section_key = str(i)
    if section_key in radon_info:
        section_info = radon_info[section_key]
        return (
            section_info["crop_rate"],
            section_info["thr"],
            section_info["theta_start"],
            section_info["theta_end"],
        )

    print(f"No data found for section {i}", flush=True)
    return None, None, None, None


def _build_initial_output_df(df_templates_star: pd.DataFrame, image_col: str) -> pd.DataFrame:
    output_df_star = pd.DataFrame(data=[[0] * len(OUTPUT_COLUMNS)], columns=OUTPUT_COLUMNS)
    output_df_star = output_df_star.reindex(range(len(df_templates_star)))
    output_df_star.fillna(0, inplace=True)
    output_df_star["rln2DAverageimageName"] = df_templates_star[image_col].astype(str).tolist()
    return output_df_star


def _resolve_process_count(requested: int, template_count: int) -> int:
    if template_count <= 0:
        return 1
    if requested <= 0:
        requested = os.cpu_count() or 1
    return max(1, min(int(requested), int(template_count)))


def _pin_worker_cuda_device() -> None:
    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
    except Exception:
        device_count = 0
    if device_count <= 0:
        return

    try:
        ident = getattr(multiprocessing.current_process(), "_identity", ())
        worker_rank = int(ident[0]) if ident else 1
        device_id = (worker_rank - 1) % device_count
        cp.cuda.Device(device_id).use()
        print(
            f">>> WORKER_INIT pid={os.getpid()} worker_rank={worker_rank} "
            f"cuda_device={device_id} visible_cuda_devices={device_count}",
            flush=True,
        )
    except Exception as exc:
        print(f">>> WARNING: failed to pin worker pid={os.getpid()} to a CUDA device: {exc}", flush=True)


def _init_worker(config: dict[str, Any], output_df: pd.DataFrame, radon_info: dict[str, Any]) -> None:
    global _WORKER_CONFIG
    global _WORKER_OUTPUT_DF
    global _WORKER_RADON_INFO

    _WORKER_CONFIG = dict(config)
    _WORKER_OUTPUT_DF = output_df.copy()
    _WORKER_RADON_INFO = dict(radon_info)
    _pin_worker_cuda_device()


def _run_template_analysis(task: TemplateTask, output_filename: str) -> TemplateResult:
    if _WORKER_CONFIG is None or _WORKER_RADON_INFO is None:
        raise RuntimeError("Worker was not initialized")

    started = time.time()
    cfg = _WORKER_CONFIG
    i = int(task.index)
    average2d_filename = task.average2d_filename
    section0 = int(task.section0)

    image = readmrc(average2d_filename, section=section0, mode="gpu")
    image = zoom(image, cfg["zoom_factor"])
    crop_rate_temp, thr_temp, theta_start_temp, theta_end_temp = get_parameters_for_section(_WORKER_RADON_INFO, i)

    # Keep this first Radon pass: Template_centerfitting reads the STAR before
    # its own internal RadonAnalyzer refreshes it.
    RadonAnalyzer(
        output_filename,
        i,
        image,
        crop_rate=crop_rate_temp,
        thr=thr_temp,
        theta_start=theta_start_temp,
        theta_end=theta_end_temp,
    )
    centerfit = Template_centerfitting(
        output_filename,
        i,
        sigma1=cfg["initial_sigma1"],
        sigma2=cfg["initial_sigma2"],
        image=image,
        crop_rate=crop_rate_temp,
        thr=thr_temp,
        theta_start=theta_start_temp,
        theta_end=theta_end_temp,
        template_size=cfg["template_size"],
        sigma_range=cfg["sigma_range"],
        sigma_step=cfg["sigma_step"],
    )
    centerfit.centerfinder()
    centerfit.fit_sigma()
    Curvefitting(
        output_filename,
        i,
        image,
        kappa_start=cfg["curve_kappa_start"],
        kappa_end=cfg["curve_kappa_end"],
        kappa_step=cfg["curve_kappa_step"],
    )

    get_mem_mask = mem_mask(output_filename, i, image, edge_sigma=cfg["edge_sigma"])
    membrane_mask = np.asarray(get_mem_mask.generate_mem_mask())

    output_df_star = starfile.read(output_filename)
    masks_filename = cfg["masks_filename"]
    averaged_filename = cfg["averaged_filename"]
    output_df_star.loc[i, "rlnMembraneMaskName"] = f"{i + 1:06d}@{masks_filename}"

    mem_average = average_membrane(
        output_filename,
        i,
        image,
        extra_mem_dist=cfg["extra_mem_dist"],
        sigma=cfg["mem_edge_sigma"],
        x0=output_df_star.loc[i, "rlnCenterY"],
        y0=output_df_star.loc[i, "rlnCenterX"],
        theta=output_df_star.loc[i, "rlnAngleTheta"] * np.pi / 180,
        membrane_distance=output_df_star.loc[i, "rlnMembraneDistance"],
        kappa=output_df_star.loc[i, "rlnCurveKappa"],
    )
    mem_average.calculate_membrane_average()
    averaged_2d_membrane = cp.asnumpy(mem_average.generate_2d_average_mem())

    if cfg["select_kappa_template"] == i:
        mem_average.kappa_templates_generator(
            kappa_start=cfg["kappa_start"],
            kappa_end=cfg["kappa_end"],
            kappa_num=cfg["kappa_num"],
        )

    output_df_star.loc[i, "rlnAveragedMembraneName"] = f"{i + 1:06d}@{averaged_filename}"
    row = {
        col: _normalise_scalar(output_df_star.loc[i, col])
        for col in output_df_star.columns
    }
    elapsed = time.time() - started
    print(f"file {average2d_filename} section {section0 + 1} finished in {elapsed:.2f}s", flush=True)

    return TemplateResult(
        index=i,
        row=row,
        membrane_mask=membrane_mask,
        averaged_membrane=averaged_2d_membrane,
        average2d_filename=average2d_filename,
        section0=section0,
        elapsed_sec=elapsed,
    )


def _process_template_worker(task: TemplateTask) -> TemplateResult:
    if _WORKER_CONFIG is None or _WORKER_OUTPUT_DF is None:
        raise RuntimeError("Worker was not initialized")

    worker_star = os.path.join(
        _WORKER_CONFIG["temp_dir"],
        f"mem_analysis_template_{task.index:06d}_{os.getpid()}.star",
    )
    starfile.write(_WORKER_OUTPUT_DF.copy(), worker_star, overwrite=True)
    try:
        return _run_template_analysis(task, worker_star)
    finally:
        try:
            os.remove(worker_star)
        except OSError:
            pass
        try:
            cp.cuda.Stream.null.synchronize()
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass


def _run_tasks(
    *,
    tasks: list[TemplateTask],
    procs: int,
    config: dict[str, Any],
    output_df: pd.DataFrame,
    radon_info: dict[str, Any],
) -> list[TemplateResult]:
    if procs <= 1:
        _init_worker(config, output_df, radon_info)
        results = []
        for task in tasks:
            results.append(_process_template_worker(task))
        return results

    ctx = multiprocessing.get_context("spawn")
    results: list[TemplateResult] = []
    completed = 0
    total = len(tasks)
    with ctx.Pool(processes=procs, initializer=_init_worker, initargs=(config, output_df, radon_info)) as pool:
        for result in pool.imap_unordered(_process_template_worker, tasks, chunksize=1):
            results.append(result)
            completed += 1
            print(
                f">>> {completed} / {total} templates finished "
                f"(template_index={result.index}, elapsed={result.elapsed_sec:.2f}s).",
                flush=True,
            )
    return results


def run_membrane_analysis(args: argparse.Namespace) -> None:
    templates_starfile_name = args.templates_starfile_name
    output_filename = args.output_filename
    particle_starfile_name = args.particle_starfile_name

    star_tables_templates = read_star_any(templates_starfile_name)
    _, df_templates_star, templates_image_col = find_star_column(
        star_tables_templates, candidates=["rlnImageName", "rlnReferenceImage"]
    )
    output_df_star = _build_initial_output_df(df_templates_star, templates_image_col)
    average_2d_lst = [
        parse_relion_image_name(x)
        for x in df_templates_star[templates_image_col].astype(str).tolist()
    ]
    if not average_2d_lst:
        raise ValueError(f"No templates found in STAR file: {templates_starfile_name}")

    masks_filename = _derive_masks_filename(average_2d_lst[0][0])
    averaged_filename = _derive_averaged_filename(average_2d_lst[0][0])
    starfile.write(output_df_star, output_filename, overwrite=True)

    zoom_factor = get_zoom_factor(particle_starfile_name, templates_starfile_name)
    print("zoom_factor", zoom_factor, flush=True)

    with open(args.info_json, "r") as file:
        radon_info = json.load(file)

    select_kappa_template = int(args.kappa_template)
    if select_kappa_template < 0:
        select_kappa_template = -1

    tasks = [
        TemplateTask(index=i, average2d_filename=average2d_filename, section0=section0)
        for i, (average2d_filename, section0) in enumerate(average_2d_lst)
    ]
    requested_procs = int(args.procs)
    procs = _resolve_process_count(requested_procs, len(tasks))

    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
    except Exception:
        device_count = 0
    cpu_count = os.cpu_count() or 1
    print(
        f">>> Found {len(tasks)} template(s). CPU count: {cpu_count}; "
        f"CUDA devices visible: {device_count}; using {procs} template worker process(es).",
        flush=True,
    )
    if requested_procs > len(tasks):
        print(
            f">>> Requested --procs={requested_procs}, capped to template count ({len(tasks)}).",
            flush=True,
        )
    if device_count > 0 and procs > device_count:
        print(
            f">>> WARNING: using {procs} worker processes with only {device_count} visible CUDA device(s). "
            "This follows template-level CPU parallelism, but may oversubscribe GPU memory.",
            flush=True,
        )

    config = {
        "zoom_factor": zoom_factor,
        "initial_sigma1": float(args.sigma1),
        "initial_sigma2": float(args.sigma2),
        "template_size": int(args.template_size),
        "sigma_range": int(args.sigma_range),
        "sigma_step": float(args.sigma_step),
        "curve_kappa_start": float(args.curve_kappa_start),
        "curve_kappa_end": float(args.curve_kappa_end),
        "curve_kappa_step": float(args.curve_kappa_step),
        "edge_sigma": float(args.edge_sigma),
        "extra_mem_dist": int(args.extra_mem_dist),
        "mem_edge_sigma": float(args.mem_edge_sigma),
        "select_kappa_template": select_kappa_template,
        "kappa_num": int(args.kappanum),
        "kappa_start": float(args.kappastart),
        "kappa_end": float(args.kappaend),
        "masks_filename": masks_filename,
        "averaged_filename": averaged_filename,
    }

    with tempfile.TemporaryDirectory(prefix="mxt_membrane_analysis_") as temp_dir:
        config["temp_dir"] = temp_dir
        results = _run_tasks(
            tasks=tasks,
            procs=procs,
            config=config,
            output_df=output_df_star,
            radon_info=radon_info,
        )

    results.sort(key=lambda result: result.index)
    for result in results:
        for col, value in result.row.items():
            output_df_star.loc[result.index, col] = value

    membrane_masks = np.asarray([result.membrane_mask for result in results])
    averaged_membranes = np.asarray([result.averaged_membrane for result in results])
    savemrc(membrane_masks, masks_filename)
    savemrc(averaged_membranes, averaged_filename)
    starfile.write(output_df_star, output_filename, overwrite=True)

    total_elapsed = sum(result.elapsed_sec for result in results)
    print(f">>> Template worker compute time total: {total_elapsed:.2f}s", flush=True)
    print(">>> Membrane analysis COMPLETED!", flush=True)
    print(f">>> Output STAR: {output_filename}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--templates_starfile_name", "-ts", type=str, default="templates_selected.star", help="star file name of templates")
    parser.add_argument("--output_filename", "-o", type=str, default="mem_analysis.star", help="output star file name")
    parser.add_argument("--particle_starfile_name", "-ps", type=str, default="particles_selected.star", help="star file name of particles")
    parser.add_argument("--kappa_template", "-k", type=int, default=-1, help="0-based template index for kappa template generation; -1 disables it")
    parser.add_argument("--kappanum", "-kn", type=int, default=40, help="number of kappa templates")
    parser.add_argument("--kappastart", "-ks", type=float, default=-0.008, help="start value of kappa templates")
    parser.add_argument("--kappaend", "-ke", type=float, default=0.008, help="end value of kappa templates")
    parser.add_argument("--info_json", "-j", type=str, default="radonanalysis_info.json", help="json file name of radon analysis info")
    parser.add_argument("--sigma1", "-s1", type=float, default=6, help="sigma1 of template center fitting")
    parser.add_argument("--sigma2", "-s2", type=float, default=6, help="sigma2 of template center fitting")
    parser.add_argument("--template_size", type=int, default=64, help="template size of template center fitting")
    parser.add_argument("--sigma_range", type=int, default=3, help="sigma range of template center fitting")
    parser.add_argument("--sigma_step", type=float, default=0.5, help="sigma step of template center fitting")
    parser.add_argument("--curve_kappa_start", type=float, default=-0.01, help="start value of kappa in curve fitting")
    parser.add_argument("--curve_kappa_end", type=float, default=0.01, help="end value of kappa in curve fitting")
    parser.add_argument("--curve_kappa_step", type=float, default=0.0002, help="step of kappa in curve fitting")
    parser.add_argument("--edge_sigma", type=float, default=3, help="sigma of edge detection in membrane mask generation")
    parser.add_argument("--extra_mem_dist", type=int, default=15, help="extra membrane distance in membrane average")
    parser.add_argument("--mem_edge_sigma", type=float, default=5, help="sigma of membrane average")
    parser.add_argument(
        "--procs",
        "--cpu",
        dest="procs",
        type=int,
        default=1,
        help=(
            "Number of template worker processes. Use 96 to process up to 96 templates "
            "concurrently; 0 auto-detects CPU count and caps at the template count."
        ),
    )
    return parser


def main() -> None:
    parser = build_parser()
    run_membrane_analysis(parser.parse_args())


if __name__ == "__main__":
    main()
