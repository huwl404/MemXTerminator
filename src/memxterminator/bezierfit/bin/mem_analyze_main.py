from cryosparc.dataset import Dataset
import numpy as np
import cupy as cp
from cupyx.scipy.ndimage import zoom
import mrcfile
from ..lib.bezierfit import Coarsefit, GA_Refine
import json
import argparse
import multiprocessing
import os

from memxterminator.path_resolve import infer_input_base_dir, normalise_dir, resolve_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', type=str, default='./templates_selected.cs')
    parser.add_argument('--particle', type=str, default='./particles_selected.cs')
    parser.add_argument('--output', type=str, default='./control_points.json')
    parser.add_argument(
        '--input_base_dir',
        type=str,
        default=None,
        help=(
            "Base directory used to resolve relative paths stored inside CryoSPARC .cs files "
            "(e.g. 'J220/extract/...'). If omitted, auto-infer from the template .cs path "
            "(CryoSPARC layout aware)."
        ),
    )
    parser.add_argument('--num_points', type=int, default=600)
    parser.add_argument('--degree', type=int, default=3)
    parser.add_argument('--coarsefit_iter', type=int, default=300)
    parser.add_argument('--coarsefit_cpus', type=int, default=20)
    parser.add_argument('--cur_penalty_thr', type=float, default=0.05)
    parser.add_argument('--dithering_range', type=int, default=50)
    parser.add_argument('--refine_iter', type=int, default=700)
    parser.add_argument('--refine_cpus', type=int, default=12)
    parser.add_argument('--physical_membrane_dist', type=int, default=35)
    args = parser.parse_args()

    if args.input_base_dir is not None and str(args.input_base_dir).strip() != "":
        input_base_dir = normalise_dir(args.input_base_dir)
    else:
        input_base_dir = infer_input_base_dir(args.template)
    print(f">>> input_base_dir: {input_base_dir}")

    template_dset = Dataset.load(args.template)
    particle_dset = Dataset.load(args.particle)

    template_filenames_list = np.array(template_dset['blob/path']).astype(str)
    template_idx_list = np.array(template_dset['blob/idx'])
    particle_pixel_szs = np.array(particle_dset['blob/psize_A'])
    template_pixel_szs = np.array(template_dset['blob/psize_A'])
    template_shape = np.array(template_dset['blob/shape'])[0]
    particle_shape = np.array(particle_dset['blob/shape'])[0]
    zoom_factor = particle_shape[0] / template_shape[0]

    template_pixel_szs = particle_pixel_szs

    control_points_with_idx = {}

    multiprocessing.set_start_method('spawn', force=True)
    template_paths = [resolve_path(p, base_dir=input_base_dir) for p in template_filenames_list.tolist()]
    if template_paths and not os.path.exists(template_paths[0]):
        raise SystemExit(
            "ERROR: Cannot find template stack referenced in template .cs.\n"
            f"  template_cs: {args.template}\n"
            f"  inferred input_base_dir: {input_base_dir}\n"
            f"  example missing resolved path: {template_paths[0]}\n"
            "Fix: re-run with --input_base_dir /path/to/cryosparc_project_root\n"
        )

    for template_filename, template_idx, template_pixel_sz in zip(template_paths, template_idx_list, template_pixel_szs):
        with mrcfile.open(str(template_filename), permissive=True) as mrc:
            template_original = mrc.data[template_idx]
            template_original = cp.array(template_original)
            template_image = zoom(template_original, zoom_factor)
            template_image_cpu = template_image.get()
            # template_image_cpu = mrc.data[template_idx]
        coarsefit = Coarsefit(template_image_cpu, args.num_points, args.degree, args.coarsefit_iter, args.coarsefit_cpus)
        initial_control_points = coarsefit()
        ga_refine = GA_Refine(template_image_cpu, template_pixel_sz, args.cur_penalty_thr, args.dithering_range, args.refine_iter, args.refine_cpus, args.physical_membrane_dist)
        refined_control_points = ga_refine(initial_control_points, template_image_cpu)
        # save refined control points and template_idx into a dictionary and save it as a JSON file
        control_points_with_idx[str(template_idx)] = refined_control_points.tolist()
        print(f'Template {template_filename} {template_idx} done, control points: {refined_control_points}')
        with open(args.output, 'w') as f:
            json.dump(control_points_with_idx, f)
            f.write('\n')
        # Free cached blocks from the *default* CuPy pool. Creating a new MemoryPool()
        # and freeing it is a no-op for existing allocations.
        cp.get_default_memory_pool().free_all_blocks()
