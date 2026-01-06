import mrcfile
import starfile
import numpy as np
import cupy as cp
import pandas as pd
import matplotlib.pyplot as plt
from cupyx.scipy.ndimage import convolve
from collections.abc import Mapping

def readmrc(filename, section=0, mode='cpu'):
    image = mrcfile.open(filename)
    if len(image.data.shape) == 2:
        gray_image = image.data.copy()
    elif len(image.data.shape) == 3: 
        gray_image = image.data[section].copy()
    else:
        raise ValueError("Unsupported image dimensions")
    if mode == 'cpu':
        return gray_image
    elif mode == 'gpu':
        return cp.asarray(gray_image)

def savemrc(image, filename):
    image = image.astype(np.float32)
    with mrcfile.new(filename, overwrite=True) as mrc:
        mrc.set_data(image)


def distance(x1, y1, x2, y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def create_starfile(Average_raw2dimage_filename):
    filename = 'mem_analysis.star'
    images = mrcfile.open(Average_raw2dimage_filename)
    image_sections = images.data.shape[0]
    # df_star = pd.DataFrame(data=[[0] * 9], columns=['rlnRawImageName', 'rln2DAverageimageName', 'rlnCenterX', 'rlnCenterY', 'rlnAngleTheta', 'rlnMembraneDistance', 'rlnSigma1', 'rlnSigma2', 'rlnCurveKappa'])
    df_star = pd.DataFrame(data=[[0] * 8], columns=['rln2DAverageimageName', 'rlnCenterX', 'rlnCenterY', 'rlnAngleTheta', 'rlnMembraneDistance', 'rlnSigma1', 'rlnSigma2', 'rlnCurveKappa'])
    for i in range(image_sections):
        # df_star.loc[i] = [f'{rawimage_name}', f'{i+1:06d}@{Average_raw2dimage_filename}', 0, 0, 0, 0, 0, 0, 0]
        df_star.loc[i] = [f'{i+1:06d}@{Average_raw2dimage_filename}', 0, 0, 0, 0, 0, 0, 0]
    starfile.write(df_star, filename, overwrite=True)

def readstar(starfile_name):
    star = starfile.read(starfile_name)
    return star

def write_star(df, starfile_name='mem_analysis.star'):
    starfile.write(df, starfile_name, overwrite=True)

def create_gaussian_low_pass_filter(image, cutoff_frequency):
    shape = image.shape
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    y, x = cp.ogrid[-crow:rows-crow, -ccol:cols-ccol]
    distance = cp.sqrt(x*x + y*y)
    mask = cp.exp(-(distance**2) / (2*cutoff_frequency**2))

    f = cp.fft.fft2(image)
    fshift = cp.fft.fftshift(f)
    fshift_filtered = fshift * mask
    f_ishift = cp.fft.ifftshift(fshift_filtered)
    img_back = cp.fft.ifft2(f_ishift)
    img_back = cp.real(img_back)
    return img_back


def gaussian_kernel(size, sigma=1.0):
    size = int(size) // 2
    x, y = cp.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * cp.pi * sigma**2)
    g =  cp.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g


def read_star_any(path: str) -> dict:
    """
    Read a RELION/STAR file and always return a dict of tabular blocks.

    Notes
    -----
    - For STAR files with a single data block, `starfile.read()` often returns a
      `pandas.DataFrame`.
    - For RELION 3+ STAR files, there can be multiple data blocks (e.g.
      `data_optics`, `data_particles`), in which case `starfile.read()` may
      return a dict mapping block name -> `pandas.DataFrame`.

    This helper provides a compatibility layer that:
    - tries to use `always_dict=True` when supported by the installed `starfile`
    - otherwise normalises the return type to `dict[str, pandas.DataFrame]`
    """
    try:
        obj = starfile.read(path, always_dict=True)
    except TypeError:
        obj = starfile.read(path)

    if isinstance(obj, pd.DataFrame):
        return {"__single__": obj}

    if isinstance(obj, Mapping):
        # Some STAR readers may return non-tabular blocks as dicts; we keep only
        # blocks that can be represented as DataFrames.
        tables = {}
        for block_name, block in obj.items():
            if isinstance(block, pd.DataFrame):
                tables[block_name] = block
                continue
            if isinstance(block, dict):
                try:
                    tables[block_name] = pd.DataFrame([block])
                except Exception:
                    continue
        if not tables:
            raise TypeError(f"No tabular blocks found when reading STAR file: {path}")
        return tables

    raise TypeError(f"Unsupported return type from starfile.read({path!r}): {type(obj)}")


def find_star_column(star_tables: dict, candidates: list[str]):
    """
    Find the first STAR block that contains any candidate column.

    Returns
    -------
    (block_name, df, column_name)
    """
    for block_name, df in star_tables.items():
        for col in candidates:
            if col in df.columns:
                return block_name, df, col
    available = {block_name: list(df.columns) for block_name, df in star_tables.items()}
    raise KeyError(
        "Cannot find any of the candidate columns in the STAR file blocks. "
        f"candidates={candidates}, available_blocks={list(available.keys())}"
    )


def parse_relion_image_name(image_name: str) -> tuple[str, int]:
    """
    Parse a RELION image reference string.

    RELION commonly stores images as `N@path/to/stack.mrcs`, where `N` is a
    1-based index into the MRC stack. This function returns a tuple of
    `(mrc_path, section_0based)`.
    """
    if pd.isna(image_name):
        raise ValueError("RELION image reference is NaN/None")
    s = str(image_name).strip()
    left, sep, right = s.partition("@")
    if sep:
        left = left.strip()
        right = right.strip()
        if left == "" or right == "":
            raise ValueError(f"Invalid RELION image reference (expected N@path): {s!r}")
        index_1based = int(left)
        if index_1based < 1:
            raise ValueError(f"RELION image index must be >= 1, got {index_1based} in {s!r}")
        return right, index_1based - 1

    # Some STAR files may store direct image paths without an explicit stack index.
    return s, 0
