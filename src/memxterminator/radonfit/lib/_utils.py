import mrcfile
import starfile
import numpy as np
import cupy as cp
import pandas as pd
import matplotlib.pyplot as plt
from cupyx.scipy.ndimage import convolve
from collections.abc import Mapping
from typing import Optional, Tuple

def readmrc(filename, section=0, mode='cpu'):
    with mrcfile.open(filename, permissive=True) as mrc:
        data = mrc.data
        if len(data.shape) == 2:
            gray_image = data.copy()
        elif len(data.shape) == 3:
            gray_image = data[int(section)].copy()
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


def parse_relion_image_name_1based(image_name: str) -> tuple[str, int]:
    """
    Parse a RELION image reference and return a 1-based index.

    Returns
    -------
    (mrc_path, index_1based)
    """
    mrc_path, section0 = parse_relion_image_name(image_name)
    return mrc_path, section0 + 1


def _normalise_star_block_name(block_name: str) -> str:
    name = str(block_name).strip().lower()
    if name.startswith("data_"):
        name = name[len("data_"):]
    return name


def get_relion_optics_table(star_tables: Mapping) -> Optional[pd.DataFrame]:
    """
    Return the RELION optics table if present (RELION 3.1+), otherwise None.
    """
    # Prefer explicit optics block names.
    for block_name, df in star_tables.items():
        if isinstance(df, pd.DataFrame) and _normalise_star_block_name(block_name) == "optics":
            return df

    # Fallback: any block name that contains 'optics'
    for block_name, df in star_tables.items():
        if isinstance(df, pd.DataFrame) and "optics" in _normalise_star_block_name(block_name):
            return df

    # Fallback heuristic: presence of typical optics columns
    for _, df in star_tables.items():
        if not isinstance(df, pd.DataFrame):
            continue
        if "rlnImagePixelSize" in df.columns and "rlnOpticsGroup" in df.columns:
            return df

    return None


def get_relion_particles_table(star_tables: Mapping, required_columns: Optional[list[str]] = None) -> pd.DataFrame:
    """
    Select the most likely per-particle table from a STAR file.

    RELION 3.1+ STAR files often contain multiple blocks such as an optics table
    and a particles table. This helper picks the appropriate DataFrame.
    """
    if required_columns is None:
        required_columns = ["rlnImageName"]

    tables = [(name, df) for name, df in star_tables.items() if isinstance(df, pd.DataFrame)]
    if not tables:
        raise TypeError("No tabular STAR blocks (DataFrames) available")
    if len(tables) == 1:
        return tables[0][1]

    candidates = []
    for block_name, df in tables:
        cols = set(df.columns)
        if not all(col in cols for col in required_columns):
            continue

        score = 0
        norm_name = _normalise_star_block_name(block_name)
        if "particles" in norm_name:
            score += 10_000
        if "images" in norm_name:
            score += 1_000

        for col in [
            "rlnMicrographName",
            "rlnCoordinateX",
            "rlnCoordinateY",
            "rlnAnglePsi",
            "rlnClassNumber",
            "rlnOriginX",
            "rlnOriginY",
            "rlnOriginXAngst",
            "rlnOriginYAngst",
            "rlnOpticsGroup",
        ]:
            if col in cols:
                score += 50

        score += len(cols)
        candidates.append((score, block_name, df))

    if not candidates:
        available = {name: list(df.columns) for name, df in tables}
        raise KeyError(
            "Cannot find a suitable particles table in STAR blocks. "
            f"required_columns={required_columns}, available_blocks={list(available.keys())}"
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][2]


def get_relion_image_pixel_size_angstrom(
    df_particles: pd.DataFrame,
    star_tables: Optional[Mapping] = None,
    optics_df: Optional[pd.DataFrame] = None,
) -> pd.Series:
    """
    Return particle image pixel size in Å/pixel as a Series aligned to df_particles.

    This is primarily used to convert origin shifts in Å (`rlnOriginXAngst`) to
    pixels when RELION 3.1+ writes translation parameters in physical units.
    """
    if "rlnImagePixelSize" in df_particles.columns:
        px = pd.to_numeric(df_particles["rlnImagePixelSize"], errors="coerce")
        if px.isna().any():
            raise ValueError("Found NaN in rlnImagePixelSize column")
        return px

    if optics_df is None and star_tables is not None:
        optics_df = get_relion_optics_table(star_tables)

    if optics_df is None:
        raise KeyError(
            "Cannot convert origin shifts in Å to pixels: optics table not found and "
            "rlnImagePixelSize is not present in the particles table."
        )

    if "rlnImagePixelSize" not in optics_df.columns:
        raise KeyError(
            "Optics table does not contain rlnImagePixelSize, cannot convert Å shifts to pixels."
        )

    if "rlnOpticsGroup" in df_particles.columns and "rlnOpticsGroup" in optics_df.columns:
        pixel_size_map = dict(
            zip(
                pd.to_numeric(optics_df["rlnOpticsGroup"], errors="coerce").astype(int),
                pd.to_numeric(optics_df["rlnImagePixelSize"], errors="coerce").astype(float),
            )
        )
        px = pd.to_numeric(df_particles["rlnOpticsGroup"], errors="coerce").astype(int).map(pixel_size_map)
        if px.isna().any():
            raise KeyError(
                "Some particles have an rlnOpticsGroup that cannot be mapped to optics rlnImagePixelSize."
            )
        return px.astype(float)

    if len(optics_df) == 1:
        px = float(pd.to_numeric(optics_df["rlnImagePixelSize"].iloc[0], errors="coerce"))
        if not np.isfinite(px) or px <= 0:
            raise ValueError(f"Invalid rlnImagePixelSize value in optics table: {px!r}")
        return pd.Series(px, index=df_particles.index, dtype=float)

    raise KeyError(
        "Multiple optics groups present but particles table has no rlnOpticsGroup column."
    )


def get_relion_origin_shifts_pixels(
    df_particles: pd.DataFrame,
    star_tables: Optional[Mapping] = None,
    optics_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.Series, pd.Series]:
    """
    Return (origin_x_pix, origin_y_pix) as pixel units for a RELION particles table.

    Supports both:
    - `rlnOriginX` / `rlnOriginY` (pixels)
    - `rlnOriginXAngst` / `rlnOriginYAngst` (Å), converted using `rlnImagePixelSize`
    """
    if "rlnOriginX" in df_particles.columns and "rlnOriginY" in df_particles.columns:
        dx = pd.to_numeric(df_particles["rlnOriginX"], errors="coerce").fillna(0.0)
        dy = pd.to_numeric(df_particles["rlnOriginY"], errors="coerce").fillna(0.0)
        return dx, dy

    angst_pairs = [
        ("rlnOriginXAngst", "rlnOriginYAngst"),
        ("rlnOriginXAngstrom", "rlnOriginYAngstrom"),
    ]
    for col_x, col_y in angst_pairs:
        if col_x in df_particles.columns and col_y in df_particles.columns:
            dx_a = pd.to_numeric(df_particles[col_x], errors="coerce").fillna(0.0)
            dy_a = pd.to_numeric(df_particles[col_y], errors="coerce").fillna(0.0)
            px = get_relion_image_pixel_size_angstrom(
                df_particles, star_tables=star_tables, optics_df=optics_df
            )
            if (px <= 0).any():
                raise ValueError("Found non-positive pixel size while converting Å shifts to pixels")
            return dx_a / px, dy_a / px

    raise KeyError(
        "Cannot find origin shift columns: expected either (rlnOriginX, rlnOriginY) in pixels "
        "or (rlnOriginXAngst, rlnOriginYAngst) in Å."
    )
