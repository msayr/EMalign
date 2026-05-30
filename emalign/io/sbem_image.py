'''
Utilities for finding and reading files produced by the SBEM Image software.
'''
from pathlib import Path
from glob import glob
import logging
import os
import re

FILE_EXT = ".tif"

_TILE_YX_POS = {}
_TILE_YX_SOURCE = None
_TILE_YX_PROJECT_ROOT = None


def _clean_resolution(v):
    v = float(v)
    return int(v) if v.is_integer() else v


def _find_project_root(path):
    parts = os.path.normpath(str(path)).split(os.sep)

    if "tiles" not in parts:
        return None

    tiles_i = parts.index("tiles")
    root = os.sep.join(parts[:tiles_i])

    return root if root else os.sep


def _infer_grid_name(path):
    parts = os.path.normpath(str(path)).split(os.sep)

    for part in parts:
        if re.fullmatch(r"g\d+", part):
            return part

    return None


def _read_pixel_sizes(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()

    m = re.search(r"pixel_size\s*=\s*\[([^\]]+)\]", content)

    if m is None:
        return None

    return [
        _clean_resolution(x.strip())
        for x in m.group(1).split(",")
        if x.strip()
    ]


def get_tileset_resolution(tileset_path):
    """
    Return (tileset_path, (y_res, x_res)) or None.

    Works for paths like:
        project/tiles/g0000/t0000/
        project/tiles/g0000/
    """

    grid_name = _infer_grid_name(tileset_path)

    if grid_name is None:
        logging.warning(f"Could not infer grid index from path: {tileset_path}")
        return None

    grid_idx = int(grid_name[1:])

    project_root = _find_project_root(tileset_path)

    if project_root is None:
        logging.warning(f"Could not find 'tiles' directory in path: {tileset_path}")
        return None

    logs_dir = os.path.join(project_root, "meta", "logs")
    config_files = sorted(glob(os.path.join(logs_dir, "config_*.txt")))

    if not config_files:
        logging.warning(f"No config_*.txt files found in {logs_dir}")
        return None

    resolutions = []

    for cfg in config_files:
        pixel_sizes = _read_pixel_sizes(cfg)

        if pixel_sizes is None:
            logging.warning(f"No pixel_size found in {cfg}")
            continue

        if grid_idx >= len(pixel_sizes):
            logging.warning(f"{grid_name} not found in pixel_size list in {cfg}")
            continue

        res = pixel_sizes[grid_idx]
        resolutions.append((cfg, (res, res)))

    if not resolutions:
        logging.warning(f"Could not determine resolution for {tileset_path}")
        return None

    return tileset_path, resolutions[-1][1]

def get_tilesets(main_dir, resolution, dir_patterns=None, num_workers=None):
    """
    Return sorted tXXXX directories matching resolution.

    Expected image location:
        main_dir/tiles/gXXXX/tXXXX/*.tif

    Returns:
        [
            ".../tiles/g0000/t0000/",
            ".../tiles/g0000/t0001/",
            ".../tiles/g0001/t0000/",
        ]
    """

    if dir_patterns is None:
        dir_patterns = []

    target_resolution = tuple(_clean_resolution(v) for v in resolution)

    tiles_root = os.path.join(main_dir, "tiles")
    logs_dir = os.path.join(main_dir, "meta", "logs")

    if not os.path.isdir(tiles_root):
        logging.warning(f"Could not find tiles directory: {tiles_root}")
        return []

    if not os.path.isdir(logs_dir):
        logging.warning(f"Could not find logs directory: {logs_dir}")
        return []

    config_files = sorted(glob(os.path.join(logs_dir, "config_*.txt")))

    if not config_files:
        logging.warning(f"No config_*.txt files found in {logs_dir}")
        return []

    latest_config = config_files[-1]
    pixel_sizes = _read_pixel_sizes(latest_config)

    if pixel_sizes is None:
        logging.warning(f"No pixel_size found in {latest_config}")
        return []

    tile_dirs = sorted(glob(os.path.join(tiles_root, "g*", "t*", "")))

    stack_list = []

    for tile_dir in tile_dirs:
        grid_name = Path(tile_dir).parent.name

        if not re.fullmatch(r"g\d+", grid_name):
            continue

        grid_idx = int(grid_name[1:])

        if grid_idx >= len(pixel_sizes):
            logging.warning(f"No pixel_size entry for {grid_name}")
            continue

        res = pixel_sizes[grid_idx]
        grid_resolution = (res, res)

        if grid_resolution != target_resolution:
            continue

        if dir_patterns:
            norm_path = os.path.normpath(tile_dir)
            if not any(pattern in norm_path for pattern in dir_patterns):
                continue

        if glob(os.path.join(tile_dir, f"*{FILE_EXT}")):
            stack_list.append(os.path.abspath(tile_dir))

    return sorted(stack_list)


def _basename(path):
    return Path(str(path).replace("\\", os.sep)).name


def _tile_key_from_name(path):
    """Return the grid/tile identifier shared by all z-slices of one tile."""

    fname = _basename(path)
    m = re.search(r"_g(\d+)_t(\d+)_s\d+", fname)

    if m is None:
        return None

    return int(m.group(1)), int(m.group(2))


def _lookup_keys(path):
    keys = [_basename(path)]
    tile_key = _tile_key_from_name(path)

    if tile_key is not None:
        keys.append(tile_key)

    return keys


def _build_tile_yx_pos_map_from_imagelist(imagelist_path):
    entries = []

    with open(imagelist_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split(";")

            if len(parts) < 3:
                continue

            raw_path = parts[0]
            fname = _basename(raw_path)
            tile_key = _tile_key_from_name(fname)

            try:
                x = int(parts[1])
                y = int(parts[2])
            except ValueError:
                continue

            entries.append((fname, tile_key, y, x))

    if not entries:
        return {}

    x_vals = sorted({x for _, _, _, x in entries})
    y_vals = sorted({y for _, _, y, _ in entries})

    x_to_col = {x: i for i, x in enumerate(x_vals)}
    y_to_row = {y: i for i, y in enumerate(y_vals)}

    tile_map = {}

    for fname, tile_key, y, x in entries:
        pos = (y_to_row[y], x_to_col[x])
        tile_map[fname] = pos

        if tile_key is not None:
            tile_map[tile_key] = pos

    return tile_map


def _ensure_tile_yx_pos_map(n):
    global _TILE_YX_POS, _TILE_YX_SOURCE, _TILE_YX_PROJECT_ROOT

    lookup_keys = _lookup_keys(n)
    project_root = _find_project_root(n)

    if (
        project_root is not None
        and project_root == _TILE_YX_PROJECT_ROOT
        and any(key in _TILE_YX_POS for key in lookup_keys)
    ):
        return

    if project_root is None:
        raise KeyError(
            f"No cached tile position found for {_basename(n)!r}, and project root "
            f"could not be inferred from path {n!r}."
        )

    logs_dir = os.path.join(project_root, "meta", "logs")
    imagelist_files = sorted(glob(os.path.join(logs_dir, "imagelist_*.txt")))

    if not imagelist_files:
        raise FileNotFoundError(f"No imagelist_*.txt files found in {logs_dir}")

    for imagelist_path in imagelist_files:
        tile_map = _build_tile_yx_pos_map_from_imagelist(imagelist_path)

        if any(key in tile_map for key in lookup_keys):
            _TILE_YX_POS = tile_map
            _TILE_YX_SOURCE = imagelist_path
            _TILE_YX_PROJECT_ROOT = project_root
            return

    raise KeyError(
        f"No tile position found for {_basename(n)!r} in imagelist files under {logs_dir}"
    )


def parse_yx_pos_from_name(n):
    """
    Return relative tile position as (y, x).

    Compatible with the VolumeScope interface:
        parse_yx_pos_from_name(path) -> tuple[int, int]
    """

    _ensure_tile_yx_pos_map(n)

    for key in _lookup_keys(n):
        if key in _TILE_YX_POS:
            return _TILE_YX_POS[key]

    raise KeyError(
        f"No tile position found for {_basename(n)!r} in {_TILE_YX_SOURCE}"
    )


def parse_slice_from_name(n):
    """
    Extract z-slice index from filenames like:
        <project>_g0001_t0002_s00003.tif
    """

    m = re.search(r"_s(\d+)", _basename(n))

    if m is None:
        raise ValueError(f"Could not parse slice index from filename: {n}")

    return int(m.group(1))
