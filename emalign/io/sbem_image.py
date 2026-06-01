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


def _natural_sort_key(path):
    """Return a key that sorts stack_2 before stack_10."""

    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(path))
    ]


def _is_sbem_project_root(path):
    return (
        os.path.isdir(os.path.join(path, "tiles"))
        and os.path.isdir(os.path.join(path, "meta", "logs"))
    )


def _physical_project_roots(main_dir):
    """Return SBEM Image acquisition roots under ``main_dir`` in stack order.

    A legacy project has ``main_dir/tiles`` directly.  A multi-stack project
    has one or more child directories, each of which contains its own
    ``tiles`` and ``meta/logs`` directories.
    """

    main_dir = os.path.abspath(main_dir)

    if _is_sbem_project_root(main_dir):
        return [main_dir]

    if not os.path.isdir(main_dir):
        return []

    roots = []
    for child in os.scandir(main_dir):
        if child.is_dir() and _is_sbem_project_root(child.path):
            roots.append(os.path.abspath(child.path))

    return sorted(roots, key=_natural_sort_key)


def _sibling_physical_project_roots(project_root):
    """Return sibling acquisition roots for the root containing a tile path."""

    if project_root is None:
        return []

    parent = os.path.dirname(os.path.abspath(project_root))
    roots = _physical_project_roots(parent)

    if os.path.abspath(project_root) in roots and len(roots) > 1:
        return roots

    return [os.path.abspath(project_root)]


def _tile_relpath(tile_dir):
    """Return the path below ``tiles`` for a gXXXX/tXXXX directory."""

    parts = os.path.normpath(str(tile_dir)).split(os.sep)

    if "tiles" not in parts:
        return None

    tiles_i = parts.index("tiles")
    rel_parts = parts[tiles_i + 1:]

    if len(rel_parts) < 2:
        return None

    return os.path.join(*rel_parts[:2])


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

def _tile_dir_resolution(tile_dir, pixel_sizes):
    grid_name = Path(tile_dir).parent.name

    if not re.fullmatch(r"g\d+", grid_name):
        return None

    grid_idx = int(grid_name[1:])

    if grid_idx >= len(pixel_sizes):
        logging.warning(f"No pixel_size entry for {grid_name}")
        return None

    res = pixel_sizes[grid_idx]
    return res, res


def _latest_pixel_sizes(project_root):
    logs_dir = os.path.join(project_root, "meta", "logs")

    if not os.path.isdir(logs_dir):
        logging.warning(f"Could not find logs directory: {logs_dir}")
        return None

    config_files = sorted(glob(os.path.join(logs_dir, "config_*.txt")))

    if not config_files:
        logging.warning(f"No config_*.txt files found in {logs_dir}")
        return None

    latest_config = config_files[-1]
    pixel_sizes = _read_pixel_sizes(latest_config)

    if pixel_sizes is None:
        logging.warning(f"No pixel_size found in {latest_config}")
        return None

    return pixel_sizes


def get_tile_paths(stack_path):
    """Return TIFF paths for one logical SBEM tile stack.

    For legacy projects this is the sorted content of ``stack_path``.  For
    multi-stack projects, ``stack_path`` identifies a ``tiles/gXXXX/tXXXX``
    directory in one physical acquisition stack and this function concatenates
    matching tile directories from sibling acquisition stacks.  If later
    acquisition stacks repeat a slice index, the image from the earlier stack is
    retained and the duplicate is ignored.
    """

    rel_tile = _tile_relpath(stack_path)

    if rel_tile is None:
        return sorted(glob(os.path.join(stack_path, f"*{FILE_EXT}")))

    project_root = _find_project_root(stack_path)
    project_roots = _sibling_physical_project_roots(project_root)
    slice_to_path = {}

    for root in project_roots:
        tile_dir = os.path.join(root, "tiles", rel_tile)

        if not os.path.isdir(tile_dir):
            continue

        for tile_path in sorted(
            glob(os.path.join(tile_dir, f"*{FILE_EXT}")),
            key=_natural_sort_key,
        ):
            z = parse_slice_from_name(tile_path)

            if z in slice_to_path:
                logging.info(
                    "Ignoring duplicate SBEM Image slice %s from %s; keeping %s",
                    z,
                    tile_path,
                    slice_to_path[z],
                )
                continue

            slice_to_path[z] = tile_path

    return [slice_to_path[z] for z in sorted(slice_to_path)]


def get_tilesets(main_dir, resolution, dir_patterns=None, num_workers=None):
    """
    Return sorted logical tXXXX directories matching resolution.

    Supported image locations:
        main_dir/tiles/gXXXX/tXXXX/*.tif
        main_dir/stack_*/tiles/gXXXX/tXXXX/*.tif

    In multi-stack projects, matching ``gXXXX/tXXXX`` directories from each
    physical acquisition stack are returned once as one logical tile stack.
    """

    if dir_patterns is None:
        dir_patterns = []

    target_resolution = tuple(_clean_resolution(v) for v in resolution)
    project_roots = _physical_project_roots(main_dir)

    if not project_roots:
        logging.warning(
            f"Could not find SBEM Image tiles/meta/logs under: {main_dir}"
        )
        return []

    logical_tiles = {}

    for project_root in project_roots:
        tiles_root = os.path.join(project_root, "tiles")

        if not os.path.isdir(tiles_root):
            logging.warning(f"Could not find tiles directory: {tiles_root}")
            continue

        pixel_sizes = _latest_pixel_sizes(project_root)

        if pixel_sizes is None:
            continue

        tile_dirs = sorted(
            glob(os.path.join(tiles_root, "g*", "t*", "")),
            key=_natural_sort_key,
        )

        for tile_dir in tile_dirs:
            grid_resolution = _tile_dir_resolution(tile_dir, pixel_sizes)

            if grid_resolution != target_resolution:
                continue

            if dir_patterns:
                norm_path = os.path.normpath(tile_dir)
                if not any(pattern in norm_path for pattern in dir_patterns):
                    continue

            if not glob(os.path.join(tile_dir, f"*{FILE_EXT}")):
                continue

            rel_tile = _tile_relpath(tile_dir)

            if rel_tile is None:
                continue

            # Keep the first acquisition stack as the representative path for
            # tile directories that continue across multiple physical stacks.
            logical_tiles.setdefault(rel_tile, os.path.abspath(tile_dir))

    return [logical_tiles[k] for k in sorted(logical_tiles, key=_natural_sort_key)]


def get_stack_name(stack_path):
    """Return a unique stack name for an SBEM Image tile directory.

    SBEM Image stores stacks under ``tiles/gXXXX/tXXXX``. Tile directory
    basenames such as ``t0000`` are not globally unique across grids, so include
    both the grid and tile directory names in the stack identifier.
    """

    path = Path(stack_path)
    tile_name = path.name
    grid_name = path.parent.name

    if re.fullmatch(r"g\d+", grid_name) and re.fullmatch(r"t\d+", tile_name):
        return f"{grid_name}_{tile_name}"

    return path.name

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


def _read_imagelist_entries(imagelist_path):
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

    return entries


def _build_tile_yx_pos_map(entries):
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


def _build_tile_yx_pos_map_from_imagelist(imagelist_path):
    return _build_tile_yx_pos_map(_read_imagelist_entries(imagelist_path))


def _ensure_tile_yx_pos_map(n):
    global _TILE_YX_POS, _TILE_YX_SOURCE, _TILE_YX_PROJECT_ROOT

    lookup_keys = _lookup_keys(n)
    project_root = _find_project_root(n)
    project_roots = _sibling_physical_project_roots(project_root)
    cache_root = (
        os.path.dirname(project_root) if len(project_roots) > 1 else project_root
    )

    if (
        cache_root is not None
        and cache_root == _TILE_YX_PROJECT_ROOT
        and any(key in _TILE_YX_POS for key in lookup_keys)
    ):
        return

    if project_root is None:
        raise KeyError(
            f"No cached tile position found for {_basename(n)!r}, and project root "
            f"could not be inferred from path {n!r}."
        )

    imagelist_files = []
    for root in project_roots:
        logs_dir = os.path.join(root, "meta", "logs")
        root_imagelist_files = sorted(
            glob(os.path.join(logs_dir, "imagelist_*.txt"))
        )

        if not root_imagelist_files:
            raise FileNotFoundError(f"No imagelist_*.txt files found in {logs_dir}")

        imagelist_files.extend(root_imagelist_files)

    if len(project_roots) > 1:
        entries = []
        for imagelist_path in imagelist_files:
            entries.extend(_read_imagelist_entries(imagelist_path))

        tile_map = _build_tile_yx_pos_map(entries)

        if any(key in tile_map for key in lookup_keys):
            _TILE_YX_POS = tile_map
            _TILE_YX_SOURCE = os.path.dirname(project_root)
            _TILE_YX_PROJECT_ROOT = cache_root
            return
    else:
        for imagelist_path in imagelist_files:
            tile_map = _build_tile_yx_pos_map_from_imagelist(imagelist_path)

            if any(key in tile_map for key in lookup_keys):
                _TILE_YX_POS = tile_map
                _TILE_YX_SOURCE = imagelist_path
                _TILE_YX_PROJECT_ROOT = cache_root
                return

    logs_dirs = [os.path.join(root, "meta", "logs") for root in project_roots]
    raise KeyError(
        f"No tile position found for {_basename(n)!r} in imagelist files under {logs_dirs}"
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
