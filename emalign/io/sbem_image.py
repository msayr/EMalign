"""
Utilities for finding and reading files produced by SBEM image acquisitions.
"""

import ast
import logging
import os
import re

from glob import glob

FILE_EXT = '.tif'

_SESSION_CACHE = {}


def _extract_datetime_from_name(path, prefix):
    name = os.path.basename(path)
    if not name.startswith(prefix) or not name.endswith('.txt'):
        return None
    return name[len(prefix):-4]


def _find_session_pairs(logs_dir):
    config_files = glob(os.path.join(logs_dir, 'config_*.txt'))
    imagelist_files = glob(os.path.join(logs_dir, 'imagelist_*.txt'))

    configs = {_extract_datetime_from_name(p, 'config_'): p for p in config_files}
    imagelists = {_extract_datetime_from_name(p, 'imagelist_'): p for p in imagelist_files}

    common = sorted([dt for dt in configs.keys() if dt in imagelists], reverse=True)
    return [(dt, configs[dt], imagelists[dt]) for dt in common]


def _read_pixel_size_list(config_path):
    with open(config_path, 'r') as f:
        for line in f:
            if 'pixel_size' not in line:
                continue

            _, rhs = line.split('=', 1)
            values = ast.literal_eval(rhs.strip())
            if isinstance(values, list):
                return [float(v) for v in values]
    return None


def _grid_index_from_name(grid_name):
    m = re.match(r'g(\d+)$', grid_name)
    if m is None:
        return None
    return int(m.group(1))


def _select_matching_session(grid_dir, target_pixel_size):
    stack_dir = os.path.dirname(os.path.dirname(grid_dir))
    logs_dir = os.path.join(stack_dir, 'meta', 'logs')

    if not os.path.isdir(logs_dir):
        return None

    grid_name = os.path.basename(os.path.normpath(grid_dir))
    grid_idx = _grid_index_from_name(grid_name)
    if grid_idx is None:
        return None

    for _, config_path, imagelist_path in _find_session_pairs(logs_dir):
        pixel_sizes = _read_pixel_size_list(config_path)
        if pixel_sizes is None or grid_idx >= len(pixel_sizes):
            continue

        if abs(float(pixel_sizes[grid_idx]) - float(target_pixel_size)) < 1e-6:
            return imagelist_path

    return None


def parse_stack_name(stack_path):
    grid_name = os.path.basename(os.path.normpath(stack_path))
    stack_name = os.path.basename(os.path.dirname(os.path.dirname(stack_path)))
    return f'{stack_name}_{grid_name}'


def get_tileset_resolution(tileset_path):
    grid_name = os.path.basename(os.path.normpath(tileset_path))
    grid_idx = _grid_index_from_name(grid_name)
    if grid_idx is None:
        return None

    stack_dir = os.path.dirname(os.path.dirname(tileset_path))
    logs_dir = os.path.join(stack_dir, 'meta', 'logs')

    for _, config_path, _ in _find_session_pairs(logs_dir):
        pixel_sizes = _read_pixel_size_list(config_path)
        if pixel_sizes is None or grid_idx >= len(pixel_sizes):
            continue

        px = pixel_sizes[grid_idx]
        return (tileset_path, (px, px))

    return None


def get_tilesets(main_dir, resolution, dir_pattern, num_workers):
    del num_workers  # unused for this backend

    target_pixel_size = float(resolution[0])
    stack_dirs = sorted([d for d in glob(os.path.join(main_dir, '*')) if os.path.isdir(d)])

    selected = []
    for stack_dir in stack_dirs:
        stack_name = os.path.basename(os.path.normpath(stack_dir))
        if dir_pattern and dir_pattern != [''] and not any(p in stack_name for p in dir_pattern):
            continue

        grid_dirs = sorted(glob(os.path.join(stack_dir, 'tiles', 'g*')))
        for grid_dir in grid_dirs:
            imagelist_path = _select_matching_session(grid_dir, target_pixel_size)
            if imagelist_path is None:
                continue

            grid_dir_abs = os.path.abspath(grid_dir)
            _SESSION_CACHE[grid_dir_abs] = imagelist_path
            selected.append(grid_dir_abs)

    return selected


def parse_yx_pos_from_name(n):
    m = re.search(r'_t(\d+)_s\d+\.tif$', os.path.basename(n))
    if m is None:
        raise ValueError(f'Could not parse tile index from filename: {n}')
    x = int(m.group(1))
    return (0, x)


def parse_slice_from_name(n):
    m = re.search(r'_s(\d+)\.tif$', os.path.basename(n))
    if m is None:
        raise ValueError(f'Could not parse slice index from filename: {n}')
    return int(m.group(1))


def _discover_imagelist_for_grid(grid_dir):
    grid_dir_abs = os.path.abspath(grid_dir)
    if grid_dir_abs in _SESSION_CACHE:
        return _SESSION_CACHE[grid_dir_abs]

    stack_dir = os.path.dirname(os.path.dirname(grid_dir_abs))
    logs_dir = os.path.join(stack_dir, 'meta', 'logs')
    sessions = _find_session_pairs(logs_dir)
    if not sessions:
        return None

    imagelist_path = sessions[0][2]
    _SESSION_CACHE[grid_dir_abs] = imagelist_path
    return imagelist_path


def build_slice_to_tilemap(grid_dir):
    grid_dir_abs = os.path.abspath(grid_dir)
    grid_name = os.path.basename(os.path.normpath(grid_dir_abs))
    stack_dir = os.path.dirname(os.path.dirname(grid_dir_abs))

    imagelist_path = _discover_imagelist_for_grid(grid_dir_abs)
    if imagelist_path is None:
        raise FileNotFoundError(f'Could not find imagelist_*.txt for {grid_dir_abs}')

    by_slice = {}
    missing = []

    with open(imagelist_path, 'r') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or ';' not in line:
                continue

            parts = line.split(';')
            if len(parts) < 5:
                continue

            rel_path = parts[0].replace('\\', os.sep)
            if f'{os.sep}{grid_name}{os.sep}' not in f'{os.sep}{rel_path}':
                continue

            try:
                x_nm = int(parts[1])
                y_nm = int(parts[2])
                slice_idx = int(parts[4])
            except ValueError:
                continue

            tile_path = os.path.join(stack_dir, rel_path)
            if not os.path.exists(tile_path):
                missing.append(tile_path)
                continue

            by_slice.setdefault(slice_idx, []).append((x_nm, y_nm, tile_path))

    if missing:
        preview = '\n'.join(f'  - {p}' for p in sorted(missing)[:25])
        suffix = '' if len(missing) <= 25 else f'\n  ... ({len(missing) - 25} more)'
        logging.warning(
            f'Missing tiles referenced in {os.path.basename(imagelist_path)} for {grid_name}:\n{preview}{suffix}'
        )

    slice_to_tilemap = {}
    for slice_idx, entries in by_slice.items():
        ys = sorted({y for _, y, _ in entries})
        xs = sorted({x for x, _, _ in entries})

        y_to_idx = {y: i for i, y in enumerate(ys)}
        x_to_idx = {x: i for i, x in enumerate(xs)}

        tile_map = {}
        for x_nm, y_nm, tile_path in entries:
            tile_map[(y_to_idx[y_nm], x_to_idx[x_nm])] = tile_path

        if tile_map:
            slice_to_tilemap[slice_idx] = tile_map

    if not slice_to_tilemap:
        raise RuntimeError(f'No usable tiles found for grid {grid_dir_abs} using {imagelist_path}')

    return dict(sorted(slice_to_tilemap.items()))
