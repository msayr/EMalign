"""Utilities for tile datasets whose metadata is encoded in directory/file names."""

import os
import re

from glob import glob

FILE_EXT = '.tif'


def parse_stack_name(stack_path):
    return os.path.basename(os.path.normpath(stack_path))


def get_tileset_resolution(tileset_path):
    """Manual backend does not read resolution metadata from disk."""
    del tileset_path
    return None


def get_tilesets(main_dir, resolution, dir_pattern, num_workers):
    """Find stack directories that contain manually named tile folders.

    Resolution is provided by user config and intentionally ignored.
    """
    del resolution, num_workers

    stack_dirs = sorted([d for d in glob(os.path.join(main_dir, '*')) if os.path.isdir(d)])

    selected = []
    for stack_dir in stack_dirs:
        stack_name = os.path.basename(os.path.normpath(stack_dir))
        if dir_pattern and dir_pattern != [''] and not any(p in stack_name for p in dir_pattern):
            continue

        has_manual_tiles = len(glob(os.path.join(stack_dir, 'tiles', 't*_??_??'))) > 0
        if has_manual_tiles:
            selected.append(os.path.abspath(stack_dir))

    return selected


def build_slice_to_tilemap(stack_path):
    """Build {slice: {(x, y): path}} for manual directory layouts.

    Expected structure:
      <stack>/tiles/t<set>_<x>_<y>/*_s00000.tif
    """
    tile_paths = glob(os.path.join(stack_path, 'tiles', 't*_??_??', f'*{FILE_EXT}'))
    if not tile_paths:
        return {}

    slice_to_tilemap = {}
    for tile_path in sorted(tile_paths):
        z = parse_slice_from_name(tile_path)
        yx = parse_yx_pos_from_name(tile_path)
        if z not in slice_to_tilemap:
            slice_to_tilemap[z] = {}
        slice_to_tilemap[z][yx] = tile_path

    return slice_to_tilemap


def parse_yx_pos_from_name(n):
    """Parse (x, y) grid index from parent directory name.

    Expected directory form: .../tiles/t<set>_<x>_<y>/<file>.tif
    """
    tile_dir = os.path.basename(os.path.dirname(os.path.abspath(n)))
    m = re.match(r'^t\d+_(\d+)_(\d+)$', tile_dir)
    if m is None:
        raise ValueError(f'Could not parse tile position from parent directory: {n}')

    x_pos = int(m.group(1))
    y_pos = int(m.group(2))
    return (x_pos, y_pos)


def parse_slice_from_name(n):
    m = re.search(r'_s(\d+)\.tif$', os.path.basename(n))
    if m is None:
        raise ValueError(f'Could not parse slice index from filename: {n}')
    return int(m.group(1))
