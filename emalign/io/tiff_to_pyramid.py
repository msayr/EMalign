import argparse
import json
import logging
import glob
import os
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from tifffile import imread
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)


'''
Convert a series of TIFF images to image pyramids.
'''

TIFF_EXTENSIONS = {'.tif', '.tiff'}


class JsonNumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def natural_sort_key(path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', str(path))]


def rotate_image_pil(img, angle, center=None):
    '''Use PIL to rotate large images that opencv cannot handle without cropping.'''
    image = Image.fromarray(img)
    if center is not None and not isinstance(center, tuple):
        center = tuple(center)
    return np.array(image.rotate(angle, center=center, resample=Image.BILINEAR, expand=True))


def create_CAVE_info_file(tile_shape, shape, resolution, voxel_offset, downsample_factor, max_layer):
    '''Creates info file to view images within CAVE.'''
    chunk_size = [tile_shape, tile_shape, 1]
    voxel_offset = np.array(voxel_offset, dtype=np.float64)
    resolution = np.array(resolution, dtype=np.float64)
    downsample_factor_arr = np.array([downsample_factor, downsample_factor, 1], dtype=np.float64)
    scales = [{
        'chunk_sizes': [chunk_size], 'encoding': 'jpeg', 'key': 'volume/0',
        'resolution': resolution.astype(int).tolist(), 'size': shape,
        'voxel_offset': voxel_offset.astype(int).tolist(),
    }]
    for layer in range(1, max_layer + 2):
        resolution = resolution * downsample_factor_arr
        voxel_offset = voxel_offset / downsample_factor_arr
        scales.append({
            'chunk_sizes': [chunk_size], 'encoding': 'jpeg', 'key': 'volume/' + str(layer),
            'resolution': resolution.astype(int).tolist(), 'size': shape,
            'voxel_offset': voxel_offset.astype(int).tolist(),
        })
    return {'data_type': 'uint8', 'num_channels': 1, 'scales': scales, 'type': 'image'}


def create_metadata(shape, resolution, voxel_offset, units, missing_layers):
    '''Create the metadata.json description consumed alongside the pyramid.'''
    width, height, sections = shape
    return {
        'volume_width_px': width,
        'volume_height_px': height,
        'volume_sections': sections,
        'extension': '.jpg',
        'resolution_x': resolution[0],
        'resolution_y': resolution[1],
        'resolution_z': resolution[2],
        'units': units,
        'offset_x_px': voxel_offset[0],
        'offset_y_px': voxel_offset[1],
        'offset_z_px': voxel_offset[2],
        'missing_layers': missing_layers,
    }


def write_single_tile(args):
    '''Write a single tile - runs in thread pool.'''
    filepath, tile, encode_params, jpeg_quality = args
    Image.fromarray(tile).save(filepath, 'JPEG', quality=jpeg_quality)


def build_pyramid(image: np.ndarray, max_layer: int) -> list:
    '''Build image pyramid using Pillow bilinear resizing.'''
    pyramid = [image]
    current = image
    for _ in range(max_layer):
        height, width = current.shape[:2]
        new_size = (max(1, width // 2), max(1, height // 2))
        current = np.array(Image.fromarray(current).resize(new_size, resample=Image.BILINEAR))
        pyramid.append(current)
    return pyramid


def write_pyramid_tiles(pyramid: list, z: int, output_path: str, tile_shape: int,
                        jpeg_quality: int = 90, executor: ThreadPoolExecutor = None,
                        skip_empty_tiles: bool = True) -> None:
    '''Write pyramid tiles to disk as JPEG images.'''
    tile_dir = os.path.join(output_path, str(z))
    os.makedirs(tile_dir, exist_ok=True)
    encode_params = None
    tiles_to_write = []
    for ds_factor, data in enumerate(pyramid):
        level_dir = os.path.join(tile_dir, str(ds_factor))
        os.makedirs(level_dir, exist_ok=True)
        if data.dtype != np.uint8:
            data = (255 * np.clip(data, 0, 1)).astype(np.uint8)
        height, width = data.shape[:2]
        tiles_y = (height + tile_shape - 1) // tile_shape
        tiles_x = (width + tile_shape - 1) // tile_shape
        for y in range(tiles_y):
            y_start = y * tile_shape
            y_end = min(y_start + tile_shape, height)
            for x in range(tiles_x):
                x_start = x * tile_shape
                x_end = min(x_start + tile_shape, width)
                tile = data[y_start:y_end, x_start:x_end]
                # Match zarr_to_pyramid.py: do not write all-black tiles,
                # including black border tiles outside the image content.
                if skip_empty_tiles and not np.any(tile):
                    continue
                pad_y = tile_shape - tile.shape[0]
                pad_x = tile_shape - tile.shape[1]
                if pad_y > 0 or pad_x > 0:
                    tile = np.pad(tile, ((0, pad_y), (0, pad_x)), mode='constant', constant_values=0)
                tiles_to_write.append((os.path.join(level_dir, f'{y}_{x}.jpg'), tile, encode_params, jpeg_quality))
    if executor is not None and len(tiles_to_write) > 1:
        futures = [executor.submit(write_single_tile, args) for args in tiles_to_write]
        for future in futures:
            future.result()
    else:
        for args in tiles_to_write:
            write_single_tile(args)




def collect_tiff_paths(input_path: str, pattern: str = '*') -> list:
    '''Return a naturally sorted list of TIFF paths from a file, directory, or glob.'''
    path = Path(input_path)

    if path.is_dir():
        paths = sorted(
            [p for p in path.glob(pattern) if p.suffix.lower() in TIFF_EXTENSIONS],
            key=natural_sort_key,
        )
    else:
        paths = sorted(
            [Path(p) for p in glob.glob(input_path) if Path(p).suffix.lower() in TIFF_EXTENSIONS],
            key=natural_sort_key,
        )
        if not paths and path.is_file() and path.suffix.lower() in TIFF_EXTENSIONS:
            paths = [path]

    if not paths:
        raise FileNotFoundError(f'No TIFF files found for input {input_path!r} with pattern {pattern!r}')

    return [str(p) for p in paths]


def as_grayscale_uint8(image: np.ndarray) -> np.ndarray:
    '''Convert a TIFF image to the single-channel uint8 format used by the pyramid writer.'''
    image = np.asarray(image)

    if image.ndim == 3:
        # Support RGB/RGBA TIFFs and single-page TIFF stacks with a singleton channel/page.
        if image.shape[-1] in (3, 4):
            image = np.array(Image.fromarray(image[..., :3]).convert('L'))
        elif image.shape[0] == 1:
            image = image[0]
        elif image.shape[-1] == 1:
            image = image[..., 0]
        else:
            raise ValueError(f'Expected a 2D image or RGB/RGBA TIFF, got shape {image.shape}')
    elif image.ndim != 2:
        raise ValueError(f'Expected a 2D image or RGB/RGBA TIFF, got shape {image.shape}')

    if image.dtype == np.uint8:
        return image

    if np.issubdtype(image.dtype, np.bool_):
        return image.astype(np.uint8) * 255

    if np.issubdtype(image.dtype, np.integer):
        info = np.iinfo(image.dtype)
        if info.max <= 255 and info.min >= 0:
            return image.astype(np.uint8)
        image = image.astype(np.float32)
        return np.clip((image - info.min) / (info.max - info.min) * 255, 0, 255).astype(np.uint8)

    image = image.astype(np.float32)
    finite = np.isfinite(image)
    if not np.any(finite):
        return np.zeros(image.shape, dtype=np.uint8)

    min_value = float(np.min(image[finite]))
    max_value = float(np.max(image[finite]))
    if max_value <= 1.0 and min_value >= 0.0:
        return np.clip(image * 255, 0, 255).astype(np.uint8)
    if max_value == min_value:
        return np.zeros(image.shape, dtype=np.uint8)

    return np.clip((image - min_value) / (max_value - min_value) * 255, 0, 255).astype(np.uint8)


def tiff_series_to_pyramid(input_path: str,
                           output_path: str,
                           pattern: str = '*',
                           max_layer: int = 5,
                           tile_shape: int = 1024,
                           downsample_factor: int = 2,
                           num_threads: int = 1,
                           rotate: int = 0,
                           duplicate_missing_slices: bool = True,
                           rotation_center: tuple = None,
                           resolution: list = None,
                           voxel_offset: list = None,
                           units: str = 'nanometer',
                           jpeg_quality: int = 90,
                           skip_empty_tiles: bool = True) -> None:
    '''Convert a directory/glob of TIFF slices to CATMAID/CAVE-compatible pyramid tiles.'''
    if downsample_factor != 2:
        raise NotImplementedError('Downsample factor different from 2 is not implemented.')

    tiff_paths = collect_tiff_paths(input_path, pattern)

    os.makedirs(output_path, exist_ok=True)
    if os.path.exists(os.path.join(output_path, 'info')):
        response = input('An info file already exists. You risk overwriting data.\nY to continue or ENTER to exit: ').lower()
        if response != 'y':
            sys.exit('Exiting.')

    rotation_angle = rotate % 360 if abs(rotate) > 360 else rotate
    resolution = resolution or [1, 1, 1]
    voxel_offset = voxel_offset or [0, 0, 0]

    logging.info(f'Collected {len(tiff_paths)} TIFF slices; output z indices will be 0-{len(tiff_paths) - 1}.')

    tile_executor = ThreadPoolExecutor(max_workers=num_threads)
    slice_has_data = {}
    copy_from = {}
    output_shape = None

    try:
        for z, tiff_path in enumerate(tqdm(tiff_paths, desc='Processing TIFF slices', unit='slices', dynamic_ncols=True)):
            try:
                data = as_grayscale_uint8(imread(tiff_path))

                if rotation_angle == 90:
                    data = np.rot90(data, k=3)
                elif rotation_angle == 180:
                    data = np.rot90(data, k=2)
                elif rotation_angle == 270:
                    data = np.rot90(data, k=1)
                elif rotation_angle not in (0, 360):
                    data = rotate_image_pil(data, rotation_angle, center=rotation_center)

                if output_shape is None:
                    output_shape = data.shape
                elif data.shape != output_shape:
                    raise ValueError(
                        f'TIFF slice {tiff_path} has shape {data.shape}; expected {output_shape}. '
                        'All slices must have the same dimensions after rotation.'
                    )

                has_data = np.any(data)
                slice_has_data[z] = has_data
                if has_data:
                    pyramid = build_pyramid(data, max_layer)
                    write_pyramid_tiles(pyramid, z, output_path, tile_shape, jpeg_quality,
                                        executor=tile_executor, skip_empty_tiles=skip_empty_tiles)
            except Exception as exc:
                logging.error(f'Error processing TIFF slice {tiff_path}: {exc}')
                slice_has_data[z] = False
                if not duplicate_missing_slices:
                    raise
    finally:
        tile_executor.shutdown(wait=True)

    if output_shape is None:
        raise ValueError('No TIFF slices were processed')

    if duplicate_missing_slices:
        last_valid_z = None
        for z in range(len(tiff_paths)):
            if slice_has_data.get(z, False):
                last_valid_z = z
            elif last_valid_z is not None:
                copy_from[z] = last_valid_z

        for dst_z, src_z in tqdm(copy_from.items(), desc='Copying empty slices', unit='slices', dynamic_ncols=True):
            src_dir = os.path.join(output_path, str(src_z))
            dst_dir = os.path.join(output_path, str(dst_z))
            if os.path.exists(src_dir):
                shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

    y, x = output_shape
    info = create_CAVE_info_file(tile_shape, [x, y, len(tiff_paths)], resolution, voxel_offset, downsample_factor, max_layer)
    with open(os.path.join(output_path, 'info'), 'w') as f:
        json.dump(info, f, indent=2, cls=JsonNumpyEncoder)

    missing_layers = [z for z in range(len(tiff_paths)) if not slice_has_data.get(z, False)]
    metadata = create_metadata(
        [x, y, len(tiff_paths)], resolution, voxel_offset, units, missing_layers
    )
    with open(os.path.join(output_path, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2, cls=JsonNumpyEncoder)

    logging.info('Done!')
    logging.info(f'Output written at: {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert a series of TIFF images to image pyramid tiles.')
    parser.add_argument('-i', '--input', dest='input_path', required=True, help='Input TIFF file, directory, or glob pattern.')
    parser.add_argument('-o', '--output', dest='output_path', required=True, help='Output directory for pyramid tiles.')
    parser.add_argument('-p', '--pattern', dest='pattern', default='*', help='Glob used when input is a directory (default: *, filtered to .tif/.tiff).')
    parser.add_argument('-l', '--max-layer', dest='max_layer', type=int, default=5, help='Maximum pyramid depth (default: 5).')
    parser.add_argument('--tile-shape', dest='tile_shape', type=int, default=1024, help='Tile size in pixels (default: 1024).')
    parser.add_argument('-ds', '--downsample-factor', dest='downsample_factor', type=int, default=2, help='Downsampling factor between levels (default: 2, other values not supported).')
    parser.add_argument('-c', '--cores', dest='num_threads', type=int, default=1, help='Number of concurrent tile-writing threads (default: 1).')
    parser.add_argument('--error-missing', dest='duplicate_missing_slices', default=True, action='store_false', help='Raise on unreadable/empty slices instead of duplicating the last valid slice.')
    parser.add_argument('--rotate', dest='rotate', type=int, default=0, help='Rotation angle in degrees (default: 0).')
    parser.add_argument('--rotation-center', metavar=('X', 'Y'), dest='rotation_center', nargs=2, type=float, default=None, help='Rotation center for non-90-degree rotations.')
    parser.add_argument('--resolution', metavar=('X', 'Y', 'Z'), dest='resolution', nargs=3, type=float, default=None, help='Voxel resolution in XYZ order (default: 1 1 1 nanometers).')
    parser.add_argument('--voxel-offset', metavar=('X', 'Y', 'Z'), dest='voxel_offset', nargs=3, type=float, default=None, help='Voxel offset in XYZ order (default: 0 0 0 pixels).')
    parser.add_argument('--units', default='nanometer', help='Physical units for resolution values in metadata.json (default: nanometer).')
    parser.add_argument('--jpeg-quality', dest='jpeg_quality', type=int, default=90, help='JPEG quality 0-100 (default: 90).')
    parser.add_argument('--write-empty-tiles', dest='skip_empty_tiles', default=True, action='store_false', help='Write all-black tiles instead of matching zarr_to_pyramid.py by skipping them.')

    args = parser.parse_args()
    tiff_series_to_pyramid(**vars(args))
