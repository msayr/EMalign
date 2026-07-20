import argparse
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import tensorstore as ts
from tifffile import imwrite
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)


"""
Convert a final aligned zarr container to a lossless TIFF image series.

Each z slice is written as one uncompressed .tiff file. The source dtype and
pixel values are preserved exactly; no scaling, normalization, compression, or
other lossy processing is applied.

CLI range options use Python-style half-open intervals: the start coordinate is
included and the end coordinate is excluded. For example, ``--slice-range 5 10``
writes z slices 5, 6, 7, 8, and 9. Passing a single z index such as
``--slice-range 5`` writes from slice 5 through the final slice.
"""


def _parse_coordinate_values(values, value_name):
    """Parse integer coordinate values from CLI or programmatic inputs."""
    if values is None:
        return None

    parsed_values = []
    for value in values:
        if isinstance(value, str):
            tokens = [
                token for token in re.split(r'[\s,]+', value.strip('()[] '))
                if token
            ]
            try:
                parsed_values.extend(int(token) for token in tokens)
            except ValueError as err:
                raise ValueError(
                    f'{value_name} values must be integers: {value!r}'
                ) from err
        else:
            try:
                parsed_values.append(int(value))
            except ValueError as err:
                raise ValueError(
                    f'{value_name} values must be integers: {value!r}'
                ) from err
    return parsed_values


def _normalize_slice_range(slice_range, z_size):
    """Validate and normalize an optional z slice range.

    The range follows Python slicing semantics: start is included and end is
    excluded. A single start value means export from that z slice to the end of
    the volume.
    """
    slice_range = _parse_coordinate_values(slice_range, 'Slice range')
    if slice_range is None:
        return [0, z_size]
    if len(slice_range) == 1:
        return [slice_range[0], z_size]
    if len(slice_range) == 2:
        return slice_range
    raise ValueError('Slice range must have 1 (start) or 2 elements ([start, end]).')


def _validate_slice_range(slice_range, z_size):
    """Validate that a slice range is inside the zarr volume."""
    start, end = slice_range
    if start < 0 or end < 0:
        raise ValueError('Slice range values must be non-negative.')
    if end <= start:
        raise ValueError('Invalid slice range: upper bound must exceed lower bound.')
    if start >= z_size or end > z_size:
        raise ValueError(f'Slice range {slice_range} is outside z dimension 0:{z_size}.')


def _normalize_crop(crop):
    """Validate and normalize an optional x-y crop rectangle."""
    crop = _parse_coordinate_values(crop, 'Crop')
    if crop is None:
        return None
    if len(crop) == 2:
        x0, y0 = 0, 0
        x1, y1 = crop
    elif len(crop) == 4:
        x0, y0, x1, y1 = crop
    else:
        raise ValueError(
            'Crop must have 2 elements ([x_max y_max]) or 4 elements '
            '([x_min y_min x_max y_max]).'
        )
    return [x0, y0, x1, y1]


def _validate_crop(crop, y_size, x_size):
    """Validate that a crop rectangle is inside the x-y image plane."""
    if crop is None:
        return
    x0, y0, x1, y1 = crop
    if any(value < 0 for value in crop):
        raise ValueError('Crop coordinates must be non-negative.')
    if x1 <= x0 or y1 <= y0:
        raise ValueError(
            'Invalid crop: bottom-right coordinate must be below and to the right of top-left.'
        )
    if x1 > x_size or y1 > y_size:
        raise ValueError(
            f'Crop {crop} is outside x-y dimensions x=0:{x_size}, y=0:{y_size}.'
        )


def _format_tiff_path(output_path, prefix, z_index, digits):
    """Build the output path for a single TIFF slice."""
    return os.path.join(output_path, f'{prefix}{z_index:0{digits}d}.tiff')


def _write_tiff_slice(dataset, z_index, output_path, prefix, digits, overwrite, crop):
    """Read one zarr slice and write it as an uncompressed TIFF."""
    output_file = _format_tiff_path(output_path, prefix, z_index, digits)
    if os.path.exists(output_file) and not overwrite:
        raise FileExistsError(
            f'Output file already exists: {output_file}. Use --overwrite to replace it.'
        )

    data = dataset[z_index].read().result()
    data = np.asarray(data)
    if crop is not None:
        x0, y0, x1, y1 = crop
        data = data[..., y0:y1, x0:x1]
    imwrite(output_file, data, compression=None)
    return output_file


def zarr_to_tiff_series(dataset_path,
                        output_path,
                        slice_range=None,
                        prefix='slice_',
                        digits=6,
                        num_threads=1,
                        overwrite=False,
                        crop=None):
    """
    Convert a zarr dataset into an uncompressed TIFF image series.

    Args:
        dataset_path: Path to the aligned zarr dataset.
        output_path: Directory where .tiff files will be written.
        slice_range: Optional z range as [start] or [start, end]. Start is
            inclusive and end is exclusive. Passing only start writes from that
            z slice through the final slice.
        prefix: Filename prefix for each TIFF.
        digits: Zero-padding width for z indices in filenames.
        num_threads: Number of concurrent slice-writing threads.
        overwrite: Whether to overwrite existing output TIFF files.
        crop: Optional x-y crop rectangle as [x_max, y_max] or
            [x_min, y_min, x_max, y_max]. Maximum coordinates are exclusive.
    """
    os.makedirs(output_path, exist_ok=True)

    dataset = ts.open({
        'driver': 'zarr',
        'kvstore': {
            'driver': 'file',
            'path': dataset_path,
        }
    }, read=True).result()

    if len(dataset.shape) < 3:
        raise ValueError(
            f'Expected at least a 3D zarr dataset with z as the first axis, got shape {dataset.shape}.'
        )

    slice_range = _normalize_slice_range(slice_range, dataset.shape[0])
    _validate_slice_range(slice_range, dataset.shape[0])
    start, end = slice_range

    crop = _normalize_crop(crop)
    _validate_crop(crop, dataset.shape[-2], dataset.shape[-1])

    logging.info(f'Input zarr: {dataset_path}')
    logging.info(f'Input shape: {dataset.shape}')
    logging.info(f'Input dtype: {dataset.dtype}')
    logging.info(f'Writing slices {start}:{end} as uncompressed TIFF image series')
    if crop is not None:
        x0, y0, x1, y1 = crop
        logging.info(f'Cropping x-y rectangle: x={x0}:{x1}, y={y0}:{y1}')
    logging.info(f'Output directory: {output_path}')

    z_indices = list(range(start, end))
    if num_threads < 1:
        raise ValueError('Number of threads must be at least 1.')

    if num_threads == 1:
        for z_index in tqdm(
                z_indices, desc='Writing TIFF slices', unit='slices', dynamic_ncols=True):
            _write_tiff_slice(dataset, z_index, output_path, prefix, digits, overwrite, crop)
    else:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(
                    _write_tiff_slice,
                    dataset,
                    z_index,
                    output_path,
                    prefix,
                    digits,
                    overwrite,
                    crop,
                )
                for z_index in z_indices
            ]
            for future in tqdm(
                    as_completed(futures), total=len(futures),
                    desc='Writing TIFF slices', unit='slices', dynamic_ncols=True):
                future.result()

    logging.info('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert a final aligned zarr dataset to an uncompressed TIFF image series.'
    )
    parser.add_argument('-i', '--input',
                        metavar='DATASET_PATH',
                        dest='dataset_path',
                        required=True,
                        type=str,
                        help='Path to the aligned zarr dataset.')
    parser.add_argument('-o', '--output',
                        metavar='OUTPUT_PATH',
                        dest='output_path',
                        required=True,
                        type=str,
                        help='Output directory for the TIFF image series.')
    parser.add_argument('-z', '--slice-range',
                        metavar='Z',
                        dest='slice_range',
                        nargs='+',
                        default=None,
                        help=(
                            'Optional z slice range. Pass either START to write from that z slice '
                            'through the final slice, or START END to write the half-open interval '
                            '[START, END), including START and excluding END. Example: '
                            '--slice-range 5 10 writes slices 5, 6, 7, 8, and 9. Default: all slices.'
                        ))
    parser.add_argument('--prefix',
                        metavar='PREFIX',
                        dest='prefix',
                        type=str,
                        default='slice_',
                        help='Filename prefix for output TIFFs (default: slice_).')
    parser.add_argument('--digits',
                        metavar='DIGITS',
                        dest='digits',
                        type=int,
                        default=6,
                        help='Zero-padding width for output slice indices (default: 6).')
    parser.add_argument('-c', '--cores',
                        metavar='CORES',
                        dest='num_threads',
                        type=int,
                        default=1,
                        help='Number of concurrent writer threads (default: 1).')
    parser.add_argument('--crop',
                        metavar='COORD',
                        dest='crop',
                        nargs='+',
                        default=None,
                        help=(
                            'Optional x-y crop rectangle. Pass either "x_max y_max" to crop from '
                            '(0, 0), or "x_min y_min x_max y_max". Coordinates may be passed '
                            'as separate values or one quoted string. Maximum coordinates are exclusive.'
                        ))
    parser.add_argument('--overwrite',
                        dest='overwrite',
                        action='store_true',
                        help='Overwrite existing TIFF files in the output directory.')

    args = parser.parse_args()
    try:
        zarr_to_tiff_series(**vars(args))
    except ValueError as err:
        parser.error(str(err))
