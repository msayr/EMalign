"""Create an XY-downsampled copy of a Zarr image stack.

This standalone utility reproduces the inspection-stack downsampling done by
``align_dataset_z`` without running alignment. Only the Y and X axes are
resampled; the Z axis is copied slice-for-slice and remains unchanged.

Example:
    python -m emalign.downsample_stack_xy /path/source.zarr /path/10x_source.zarr
"""

import argparse
import json
import logging
import os
from typing import Iterable, Optional

import numpy as np
import scipy.ndimage
import tensorstore as ts
from tqdm import tqdm

DEFAULT_DOWNSAMPLE_FACTOR = 10
DEFAULT_CHUNK_SIZE = [1, 1024, 1024]

logging.basicConfig(level=logging.INFO)


def _open_zarr(path: str, *, read: bool = False, create: bool = False,
               delete_existing: bool = False, shape: Optional[Iterable[int]] = None,
               chunks: Optional[Iterable[int]] = None, dtype: Optional[ts.dtype] = None,
               fill_value: Optional[int] = None) -> ts.TensorStore:
    """Open or create a TensorStore-backed Zarr array."""
    spec = {
        "driver": "zarr",
        "kvstore": {"driver": "file", "path": os.path.abspath(path)},
    }

    if create:
        if shape is None or chunks is None or dtype is None:
            raise ValueError("shape, chunks, and dtype are required when creating a store")
        spec.update({
            "metadata": {
                "zarr_format": 2,
                "shape": list(shape),
                "chunks": list(chunks),
            },
            "key_encoding": "/",
            "transform": {"input_labels": ["z", "y", "x"]},
        })
        kwargs = {
            "create": True,
            "delete_existing": delete_existing,
            "dtype": dtype,
        }
        if fill_value is not None:
            kwargs["fill_value"] = fill_value
        store = ts.open(spec, **kwargs).result()

        parent = os.path.dirname(os.path.abspath(path))
        if parent.endswith(".zarr"):
            with open(os.path.join(parent, ".zgroup"), "w") as f:
                json.dump({"zarr_format": 2}, f)

        return store

    return ts.open(spec, read=read).result()


def _read_attrs(path: str) -> dict:
    attrs_path = os.path.join(path, ".zattrs")
    if not os.path.exists(attrs_path):
        return {}
    with open(attrs_path, "r") as f:
        return json.load(f)


def _write_attrs(path: str, attrs: dict) -> None:
    if not attrs:
        return
    with open(os.path.join(path, ".zattrs"), "w") as f:
        json.dump(attrs, f, indent=2)


def _scaled_spatial_attrs(attrs: dict, factor: int) -> dict:
    """Scale common ZYX metadata while leaving Z untouched."""
    scaled = dict(attrs)

    for key in ("resolution", "voxel_size"):
        if key in scaled and len(scaled[key]) >= 3:
            values = list(scaled[key])
            values[1] *= factor
            values[2] *= factor
            scaled[key] = values

    for key in ("voxel_offset",):
        if key in scaled and len(scaled[key]) >= 3:
            values = list(scaled[key])
            values[1] //= factor
            values[2] //= factor
            scaled[key] = values

    # Existing align_dataset_z downsampled stores keep offset in voxel units, but
    # many input stores use physical-unit offsets. Preserve the source value to
    # avoid silently changing coordinate systems.
    return scaled


def _downsample_slice_xy(image: np.ndarray, factor: int, order: int) -> np.ndarray:
    """Downsample a single 2D slice to floor(shape / factor)."""
    if image.ndim != 2:
        raise ValueError(f"Expected 2D Z slices, got {image.ndim}D slice")

    out_shape = (image.shape[0] // factor, image.shape[1] // factor)
    if out_shape[0] < 1 or out_shape[1] < 1:
        raise ValueError(
            f"Downsample factor {factor} is too large for slice shape {image.shape}"
        )

    zoom = (out_shape[0] / image.shape[0], out_shape[1] / image.shape[1])
    downsampled = scipy.ndimage.zoom(image, zoom, order=order, prefilter=order > 1)

    if downsampled.shape != out_shape:
        downsampled = downsampled[:out_shape[0], :out_shape[1]]

    return downsampled.astype(image.dtype, copy=False)


def downsample_stack_xy(source_path: str, output_path: str, *, factor: int = DEFAULT_DOWNSAMPLE_FACTOR,
                        overwrite: bool = False, interpolation: str = "linear") -> str:
    """Create a Zarr stack downsampled by ``factor`` in Y and X only.

    Args:
        source_path: Input Zarr array path with shape ``[z, y, x]``.
        output_path: Output Zarr array path to create.
        factor: Integer downsampling factor for Y and X. Z is unchanged.
        overwrite: Delete and recreate ``output_path`` if it already exists.
        interpolation: ``linear`` for greyscale images or ``nearest`` for labels.

    Returns:
        The output path.
    """
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")
    if factor == 1 and os.path.abspath(source_path) == os.path.abspath(output_path):
        raise ValueError("source and output paths must differ when factor is 1")
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(f"Output path already exists: {output_path}")
    if interpolation not in {"linear", "nearest"}:
        raise ValueError("interpolation must be 'linear' or 'nearest'")

    source = _open_zarr(source_path, read=True)
    source_shape = list(source.domain.exclusive_max)
    if len(source_shape) != 3:
        raise ValueError(f"Expected source shape [z, y, x], got {source_shape}")

    output_shape = [source_shape[0], source_shape[1] // factor, source_shape[2] // factor]
    if output_shape[1] < 1 or output_shape[2] < 1:
        raise ValueError(f"factor {factor} is too large for source shape {source_shape}")

    chunks = [DEFAULT_CHUNK_SIZE[0], min(DEFAULT_CHUNK_SIZE[1], output_shape[1]),
              min(DEFAULT_CHUNK_SIZE[2], output_shape[2])]
    output = _open_zarr(
        output_path,
        create=True,
        delete_existing=overwrite,
        shape=output_shape,
        chunks=chunks,
        dtype=source.dtype,
    )

    order = 1 if interpolation == "linear" else 0
    logging.info("Downsampling %s -> %s", source_path, output_path)
    logging.info("Input shape %s, output shape %s; Z is unchanged", source_shape, output_shape)

    for z in tqdm(range(source_shape[0]), desc="Downsampling Z slices"):
        image = source[z, :, :].read().result()
        downsampled = _downsample_slice_xy(image, factor, order)
        output[z:z + 1, :, :].write(downsampled).result()

    _write_attrs(output_path, _scaled_spatial_attrs(_read_attrs(source_path), factor))
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a standalone XY-downsampled copy of a Zarr image stack. Z is not downsampled."
    )
    parser.add_argument("source", help="Input Zarr array path with shape [z, y, x]")
    parser.add_argument("output", help="Output Zarr array path to create")
    parser.add_argument("-f", "--factor", type=int, default=DEFAULT_DOWNSAMPLE_FACTOR,
                        help=f"Integer XY downsampling factor (default: {DEFAULT_DOWNSAMPLE_FACTOR})")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output store if it exists")
    parser.add_argument("--interpolation", choices=("linear", "nearest"), default="linear",
                        help="Interpolation to use for each XY slice (default: linear)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    downsample_stack_xy(
        args.source,
        args.output,
        factor=args.factor,
        overwrite=args.overwrite,
        interpolation=args.interpolation,
    )


if __name__ == "__main__":
    main()
