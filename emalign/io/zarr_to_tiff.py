#!/usr/bin/env python3

import argparse
from pathlib import Path

import zarr
import tifffile as tiff


def convert_zarr_to_tiff_series(input_dir, output_dir, prefix=None):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    if not (input_dir / ".zarray").exists():
        raise ValueError(f"Input does not look like a Zarr array: {input_dir}")

    arr = zarr.open(str(input_dir), mode="r")

    print(f"Opened Zarr array: {input_dir}")
    print(f"Shape: {arr.shape}")
    print(f"Dtype: {arr.dtype}")

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D Zarr array, got shape: {arr.shape}")

    if prefix is None:
        prefix = input_dir.name

    n_z = arr.shape[0]
    pad = len(str(n_z - 1))

    for z in range(n_z):
        slice_2d = arr[z, :, :]

        out_path = output_dir / f"{prefix}_z{z:0{pad}d}.tif"

        tiff.imwrite(
            out_path,
            slice_2d,
            imagej=True,
            metadata={"axes": "YX"},
        )

        print(f"Saved {z + 1}/{n_z}: {out_path}")

    print(f"Done. Saved {n_z} TIFF slices to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a 3D Zarr array container to a TIFF image series."
    )

    parser.add_argument(
        "-i", "--input-dir",
        required=True,
        help="Input Zarr array directory, e.g. /path/to/arasia",
    )

    parser.add_argument(
        "-o", "--output-dir",
        required=True,
        help="Output directory for TIFF slices",
    )

    parser.add_argument(
        "-p", "--prefix",
        default=None,
        help="Optional filename prefix for output slices",
    )

    args = parser.parse_args()

    convert_zarr_to_tiff_series(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        prefix=args.prefix,
    )


if __name__ == "__main__":
    main()
