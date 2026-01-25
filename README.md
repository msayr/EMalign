# EMalign

A Python package for aligning Serial Block-face Electron Microscopy (SBEM) image tiles into 3D volumetric stacks using [SOFIMA](https://github.com/google-research/sofima) (Scalable Optical Flow-based Image Montaging and Alignment).

EMalign processes TIF tiles from ThermoFisher's VolumeScope/MAPs software and outputs aligned Zarr volumes suitable for downstream analysis and visualization.

## Features

- **XY Alignment (Within-Slice Stitching)**: Stitches individual tile images within each microscopy slice using rigid and elastic alignment
- **Z Alignment (Cross-Slice Alignment)**: Aligns consecutive slices along the Z-axis using optical flow-based methods
- **GPU Acceleration**: JAX/XLA-based computing for efficient GPU utilization with multi-GPU support
- **Zarr Output**: TensorStore integration for efficient chunked volumetric storage
- **Progress Tracking**: Optional MongoDB backend for monitoring long-running jobs
- **Neuroglancer Integration**: Built-in 3D visualization for quality control
- **Extensible Backend System**: Support for custom microscope file formats

## Installation

```bash
pip install -e .
```

### Dependencies

Core dependencies:
- numpy
- tensorstore
- JAX/XLA (for GPU acceleration)
- OpenCV (cv2)
- sofima
- connectomics

Optional dependencies:
- pymongo (progress tracking)
- neuroglancer (visualization)

## Quick Start

### 1. XY Alignment (Tile Stitching)

**Prepare configuration** (no GPU required):
```bash
python -m emalign.prep_config_xy \
  -m /path/to/tiles \
  -p /path/to/project_dir \
  -o output_name \
  -res 8.0 \
  -c 4
```

**Execute alignment** (GPU required):
```bash
CUDA_VISIBLE_DEVICES=0 python -m emalign.align_dataset_xy \
  -cfg /path/to/main_config.json \
  -c 4 \
  --overwrite
```

### 2. Z Alignment (Cross-Slice Alignment)

**Prepare configuration** (no GPU required):
```bash
python -m emalign.prep_config_z \
  -cfg /path/to/main_config.json \
  -cfg-z /path/to/z_config.json \
  -o /path/to/config/z_config/ \
  -c 4 \
  --exclude /flow _mask
```

**Execute alignment** (GPU required):
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m emalign.align_dataset_z \
  -cfg /path/to/config/z_config/ \
  -cfg-z /path/to/z_config.json \
  -c 4 \
  -ds 10
```

### 3. Inspect Results

```bash
python -m emalign.inspect_dataset /path/to/output.zarr
```

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                          EMalign Pipeline                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Input: TIF Tiles (VolumeScope/MAPs format)                         │
│         Tile_XX-YY_sZZZZ.tif                                        │
│                                                                     │
│  ┌─────────────┐     ┌─────────────┐                                │
│  │ prep_config │────>│ align_xy    │  XY Alignment                  │
│  │    _xy      │     │             │  (within-slice stitching)      │
│  └─────────────┘     └──────┬──────┘                                │
│                             │                                       │
│                             v                                       │
│  ┌─────────────┐     ┌─────────────┐                                │
│  │ prep_config │────>│ align_z     │  Z Alignment                   │
│  │    _z       │     │             │  (cross-slice alignment)       │
│  └─────────────┘     └──────┬──────┘                                │
│                             │                                       │
│                             v                                       │
│  Output: Aligned Zarr Volume (.zarr)                                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Configuration

### XY Main Configuration (`main_config.json`)

```json
{
  "project_name": "my_project",
  "main_dir": "/path/to/tiles",
  "output_path": "/path/to/output.zarr",
  "resolution": [8.0, 8.0],
  "offset": [0, 0, 0],
  "stride": 20,
  "apply_gaussian": false,
  "apply_clahe": false,
  "stack_configs": {
    "stack_name": "/path/to/stack_config.json"
  },
  "mongodb_config_filepath": null
}
```

### Z Parameters Configuration (`config_z.json`)

```json
{
  "scale_flow": 0.5,
  "stride": 20,
  "patch_size": [160, 160],
  "max_deviation": 5,
  "max_magnitude": 0,
  "step_slices": 1,
  "yx_target_resolution": [8.0, 8.0],
  "k0": 0.01,
  "k": 0.4,
  "gamma": 0.5,
  "flow": {},
  "mesh": {},
  "warp": {}
}
```

## GPU Setup

EMalign requires proper environment configuration for GPU acceleration. These settings must be applied **before** importing JAX:

```bash
# Prevent GPU memory pre-allocation (critical for avoiding OOM)
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# Performance tuning
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Select GPUs
export CUDA_VISIBLE_DEVICES=0,1
```

These environment variables are automatically set in the main entry points (`align_dataset_xy.py` and `align_dataset_z.py`).

## Project Structure

```
emalign/
├── align_dataset_xy.py      # Main entry: XY alignment execution
├── align_dataset_z.py       # Main entry: Z alignment execution
├── prep_config_xy.py        # XY config generator (no GPU)
├── prep_config_z.py         # Z config generator (no GPU)
├── inspect_dataset.py       # Neuroglancer visualization
│
├── align_xy/                # XY stitching modules
│   ├── stitch_ongrid.py     # Rigid alignment
│   ├── stitch_offgrid.py    # Elastic alignment
│   ├── tile_map_positions.py
│   └── render.py
│
├── align_z/                 # Z alignment modules
│   ├── align_z.py           # Core optical flow alignment
│   ├── config.py            # Config validation
│   └── render.py
│
├── arrays/                  # Data structures
│   ├── stacks.py            # Stack management
│   ├── tile_map.py          # TileMap class
│   └── overlap.py
│
├── io/                      # I/O utilities
│   ├── store.py             # TensorStore/Zarr operations
│   ├── progress.py          # MongoDB progress tracking
│   ├── tif.py               # TIF file handling
│   └── volumescope.py       # VolumeScope format parser
│
├── scripts/                 # Lower-level scripts
│   ├── align_stack_xy.py
│   ├── align_stack_z.py
│   └── fuse_stacks_xy.py
│
└── visualize/               # Neuroglancer integration
    └── nglancer.py
```

## API Usage

### Stack Management

```python
from emalign.arrays.stacks import Stack

stack = Stack(stack_name, tile_maps_paths, tile_maps_invert)
tile_map = stack.get_tile_map(z, apply_gaussian=True, apply_clahe=False)
```

### TensorStore Operations

```python
from emalign.io.store import open_store, write_ndarray, write_slice

# Open or create a Zarr store
dataset = open_store(
    path='/path/to/output.zarr',
    mode='a',
    dtype='uint8',
    shape=[100, 2048, 2048],
    chunks=[1, 1024, 1024]
)

# Write data
write_ndarray(dataset, arr, z=0, offsets=[0, 0], validate=True)
write_slice(dataset, arr, z=0, x_offset=0, y_offset=0)
```

### Progress Tracking

```python
from emalign.io.progress import get_mongo_client, get_mongo_db, check_progress, log_progress

client = get_mongo_client(config_filepath)
db = get_mongo_db(client, project_name)
status = check_progress(db, dataset_name, z)
log_progress(db, dataset_name, z, {'status': 'complete'})
```

## Command Line Reference

### prep_config_xy

```
python -m emalign.prep_config_xy [options]

Options:
  -m, --main-dir        Path to tile directory
  -p, --project-dir     Project output directory
  -o, --output-name     Output name
  -res, --resolution    Tile resolution in nm
  -c, --num-workers     Number of parallel workers
```

### align_dataset_xy

```
python -m emalign.align_dataset_xy [options]

Options:
  -cfg, --config        Path to main_config.json
  -c, --num-workers     Number of parallel workers
  --overwrite           Overwrite existing output
```

### prep_config_z

```
python -m emalign.prep_config_z [options]

Options:
  -cfg, --config           Path to XY main_config.json
  -cfg-z, --config-z       Path to Z parameters config
  -o, --output-dir         Output directory for configs
  -c, --num-workers        Number of parallel workers
  --exclude                Patterns to exclude from processing
  --test-transitions       Test alignment on transition zones (requires GPU)
  --force-overwrite        Overwrite existing configs
```

### align_dataset_z

```
python -m emalign.align_dataset_z [options]

Options:
  -cfg, --config           Path to Z config directory
  -cfg-z, --config-z       Path to Z parameters config
  -c, --num-workers        Number of parallel workers
  -ds, --downsample        Downsampling factor for inspection
  --wipe-progress          Reprocess specific stacks
```

## Troubleshooting

**GPU Out of Memory (OOM)**
- Ensure `XLA_PYTHON_CLIENT_PREALLOCATE=false` is set
- Reduce batch size or number of workers
- Monitor GPU memory with `nvidia-smi`

**Slow Performance**
- Verify `CUDA_VISIBLE_DEVICES` is correctly set
- Check thread settings (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`)

**Path Errors**
- Always use absolute paths for TensorStore operations
- Verify all input directories exist before running

**Reprocessing Specific Stacks**
- Use `--wipe-progress <stack>` flag to clear progress for a specific stack
- This allows reprocessing without starting from scratch

## License

See LICENSE file for details.

## Author

Valentin Gillet (valentin.gillet@biol.lu.se)

## Acknowledgments

EMalign is built on [SOFIMA](https://github.com/google-research/sofima) by Google Research for optical flow-based image alignment.
