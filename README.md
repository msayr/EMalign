# EMalign

A Python package for aligning Serial Block-face Electron Microscopy (SBEM) image tiles into 3D volumetric stacks using [SOFIMA](https://github.com/google-research/sofima) (Scalable Optical Flow-based Image Montaging and Alignment).

## Installation

Clone the repository locally
```bash
git clone https://github.com/Heinze-lab/EMalign.git
```

Create a new environment and activate it (here using conda)
```bash
conda create -n myenv python=3.12
conda activate myenv
```

Install JAX with CUDA support (recommended)
```bash
pip install jax[cuda12]   # Replace cuda12 with your version
```

Install dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### 1. XY Alignment (Tile Stitching)

**Prepare configuration**:
```bash
python prep_config_xy \
  -i /path/to/tiles/directory \
  -p /path/to/project_dir \
  -o /path/to/zarr.zarr \
  -res 10 10 \
  -c 4
```

**Execute alignment**:
```bash
CUDA_VISIBLE_DEVICES=0 python align_dataset_xy \
  -cfg /path/to/main_config.json \
  -c 4
```

### 2. Z Alignment (Cross-Slice Alignment)

**Prepare configuration**:
```bash
python -m emalign.prep_config_z \
  -p /path/to/project_dir \
  -cfg-z /path/to/z_config.json \
  -c 4
```

**Execute alignment**:
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

## Configuration

Documentation about configuration files can be found [here](/docs/config.md).

## API Usage

## License

See LICENSE file for details.

## Author

Valentin Gillet (valentin.gillet@biol.lu.se)

## Acknowledgments

EMalign is built on [SOFIMA](https://github.com/google-research/sofima) by Google Research for optical flow-based image alignment.
