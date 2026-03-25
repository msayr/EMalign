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

Install JAX with CUDA support to make use of GPU (recommended)
```bash
pip install jax[cuda12]   # Replace cuda12 with your version
```

Install dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

_Installation was tested on Ubuntu 20.04 with Python 3.12._

## Other requirements

A MongoDB database is currently required for metadata and progress tracking. The free community edition is sufficient for this. It can be found [here](https://www.mongodb.com/products/self-managed/community-edition) along with installation tutorials.

Neuroglancer is currently required for data visualization. Its installation may require extra steps described [here](https://github.com/google/neuroglancer?tab=readme-ov-file#building).

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
python prep_config_z.py \
  -p /path/to/project_dir \
  -cfg-z /path/to/z_config.json \
  -c 4
```

**Execute alignment**:
```bash
CUDA_VISIBLE_DEVICES=0,1 python align_dataset_z.py \
  -cfg /path/to/config/z_config/ \
  -cfg-z /path/to/z_config.json \
  -c 4 \
  -ds 10
```

### 3. Inspect Results

```bash
python inspect_dataset.py -d /path/to/output.zarr/dataset
```

See [docs](/docs/visualization.md) for more details on how to use `inspect_dataset.py`.

## Configuration

Documentation about configuration files can be found [here](/docs/config.md).

## API Usage

## License

See LICENSE file for details.

## Author

Valentin Gillet (valentin.gillet@biol.lu.se)

## Acknowledgments

EMalign is built on [SOFIMA](https://github.com/google-research/sofima) by Google Research for optical flow-based image alignment.
