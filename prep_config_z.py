'''Prepare configuration files for Z alignment.

Usage:
    python -m emalign.prep_config_z \\
        -p /path/to/project_dir \\
        -cfg-z /path/to/z_config.json \\
        -c 4 \\
        --force-overwrite
'''

import argparse
import json
import logging
import numpy as np
import os
import sys

from glob import glob
from typing import List, Optional

from emprocess.utils.io import get_dataset_attributes

from emalign.align_z.config import add_config_metadata, validate_config_directory, CONFIG_VERSION
from emalign.align_z.utils import compute_alignment_path, determine_initial_offset, get_ordered_datasets


logging.basicConfig(level=logging.INFO)
logging.getLogger('absl').setLevel(logging.WARNING)
logging.getLogger('jax._src.xla_bridge').setLevel(logging.WARNING)

# Constants
PAD_OFFSET = np.array([1000, 1000])  # Offset to add to origin for drift correction


def load_configs_from_files(config_paths, exclude):
    '''Load configuration from XY main config files.

    Args:
        config_paths: List of paths to main config files
        exclude: List of patterns to exclude from datasets

    Returns:
        tuple: (datasets, z_offsets, yx_target_resolution,
                project_name, mongodb_config_filepath, output_path)
    '''
    try:
        with open(config_paths[0], 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f'Invalid JSON in config file {config_paths[0]}: {e}')
    except FileNotFoundError:
        raise FileNotFoundError(f'Config file not found: {config_paths[0]}')

    project_name = config.get('project_name')
    if not project_name:
        if 'output_path' not in config:
            raise KeyError(f'Config file {config_paths[0]} missing both "project_name" and "output_path"')
        project_name = os.path.basename(config['output_path']).rstrip('.zarr')
    mongodb_config_filepath = config.get('mongodb_config_filepath')

    if 'output_path' not in config:
        raise KeyError(f'Config file {config_paths[0]} missing required field "output_path"')
    output_path = config['output_path']

    if 'resolution' not in config and 'yx_target_resolution' not in config:
        raise KeyError(f'Config file {config_paths[0]} missing both "resolution" and "yx_target_resolution"')
    yx_target_resolution = config['resolution'][0] if 'resolution' in config else config['yx_target_resolution'][0]

    # Get list of datasets and offsets
    try:
        datasets, z_offsets = get_ordered_datasets(config_paths, exclude=exclude)
    except Exception as e:
        raise RuntimeError(f'Failed to load datasets from config files: {e}')

    return (datasets, z_offsets, yx_target_resolution,
            project_name, mongodb_config_filepath, output_path)


def create_alignment_configs(datasets, z_offsets, output_configs_dir, config_z,
                             destination_path, project_name, mongodb_config_filepath,
                             yx_target_resolution, save_downsampled, num_workers):
    '''Create alignment configuration files for all datasets.

    Args:
        datasets: List of tensorstore datasets
        z_offsets: Array of z offsets for each dataset
        output_configs_dir: Directory to store config files
        config_z: Z alignment configuration dictionary
        destination_path: Path to output zarr
        project_name: Name of the project
        mongodb_config_filepath: Path to MongoDB config
        yx_target_resolution: Target resolution in YX
        save_downsampled: Downsampling factor
        num_workers: Number of worker threads

    Returns:
        tuple: (root_stack, paths, reverse_order, root_offset)
    '''
    logging.info('Creating Z align configuration files...')
    logging.info(f'Configuration files will be stored at: \n    {output_configs_dir}\n')
    logging.info('Computing alignment path...')

    # Compute the paths. Some stacks may be disconnected in some parts of the dataset
    root_stack, paths, reverse_order, ds_bounds = compute_alignment_path(
        datasets, z_offsets, target_resolution=yx_target_resolution)
    
    # Determine where to start to ensure that everything fits within the canvas
    logging.info('Computing padding...')
    root_offset = determine_initial_offset(datasets, paths)
    pad_offset = PAD_OFFSET.copy()  # pad offsets to correct for any drift
    root_offset += pad_offset

    # Write alignment plan
    align_plan = {
        'root_stack': root_stack,
        'paths': paths,
        'reverse_order': reverse_order,
        'root_offset': root_offset.tolist(),
        'pad_offset': pad_offset.tolist(),
        'yx_target_resolution': yx_target_resolution,
        'dataset_local_bounds': ds_bounds,
        'destination_path': destination_path,
        'project_name': project_name
    }
    align_plan = add_config_metadata(align_plan)

    os.makedirs(output_configs_dir, exist_ok=True)
    with open(os.path.join(output_configs_dir, '00_align_plan.json'), 'w') as f:
        json.dump(align_plan, f, indent=2)

    # Write configs for each dataset
    done = []
    for i, (path, order) in enumerate(zip(paths, reverse_order)):
        for dataset_name in path:
            if dataset_name in done:
                continue
            idx = [os.path.basename(os.path.abspath(d.kvstore.path)) == dataset_name for d in datasets].index(True)
            dataset = datasets[idx]
            z_offset = int(z_offsets[idx, 0]) + ds_bounds[dataset_name][0]
            config_path = os.path.join(output_configs_dir, f'z_{dataset_name}.json')

            if dataset_name == path[0] and i == 0:
                # Very first dataset to align should be root and has no reference
                assert dataset_name == root_stack, f'First dataset ({dataset_name}) of the path is not the root stack ({root_stack})'
                first_slice = None
                xy_offset = list(map(int, root_offset))
            else:
                first_slice = z_offset - 1  # First slice is last slice from previous dataset
                xy_offset = [0, 0]

            config = {
                'destination_path': destination_path,
                'dataset_path': os.path.abspath(dataset.kvstore.path),
                'dataset_name': dataset_name,
                'alignment_path': path,
                'reverse_order': order,
                'project_name': project_name,
                'mongodb_config_filepath': mongodb_config_filepath,
                'z_offset': z_offset,
                'xy_offset': xy_offset,
                'local_z_min': ds_bounds[dataset_name][0],
                'local_z_max': ds_bounds[dataset_name][1],
                'scale': config_z['scale_flow'],
                'flow_config': config_z['flow'],
                'mesh_config': config_z['mesh'],
                'warp_config': config_z['warp'],
                'first_slice': first_slice,
                'yx_target_resolution': yx_target_resolution,
                'save_downsampled': save_downsampled,
                'num_workers': num_workers,
                'overwrite': False
            }
            config = add_config_metadata(config)

            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            done.append(dataset_name)

    logging.info(f'Configuration files were created at {output_configs_dir}')

    return root_stack, paths, reverse_order, root_offset


def prep_config_z(project_dir: str,
                  config_z_path: str,
                  config_paths: List[str] = None,
                  destination_path: Optional[str] = None,
                  exclude: List[str] = None,
                  num_workers: int = 1,
                  save_downsampled: float = 10,
                  force_overwrite: bool = False) -> str:
    '''Generate Z alignment configuration files.

    Args:
        project_dir: Directory containing the project: config directory, and output zarr
        config_z_path: Path to Z alignment parameters config
        config_paths: List of paths to XY main config files (optional, derived from project_dir if not provided)
        destination_path: Path to output zarr (optional, derived from config if not provided)
        exclude: List of patterns to exclude from datasets
        num_workers: Number of worker threads
        save_downsampled: Downsampling factor for inspection store
        force_overwrite: Whether to overwrite existing configs

    Returns:
        str: Path to the created config directory
    '''
    if exclude is None:
        exclude = []

    if config_paths is None:
        # Attempt to find the config in the project directory
        config_paths = [os.path.join(project_dir, 'config/xy_config/main_config.json')]
        if not os.path.exists(config_paths[0]):
            raise FileNotFoundError(f'Main config file not found in the project directory: {config_paths[0]}')
        logging.info(f'Config file location was determined from project directory:\n{config_paths[0]}\n')

    # Check if output directory already has configs
    output_configs_dir = os.path.join(project_dir, 'config', 'z_config')
    existing_configs = glob(os.path.join(output_configs_dir, 'z_*.json'))

    if existing_configs and not force_overwrite:
        response = input(f'Config files already exist at {output_configs_dir}.\nOverwrite? [y/N] ')
        if response.lower() != 'y':
            logging.info('Exiting without overwriting existing config')
            sys.exit(0)
        logging.info('Overwriting existing config files')

    # Load Z alignment parameters
    try:
        with open(config_z_path, 'r') as f:
            config_z = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f'Z alignment config file not found: {config_z_path}')
    except json.JSONDecodeError as e:
        raise ValueError(f'Invalid JSON in Z alignment config file {config_z_path}: {e}')

    # Load datasets from XY configs
    logging.info('Loading datasets from XY configuration files...')
    (datasets, z_offsets, yx_target_resolution,
     project_name, mongodb_config_filepath, xy_output_path) = load_configs_from_files(
        config_paths, exclude)

    # Determine destination path
    if destination_path is None:
        destination_path = os.path.join(os.path.abspath(xy_output_path), project_name)
    else:
        destination_path = os.path.join(os.path.abspath(destination_path), project_name)

    # Print dataset info
    logging.info('Datasets Z offsets:')
    for dataset, z in zip(datasets, z_offsets):
        yx_res = get_dataset_attributes(dataset)['resolution'][1:]
        logging.info(f'    {z[0]} (res: {yx_res}): {os.path.basename(os.path.abspath(dataset.kvstore.path))}')

    if isinstance(yx_target_resolution, list):
        yx_target_resolution = np.min(yx_target_resolution, axis=0).tolist()

    logging.info(f'Target resolution (yx): {yx_target_resolution}\n')
    logging.info(f'Destination path: {destination_path}\n')

    # Create configuration files
    root_stack, paths, reverse_order, root_offset = create_alignment_configs(
        datasets, z_offsets, output_configs_dir, config_z,
        destination_path, project_name, mongodb_config_filepath,
        yx_target_resolution, save_downsampled, num_workers
    )

    # Validate created configs
    is_valid, errors, warnings = validate_config_directory(output_configs_dir)

    if warnings:
        for warning in warnings:
            logging.warning(warning)

    if not is_valid:
        for error in errors:
            logging.error(error)
        raise RuntimeError('Created configuration files are invalid')

    logging.info(f'Config version: {CONFIG_VERSION}')
    logging.info(f'Root stack: {root_stack}')
    logging.info(f'Number of alignment paths: {len(paths)}')
    logging.info(f'\nConfiguration complete!')
    logging.info(f'Config directory: {output_configs_dir}')
    logging.info(f'\nTo run alignment:')
    logging.info(f'  CUDA_VISIBLE_DEVICES=0,1 python -m emalign.align_dataset_z \\')
    logging.info(f'    -p {project_dir} \\')
    logging.info(f'    -c {num_workers}')

    return output_configs_dir


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Prepare configuration files for Z alignment.')

    # Required arguments
    parser.add_argument('-p', '--project-dir',
                        metavar='PROJECT_DIR',
                        dest='project_dir',
                        required=True,
                        type=str,
                        help='Directory where the config will be written.')
    parser.add_argument('-cfg-z', '--config-z',
                        metavar='CONFIG_Z_PATH',
                        dest='config_z_path',
                        required=True,
                        type=str,
                        help='Path to Z alignment parameters config')

    # Optional arguments
    parser.add_argument('-cfg', '--config',
                        metavar='CONFIG_PATHS',
                        dest='config_paths',
                        nargs='+',
                        type=str,
                        default=None,
                        help='Path(s) to XY main config file(s)')
    parser.add_argument('-d', '--destination',
                        metavar='DESTINATION',
                        dest='destination_path',
                        type=str,
                        default=None,
                        help='Path to output zarr (default: derived from XY config)')
    parser.add_argument('--exclude',
                        metavar='EXCLUDE',
                        dest='exclude',
                        type=str,
                        nargs='+',
                        default=[],
                        help='Patterns to exclude from datasets')
    parser.add_argument('-c', '--cores',
                        metavar='CORES',
                        dest='num_workers',
                        type=int,
                        default=1,
                        help='Number of threads to use. Default: 1')
    parser.add_argument('-ds', '--downsample-scale',
                        metavar='SCALE',
                        dest='save_downsampled',
                        type=float,
                        default=10,
                        help='Downsampling factor for inspection store. Default: 10')
    parser.add_argument('--force-overwrite',
                        dest='force_overwrite',
                        action='store_true',
                        default=False,
                        help='Force overwrite of existing config files')

    args = parser.parse_args()

    prep_config_z(**vars(args))
