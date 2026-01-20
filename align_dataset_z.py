import os
# To prevent running out of memory because of preallocation
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false' # Avoid OOM by allocating memory flexibly but may slow things down
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # Log only ERRORS for XLA
# os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'cuda_async'

# Influences performance
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

import argparse
import json
import logging
import numpy as np
import tensorstore as ts
import sys

from glob import glob
from inspect import signature
from typing import List, Optional, Tuple
from emprocess.utils.io import get_dataset_attributes, set_dataset_attributes

from emalign.align_z.utils import compute_alignment_path, determine_initial_offset, get_ordered_datasets
from emalign.scripts.align_stack_z import align_stack_z
from emalign.io import open_store
from emalign.io.progress import get_mongo_client, get_mongo_db, wipe_progress

logging.basicConfig(level=logging.INFO)
logging.getLogger('absl').setLevel(logging.WARNING)
logging.getLogger('jax._src.xla_bridge').setLevel(logging.WARNING)

# Constants
PAD_OFFSET = np.array([1000, 1000]) # Offset to add to the origin of a dataset to account for drift
CHUNK_SIZE = [1, 1024, 1024] # For store creation
NUM_WORKERS = 1 
DOWNSAMPLE_SCALE = 10 # For creation of the downsampled inspection store


def load_configs_from_directory(config_paths, destination_path):
    '''Load configuration files from a directory of z*.json files.

    Args:
        config_paths: List containing a single path to directory with z*.json files
        destination_path: Optional destination path override

    Returns:
        tuple: (datasets, z_offsets, yx_target_resolution, output_configs_dir,
                project_name, mongodb_config_filepath, destination_path, create_configs)
    '''
    output_configs_dir = config_paths[0]
    z_config_files = glob(os.path.join(output_configs_dir, 'z*.json'))

    if not z_config_files:
        raise FileNotFoundError(f'No z*.json config files found in directory: {output_configs_dir}')

    try:
        with open(z_config_files[0], 'r') as f:
            first_config = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f'Invalid JSON in config file {z_config_files[0]}: {e}')
    except FileNotFoundError:
        raise FileNotFoundError(f'Config file not found: {z_config_files[0]}')

    project_name = first_config.get('project_name')
    if not project_name:
        if 'destination_path' not in first_config:
            raise KeyError(f'Config file {z_config_files[0]} missing both "project_name" and "destination_path"')
        project_name = os.path.basename(first_config['destination_path']).rstrip('.zarr')
    mongodb_config_filepath = first_config.get('mongodb_config_filepath')

    if destination_path is None:
        destination_path = os.path.abspath(first_config['destination_path'])
    else:
        destination_path = os.path.join(os.path.abspath(destination_path), project_name)

    # Get list of datasets and offsets
    datasets = []
    z_offsets = []
    for p in z_config_files:
        try:
            with open(p, 'r') as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f'Invalid JSON in config file {p}: {e}')
        except FileNotFoundError:
            raise FileNotFoundError(f'Config file not found: {p}')

        if 'dataset_path' not in config:
            raise KeyError(f'Config file {p} missing required field "dataset_path"')
        if 'z_offset' not in config:
            raise KeyError(f'Config file {p} missing required field "z_offset"')

        spec = {
                'driver': 'zarr',
                'kvstore': {
                    'driver': 'file',
                    'path': config['dataset_path']
                }
            }
        try:
            datasets.append(ts.open(spec).result())
        except Exception as e:
            raise IOError(f"Failed to open dataset at {config['dataset_path']}: {e}")

        z_offsets.append([config['z_offset']] + config.get('xy_offset', [0,0]))
    z_offsets = np.array(z_offsets)
    datasets = [datasets[i] for i in np.argsort(z_offsets[:, 0])]
    z_offsets = z_offsets[np.argsort(z_offsets[:, 0])]

    yx_target_resolution = float('inf')
    for config_file in z_config_files:
        with open(config_file, 'r') as f:
            c = json.load(f)
            r = c['resolution'][0] if 'resolution' in c else c.get('yx_target_resolution', [float('inf')])[0]
            yx_target_resolution = min(yx_target_resolution, r)

    create_configs = False

    return (datasets, z_offsets, yx_target_resolution, output_configs_dir,
            project_name, mongodb_config_filepath, destination_path, create_configs)


def load_configs_from_files(config_paths, destination_path, exclude):
    '''Load configuration from main config files.

    Args:
        config_paths: List of paths to main config files
        destination_path: Optional destination path override
        exclude: List of patterns to exclude from datasets

    Returns:
        tuple: (datasets, z_offsets, yx_target_resolution, output_configs_dir,
                project_name, mongodb_config_filepath, destination_path, create_configs)
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

    if destination_path is None:
        if 'output_path' not in config:
            raise KeyError(f'Config file {config_paths[0]} missing required field "output_path"')
        destination_path = config['output_path']

    output_configs_dir = os.path.join(os.path.dirname(os.path.abspath(destination_path)), 'config', 'z_config')
    destination_path = os.path.join(os.path.abspath(destination_path), project_name)
    create_configs = True

    if 'resolution' not in config and 'yx_target_resolution' not in config:
        raise KeyError(f'Config file {config_paths[0]} missing both "resolution" and "yx_target_resolution"')
    yx_target_resolution = config['resolution'][0] if 'resolution' in config else config['yx_target_resolution'][0]

    # Get list of datasets and offsets
    try:
        datasets, z_offsets = get_ordered_datasets(config_paths, exclude=['/flow', '_mask'] + exclude)
    except Exception as e:
        raise RuntimeError(f'Failed to load datasets from config files: {e}')

    return (datasets, z_offsets, yx_target_resolution, output_configs_dir,
            project_name, mongodb_config_filepath, destination_path, create_configs)


def create_alignment_configs(datasets, z_offsets, output_configs_dir, config_z,
                            destination_path, project_name, mongodb_config_filepath,
                            yx_target_resolution, save_downsampled, num_workers,
                            start_over):
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
        start_over: Whether to wipe existing progress

    Returns:
        tuple: (root_stack, paths, reverse_order, root_offset)
    '''
    if start_over:
        try:
            input('WARNING: All progress will be wiped and all datasets will be processed.\nPress ENTER to continue or CTRL+C to abort\n')
        except KeyboardInterrupt:
            logging.info('\nAborted by user')
            sys.exit(0)

        client = get_mongo_client(mongodb_config_filepath)
        db = get_mongo_db(client, project_name)
        for dataset in datasets:
            dataset_name = os.path.basename(os.path.abspath(dataset.kvstore.path))
            wipe_progress(db, dataset_name)
            attrs = get_dataset_attributes(dataset)
            attrs['z_aligned'] = False
            set_dataset_attributes(dataset, attrs)

    logging.info('Creating Z align configuration files...')
    logging.info(f'Configuration files will be stored at: \n    {output_configs_dir}\n')
    logging.info('Computing alignment path...')

    # Compute the paths. Some stacks may be disconnected in some parts of the dataset
    root_stack, paths, reverse_order, ds_bounds = compute_alignment_path(datasets, z_offsets, target_resolution=yx_target_resolution)
    logging.info('Computing padding...')
    root_offset = determine_initial_offset(datasets, paths)

    pad_offset = PAD_OFFSET.copy()  # pad offsets to correct for any drift
    root_offset += pad_offset

    align_plan = {
        'root_stack': root_stack,
        'paths': paths,
        'reverse_order': reverse_order,
        'root_offset': root_offset.tolist(),
        'pad_offset': pad_offset.tolist(),
        'yx_target_resolution': yx_target_resolution,
        'dataset_local_bounds': ds_bounds
    }
    os.makedirs(output_configs_dir, exist_ok=True)
    with open(os.path.join(output_configs_dir, '00_align_plan.json'), 'w') as f:
        json.dump(align_plan, f, indent='')

    for i, (path, order) in enumerate(zip(paths, reverse_order)):
        for dataset_name in path:
            idx = [os.path.basename(os.path.abspath(d.kvstore.path)) == dataset_name for d in datasets].index(True)
            dataset = datasets[idx]
            z_offset = int(z_offsets[idx, 0]) + ds_bounds[dataset_name][0] # z offset is not correct anymore because the bounds may have been shifted
            config_path = os.path.join(output_configs_dir, f'z_{dataset_name}.json')

            if dataset_name == path[0] and i == 0:
                # Very first dataset to align should be root and has no reference
                assert dataset_name == root_stack, f'First dataset ({dataset_name}) of the path is not the root stack ({root_stack})'
                first_slice = None
                xy_offset = list(map(int, root_offset))
            else:
                first_slice = z_offset - 1 # First slice is last slice from previous dataset
                xy_offset = [0,0]

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

            with open(config_path, 'w') as f:
                json.dump(config, f, indent='')
    logging.info(f'Configuration files were created at {output_configs_dir}')

    return root_stack, paths, reverse_order, root_offset


def initialize_destination_stores(destination_path, datasets, z_offsets, save_downsampled,
                                  project_name, start_over):
    '''Create or open destination zarr stores.

    Args:
        destination_path: Path to main output zarr
        datasets: List of tensorstore datasets
        z_offsets: Array of z offsets
        save_downsampled: Downsampling factor
        project_name: Name of the project
        start_over: Whether to recreate stores

    Returns:
        tuple: (destination, destination_mask, ds_destination, ds_project_output_path)
    '''
    ds_project_output_path = os.path.join(os.path.dirname(destination_path), f'{save_downsampled}x_' + project_name)
    create_new = not os.path.exists(destination_path) or start_over

    if create_new:
        logging.info(f'Creating project dataset at: \n    {destination_path}\n')
        if start_over:
            logging.info(f'Previous dataset will be overwritten')
            open_mode = 'w'
        else:
            open_mode = 'a'
        # Create container at destination if it doesn't exist or if user wants to start over
        # Shape destination starts as largest yx and last offset + shape of last dataset
        # yx could change shape based on warping but z should stay like this for this project
        shapes = np.array([dataset.shape for dataset in datasets])
        dest_shape = np.append(shapes[-1, 0] + z_offsets[-1, 0],
                               shapes[:, 1:].max(0)).tolist()

        destination = open_store(
            destination_path, mode=open_mode, dtype=ts.uint8, shape=dest_shape, chunks=CHUNK_SIZE)
        destination_mask = open_store(
            destination_path + '_mask', mode=open_mode, dtype=ts.bool, shape=dest_shape, chunks=CHUNK_SIZE)

        logging.info(f'Creating downsampled project dataset ({save_downsampled}) at: \n    {ds_project_output_path}\n')
        shapes_ds = np.array([dataset.shape for dataset in datasets])
        shapes_ds[:, 1:] = shapes_ds[:, 1:] // save_downsampled
        dest_shape_ds = np.append(shapes_ds[-1, 0] + z_offsets[-1, 0],
                                  shapes_ds[:, 1:].max(0)).tolist()

        ds_destination = open_store(
            ds_project_output_path, mode=open_mode, dtype=ts.uint8, shape=dest_shape_ds, chunks=CHUNK_SIZE)
    else:
        logging.info(f'Opening existing project dataset at: \n    {destination_path}\n')
        destination = open_store(
            destination_path, mode='r+', dtype=ts.uint8)
        destination_mask = open_store(
            destination_path + '_mask', mode='r+', dtype=ts.bool)

        logging.info(f'Opening existing downsampled project dataset ({save_downsampled}) at: \n    {ds_project_output_path}\n')
        ds_destination = open_store(
            ds_project_output_path, mode='r+', dtype=ts.uint8)

    return destination, destination_mask, ds_destination, ds_project_output_path


def execute_alignment(paths, output_configs_dir, root_stack,
                     num_workers, wipe_progress_stack):
    '''Execute the alignment for all datasets.

    Args:
        paths: List of alignment paths
        output_configs_dir: Directory containing config files
        root_stack: Name of the root stack
        num_workers: Number of worker threads
        wipe_progress_stack: Optional stack name to wipe progress for
    '''
    for i, path in enumerate(paths):
        for dataset_name in path:
            config_path = os.path.join(output_configs_dir, f'z_{dataset_name}.json')
            if dataset_name == path[0] and i == 0:
                assert dataset_name == root_stack, f'First dataset ({dataset_name}) of the path is not the root stack ({root_stack})'

            # Load configuration
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
            except FileNotFoundError:
                raise FileNotFoundError(f'Config file not found for dataset {dataset_name}: {config_path}')
            except json.JSONDecodeError as e:
                raise ValueError(f'Invalid JSON in config file {config_path}: {e}')

            config['num_workers'] = num_workers
            config['wipe_progress_flag'] = (dataset_name == wipe_progress_stack)

            # Start alignment
            try:
                params = signature(align_stack_z).parameters
                relevant_args = {k: v for k, v in config.items() if k in params}
                align_stack_z(**relevant_args)
            except KeyError as e:
                raise RuntimeError(f'Missing required parameter for {dataset_name}: {e}')
            except ValueError as e:
                raise RuntimeError(f'Invalid parameter value for {dataset_name}: {e}')
            except IOError as e:
                raise RuntimeError(f'IO error processing {dataset_name}: {e}')
            except Exception as e:
                logging.exception(f'Unexpected error processing {dataset_name}')
                raise RuntimeError(f'Error processing {dataset_name}: {e}')


def align_dataset_z(config_paths: List[str],
                    config_z_path: str,
                    destination_path: Optional[str],
                    exclude: List[str],
                    num_workers: int,
                    save_downsampled: float,
                    no_align: bool,
                    start_over: bool,
                    wipe_progress_stack: Optional[str] = None) -> None:

    #---------- Open/prepare configs ----------#
    # Read common Z alignments parameters
    try:
        with open(config_z_path, 'r') as f:
            config_z = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f'Z alignment config file not found: {config_z_path}')
    except json.JSONDecodeError as e:
        raise ValueError(f'Invalid JSON in Z alignment config file {config_z_path}: {e}')

    # If config_paths only has one item that is a dir, we assume it contains the config files
    if len(config_paths) == 1 and os.path.isdir(config_paths[0]):
        # Load dataset paths from z configs
        (datasets, z_offsets, yx_target_resolution, output_configs_dir,
         project_name, mongodb_config_filepath, destination_path, create_configs) = \
            load_configs_from_directory(config_paths, destination_path)
    else:
        # Load dataset paths from main config files
        (datasets, z_offsets, yx_target_resolution, output_configs_dir,
         project_name, mongodb_config_filepath, destination_path, create_configs) = \
            load_configs_from_files(config_paths, destination_path, exclude)
        create_configs = (not os.path.exists(output_configs_dir)) or start_over

    #---------- Compute alignment path and initial offset ----------#
    # Print some info
    logging.info('Datasets Z offsets:')
    for dataset, z in zip(datasets, z_offsets):
        yx_res = get_dataset_attributes(dataset)['resolution'][1:]
        logging.info(f'    {z[0]} (res: {yx_res}): {os.path.basename(os.path.abspath(dataset.kvstore.path))}')

    if isinstance(yx_target_resolution, list):
        yx_target_resolution = np.min(yx_target_resolution, axis=0).tolist()

    logging.info(f'Target resolution (yx): {yx_target_resolution}\n')

    if create_configs or start_over:
        # Create config files
        if start_over:
            create_configs = True

        root_stack, paths, reverse_order, root_offset = create_alignment_configs(
            datasets, z_offsets, output_configs_dir, config_z,
            destination_path, project_name, mongodb_config_filepath,
            yx_target_resolution, save_downsampled, num_workers, start_over
        )
        if no_align:
            logging.info('No alignment this time!')
            return
    else:
        # Read configuration
        align_plan_path = os.path.join(output_configs_dir, '00_align_plan.json')
        try:
            with open(align_plan_path, 'r') as f:
                align_plan = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f'Alignment plan file not found: {align_plan_path}')
        except json.JSONDecodeError as e:
            raise ValueError(f'Invalid JSON in alignment plan file {align_plan_path}: {e}')

        try:
            root_stack = align_plan['root_stack']
            paths = align_plan['paths']
            reverse_order = align_plan['reverse_order']
            root_offset = align_plan['root_offset']
        except KeyError as e:
            raise KeyError(f'Alignment plan file {align_plan_path} missing required field: {e}')
        logging.info(f'Using existing configuration files from:\n    {output_configs_dir}\n')

    #---------- Prepare destinations ----------#
    destination, destination_mask, ds_destination, ds_project_output_path = \
        initialize_destination_stores(destination_path, datasets, z_offsets,
                                     save_downsampled, project_name, start_over)

    #---------- Start alignment ----------#
    execute_alignment(paths, output_configs_dir, root_stack,
                     num_workers, wipe_progress_stack)

    logging.info('Done!')
    logging.info(f'Output: {destination_path}')


if __name__ == '__main__':


    parser=argparse.ArgumentParser('Script aligning tiles in Z based on SOFIMA (Scalable Optical Flow-based Image Montaging and Alignment).\n\
                                   The dataset must have been aligned in XY and written to a zarr container, before using this script.\n\
                                    This script was written to match the file structure produced by the ThermoFisher MAPs software.')
    # Required
    parser.add_argument('-cfg', '--config',
                        metavar='CONFIG_PATHS',
                        dest='config_paths',
                        required=True,
                        nargs='+',
                        type=str,
                        help='Path to the main task configs. \
                              Can provide one or multiple configs with the same project name to align all stacks they point to')
    parser.add_argument('-cfg-z', '--config-z',
                        metavar='CONFIG_Z_PATH',
                        dest='config_z_path',
                        required=True,
                        type=str,
                        help='Path to the z alignment task config.')
    parser.add_argument('-d', '--destination',
                        metavar='DESTINATION',
                        dest='destination_path',
                        type=str,
                        default=None,
                        help='Path to the zarr container where the final alignment will be written.')
    
    # Not required
    parser.add_argument('--exclude',
                        metavar='EXCLUDE',
                        dest='exclude',
                        type=str,
                        nargs='+',
                        default=[],
                        help='Patterns to exclude from the datasets to align.')
    parser.add_argument('-c', '--cores',
                        metavar='CORES',
                        dest='num_workers',
                        type=int,
                        default=NUM_WORKERS,
                        help=f'Number of threads to use for processing. Default: {NUM_WORKERS}')
    parser.add_argument('-ds', '--downsample-scale',
                        metavar='SCALE',
                        dest='save_downsampled',
                        type=float,
                        default=DOWNSAMPLE_SCALE,
                        help=f'Factor to use for downsampling the dataset that will be saved for inspection. Default: {DOWNSAMPLE_SCALE}')
    parser.add_argument('--no-align',
                        dest='no_align',
                        default=False,
                        action='store_true',
                        help='Do not align and only create configuration files. Default: False')
    parser.add_argument('--start-over',
                        dest='start_over',
                        default=False,
                        action='store_true',
                        help='Deletes existing output dataset and start over. Default: False')
    parser.add_argument('--wipe-progress',
                        dest='wipe_progress_stack',
                        type=str,
                        default=None,
                        help='Wipe progress for a specific stack before starting.')

    args=parser.parse_args()

    try:
        GPU_ids = os.environ['CUDA_VISIBLE_DEVICES']
    except KeyError:
        logging.error('No GPUs specified. Set CUDA_VISIBLE_DEVICES environment variable.')
        logging.error('Example: CUDA_VISIBLE_DEVICES=0,1 python align_dataset_z.py ...')
        sys.exit(1)
    logging.info(f'Using GPU IDs: {GPU_ids}')

    align_dataset_z(**vars(args))   
