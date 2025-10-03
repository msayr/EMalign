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
from emprocess.utils.io import get_dataset_attributes, set_dataset_attributes

from emalign.align_z.utils import compute_alignment_path, determine_initial_offset, get_ordered_datasets
from emalign.scripts.align_stack_z import align_stack_z

logging.basicConfig(level=logging.INFO)
logging.getLogger('absl').setLevel(logging.WARNING)
logging.getLogger('jax._src.xla_bridge').setLevel(logging.WARNING)


def align_dataset_z(config_paths,
                    config_z_path,
                    destination_path,
                    exclude,
                    num_workers, 
                    save_downsampled,
                    no_align,
                    start_over,
                    wipe_progress_stack=None):
    
    #---------- Open/prepare configs ----------#
    # Read common Z alignments parameters
    with open(config_z_path, 'r') as f:
        config_z = json.load(f)

    # If config_paths only has one item that is a dir, we assume it contains the config files
    if len(config_paths) == 1 and os.path.isdir(config_paths[0]):
        # Load dataset paths from z configs
        output_configs_dir = config_paths[0]
        config_paths = glob(os.path.join(output_configs_dir, 'z*.json'))

        with open(config_paths[0], 'r') as f:
            first_config = json.load(f)
            project_name = first_config.get('project_name')
            if not project_name:
                project_name = os.path.basename(first_config['destination_path']).rstrip('.zarr')
            mongodb_config_filepath = first_config.get('mongodb_config_filepath')

        if destination_path is None:
            destination_path = os.path.abspath(first_config['destination_path'])
        else:
            destination_path = os.path.join(os.path.abspath(destination_path), project_name)

        # Get list of datasets and offsets
        datasets = []
        z_offsets = []
        for p in config_paths:
            with open(p, 'r') as f:
                config = json.load(f)
            spec = {
                    'driver': 'zarr',
                    'kvstore': {
                        'driver': 'file',
                        'path': config['dataset_path']
                    }
                }
            datasets.append(ts.open(spec).result())
            z_offsets.append([config['z_offset']] + config.get('xy_offset', [0,0]))
        z_offsets = np.array(z_offsets)
        datasets = [datasets[i] for i in np.argsort(z_offsets[:, 0])]
        z_offsets = z_offsets[np.argsort(z_offsets[:, 0])]

        yx_target_resolution = float('inf')
        for config_path in config_paths:
            with open(config_path, 'r') as f:
                c = json.load(f)
                r = c['resolution'][0] if 'resolution' in c else c.get('yx_target_resolution', [float('inf')])[0]
                yx_target_resolution = min(yx_target_resolution, r)

        create_configs = False
    else:
        # Load dataset paths from main config files
        with open(config_paths[0], 'r') as f:
            config = json.load(f)
        project_name = config.get('project_name')
        if not project_name:
            project_name = os.path.basename(config['output_path']).rstrip('.zarr')
        mongodb_config_filepath = config.get('mongodb_config_filepath')

        if destination_path is None:
            destination_path = config['output_path']

        output_configs_dir = os.path.join(os.path.dirname(os.path.abspath(destination_path)), '03_config_z')
        destination_path = os.path.join(os.path.abspath(destination_path), project_name)
        create_configs = (not os.path.exists(output_configs_dir)) or start_over

        yx_target_resolution = config['resolution'][0] if 'resolution' in config else config['yx_target_resolution'][0]

        # Get list of datasets and offsets
        datasets, z_offsets = get_ordered_datasets(config_paths, exclude=['/flow', '_mask'] + exclude)

    #---------- Compute alignment path and initial offset ----------#
    # Print some info
    logging.info('Datasets Z offsets:')
    for dataset, z in zip(datasets, z_offsets):
        yx_res = get_dataset_attributes(dataset)['resolution'][1:]
        logging.info(f'    {z[0]} (res: {yx_res}): {dataset.kvstore.path.split("/")[-1]}')

    if isinstance(yx_target_resolution, list):
        yx_target_resolution = np.min(yx_target_resolution, axis=0).tolist()

    logging.info(f'Target resolution (yx): {yx_target_resolution}\n')
    
    if create_configs or start_over:
        # Create config files
        if start_over:
            create_configs = True
            try:
                input('WARNING: Progress will be wiped and all datasets will be processed.\nPress ENTER/ESC to resume or CTRL+C to abord\n')
            except KeyboardInterrupt:
                print('\nExiting...')
                sys.exit()
            
            for dataset in datasets:
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
    
        pad_offset = np.array([1000,1000]) # pad offsets to correct for any drift
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
        if no_align:
            logging.info('No alignment this time!')
            return
    else:
        # Read configuration
        with open(os.path.join(output_configs_dir, '00_align_plan.json'), 'r') as f:
            align_plan = json.load(f)

        root_stack = align_plan['root_stack']
        paths = align_plan['paths']
        reverse_order = align_plan['reverse_order']
        root_offset = align_plan['root_offset']
        logging.info(f'Using existing configuration files from:\n    {output_configs_dir}\n')
    
    #---------- Prepare destinations ----------#
    # Downsampled destination will serve for inspection
    ds_project_output_path = os.path.join(os.path.dirname(destination_path), f'{save_downsampled}x_' + project_name)
    if not os.path.exists(destination_path) or start_over:
        logging.info(f'Creating project dataset at: \n    {destination_path}\n')
        # Create container at destination if it doesn't exist or if user wants to start over
        # Shape destination starts as largest yx and last offset + shape of last dataset
        # yx could change shape based on warping but z should stay like this for this project
        shapes = np.array([dataset.shape for dataset in datasets])
        dest_shape = np.append(shapes[-1, 0] + z_offsets[-1, 0], 
                               shapes[:, 1:].max(0))
        destination = ts.open({'driver': 'zarr',
                               'kvstore': {
                                   'driver': 'file',
                                   'path': destination_path,
                                           },
                               'metadata':{
                                   'shape': dest_shape.tolist(),
                                   'chunks':[1,512,512]
                                           },
                               'transform': {'input_labels': ['z', 'y', 'x']}
                               },
                               dtype=ts.uint8, 
                               create=True,
                               delete_existing=True
                               ).result()
        destination_mask = ts.open({'driver': 'zarr',
                               'kvstore': {
                                   'driver': 'file',
                                   'path': destination_path + '_mask',
                                           },
                               'metadata':{
                                   'shape': dest_shape.tolist(),
                                   'chunks':[1,512,512]
                                           },
                               'transform': {'input_labels': ['z', 'y', 'x']}
                               },
                               dtype=ts.bool, 
                               create=True,
                               delete_existing=True
                               ).result()
        
        logging.info(f'Creating downsampled project dataset ({save_downsampled}) at: \n    {ds_project_output_path}\n')
        shapes = np.array([dataset.shape for dataset in datasets])//save_downsampled
        dest_shape = np.append(shapes[-1, 0] + z_offsets[-1, 0], 
                               shapes[:, 1:].max(0))
        ds_destination = ts.open({'driver': 'zarr',
                               'kvstore': {
                                   'driver': 'file',
                                   'path': ds_project_output_path,
                                           },
                               'metadata':{
                                   'shape': dest_shape.tolist(),
                                   'chunks':[1,512,512]
                                           },
                               'transform': {'input_labels': ['z', 'y', 'x']}
                               },
                               dtype=ts.uint8, 
                               create=True,
                               delete_existing=True
                               ).result()
    else:
        logging.info(f'Opening existing project dataset at: \n    {destination_path}\n')
        destination = ts.open({'driver': 'zarr',
                            'kvstore': {
                                    'driver': 'file',
                                    'path': destination_path,
                                        }
                            },
                            dtype=ts.uint8
                            ).result()
        destination_mask = ts.open({'driver': 'zarr',
                            'kvstore': {
                                    'driver': 'file',
                                    'path': destination_path + '_mask',
                                        }
                            },
                            dtype=ts.bool
                            ).result()
        
        logging.info(f'Opening existing downsampled project dataset ({save_downsampled}) at: \n    {ds_project_output_path}\n')
        ds_destination = ts.open({'driver': 'zarr',
                            'kvstore': {
                                    'driver': 'file',
                                    'path': ds_project_output_path,
                                        }
                            },
                            dtype=ts.uint8
                            ).result()
        
    #---------- Start alignment ----------#
    for i, (path, order) in enumerate(zip(paths, reverse_order)):
        for dataset_name in path:
            config_path = os.path.join(output_configs_dir, f'z_{dataset_name}.json')
            if dataset_name == path[0] and i == 0:
                assert dataset_name == root_stack, f'First dataset ({dataset_name}) of the path is not the root stack ({root_stack})'

            # Load configuration
            with open(config_path, 'r') as f:
                config = json.load(f)
            config['num_workers'] = num_workers
            config['wipe_progress_flag'] = (dataset_name == wipe_progress_stack)

            # Start alignment
            try:
                params = signature(align_stack_z).parameters
                relevant_args = {k: v for k, v in config.items() if k in params}
                align_stack_z(**relevant_args)
            except Exception as e:
                raise RuntimeError(f'Error with {dataset_name}: ' + str(e))
            
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
                        default=1,
                        help='Number of threads to use for processing. Default: 0 (all cores available)')
    parser.add_argument('-ds', '--downsample-scale',
                        metavar='SCALE',
                        dest='save_downsampled',
                        type=float,
                        default=10,
                        help='Factor to use for downsampling the dataset that will be saved for inspection. Default: 10')
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
    except Exception:
        print('To select GPUs, specify it before running python, e.g.: CUDA_VISIBLE_DEVICES=0,1 python script.py')
        sys.exit()
    print(f'Available GPU IDs: {GPU_ids}')

    align_dataset_z(**vars(args))   
