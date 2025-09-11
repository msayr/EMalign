import argparse
import json
import logging
import os
from emalign.align_xy.stitch_offgrid import stitch_images
from emalign.io.mongo import check_progress
from emalign.io.store import write_slice
from pymongo import MongoClient
import tensorstore as ts

from glob import glob
from tqdm import tqdm
from emprocess.utils.io import get_dataset_attributes, set_dataset_attributes
from emprocess.utils.mask import compute_greyscale_mask

from emalign.align_xy.prep import create_configs_fused_stacks
from emalign.arrays.utils import _compute_laplacian_var, _compute_sobel_mean, _compute_grad_mag, downsample


# TODO: add a first slice test to make sure it is not missing images


def get_fused_configs(
        main_config_path,
        scale=0.1
        ):
    '''Gather or compute configuration files for groups of stacks to fuse.

    Args:
        config_path (str): Absolute path to the main_config.json file for this project.
        scale (float, optional): Scale to downsample images for determining offset using SIFT. Defaults to 0.1.

    Returns:
        fused_configs (list of `dict`): list of configuration file per segment of stacks to fuse.
    '''

    # Output directory for the config files
    output_dir = os.path.dirname(os.path.abspath(main_config_path))

    # Check for existing files
    config_filepaths = glob(os.path.join(output_dir, 'fuse_xy*.json'))

    if len(config_filepaths) == 0:
        # Compute and write configuration files
        overlapping_groups = create_configs_fused_stacks(main_config_path, scale=scale)

        pbar = tqdm(overlapping_groups, position=0, desc='Looking for overlapping stacks')
        for i, config in enumerate(pbar):
            filepath = os.path.join(output_dir, f'fuse_xy_{config['zmin']}_{config['zmax']}_{i}.json')
            if not os.path.exists(filepath):
                with open(filepath, 'w') as f:
                    json.dump(config, f, indent='')
    else:
        # Load configuration files
        overlapping_groups = []
        pbar = tqdm(config_filepaths, position=0, desc='Loading existing configurations')
        for filepath in pbar:
            with open(filepath, 'r') as f:
                overlapping_groups.append(json.load(f))

    logging.info(f'Found {len(overlapping_groups)} segments of stacks.')
    return overlapping_groups


def fuse_stacks_group(config, 
                      db_name,
                      scale=0.1, 
                      patch_size=160, 
                      stride=40, 
                      img_on_top='auto', 
                      img_q_fun=None, 
                      destination_path=None,
                      target_res=None,
                      overwrite=False,
                      num_workers=1):
    '''Fuse a group of stacks that overlap on the XY plane.

    Args:
        config (dict): Configuration dictionnary containing the paths to the stacks to align.
        db_name (str): Name of the MongoDB database to write progress documents to.
        scale (float): Scale to downsample images to when determining offset using SIFT. Defaults to 0.1.
        patch_size (int, optional): Patch size used to compute the flow map using `sofima.flow_field.JAXMaskedXCorrWithStatsCalculator`. 
            Defaults to 160.
        stride (int, optional): Stride to compute flow map using `sofima.flow_field.JAXMaskedXCorrWithStatsCalculator`. 
            Defaults to 40.
        img_on_top (str, optional): What image should be on top. One of: auto, 1, 2. Defaults to 'auto'.
        img_q_fun (callable, optional): If img_on_top is set to auto, function taking image and mask as arguments, returns a value higher for higher quality/sharpness. 
            Defaults to None.
        overwrite (bool, optional): Whether to delete destination and start over. Defaults to False.
        num_workers (int, optional): Number of threads used to render the final image by `sofima.warp.ndimage_warp`. Defaults to 1.
    '''


    if img_on_top == 'auto' and img_q_fun is None:
        raise ValueError('img_on_top set to auto. Please provide img_q_fun.')
    
    # Open datasets
    datasets = []
    for z_offset, ds_path in zip(config['z_offsets'], config['dataset_paths']):
        # Open dataset
        ds = ts.open({'driver': 'zarr',
                        'kvstore': {
                                'driver': 'file',
                                'path': ds_path,
                                    }},
                            read=True).result()
        
        # Limit to the overlapping range only
        zmin = config['zmin'] - z_offset
        zmax = config['zmax'] - z_offset
        ds = ds[zmin:zmax]

        # In case we need to resample
        if target_res is not None:
            s = get_dataset_attributes(ds)['resolution'][-1] / target_res
        else:
            s = 1
        
        # Open mask if exists
        ds_mask_path = os.path.abspath(ds_path) + '_mask'
        if os.path.exists(ds_mask_path):
            ds_mask = ts.open({'driver': 'zarr',
                            'kvstore': {
                            'driver': 'file',
                            'path': ds_mask_path,
                            }},
                            read=True).result()
            ds_mask = ds_mask[zmin:zmax]
        else:
            ds_mask = None
        datasets.append({'dataset': ds, 'dataset_mask': ds_mask, 'target_scale': s, 'zmin': zmin})

    # Create destination
    if overwrite:
        logging.warning('Existing dataset will be deleted and aligned from scratch.')

    # Prepare destination    
    destination_name = '_'.join([os.path.basename(os.path.abspath(ds)) for ds in config['dataset_paths']])
    destination_name += '_fused'
    z_shape = config['zmax'] - config['zmin']

    if destination_path is None:
        destination_basepath = os.path.dirname(os.path.abspath(config['dataset_paths'][0]))
        destination_path = os.path.join(destination_basepath, destination_name)
    destination_mask_path = os.path.join(destination_basepath, destination_name + '_mask')

    if overwrite or not os.path.exists(destination_path):
        # Create destination from scratch
        destination = ts.open({'driver': 'zarr',
                            'kvstore': {
                                'driver': 'file',
                                'path': destination_path,
                                        },
                            'metadata':{
                                'shape': [z_shape, 1, 1],
                                'chunks':[1,512,512]
                                        },
                            'transform': {'input_labels': ['z', 'y', 'x']}
                            },
                            dtype=ts.uint8, 
                            create=True,
                            delete_existing=True).result() 
          
        destination_mask = ts.open({'driver': 'zarr',
                            'kvstore': {
                                'driver': 'file',
                                'path': destination_mask_path,
                                        },
                            'metadata':{
                                'shape': [z_shape, 1, 1],
                                'chunks':[1,512,512]
                                        },
                            'transform': {'input_labels': ['z', 'y', 'x']}
                            },
                            dtype=ts.bool,
                            create=True,
                            delete_existing=True).result()   
    else:
        # Load existing destination
        destination = ts.open({'driver': 'zarr',
                            'kvstore': {
                                'driver': 'file',
                                'path': destination_path,
                                        },
                            },
                            dtype=ts.uint8).result()  
        
        destination_mask = ts.open({'driver': 'zarr',
                            'kvstore': {
                                'driver': 'file',
                                'path': destination_mask_path,
                                        },
                            },
                            dtype=ts.bool).result()        
    
    # Track progress
    db_host=None
    collection_name ='FUSE_XY_' + destination_name
    client = MongoClient(db_host)
    db = client[db_name]
    collection_progress = db[collection_name]

    # Start stitching
    k0 = 0.01
    k = 0.1
    gamma = 0.5 
    
    pbarz = tqdm(range(z_shape), position=1)
    for z in pbarz:
        if check_progress({'stack_name': destination_name, 'z': z}, db_host, db_name, collection_name) and not overwrite:
            pbarz.set_description(f'Skipping {z}...')
            continue
        pbarz.set_description(f'Fusing stacks...')
        canvas = None
        canvas_mask = None
        pbar_stacks = tqdm(datasets, position=2, leave=False)
        for stack in pbar_stacks:
            pbar_stacks.set_description(f'Slice {z} in progress...')
            dataset, dataset_mask, target_scale, zmin = stack.values()

            # Load image
            img = dataset[z + zmin].read().result()
            if not img.any():
                continue

            # Load or compute mask
            if dataset_mask is None:
                mask = compute_greyscale_mask(img)
            else:
                mask = dataset_mask[z + zmin].read().result()

            # Resample to the correct resolution
            img = downsample(img, target_scale)
            mask = downsample(mask, target_scale)
            
            if canvas is None:
                # First image
                canvas = img.copy()
                canvas_mask = mask.copy()
                continue
            
            # Stitch images to canvas
            try:
                canvas, canvas_mask = stitch_images(canvas, 
                                                    img,
                                                    mask1=canvas_mask, 
                                                    mask2=mask,
                                                    scale=scale,
                                                    patch_size=patch_size,
                                                    stride=stride,
                                                    parallelism=num_workers,
                                                    img_on_top=img_on_top,
                                                    img_q_fun=img_q_fun,
                                                    k0=k0,
                                                    k=k,
                                                    gamma=gamma)
            except Exception as e:
                # TODO: fix this. Error gets messy because of tqdm bars
                print()
                print()
                print()
                print(f'Error in stack (z = {z}): {stack}')
                raise(e)
            
            if pbar_stacks.n == pbar_stacks.total-1:
                pbar_stacks.set_description('Writing slice...')
                destination, _ = write_slice(destination, canvas, z)
                destination_mask, _ = write_slice(destination_mask, canvas_mask, z)

        # Log progress
        doc = {
            'stack_name': destination_name,
            'z': z,
            'mesh_parameters':{
                            'stride':stride,
                            'patch_size':patch_size,
                            'k0':k0,
                            'k':k,
                            'gamma':gamma
                            },
            'empty_slice': canvas is None,
            'scale': scale,
            'img_on_top': img_on_top
                }
        collection_progress.insert_one(doc)

    # Destination takes the same attributes as the stacks we just processed
    attributes = get_dataset_attributes(datasets[0]['dataset'])
    attributes['resolution'][1] = attributes['resolution'][2] = target_res
    attributes['voxel_size'] = attributes['resolution']
    attributes['voxel_offset'][0] = config['zmin']
    attributes['offset'][0] = config['zmin'] * attributes['resolution'][0]
    attributes['z_aligned'] = False # This should not exist but let's be safe
    set_dataset_attributes(destination, attributes)
    set_dataset_attributes(destination_mask, attributes)
    

def align_fused_stacks_xy(config_path,
                          scale=0.1,
                          patch_size=160,
                          stride=40,
                          img_on_top='auto',
                          overwrite=False,
                          num_workers=1):
    '''Align groups of overlapping stacks one after the other.

    Args:
        config_path (_type_): _description_
        scale (float, optional): _description_. Defaults to 0.1.
        patch_size (int, optional): _description_. Defaults to 160.
        stride (int, optional): _description_. Defaults to 40.
        img_on_top (str, optional): _description_. Defaults to 'auto'.
        overwrite (bool, optional): _description_. Defaults to False.
        num_workers (int, optional): _description_. Defaults to 1.
    '''
    
    with open(config_path, 'r') as f:
        main_config = json.load(f)
    target_res = main_config['resolution'][-1]

    project = os.path.basename(main_config['output_path']).rstrip('.zarr')
    db_name=f'alignment_progress_{project}'

    fused_configs = get_fused_configs(config_path,
                                      0.1)
    
    # Function to determine image quality to choose which one is on top
    # Highest value == on top
    # laplacian variance is sensitive to contrast and is thus weighted lower
    img_q_fun = lambda img, m: _compute_laplacian_var(img, m)*0.5 + _compute_sobel_mean(img, m) + _compute_grad_mag(img, m)*100
    
    pbar = tqdm(fused_configs, position=0, leave=True)
    for config in pbar:
        pbar.set_description(f'z = {config['zmin']} - {config['zmax']}: Processing group of stacks...')
        fuse_stacks_group(config, 
                          db_name=db_name,
                          scale=scale,
                          patch_size=patch_size, 
                          stride=stride, 
                          target_res=target_res,
                          img_on_top=img_on_top, 
                          img_q_fun=img_q_fun, 
                          overwrite=overwrite,
                          num_workers=num_workers)
    logging.info(f'All {len(fused_configs)} stacks were fused!')


if __name__ == '__main__':


    parser=argparse.ArgumentParser('Script aligning tiles in XY based on SOFIMA (Scalable Optical Flow-based Image Montaging and Alignment). \n\
                                    This script was written to match the file structure produced by the ThermoFisher MAPs software.')
    parser.add_argument('-cfg', '--config',
                        metavar='CONFIG_PATH',
                        dest='config_path',
                        required=True,
                        type=str,
                        help='Path to the main task config.')
    parser.add_argument('-c', '--cores',
                        metavar='CORES',
                        dest='num_workers',
                        required=True,
                        type=int,
                        help='Number of threads to use for rendering. Default: 1')
    args=parser.parse_args()

    align_fused_stacks_xy(**vars(args))