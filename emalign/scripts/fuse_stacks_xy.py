import argparse
import json
import logging
import os
import traceback
from datetime import datetime, UTC


def _limited_thread_env(num_workers):
    """Return environment settings that cap common CPU thread pools."""
    if num_workers is None or num_workers < 1:
        return {}

    worker_count = str(num_workers)
    return {
        'OMP_NUM_THREADS': worker_count,
        'OPENBLAS_NUM_THREADS': worker_count,
        'MKL_NUM_THREADS': worker_count,
        'NUMEXPR_NUM_THREADS': worker_count,
        'VECLIB_MAXIMUM_THREADS': worker_count,
        'XLA_FLAGS': (
            f'--xla_cpu_multi_thread_eigen=true '
            f'intra_op_parallelism_threads={worker_count}'
        ),
    }


def configure_thread_env(num_workers):
    """Set CPU thread limits before NumPy/JAX/SOFIMA initialize their runtimes."""
    for name, value in _limited_thread_env(num_workers).items():
        os.environ.setdefault(name, value)


def _preconfigure_thread_env_from_cli(argv=None):
    """Apply ``--cores``/``-c`` limits early enough for imported native libraries."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-c', '--cores', dest='num_workers', type=int)
    args, _ = parser.parse_known_args(argv)
    configure_thread_env(args.num_workers)


_preconfigure_thread_env_from_cli()

from emalign.align_xy.stitch_offgrid import stitch_images
from emalign.io.progress import get_mongo_client, get_mongo_db, log_progress, wipe_progress
from emalign.io.store import write_data, open_store
import tensorstore as ts

from glob import glob
from tqdm import tqdm

from emalign.align_xy.prep import create_configs_fused_stacks
from emalign.arrays.utils import compute_laplacian_var, compute_sobel_mean, compute_grad_mag, resample
from emalign.io.store import get_store_attributes, set_store_attributes
from emalign.io.process.mask import compute_greyscale_mask


# TODO: add a first slice test to make sure it is not missing images


def _json_default(obj):
    """Serialize NumPy/TensorStore values in diagnostic logs."""
    if hasattr(obj, 'item'):
        return obj.item()
    return str(obj)


def log_failed_fuse_image(log_path, record):
    """Append one failed stack/slice fusion attempt to a JSONL log."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'a') as f:
        f.write(json.dumps(record, default=_json_default) + '\n')


def build_failed_fuse_record(stack, z, global_slice_index, stage, error):
    """Build a compact diagnostic record for a stack image that could not be fused."""
    return {
        'timestamp': datetime.now(UTC).isoformat(),
        'stage': stage,
        'local_slice': z,
        'global_slice': global_slice_index,
        'dataset_path': stack.get('dataset_path'),
        'dataset_mask_path': stack.get('dataset_mask_path'),
        'source_zmin': stack.get('zmin'),
        'source_slice': z + stack.get('zmin', 0),
        'target_scale': stack.get('target_scale'),
        'error_type': type(error).__name__,
        'error': str(error),
        'traceback': ''.join(traceback.format_exception(type(error), error, error.__traceback__)),
    }


def summarize_failed_fuse_record(record):
    """Return a shorter failed-fusion record for progress metadata."""
    return {k: v for k, v in record.items() if k != 'traceback'}


def has_fuse_progress(db, collection_name, step_name, local_slice_index):
    """Return True when a fuse slice has any progress record."""
    collection = db[collection_name]
    return collection.count_documents({
        'step_name': step_name,
        'local_slice': local_slice_index,
    }) > 0


def has_completed_fuse_progress(db, collection_name, step_name, local_slice_index):
    """Return True only when a fuse slice has a completed progress record."""
    collection = db[collection_name]
    completed_filter = {
        'step_name': step_name,
        'local_slice': local_slice_index,
        '$or': [
            {'completed': True},
            {'failed_image_count': 0},
            {
                'completed': {'$exists': False},
                'failed_image_count': {'$exists': False},
            },
        ],
    }
    return collection.count_documents(completed_filter) > 0


def is_fuse_config(config):
    """Return True when a JSON object has the required fused-stack config fields."""
    required_fields = {'zmin', 'zmax', 'z_offsets', 'dataset_paths'}
    return isinstance(config, dict) and required_fields.issubset(config)


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
    config_filepaths = glob(os.path.join(output_dir, 'fuse_xy_*.json'))

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
                config = json.load(f)
            if is_fuse_config(config):
                overlapping_groups.append(config)
            else:
                logging.warning('Ignoring non-fuse-stack configuration file: %s', filepath)

    logging.info(f'Found {len(overlapping_groups)} segments of stacks.')
    return overlapping_groups


def fuse_stacks_group(config, 
                      project_name,
                      mongodb_config_filepath=None,
                      scale=0.1, 
                      patch_size=160, 
                      stride=40, 
                      img_on_top='auto', 
                      img_q_fun=None, 
                      destination_path=None,
                      target_res=None,
                      overwrite=False,
                      wipe_progress_flag=False,
                      retry_missing_slices=True,
                      num_workers=1,
                      max_canvas_scale=1.5):
    '''Fuse a group of stacks that overlap on the XY plane.

    Args:
        config (dict): Configuration dictionnary containing the paths to the stacks to align.
        project_name (str): Name of the project.
        mongodb_config_filepath (str, optional): Path to the MongoDB configuration file. Defaults to None.
        scale (float): Scale to downsample images to when determining offset using SIFT. Defaults to 0.1.
        patch_size (int, optional): Patch size used to compute the flow map using `sofima.flow_field.JAXMaskedXCorrWithStatsCalculator`. 
            Defaults to 160.
        stride (int, optional): Stride to compute flow map using `sofima.flow_field.JAXMaskedXCorrWithStatsCalculator`. 
            Defaults to 40.
        img_on_top (str, optional): What image should be on top. One of: auto, 1, 2. Defaults to 'auto'.
        img_q_fun (callable, optional): If img_on_top is set to auto, function taking image and mask as arguments, returns a value higher for higher quality/sharpness. 
            Defaults to None.
        overwrite (bool, optional): Whether to delete destination and start over. Defaults to False.
        wipe_progress_flag (bool): Whether to wipe progress for the stack. Defaults to False.
        retry_missing_slices (bool): Whether to retry slices with incomplete progress records.
            If False, any slice with an existing progress record is skipped. Defaults to True.
        num_workers (int, optional): Number of threads used to render the final image by `sofima.warp.ndimage_warp`. Defaults to 1.
        max_canvas_scale (float or None, optional): Maximum fused canvas shape as a multiple of the
            larger input image. Set to None to disable. Defaults to 1.5.
    '''


    if img_on_top == 'auto' and img_q_fun is None:
        raise ValueError('img_on_top set to auto. Please provide img_q_fun.')
    
    # Prepare destination name
    destination_name = '_'.join([os.path.basename(os.path.abspath(ds)) for ds in config['dataset_paths']])
    destination_name += '_fused'

    client = get_mongo_client(mongodb_config_filepath)
    db = get_mongo_db(client, project_name)

    if wipe_progress_flag:
        logging.info(f"Wiping progress for stack: {destination_name}")
        wipe_progress(db, destination_name)

    # Open datasets
    datasets = []
    for z_offset, ds_path in zip(config['z_offsets'], config['dataset_paths']):
        # Open dataset
        ds = open_store(ds_path, mode='r')
        
        # Limit to the overlapping range only
        zmin = config['zmin'] - z_offset
        zmax = config['zmax'] - z_offset
        ds = ds[zmin:zmax]

        # In case we need to resample
        if target_res is not None:
            s = get_store_attributes(ds)['resolution'][-1] / target_res
        else:
            s = 1
        
        # Open mask if exists
        ds_mask_path = os.path.abspath(ds_path) + '_mask'
        if os.path.exists(ds_mask_path):
            ds_mask = open_store(ds_mask_path, mode='r', dtype=ts.bool)
            ds_mask = ds_mask[zmin:zmax]
        else:
            ds_mask_path = None
            ds_mask = None
        datasets.append({
            'dataset': ds,
            'dataset_mask': ds_mask,
            'target_scale': s,
            'zmin': zmin,
            'dataset_path': os.path.abspath(ds_path),
            'dataset_mask_path': ds_mask_path,
        })

    # Create destination
    if overwrite:
        logging.warning('Existing dataset will be deleted and aligned from scratch.')

    z_shape = config['zmax'] - config['zmin']

    destination_basepath = os.path.dirname(os.path.abspath(config['dataset_paths'][0]))
    if destination_path is None:
        destination_path = os.path.join(destination_basepath, destination_name)
    else:
        destination_path = os.path.abspath(destination_path)
        destination_basepath = os.path.dirname(destination_path)
    destination_mask_path = os.path.join(destination_basepath, destination_name + '_mask')
    failed_alignment_log_path = os.path.join(destination_basepath, destination_name + '_failed_alignments.jsonl')

    if overwrite or not os.path.exists(destination_path):
        # Create destination from scratch
        destination = open_store(
            destination_path,
            mode='w',
            dtype=ts.uint8,
            shape=[z_shape, 1, 1],
            chunks=[1, 512, 512]
        )

        destination_mask = open_store(
            destination_mask_path,
            mode='w',
            dtype=ts.bool,
            shape=[z_shape, 1, 1],
            chunks=[1, 512, 512]
        )
    else:
        # Load existing destination
        destination = open_store(destination_path, mode='r+', dtype=ts.uint8)
        destination_mask = open_store(destination_mask_path, mode='r+', dtype=ts.bool)        
    
    # Start stitching
    k0 = 0.01
    k = 0.1
    gamma = 0.5 
    step_name = "fuse_xy"
    
    pbarz = tqdm(range(z_shape), position=1)
    for z in pbarz:
        global_slice_index = z + config['zmin']
        if not overwrite:
            if has_completed_fuse_progress(db, destination_name, step_name, z):
                pbarz.set_description(f'Skipping completed {z}...')
                continue
            if not retry_missing_slices and has_fuse_progress(db, destination_name, step_name, z):
                pbarz.set_description(f'Skipping previously attempted {z}...')
                continue
        pbarz.set_description(f'Fusing stacks...')
        canvas = None
        canvas_mask = None
        failed_images = []
        pbar_stacks = tqdm(datasets, position=2, leave=False)
        for stack in pbar_stacks:
            pbar_stacks.set_description(f'Slice {z} in progress...')
            dataset = stack['dataset']
            dataset_mask = stack['dataset_mask']
            target_scale = stack['target_scale']
            zmin = stack['zmin']

            try:
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
                img = resample(img, target_scale)
                mask = resample(mask, target_scale)
            except Exception as e:
                failed_record = build_failed_fuse_record(stack, z, global_slice_index, 'load_or_prepare', e)
                log_failed_fuse_image(failed_alignment_log_path, failed_record)
                failed_summary = summarize_failed_fuse_record(failed_record)
                failed_images.append(failed_summary)
                logging.exception(
                    'Skipping stack image that could not be loaded/prepared for fusion: %s',
                    failed_summary,
                )
                continue
            
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
                                                    max_canvas_scale=max_canvas_scale,
                                                    k0=k0,
                                                    k=k,
                                                    gamma=gamma)
            except Exception as e:
                failed_record = build_failed_fuse_record(stack, z, global_slice_index, 'stitch', e)
                log_failed_fuse_image(failed_alignment_log_path, failed_record)
                failed_summary = summarize_failed_fuse_record(failed_record)
                failed_images.append(failed_summary)
                logging.exception(
                    'Skipping stack image that could not be stitched into fused slice: %s',
                    failed_summary,
                )
                continue
            

        if canvas is not None:
            pbarz.set_description('Writing slice...')
            destination, _ = write_data(destination, canvas, z)
            destination_mask, _ = write_data(destination_mask, canvas_mask, z)

        # Log progress
        completed = len(failed_images) == 0
        metadata = {
            'mesh_parameters':{
                            'stride':stride,
                            'patch_size':patch_size,
                            'k0':k0,
                            'k':k,
                            'gamma':gamma
                            },
            'empty_slice': canvas is None,
            'completed': completed,
            'status': 'completed' if completed else 'incomplete',
            'failed_image_count': len(failed_images),
            'failed_images': failed_images,
            'failed_alignment_log_path': failed_alignment_log_path,
            'scale': scale,
            'img_on_top': img_on_top,
            'max_canvas_scale': max_canvas_scale
                }
        log_progress(db, destination_name, step_name, global_slice_index, z, metadata)

    # Destination takes the same attributes as the stacks we just processed
    attributes = get_store_attributes(datasets[0]['dataset'])
    attributes['resolution'][1] = attributes['resolution'][2] = target_res
    attributes['voxel_size'] = attributes['resolution']
    attributes['voxel_offset'][0] = config['zmin']
    attributes['offset'][0] = config['zmin'] * attributes['resolution'][0]
    attributes['z_aligned'] = False # This should not exist but let's be safe
    set_store_attributes(destination, attributes)
    set_store_attributes(destination_mask, attributes)
    

def align_fused_stacks_xy(config_path,
                          scale=0.1,
                          patch_size=160,
                          stride=40,
                          img_on_top='auto',
                          overwrite=False,
                          wipe_progress_stack=None,
                          retry_missing_slices=True,
                          num_workers=1,
                          max_canvas_scale=1.5):
    '''Align groups of overlapping stacks one after the other.

    Args:
        config_path (_type_): _description_
        scale (float, optional): _description_. Defaults to 0.1.
        patch_size (int, optional): _description_. Defaults to 160.
        stride (int, optional): _description_. Defaults to 40.
        img_on_top (str, optional): _description_. Defaults to 'auto'.
        overwrite (bool, optional): _description_. Defaults to False.
        wipe_progress_stack (str, optional): Name of the stack to wipe progress for. Defaults to None.
        retry_missing_slices (bool): Whether to retry slices with incomplete progress records. Defaults to True.
        num_workers (int, optional): _description_. Defaults to 1.
        max_canvas_scale (float or None, optional): Maximum fused canvas shape as a multiple of the larger input image.
    '''
    
    with open(config_path, 'r') as f:
        main_config = json.load(f)
    target_res = main_config['resolution'][-1]

    project_name = main_config.get('project_name')
    if not project_name:
        project_name = os.path.basename(main_config['output_path']).rstrip('.zarr')
    mongodb_config_filepath = main_config.get('mongodb_config_filepath')


    fused_configs = get_fused_configs(config_path,
                                      scale=scale)
    
    # Function to determine image quality to choose which one is on top
    # Highest value == on top
    # laplacian variance is sensitive to contrast and is thus weighted lower
    img_q_fun = lambda img, m: compute_laplacian_var(img, m)*0.5 + compute_sobel_mean(img, m) + compute_grad_mag(img, m)*100
    
    pbar = tqdm(fused_configs, position=0, leave=True)
    for config in pbar:
        pbar.set_description(f'z = {config['zmin']} - {config['zmax']}: Processing group of stacks...')
        destination_name = '_'.join([os.path.basename(os.path.abspath(ds)) for ds in config['dataset_paths']])
        destination_name += '_fused'
        wipe_this_stack = (destination_name == wipe_progress_stack)

        fuse_stacks_group(config,
                          project_name=project_name,
                          mongodb_config_filepath=mongodb_config_filepath,
                          scale=scale,
                          patch_size=patch_size, 
                          stride=stride, 
                          target_res=target_res,
                          img_on_top=img_on_top, 
                          img_q_fun=img_q_fun, 
                          overwrite=overwrite,
                          wipe_progress_flag=wipe_this_stack,
                          retry_missing_slices=retry_missing_slices,
                          num_workers=num_workers,
                          max_canvas_scale=max_canvas_scale)
    logging.info(f'All {len(fused_configs)} stacks were fused!')


def build_parser():
    parser = argparse.ArgumentParser('Script aligning tiles in XY based on SOFIMA (Scalable Optical Flow-based Image Montaging and Alignment). \n                                    This script was written to match the file structure produced by the ThermoFisher MAPs software.')
    parser.add_argument('-cfg', '--config',
                        metavar='CONFIG_PATH',
                        dest='config_path',
                        required=True,
                        type=str,
                        help='Path to the main task config.')
    parser.add_argument('-c', '--cores',
                        metavar='CORES',
                        dest='num_workers',
                        required=False,
                        default=1,
                        type=int,
                        help='Number of threads to use for rendering and imported native CPU thread pools. Default: 1')
    parser.add_argument('--scale',
                        dest='scale',
                        type=float,
                        default=0.1,
                        help='Downsampling scale used when estimating XY offsets with SIFT. Default: 0.1')
    parser.add_argument('--patch-size',
                        dest='patch_size',
                        type=int,
                        default=160,
                        help='Patch size used to compute the optical flow map. Default: 160')
    parser.add_argument('--stride',
                        dest='stride',
                        type=int,
                        default=40,
                        help='Stride used to compute the optical flow map. Default: 40')
    parser.add_argument('--img-on-top',
                        dest='img_on_top',
                        choices=['auto', '1', '2'],
                        default='auto',
                        help='Which image should be rendered on top during fusion. Choices: auto, 1, 2. Default: auto')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing dataset.')
    parser.add_argument('--wipe-progress',
                        dest='wipe_progress_stack',
                        type=str,
                        default=None,
                        help='Wipe progress for a specific stack before starting.')
    parser.add_argument('--no-retry-missing-slices',
                        dest='retry_missing_slices',
                        action='store_false',
                        help='Skip slices with any existing progress record, including incomplete slices. Default: retry incomplete slices.')
    parser.set_defaults(retry_missing_slices=True)
    parser.add_argument('--max-canvas-scale',
                        dest='max_canvas_scale',
                        type=float,
                        default=1.5,
                        help='Maximum fused XY canvas shape as a multiple of the larger input image. Default: 1.5')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    align_fused_stacks_xy(config_path=args.config_path,
                          scale=args.scale,
                          patch_size=args.patch_size,
                          stride=args.stride,
                          img_on_top=args.img_on_top,
                          num_workers=args.num_workers,
                          overwrite=args.overwrite,
                          wipe_progress_stack=args.wipe_progress_stack,
                          retry_missing_slices=args.retry_missing_slices,
                          max_canvas_scale=args.max_canvas_scale)


if __name__ == '__main__':
    main()
