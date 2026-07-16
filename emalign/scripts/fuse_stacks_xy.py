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


def _extract_cores_arg(argv):
    """Return an exact ``-c``/``--cores`` value without misreading ``-cfg``."""
    if argv is None:
        argv = os.sys.argv[1:]

    for i, arg in enumerate(argv):
        if arg == '-c' or arg == '--cores':
            if i + 1 < len(argv):
                return argv[i + 1]
            return None
        if arg.startswith('--cores='):
            return arg.split('=', 1)[1]
    return None


def _preconfigure_thread_env_from_cli(argv=None):
    """Apply exact ``--cores``/``-c`` limits early enough for imported native libraries."""
    cores = _extract_cores_arg(argv)
    if cores is None:
        return
    try:
        configure_thread_env(int(cores))
    except ValueError:
        return


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
import numpy as np
import importlib.util



def _load_manual_reference_images(config, target_res, gui_downsample=0.1):
    """Load one display-resolution reference image per dataset in a fuse config."""
    images = []
    for z_offset, ds_path in zip(config['z_offsets'], config['dataset_paths']):
        ds = open_store(ds_path, mode='r')
        source_z = config['zmin'] - z_offset
        img = ds[source_z].read().result()
        attrs = get_store_attributes(ds)
        scale = attrs['resolution'][-1] / target_res * gui_downsample
        images.append(resample(img, scale))
    return images


def _normalise_manual_offsets(offsets, scale=1):
    """Move offsets to origin, convert from display pixels, and return plain ints."""
    arr = np.asarray(offsets, dtype=float)
    arr = arr - arr.min(axis=0)
    arr = np.rint(arr / scale).astype(int)
    return [[int(y), int(x)] for y, x in arr]


def _prompt_manual_offsets_cli(config, initial_offsets):
    """Allow manual offset entry when an interactive matplotlib backend is unavailable."""
    offsets = [list(map(int, offset)) for offset in initial_offsets]
    print('Manual XY guide offsets are shown as [y, x] pixels at the fused output resolution.')
    for i, (path, offset) in enumerate(zip(config['dataset_paths'], offsets)):
        answer = input(f'{i}: {os.path.basename(path)} offset {offset}; press Enter to keep or enter y,x: ').strip()
        if answer:
            y, x = answer.replace(',', ' ').split()[:2]
            offsets[i] = [int(float(y)), int(float(x))]
    return _normalise_manual_offsets(offsets)


def _set_manual_xy_view(ax, offsets, images, margin_fraction=0.2):
    """Zoom the manual XY axes out enough to show all draggable stack images."""
    offsets = np.asarray(offsets, dtype=float)
    min_yx = offsets.min(axis=0)
    max_yx = np.max([
        offsets[i] + np.array(images[i].shape[:2])
        for i in range(len(images))
    ], axis=0)
    span_yx = np.maximum(max_yx - min_yx, 1)
    margin_yx = np.maximum(span_yx * margin_fraction, 25)
    ax.set_xlim(min_yx[1] - margin_yx[1], max_yx[1] + margin_yx[1])
    ax.set_ylim(max_yx[0] + margin_yx[0], min_yx[0] - margin_yx[0])


def confirm_manual_xy_offsets(config, target_res, gui_downsample=0.1):
    """Open a draggable matplotlib UI to confirm rough stack XY positions."""
    full_res_offsets = [[0, 0] for _ in config['dataset_paths']]
    if importlib.util.find_spec('matplotlib') is None:
        return _prompt_manual_offsets_cli(config, full_res_offsets)

    images = _load_manual_reference_images(config, target_res, gui_downsample=gui_downsample)
    offsets = [[0, 0] for _ in images]

    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.widgets import Button, RadioButtons

    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(left=0.24, bottom=0.22, right=0.98, top=0.9)
    artists = []
    selected = {'index': 0, 'press': None}
    colors = ['red', 'cyan', 'yellow', 'lime', 'magenta', 'orange']
    for i, (img, path) in enumerate(zip(images, config['dataset_paths'])):
        artist = ax.imshow(
            img,
            alpha=0.45,
            cmap='gray',
            origin='upper',
            extent=(0, img.shape[1], img.shape[0], 0),
        )
        artists.append(artist)
        ax.text(5, 20 + i * 25, str(i), color=colors[i % len(colors)], fontsize=14, weight='bold')
    ax.set_title('Manual XY guide: select a stack, then left-drag it into rough alignment')
    legend_handles = [
        mpatches.Patch(color=colors[i % len(colors)], label=f'{i}: {os.path.basename(path)}')
        for i, path in enumerate(config['dataset_paths'])
    ]
    ax.legend(handles=legend_handles, loc='upper right', title='Stack index')
    instructions = (
        '1) Select stack index at left.\n'
        '2) Left-click and drag inside the image panel to move that stack.\n'
        '3) Use Zoom out/Fit all if stacks move off-screen.\n'
        'Toolbar pan/zoom still works for navigating the canvas.'
    )
    fig.text(0.24, 0.05, instructions, va='bottom')
    _set_manual_xy_view(ax, offsets, images)

    radio_ax = fig.add_axes([0.02, 0.35, 0.18, 0.5])
    radio = RadioButtons(radio_ax, [str(i) for i in range(len(images))], active=0)
    radio_ax.set_title('Move stack')

    def select_stack(label):
        selected['index'] = int(label)

    radio.on_clicked(select_stack)

    def on_press(event):
        if event.inaxes != ax or event.button != 1 or event.xdata is None or event.ydata is None:
            return
        idx = selected['index']
        selected['press'] = (event.xdata, event.ydata, offsets[idx][:])

    def on_motion(event):
        if selected['press'] is None or event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        x0, y0, old = selected['press']
        idx = selected['index']
        offsets[idx] = [int(round(old[0] + event.ydata - y0)), int(round(old[1] + event.xdata - x0))]
        artists[idx].set_extent((
            offsets[idx][1],
            offsets[idx][1] + images[idx].shape[1],
            offsets[idx][0] + images[idx].shape[0],
            offsets[idx][0],
        ))
        fig.canvas.draw_idle()

    def on_release(event):
        selected['press'] = None

    accepted = {'value': False}

    def accept(event):
        accepted['value'] = True
        plt.close(fig)

    def fit_all(event):
        _set_manual_xy_view(ax, offsets, images)
        fig.canvas.draw_idle()

    def zoom_out(event):
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        half_w = abs(x1 - x0)
        half_h = abs(y1 - y0)
        ax.set_xlim(cx - half_w, cx + half_w)
        ax.set_ylim(cy + half_h, cy - half_h)
        fig.canvas.draw_idle()

    button = Button(fig.add_axes([0.84, 0.06, 0.12, 0.07]), 'Accept')
    button.on_clicked(accept)
    fit_button = Button(fig.add_axes([0.70, 0.06, 0.12, 0.07]), 'Fit all')
    fit_button.on_clicked(fit_all)
    zoom_button = Button(fig.add_axes([0.56, 0.06, 0.12, 0.07]), 'Zoom out')
    zoom_button.on_clicked(zoom_out)
    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)
    plt.show()
    if not accepted['value']:
        return _prompt_manual_offsets_cli(config, full_res_offsets)
    return _normalise_manual_offsets(offsets, scale=gui_downsample)


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
                      max_canvas_scale=1.5,
                      manual_xy_offsets=None):
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
        manual_xy_offsets (list[list[int]] or None): Optional rough [y, x] positions for each
            dataset, in the same order as config['dataset_paths'].
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
            'manual_xy_offset': (
                None
                if manual_xy_offsets is None
                else np.asarray(manual_xy_offsets[len(datasets)], dtype=int)
            ),
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
        canvas_origin = None
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
                canvas_origin = stack['manual_xy_offset'] if stack['manual_xy_offset'] is not None else np.array([0, 0])
                continue
            
            # Stitch images to canvas
            try:
                initial_offset = None
                if stack['manual_xy_offset'] is not None:
                    initial_offset = stack['manual_xy_offset'] - canvas_origin
                canvas, canvas_mask = stitch_images(canvas, 
                                                    img,
                                                    mask1=canvas_mask, 
                                                    mask2=mask,
                                                    scale=scale,
                                                    initial_offset=initial_offset,
                                                    use_initial_offset_only=initial_offset is not None,
                                                    patch_size=patch_size,
                                                    stride=stride,
                                                    parallelism=num_workers,
                                                    img_on_top=img_on_top,
                                                    img_q_fun=img_q_fun,
                                                    max_canvas_scale=max_canvas_scale,
                                                    k0=k0,
                                                    k=k,
                                                    gamma=gamma)
                if initial_offset is not None:
                    all_offsets = np.stack([canvas_origin, stack['manual_xy_offset']])
                    canvas_origin = all_offsets.min(axis=0)
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
                          max_canvas_scale=1.5,
                          manual_xy=False):
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
        manual_xy (bool): Prompt for draggable manual rough XY offsets before fusing each group.
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
        manual_xy_offsets = None
        if manual_xy:
            manual_xy_offsets = confirm_manual_xy_offsets(config, target_res)
            config['manual_xy_offsets'] = manual_xy_offsets
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
                          max_canvas_scale=max_canvas_scale,
                          manual_xy_offsets=manual_xy_offsets)
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
    parser.add_argument('--manual-xy',
                        dest='manual_xy',
                        action='store_true',
                        help='Open a draggable GUI before each fuse group so users can confirm or correct rough relative XY stack positions.')
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
                          max_canvas_scale=args.max_canvas_scale,
                          manual_xy=args.manual_xy)


if __name__ == '__main__':
    main()
