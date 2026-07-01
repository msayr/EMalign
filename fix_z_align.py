#!/usr/bin/env python3
"""Repair a bad Z-alignment boundary without rebuilding the whole stack.

This utility should usually be pointed at ``00_align_plan.json`` and a
destination/global slice number from the final aligned dataset. It selects the
matching stack config, re-aligns a small preview range, opens a Neuroglancer
inspection view, and, on CLI confirmation, propagates the repaired alignment
from the first bad slice to the end of that stack by rerunning Z alignment for
the suffix only.
"""

import argparse
import copy
import json
import inspect
import logging
import os
from glob import glob
from typing import Optional, Tuple



logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def _parse_slice_range(value: str) -> Tuple[int, int]:
    """Parse ``N`` or ``N:M`` / ``N-M`` into an inclusive global slice range."""
    value = value.strip()
    sep = ':' if ':' in value else '-' if '-' in value else None
    if sep is None:
        start = end = int(value)
    else:
        start_s, end_s = value.split(sep, maxsplit=1)
        start, end = int(start_s), int(end_s)
    if start < 0 or end < start:
        raise argparse.ArgumentTypeError('Slice range must be N or START:END with 0 <= START <= END.')
    return start, end


def _stack_bounds(config: dict) -> Tuple[int, int]:
    """Return the local [min, max) bounds covered by the original stack config."""
    if config.get('local_z_min') is not None and config.get('local_z_max') is not None:
        return int(config['local_z_min']), int(config['local_z_max'])

    from emalign.io.store import open_store

    dataset = open_store(os.path.abspath(config['dataset_path']), mode='r')
    default_min = int(dataset.domain.inclusive_min[0])
    default_max = int(dataset.domain.exclusive_max[0])
    return default_min, default_max


def _global_z_for_local(config: dict, original_local_min: int, local_z: int) -> int:
    """Map an input stack-local z index to the destination/global z index."""
    return int(config.get('z_offset', 0)) + int(local_z) - int(original_local_min)


def _global_bounds(config: dict, original_local_min: int, original_local_max: int) -> Tuple[int, int]:
    """Return destination/global [min, max) bounds for a stack config."""
    return (
        _global_z_for_local(config, original_local_min, original_local_min),
        _global_z_for_local(config, original_local_min, original_local_max),
    )


def _global_to_local(config: dict, original_local_min: int, global_z: int) -> int:
    """Map a destination/global z index to an input stack-local z index."""
    return int(original_local_min) + int(global_z) - int(config.get('z_offset', 0))


def _load_json(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def _load_dataset_configs(config_dir: str) -> dict:
    configs = {}
    for path in glob(os.path.join(config_dir, 'z_*.json')):
        cfg = _load_json(path)
        if 'dataset_name' in cfg:
            cfg['_config_path'] = path
            configs[cfg['dataset_name']] = cfg
    if not configs:
        raise FileNotFoundError(f'No z_*.json stack configs found in {config_dir}.')
    return configs


def _is_align_plan(config: dict) -> bool:
    return 'paths' in config and 'dataset_local_bounds' in config and 'dataset_path' not in config


def _config_for_global_range(config_path: str, slice_range: Tuple[int, int]) -> dict:
    """Load the stack config containing an inclusive global slice range."""
    config = _load_json(config_path)
    if not _is_align_plan(config):
        return config

    config_dir = os.path.dirname(os.path.abspath(config_path))
    dataset_configs = _load_dataset_configs(config_dir)
    start, end = slice_range
    matches = []
    for dataset_name, dataset_config in dataset_configs.items():
        local_min, local_max = _stack_bounds(dataset_config)
        global_min, global_max = _global_bounds(dataset_config, local_min, local_max)
        if global_min <= start <= end < global_max:
            matches.append((global_min, dataset_name, dataset_config))

    if not matches:
        ranges = []
        for dataset_name, dataset_config in sorted(dataset_configs.items()):
            local_min, local_max = _stack_bounds(dataset_config)
            global_min, global_max = _global_bounds(dataset_config, local_min, local_max)
            ranges.append(f'{dataset_name}=[{global_min}, {global_max})')
        raise ValueError(
            f'Global repair range {start}:{end} is not fully contained in one stack from {config_path}. '
            f'Available global ranges: {"; ".join(ranges)}'
        )

    matches.sort()
    selected = matches[0][2]
    LOGGER.info('Selected stack config %s from align plan for global range %d:%d.', selected.get('_config_path', selected['dataset_name']), start, end)
    return selected


def _normalize_slice_range(config: dict, original_local_min: int, original_local_max: int, slice_range: Tuple[int, int]) -> Tuple[int, int]:
    """Return a local inclusive range from an inclusive destination/global z range."""
    start, end = slice_range
    global_min, global_max_exclusive = _global_bounds(config, original_local_min, original_local_max)
    if global_min <= start <= end < global_max_exclusive:
        return (
            _global_to_local(config, original_local_min, start),
            _global_to_local(config, original_local_min, end),
        )

    raise ValueError(
        f'Global repair range {start}:{end} is outside destination/global bounds '
        f'[{global_min}, {global_max_exclusive}) for stack {config.get("dataset_name", "<unknown>")}. '
        'Pass the global slice number from the final aligned dataset, or pass 00_align_plan.json '
        'so the script can select the correct stack config automatically.'
    )


def _delete_progress_suffix(db, dataset_name: str, start_local: int, start_global: int, *, include_mesh: bool) -> None:
    """Remove cached progress docs that would otherwise cause suffix repair to be skipped."""
    collection = db[dataset_name]
    result = collection.delete_many({
        '$or': [
            {'step_name': 'flow_z', 'local_slice': {'$gte': int(start_local)}},
            {'step_name': 'render_z', 'global_slice': {'$gte': int(start_global)}},
        ]
    })
    LOGGER.info('Deleted %d cached flow/render MongoDB progress documents.', result.deleted_count)
    if include_mesh:
        mesh_result = collection.delete_many({'step_name': 'mesh_relax_z'})
        LOGGER.info('Deleted %d cached mesh MongoDB progress documents.', mesh_result.deleted_count)


def _make_repair_config(base_config: dict, original_local_min: int, local_start: int, local_stop_exclusive: int) -> dict:
    """Build an align_stack_z config for a suffix/subrange starting at local_start."""
    config = copy.deepcopy(base_config)
    config.pop('_config_path', None)
    config['local_z_min'] = int(local_start)
    config['local_z_max'] = int(local_stop_exclusive)
    config['z_offset'] = _global_z_for_local(base_config, original_local_min, local_start)
    config['first_slice'] = config['z_offset'] - 1
    config['overwrite'] = True
    config['wipe_progress_flag'] = False
    return config


def _call_align_stack_z(align_stack_z, config: dict) -> None:
    """Call align_stack_z with only the keyword arguments accepted by this checkout."""
    params = inspect.signature(align_stack_z).parameters
    relevant_args = {k: v for k, v in config.items() if k in params}
    ignored = sorted(set(config) - set(relevant_args))
    if ignored:
        LOGGER.debug('Ignoring config keys not accepted by align_stack_z: %s', ', '.join(ignored))
    align_stack_z(**relevant_args)


def _confirm(prompt: str) -> bool:
    answer = input(f'{prompt} [y/N]: ').strip().lower()
    return answer in {'y', 'yes'}


def repair_z_alignment(config_path: str,
                       slice_range: Tuple[int, int],
                       preview_after: int,
                       inspect_port: int,
                       skip_preview: bool = False,
                       project_name: Optional[str] = None) -> None:
    from emalign.inspect_dataset import inspect_dataset
    from emalign.io.progress import get_mongo_client, get_mongo_db
    from emalign.scripts.align_stack_z import align_stack_z

    base_config = _config_for_global_range(config_path, slice_range)

    if project_name is not None:
        base_config['project_name'] = project_name
    if not base_config.get('project_name'):
        base_config['project_name'] = os.path.basename(base_config['destination_path']).rstrip('.zarr')

    dataset_name = base_config['dataset_name']
    original_min, original_max = _stack_bounds(base_config)
    repair_start, repair_end = _normalize_slice_range(base_config, original_min, original_max, slice_range)
    start_global = _global_z_for_local(base_config, original_min, repair_start)
    LOGGER.info(
        'Interpreting requested global slice range %d:%d -> local z %d:%d in %s.',
        slice_range[0],
        slice_range[1],
        repair_start,
        repair_end,
        base_config['dataset_name'],
    )
    if repair_start == original_min and base_config.get('first_slice') is None and base_config.get('reference_path') is None and start_global <= 0:
        raise ValueError('Cannot repair the first slice of a root stack because there is no previous aligned destination slice to anchor to.')

    preview_stop = min(original_max, max(repair_end + 1, repair_start + 1) + max(0, preview_after))

    client = get_mongo_client(base_config.get('mongodb_config_filepath'))
    db = get_mongo_db(client, base_config['project_name'])

    if not skip_preview:
        LOGGER.info('Realigning preview range local z [%d, %d) for %s.', repair_start, preview_stop, dataset_name)
        _delete_progress_suffix(db, dataset_name, repair_start, start_global, include_mesh=True)
        preview_config = _make_repair_config(base_config, original_min, repair_start, preview_stop)
        _call_align_stack_z(align_stack_z, preview_config)

        inspect_min = max(0, start_global - 2)
        inspect_max = _global_z_for_local(base_config, original_min, preview_stop - 1) + 3
        LOGGER.info('Opening Neuroglancer for destination slices [%d, %d).', inspect_min, inspect_max)
        inspect_dataset(base_config['destination_path'],
                        bounding_box=[inspect_min, inspect_max],
                        keep_missing=True,
                        bind_port=inspect_port)

        if not _confirm('Does the previewed Z alignment look correct and should the repair be propagated to the end of the stack?'):
            LOGGER.warning('Repair was not propagated. Preview slices already written to the destination remain in place.')
            return

    LOGGER.info('Propagating repaired alignment from local z %d through %d for %s.', repair_start, original_max, dataset_name)
    _delete_progress_suffix(db, dataset_name, repair_start, start_global, include_mesh=True)
    final_config = _make_repair_config(base_config, original_min, repair_start, original_max)
    _call_align_stack_z(align_stack_z, final_config)

    LOGGER.info('Repair complete. Opening final Neuroglancer inspection view.')
    inspect_dataset(base_config['destination_path'],
                    bounding_box=[max(0, start_global - 2), min(start_global + 2 * preview_after + 5, start_global + (original_max - repair_start))],
                    keep_missing=True,
                    bind_port=inspect_port)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Repair and propagate a problematic Z alignment in an existing EMalign stack.',
        epilog=(
            'CONFIG_FILE should usually be 00_align_plan.json from config/z_config. The script '
            'uses global destination slice numbers only, selects the matching stack config, '
            'and repairs from that global slice onward. A per-stack z_*.json config is still '
            'accepted for advanced use, but --slice-range remains global.'
        )
    )
    parser.add_argument(
        'config_file',
        nargs='?',
        help=(
            'Path to 00_align_plan.json from config/z_config, or an advanced per-stack z_*.json config. '
            'This is not the destination_path/final aligned stack directory.'
        )
    )
    parser.add_argument(
        '--config-file',
        '--config_file',
        dest='config_file_option',
        help=(
            'Path to 00_align_plan.json from config/z_config, or an advanced per-stack z_*.json config. '
            'Use this if you prefer a named flag instead of the positional CONFIG_FILE.'
        )
    )
    parser.add_argument('--slice-range', required=True, type=_parse_slice_range,
                        help='Bad destination/global slice or inclusive global range to preview, e.g. "531" or "531:536". Local stack slice numbers are not accepted.')
    parser.add_argument('--preview-after', type=int, default=5,
                        help='Number of additional slices after --slice-range to include in the preview realignment.')
    parser.add_argument('--port', type=int, default=55555, help='Neuroglancer bind port for inspection.')
    parser.add_argument('--skip-preview', action='store_true',
                        help='Do not pause for Neuroglancer confirmation; immediately propagate to the end of the stack.')
    parser.add_argument('--project-name', default=None, help='Override project_name from the config file.')
    args = parser.parse_args()
    config_file = args.config_file_option or args.config_file
    if config_file is None:
        parser.error('CONFIG_FILE is required. Provide 00_align_plan.json, or an advanced per-stack z_*.json config, positionally or with --config-file.')
    if args.config_file_option and args.config_file and args.config_file_option != args.config_file:
        parser.error('Provide CONFIG_FILE either positionally or with --config-file, not both with different values.')

    repair_z_alignment(config_file, args.slice_range, args.preview_after, args.port, args.skip_preview, args.project_name)


if __name__ == '__main__':
    main()
