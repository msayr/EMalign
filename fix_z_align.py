#!/usr/bin/env python3
"""Repair a bad Z-alignment boundary without rebuilding the whole stack.

This utility reuses an existing ``align_stack_z.py`` JSON config, re-aligns a
small user-selected preview range, opens a Neuroglancer inspection view, and, on
CLI confirmation, propagates the repaired alignment from the first bad slice to
the end of that stack by rerunning Z alignment for the suffix only.
"""

import argparse
import copy
import json
import logging
import os
from typing import Optional, Tuple

from emalign.inspect_dataset import inspect_dataset
from emalign.io.progress import get_mongo_client, get_mongo_db
from emalign.io.store import open_store
from emalign.scripts.align_stack_z import align_stack_z


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def _parse_slice_range(value: str) -> Tuple[int, int]:
    """Parse ``N`` or ``N:M`` / ``N-M`` into an inclusive local slice range."""
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
    dataset = open_store(os.path.abspath(config['dataset_path']), mode='r')
    default_min = int(dataset.domain.inclusive_min[0])
    default_max = int(dataset.domain.exclusive_max[0])
    return int(config.get('local_z_min', default_min) or default_min), int(config.get('local_z_max', default_max) or default_max)


def _global_z_for_local(config: dict, original_local_min: int, local_z: int) -> int:
    """Map an input stack-local z index to the destination/global z index."""
    return int(config.get('z_offset', 0)) + int(local_z) - int(original_local_min)


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
    config['local_z_min'] = int(local_start)
    config['local_z_max'] = int(local_stop_exclusive)
    config['z_offset'] = _global_z_for_local(base_config, original_local_min, local_start)
    config['first_slice'] = config['z_offset'] - 1
    config['overwrite'] = True
    config['wipe_progress_flag'] = False
    return config


def _confirm(prompt: str) -> bool:
    answer = input(f'{prompt} [y/N]: ').strip().lower()
    return answer in {'y', 'yes'}


def repair_z_alignment(config_path: str,
                       slice_range: Tuple[int, int],
                       preview_after: int,
                       inspect_port: int,
                       skip_preview: bool = False,
                       project_name: Optional[str] = None) -> None:
    with open(config_path, 'r') as f:
        base_config = json.load(f)

    if project_name is not None:
        base_config['project_name'] = project_name
    if not base_config.get('project_name'):
        base_config['project_name'] = os.path.basename(base_config['destination_path']).rstrip('.zarr')

    dataset_name = base_config['dataset_name']
    original_min, original_max = _stack_bounds(base_config)
    repair_start, repair_end = slice_range
    if repair_start < original_min or repair_end >= original_max:
        raise ValueError(f'Repair range {repair_start}:{repair_end} is outside configured stack bounds [{original_min}, {original_max}).')
    if repair_start == original_min and base_config.get('first_slice') is None and base_config.get('reference_path') is None:
        raise ValueError('Cannot repair the first slice of a root stack because there is no previous aligned slice to anchor to.')

    preview_stop = min(original_max, max(repair_end + 1, repair_start + 1) + max(0, preview_after))
    start_global = _global_z_for_local(base_config, original_min, repair_start)

    client = get_mongo_client(base_config.get('mongodb_config_filepath'))
    db = get_mongo_db(client, base_config['project_name'])

    if not skip_preview:
        LOGGER.info('Realigning preview range local z [%d, %d) for %s.', repair_start, preview_stop, dataset_name)
        _delete_progress_suffix(db, dataset_name, repair_start, start_global, include_mesh=True)
        preview_config = _make_repair_config(base_config, original_min, repair_start, preview_stop)
        align_stack_z(**preview_config)

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
    align_stack_z(**final_config)

    LOGGER.info('Repair complete. Opening final Neuroglancer inspection view.')
    inspect_dataset(base_config['destination_path'],
                    bounding_box=[max(0, start_global - 2), min(start_global + 2 * preview_after + 5, start_global + (original_max - repair_start))],
                    keep_missing=True,
                    bind_port=inspect_port)


def main() -> None:
    parser = argparse.ArgumentParser(description='Repair and propagate a problematic Z alignment in an existing EMalign stack.')
    parser.add_argument('config_file', help='Existing align_stack_z JSON config for the stack to repair.')
    parser.add_argument('--slice-range', required=True, type=_parse_slice_range,
                        help='Input stack-local bad slice or inclusive range to preview, e.g. "201" or "201:205".')
    parser.add_argument('--preview-after', type=int, default=5,
                        help='Number of additional slices after --slice-range to include in the preview realignment.')
    parser.add_argument('--port', type=int, default=55555, help='Neuroglancer bind port for inspection.')
    parser.add_argument('--skip-preview', action='store_true',
                        help='Do not pause for Neuroglancer confirmation; immediately propagate to the end of the stack.')
    parser.add_argument('--project-name', default=None, help='Override project_name from the config file.')
    args = parser.parse_args()

    repair_z_alignment(args.config_file, args.slice_range, args.preview_after, args.port, args.skip_preview, args.project_name)


if __name__ == '__main__':
    main()
