'''
Utilities for finding and reading metadata produced by the ThermoFisher Volumescope microscope.
'''


import logging
import os
import re

from concurrent import futures
from glob import glob
from tqdm import tqdm

def get_tileset_resolution(tileset_path):
    '''
    Find resolution of a stack by reading an .info file located in the stack directory.
    '''
    info=None
    with os.scandir(tileset_path) as entries:
        for entry in entries:
            if entry.name.endswith('.info'):
                info = entry.path
                break

    if info is None:
        return None

    with open(info, 'r') as f:
        content = f.readlines()

    resolution = None
    for line in content:
        if 'resolution' in line.lower() or 'pixel size' in line.lower():
            matches = re.findall(r'\d+', line)
            if len(matches) >= 2:
                resolution = tuple(map(int, matches[:2]))
                break

    # Fallback to line 5 for backward compatibility with existing .info files
    if resolution is None:
        if len(content) > 5:
            matches = re.findall(r'\d+', content[5])
            if len(matches) >= 2:
                resolution = tuple(map(int, matches[:2]))
            else:
                logging.warning(f'Could not determine resolution from .info file in {tileset_path}')
                return None
        else:
            logging.warning(f'Could not determine resolution from .info file in {tileset_path} (insufficient lines)')
            return None

    return (tileset_path, resolution)


def get_tilesets(main_dir, resolution, dir_pattern, num_workers):

    # Get all directories containing tilesets that are present in main_dir
    tileset_dirs = glob(os.path.join(main_dir, '*', ''))

    stack_list = []
    # Find the ones with the right resolution
    with futures.ThreadPoolExecutor(num_workers) as tpe:
        fs = []
        for d in tileset_dirs:
            fs.append(tpe.submit(get_tileset_resolution, d))

        for f in tqdm(futures.as_completed(fs), total=len(fs), desc=f'Looking for resolution: {resolution}', leave=False):
            result = f.result()
            if result is None:
                continue
            # Find the directory with the right pattern if relevant
            for d in dir_pattern:
                if d in result[0].split('/')[-2] and result[1] == tuple(resolution):
                    stack_list.append(result[0])
    return sorted(stack_list)