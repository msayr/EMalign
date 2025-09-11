import cv2
import numpy as np

from emprocess.utils.transform import rotate_image
from emprocess.utils.io import get_dataset_attributes

# WRITE
def write_slice(dataset, arr, z, x_offset=0, y_offset=0):

    y,x = arr.shape
    new_max = np.array([z+1, y+y_offset, x+x_offset], dtype=int)
    current_max = np.array(dataset.domain.exclusive_max, dtype=int)
    if np.any(current_max < new_max):
        new_max = np.max([current_max, new_max], axis=0)
        dataset = dataset.resize(exclusive_max=new_max, expand_only=True).result()
    try:
        return dataset, dataset[z:z+1, y_offset:y+y_offset, x_offset:x+x_offset].write(arr).result()
    except Exception as e:
        raise e


# READ
def find_ref_slice(dataset, z=None, reverse=False):

    '''Find first or last non-black slice of an image stack.

    Args:
        dataset (tensorstore.TensorStore): A dataset containing the image data.
        z (int or None): Z index to start from (axis=0). If None, will pick a start based on reverse. Defaults to None.
        reverse (bool): Whether to look for images at indices higher than z (False) or lower than z (True). Defaults to False.
    
    Returns:
        tuple: tuple of image np.ndarray and corresponding z index.
    '''
    
    increment = -1 if reverse else 1

    if z is None and reverse:
        z = dataset.domain.exclusive_max[0] - 1 
    elif z is None:
        z = dataset.domain.inclusive_min[0]

    img = dataset[z].read().result()

    while not img.any():
        z += increment
        img = dataset[z].read().result()
    return img, z


def get_data_samples(dataset, step_slices, yx_target_resolution):

    resolution = np.array(get_dataset_attributes(dataset)['resolution'])[1:]

    z_max = dataset.domain.exclusive_max[0]-1

    z_list = np.arange(0, z_max, step_slices)
    z_list = np.append(z_list, z_max) if z_max not in z_list else z_list    

    data = []
    for z in z_list:
        arr = dataset[z].read().result()
        while not arr.any():
            z += 1
            arr = dataset[z].read().result()
        
        if np.any(resolution < yx_target_resolution):
            fy, fx = resolution/yx_target_resolution
            arr = cv2.resize(arr, None, fx=fx, fy=fy)
        elif np.any(resolution > yx_target_resolution):
            raise RuntimeError(f'Dataset resolution ({resolution.tolist()}) must be lower \
                               than target resolution ({yx_target_resolution.tolist()})')
        data.append(arr)

    return np.array(data)