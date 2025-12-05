import cv2
import numpy as np

from emprocess.utils.mask import mask_to_bbox

def resample(array, ratio):
    '''
    Resize an array by a scaling ratio using OpenCV interpolation.

    Parameters:
    -----------
    array : np.ndarray
        Input array to resize
    ratio : float
        Scaling factor. Values < 1 downsample, values > 1 upsample.
        If 1, returns original array without change.

    Returns:
    --------
    np.ndarray
        Resized array

    Raises:
    -------
    ValueError
        If ratio is not positive

    Notes:
    ------
    - Uses cv2.resize with default interpolation (bilinear)
    - Boolean arrays are converted to uint8, resized, then converted back to bool
    - Returns original array without copying if ratio == 1
    '''
    if ratio <= 0:
        raise ValueError(f"Ratio must be positive, got {ratio}")

    if ratio == 1:
        return array

    if array.dtype == bool:
        # cv2.resize doesn't want boolean
        array = cv2.resize(array.astype(np.uint8), None, fx=ratio, fy=ratio).astype(bool)
    else:
        array = cv2.resize(array, None, fx=ratio, fy=ratio)
    return array

# PAD
def xy_offset_to_pad(offset):
    pad = np.zeros([2,2], dtype=int)
    x,y = [int(i) for i in offset]
    
    if y > 0:
        pad[0][1] = y
    else:
        pad[0][0] = abs(y)
    
    if x > 0:
        pad[1][1] = x
    else:
        pad[1][0] = abs(x)

    return pad


def pad_to_shape(array, target_shape, direction=None, axis=None, pad_value=0):
    '''
    Pad an array to match a shape along specified axes. If the target shape is smaller 
    than the array's shape in a dimension, no padding is added to that dimension.
    
    Parameters:
    -----------
    array : np.ndarray
        Input array to pad
    target_shape : list or tuple
        Target shape for the specified axes
    direction : list
        Direction to pad for each axis (0=left/start, 1=right/end)
    axis : list
        Axes to apply padding to (supports negative indexing)
    pad_value : 
        Value to use for padding
        
    Returns:
    --------
    np.ndarray
        Padded array
    '''
    # Convert axis to positive indices and validate
    if axis is not None:
        axis = np.array(axis)
        axis = np.where(axis < 0, array.ndim + axis, axis)
    else:
        axis=np.arange(0, array.ndim)

    if direction is None:
        direction = np.ones_like(axis)

    # Validate inputs
    assert len(target_shape) == len(axis), 'Target_shape length must match axis length'
    assert len(direction) == len(axis), 'Direction length must match axis length'
    assert all(0 <= ax < array.ndim for ax in axis), 'Axis indices out of bounds'
    
    # Calculate padding needed for each specified axis
    pad_sizes = np.array(target_shape) - np.array(array.shape)[axis]
    pad_sizes = np.max([pad_sizes, np.zeros_like(pad_sizes)], axis=0)  # No negative padding
    
    # Create padding specification for np.pad
    pad_width = np.zeros((array.ndim, 2), dtype=int)
    for i, (ax, pad_size, dir_val) in enumerate(zip(axis, pad_sizes, direction)):
        if pad_size > 0:
            if dir_val == 0:  # Pad at start
                pad_width[ax, 0] = pad_size
            else:  # Pad at end (default)
                pad_width[ax, 1] = pad_size
    
    return np.pad(array, pad_width, constant_values=pad_value)


def homogenize_arrays_shape(arrs, pad_value=0):
    max_shape = np.max([a.shape for a in arrs],axis=0)
    return np.array([pad_to_shape(a, max_shape, pad_value=pad_value) for a in arrs])


# ASSESS QUALITY
def _compute_laplacian_var(arr, mask=None):
    '''
    Compute laplacian variance. Provides an indication of image sharpness but is sensitive to contrast.
    '''
    if mask is not None:
        l = cv2.Laplacian(arr, cv2.CV_64F)[mask]
    else:
        l = cv2.Laplacian(arr, cv2.CV_64F)
    return np.var(l)


def _compute_sobel_mean(arr, mask=None):
    '''
    Apply Sobel operator. Provides an indication of image sharpness
    '''
    if mask is not None:
        sobel_x = cv2.Sobel(arr, cv2.CV_64F, 1, 0, ksize=5)[mask]
        sobel_y = cv2.Sobel(arr, cv2.CV_64F, 0, 1, ksize=5)[mask]
    else:
        sobel_x = cv2.Sobel(arr, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(arr, cv2.CV_64F, 0, 1, ksize=5)
    sobel = np.sqrt(sobel_x**2 + sobel_y**2)
    return np.mean(sobel)


def _compute_grad_mag(arr, mask=None):
    '''
    Compute gradient magnitude. Provides an indication of pixel variation, which can be used to measure sharpness/quality. 
    '''
    gy, gx = np.gradient(arr)

    if mask is not None:
        gnorm = np.sqrt(gx**2 + gy**2)[mask]
    else:
        gnorm = np.sqrt(gx**2 + gy**2)
    return np.average(gnorm)


def compute_laplacian_var_diff(overlap1, 
                               overlap2, 
                               mask=None):

    '''
    Compute a metric ([0,1]) describing how well two arrays overlap, based on laplacian filter.
    If score is 1, overlapping regions have the same edge content and therefore overlap well.
    '''
    
    lap_var1 = _compute_laplacian_var(overlap1, mask)
    lap_var2 = _compute_laplacian_var(overlap2, mask)

    # Calculate an index of difference in edge content (variance of laplacian)
    # Between 0 and 1, low means exact same content, 1 means different
    return 1 - abs(lap_var1 - lap_var2) / max(lap_var1, lap_var2)