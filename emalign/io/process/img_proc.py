import cv2
import numpy as np

from emalign.io.process.mask import compute_greyscale_mask

def downsample(img, ratio=0.3):
    if ratio == 1:
        return img
    return cv2.resize(img,
                     (0,0), 
                     fx=ratio, 
                     fy=ratio, 
                     interpolation=cv2.INTER_NEAREST)


def process_image(img, process_scheme, compute_mask=False):
    '''
    Run image processing based on a dictionary of function keyword to a dictionary of function arguments name to values.
    If a function has no additional parameter, pass an empty dictionary as argument.
    '''

    proc_fun = {'gaussian': proc_gaussian,
                'clahe': proc_clahe,
                'equalize': proc_equalize}

    if 'invert' in process_scheme:
        img = proc_invert(img)

    if compute_mask:
        if img.any():
            mask = compute_greyscale_mask(img, 0)
        else:
            mask = np.zeros_like(img).astype(bool)
    else:
        mask = None

    if len(process_scheme) == 0:
        return img, mask
    
    for fun, kwargs in process_scheme.items():
        if fun == 'invert':
            continue
        img = proc_fun[fun](img, mask, **kwargs)

    return img, mask


def proc_invert(img):
    '''
    Invert image.
    '''
    return np.invert(img).astype(np.uint8)


def proc_gaussian(img, mask=None, kernel_size=(3,3), sigma=0.7):
    '''
    Apply gaussian filter to image by convolving a kernel of size kernel_size.
    Sigma value corresponds to how strong the effect is (peak of gaussian).
    '''
    if mask is not None:
        processed_img = img.copy()
        processed_img[mask] = cv2.GaussianBlur(processed_img[mask], kernel_size, sigma).squeeze()
        return processed_img
    else:
        return cv2.GaussianBlur(img, kernel_size, sigma)
    

def proc_clahe(img, mask=None, clip_limit=2, tile_grid_size=(10,10)):
    '''
    Apply CLAHE (Constrast Limited Adaptive Histogram Equalization) to enhance contrast.
    '''
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if mask is not None:
        processed_img = img.copy()
        processed_img[mask] = clahe.apply(processed_img[mask]).squeeze()
        return processed_img
    else:
        return clahe.apply(img)


def proc_equalize(img, mask=None):
    '''
    Equalize image histogram to get more consistent contrast and brightness.
    '''
    if mask is not None:
        processed_img = img.copy()
        processed_img[mask] = cv2.equalizeHist(processed_img[mask]).squeeze()
        return processed_img
    else:
        return cv2.equalizeHist(img)
