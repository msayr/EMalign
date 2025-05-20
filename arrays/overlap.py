import logging
import cv2
import numpy as np

from emprocess.utils.transform import rotate_image
from emprocess.utils.mask import mask_to_bbox  

from .sift import estimate_transform_sift
from .utils import compute_laplacian_var_diff, homogenize_arrays_shape, xy_offset_to_pad


def get_overlap(img1, img2, offset, rotation=0, pad=0, homogenize_shapes=False):
    '''
    Extract overlapping parts of two images based on an offset and rotation from img2 to img1.
    '''
        
    if rotation != 0:
        if img2.dtype == bool:
            img2 = rotate_image(img2.astype(np.uint8), rotation).astype(bool)
        else:
            img2 = rotate_image(img2, rotation)
    
    offset = offset[::-1]
    if offset[1] > 0:
        ox = img2.shape[1] - int(abs(offset[1])) + pad
        crop2 = img2[:, -ox:]
        crop1 = img1[:, :ox]
    else:
        ox = img1.shape[1] - int(abs(offset[1])) + pad
        crop1 = img1[:, -ox:]
        crop2 = img2[:, :ox]

    if offset[0] < 0:
        oy = img1.shape[0] - int(abs(offset[0])) + pad
        crop1 = crop1[-oy:, :]
        crop2 = crop2[:oy, :]
    else:
        oy = img2.shape[0] - int(abs(offset[0])) + pad
        crop1 = crop1[:oy, :]
        crop2 = crop2[-oy:, :]

    if homogenize_shapes:
        y, x = np.min([crop1.shape, crop2.shape], axis=0)
        crop1 = crop1[:y,:x]
        crop2 = crop2[:y,:x]

    return crop1, crop2


def get_overlap_warp(ref_img, mov_img, ref_mask, mov_mask, M, mov_img_shape, ref_img_offset):
    
    mov_img = cv2.warpAffine(mov_img, M, mov_img_shape[::-1])  
    ref_mask = cv2.warpAffine(ref_mask.astype(np.uint8), M, mov_img_shape[::-1]).astype(bool)

    # Pad moving image so it matches the reference
    ref_img = np.pad(ref_img, xy_offset_to_pad(ref_img_offset))
    mov_mask = np.pad(mov_mask, xy_offset_to_pad(ref_img_offset))

    # Make sure that images have the same shape for sofima
    mov_img, ref_img = homogenize_arrays_shape([mov_img, ref_img])
    ref_mask, mov_mask = homogenize_arrays_shape([ref_mask, mov_mask])

    mask = ref_mask & mov_mask
    y1,y2,x1,x2 = mask_to_bbox(mask)

    return ref_img[y1:y2, x1:x2], mov_img[y1:y2, x1:x2]


def check_overlap(img1, 
                  img2, 
                  xy_offset, 
                  theta, 
                  threshold=0.5, 
                  scale=(0.3, 0.5), 
                  refine=True):

    '''
    Compute a metric describing how well images overlap, based on a given offset and rotation. 
    '''

    # Index of sharpness using Laplacian
    overlap = get_overlap(img1, img2, xy_offset, theta)

    if overlap is not None:
        overlap1, overlap2, mask = overlap

        lap_variance_diff = compute_laplacian_var_diff(overlap1, overlap2, mask)

        if refine and lap_variance_diff < threshold:
            logging.debug('Refining overlap estimation...')
            # Retry the overlap, it can often get better
            try:
                xy_offset, theta = estimate_transform_sift(overlap1, overlap2, scale=scale[0])
            except:
                xy_offset, theta = estimate_transform_sift(overlap1, overlap2, scale=scale[1])
            res = get_overlap(overlap1, overlap2, xy_offset, theta)
            
            if res is not None:
                overlap1, overlap2, mask = res
                lap_variance_diff = compute_laplacian_var_diff(overlap1, overlap2, mask)
            else:
                lap_variance_diff = 0
    else:
        # Images do not overlap (displacement is larger than image itself)
        lap_variance_diff = 0

    return lap_variance_diff
