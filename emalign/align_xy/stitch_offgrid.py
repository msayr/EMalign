'''
Functions for stitching images existing on the same XY plane.
Contrary to tilemaps, these functions are for images that are not on a predictable grid. 
'''

import cv2
import numpy as np
import jax.numpy as jnp

from connectomics.common import bounding_box
from sofima import flow_field, flow_utils, mesh
from sofima.warp import ndimage_warp

from emalign.io.process.mask import compute_greyscale_mask, mask_to_bbox
from emalign.io.process.transform import rotate_image

from ..arrays.sift import estimate_transform_sift
from ..arrays.utils import homogenize_arrays_shape, xy_offset_to_pad, compute_laplacian_var
from ..arrays.overlap import get_overlap
from .utils import mask_to_mesh


class TransformEstimationError(RuntimeError):
    """Raised when SIFT cannot estimate an affine transform for image stitching."""


class CanvasSizeError(RuntimeError):
    """Raised when a fused stitch would create an implausibly large canvas."""


def _max_fused_canvas_shape(img1_shape, img2_shape, max_canvas_scale):
    """Return the largest acceptable YX shape for fusing overlapping stack images."""
    return np.ceil(np.maximum(img1_shape, img2_shape) * max_canvas_scale).astype(int)


def _raise_if_canvas_too_large(candidate_shape, img1_shape, img2_shape, max_canvas_scale, stage):
    """Reject runaway affine/elastic transforms before a massive canvas is allocated or written."""
    if max_canvas_scale is None:
        return

    max_shape = _max_fused_canvas_shape(img1_shape, img2_shape, max_canvas_scale)
    candidate_shape = np.array(candidate_shape)
    if np.any(candidate_shape > max_shape):
        raise CanvasSizeError(
            f'Fused XY canvas exceeds allowed size during {stage}. '
            f'Got YX shape {tuple(map(int, candidate_shape))}, maximum allowed '
            f'{tuple(map(int, max_shape))} from input shapes '
            f'{tuple(map(int, img1_shape))} and {tuple(map(int, img2_shape))}.'
        )


def get_elastic_mesh(pre, 
                     post, 
                     pre_mask, 
                     post_mask, 
                     patch_size, 
                     stride,
                     k0=0.01,
                     k=0.1,
                     gamma=0.5):
    
    # Estimate flow
    mfc = flow_field.JAXMaskedXCorrWithStatsCalculator()
    flow = mfc.flow_field(pre, post, (patch_size,patch_size),
                            (stride,stride), batch_size=256,
                            pre_mask=~pre_mask, 
                            post_mask=~post_mask)
    x = mask_to_mesh(post_mask, flow.shape[-2:])

    # Clean flow
    pad = patch_size // 2 // stride
    flow = np.pad(flow, [[0, 0], [pad, pad - 1], [pad, pad - 1]], constant_values=np.nan)
    x = np.pad(x, [[0, 0], [0, 0], [pad, pad - 1], [pad, pad - 1]], constant_values=np.nan)

    kwargs = {"min_peak_ratio": 1.4, "min_peak_sharpness": 1.4, "max_deviation": 5, "max_magnitude": 0}
    flow = flow_utils.clean_flow(flow[:, None, ...], **kwargs)

    mesh_config = mesh.IntegrationConfig(dt=0.001, gamma=gamma, k0=k0, k=k, stride=(stride, stride), num_iters=1000,
                                            max_iters=100000, stop_v_max=0.005, dt_max=1000, start_cap=0.01,
                                            final_cap=10, prefer_orig_order=False)
    x, _, _ = mesh.relax_mesh(jnp.array(x), -flow, mesh_config)
    return np.array(x)[:,0,...]


def render_fused_slice(pre, 
                       post,
                       pre_mask,
                       post_mask,
                       x,
                       stride,
                       bbox_post=[0,-1,0,-1],
                       work_size=512,
                       overlap=1,
                       parallelism=1,
                       output_shape=None,
                       post_on_top=False,
                       resize_canvas=True,
                       max_canvas_scale=1.5,
                       expected_shape=None,
                       
                       ):
    
    y1,y2,x1,x2 = bbox_post
    post = post[y1:y2, x1:x2]
    post_mask = post_mask[y1:y2, x1:x2]

    if output_shape is None:
        output_shape = post.shape

    data_bbox = bounding_box.BoundingBox(start=(0, 0), 
                                        size=(output_shape[-1], output_shape[-2]))
    # Warp image
    warped_img = ndimage_warp(
        post, 
        x, 
        stride=(stride, stride), 
        work_size=(work_size, work_size), 
        overlap=(overlap,overlap),
        image_box=data_bbox,
        parallelism=parallelism
    )

    # Warp mask
    warped_mask = ndimage_warp(
        post_mask, 
        x, 
        stride=(stride, stride),
        work_size=(work_size, work_size),
        overlap=(overlap,overlap),
        image_box=data_bbox,
        parallelism=parallelism
    )

    # Stitch images together
    stitched = pre.copy()
    if post_on_top:
        m = warped_mask
    else:
        m = warped_mask & ~pre_mask[y1:y2,x1:x2]

    stitched[y1:y2,x1:x2][m] = warped_img[m]
    stitched_mask = pre_mask
    stitched_mask[y1:y2,x1:x2] |= warped_mask
    
    if resize_canvas:
        y1,y2,x1,x2 = mask_to_bbox(stitched_mask)
        stitched = stitched[y1:y2,x1:x2]
        stitched_mask = stitched_mask[y1:y2,x1:x2]

    if expected_shape is not None:
        _raise_if_canvas_too_large(
            stitched.shape,
            expected_shape[0],
            expected_shape[1],
            max_canvas_scale,
            'elastic render',
        )
    return stitched, stitched_mask


def stitch_images(img1, 
                  img2,
                  mask1=None, 
                  mask2=None,
                  scale=0.1,
                  initial_offset=None,
                  use_initial_offset_only=False,
                  patch_size=160,
                  stride=40,
                  parallelism=1,
                  img_on_top='auto',
                  img_q_fun=None,
                  resize_canvas=True,
                  max_canvas_scale=1.5,
                  **kwargs):
    
    '''Stitch two images on the same slice. 
    
    Img1 is the reference, img2 is the moving image.
    Img1 is the one being rotated and offset when necessary, because the rotation would result in a staircase artifact at the boundaries of the mesh.

    initial_offset: optional (y, x) top-left offset of img2 relative to img1.
        When provided, images are padded into that rough arrangement before registration.
    use_initial_offset_only: if True, skip SIFT affine estimation and use initial_offset
        directly as the affine guide before elastic refinement.
    img_q_fun: function taking image and mask as arguments, returns a value higher for higher quality/sharpness.
    e.g.: img_q_fun = lambda img, m: compute_laplacian_var(img, m)*0.5 + compute_sobel_mean(img, m) + compute_grad_mag(img, m)*100
    '''

    original_img1_shape = np.array(img1.shape)
    original_img2_shape = np.array(img2.shape)

    # Compute mask if not provided
    if mask1 is None:
        mask1 = compute_greyscale_mask(img1)
    if mask2 is None:
        mask2 = compute_greyscale_mask(img2)
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)

    if initial_offset is not None:
        initial_offset = np.asarray(initial_offset, dtype=int)
        if initial_offset.shape != (2,):
            raise ValueError('initial_offset must be a two-value (y, x) offset')
        pad1_before = np.maximum(initial_offset, 0)
        pad2_before = np.maximum(-initial_offset, 0)
        target_shape = np.maximum(pad1_before + np.array(img1.shape), pad2_before + np.array(img2.shape))
        pad1_after = target_shape - pad1_before - np.array(img1.shape)
        pad2_after = target_shape - pad2_before - np.array(img2.shape)
        img1 = np.pad(img1, [(pad1_before[0], pad1_after[0]), (pad1_before[1], pad1_after[1])])
        mask1 = np.pad(mask1, [(pad1_before[0], pad1_after[0]), (pad1_before[1], pad1_after[1])])
        img2 = np.pad(img2, [(pad2_before[0], pad2_after[0]), (pad2_before[1], pad2_after[1])])
        mask2 = np.pad(mask2, [(pad2_before[0], pad2_after[0]), (pad2_before[1], pad2_after[1])])
        if use_initial_offset_only:
            M = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
            img1_shape = np.array(img1.shape)
            img2_offset = np.array([0, 0])
        else:
            M, img1_shape, img2_offset, robust_estimate, robustness_metrics = estimate_transform_sift(
                img2, img1, scale, ref_mask=mask2, mov_mask=mask1, refine_estimate=True)
    else:
        M, img1_shape, img2_offset, robust_estimate, robustness_metrics = estimate_transform_sift(
            img2,
            img1,
            scale,
            ref_mask=mask2,
            mov_mask=mask1,
            refine_estimate=True,
        )
    if M is None or img1_shape is None or img2_offset is None:
        raise TransformEstimationError(
            'SIFT could not estimate a valid transform between the fused canvas and stack image '
            f'(scale={scale}, robust_estimate={robust_estimate}, '
            f'robustness_metrics={robustness_metrics}).'
        )
    _raise_if_canvas_too_large(
        img1_shape,
        original_img1_shape,
        original_img2_shape,
        max_canvas_scale,
        'affine warp',
    )

    # Pad moving image so it matches the reference
    padded_img2_shape = np.array(img2.shape) + np.sum(xy_offset_to_pad(img2_offset), axis=1)
    _raise_if_canvas_too_large(
        padded_img2_shape,
        original_img1_shape,
        original_img2_shape,
        max_canvas_scale,
        'moving-image padding',
    )

    img1 = cv2.warpAffine(img1, M, img1_shape[::-1])  
    mask1 = cv2.warpAffine(mask1.astype(np.uint8), M, img1_shape[::-1]).astype(bool)

    img2 = np.pad(img2, xy_offset_to_pad(img2_offset))
    mask2 = np.pad(mask2, xy_offset_to_pad(img2_offset))

    # Make sure that images have the same shape for sofima
    homogenized_shape = np.maximum(img1.shape, img2.shape)
    _raise_if_canvas_too_large(
        homogenized_shape,
        original_img1_shape,
        original_img2_shape,
        max_canvas_scale,
        'shape homogenization',
    )
    img1, img2 = homogenize_arrays_shape([img1, img2])
    mask1, mask2 = homogenize_arrays_shape([mask1, mask2])

    if img_on_top == 'auto':
        # Determine automatically what image to put on top
        mask = mask1 & mask2
        y1,y2,x1,x2 = mask_to_bbox(mask)
        l1 = img_q_fun(img1[y1:y2, x1:x2], mask[y1:y2, x1:x2])
        l2 = img_q_fun(img2[y1:y2, x1:x2], mask[y1:y2, x1:x2])
        if l1 > l2:
            post_on_top = False
        else:
            post_on_top = True
    elif img_on_top == '1':
        post_on_top = False
    elif img_on_top == '2':
        post_on_top = True

    # Mesh represents transformation from img2 to img1. We can crop img1 to avoid OOM errors
    pad = 200

    y1,y2,x1,x2 = mask_to_bbox(mask2)
    y1,x1 = np.max([np.array([y1,x1]) - pad, [0,0]], axis=0)
    y2,x2 = np.min([[y2 + pad, x2 + pad], img2.shape], axis=0)
    
    x = get_elastic_mesh(img1[y1:y2, x1:x2], 
                         img2[y1:y2, x1:x2], 
                         mask1[y1:y2, x1:x2], 
                         mask2[y1:y2, x1:x2], 
                         patch_size, 
                         stride,
                         **kwargs)
    
    return render_fused_slice(img1, 
                              img2, 
                              mask1, 
                              mask2,
                              x,
                              stride,
                              bbox_post=[y1,y2,x1,x2],
                              work_size=512,
                              overlap=5,
                              parallelism=parallelism,
                              post_on_top=post_on_top,
                              resize_canvas=resize_canvas,
                              max_canvas_scale=max_canvas_scale,
                              expected_shape=(original_img1_shape, original_img2_shape))
