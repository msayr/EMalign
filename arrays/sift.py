import cv2
import numpy as np

from emprocess.utils.img_proc import downsample


def adjust_matrix_to_shape(mov_img, M):

    y, x = mov_img.shape[:2]
    corners = np.array([[0, 0], [x, 0], [x, y], [0, y]], dtype=np.float32).reshape(-1, 1, 2)
    transformed_corners = cv2.transform(corners, M)
    
    min_x = np.floor(np.min(transformed_corners[:, 0, 0])).astype(int)
    min_y = np.floor(np.min(transformed_corners[:, 0, 1])).astype(int)
    max_x = np.ceil(np.max(transformed_corners[:, 0, 0])).astype(int)
    max_y = np.ceil(np.max(transformed_corners[:, 0, 1])).astype(int)
    
    # Create adjusted transformation matrix that accounts for shifts
    translation_adj = np.array([
        [1, 0, -min(min_x, 0)],
        [0, 1, -min(min_y, 0)]
    ], dtype=np.float32)
    
    adjusted_M = M.copy()
    adjusted_M[:, 2] += translation_adj[:, 2]

    # Get output shape
    output_w = np.ceil(max_x-min(min_x, 0))
    output_h = np.ceil(max_y-min(min_y, 0))
    mov_shape = (output_h, output_w)

    # Get transformation for the other image to match
    ref_offset = M[:, 2] - adjusted_M[:, 2]

    return adjusted_M, mov_shape, ref_offset


def estimate_transform_sift(ref_img, 
                            mov_img, 
                            scale=1.0, 
                            refine_estimate=True,
                            return_upscaled_matrix=True):
    '''Estimate transformation (xy offset and rotation) from img2 to img1 using SIFT.

    Args:
        ref_img (np.ndarray): Reference greyscale image.
        mov_img (np.ndarray): Moving greyscale image.
        scale (float, optional): Scale to downsample images to for computing the offset. Defaults to 1.
        refine_estimate (bool, optional): Whether to try again with higher resolution if the first estimate is found to be invalid. Defaults to True.
        return_upscaled_matrix (bool, optional): Whether to return the matrix corresponding to the transformation to apply to the original image (as opposed to the downsampled one). Defaults to True.

    Returns:
        tuple of: 
            M (np.ndarray): Affine transformation matrix to apply to mov_img.
            output_shape (tuple): (y,x) shape for mov_img after transformation.
            ref_offset (nd.array): (x,y) offset to apply to ref_img for it to match with mov_img.
            robust_estimate (bool): Whether the estimate was valid based on the number and proportion of good matches.
    '''
    # knnMatch will return an error if there are too many keypoints so we limit their number
    max_features=250000

    # Downsample images for faster computations
    ds_ref_img = downsample(ref_img, scale)
    ds_mov_img = downsample(mov_img, scale)

    # Find keypoints using SIFT
    sift = cv2.SIFT_create(nfeatures=max_features)
    kp1, des1 = sift.detectAndCompute(ds_ref_img,None)
    kp2, des2 = sift.detectAndCompute(ds_mov_img,None)

    if len(kp1) and len(kp2):
        # Match keypoints to each other
        # Brute force matchers is slower than flann, but it is exact
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2,k=2)

        good_matches = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good_matches.append(m)
        
        good_matches = sorted(good_matches, key=lambda x: x.distance)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
        src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

        # Estimate affine transformation matrix
        try:
            M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        except cv2.error as e:
            if 'count >= 0' in e.err:
                M = None
            else:
                raise e
        except Exception as e:
            raise e
    else:
        M = None
               
    if M is None or np.isnan(M).all():
        output_shape = None
        ref_offset = None
        robust_estimate = False
    else:
        if return_upscaled_matrix:
            M[:,2] /= scale
            M, output_shape, ref_offset = adjust_matrix_to_shape(mov_img, M)
        else:
            M, output_shape, ref_offset = adjust_matrix_to_shape(mov_img, M)
        # Check that we have enough good matches, enough inliers, and a good proportion of inliers
        robust_estimate = (len(good_matches)>10) and \
                          (inliers.sum()>10) and (inliers.sum() / len(good_matches) > 0.6)
        

    if refine_estimate and not robust_estimate and scale<0.9:
        return estimate_transform_sift(ref_img, mov_img, scale=scale+0.1, refine_estimate=False)
    else:
        return M, output_shape, ref_offset, robust_estimate