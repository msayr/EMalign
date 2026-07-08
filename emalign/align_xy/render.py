import numpy as np
import warnings

from sofima import warp

from emalign.io.process.mask import mask_to_bbox

from .utils import check_stitch
from ..io.store import write_data


def _max_expected_canvas_shape(tile_map, max_canvas_scale):
    '''Return the largest acceptable stitched YX shape for an on-grid tile map.'''
    tile_keys = np.array(list(tile_map.keys()))
    tile_space = np.array([tile_keys[:, 1].max() + 1, tile_keys[:, 0].max() + 1])
    max_tile_shape = np.max([tile.shape for tile in tile_map.values()], axis=0)
    return np.ceil(tile_space * max_tile_shape * max_canvas_scale).astype(int)


def render_slice_xy(destination,
                    z,
                    tile_map,
                    meshes,
                    stride,
                    tile_masks=None,
                    parallelism=1,
                    margin=50,
                    dest_mask=None,
                    return_render=False,
                    resize_canvas=True,
                    min_stitch_score=0,
                    max_canvas_scale=1.25,
                    **kwargs):
    '''Render an aligned image from a tile map.

    Use a tile_map and corresponding meshes to produce an aligned image and mask. 
    Overlaps are assessed with check_stitch to produce a stitch_score that will be logged to find flawed slices.
    The score is based on a laplacian filter, between 0 (no similarity) and 1 (exact match).

    Args:
        destination (`tensorstore.TensorStore`): Zarr store where to write aligned slice.
        z (int): Z index at which to write the slice (axis at first position).
        tile_map (dict of `np.ndarray`): Dictionary from [x,y] tile position to [y,x] image.
        meshes (dict of `np.ndarray`): Dictionary from [x,y] tile position to [2, z, x, y] array of mesh positions. Order of keys determines the order of render.
        stride (int): Step used to determine mesh node positions.
        tile_masks (dict of `np.ndarray`, optional): Dictionary from [x,y] tile position to [y,x] boolean masks corresponding to tile_map. Defaults to None.
        parallelism (int, optional): Number of threads used by warp.render_tiles to warp tiles in parallel (max one thread per tile). Defaults to 1.
        margin (int, optional): Number of pixels cropped from each tile's boundaries to remove artifacts from deformation. Defaults to 50.
        dest_mask (_type_, optional): Zarr store where to write aligned slice's mask. Defaults to None.
        return_render (bool, optional): Whether to return the aligned image rather than writing it. Defaults to False.
        resize_canvas (bool, optional): Whether the image to the size of a bounding box defined by the mask. Defaults to True.
        max_canvas_scale (float or None, optional): Maximum allowed rendered canvas size as a multiple of the
            tile grid's unaligned bounding box. If the alignment moves tiles so far that the cropped mask bounding
            box exceeds this limit, the stitch is treated as failed and no data is written. Set to None to disable.
        **kwargs (optional): Additional arguments passed to warp.render_tiles. 
            e.g.: margin_overrides provides specific margins per direction per tile.

    Returns:
        int: 
            If return_render == False (Default): stitch score describing how well overlaps match, between 0 and 1 as defined by check_stitch. 
            If return_render == True: tuple of: aligned image, stitch score.
    '''

    if len(tile_map) > 1:
        # warp.render_tiles only uses workers to distribute tiles
        parallelism = min(len(tile_map.keys()), parallelism)

        # Render stitched image
        stitched, mask, warped_tiles = warp.render_tiles(tile_map, meshes, 
                                                    tile_masks=tile_masks, 
                                                    parallelism=parallelism, 
                                                    stride=(stride, stride), 
                                                    return_warped_tiles=True,
                                                    margin=margin,
                                                    **kwargs)
        # Evaluate overlap
        stitch_score = check_stitch(warped_tiles, margin)
    else:
        stitched = list(tile_map.values())[0]
        mask = np.ones_like(list(tile_map.values())[0]).astype(bool)
        stitch_score = 1
    
    if resize_canvas:
        y1,y2,x1,x2 = mask_to_bbox(mask)
        stitched = stitched[y1:y2,x1:x2]
        mask = mask[y1:y2,x1:x2]

    if max_canvas_scale is not None:
        max_shape = _max_expected_canvas_shape(tile_map, max_canvas_scale)
        if np.any(np.array(stitched.shape) > max_shape):
            warnings.warn(
                'Rendered XY canvas exceeds allowed size; treating stitch as failed. '
                f'Got YX shape {stitched.shape}, maximum allowed {tuple(map(int, max_shape))}.'
            )
            stitch_score = np.zeros_like(np.atleast_1d(stitch_score), dtype=float).tolist()

    if return_render:
        return stitched, stitch_score
    elif np.min(stitch_score) > min_stitch_score:
        # Stitch good enough, write data
        destination, _ = write_data(destination, stitched, z)

        if dest_mask is not None:
            dest_mask, _ = write_data(dest_mask, mask, z)
            return destination, dest_mask, stitch_score
        return destination, stitch_score
    else:
        # Bad stitch, don't write data
        if dest_mask is not None:
            return destination, dest_mask, stitch_score
        return destination, stitch_score
