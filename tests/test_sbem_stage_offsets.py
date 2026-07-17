import numpy as np
from emalign.align_xy.stitch_ongrid import _apply_tile_origin_offsets


def test_apply_tile_origin_offsets_converts_origins_to_sofima_offsets():
    tile_map = {
        (0, 0): np.zeros((100, 200), dtype=np.uint8),
        (1, 0): np.zeros((100, 200), dtype=np.uint8),
        (0, 1): np.zeros((100, 200), dtype=np.uint8),
    }
    cx = np.zeros((2, 1, 2, 2), dtype=float)
    cy = np.zeros((2, 1, 2, 2), dtype=float)
    tile_origins = {
        (0, 0): (0, 0),
        (1, 0): (3, 180),
        (0, 1): (90, -4),
    }

    cx, cy = _apply_tile_origin_offsets(cx, cy, tile_map, tile_origins)

    assert cx[0, 0, 0, 0] == -20
    assert cx[1, 0, 0, 0] == 3
    assert cy[0, 0, 0, 0] == -4
    assert cy[1, 0, 0, 0] == -10

