import sys
import types

import numpy as np

# The test environment may not provide OpenCV's native libGL dependency. Stub the
# small import surface needed while testing render's canvas guard.
sys.modules.setdefault('cv2', types.SimpleNamespace(error=Exception))
sofima = types.ModuleType('sofima')
sofima.warp = types.SimpleNamespace(render_tiles=None)
sys.modules.setdefault('sofima', sofima)
sys.modules.setdefault('sofima.warp', sofima.warp)

from emalign.align_xy import render


class DummyDestination:
    pass


def test_render_slice_xy_rejects_disproportionate_canvas(monkeypatch):
    tile_map = {
        (0, 0): np.ones((10, 10), dtype=np.uint8),
        (1, 0): np.ones((10, 10), dtype=np.uint8),
    }
    meshes = {key: np.zeros((2, 1, 1, 1), dtype=np.float32) for key in tile_map}

    def fake_render_tiles(*args, **kwargs):
        stitched = np.ones((10, 200), dtype=np.uint8)
        mask = np.ones_like(stitched, dtype=bool)
        warped_tiles = {
            (0, 0): (0, 0, np.ones((10, 10), dtype=np.uint8)),
            (1, 0): (190, 0, np.ones((10, 10), dtype=np.uint8)),
        }
        return stitched, mask, warped_tiles

    monkeypatch.setattr(render.warp, 'render_tiles', fake_render_tiles)
    monkeypatch.setattr(render, 'check_stitch', lambda warped_tiles, margin: [1.0])
    writes = []
    monkeypatch.setattr(render, 'write_data', lambda *args, **kwargs: writes.append(args) or (args[0], None))

    destination, stitch_score = render.render_slice_xy(
        DummyDestination(),
        0,
        tile_map,
        meshes,
        stride=10,
        resize_canvas=True,
        min_stitch_score=0.8,
    )

    assert isinstance(destination, DummyDestination)
    assert stitch_score == [0.0]
    assert writes == []


def test_render_slice_xy_allows_canvas_within_grid_limit(monkeypatch):
    tile_map = {
        (0, 0): np.ones((10, 10), dtype=np.uint8),
        (1, 0): np.ones((10, 10), dtype=np.uint8),
    }
    meshes = {key: np.zeros((2, 1, 1, 1), dtype=np.float32) for key in tile_map}

    def fake_render_tiles(*args, **kwargs):
        stitched = np.ones((10, 20), dtype=np.uint8)
        mask = np.ones_like(stitched, dtype=bool)
        warped_tiles = {
            (0, 0): (0, 0, np.ones((10, 10), dtype=np.uint8)),
            (1, 0): (10, 0, np.ones((10, 10), dtype=np.uint8)),
        }
        return stitched, mask, warped_tiles

    monkeypatch.setattr(render.warp, 'render_tiles', fake_render_tiles)
    monkeypatch.setattr(render, 'check_stitch', lambda warped_tiles, margin: [1.0])
    writes = []
    monkeypatch.setattr(render, 'write_data', lambda *args, **kwargs: writes.append(args) or (args[0], None))

    _, stitch_score = render.render_slice_xy(
        DummyDestination(),
        0,
        tile_map,
        meshes,
        stride=10,
        resize_canvas=True,
        min_stitch_score=0.8,
    )

    assert stitch_score == [1.0]
    assert len(writes) == 1
