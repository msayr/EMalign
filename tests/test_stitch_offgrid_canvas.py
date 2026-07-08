import sys
import types

import numpy as np
import pytest

sys.modules.setdefault('cv2', types.SimpleNamespace(error=Exception))

connectomics = types.ModuleType('connectomics')
connectomics.common = types.ModuleType('connectomics.common')
connectomics.common.bounding_box = types.SimpleNamespace(BoundingBox=object)
sys.modules.setdefault('connectomics', connectomics)
sys.modules.setdefault('connectomics.common', connectomics.common)
sys.modules.setdefault('connectomics.common.bounding_box', connectomics.common.bounding_box)

sofima = sys.modules.get('sofima', types.ModuleType('sofima'))
sofima.flow_field = types.SimpleNamespace(JAXMaskedXCorrWithStatsCalculator=object)
sofima.flow_utils = types.SimpleNamespace(clean_flow=lambda flow, **kwargs: flow)
sofima.mesh = types.SimpleNamespace(IntegrationConfig=object, relax_mesh=lambda *args, **kwargs: (args[0], None, None))
sofima.warp = getattr(sofima, 'warp', types.SimpleNamespace())
sofima.warp.ndimage_warp = lambda *args, **kwargs: args[0]
sys.modules['sofima'] = sofima
sys.modules.setdefault('sofima.flow_field', sofima.flow_field)
sys.modules.setdefault('sofima.flow_utils', sofima.flow_utils)
sys.modules.setdefault('sofima.mesh', sofima.mesh)
sys.modules['sofima.warp'] = sofima.warp

from emalign.align_xy.stitch_offgrid import CanvasSizeError, _raise_if_canvas_too_large


def test_fused_canvas_guard_rejects_runaway_shape():
    with pytest.raises(CanvasSizeError, match='affine warp'):
        _raise_if_canvas_too_large(
            candidate_shape=(100, 1000),
            img1_shape=np.array((100, 100)),
            img2_shape=np.array((100, 120)),
            max_canvas_scale=1.5,
            stage='affine warp',
        )


def test_fused_canvas_guard_allows_bounded_shape():
    _raise_if_canvas_too_large(
        candidate_shape=(120, 150),
        img1_shape=np.array((100, 100)),
        img2_shape=np.array((100, 120)),
        max_canvas_scale=1.5,
        stage='affine warp',
    )
