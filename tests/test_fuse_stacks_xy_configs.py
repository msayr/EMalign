import importlib
import sys
import types


def _install_import_stubs():
    cv2 = types.ModuleType('cv2')
    cv2.error = Exception
    cv2.warpAffine = lambda img, *args, **kwargs: img
    sys.modules['cv2'] = cv2

    connectomics = types.ModuleType('connectomics')
    connectomics.common = types.ModuleType('connectomics.common')
    connectomics.common.bounding_box = types.SimpleNamespace(BoundingBox=object)
    sys.modules.setdefault('connectomics', connectomics)
    sys.modules.setdefault('connectomics.common', connectomics.common)
    sys.modules.setdefault('connectomics.common.bounding_box', connectomics.common.bounding_box)

    sofima = types.ModuleType('sofima')
    sofima.flow_field = types.SimpleNamespace(JAXMaskedXCorrWithStatsCalculator=object)
    sofima.flow_utils = types.SimpleNamespace(clean_flow=lambda flow, **kwargs: flow)
    sofima.mesh = types.SimpleNamespace(IntegrationConfig=object, relax_mesh=lambda *args, **kwargs: (args[0], None, None))
    sofima.warp = types.SimpleNamespace(ndimage_warp=lambda *args, **kwargs: args[0])
    sys.modules.setdefault('sofima', sofima)
    sys.modules.setdefault('sofima.flow_field', sofima.flow_field)
    sys.modules.setdefault('sofima.flow_utils', sofima.flow_utils)
    sys.modules.setdefault('sofima.mesh', sofima.mesh)
    sys.modules.setdefault('sofima.warp', sofima.warp)

    tensorstore = types.ModuleType('tensorstore')
    tensorstore.bool = bool
    tensorstore.uint8 = int
    tensorstore.dtype = type
    tensorstore.TensorStore = object
    sys.modules.setdefault('tensorstore', tensorstore)


def _module():
    _install_import_stubs()
    return importlib.import_module('emalign.scripts.fuse_stacks_xy')


def test_is_fuse_config_accepts_required_fields():
    module = _module()

    assert module.is_fuse_config({
        'zmin': 0,
        'zmax': 10,
        'z_offsets': [0],
        'dataset_paths': ['/tmp/stack.zarr'],
    })


def test_is_fuse_config_rejects_diagnostics_json():
    module = _module()

    assert not module.is_fuse_config({
        'fallback_group_count': 1,
        'groups': [],
    })
