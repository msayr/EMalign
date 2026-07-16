import importlib
import os
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


def test_build_parser_exposes_fusion_parameter_overrides():
    module = _module()

    args = module.build_parser().parse_args([
        '--config', '/tmp/main_config.json',
        '--scale', '0.25',
        '--patch-size', '96',
        '--stride', '24',
        '--img-on-top', '2',
    ])

    assert args.config_path == '/tmp/main_config.json'
    assert args.scale == 0.25
    assert args.patch_size == 96
    assert args.stride == 24
    assert args.img_on_top == '2'


def test_main_passes_fusion_parameter_overrides(monkeypatch):
    module = _module()
    captured = {}

    def fake_align_fused_stacks_xy(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(module, 'align_fused_stacks_xy', fake_align_fused_stacks_xy)
    monkeypatch.setattr(sys, 'argv', [
        'fuse_stacks_xy',
        '--config', '/tmp/main_config.json',
        '--scale', '0.2',
        '--patch-size', '128',
        '--stride', '32',
        '--img-on-top', '1',
    ])

    module.main()

    assert captured['config_path'] == '/tmp/main_config.json'
    assert captured['scale'] == 0.2
    assert captured['patch_size'] == 128
    assert captured['stride'] == 32
    assert captured['img_on_top'] == '1'


def test_thread_env_uses_core_count_without_overriding_existing_env(monkeypatch):
    module = _module()
    thread_env_names = [
        'OMP_NUM_THREADS',
        'OPENBLAS_NUM_THREADS',
        'MKL_NUM_THREADS',
        'NUMEXPR_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS',
        'XLA_FLAGS',
    ]
    for name in thread_env_names:
        monkeypatch.delenv(name, raising=False)

    module.configure_thread_env(2)

    assert os.environ['OMP_NUM_THREADS'] == '2'
    assert os.environ['OPENBLAS_NUM_THREADS'] == '2'
    assert os.environ['MKL_NUM_THREADS'] == '2'
    assert os.environ['NUMEXPR_NUM_THREADS'] == '2'
    assert os.environ['VECLIB_MAXIMUM_THREADS'] == '2'
    assert 'intra_op_parallelism_threads=2' in os.environ['XLA_FLAGS']

    monkeypatch.setenv('OMP_NUM_THREADS', '8')
    module.configure_thread_env(1)

    assert os.environ['OMP_NUM_THREADS'] == '8'


def test_early_core_parser_does_not_treat_config_flag_as_cores():
    module = _module()

    assert module._extract_cores_arg([
        '-cfg', '/tmp/main_config.json',
        '-c', '2',
    ]) == '2'
    assert module._extract_cores_arg([
        '-cfg', '/tmp/main_config.json',
        '--cores=3',
    ]) == '3'
    assert module._extract_cores_arg([
        '-cfg', '/tmp/main_config.json',
    ]) is None


def test_build_parser_exposes_max_overlap_limit():
    module = _module()

    args = module.build_parser().parse_args([
        '--config', '/tmp/main_config.json',
        '--max-overlap', '25',
    ])

    assert args.max_overlap_percent == 25


def test_main_passes_max_overlap_limit(monkeypatch):
    module = _module()
    captured = {}

    def fake_align_fused_stacks_xy(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(module, 'align_fused_stacks_xy', fake_align_fused_stacks_xy)
    monkeypatch.setattr(sys, 'argv', [
        'fuse_stacks_xy',
        '--config', '/tmp/main_config.json',
        '--max-overlap', '50',
    ])

    module.main()

    assert captured['max_overlap_percent'] == 50


def test_overlap_limit_rejects_excessive_overlap():
    _install_import_stubs()
    import numpy as np
    from emalign.align_xy.stitch_offgrid import OverlapLimitError, _raise_if_overlap_too_large

    existing = np.zeros((10, 10), dtype=bool)
    moving = np.zeros((10, 10), dtype=bool)
    existing[:, :5] = True
    moving[:, :8] = True

    try:
        _raise_if_overlap_too_large(existing, moving, 50)
    except OverlapLimitError:
        pass
    else:
        raise AssertionError('Expected OverlapLimitError for 62.5% overlap with a 50% limit')


def test_overlap_limit_allows_overlap_at_limit():
    _install_import_stubs()
    import numpy as np
    from emalign.align_xy.stitch_offgrid import _raise_if_overlap_too_large

    existing = np.zeros((10, 10), dtype=bool)
    moving = np.zeros((10, 10), dtype=bool)
    existing[:, :5] = True
    moving[:, :10] = True

    _raise_if_overlap_too_large(existing, moving, 50)
