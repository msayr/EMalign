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
    assert 'max_canvas_scale' not in vars(args)


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


def test_create_configs_keeps_z_overlap_group_together_when_sift_is_disconnected(monkeypatch, tmp_path):
    _install_import_stubs()
    prep = importlib.import_module('emalign.align_xy.prep')

    class FakeKvStore:
        def __init__(self, path):
            self.path = path

    class FakeDataset:
        def __init__(self, path):
            self.kvstore = FakeKvStore(path)
            self.shape = (5, 4, 4)

    datasets = [
        FakeDataset('/tmp/output/xy_intermediate/01_g0000_t0000/'),
        FakeDataset('/tmp/output/xy_intermediate/01_g0000_t0001/'),
        FakeDataset('/tmp/output/xy_intermediate/01_g0008_t0000/'),
    ]

    monkeypatch.setattr(
        prep,
        'get_ordered_datasets',
        lambda *args, **kwargs: (datasets, prep.np.array([[10], [10], [10]])),
    )
    monkeypatch.setattr(prep, 'get_store_attributes', lambda ds: {'resolution': [50, 50]})
    monkeypatch.setattr(prep, 'find_ref_slice', lambda ds, z: (prep.np.zeros((4, 4)), None))
    monkeypatch.setattr(prep, 'resample', lambda img, scale: img)

    def fake_estimate_transform_sift(*args, **kwargs):
        return None, None, False, None

    monkeypatch.setattr(prep, 'estimate_transform_sift', fake_estimate_transform_sift)

    main_config_path = tmp_path / 'main_config.json'
    main_config_path.write_text('{"resolution": [50, 50]}')

    configs = prep.create_configs_fused_stacks(str(main_config_path))

    assert len(configs) == 1
    assert configs[0]['dataset_paths'] == [dataset.kvstore.path for dataset in datasets]
    assert configs[0]['z_offsets'] == [10, 10, 10]
    assert configs[0]['zmin'] == 10
    assert configs[0]['zmax'] == 15
