from pathlib import Path

from emalign.io import sbem_image
from emalign.io.backend import get_io_backend


def _write_project(tmp_path):
    project = tmp_path / "sample"
    logs = project / "meta" / "logs"
    logs.mkdir(parents=True)
    (logs / "config_20240530.txt").write_text(
        "pixel_size = [4.0, 8.0]\n",
        encoding="utf-8",
    )
    (logs / "imagelist_20240530.txt").write_text(
        "\n".join(
            [
                r"tiles\g0000\t0000\sample_g0000_t0000_s00000.tif;-608632;-619022;0;0",
                r"tiles\g0000\t0001\sample_g0000_t0001_s00000.tif;-421332;-619022;0;0",
                r"tiles\g0001\t0000\sample_g0001_t0000_s00000.tif;-683129;-493211;0;0",
            ]
        ),
        encoding="utf-8",
    )

    for grid, tile in [(0, 0), (0, 1), (1, 0)]:
        tile_dir = project / "tiles" / f"g{grid:04d}" / f"t{tile:04d}"
        tile_dir.mkdir(parents=True)
        for z in [0, 1]:
            (tile_dir / f"sample_g{grid:04d}_t{tile:04d}_s{z:05d}.tif").touch()

    return project


def _reset_cache():
    sbem_image._TILE_YX_POS = {}
    sbem_image._TILE_YX_SOURCE = None
    sbem_image._TILE_YX_PROJECT_ROOT = None


def test_backend_registration_exposes_sbem_image():
    backend = get_io_backend("sbem_image")

    assert backend is sbem_image
    assert backend.FILE_EXT == ".tif"


def test_get_tileset_resolution_reads_grid_pixel_size(tmp_path):
    project = _write_project(tmp_path)

    assert sbem_image.get_tileset_resolution(
        str(project / "tiles" / "g0000" / "t0000")
    ) == (str(project / "tiles" / "g0000" / "t0000"), (4, 4))
    assert sbem_image.get_tileset_resolution(
        str(project / "tiles" / "g0001" / "t0000")
    ) == (str(project / "tiles" / "g0001" / "t0000"), (8, 8))


def test_get_tilesets_filters_by_grid_resolution(tmp_path):
    project = _write_project(tmp_path)

    assert sbem_image.get_tilesets(str(project), (4, 4), [], 1) == [
        str(project / "tiles" / "g0000" / "t0000"),
        str(project / "tiles" / "g0000" / "t0001"),
    ]
    assert sbem_image.get_tilesets(str(project), (8, 8), [], 1) == [
        str(project / "tiles" / "g0001" / "t0000"),
    ]


def test_parse_tile_position_uses_imagelist_tile_key_for_all_slices(tmp_path):
    _reset_cache()
    project = _write_project(tmp_path)

    upper_slice = (
        project / "tiles" / "g0000" / "t0000" / "sample_g0000_t0000_s00001.tif"
    )
    other_tile = (
        project / "tiles" / "g0001" / "t0000" / "sample_g0001_t0000_s00001.tif"
    )

    assert sbem_image.parse_yx_pos_from_name(upper_slice) == (0, 1)
    assert sbem_image.parse_yx_pos_from_name(other_tile) == (1, 0)
    assert (
        sbem_image.parse_slice_from_name(
            Path(r"tiles\g0000\t0000\sample_g0000_t0000_s00001.tif")
        )
        == 1
    )
