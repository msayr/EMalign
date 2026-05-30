import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_prep_config_xy_direct_script_help():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / 'emalign' / 'prep_config_xy.py'), '--help'],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert '--resolution' in result.stdout
    assert '--mode' in result.stdout


def test_prep_config_xy_module_help():
    result = subprocess.run(
        [sys.executable, '-m', 'emalign.prep_config_xy', '--help'],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert '--resolution' in result.stdout
    assert '--mode' in result.stdout
