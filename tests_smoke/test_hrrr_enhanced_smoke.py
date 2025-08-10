import importlib
import sys
from pathlib import Path


# Ensure project root is on sys.path for import
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def test_import_module():
    mod = importlib.import_module("hrrr_enhanced")
    assert hasattr(mod, "extract_specific_locations_enhanced")
    assert hasattr(mod, "extract_full_grid_enhanced")
    assert hasattr(mod, "extract_region_data_enhanced")


