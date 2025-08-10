import sys
from pathlib import Path

import pytest


# Ensure project root on path
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _first_existing_grib_file(datadir: Path, date_time_candidates):
    from prereise.gather.winddata.hrrr.helpers import formatted_filename

    for dt in date_time_candidates:
        for hours_forecasted in ("0", "1"):
            rel = formatted_filename(dt, hours_forecasted=hours_forecasted)
            fp = datadir / rel
            if fp.exists():
                return fp
    return None


def test_open_one_grib_hour():
    import pygrib
    from config_unified import DEFAULT_CONFIG
    from prereise.gather.const import get_grib_data_path

    datadir = Path(get_grib_data_path())
    assert datadir.exists(), f"GRIB data path does not exist: {datadir}"

    # Try configured start date and a couple of adjacent hours to increase hit rate
    start = DEFAULT_CONFIG.start_date
    candidates = [start]
    try:
        from datetime import timedelta

        candidates.extend([start + timedelta(hours=1), start - timedelta(hours=1)])
    except Exception:
        pass

    grib_fp = _first_existing_grib_file(datadir, candidates)
    if grib_fp is None:
        pytest.skip("No GRIB file found for candidate hours; skipping fast GRIB smoke test")

    # Open and read just the first message
    with pygrib.open(str(grib_fp)) as grbs:
        first = next(iter(grbs))
        assert getattr(first, "name", None) is not None


