"""
Prereise Essentials - Consolidated essential functions from the prereise module.

This file contains only the functions that are actually used in the codebase,
eliminating the need for the entire 45MB prereise folder.
"""

import os
import sys
from typing import List


# Constants from prereise.gather.winddata.hrrr.constants
DEFAULT_PRODUCT = "subh"
DEFAULT_HOURS_FORECASTED = "0"


def get_grib_data_path():
    """Automatically detect the appropriate GRIB data path based on OS."""
    if sys.platform.startswith("win"):
        # Windows paths to try
        windows_paths = [
            r"../../data/hrrr",
        ]

        for path in windows_paths:
            if os.path.exists(path):
                # Check if it has any content
                try:
                    items = os.listdir(path)
                    if items:
                        return path
                except Exception:
                    continue

        return r"data"  # Default fallback
    else:
        # Linux paths to try - prioritize the actual data directory
        linux_paths = [
            "/research/alij/hrrr",  # Primary path for full dataset
            "/local/alij/hrrr",  # Alternative path
            "./data",  # Local test data
            "data",  # Local test data (relative path)
        ]

        for path in linux_paths:
            if os.path.exists(path):
                try:
                    items = os.listdir(path)
                    if items:
                        print(f"✅ Found GRIB data at: {path}")
                        return path
                except Exception as e:
                    print(f"⚠️  Error checking {path}: {e}")
                    continue

        # If no paths found, return the primary path for Linux
        print(
            f"⚠️  No GRIB data found in expected locations, using default: /research/alij/hrrr"
        )
        return "/research/alij/hrrr"  # Default fallback for Linux


def formatted_filename(
    dt, product=DEFAULT_PRODUCT, hours_forecasted=DEFAULT_HOURS_FORECASTED
):
    """Deterministically returns a grib filename

    :param datetime.datetime dt: datetime associated with
        the data being stored
    :param string product: product associated with the
        data being stored
    :param string hours_forecasted: how many hours into
        the future the data is forecasted

    :return: (*str*) -- a filename
    """
    hours_forecasted = hours_forecasted.zfill(2)
    return f"{dt.strftime('%Y%m%d')}/hrrr.t{dt.strftime('%H')}z.wrf{product}f{hours_forecasted}.grib2"


def get_indices_that_contain_selector(input_list: List[str], selectors: List[str]) -> List[int]:
    """Generates list of indices of strings in input_list that
    contain a string inside of selectors

    :param list input_list: list of strings
    :param list selectors: list of strings

    :return: (*list*) -- list of indices of strings in input_list
        that contain a string inside of selectors
    """
    return [
        i
        for i, item in enumerate(input_list)
        if any([selector in item for selector in selectors])
    ]


# Note: extract_full_grid_day_by_day is a very large function (250+ lines)
# that would need to be copied separately if needed, or we can keep
# the import from the original location for that specific function.
