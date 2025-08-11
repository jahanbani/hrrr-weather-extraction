import os
import sys
from datetime import timedelta

import pandas as pd

# Map state abbreviations to state name
abv2state = {
    "AK": "Alaska",
    "AL": "Alabama",
    "AR": "Arkansas",
    "AZ": "Arizona",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "IA": "Iowa",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "MA": "Massachusetts",
    "MD": "Maryland",
    "ME": "Maine",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MO": "Missouri",
    "MS": "Mississippi",
    "MT": "Montana",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "NE": "Nebraska",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NV": "Nevada",
    "NY": "New York",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VA": "Virginia",
    "VT": "Vermont",
    "WA": "Washington",
    "WI": "Wisconsin",
    "WV": "West Virginia",
    "WY": "Wyoming",
}

SELECTORS = {
    "UWind80": "u",
    "VWind80": "v",
    "UWind10": "10u",
    "VWind10": "10v",
    "rad": "dswrf",
    "vbd": "vbdsf",
    "vdd": "vddsf",
    "2tmp": "2t",
}

OUTDIR = "./"

# study year; which year to study
YEAR = 2023
# Use a single day that we know exists
START = pd.to_datetime("2022-12-31 01:00")
END = pd.to_datetime("2023-02-01 01:00")
# START = pd.to_datetime("2019-12-31 01:00")
# END = pd.to_datetime("2021-01-02 00:00")

# Automatic path detection for Windows vs Linux


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


DATADIR = get_grib_data_path()

# print(f"start time is {START}")
# print(f"end time is {END}")
print(f"Using DATADIR: {DATADIR}")
SEARCHSTRING = "V[B,D]DSF|DSWRF|TMP:2 m|(?:U|V)GRD:(?:10|80) m"
TZ = 8
