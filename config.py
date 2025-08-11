#!/usr/bin/env python3
"""
Configuration file for HRRR data extraction.
Modify these settings to change output directories and other parameters.
"""

import os

# =============================================================================
# OUTPUT DIRECTORY CONFIGURATION
# =============================================================================

# Base output directory for all extracted data
OUTPUT_BASE_DIR = "./extracted_data"

# Subdirectories for different types of data
WIND_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "wind")
SOLAR_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "solar")
FULL_GRID_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "full_grid")

# =============================================================================
# DATA SOURCE CONFIGURATION
# =============================================================================

# GRIB data directory paths (auto-detected by OS)
DEFAULT_WINDOWS_DATA_DIR = "../../data/hrrr"  # Windows: ./data/YYYYMMDD/
DEFAULT_LINUX_DATA_DIR = "/research/alij/hrrr"  # Linux: /research/alij/hrrr/YYYYMMDD/

# Default forecast hours to process (f00 and f01 files)
DEFAULT_HOURS_FORECASTED = ["0", "1"]

# Default compression for parquet files
DEFAULT_COMPRESSION = "snappy"

# =============================================================================
# PROCESSING CONFIGURATION
# =============================================================================

# Default number of workers for parallel processing
DEFAULT_NUM_WORKERS = None  # Auto-detect if None

# Default maximum file groups to process
DEFAULT_MAX_FILE_GROUPS = 50000  # Increased for 256GB system

# Enable parallel processing by default
DEFAULT_USE_PARALLEL = True

# Enable resume functionality by default
DEFAULT_ENABLE_RESUME = True

# =============================================================================
# VARIABLE SELECTORS
# =============================================================================

# Default wind variables to extract
DEFAULT_WIND_SELECTORS = {
    "UWind80": "U component of wind",  # U component of wind at 80m
    "VWind80": "V component of wind",  # V component of wind at 80m
}

# Default solar variables to extract
DEFAULT_SOLAR_SELECTORS = {
    "rad": "Mean surface downward short-wave radiation flux",  # Mean surface downward short-wave radiation flux
    "vbd": "Visible Beam Downward Solar Flux",  # Visible Beam Downward Solar Flux
    "vdd": "Visible Diffuse Downward Solar Flux",  # Visible Diffuse Downward Solar Flux
    "2tmp": "2 metre temperature",  # 2 metre temperature
    "UWind10": "10 metre U wind component",  # 10 metre U wind component
    "VWind10": "10 metre V wind component",  # 10 metre V wind component
}

# =============================================================================
# FILE PATHS
# =============================================================================

# Default CSV file paths
DEFAULT_WIND_CSV_PATH = "wind.csv"
DEFAULT_SOLAR_CSV_PATH = "solar.csv"

# =============================================================================
# DATE RANGES
# =============================================================================

# Default date range for testing (2 hours)
DEFAULT_TEST_START = "2023-01-01 00:00:00"
DEFAULT_TEST_END = "2023-01-01 02:00:00"

# Default date range for full extraction (entire year)
DEFAULT_FULL_START = "2023-01-01 00:00:00"
DEFAULT_FULL_END = "2023-12-31 23:00:00"

# Default date range for month extraction
DEFAULT_MONTH_START = "2023-01-01 00:00:00"
DEFAULT_MONTH_END = "2023-01-31 23:00:00"

# Default date range for week extraction
DEFAULT_WEEK_START = "2023-01-01 00:00:00"
DEFAULT_WEEK_END = "2023-01-07 23:00:00"

# Default date range for day extraction
DEFAULT_DAY_START = "2023-01-01 00:00:00"
DEFAULT_DAY_END = "2023-01-01 23:00:00"
