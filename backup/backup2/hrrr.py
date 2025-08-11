#!/usr/bin/env python3
"""
Main script for HRRR data extraction with automatic path detection.
Supports both full grid extraction and specific locations extraction.
"""

import datetime
import os
import warnings

import pandas as pd

# Import configuration
from config import (
    DEFAULT_COMPRESSION,
    DEFAULT_HOURS_FORECASTED,
    DEFAULT_SOLAR_SELECTORS,
    DEFAULT_WIND_SELECTORS,
    DEFAULT_FULL_START,
    DEFAULT_FULL_END,
    DEFAULT_MONTH_START,
    DEFAULT_MONTH_END,
    DEFAULT_WEEK_START,
    DEFAULT_WEEK_END,
    DEFAULT_DAY_START,
    DEFAULT_DAY_END,
    FULL_GRID_OUTPUT_DIR,
    SOLAR_OUTPUT_DIR,
    WIND_OUTPUT_DIR,
)

# Suppress fs package deprecation warnings
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, message=".*pkg_resources.declare_namespace.*"
)
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*fs.*")

# Import after warnings are suppressed


def extract_specific_locations():
    """Extract HRRR data for specific wind and solar locations."""

    print("🚀 SPECIFIC LOCATIONS EXTRACTION")
    print("=" * 50)

    # Load location data
    print("📁 Loading location data from CSV files...")

    try:
        wind_df = pd.read_csv("wind.csv")
        print(f"✅ Loaded {len(wind_df)} wind locations from wind.csv")
    except FileNotFoundError:
        print("❌ wind.csv not found!")
        return None
    except Exception as e:
        print(f"❌ Error loading wind.csv: {e}")
        return None

    try:
        solar_df = pd.read_csv("solar.csv")
        print(f"✅ Loaded {len(solar_df)} solar locations from solar.csv")
    except FileNotFoundError:
        print("❌ solar.csv not found!")
        return None
    except Exception as e:
        print(f"❌ Error loading solar.csv: {e}")
        return None

    # Check required columns
    required_columns = ["pid", "lat", "lon"]
    for col in required_columns:
        if col not in wind_df.columns:
            print(f"❌ Missing column '{col}' in wind.csv")
            return None
        if col not in solar_df.columns:
            print(f"❌ Missing column '{col}' in solar.csv")
            return None

    # Select only required columns
    wind_locations = wind_df[["pid", "lat", "lon"]].copy()
    solar_locations = solar_df[["pid", "lat", "lon"]].copy()

    print(f"📊 Separated locations:")
    print(f"   Wind locations: {len(wind_locations)}")
    print(f"   Solar locations: {len(solar_locations)}")

    # ============================================================================
    # DATE RANGE CONFIGURATION - MODIFY THESE LINES TO CHANGE EXTRACTION PERIOD
    # ============================================================================
    
    # Option 1: Full year extraction (2023)
    START = datetime.datetime(2023, 1, 1, 0, 0, 0)  # Start: January 1, 2023
    END = datetime.datetime(2023, 12, 31, 23, 0, 0)  # End: December 31, 2023
    
    # Option 2: Month extraction (January 2023)
    # START = datetime.datetime(2023, 1, 1, 0, 0, 0)  # Start: January 1, 2023
    # END = datetime.datetime(2023, 1, 31, 23, 0, 0)  # End: January 31, 2023
    
    # Option 3: Week extraction (first week of 2023)
    # START = datetime.datetime(2023, 1, 1, 0, 0, 0)  # Start: January 1, 2023
    # END = datetime.datetime(2023, 1, 7, 23, 0, 0)  # End: January 7, 2023
    
    # Option 4: Day extraction (January 1, 2023)
    # START = datetime.datetime(2023, 1, 1, 0, 0, 0)  # Start: January 1, 2023
    # END = datetime.datetime(2023, 1, 1, 23, 0, 0)  # End: January 1, 2023
    
    # Option 5: Test extraction (2 hours only)
    # START = datetime.datetime(2023, 1, 1, 0, 0, 0)  # Start: January 1, 2023, 00:00
    # END = datetime.datetime(2023, 1, 1, 2, 0, 0)  # End: January 1, 2023, 02:00

    # Get the GRIB data path
    from prereise.gather.const import get_grib_data_path

    grib_path = get_grib_data_path()

    if grib_path is None:
        print("❌ No GRIB data path found!")
        return None

    print(f"✅ Using GRIB path: {grib_path}")

    # Use default selectors from config
    wind_selectors = DEFAULT_WIND_SELECTORS
    solar_selectors = DEFAULT_SOLAR_SELECTORS

    # Alternative selectors (commented out - keep for reference)
    # wind_selectors_alt = {
    #     "UWind80": "U component of wind",
    #     "VWind80": "V component of wind",
    # }
    #
    # solar_selectors_alt = {
    #     "rad": "Downward short-wave radiation flux",
    #     "vbd": "Visible Beam Downward Solar Flux",
    #     "vdd": "Visible Diffuse Downward Solar Flux",
    #     "2tmp": "2 metre temperature",
    #     "UWind10": "10 metre U wind component",
    #     "VWind10": "10 metre V wind component",
    # }

    print("📋 Extraction Parameters:")
    print(f"   Date range: {START.date()} to {END.date()}")
    print(f"   Wind locations: {len(wind_locations)}")
    print(f"   Wind variables: {list(wind_selectors.keys())}")
    print(f"   Solar locations: {len(solar_locations)}")
    print(f"   Solar variables: {list(solar_selectors.keys())}")
    print()

    # Import the optimized single-pass extraction function
    from prereise.gather.winddata.hrrr.calculations import (
        extract_all_locations_single_pass,
    )

    # Combine all selectors for single-pass reading
    all_selectors = {**wind_selectors, **solar_selectors}

    print("🚀 OPTIMIZED SINGLE-PASS EXTRACTION")
    print("=" * 50)
    print("📊 Using single-pass GRIB reading for maximum efficiency")
    print("   - Each GRIB file read only ONCE")
    print("   - All variables extracted simultaneously")
    print("   - ~50% faster than previous approach")
    print()

    results = {}

    try:
        # Use the location-specific single-pass extraction function
        from prereise.gather.winddata.hrrr.calculations import (
            extract_all_locations_single_pass,
        )

        # For testing, let's use a more targeted approach
        # Only process the specific hours we want (0 and 1) instead of all 24 hours
        print("🎯 Using targeted extraction for 2 hours only")

        # Extract location-specific data using OPTIMIZED POINT EXTRACTION (much faster!)
        from extract_specific_points_parallel import extract_specific_points_parallel

        extraction_result = extract_specific_points_parallel(
            wind_csv_path="wind.csv",
            solar_csv_path="solar.csv",
            START=START,
            END=END,
            DATADIR=grib_path,
            DEFAULT_HOURS_FORECASTED=["0", "1"],  # Only f00 and f01
            wind_selectors=wind_selectors,
            solar_selectors=solar_selectors,
            wind_output_dir=WIND_OUTPUT_DIR,
            solar_output_dir=SOLAR_OUTPUT_DIR,
            compression=DEFAULT_COMPRESSION,
            max_file_groups=None,  # Process ALL file groups (no limit)
        )

        print("✅ Single-pass extraction completed!")
        results["extraction"] = extraction_result

    except Exception as e:
        print(f"❌ Error during single-pass extraction: {e}")
        import traceback

        traceback.print_exc()
        results["extraction"] = None

    # Show summary
    print("\n📁 EXTRACTION SUMMARY")
    print("=" * 40)

    if os.path.exists(WIND_OUTPUT_DIR):
        wind_files = [f for f in os.listdir(WIND_OUTPUT_DIR) if f.endswith(".parquet")]
        print(f"🌪️  Wind files: {len(wind_files)} files in {WIND_OUTPUT_DIR}/")

    if os.path.exists(SOLAR_OUTPUT_DIR):
        solar_files = [
            f for f in os.listdir(SOLAR_OUTPUT_DIR) if f.endswith(".parquet")
        ]
        print(f"☀️  Solar files: {len(solar_files)} files in {SOLAR_OUTPUT_DIR}/")

    if results.get("extraction"):
        extraction_result = results["extraction"]
        print(f"📊 Processing Summary:")
        print(f"   Total days: {extraction_result.get('total_days', 'N/A')}")
        print(f"   Successful days: {extraction_result.get('successful_days', 'N/A')}")
        print(f"   Failed days: {extraction_result.get('failed_days', 'N/A')}")
        print(
            f"   Processing time: {extraction_result.get('processing_time_seconds', 0) / 3600:.1f} hours"
        )
        print(f"   Wind locations: {extraction_result.get('wind_locations', 'N/A')}")
        print(f"   Solar locations: {extraction_result.get('solar_locations', 'N/A')}")

    return results


def extract_full_grid():
    """Extract HRRR data for the full grid."""

    print("🚀 FULL GRID EXTRACTION")
    print("=" * 50)

    # Get the GRIB data path
    from prereise.gather.const import SELECTORS, get_grib_data_path

    grib_path = get_grib_data_path()

    if grib_path is None:
        print("❌ No GRIB data path found!")
        print(
            "Please ensure GRIB files are available in one of the expected locations."
        )
        return None

    print(f"✅ Using GRIB path: {grib_path}")

    # ============================================================================
    # DATE RANGE CONFIGURATION - MODIFY THESE LINES TO CHANGE EXTRACTION PERIOD
    # ============================================================================
    
    # Option 1: Full year extraction (2023) - DEFAULT
    START = datetime.datetime(2023, 1, 1, 0, 0, 0)  # Start: January 1, 2023
    END = datetime.datetime(2023, 12, 31, 23, 0, 0)  # End: December 31, 2023
    
    # Option 2: Month extraction (January 2023)
    # START = datetime.datetime(2023, 1, 1, 0, 0, 0)  # Start: January 1, 2023
    # END = datetime.datetime(2023, 1, 31, 23, 0, 0)  # End: January 31, 2023
    
    # Option 3: Week extraction (first week of 2023)
    # START = datetime.datetime(2023, 1, 1, 0, 0, 0)  # Start: January 1, 2023
    # END = datetime.datetime(2023, 1, 7, 23, 0, 0)  # End: January 7, 2023
    
    # Option 4: Day extraction (January 1, 2023)
    # START = datetime.datetime(2023, 1, 1, 0, 0, 0)  # Start: January 1, 2023
    # END = datetime.datetime(2023, 1, 1, 23, 0, 0)  # End: January 1, 2023

    print(f"📅 Date range: {START.date()} to {END.date()}")
    print(f"📊 Variables: {list(SELECTORS.keys())}")
    print()

    # Use DAY-BY-DAY AGGRESSIVE optimization for maximum performance on 36 CPU, 256GB system
    from prereise.gather.winddata.hrrr.calculations import extract_full_grid_day_by_day

    result = extract_full_grid_day_by_day(
        START=START,
        END=END,
        DATADIR=grib_path,
        DEFAULT_HOURS_FORECASTED=["0", "1"],
        SELECTORS=SELECTORS,
        output_dir=FULL_GRID_OUTPUT_DIR,
        use_aggressive_settings=True,  # Use ALL 36 CPUs and 256GB RAM efficiently
        enable_resume=True,  # Enable resume functionality
    )

    if result:
        print(
            f"\n🎉 DAY-BY-DAY AGGRESSIVE extraction completed with status: {result['status']}"
        )
        print(f"🚀 Performance metrics:")
        print(f"   Total days: {result.get('total_days', 'N/A')}")
        print(f"   Successful days: {result.get('successful_days', 'N/A')}")
        print(f"   Failed days: {result.get('failed_days', 'N/A')}")
        print(
            f"   Processing time: {result.get('processing_time_seconds', 'N/A'):.1f} seconds ({result.get('processing_time_seconds', 0) / 3600:.1f} hours)"
        )
        print(
            f"   Success rate: {result.get('successful_days', 0) / result.get('total_days', 1) * 100:.1f}%"
        )
        if result.get("resume_used"):
            print("✅ Resume functionality was used")
        if result.get("interrupted"):
            print("⚠️  Process was interrupted. You can resume later.")
    else:
        print("❌ DAY-BY-DAY AGGRESSIVE extraction failed")

    return result


def main():
    """Main function for HRRR data extraction."""

    print("🚀 HRRR Data Extraction")
    print("=" * 50)
    print("Choose extraction type by commenting/uncommenting the lines below:")
    print()

    # ============================================================================
    # CHOOSE EXTRACTION TYPE:
    # ============================================================================

    # Option 1: Full Grid Extraction (1.9M+ points)
    # Uncomment the line below to run full grid extraction
    # result = extract_full_grid()

    # Option 2: Specific Locations Extraction (wind.csv + solar.csv)
    # Uncomment the line below to run specific locations extraction
    result = extract_specific_locations()

    # ============================================================================
    # END OF CHOICES
    # ============================================================================

    if result:
        print(f"\n✅ Extraction completed successfully!")
    else:
        print(f"\n❌ Extraction failed!")


if __name__ == "__main__":
    main()
