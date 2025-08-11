#!/usr/bin/env python3
"""
SINGLE-PASS Day-by-day HRRR data extraction for specific wind and solar locations.
This version reads ALL variables from each GRIB file in ONE pass for maximum efficiency.
"""

import datetime
import os
import time
import warnings
import gc
import re
import platform
import multiprocessing as mp
from functools import partial
from collections import defaultdict
from pathlib import Path

# Set multiprocessing start method for Linux compatibility
if platform.system() == "Linux":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # If already set, continue
        pass

# Optional import for system monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️  psutil not available - system monitoring disabled")

import numpy as np
import pandas as pd
import pygrib
from scipy.spatial import KDTree
from tqdm import tqdm

# Import configuration
from config import (
    WIND_OUTPUT_DIR,
    SOLAR_OUTPUT_DIR,
    DEFAULT_WIND_SELECTORS,
    DEFAULT_SOLAR_SELECTORS,
    DEFAULT_HOURS_FORECASTED,
    DEFAULT_COMPRESSION,
)

# Suppress fs package deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, 
                       message=".*pkg_resources.declare_namespace.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, 
                       message=".*fs.*")

# Suppress cryptography deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, 
                       message=".*TripleDES.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, 
                       message=".*cryptography.*")


def get_data_directory():
    """Get the appropriate data directory based on the operating system."""
    system = platform.system()
    if system == "Windows":
        return "data"
    elif system == "Linux":
        return "/research/alij/hrrr"
    else:
        return "/research/alij/hrrr"


# Global cache for grid coordinates to avoid recalculation
_GRID_CACHE = {}

def get_wind_data_lat_long_cached(dt, directory, hours_forecasted="0"):
    """Returns the latitude and longitudes with caching for performance."""
    date_str = dt.strftime("%Y%m%d")
    
    # Check cache first
    if date_str in _GRID_CACHE:
        return _GRID_CACHE[date_str]
    
    try:
        # Check if directory is already a date directory (e.g., data/20230101)
        # or if we need to create a date subdirectory (e.g., data/20230101/20230101)
        if os.path.basename(directory) == date_str:
            # Directory is already a date directory, use it directly
            search_dir = directory
        else:
            # Directory is a base directory, create date subdirectory
            search_dir = os.path.join(directory, date_str)
        
        if not os.path.exists(search_dir):
            print(f"No data directory found: {search_dir}")
            return None
        
        import glob
        grib_pattern = os.path.join(search_dir, "*.grib2")
        grib_files = glob.glob(grib_pattern)
        
        if not grib_files:
            print(f"No GRIB files found in {search_dir}")
            return None
        
        # Filter out subset files and index files which appear to be corrupted
        valid_files = [f for f in grib_files if "subset_" not in os.path.basename(f) and not f.endswith('.idx')]
        
        if not valid_files:
            print(f"No valid GRIB files found in {search_dir} (only subset/index files available)")
            return None
        
        # Use the first valid GRIB file to get grid coordinates
        grb_file = valid_files[0]
        grb = pygrib.open(grb_file)
        grb_message = grb.select(name="U component of wind")[0]
        
        lats, lons = grb_message.latlons()
        grb.close()
        
        # Cache the result
        _GRID_CACHE[date_str] = (lats, lons)
        
        return lats, lons
        
    except Exception as e:
        print(f"Error getting grid coordinates: {e}")
        return None


def find_closest_grid_points(points, grid_lats, grid_lons):
    """Find the closest grid points for given lat/lon coordinates."""
    if grid_lats is None or grid_lons is None:
        return None
    
    # Flatten the grid coordinates
    grid_points = np.column_stack([grid_lats.flatten(), grid_lons.flatten()])
    
    # Create KDTree for efficient nearest neighbor search
    tree = KDTree(grid_points)
    
    # Find closest points
    distances, indices = tree.query(points[['lat', 'lon']].values)
    
    # Convert back to 2D indices
    grid_shape = grid_lats.shape
    row_indices = indices // grid_shape[1]
    col_indices = indices % grid_shape[1]
    
    return list(zip(row_indices, col_indices))


def extract_values_for_points_vectorized(grb, point_indices, grid_lats, grid_lons):
    """Extract values for specific grid points from a GRIB message (vectorized)."""
    if point_indices is None:
        return None
    
    try:
        # Get the full grid data
        values_2d = grb.values
        
        # Extract values for the specific points (vectorized)
        rows = [row for row, col in point_indices]
        cols = [col for row, col in point_indices]
        
        # Use advanced indexing for better performance
        extracted_values = values_2d[rows, cols]
        
        # Convert to float32 to save memory
        return np.array(extracted_values, dtype=np.float32)
        
    except MemoryError as e:
        print(f"Memory error in extraction: {e}")
        return None
    except Exception as e:
        print(f"Error extracting values: {e}")
        return None
    finally:
        # Explicitly delete to free memory
        if 'values_2d' in locals():
            del values_2d


def group_grib_files_by_time(files):
    """Group GRIB files by their forecast time."""
    grouped = defaultdict(list)
    
    for file in files:
        filename = os.path.basename(file)
        
        # Handle Linux GRIB file naming patterns
        # Pattern 1: hrrr.t10z.wrfsubhf01.grib2 (hour 10, forecast 01)
        match = re.search(r'hrrr\.t(\d{2})z\.wrfsubhf(\d{2})\.grib2', filename)
        if match:
            hour = match.group(1)
            forecast = match.group(2)
            # Convert forecast to single digit: "00" -> "0", "01" -> "1"
            forecast_single = str(int(forecast))
            time_key = f"{hour}_{forecast_single}"
            grouped[time_key].append(file)
            continue
        
        # Pattern 2: hrrr.t10z.wrfsubhf01.grib2.47d85.idx (index files)
        match = re.search(r'hrrr\.t(\d{2})z\.wrfsubhf(\d{2})\.grib2\.\w+\.idx', filename)
        if match:
            # Skip index files
            continue
        
        # Pattern 3: Other patterns - try to extract hour from filename
        match = re.search(r'hrrr\.t(\d{2})z', filename)
        if match:
            hour = match.group(1)
            # Use a default forecast value
            forecast = "0"
            time_key = f"{hour}_{forecast}"
            grouped[time_key].append(file)
            continue
        
        # If no pattern matches, use the filename as the key
        grouped[filename].append(file)
    
    return grouped


def process_single_grib_file_single_pass(file_path, wind_selectors, solar_selectors, 
                                       wind_indices, solar_indices, grid_lats, grid_lons, day_date):
    """
    Process a single GRIB file and extract ALL variables in ONE pass.
    This uses the same approach as the existing implementation for correct timestamp handling.
    """
    
    wind_values = {}
    solar_values = {}
    timestamps = []
    
    try:
        # Direct extraction for specific points - no chunking needed
        import gc
        
        # Combine all selectors
        all_selectors = {**wind_selectors, **solar_selectors}
        
        with pygrib.open(file_path) as grbs:
            # Track which variables we've found for each time offset
            found_variables = {var_name: set() for var_name in all_selectors.keys()}
            
            # Single pass through all GRIB messages
            for grb in grbs:
                # Check if this message is for our target date
                try:
                    message_date = grb.validDate
                    if message_date.date() != day_date.date():
                        continue  # Skip messages from other dates
                except:
                    # If we can't get the date, continue processing
                    pass
                
                grb_name = grb.name
                
                # Check if this message contains any of our target variables
                for var_name, var_selector in all_selectors.items():
                    if grb_name == var_selector:
                        # Get base timestamp from GRIB message
                        base_timestamp = pd.Timestamp(
                            year=grb.year, month=grb.month, day=grb.day,
                            hour=grb.hour, minute=grb.minute
                        )
                        
                        # Check for time offsets in this message
                        grb_str = str(grb)
                        time_offsets_found = []
                        
                        # Look for all quarter-hourly intervals (00, 15, 30, 45 minutes)
                        for offset, minute in [(0, 0), (15, 15), (30, 30), (45, 45)]:
                             if offset == 0:
                                 # For 00 offset, check if it's the base time or if no offset is mentioned
                                 if "0 mins" in grb_str or "0m mins" in grb_str or "mins" not in grb_str:
                                     time_offsets_found.append((offset, minute))
                             else:
                                 # For 15, 30, 45-minute offsets, look for specific mentions
                                 if f"{offset}m mins" in grb_str or f"{offset} mins" in grb_str:
                                     time_offsets_found.append((offset, minute))
                        
                        # If no specific offsets found, assume it's the base time (00)
                        if not time_offsets_found:
                            time_offsets_found = [(0, 0)]
                        
                        # Process each time offset found
                        for offset, minute in time_offsets_found:
                            dt = base_timestamp + pd.Timedelta(minutes=minute)
                            
                            # Skip if we already have this variable at this time
                            if dt in found_variables[var_name]:
                                continue
                            
                            # Extract values directly for our specific points
                            values_2d = grb.values
                            
                            # Extract data for wind locations
                            if var_name in wind_selectors and wind_indices is not None:
                                wind_extracted = []
                                for idx in wind_indices:
                                    row, col = idx
                                    wind_extracted.append(values_2d[row, col])
                                
                                if var_name not in wind_values:
                                    wind_values[var_name] = {}
                                wind_values[var_name][dt] = np.array(wind_extracted, dtype=np.float32)
                                
                                if dt not in timestamps:
                                    timestamps.append(dt)
                            
                            # Extract data for solar locations
                            if var_name in solar_selectors and solar_indices is not None:
                                solar_extracted = []
                                for idx in solar_indices:
                                    row, col = idx
                                    solar_extracted.append(values_2d[row, col])
                                
                                if var_name not in solar_values:
                                    solar_values[var_name] = {}
                                solar_values[var_name][dt] = np.array(solar_extracted, dtype=np.float32)
                                
                                if dt not in timestamps:
                                    timestamps.append(dt)
                            
                            found_variables[var_name].add(dt)
                        
                        # Don't break - continue processing other messages for the same variable
                        # This allows us to find multiple time offsets in the same file
                
                # Force garbage collection after each message
                del grb
                gc.collect()
                
                # Don't exit early - continue processing all messages to find all variables
            
            # Report missing variables
            found_var_names = {var_name for var_name, timestamps in found_variables.items() if timestamps}
            missing_vars = set(all_selectors.keys()) - found_var_names
            if missing_vars:
                print(f"  Missing variables in {os.path.basename(file_path)}: {missing_vars}")
                
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        gc.collect()
        
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        # Return empty values for all variables
        for var_name in wind_selectors.keys():
            wind_values[var_name] = {}
        for var_name in solar_selectors.keys():
            solar_values[var_name] = {}
    
    return wind_values, solar_values, timestamps


def process_single_day_single_pass(day_date, wind_locations, solar_locations, data_dir, 
                                  wind_selectors, solar_selectors, wind_output_dir, solar_output_dir,
                                  compression="snappy"):
    """
    Process a single day of HRRR data using SINGLE-PASS GRIB reading.
    """
    
    print(f"🌅 Processing day: {day_date.strftime('%Y-%m-%d')}")
    start_time = time.time()
    
    # Get grid coordinates for this day (with caching)
    grid_coords = get_wind_data_lat_long_cached(day_date, data_dir)
    if grid_coords is None:
        return {
            'day': day_date.strftime('%Y-%m-%d'),
            'status': 'failed',
            'error': 'No grid coordinates available',
            'processing_time': 0
        }
    
    grid_lats, grid_lons = grid_coords
    
    # Find closest grid points for wind and solar locations
    wind_indices = find_closest_grid_points(wind_locations, grid_lats, grid_lons)
    solar_indices = find_closest_grid_points(solar_locations, grid_lats, grid_lons)
    
    # Get GRIB files for this day - PROPERLY HANDLE QUARTER-HOURLY DATA
    date_str = day_date.strftime("%Y%m%d")
    
    # Check if directory is already a date directory (e.g., data/20230101)
    # or if we need to create a date subdirectory (e.g., data/20230101/20230101)
    if os.path.basename(data_dir) == date_str:
        # Directory is already a date directory, use it directly
        search_dir = data_dir
    else:
        # Directory is a base directory, create date subdirectory
        search_dir = os.path.join(data_dir, date_str)
    
    if not os.path.exists(search_dir):
        return {
            'day': day_date.strftime('%Y-%m-%d'),
            'status': 'failed',
            'error': f'No data directory found: {search_dir}',
            'processing_time': 0
        }
    
    # Find ALL GRIB files for this day (including quarter-hourly)
    import glob
    # Look for files in the date-specific subdirectory
    grib_pattern = os.path.join(search_dir, "*.grib2")
    all_grib_files = glob.glob(grib_pattern)
    
    # Filter files for this specific date - use simpler approach
    valid_files = []
    for file_path in all_grib_files:
        filename = os.path.basename(file_path)
        # Skip subset files and index files
        if "subset_" in filename or filename.endswith('.idx'):
            continue
        
        # Accept all non-subset GRIB files in the date directory
        valid_files.append(file_path)
    
    if not valid_files:
        return {
            'day': day_date.strftime('%Y-%m-%d'),
            'status': 'failed',
            'error': f'No valid GRIB files found for {date_str}',
            'processing_time': 0
        }
    
    print(f"📁 Found {len(valid_files)} GRIB files for {day_date.strftime('%Y-%m-%d')}")
    
    # Group files by time (this will include f00 and f01 files for quarter-hourly data)
    file_groups = group_grib_files_by_time(valid_files)
    
    # Check what forecast files are actually available
    available_forecasts = set()
    for time_key in file_groups.keys():
        hour, forecast = time_key.split('_')
        available_forecasts.add(forecast)
    
    # Verify we have the expected quarter-hourly data
    # For a typical day, we expect 24 hours × 2 forecast files = 48 files
    # But the actual number can vary based on data availability
    expected_times = []
    for hour in range(24):  # All hours in the day
        for forecast in DEFAULT_HOURS_FORECASTED:  # f00 and f01
            expected_times.append(f"{hour:02d}_{forecast}")
    
    missing_times = [t for t in expected_times if t not in file_groups]
    
    # Only warn if we're missing a significant portion of expected files
    # For a typical day, having 48 files is good, even if some time groups are missing
    if len(valid_files) >= 40:  # If we have at least 40 files, that's good
        print(f"✅ Found {len(valid_files)} GRIB files - sufficient data available")
        if missing_times:  # Warn about ANY missing time groups
            print(f"⚠️  Warning: {len(missing_times)} time groups missing:")
            for missing_time in sorted(missing_times)[:10]:  # Show first 10 missing times
                print(f"   - Missing: {missing_time}")
            if len(missing_times) > 10:
                print(f"   ... and {len(missing_times) - 10} more")
    elif len(missing_times) > len(expected_times) * 0.2:  # Missing more than 20%
        print(f"⚠️  Missing {len(missing_times)} time groups (out of {len(expected_times)} expected)")
        print(f"📊 Found {len(file_groups)} time groups")
    else:
        print(f"📊 Found {len(file_groups)} time groups (missing {len(missing_times)} minor gaps)")
    
    # Initialize data storage for this day
    wind_data = defaultdict(dict)  # {var_name: {timestamp: values}}
    solar_data = defaultdict(dict)  # {var_name: {timestamp: values}}
    all_timestamps = []
    
    # Sort time groups chronologically for proper timestamp ordering
    sorted_time_groups = sorted(file_groups.items(), key=lambda x: (int(x[0].split('_')[0]), int(x[0].split('_')[1])))
    
    print(f"⏰ Processing {len(sorted_time_groups)} time groups in chronological order...")
    
    # Process each time group - PROPERLY HANDLE QUARTER-HOURLY TIMESTAMPS
    for time_key, files in sorted_time_groups:
        hour, forecast = time_key.split('_')
        
        # Create base timestamp for this forecast
        forecast_int = int(forecast)
        base_time = day_date.replace(hour=int(hour), minute=0, second=0)
        
        # Process each file in the time group (SINGLE PASS!)
        for file_path in files:
            # Extract ALL variables from this file in one pass
            wind_values, solar_values, file_timestamps = process_single_grib_file_single_pass(
                file_path, wind_selectors, solar_selectors, 
                wind_indices, solar_indices, grid_lats, grid_lons, day_date
            )
            
            # Add the extracted values to our data storage
            for var_name, time_data in wind_values.items():
                for timestamp, values in time_data.items():
                    wind_data[var_name][timestamp] = values
                    if timestamp not in all_timestamps:
                        all_timestamps.append(timestamp)
            
            for var_name, time_data in solar_values.items():
                for timestamp, values in time_data.items():
                    solar_data[var_name][timestamp] = values
                    if timestamp not in all_timestamps:
                        all_timestamps.append(timestamp)
    
    # Sort all timestamps chronologically
    all_timestamps.sort()
    print(f"📊 Found {len(all_timestamps)} unique timestamps")
    
    # Convert to the format expected by the saving logic
    wind_data_final = defaultdict(list)
    solar_data_final = defaultdict(list)
    timestamps_final = []
    
    for timestamp in all_timestamps:
        timestamps_final.append(timestamp)
        
        # Add wind data for this timestamp
        for var_name in wind_selectors.keys():
            if var_name in wind_data and timestamp in wind_data[var_name]:
                wind_data_final[var_name].append(wind_data[var_name][timestamp])
            else:
                # Fill with NaN if no data for this timestamp
                wind_data_final[var_name].append(np.full(len(wind_indices) if wind_indices else 0, np.nan))
        
        # Add solar data for this timestamp
        for var_name in solar_selectors.keys():
            if var_name in solar_data and timestamp in solar_data[var_name]:
                solar_data_final[var_name].append(solar_data[var_name][timestamp])
            else:
                # Fill with NaN if no data for this timestamp
                solar_data_final[var_name].append(np.full(len(solar_indices) if solar_indices else 0, np.nan))
    
    # Convert to DataFrames and save (batch write for better performance)
    try:
        # Create output directories once
        wind_dirs = {}
        solar_dirs = {}
        
        for var_name in wind_data.keys():
            var_dir = os.path.join(wind_output_dir, var_name)
            os.makedirs(var_dir, exist_ok=True)
            wind_dirs[var_name] = var_dir
        
        for var_name in solar_data.keys():
            var_dir = os.path.join(solar_output_dir, var_name)
            os.makedirs(var_dir, exist_ok=True)
            solar_dirs[var_name] = var_dir
        
        # Save wind data
        if wind_data_final and wind_indices is not None:
            for var_name, values_list in wind_data_final.items():
                if values_list:
                    # Create DataFrame
                    df = pd.DataFrame(values_list, columns=wind_locations['pid'].values)
                    df.index = timestamps_final[:len(df)]
                    
                    # Save to parquet
                    output_file = os.path.join(wind_dirs[var_name], f"{day_date.strftime('%Y%m%d')}.parquet")
                    df.to_parquet(output_file, compression=compression)
                    print(f"✅ Saved wind {var_name}: {len(df)} rows")
        
        # Save solar data
        if solar_data_final and solar_indices is not None:
            for var_name, values_list in solar_data_final.items():
                if values_list:
                    # Create DataFrame
                    df = pd.DataFrame(values_list, columns=solar_locations['pid'].values)
                    df.index = timestamps_final[:len(df)]
                    
                    # Save to parquet
                    output_file = os.path.join(solar_dirs[var_name], f"{day_date.strftime('%Y%m%d')}.parquet")
                    df.to_parquet(output_file, compression=compression)
                    print(f"✅ Saved solar {var_name}: {len(df)} rows")
        
        # Calculate and save wind speeds from U and V components
        if wind_data_final and wind_indices is not None:
            # Calculate WindSpeed80 from UWind80 and VWind80
            if "UWind80" in wind_data_final and "VWind80" in wind_data_final:
                uwind80_df = pd.DataFrame(wind_data_final["UWind80"], columns=wind_locations['pid'].values)
                vwind80_df = pd.DataFrame(wind_data_final["VWind80"], columns=wind_locations['pid'].values)
                uwind80_df.index = timestamps_final[:len(uwind80_df)]
                vwind80_df.index = timestamps_final[:len(vwind80_df)]
                
                # Calculate wind speed: sqrt(U^2 + V^2)
                wind_speed_80 = np.sqrt(uwind80_df**2 + vwind80_df**2)
                
                # Save WindSpeed80
                wind_speed_dir = os.path.join(wind_output_dir, "WindSpeed80")
                os.makedirs(wind_speed_dir, exist_ok=True)
                output_file = os.path.join(wind_speed_dir, f"{day_date.strftime('%Y%m%d')}.parquet")
                wind_speed_80.to_parquet(output_file, compression=compression)
                print(f"✅ Saved WindSpeed80: {len(wind_speed_80)} rows")
            
        if solar_data_final and solar_indices is not None:
            # Calculate WindSpeed10 from UWind10 and VWind10
            if "UWind10" in solar_data_final and "VWind10" in solar_data_final:
                uwind10_df = pd.DataFrame(solar_data_final["UWind10"], columns=solar_locations['pid'].values)
                vwind10_df = pd.DataFrame(solar_data_final["VWind10"], columns=solar_locations['pid'].values)
                uwind10_df.index = timestamps_final[:len(uwind10_df)]
                vwind10_df.index = timestamps_final[:len(vwind10_df)]
                
                # Calculate wind speed: sqrt(U^2 + V^2)
                wind_speed_10 = np.sqrt(uwind10_df**2 + vwind10_df**2)
                
                # Save WindSpeed10
                solar_wind_speed_dir = os.path.join(solar_output_dir, "WindSpeed10")
                os.makedirs(solar_wind_speed_dir, exist_ok=True)
                output_file = os.path.join(solar_wind_speed_dir, f"{day_date.strftime('%Y%m%d')}.parquet")
                wind_speed_10.to_parquet(output_file, compression=compression)
                print(f"✅ Saved WindSpeed10: {len(wind_speed_10)} rows")
        
        processing_time = time.time() - start_time
        
        return {
            'day': day_date.strftime('%Y-%m-%d'),
            'status': 'success',
            'wind_variables': len(wind_data_final),
            'solar_variables': len(solar_data_final),
            'wind_locations': len(wind_locations) if wind_indices else 0,
            'solar_locations': len(solar_locations) if solar_indices else 0,
            'processing_time': processing_time,
            'total_timestamps': len(timestamps_final)
        }
        
    except Exception as e:
        return {
            'day': day_date.strftime('%Y-%m-%d'),
            'status': 'failed',
            'error': str(e),
            'processing_time': time.time() - start_time
        }
    finally:
        # Clear memory
        del wind_data, solar_data, wind_data_final, solar_data_final, all_timestamps, timestamps_final
        gc.collect()


def extract_specific_points_daily_single_pass(
    wind_csv_path,
    solar_csv_path,
    START,
    END,
    DATADIR=None,
    DEFAULT_HOURS_FORECASTED=DEFAULT_HOURS_FORECASTED,
    wind_selectors=None,
    solar_selectors=None,
    wind_output_dir=WIND_OUTPUT_DIR,
    solar_output_dir=SOLAR_OUTPUT_DIR,
    compression=DEFAULT_COMPRESSION,
    use_parallel=True,
    num_workers=None,
    enable_resume=True,
    batch_size=36,  # Process 36 days at a time (one day per CPU for full utilization)
    max_memory_gb=250.0,  # Memory limit for 256GB system (use full RAM)
):
    """
    Extract HRRR data for specific locations using SINGLE-PASS day-by-day processing.
    This version reads each GRIB file only ONCE and extracts ALL variables.
    """
    
    print("🚀 SINGLE-PASS DAY-BY-DAY HRRR EXTRACTION")
    print("=" * 50)
    print("📊 OPTIMIZATION: Each GRIB file read only ONCE, all variables extracted")
    
    # Auto-detect data directory if not provided
    if DATADIR is None:
        DATADIR = get_data_directory()
    
    print(f"📁 Data directory: {DATADIR}")
    print(f"📊 Wind output: {wind_output_dir}")
    print(f"☀️  Solar output: {solar_output_dir}")
    
    # Load location data
    print("📊 Loading location data...")
    wind_df = pd.read_csv(wind_csv_path)
    solar_df = pd.read_csv(solar_csv_path)
    
    wind_locations = wind_df[["pid", "lat", "lon"]].copy()
    solar_locations = solar_df[["pid", "lat", "lon"]].copy()
    
    print(f"🌪️  Wind locations: {len(wind_locations)}")
    print(f"☀️  Solar locations: {len(solar_locations)}")
    
    # Set default selectors if not provided
    if wind_selectors is None:
        wind_selectors = DEFAULT_WIND_SELECTORS
    if solar_selectors is None:
        solar_selectors = DEFAULT_SOLAR_SELECTORS
    
    print(f"🌪️  Wind variables: {list(wind_selectors.keys())}")
    print(f"☀️  Solar variables: {list(solar_selectors.keys())}")
    
    # Generate date range
    date_range = []
    current_date = START
    while current_date <= END:
        date_range.append(current_date)
        current_date += datetime.timedelta(days=1)
    
    print(f"📅 Processing {len(date_range)} days: {START.strftime('%Y-%m-%d')} to {END.strftime('%Y-%m-%d')}")
    print(f"⏰ Quarter-hourly data: f00 (00 min) + f01 (15, 30, 45 min)")
    print(f"📊 Expected data points per day: 96 (24 hours × 4 time offsets per hour)")
    
    # Show system resource usage
    import multiprocessing as mp
    if PSUTIL_AVAILABLE:
        import psutil
        cpu_count = psutil.cpu_count()
        memory = psutil.virtual_memory()
        print(f"🖥️  System Resources: {cpu_count} CPUs, {memory.total / (1024**3):.1f}GB RAM")
    else:
        cpu_count = mp.cpu_count()
        print(f"🖥️  System Resources: {cpu_count} CPUs detected")
    
    # Ensure we're using all available CPUs
    if num_workers is None:
        num_workers = cpu_count
        # Use ALL available CPUs for maximum performance
        print(f"⚡ Using {num_workers} workers (ALL {cpu_count} available CPUs)")
    
    # Set memory limit to use full 256 GB RAM
    if max_memory_gb < 250:
        max_memory_gb = 250.0  # Use full 256 GB RAM
        print(f"💾 Memory limit set to {max_memory_gb} GB")
    
    # Check for resume capability
    completed_days = set()
    if enable_resume:
        print("🔍 Checking for completed days...")
        for var_name in wind_selectors.keys():
            var_dir = os.path.join(wind_output_dir, var_name)
            if os.path.exists(var_dir):
                for file in os.listdir(var_dir):
                    if file.endswith('.parquet'):
                        day_str = file.replace('.parquet', '')
                        completed_days.add(day_str)
        
        for var_name in solar_selectors.keys():
            var_dir = os.path.join(solar_output_dir, var_name)
            if os.path.exists(var_dir):
                for file in os.listdir(var_dir):
                    if file.endswith('.parquet'):
                        day_str = file.replace('.parquet', '')
                        completed_days.add(day_str)
        
        if completed_days:
            print(f"✅ Found {len(completed_days)} completed days")
            # Filter out completed days
            date_range = [d for d in date_range if d.strftime('%Y%m%d') not in completed_days]
            print(f"📅 Remaining days to process: {len(date_range)}")
    
    if not date_range:
        print("✅ All days already processed!")
        return {
            'status': 'completed',
            'total_days': len(completed_days),
            'completed_days': len(completed_days),
            'failed_days': 0
        }
    
    # Process days in batches
    results = []
    total_start_time = time.time()
    
    # Use parallel processing for larger datasets (more than 1 day OR more than 10 locations)
    # This ensures yearly runs always use all CPUs
    if len(date_range) > 1 or len(wind_locations) + len(solar_locations) > 10:
        # Use parallel processing for larger datasets
        if use_parallel and num_workers and num_workers > 1:
            print(f"🚀 Using parallel processing with {num_workers} workers")
            print(f"📦 Processing in batches of {batch_size} days")
            
            # Process in batches to avoid overwhelming the system
            for i in range(0, len(date_range), batch_size):
                batch_dates = date_range[i:i + batch_size]
                print(f"\n📦 Processing batch {i//batch_size + 1}/{(len(date_range) + batch_size - 1)//batch_size}")
                
                # Create partial function with fixed parameters
                process_func = partial(
                    process_single_day_single_pass,
                    wind_locations=wind_locations,
                    solar_locations=solar_locations,
                    data_dir=DATADIR,
                    wind_selectors=wind_selectors,
                    solar_selectors=solar_selectors,
                    wind_output_dir=wind_output_dir,
                    solar_output_dir=solar_output_dir,
                    compression=compression
                )
                
                # Process batch in parallel with error handling
                try:
                    with mp.Pool(processes=num_workers) as pool:
                        batch_results = pool.map(process_func, batch_dates)
                except Exception as e:
                    print(f"⚠️  Parallel processing failed: {e}")
                    print("🔄 Falling back to sequential processing...")
                    # Fall back to sequential processing
                    batch_results = []
                    for day_date in batch_dates:
                        try:
                            result = process_func(day_date)
                            batch_results.append(result)
                        except Exception as day_error:
                            print(f"❌ Error processing {day_date.strftime('%Y-%m-%d')}: {day_error}")
                            batch_results.append({
                                'day': day_date.strftime('%Y-%m-%d'),
                                'status': 'failed',
                                'error': str(day_error),
                                'processing_time': 0
                            })
                except (OSError, PermissionError) as e:
                    print(f"⚠️  System error with parallel processing: {e}")
                    print("🔄 Falling back to sequential processing...")
                    # Fall back to sequential processing
                    batch_results = []
                    for day_date in batch_dates:
                        try:
                            result = process_func(day_date)
                            batch_results.append(result)
                        except Exception as day_error:
                            print(f"❌ Error processing {day_date.strftime('%Y-%m-%d')}: {day_error}")
                            batch_results.append({
                                'day': day_date.strftime('%Y-%m-%d'),
                                'status': 'failed',
                                'error': str(day_error),
                                'processing_time': 0
                            })
                
                results.extend(batch_results)
                
                # Clear memory after each batch
                gc.collect()
                
        else:
            print("🔄 Using sequential processing")
            
            # Process days sequentially
            for day_date in tqdm(date_range, desc="Processing days"):
                result = process_single_day_single_pass(
                    day_date,
                    wind_locations,
                    solar_locations,
                    DATADIR,
                    wind_selectors,
                    solar_selectors,
                    wind_output_dir,
                    solar_output_dir,
                    compression
                )
                results.append(result)
                
                # Clear memory after each day
                gc.collect()
    else:
        print("🔄 Using sequential processing for very small dataset")
        
        # Process days sequentially for very small datasets
        for day_date in tqdm(date_range, desc="Processing days"):
            result = process_single_day_single_pass(
                day_date,
                wind_locations,
                solar_locations,
                DATADIR,
                wind_selectors,
                solar_selectors,
                wind_output_dir,
                solar_output_dir,
                compression
            )
            results.append(result)
            
            # Clear memory after each day
            gc.collect()
    
    # Summary
    total_time = time.time() - total_start_time
    successful_days = [r for r in results if r['status'] == 'success']
    failed_days = [r for r in results if r['status'] == 'failed']
    
    print("\n📊 EXTRACTION SUMMARY")
    print("=" * 30)
    print(f"✅ Successful days: {len(successful_days)}")
    print(f"❌ Failed days: {len(failed_days)}")
    print(f"⏱️  Total time: {total_time/3600:.1f} hours")
    
    if successful_days:
        avg_time = sum(r['processing_time'] for r in successful_days) / len(successful_days)
        print(f"⏱️  Average time per day: {avg_time/60:.1f} minutes")
    
    if failed_days:
        print("\n❌ Failed days:")
        for result in failed_days:
            print(f"   {result['day']}: {result.get('error', 'Unknown error')}")
    
    return {
        'status': 'completed',
        'total_days': len(date_range),
        'successful_days': len(successful_days),
        'failed_days': len(failed_days),
        'total_time': total_time,
        'results': results
    }


if __name__ == "__main__":
    # Example usage
    START = datetime.datetime(2023, 1, 1, 0, 0, 0)
    END = datetime.datetime(2023, 1, 7, 23, 0, 0)  # One week test
    
    result = extract_specific_points_daily_single_pass(
        wind_csv_path="wind.csv",
        solar_csv_path="solar.csv",
        START=START,
        END=END,
        num_workers=None,  # Use default from config (all CPUs)
        batch_size=36,  # Process 36 days at a time (one day per CPU for full utilization)
    )
    
    print(f"\n🎯 Final result: {result}") 
