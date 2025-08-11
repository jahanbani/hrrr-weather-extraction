#!/usr/bin/env python3
"""
Day-by-day HRRR data extraction for specific wind and solar locations.
This version processes one day at a time to avoid memory accumulation and provide fault tolerance.
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


def get_wind_data_lat_long(dt, directory, hours_forecasted="0"):
    """Returns the latitude and longitudes of the various wind grid sectors."""
    try:
        date_str = dt.strftime("%Y%m%d")
        date_dir = os.path.join(directory, date_str)
        
        import glob
        grib_pattern = os.path.join(date_dir, "*.grib2")
        grib_files = glob.glob(grib_pattern)
        
        if not grib_files:
            print(f"No GRIB files found in {date_dir}")
            return None
        
        # Filter out subset files which appear to be corrupted
        valid_files = [f for f in grib_files if "subset_" not in os.path.basename(f)]
        
        if not valid_files:
            print(f"No valid GRIB files found in {date_dir} (only subset files available)")
            return None
        
        # Use the first valid GRIB file to get grid coordinates
        grb_file = valid_files[0]
        grb = pygrib.open(grb_file)
        grb_message = grb.select(name="U component of wind")[0]
        
        lats, lons = grb_message.latlons()
        grb.close()
        
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


def extract_values_for_points(grb, point_indices, grid_lats, grid_lons):
    """Extract values for specific grid points from a GRIB message."""
    if point_indices is None:
        return None
    
    try:
        # Get the full grid data
        values_2d = grb.values
        
        # Extract values for the specific points
        extracted_values = []
        for row, col in point_indices:
            if 0 <= row < values_2d.shape[0] and 0 <= col < values_2d.shape[1]:
                extracted_values.append(values_2d[row, col])
            else:
                extracted_values.append(np.nan)
        
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
        # Extract time information from filename
        # Example: hrrr.t00z.wrfsubhf00.grib2
        match = re.search(r'hrrr\.t(\d{2})z\.wrfsubhf(\d{2})\.grib2', os.path.basename(file))
        if match:
            hour = match.group(1)
            forecast = match.group(2)
            time_key = f"{hour}_{forecast}"
            grouped[time_key].append(file)
    
    return grouped


def process_single_day(day_date, wind_locations, solar_locations, data_dir, 
                      wind_selectors, solar_selectors, wind_output_dir, solar_output_dir,
                      compression="snappy"):
    """
    Process a single day of HRRR data.
    
    Args:
        day_date: datetime object for the day to process
        wind_locations: DataFrame with wind location data
        solar_locations: DataFrame with solar location data
        data_dir: Directory containing GRIB data
        wind_selectors: Dictionary of wind variables to extract
        solar_selectors: Dictionary of solar variables to extract
        wind_output_dir: Output directory for wind data
        solar_output_dir: Output directory for solar data
        compression: Compression for parquet files
    
    Returns:
        dict: Results summary for the day
    """
    
    print(f"🌅 Processing day: {day_date.strftime('%Y-%m-%d')}")
    start_time = time.time()
    
    # Get grid coordinates for this day
    grid_coords = get_wind_data_lat_long(day_date, data_dir)
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
    
    # Get GRIB files for this day
    date_str = day_date.strftime("%Y%m%d")
    date_dir = os.path.join(data_dir, date_str)
    
    if not os.path.exists(date_dir):
        return {
            'day': day_date.strftime('%Y-%m-%d'),
            'status': 'failed',
            'error': f'No data directory found: {date_dir}',
            'processing_time': 0
        }
    
    # Group files by time
    import glob
    grib_files = glob.glob(os.path.join(date_dir, "*.grib2"))
    valid_files = [f for f in grib_files if "subset_" not in os.path.basename(f)]
    
    if not valid_files:
        return {
            'day': day_date.strftime('%Y-%m-%d'),
            'status': 'failed',
            'error': 'No valid GRIB files found',
            'processing_time': 0
        }
    
    file_groups = group_grib_files_by_time(valid_files)
    
    # Initialize data storage for this day
    wind_data = defaultdict(list)
    solar_data = defaultdict(list)
    timestamps = []
    
    # Process each time group
    for time_key, files in file_groups.items():
        hour, forecast = time_key.split('_')
        
        # Create timestamp for this forecast
        forecast_time = day_date.replace(hour=int(hour)) + datetime.timedelta(hours=int(forecast))
        timestamps.append(forecast_time)
        
        # Process each file in the time group
        for file_path in files:
            try:
                with pygrib.open(file_path) as grb:
                    # Extract wind variables
                    if wind_indices is not None:
                        for var_name, selector in wind_selectors.items():
                            try:
                                # Find the GRIB message
                                messages = grb.select(shortName=selector)
                                if messages:
                                    grb_message = messages[0]
                                    
                                    # Extract values for wind locations
                                    values = extract_values_for_points(grb_message, wind_indices, grid_lats, grid_lons)
                                    if values is not None:
                                        wind_data[var_name].append(values)
                                    else:
                                        wind_data[var_name].append(np.full(len(wind_locations), np.nan))
                                else:
                                    wind_data[var_name].append(np.full(len(wind_locations), np.nan))
                            except Exception as e:
                                print(f"Error processing wind variable {var_name}: {e}")
                                wind_data[var_name].append(np.full(len(wind_locations), np.nan))
                    
                    # Extract solar variables
                    if solar_indices is not None:
                        for var_name, selector in solar_selectors.items():
                            try:
                                # Find the GRIB message
                                messages = grb.select(shortName=selector)
                                if messages:
                                    grb_message = messages[0]
                                    
                                    # Extract values for solar locations
                                    values = extract_values_for_points(grb_message, solar_indices, grid_lats, grid_lons)
                                    if values is not None:
                                        solar_data[var_name].append(values)
                                    else:
                                        solar_data[var_name].append(np.full(len(solar_locations), np.nan))
                                else:
                                    solar_data[var_name].append(np.full(len(solar_locations), np.nan))
                            except Exception as e:
                                print(f"Error processing solar variable {var_name}: {e}")
                                solar_data[var_name].append(np.full(len(solar_locations), np.nan))
                                
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                continue
    
    # Convert to DataFrames and save
    try:
        # Save wind data
        if wind_data and wind_indices is not None:
            for var_name, values_list in wind_data.items():
                if values_list:
                    # Create DataFrame
                    df = pd.DataFrame(values_list, columns=wind_locations['pid'].values)
                    df.index = timestamps[:len(df)]
                    
                    # Create output directory
                    var_dir = os.path.join(wind_output_dir, var_name)
                    os.makedirs(var_dir, exist_ok=True)
                    
                    # Save to parquet
                    output_file = os.path.join(var_dir, f"{day_date.strftime('%Y%m%d')}.parquet")
                    df.to_parquet(output_file, compression=compression)
                    print(f"✅ Saved wind {var_name}: {output_file}")
        
        # Save solar data
        if solar_data and solar_indices is not None:
            for var_name, values_list in solar_data.items():
                if values_list:
                    # Create DataFrame
                    df = pd.DataFrame(values_list, columns=solar_locations['pid'].values)
                    df.index = timestamps[:len(df)]
                    
                    # Create output directory
                    var_dir = os.path.join(solar_output_dir, var_name)
                    os.makedirs(var_dir, exist_ok=True)
                    
                    # Save to parquet
                    output_file = os.path.join(var_dir, f"{day_date.strftime('%Y%m%d')}.parquet")
                    df.to_parquet(output_file, compression=compression)
                    print(f"✅ Saved solar {var_name}: {output_file}")
        
        processing_time = time.time() - start_time
        
        return {
            'day': day_date.strftime('%Y-%m-%d'),
            'status': 'success',
            'wind_variables': len(wind_data),
            'solar_variables': len(solar_data),
            'wind_locations': len(wind_locations) if wind_indices else 0,
            'solar_locations': len(solar_locations) if solar_indices else 0,
            'processing_time': processing_time
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
        del wind_data, solar_data, timestamps
        gc.collect()


def extract_specific_points_daily(
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
    batch_size=7,  # Process 7 days at a time (one week)
):
    """
    Extract HRRR data for specific locations using day-by-day processing.
    
    This approach processes one day at a time to avoid memory accumulation
    and provide better fault tolerance.
    """
    
    print("🚀 DAY-BY-DAY HRRR EXTRACTION")
    print("=" * 50)
    
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
    
    if use_parallel and num_workers and num_workers > 1:
        print(f"🚀 Using parallel processing with {num_workers} workers")
        print(f"📦 Processing in batches of {batch_size} days")
        
        # Process in batches to avoid overwhelming the system
        for i in range(0, len(date_range), batch_size):
            batch_dates = date_range[i:i + batch_size]
            print(f"\n📦 Processing batch {i//batch_size + 1}/{(len(date_range) + batch_size - 1)//batch_size}")
            
            # Create partial function with fixed parameters
            process_func = partial(
                process_single_day,
                wind_locations=wind_locations,
                solar_locations=solar_locations,
                data_dir=DATADIR,
                wind_selectors=wind_selectors,
                solar_selectors=solar_selectors,
                wind_output_dir=wind_output_dir,
                solar_output_dir=solar_output_dir,
                compression=compression
            )
            
            # Process batch in parallel
            with mp.Pool(processes=num_workers) as pool:
                batch_results = pool.map(process_func, batch_dates)
            
            results.extend(batch_results)
            
            # Clear memory after each batch
            gc.collect()
            
    else:
        print("🔄 Using sequential processing")
        
        # Process days sequentially
        for day_date in tqdm(date_range, desc="Processing days"):
            result = process_single_day(
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
    
    result = extract_specific_points_daily(
        wind_csv_path="wind.csv",
        solar_csv_path="solar.csv",
        START=START,
        END=END,
        num_workers=4,  # Use 4 workers for testing
        batch_size=3,   # Process 3 days at a time
    )
    
    print(f"\n🎯 Final result: {result}") 