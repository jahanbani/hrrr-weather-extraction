#!/usr/bin/env python3
"""
Parallel function to extract HRRR data for specific wind and solar locations.
This version uses multiprocessing for better CPU utilization during GRIB file reading.
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
from powersimdata.utility.distance import ll2uv
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
    DEFAULT_MAX_FILE_GROUPS,
)

# Suppress fs package deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, 
                       message=".*pkg_resources.declare_namespace.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, 
                       message=".*fs.*")


def get_data_directory():
    """Get the appropriate data directory based on the operating system.
    
    Returns:
        str: Data directory path
    """
    system = platform.system()
    if system == "Windows":
        return "data"  # Change this for Windows
    elif system == "Linux":
        return "/research/alij/hrrr"  # Change this for Linux
    else:
        # Default to Linux path for other Unix-like systems
        return "/research/alij/hrrr"


def get_wind_data_lat_long(dt, directory, hours_forecasted="0"):
    """Returns the latitude and longitudes of the various wind grid sectors.
    
    Args:
        dt (datetime): Date and time of the grib data
        directory (str): Directory where the data is located
        hours_forecasted (str): Forecast hour (default "0")
        
    Returns:
        tuple: A tuple of 2 same lengthed numpy arrays, first one being
            latitude and second one being longitude.
    """
    try:
        # Construct the date-specific directory path
        date_str = dt.strftime("%Y%m%d")
        date_dir = os.path.join(directory, date_str)
        
        # Try to find any available GRIB file in the date directory
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
        # (all GRIB files for the same date should have the same grid)
        first_grib_file = valid_files[0]
        print(f"Using {os.path.basename(first_grib_file)} for grid coordinates")
        
        gribs = pygrib.open(first_grib_file)
        grib = next(gribs)
        return grib.latlons()
    except Exception as e:
        print(f"Error getting grid coordinates: {e}")
        print(f"File: {first_grib_file}")
        print(f"File exists: {os.path.exists(first_grib_file)}")
        print(f"File size: {os.path.getsize(first_grib_file) if os.path.exists(first_grib_file) else 'N/A'} bytes")
        return None


def find_closest_grid_points(points, grid_lats, grid_lons):
    """Find the closest grid points for each location.
    
    Args:
        points (pd.DataFrame): DataFrame with lat/lon points
        grid_lats (np.array): 2D array of grid latitudes
        grid_lons (np.array): 2D array of grid longitudes
        
    Returns:
        np.array: Array of closest grid indices
    """
    # Flatten grid coordinates
    grid_lats_flat = grid_lats.flatten()
    grid_lons_flat = grid_lons.flatten()
    
    # Create unit vectors for grid points
    grid_lat_lon_unit_vectors = [ll2uv(i, j) for i, j in zip(grid_lons_flat, grid_lats_flat)]
    
    # Create KDTree for efficient nearest neighbor search
    tree = KDTree(grid_lat_lon_unit_vectors)
    
    # Create unit vectors for point locations
    point_unit_vectors = [ll2uv(lon, lat) for lat, lon in zip(points.lat.values, points.lon.values)]
    
    # Find closest grid points
    _, indices = tree.query(point_unit_vectors)
    
    return indices


def extract_values_for_points(grb, point_indices, grid_lats, grid_lons):
    """Extract values for specific point indices from GRIB message.
    
    Args:
        grb: GRIB message object
        point_indices (np.array): Array of point indices
        grid_lats (np.array): 2D array of grid latitudes
        grid_lons (np.array): 2D array of grid longitudes
        
    Returns:
        np.array: Extracted values for the points
    """
    try:
        # Convert point indices to 2D indices
        n_lats, n_lons = grid_lats.shape
        lat_indices, lon_indices = np.unravel_index(point_indices, (n_lats, n_lons))
        
        # Extract values using advanced indexing
        values_2d = grb.values
        point_values = values_2d[lat_indices, lon_indices]
        
        # Convert to float32 to save memory
        return point_values.astype(np.float32)
    except MemoryError as e:
        print(f"Memory error in extraction: {e}")
        # Fallback: process in smaller batches
        try:
            batch_size = 1000  # Process 1k points at a time
            point_values = []
            for i in range(0, len(point_indices), batch_size):
                batch_indices = point_indices[i:i+batch_size]
                lat_indices, lon_indices = np.unravel_index(batch_indices, (n_lats, n_lons))
                values_2d = grb.values
                batch_values = values_2d[lat_indices, lon_indices]
                point_values.extend(batch_values)
                del values_2d  # Explicitly delete to free memory
                gc.collect()
            return np.array(point_values, dtype=np.float32)
        except Exception as e2:
            print(f"Fallback extraction also failed: {e2}")
            return None
    except Exception as e:
        print(f"Error in point extraction: {e}")
        return None


def group_grib_files_by_time(files):
    """Group GRIB files by time for efficient processing.
    
    Args:
        files (list): List of GRIB file paths
        
    Returns:
        dict: Dictionary of grouped files
    """
    file_groups = {}
    
    for file_path in files:
        # Extract hour and forecast type from HRRR naming convention
        filename = os.path.basename(file_path)
        match = re.search(r'hrrr\.t(\d{2})z\.wrfsubhf(\d{2})\.grib2', filename)
        if not match:
            continue
            
        hour_str, forecast_str = match.groups()
        fxx = f"f{forecast_str}"
        
        # Extract date from directory path
        dir_path = os.path.dirname(file_path)
        date_match = re.search(r'(\d{8})', dir_path)
        if date_match:
            date_str = date_match.group(1)
            key = f"{date_str}_{hour_str}"
        else:
            key = f"hour_{hour_str}"
            
        if key not in file_groups:
            file_groups[key] = {}
        file_groups[key][fxx] = file_path
    
    return file_groups


def process_file_group_parallel(args):
    """Process a single file group in parallel.
    
    Args:
        args: Tuple containing (key, group, wind_indices, solar_indices, grid_lats, grid_lons, 
              wind_points, solar_points, wind_selectors, solar_selectors)
        
    Returns:
        tuple: (key, group_wind_data, group_solar_data)
    """
    (key, group, wind_indices, solar_indices, grid_lats, grid_lons, 
     wind_points, solar_points, wind_selectors, solar_selectors) = args
    
    group_wind_data = defaultdict(dict)
    group_solar_data = defaultdict(dict)
    
    # Process f00 files (:00 timestamps - top of the hour)
    if 'f00' in group:
        try:
            with pygrib.open(group['f00']) as grbs:
                # Read all GRIB messages once
                grb_messages = list(grbs)
                
                # Process each timestamp
                for grb in grb_messages:
                    timestamp = pd.Timestamp(
                        year=grb.year, month=grb.month, day=grb.day,
                        hour=grb.hour, minute=grb.minute
                    )
                    
                    # Check if this is a wind variable and we have wind locations
                    if len(wind_indices) > 0:
                        # Match by shortName and level for wind variables
                        for var_key, short_name in wind_selectors.items():
                            if grb.shortName == short_name:
                                # Level check for wind variables
                                if "80" in var_key and grb.level == 80:
                                    wind_values = extract_values_for_points(grb, wind_indices, grid_lats, grid_lons)
                                    if wind_values is not None:
                                        wind_columns = wind_points.pid.astype(str).tolist()
                                        group_wind_data[var_key][timestamp] = dict(zip(wind_columns, wind_values))
                                    break
                                elif "10" in var_key and grb.level == 10:
                                    wind_values = extract_values_for_points(grb, wind_indices, grid_lats, grid_lons)
                                    if wind_values is not None:
                                        wind_columns = wind_points.pid.astype(str).tolist()
                                        group_wind_data[var_key][timestamp] = dict(zip(wind_columns, wind_values))
                                    break
                    
                    # Check if this is a solar variable and we have solar locations
                    if len(solar_indices) > 0:
                        # Match by shortName for solar variables
                        for var_key, short_name in solar_selectors.items():
                            if grb.shortName == short_name:
                                solar_values = extract_values_for_points(grb, solar_indices, grid_lats, grid_lons)
                                if solar_values is not None:
                                    solar_columns = solar_points.pid.astype(str).tolist()
                                    group_solar_data[var_key][timestamp] = dict(zip(solar_columns, solar_values))
                                break
                             
        except Exception as e:
            print(f"Error processing f00 {group['f00']}: {e}")
    
    # Process f01 files (:15, :30, :45 timestamps - subhourly)
    if 'f01' in group:
        try:
            with pygrib.open(group['f01']) as grbs:
                # Read all GRIB messages once
                grb_messages = list(grbs)
                
                # Process each timestamp
                for grb in grb_messages:
                    grb_str = str(grb)
                    
                    # Process all time offsets (15 min, 30 min, 45 min)
                    for offset, minute in [(15, 15), (30, 30), (45, 45)]:
                        if f"{offset}m mins" in grb_str or f"{offset} mins" in grb_str:
                            base_timestamp = pd.Timestamp(
                                year=grb.year, month=grb.month, day=grb.day,
                                hour=grb.hour, minute=grb.minute
                            )
                            dt = base_timestamp + pd.Timedelta(minutes=minute)
                            
                            # Check if this is a wind variable and we have wind locations
                            if len(wind_indices) > 0:
                                # Match by shortName and level for wind variables
                                for var_key, short_name in wind_selectors.items():
                                    if grb.shortName == short_name:
                                        # Level check for wind variables
                                        if "80" in var_key and grb.level == 80:
                                            wind_values = extract_values_for_points(grb, wind_indices, grid_lats, grid_lons)
                                            if wind_values is not None:
                                                wind_columns = wind_points.pid.astype(str).tolist()
                                                group_wind_data[var_key][dt] = dict(zip(wind_columns, wind_values))
                                            break
                                        elif "10" in var_key and grb.level == 10:
                                            wind_values = extract_values_for_points(grb, wind_indices, grid_lats, grid_lons)
                                            if wind_values is not None:
                                                wind_columns = wind_points.pid.astype(str).tolist()
                                                group_wind_data[var_key][dt] = dict(zip(wind_columns, wind_values))
                                            break
                            
                            # Check if this is a solar variable and we have solar locations
                            if len(solar_indices) > 0:
                                # Match by shortName for solar variables
                                for var_key, short_name in solar_selectors.items():
                                    if grb.shortName == short_name:
                                        solar_values = extract_values_for_points(grb, solar_indices, grid_lats, grid_lons)
                                        if solar_values is not None:
                                            solar_columns = solar_points.pid.astype(str).tolist()
                                            group_solar_data[var_key][dt] = dict(zip(solar_columns, solar_values))
                                        break
                             
        except Exception as e:
            print(f"Error processing f01 {group['f01']}: {e}")
    
    return key, group_wind_data, group_solar_data


def extract_specific_points_parallel(
    wind_csv_path,
    solar_csv_path,
    START,
    END,
    DATADIR=None,  # Auto-detect if None
    DEFAULT_HOURS_FORECASTED=DEFAULT_HOURS_FORECASTED,  # f00 and f01 only
    wind_selectors=None,
    solar_selectors=None,
    wind_output_dir=WIND_OUTPUT_DIR,
    solar_output_dir=SOLAR_OUTPUT_DIR,
    compression=DEFAULT_COMPRESSION,
    use_parallel=True,
    num_workers=None,
    max_file_groups=None,
    enable_resume=True,
):
    """
    Extract HRRR data for specific wind and solar locations with true multiprocessing.
    
    This function reads wind and solar points from CSV files and extracts only the
    closest grid points for those specific locations, using multiprocessing for
    maximum CPU utilization during GRIB file reading.
    
    Args:
        wind_csv_path (str): Path to wind points CSV file
        solar_csv_path (str): Path to solar points CSV file
        START (datetime): Start datetime
        END (datetime): End datetime
        DATADIR (str): Directory containing GRIB files (auto-detect if None)
        DEFAULT_HOURS_FORECASTED (list): List of forecast hours to process (default: ["0", "1"])
        wind_selectors (dict): Dictionary of wind variables to extract
        solar_selectors (dict): Dictionary of solar variables to extract
        wind_output_dir (str): Output directory for wind data
        solar_output_dir (str): Output directory for solar data
        compression (str): Parquet compression
        use_parallel (bool): Whether to use parallel processing
        num_workers (int): Number of workers (auto-detect if None)
        max_file_groups (int): Maximum file groups to process
        enable_resume (bool): Whether to enable resume functionality
        
    Returns:
        dict: Summary of processing results
    """
    
    # Auto-detect data directory if not provided
    if DATADIR is None:
        DATADIR = get_data_directory()
    
    print("🚀 PARALLEL SPECIFIC POINTS EXTRACTION")
    print("=" * 60)
    print(f"Date range: {START.date()} to {END.date()}")
    print(f"Wind CSV: {wind_csv_path}")
    print(f"Solar CSV: {solar_csv_path}")
    print(f"Data directory: {DATADIR}")
    print(f"Output directories: {wind_output_dir}, {solar_output_dir}")
    print(f"Forecast hours: {DEFAULT_HOURS_FORECASTED} (f00 and f01 only)")
    print()
    
    # Auto-detect optimal settings
    if num_workers is None:
        # Use all available CPUs for maximum performance
        num_workers = mp.cpu_count()  # Use multiprocessing CPU count
    
    if max_file_groups is None:
        max_file_groups = DEFAULT_MAX_FILE_GROUPS  # Use config value
    
    print(f"🎯 Using settings:")
    print(f"   Workers: {num_workers} (using all {mp.cpu_count()} CPUs)")
    print(f"   Max file groups: {max_file_groups} (from config: {DEFAULT_MAX_FILE_GROUPS})")
    print(f"   Parallel processing: {use_parallel}")
    print(f"   Resume enabled: {enable_resume}")
    print(f"   Memory optimization: Using float32 for data storage")
    print()
    
    # Phase 1: Load point data
    print("📁 Loading point data from CSV files...")
    load_start = time.time()
    
    try:
        wind_points = pd.read_csv(wind_csv_path)
        print(f"✅ Loaded {len(wind_points)} wind points from {wind_csv_path}")
    except FileNotFoundError:
        print(f"❌ Wind CSV file not found: {wind_csv_path}")
        return None
    except Exception as e:
        print(f"❌ Error loading wind CSV: {e}")
        return None
    
    try:
        solar_points = pd.read_csv(solar_csv_path)
        print(f"✅ Loaded {len(solar_points)} solar points from {solar_csv_path}")
    except FileNotFoundError:
        print(f"❌ Solar CSV file not found: {solar_csv_path}")
        return None
    except Exception as e:
        print(f"❌ Error loading solar CSV: {e}")
        return None
    
    # Check required columns
    required_columns = ["pid", "lat", "lon"]
    for col in required_columns:
        if col not in wind_points.columns:
            print(f"❌ Missing column '{col}' in wind CSV")
            return None
        if col not in solar_points.columns:
            print(f"❌ Missing column '{col}' in solar CSV")
            return None
    
    # Select only required columns
    wind_points = wind_points[["pid", "lat", "lon"]].copy()
    solar_points = solar_points[["pid", "lat", "lon"]].copy()
    
    load_time = time.time() - load_start
    print(f"Point data loaded in {load_time:.2f}s")
    print()
    
    # Phase 2: Extract grid metadata
    print("📊 Extracting grid metadata...")
    grid_start = time.time()
    
    wind_data_lat_long = get_wind_data_lat_long(START, DATADIR)
    if wind_data_lat_long is None:
        print("❌ Failed to extract grid metadata")
        return None
    
    grid_lats, grid_lons = wind_data_lat_long[0], wind_data_lat_long[1]
    n_lats, n_lons = grid_lats.shape
    total_grid_points = n_lats * n_lons
    
    print(f"Grid dimensions: {n_lats} x {n_lons} = {total_grid_points:,} total points")
    grid_time = time.time() - grid_start
    print(f"Grid metadata extracted in {grid_time:.2f}s")
    print()
    
    # Phase 3: Find closest grid points
    print("🎯 Finding closest grid points...")
    closest_start = time.time()
    
    wind_indices = find_closest_grid_points(wind_points, grid_lats, grid_lons)
    solar_indices = find_closest_grid_points(solar_points, grid_lats, grid_lons)
    
    print(f"Wind points: {len(wind_indices)} indices")
    print(f"Solar points: {len(solar_indices)} indices")
    closest_time = time.time() - closest_start
    print(f"Closest points found in {closest_time:.2f}s")
    print()
    
    # Phase 4: Set up default selectors if not provided
    if wind_selectors is None:
        wind_selectors = DEFAULT_WIND_SELECTORS
    
    if solar_selectors is None:
        solar_selectors = DEFAULT_SOLAR_SELECTORS
    
    # Combine all selectors for processing
    all_selectors = {**wind_selectors, **solar_selectors}
    
    print(f"📋 Variables to extract:")
    print(f"   Wind variables: {list(wind_selectors.keys())} (for wind locations only)")
    print(f"   Solar variables: {list(solar_selectors.keys())} (for solar locations only)")
    print(f"   Total variables: {len(all_selectors)}")
    print()
    
    # Phase 5: Find all GRIB files
    print("🔍 Discovering GRIB files...")
    file_discovery_start = time.time()
    
    date_range = pd.date_range(start=START.date(), end=END.date(), freq="1D")
    all_files = []
    
    for date in date_range:
        date_str = date.strftime("%Y%m%d")
        date_dir = os.path.join(DATADIR, date_str)
        if os.path.exists(date_dir):
            grib_pattern = os.path.join(date_dir, "*.grib2")
            import glob
            date_files = glob.glob(grib_pattern)
            # Filter out subset files which appear to be corrupted
            valid_files = [f for f in date_files if "subset_" not in os.path.basename(f)]
            all_files.extend(valid_files)
            print(f"  Found {len(valid_files)} valid files in {date_str}")
        else:
            print(f"  No directory found for {date_str}")
    
    if not all_files:
        print(f"❌ No GRIB files found in date range {START.date()} to {END.date()}")
        return None
    
    print(f"Found {len(all_files)} total GRIB files")
    file_discovery_time = time.time() - file_discovery_start
    print(f"File discovery completed in {file_discovery_time:.2f}s")
    print()
    
    # Phase 6: Group files by time
    print("📦 Grouping files by time...")
    grouping_start = time.time()
    
    file_groups = group_grib_files_by_time(all_files)
    
    # Limit file groups if specified
    if max_file_groups is not None and len(file_groups) > max_file_groups:
        print(f"⚠️  {len(file_groups)} file groups found, processing only first {max_file_groups}")
        sorted_keys = sorted(file_groups.keys())
        file_groups = {k: file_groups[k] for k in sorted_keys[:max_file_groups]}
    else:
        print(f"Processing {len(file_groups)} file groups")
    
    grouping_time = time.time() - grouping_start
    print(f"File grouping completed in {grouping_time:.2f}s")
    print()
    
    # Phase 7: Create output directories
    print("📁 Creating output directories...")
    os.makedirs(wind_output_dir, exist_ok=True)
    os.makedirs(solar_output_dir, exist_ok=True)
    
    # Phase 8: Process each file group with multiprocessing
    print("🚀 Processing file groups with multiprocessing...")
    processing_start = time.time()
    
    # Initialize data storage
    all_wind_data = defaultdict(dict)
    all_solar_data = defaultdict(dict)
    
    # Prepare arguments for parallel processing
    process_args = []
    for key in sorted(file_groups.keys()):
        group = file_groups[key]
        args = (key, group, wind_indices, solar_indices, grid_lats, grid_lons,
                wind_points, solar_points, wind_selectors, solar_selectors)
        process_args.append(args)
    
    # Use multiprocessing for true parallel processing
    if use_parallel and num_workers > 1:
        print(f"🚀 Using {num_workers} parallel processes for file processing...")
        print(f"   Processing {len(process_args)} file groups with {num_workers} workers")
        print(f"   Expected speedup: ~{num_workers}x faster than sequential")
        print(f"   CPU cores available: {mp.cpu_count()}")
        if PSUTIL_AVAILABLE:
            print(f"   Memory available: {psutil.virtual_memory().available / (1024**3):.1f} GB")
        
        parallel_start = time.time()
        
        # Use multiprocessing Pool for true parallel processing
        with mp.Pool(processes=num_workers) as pool:
            # Process all file groups in parallel
            results = list(tqdm(
                pool.imap(process_file_group_parallel, process_args),
                total=len(process_args),
                desc="Processing file groups"
            ))
        
        # Aggregate results
        for key, group_wind_data, group_solar_data in results:
            # Aggregate wind data
            for var_key, var_data in group_wind_data.items():
                if var_key not in all_wind_data:
                    all_wind_data[var_key] = {}
                all_wind_data[var_key].update(var_data)
            
            # Aggregate solar data
            for var_key, var_data in group_solar_data.items():
                if var_key not in all_solar_data:
                    all_solar_data[var_key] = {}
                all_solar_data[var_key].update(var_data)
        
        parallel_time = time.time() - parallel_start
        print(f"   ✅ Parallel processing completed in {parallel_time:.1f}s")
        print(f"   📊 Average rate: {len(process_args)/parallel_time:.1f} files/sec")
    else:
        # Sequential processing (fallback)
        print(f"🔄 Using sequential processing (workers: {num_workers})...")
        
        for args in tqdm(process_args, desc="Processing file groups"):
            try:
                key, group_wind_data, group_solar_data = process_file_group_parallel(args)
                
                # Aggregate data
                for var_key, var_data in group_wind_data.items():
                    if var_key not in all_wind_data:
                        all_wind_data[var_key] = {}
                    all_wind_data[var_key].update(var_data)
                
                for var_key, var_data in group_solar_data.items():
                    if var_key not in all_solar_data:
                        all_solar_data[var_key] = {}
                    all_solar_data[var_key].update(var_data)
                    
            except Exception as e:
                print(f"Error processing file group {args[0]}: {e}")
                continue
    
    # Convert dictionaries to DataFrames
    print("📊 Converting to DataFrames...")
    print(f"  Debug: all_wind_data keys: {list(all_wind_data.keys())}")
    print(f"  Debug: all_solar_data keys: {list(all_solar_data.keys())}")
    
    for var_name, var_data in all_wind_data.items():
        print(f"  Debug: Processing wind variable {var_name}")
        print(f"  Debug: var_data type: {type(var_data)}")
        print(f"  Debug: var_data length: {len(var_data) if var_data else 0}")
        if var_data:
            print(f"  Debug: var_data sample keys: {list(var_data.keys())[:3]}")
            wind_df = pd.DataFrame.from_dict(var_data, orient='index').sort_index()
            wind_df.index.name = 'time'
            all_wind_data[var_name] = wind_df
            print(f"  Wind {var_name}: {len(wind_df)} timestamps, {len(wind_df.columns)} locations")
            print(f"  Debug: DataFrame empty? {wind_df.empty}")
            print(f"  Debug: DataFrame shape: {wind_df.shape}")
        else:
            print(f"  Debug: var_data is empty for {var_name}")
    
    for var_name, var_data in all_solar_data.items():
        print(f"  Debug: Processing solar variable {var_name}")
        print(f"  Debug: var_data type: {type(var_data)}")
        print(f"  Debug: var_data length: {len(var_data) if var_data else 0}")
        if var_data:
            print(f"  Debug: var_data sample keys: {list(var_data.keys())[:3]}")
            solar_df = pd.DataFrame.from_dict(var_data, orient='index').sort_index()
            solar_df.index.name = 'time'
            all_solar_data[var_name] = solar_df
            print(f"  Solar {var_name}: {len(solar_df)} timestamps, {len(solar_df.columns)} locations")
            print(f"  Debug: DataFrame empty? {solar_df.empty}")
            print(f"  Debug: DataFrame shape: {solar_df.shape}")
        else:
            print(f"  Debug: var_data is empty for {var_name}")
    
    # Force garbage collection
    gc.collect()
    
    processing_time = time.time() - processing_start
    print(f"Variable processing completed in {processing_time:.2f}s")
    print()
    
    # Phase 9: Save results
    print("💾 Saving results...")
    save_start = time.time()
    
    # Save each wind variable in its own file within variable-named subfolder
    print(f"  Debug: Saving wind data, {len(all_wind_data)} variables")
    for var_name, df in all_wind_data.items():
        print(f"  Debug: Checking wind variable {var_name}")
        print(f"  Debug: DataFrame type: {type(df)}")
        print(f"  Debug: DataFrame empty? {df.empty}")
        if hasattr(df, 'shape'):
            print(f"  Debug: DataFrame shape: {df.shape}")
        
        if not df.empty:
            # Create variable-specific subfolder
            var_subfolder = os.path.join(wind_output_dir, var_name)
            os.makedirs(var_subfolder, exist_ok=True)
            
            filename = f"{START.strftime('%Y%m%d')}_to_{END.strftime('%Y%m%d')}.parquet"
            filepath = os.path.join(var_subfolder, filename)
            df.to_parquet(filepath, compression=compression)
            print(f"  Saved wind {var_name}: {len(df)} timestamps, {len(df.columns)} locations")
        else:
            print(f"  Debug: Skipping wind {var_name} - DataFrame is empty")
    
    # Save each solar variable in its own file within variable-named subfolder
    print(f"  Debug: Saving solar data, {len(all_solar_data)} variables")
    for var_name, df in all_solar_data.items():
        print(f"  Debug: Checking solar variable {var_name}")
        print(f"  Debug: DataFrame type: {type(df)}")
        print(f"  Debug: DataFrame empty? {df.empty}")
        if hasattr(df, 'shape'):
            print(f"  Debug: DataFrame shape: {df.shape}")
        
        if not df.empty:
            # Create variable-specific subfolder
            var_subfolder = os.path.join(solar_output_dir, var_name)
            os.makedirs(var_subfolder, exist_ok=True)
            
            filename = f"{START.strftime('%Y%m%d')}_to_{END.strftime('%Y%m%d')}.parquet"
            filepath = os.path.join(var_subfolder, filename)
            df.to_parquet(filepath, compression=compression)
            print(f"  Saved solar {var_name}: {len(df)} timestamps, {len(df.columns)} locations")
        else:
            print(f"  Debug: Skipping solar {var_name} - DataFrame is empty")
    
    save_time = time.time() - save_start
    print(f"Results saved in {save_time:.2f}s")
    print()
    
    # Phase 10: Calculate and save derived wind speeds
    print("🌪️  Calculating derived wind speeds...")
    derived_start = time.time()
    
    # Calculate wind speeds from U and V components
    wind_speed_vars = []
    
    if "UWind80" in all_wind_data and "VWind80" in all_wind_data:
        wind_speed_80 = np.sqrt(all_wind_data["UWind80"]**2 + all_wind_data["VWind80"]**2)
        # Keep the same column names as the original wind components
        wind_speed_80.columns = wind_speed_80.columns
        
        # Create variable-specific subfolder for derived wind speed
        var_subfolder = os.path.join(wind_output_dir, "WindSpeed80")
        os.makedirs(var_subfolder, exist_ok=True)
        
        filename = f"{START.strftime('%Y%m%d')}_to_{END.strftime('%Y%m%d')}.parquet"
        filepath = os.path.join(var_subfolder, filename)
        wind_speed_80.to_parquet(filepath, compression=compression)
        wind_speed_vars.append("WindSpeed80")
        print(f"  Saved WindSpeed80: {len(wind_speed_80)} timestamps, {len(wind_speed_80.columns)} locations")
    
    if "UWind10" in all_wind_data and "VWind10" in all_wind_data:
        wind_speed_10 = np.sqrt(all_wind_data["UWind10"]**2 + all_wind_data["VWind10"]**2)
        # Keep the same column names as the original wind components
        wind_speed_10.columns = wind_speed_10.columns
        
        # Create variable-specific subfolder for derived wind speed
        var_subfolder = os.path.join(wind_output_dir, "WindSpeed10")
        os.makedirs(var_subfolder, exist_ok=True)
        
        filename = f"{START.strftime('%Y%m%d')}_to_{END.strftime('%Y%m%d')}.parquet"
        filepath = os.path.join(var_subfolder, filename)
        wind_speed_10.to_parquet(filepath, compression=compression)
        wind_speed_vars.append("WindSpeed10")
        print(f"  Saved WindSpeed10: {len(wind_speed_10)} timestamps, {len(wind_speed_10.columns)} locations")
    
    derived_time = time.time() - derived_start
    print(f"Derived calculations completed in {derived_time:.2f}s")
    print()
    
    # Final summary
    total_time = time.time() - processing_start
    
    print("=" * 60)
    print("📊 EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Total processing time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Wind points processed: {len(wind_indices)}")
    print(f"Solar points processed: {len(solar_indices)}")
    print(f"Wind variables extracted: {list(wind_selectors.keys())} (for wind locations only)")
    print(f"Solar variables extracted: {list(solar_selectors.keys())} (for solar locations only)")
    print(f"Derived variables: {wind_speed_vars}")
    print(f"Output directories:")
    print(f"  Wind: {wind_output_dir}")
    print(f"  Solar: {solar_output_dir}")
    
    # File statistics
    wind_files = len([f for f in os.listdir(wind_output_dir) if f.endswith('.parquet')]) if os.path.exists(wind_output_dir) else 0
    solar_files = len([f for f in os.listdir(solar_output_dir) if f.endswith('.parquet')]) if os.path.exists(solar_output_dir) else 0
    
    print(f"Files created:")
    print(f"  Wind files: {wind_files} (separate files per variable)")
    print(f"  Solar files: {solar_files} (separate files per variable)")
    print(f"Optimization: True multiprocessing for GRIB file reading")
    
    return {
        "status": "completed",
        "processing_time_seconds": total_time,
        "wind_points": len(wind_indices),
        "solar_points": len(solar_indices),
        "variables_extracted": len(all_selectors),
        "wind_variables": list(wind_selectors.keys()),
        "solar_variables": list(solar_selectors.keys()),
        "derived_variables": wind_speed_vars,
        "wind_output_dir": wind_output_dir,
        "solar_output_dir": solar_output_dir,
        "wind_files_created": wind_files,
        "solar_files_created": solar_files
    }


# Example usage function
def main():
    """Example usage of the parallel specific points extraction."""
    
    # Example parameters
    wind_csv_path = "wind.csv"
    solar_csv_path = "solar.csv"
    START = datetime.datetime(2023, 1, 1, 0, 0, 0)
    END = datetime.datetime(2023, 1, 1, 23, 0, 0)  # One day for testing
    
    # Run extraction
    result = extract_specific_points_parallel(
        wind_csv_path=wind_csv_path,
        solar_csv_path=solar_csv_path,
        START=START,
        END=END,
        DATADIR=None,  # Auto-detect based on OS
        wind_output_dir="./wind_extracted",
        solar_output_dir="./solar_extracted",
        use_parallel=True,
        enable_resume=True
    )
    
    if result:
        print(f"\n✅ Extraction completed successfully!")
        print(f"Processing time: {result['processing_time_seconds']:.1f} seconds")
    else:
        print(f"\n❌ Extraction failed!")


if __name__ == "__main__":
    main() 