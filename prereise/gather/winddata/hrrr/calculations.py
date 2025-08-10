import concurrent.futures
import datetime
import functools
import json
import multiprocessing as mp
import os
import time
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pygrib
from powersimdata.utility.distance import ll2uv
from scipy.spatial import KDTree
from tqdm import tqdm  # Assuming you're using tqdm for progress tracking

# Suppress fs package deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, 
                       message=".*pkg_resources.declare_namespace.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, 
                       message=".*fs.*")

import atexit
import signal
import sys

from prereise.gather.const import (DATADIR, END, SEARCHSTRING, SELECTORS,
                                   START, TZ, YEAR)
from prereise.gather.winddata import const
from prereise.gather.winddata.hrrr.helpers import formatted_filename
from prereise.gather.winddata.impute import linear
from prereise.gather.winddata.power_curves import (get_power,
                                                   get_state_power_curves,
                                                   get_turbine_power_curves,
                                                   shift_turbine_curve)

# Global flag to track if we're shutting down
_shutdown_requested = False

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    import gc
    import sys

    # Prevent reentrant calls
    if hasattr(signal_handler, '_shutdown_in_progress'):
        return
    signal_handler._shutdown_in_progress = True
    
    try:
        print(f"\nReceived signal {signum}. Initiating graceful shutdown...")
        
        # Force garbage collection
        gc.collect()
        
        print("Memory cleanup completed. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"Error during shutdown: {e}")
        sys.exit(1)

def cleanup_on_exit():
    """Cleanup function called on exit."""
    import gc

    # Prevent reentrant calls
    if hasattr(cleanup_on_exit, '_cleanup_in_progress'):
        return
    cleanup_on_exit._cleanup_in_progress = True
    
    try:
        print("\nPerforming final cleanup...")
        gc.collect()
        print("Cleanup completed.")
    except Exception as e:
        # Silently ignore cleanup errors to avoid reentrant issues
        pass

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # kill command

# Register cleanup function
atexit.register(cleanup_on_exit)

def check_shutdown_requested():
    """Check if shutdown has been requested."""
    global _shutdown_requested
    return _shutdown_requested

def log_error(e, filename, hours_forecasted="0", message="in something"):
    """logging error"""
    print(
        f"in {filename} ERROR: {e} occured {message} for hours forcast {hours_forecasted}"
    )
    return


def get_wind_data_lat_long(dt, directory, hours_forecasted="0"):
    """Returns the latitude and longitudes of the various
    wind grid sectors. Function assumes that there's data
    for the dt provided and the data lives in the directory.

    :param datetime.datetime dt: date and time of the grib data
    :param str directory: directory where the data is located
    :return: (*tuple*) -- A tuple of 2 same lengthed numpy arrays, first one being
        latitude and second one being longitude.
    """
    try:
        import pygrib
    except ImportError:
        print("pygrib is missing but required for this function")
        raise
    gribs = pygrib.open(
        os.path.join(
            directory,
            formatted_filename(dt, hours_forecasted=hours_forecasted),
        )
    )
    grib = next(gribs)
    # , grib['numberOfDataPoints']
    # grib.keys()
    # grib['latitudeOfFirstGridPointInDegrees']

    return grib.latlons()


def find_closest_wind_grids(wind_farms, wind_data_lat_long):
    """Uses provided wind farm data and wind grid data to calculate
    the closest wind grid to each wind farm.

    :param pandas.DataFrame wind_farms: plant data frame.
    :param tuple wind_data_lat_long: A tuple of 2 same lengthed numpy arrays, first one being
        latitude and second one being longitude.
    :return: (*numpy.array*) -- a numpy array that holds in each index i
        the index of the closest wind grid in wind_data_lat_long for wind_farms i
    """
    grid_lats, grid_lons = (
        wind_data_lat_long[0].flatten(),
        wind_data_lat_long[1].flatten(),
    )
    assert len(grid_lats) == len(grid_lons)
    grid_lat_lon_unit_vectors = [ll2uv(i, j) for i, j in zip(grid_lons, grid_lats)]

    tree = KDTree(grid_lat_lon_unit_vectors)

    wind_farm_lats = wind_farms.lat.values
    wind_farm_lons = wind_farms.lon.values

    wind_farm_unit_vectors = [
        ll2uv(i, j) for i, j in zip(wind_farm_lons, wind_farm_lats)
    ]
    _, indices = tree.query(wind_farm_unit_vectors)

    return indices


def process_grib_file(fn, wind_farm_to_closest_wind_grid_indices):
    try:
        data = {}
        with pygrib.open(fn) as grbs:
            for grb in grbs:
                for key, value in SELECTORS.items():
                    if grb.name == value:
                        data[
                            (
                                fn,
                                key,
                                grb.year,
                                grb.month,
                                grb.day,
                                grb.hour,
                                grb.minute,
                                grb.forecastTime,
                            )
                        ] = grb.values.flatten()[wind_farm_to_closest_wind_grid_indices]
        return data
    except Exception as e:
        return f"Error processing {fn}: {e}"


def process_grib_file_worker(args):
    """Worker function for multiprocessing - must be at module level and pickleable."""
    fn, wind_farm_to_closest_wind_grid_indices, selectors = args
    try:
        data = {}
        with pygrib.open(fn) as grbs:
            for grb in grbs:
                for key, value in selectors.items():
                    if grb.name == value:
                        data[
                            (
                                fn,
                                key,
                                grb.year,
                                grb.month,
                                grb.day,
                                grb.hour,
                                grb.minute,
                                grb.forecastTime,
                            )
                        ] = grb.values.flatten()[wind_farm_to_closest_wind_grid_indices]
        return data
    except Exception as e:
        return f"Error processing {fn}: {e}"


def extract_data_parallel(
    points,
    START,
    END,
    DATADIR,
    DEFAULT_HOURS_FORECASTED,
    SELECTORS,
    WINDSCEN,
    SOLARSCEN,
    use_parallel=True,
):
    """why results has two elements 0 and 1; where does it separate files with 0 and 1?"""
    wind_data_lat_long = get_wind_data_lat_long(START, DATADIR)

    wind_farm_to_closest_wind_grid_indices = find_closest_wind_grids(
        points, wind_data_lat_long
    )
    fns = [
        os.path.join(
            DATADIR,
            formatted_filename(dt, hours_forecasted=hours_forecasted),
        )
        for dt in pd.date_range(start=START, end=END, freq="1h").to_pydatetime()
        for hours_forecasted in DEFAULT_HOURS_FORECASTED
    ]
    max_workers = os.cpu_count()
    partial_process_grib_file = partial(
        process_grib_file,
        wind_farm_to_closest_wind_grid_indices=wind_farm_to_closest_wind_grid_indices,
    )
    if use_parallel:
        print("entering parallel work")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                tqdm(
                    executor.map(partial_process_grib_file, fns),
                    total=len(fns),
                    desc="Overall Progress",
                )
            )
        print("done with parallel work")
    else:
        results = []
        for fn in tqdm(fns, desc="Overall Progress"):
            results.append(partial_process_grib_file(fn))
        # results = []
        # for fn in tqdm(fns, desc="Overall Progress"):
        #     results.append(process_grib_file(fn, wind_farm_to_closest_wind_grid_indices))

    combined_results = {
        key: value for result in results for key, value in result.items()
    }

    filtered_results = {
        key: value
        for key, value in combined_results.items()
        if not (key[0].endswith("f01.grib2") and key[-1] in [0, 60])
    }

    # Assume filtered_results is your dictionary
    # Initialize a defaultdict to collect data for each variable
    variable_data = defaultdict(list)

    # Iterate over the filtered_results dictionary
    for key, data_array in filtered_results.items():
        file_path, variable_name, year, month, day, hour, minute, second = key
        # Create a timestamp from the date and time components
        timestamp = pd.Timestamp(
            year=year, month=month, day=day, hour=hour, minute=second
        )
        # Append the timestamp and data array to the list corresponding to the variable name
        variable_data[variable_name].append({timestamp: data_array})

    # some checks that the data is represeting all hours before flattening the dict
    dates = {}
    for key in variable_data.keys():
        dates[key] = []
        for item in variable_data[key]:
            for k, v in item.items():
                dates[key].append(k)
        dates[key] = sorted(dates[key])
        start = dates[key][0]
        end = dates[key][-1]
        full_dates = pd.date_range(start=start, end=end, freq="15min")
        # full_dates = pd.date_range(start=start, end=end, freq="1h")
        missing_dates = [date for date in full_dates if date not in dates[key]]
        if len(missing_dates) > 0:
            print(f"missing dates for {key}: {missing_dates}")
        # repetitions
        repetitions = [item for item in dates[key] if dates[key].count(item) > 1]
        if len(repetitions) > 0:
            print(f"repetitions for {key}: {repetitions}")

    # now that we know there are no missing or repetitions we can convert the data to df
    output = {}
    for key in variable_data.keys():
        output[key] = pd.DataFrame(
            data={k: v for item in variable_data[key] for k, v in item.items()},
            index=points["pid"].astype(str),
        ).T
        # write it to the parquet file
        fn = (
            key
            + f"_{WINDSCEN}_{SOLARSCEN}_{str(START.year)}_{str(START.month)}_to_{str(END.year)}_{str(END.month)}.parquet"
        )
        output[key].to_parquet(fn)

    output["Wind10"] = np.sqrt(pow(output["UWind10"], 2) + pow(output["VWind10"], 2))
    output["Wind80"] = np.sqrt(pow(output["UWind80"], 2) + pow(output["VWind80"], 2))
    fn = (
        "Wind10"
        + f"_{WINDSCEN}_{SOLARSCEN}_{str(START.year)}_{str(START.month)}_to_{str(END.year)}_{str(END.month)}.parquet"
    )
    output["Wind10"].to_parquet(fn)
    fn = (
        "Wind80"
        + f"_{WINDSCEN}_{SOLARSCEN}_{str(START.year)}_{str(START.month)}_to_{str(END.year)}_{str(END.month)}.parquet"
    )
    output["Wind80"].to_parquet(fn)
    return output


def extract_data_parallel_multiprocessing(
    points,
    START,
    END,
    DATADIR,
    DEFAULT_HOURS_FORECASTED,
    SELECTORS,
    WINDSCEN,
    SOLARSCEN,
    use_parallel=True,
):
    """Version using multiprocessing for true parallelization."""
    wind_data_lat_long = get_wind_data_lat_long(START, DATADIR)

    wind_farm_to_closest_wind_grid_indices = find_closest_wind_grids(
        points, wind_data_lat_long
    )
    fns = [
        os.path.join(
            DATADIR,
            formatted_filename(dt, hours_forecasted=hours_forecasted),
        )
        for dt in pd.date_range(start=START, end=END, freq="1h").to_pydatetime()
        for hours_forecasted in DEFAULT_HOURS_FORECASTED
    ]
    
    if use_parallel:
        print("entering multiprocessing work")
        max_workers = min(os.cpu_count(), len(fns))
        
        # Prepare arguments for multiprocessing
        args_list = [(fn, wind_farm_to_closest_wind_grid_indices, SELECTORS) for fn in fns]
        
        with mp.Pool(processes=max_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(process_grib_file_worker, args_list),
                    total=len(fns),
                    desc="Overall Progress (Multiprocessing)",
                )
            )
        print("done with multiprocessing work")
    else:
        results = []
        for fn in tqdm(fns, desc="Overall Progress"):
            results.append(process_grib_file_worker((fn, wind_farm_to_closest_wind_grid_indices, SELECTORS)))

    combined_results = {
        key: value for result in results for key, value in result.items()
    }

    filtered_results = {
        key: value
        for key, value in combined_results.items()
        if not (key[0].endswith("f01.grib2") and key[-1] in [0, 60])
    }

    # Assume filtered_results is your dictionary
    # Initialize a defaultdict to collect data for each variable
    variable_data = defaultdict(list)

    # Iterate over the filtered_results dictionary
    for key, data_array in filtered_results.items():
        file_path, variable_name, year, month, day, hour, minute, second = key
        # Create a timestamp from the date and time components
        timestamp = pd.Timestamp(
            year=year, month=month, day=day, hour=hour, minute=second
        )
        # Append the timestamp and data array to the list corresponding to the variable name
        variable_data[variable_name].append({timestamp: data_array})

    # some checks that the data is represeting all hours before flattening the dict
    dates = {}
    for key in variable_data.keys():
        dates[key] = []
        for item in variable_data[key]:
            for k, v in item.items():
                dates[key].append(k)
        dates[key] = sorted(dates[key])
        start = dates[key][0]
        end = dates[key][-1]
        full_dates = pd.date_range(start=start, end=end, freq="15min")
        # full_dates = pd.date_range(start=start, end=end, freq="1h")
        missing_dates = [date for date in full_dates if date not in dates[key]]
        if len(missing_dates) > 0:
            print(f"missing dates for {key}: {missing_dates}")
        # repetitions
        repetitions = [item for item in dates[key] if dates[key].count(item) > 1]
        if len(repetitions) > 0:
            print(f"repetitions for {key}: {repetitions}")

    # now that we know there are no missing or repetitions we can convert the data to df
    output = {}
    for key in variable_data.keys():
        output[key] = pd.DataFrame(
            data={k: v for item in variable_data[key] for k, v in item.items()},
            index=points["pid"].astype(str),
        ).T
        # write it to the parquet file
        fn = (
            key
            + f"_{WINDSCEN}_{SOLARSCEN}_{str(START.year)}_{str(START.month)}_to_{str(END.year)}_{str(END.month)}.parquet"
        )
        output[key].to_parquet(fn)

    output["Wind10"] = np.sqrt(pow(output["UWind10"], 2) + pow(output["VWind10"], 2))
    output["Wind80"] = np.sqrt(pow(output["UWind80"], 2) + pow(output["VWind80"], 2))
    fn = (
        "Wind10"
        + f"_{WINDSCEN}_{SOLARSCEN}_{str(START.year)}_{str(START.month)}_to_{str(END.year)}_{str(END.month)}.parquet"
    )
    output["Wind10"].to_parquet(fn)
    fn = (
        "Wind80"
        + f"_{WINDSCEN}_{SOLARSCEN}_{str(START.year)}_{str(START.month)}_to_{str(END.year)}_{str(END.month)}.parquet"
    )
    output["Wind80"].to_parquet(fn)
    return output


def calculate_pout_individual_vectorized(
    wind_speed_data,
    wind_farms,
    start_dt,
    end_dt,
):
    """
    Calculate power output for wind farms based on wind speed data and turbine power curves.

    :param pandas.DataFrame wind_speed_data: Wind speed data indexed by datetime, columns are wind farm 'pid's.
    :param pandas.DataFrame wind_farms: DataFrame containing wind farm information, including:
        'Predominant Turbine Manufacturer', 'Predominant Turbine Model Number',
        'Turbine Hub Height (Feet)', and 'pid'.
    :param str start_dt: Start date (not used in this function but kept for consistency).
    :param str end_dt: End date (not used in this function but kept for consistency).
    :return: pandas.DataFrame containing power output per wind farm at each datetime.
    """

    # Constants and required columns
    const = {
        "mfg_col": "Predominant Turbine Manufacturer",
        "model_col": "Predominant Turbine Model Number",
        "hub_height_col": "Turbine Hub Height (Feet)",
        "max_wind_speed": 30,  # Assuming maximum wind speed for the curves
        "new_curve_res": 0.01,  # Resolution for the shifted power curves
    }

    req_cols = {const["mfg_col"], const["model_col"], const["hub_height_col"], "pid"}
    if not req_cols <= set(wind_farms.columns):
        raise ValueError(f"wind_farms requires columns: {req_cols}")

    # Helper functions (as in your original code)
    def get_starting_curve_name(series, valid_names):
        """Given a wind farm series, build a single string used to look up a wind farm
        power curve. If the specific make and model aren't found, return a default.
        """
        try:
            full_name = " ".join([series[const["mfg_col"]], series[const["model_col"]]])
            full_name = full_name if full_name in valid_names else "IEC class 2"
        except:
            full_name = "IEC class 2"  # Default if any error occurs
        return full_name

    def get_shifted_curve(lookup_name, hub_height, reference_curves):
        """Get the power curve for the given turbine type, shifted to hub height."""
        shifted = shift_turbine_curve(
            reference_curves[lookup_name],
            hub_height,
            const["max_wind_speed"],
            const["new_curve_res"],
        )
        return shifted

    # Load turbine power curves
    turbine_power_curves = get_turbine_power_curves(filename="In_PowerCurves_Dut.csv")
    valid_curve_names = turbine_power_curves.columns

    # Create a lookup for turbine types
    lookup_names = wind_farms.apply(
        lambda x: get_starting_curve_name(x, valid_curve_names),
        axis=1,
    )
    lookup_values = pd.concat(
        [lookup_names.rename("curve_name"), wind_farms[const["hub_height_col"]]], axis=1
    )

    # Cached function for shifting curves
    cached_shifted_curve = functools.lru_cache(maxsize=None)(
        functools.partial(get_shifted_curve, reference_curves=turbine_power_curves)
    )

    # Compute shifted power curves for each wind farm
    shifted_power_curves = {}
    for idx, row in tqdm(
        lookup_values.iterrows(),
        total=lookup_values.shape[0],
        desc="Shifting power curves",
    ):
        pid = wind_farms.loc[idx, "pid"]
        curve_name = row["curve_name"]
        hub_height = row[const["hub_height_col"]]
        shifted_curve = cached_shifted_curve(curve_name, hub_height)
        shifted_power_curves[str(pid)] = shifted_curve

    # Convert shifted_power_curves to DataFrame
    shifted_power_curves_df = pd.DataFrame(
        shifted_power_curves
    ).T  # Transpose to have wind farms as rows
    shifted_power_curves_df.columns.name = "Speed bin (m/s)"

    # Ensure wind_farm IDs are strings to match the DataFrame columns and indices
    wind_farm_ids = wind_farms["pid"].astype(str).tolist()

    # Align wind_speed_data and shifted_power_curves with wind_farm_ids
    wind_speed_data = wind_speed_data[wind_farm_ids]
    shifted_power_curves_df = shifted_power_curves_df.loc[wind_farm_ids]

    # Get the common speed bins from the shifted_power_curves columns
    speed_bins = shifted_power_curves_df.columns.astype(float).values

    # Prepare the output DataFrame
    power_output = pd.DataFrame(
        index=wind_speed_data.index,
        columns=wind_farm_ids,
        dtype=np.float32,  # Use float32 to save memory
    )

    # Vectorized interpolation over time for each wind farm
    for w in tqdm(wind_farm_ids, desc="Calculating power output"):
        # Retrieve wind speeds and power curve for the current wind farm
        wind_speeds_w = wind_speed_data[w].values  # Shape: (n_times,)
        # Shape: (n_speed_bins,)
        power_curve_w = shifted_power_curves_df.loc[w].values

        # Perform interpolation
        power_output_w = np.interp(
            wind_speeds_w,
            speed_bins,
            power_curve_w,
            left=0,
            right=0,
        )

        # Assign the interpolated power output to the DataFrame
        power_output[w] = power_output_w

    # Round the output as per the original code
    power_output = power_output.round(4)

    return power_output


def extract_data_xarray_optimized(
    points,
    START,
    END,
    DATADIR,
    DEFAULT_HOURS_FORECASTED,
    SELECTORS,
    WINDSCEN,
    SOLARSCEN,
    use_parallel=True,
):
    """Optimized GRIB reading using xarray/cfgrib for better performance.
    
    This function provides significant performance improvements over the pygrib approach
    by using vectorized operations and better memory management.
    
    Args:
        points (pd.DataFrame): DataFrame containing lat/lon points
        START (datetime): Start datetime
        END (datetime): End datetime
        DATADIR (str): Directory containing GRIB files
        DEFAULT_HOURS_FORECASTED (list): List of forecast hours to process
        SELECTORS (dict): Dictionary of variables to extract
        WINDSCEN (str): Wind scenario name
        SOLARSCEN (str): Solar scenario name
        use_parallel (bool): Whether to use parallel processing
        
    Returns:
        dict: Dictionary of DataFrames containing extracted data
    """
    try:
        import dask.array as da
        import xarray as xr
        from dask.diagnostics import ProgressBar
    except ImportError:
        print("xarray and dask required for optimized extraction. Falling back to pygrib.")
        return extract_data_parallel(
            points, START, END, DATADIR, DEFAULT_HOURS_FORECASTED, 
            SELECTORS, WINDSCEN, SOLARSCEN, use_parallel
        )
    
    print("Using xarray-optimized GRIB reading...")
    
    # 1. Generate file list
    date_range = pd.date_range(start=START, end=END, freq="1h")
    files = []
    for dt in date_range:
        for hours_forecasted in DEFAULT_HOURS_FORECASTED:
            file_path = os.path.join(
                DATADIR,
                formatted_filename(dt, hours_forecasted=hours_forecasted),
            )
            if os.path.exists(file_path):
                files.append(file_path)
    
    if not files:
        print("No GRIB files found. Check DATADIR and file paths.")
        return {}
    
    print(f"Found {len(files)} GRIB files to process")
    
    # 2. Get grid coordinates from first file
    wind_data_lat_long = get_wind_data_lat_long(START, DATADIR)
    grid_lats, grid_lons = wind_data_lat_long[0].flatten(), wind_data_lat_long[1].flatten()
    
    # 3. Find closest grid points for each location
    grid_lat_lon_unit_vectors = [ll2uv(i, j) for i, j in zip(grid_lons, grid_lats)]
    tree = KDTree(grid_lat_lon_unit_vectors)
    
    wind_farm_unit_vectors = [ll2uv(i, j) for i, j in zip(points.lon.values, points.lat.values)]
    _, indices = tree.query(wind_farm_unit_vectors)
    
    # 4. Create mapping of point IDs to grid indices
    point_to_grid = dict(zip(points.pid.astype(str), indices))
    
    # 5. Process each variable separately for better memory management
    output = {}
    
    for var_name, var_selector in SELECTORS.items():
        print(f"Processing variable: {var_name}")
        
        try:
            # Open files for this variable
            ds = xr.open_mfdataset(
                files,
                engine='cfgrib',
                backend_kwargs={'filter_by_keys': {'shortName': var_selector}},
                chunks={'time': 24, 'latitude': 100, 'longitude': 100},
                parallel=use_parallel,
                combine='nested',
                concat_dim='time'
            )
            
            # Extract data for each point
            point_data = {}
            for pid, grid_idx in point_to_grid.items():
                # Get the lat/lon indices for this grid point
                lat_idx, lon_idx = np.unravel_index(grid_idx, (len(grid_lats), len(grid_lons)))
                
                # Extract data for this point
                point_series = ds.isel(latitude=lat_idx, longitude=lon_idx)
                point_data[pid] = point_series.to_dataframe()[var_name]
            
            # Convert to DataFrame
            df = pd.DataFrame(point_data)
            df.index.name = 'time'
            
            # Save to parquet
            fn = (
                var_name
                + f"_{WINDSCEN}_{SOLARSCEN}_{str(START.year)}_{str(START.month)}_to_{str(END.year)}_{str(END.month)}.parquet"
            )
            df.to_parquet(fn)
            output[var_name] = df
            
            # Close dataset
            ds.close()
            
        except Exception as e:
            print(f"Error processing {var_name}: {e}")
            continue
    
    # 6. Calculate derived wind speeds
    if "UWind10" in output and "VWind10" in output:
        output["Wind10"] = np.sqrt(pow(output["UWind10"], 2) + pow(output["VWind10"], 2))
        fn = (
            "Wind10"
            + f"_{WINDSCEN}_{SOLARSCEN}_{str(START.year)}_{str(START.month)}_to_{str(END.year)}_{str(END.month)}.parquet"
        )
        output["Wind10"].to_parquet(fn)
    
    if "UWind80" in output and "VWind80" in output:
        output["Wind80"] = np.sqrt(pow(output["UWind80"], 2) + pow(output["VWind80"], 2))
        fn = (
            "Wind80"
            + f"_{WINDSCEN}_{SOLARSCEN}_{str(START.year)}_{str(START.month)}_to_{str(END.year)}_{str(END.month)}.parquet"
        )
        output["Wind80"].to_parquet(fn)
    
    return output


def extract_data_cached_optimized(
    points,
    START,
    END,
    DATADIR,
    DEFAULT_HOURS_FORECASTED,
    SELECTORS,
    WINDSCEN,
    SOLARSCEN,
    use_parallel=True,
    cache_dir="./grib_cache",
):
    """Optimized GRIB reading with intelligent caching.
    
    This function adds caching to avoid re-reading the same GRIB files,
    which can significantly speed up repeated runs.
    
    Args:
        points (pd.DataFrame): DataFrame containing lat/lon points
        START (datetime): Start datetime
        END (datetime): End datetime
        DATADIR (str): Directory containing GRIB files
        DEFAULT_HOURS_FORECASTED (list): List of forecast hours to process
        SELECTORS (dict): Dictionary of variables to extract
        WINDSCEN (str): Wind scenario name
        SOLARSCEN (str): Solar scenario name
        use_parallel (bool): Whether to use parallel processing
        cache_dir (str): Directory for caching processed data
        
    Returns:
        dict: Dictionary of DataFrames containing extracted data
    """
    import hashlib
    import os
    import pickle

    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate cache key based on parameters
    cache_params = {
        'start': START,
        'end': END,
        'datadir': DATADIR,
        'hours_forecasted': DEFAULT_HOURS_FORECASTED,
        'selectors': SELECTORS,
        'points_hash': hashlib.md5(points.to_string().encode()).hexdigest()
    }
    
    cache_key = hashlib.md5(str(cache_params).encode()).hexdigest()
    cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
    
    # Check if cached data exists
    if os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading cache: {e}")
    
    # Process data normally
    print("Processing GRIB files (no cache found)...")
    result = extract_data_parallel(
        points, START, END, DATADIR, DEFAULT_HOURS_FORECASTED,
        SELECTORS, WINDSCEN, SOLARSCEN, use_parallel
    )
    
    # Cache the result
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        print(f"Cached data saved to {cache_file}")
    except Exception as e:
        print(f"Error saving cache: {e}")
    
    return result


def extract_data_hybrid_optimized(
    points,
    START,
    END,
    DATADIR,
    DEFAULT_HOURS_FORECASTED,
    SELECTORS,
    WINDSCEN,
    SOLARSCEN,
    use_parallel=True,
    cache_dir="./grib_cache",
):
    """Hybrid optimization combining multiple strategies.
    
    This function automatically chooses the best approach based on:
    - Dataset size (number of files)
    - Available memory
    - Whether xarray is available
    
    Args:
        points (pd.DataFrame): DataFrame containing lat/lon points
        START (datetime): Start datetime
        END (datetime): End datetime
        DATADIR (str): Directory containing GRIB files
        DEFAULT_HOURS_FORECASTED (list): List of forecast hours to process
        SELECTORS (dict): Dictionary of variables to extract
        WINDSCEN (str): Wind scenario name
        SOLARSCEN (str): Solar scenario name
        use_parallel (bool): Whether to use parallel processing
        cache_dir (str): Directory for caching processed data
        
    Returns:
        dict: Dictionary of DataFrames containing extracted data
    """
    # Calculate dataset size
    date_range = pd.date_range(start=START, end=END, freq="1h")
    n_files = len(date_range) * len(DEFAULT_HOURS_FORECASTED)
    
    print(f"Dataset size: {n_files} files over {len(date_range)} hours")
    
    # Check if xarray is available
    try:
        import xarray as xr
        xarray_available = True
    except ImportError:
        xarray_available = False
        print("xarray not available, using pygrib approach")
    
    # Choose approach based on dataset size and available tools
    if xarray_available and n_files > 100:
        print("Using xarray-optimized approach (large dataset)")
        return extract_data_xarray_optimized(
            points, START, END, DATADIR, DEFAULT_HOURS_FORECASTED,
            SELECTORS, WINDSCEN, SOLARSCEN, use_parallel
        )
    elif n_files > 50:
        print("Using cached parallel approach (medium dataset)")
        return extract_data_cached_optimized(
            points, START, END, DATADIR, DEFAULT_HOURS_FORECASTED,
            SELECTORS, WINDSCEN, SOLARSCEN, use_parallel, cache_dir
        )
    else:
        print("Using standard parallel approach (small dataset)")
        return extract_data_parallel(
            points, START, END, DATADIR, DEFAULT_HOURS_FORECASTED,
            SELECTORS, WINDSCEN, SOLARSCEN, use_parallel
        )


def create_global_grid_mapping(grid_lats, grid_lons, output_dir, format="parquet"):
    """Create a single global mapping file for the entire grid in fast format.
    
    Args:
        grid_lats: 2D array of latitudes
        grid_lons: 2D array of longitudes  
        output_dir: Output directory
        format: "parquet" (fast) or "json" (human readable)
    """
    n_lats, n_lons = grid_lats.shape
    total_grid_points = n_lats * n_lons
    
    print(f"Creating global grid mapping for {total_grid_points:,} points in {format.upper()} format...")
    
    if format.lower() == "parquet":
        # Create DataFrame for fast Parquet storage
        mapping_data = []
        for idx in range(total_grid_points):
            lat_idx, lon_idx = np.unravel_index(idx, (n_lats, n_lons))
            lat = float(grid_lats[lat_idx, lon_idx])
            lon = float(grid_lons[lat_idx, lon_idx])
            
            mapping_data.append({
                'grid_id': f"grid_{idx:06d}",
                'lat': lat,
                'lon': lon,
                'grid_index': int(idx),
                'lat_index': int(lat_idx),
                'lon_index': int(lon_idx)
            })
        
        # Create DataFrame and save as Parquet
        import pandas as pd
        mapping_df = pd.DataFrame(mapping_data)
        
        # Save to Parquet (much faster than JSON)
        mapping_filename = "global_grid_mapping.parquet"
        mapping_path = os.path.join(output_dir, "mappings", mapping_filename)
        
        # Ensure the mappings directory exists
        os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
        
        # Time the write operation
        write_start = time.time()
        mapping_df.to_parquet(mapping_path, compression="snappy", engine="pyarrow")
        write_time = time.time() - write_start
        
        file_size_mb = os.path.getsize(mapping_path) / (1024 * 1024)
        
        print(f"Global mapping saved to: {mapping_path}")
        print(f"Mapping contains {len(mapping_data)} grid points")
        print(f"File size: {file_size_mb:.1f} MB")
        print(f"Write time: {write_time:.2f}s")
        
        return mapping_df
        
    else:
        # Fallback to JSON for human readability
        mapping = {}
        for idx in range(total_grid_points):
            lat_idx, lon_idx = np.unravel_index(idx, (n_lats, n_lons))
            lat = float(grid_lats[lat_idx, lon_idx])
            lon = float(grid_lons[lat_idx, lon_idx])
            mapping[f"grid_{idx:06d}"] = {
                'lat': lat,
                'lon': lon,
                'grid_index': int(idx),
                'lat_index': int(lat_idx),
                'lon_index': int(lon_idx)
            }
        
        # Save global mapping to JSON file
        mapping_filename = "global_grid_mapping.json"
        mapping_path = os.path.join(output_dir, "mappings", mapping_filename)
        
        # Ensure the mappings directory exists
        os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
        
        with open(mapping_path, 'w') as f:
            json.dump(mapping, f, indent=2)
        
        print(f"Global mapping saved to: {mapping_path}")
        print(f"Mapping contains {len(mapping)} grid points")
        
        return mapping

def load_global_grid_mapping(output_dir, format="parquet"):
    """Load global grid mapping from file.
    
    Args:
        output_dir: Output directory containing mapping file
        format: "parquet" (fast) or "json" (human readable)
    
    Returns:
        DataFrame (Parquet) or dict (JSON)
    """
    if format.lower() == "parquet":
        mapping_path = os.path.join(output_dir, "mappings", "global_grid_mapping.parquet")
        if os.path.exists(mapping_path):
            import pandas as pd
            print(f"Loading global mapping from: {mapping_path}")
            load_start = time.time()
            mapping_df = pd.read_parquet(mapping_path)
            load_time = time.time() - load_start
            print(f"Loaded {len(mapping_df)} grid points in {load_time:.2f}s")
            return mapping_df
        else:
            print(f"Global mapping file not found: {mapping_path}")
            return None
    else:
        mapping_path = os.path.join(output_dir, "mappings", "global_grid_mapping.json")
        if os.path.exists(mapping_path):
            print(f"Loading global mapping from: {mapping_path}")
            load_start = time.time()
            with open(mapping_path, 'r') as f:
                mapping = json.load(f)
            load_time = time.time() - load_start
            print(f"Loaded {len(mapping)} grid points in {load_time:.2f}s")
            return mapping
        else:
            print(f"Global mapping file not found: {mapping_path}")
            return None



def process_variable_worker(args):
    """Worker function for multiprocessing - must be at module level and pickleable."""
    var_name, var_selector, output_dir, files, grid_lats, grid_lons, n_chunks, chunk_size, compression = args
    
    var_start_time = time.time()
    print(f"Processing variable: {var_name}")
    
    var_output_dir = os.path.join(output_dir, var_name)
    os.makedirs(var_output_dir, exist_ok=True)
    
    # Create mapping directory
    mapping_dir = os.path.join(output_dir, "mappings")
    os.makedirs(mapping_dir, exist_ok=True)
    
    # Create mapping for this variable's chunk
    n_lats, n_lons = grid_lats.shape
    total_grid_points = n_lats * n_lons
    
    var_metadata = {
        'selector': var_selector,
        'chunks': [],
        'file_size_mb': 0,
        'processing_time_seconds': 0
    }
    
    # Process chunks for this variable (sequential within variable to avoid nested parallelism)
    for chunk_idx in tqdm(range(n_chunks), desc=f"Processing {var_name}"):
        chunk_start_time = time.time()
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, total_grid_points)
        chunk_indices = list(range(start_idx, end_idx))
        
        chunk_data = extract_grid_chunk(
            chunk_indices, files, var_selector, grid_lats, grid_lons,
            use_parallel=True  # Use parallel file processing within chunks
        )
        
        if chunk_data is not None and not chunk_data.empty:
            # Round values to 3 decimal digits to reduce file size
            chunk_data = chunk_data.round(3)
            # Create mapping for this chunk
            mapping = create_chunk_mapping(chunk_indices, grid_lats, grid_lons, output_dir, var_name, chunk_idx, format="parquet")
            
            # Save chunk to parquet
            chunk_filename = f"{var_name}_chunk_{chunk_idx:04d}.parquet"
            chunk_path = os.path.join(var_output_dir, chunk_filename)
            
            chunk_data.to_parquet(
                chunk_path,
                compression=compression,
                engine='pyarrow'
            )
            
            # Update metadata
            file_size_mb = os.path.getsize(chunk_path) / (1024 * 1024)
            chunk_time = time.time() - chunk_start_time
            var_metadata['chunks'].append({
                'chunk_idx': chunk_idx,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'filename': chunk_filename,
                'file_size_mb': file_size_mb,
                'processing_time_seconds': chunk_time,
                'mapping_file': f"{var_name}_chunk_{chunk_idx:04d}_mapping.json"
            })
            var_metadata['file_size_mb'] += file_size_mb
            
            print(f"  Chunk {chunk_idx}: {file_size_mb:.1f} MB in {chunk_time:.1f}s")
    
    var_metadata['processing_time_seconds'] = time.time() - var_start_time
    print(f"Completed {var_name}: {var_metadata['file_size_mb']:.1f} MB in {var_metadata['processing_time_seconds']:.1f}s")
    
    return var_name, var_metadata


def process_chunk_with_metadata(args):
    """Process a single chunk with metadata for the new flattened approach."""
    (
        chunk_idx, files, var_selector, grid_lats, grid_lons, chunk_size, total_grid_points, output_dir, var_name, compression, period_str
    ) = args
    
    result = process_chunk((chunk_idx, files, var_selector, grid_lats, grid_lons, chunk_size, total_grid_points, output_dir, var_name, compression))
    if result is not None:
        result['period_str'] = period_str
        result['var_name'] = var_name
    return result


def process_chunk(args):
    """Top-level function for multiprocessing chunk processing."""
    (
        chunk_idx, files, var_selector, grid_lats, grid_lons, chunk_size, total_grid_points, output_dir, var_name, compression
    ) = args
    start_idx = chunk_idx * chunk_size
    end_idx = min(start_idx + chunk_size, total_grid_points)
    chunk_indices = list(range(start_idx, end_idx))
    chunk_data = extract_grid_chunk(
        chunk_indices, files, var_selector, grid_lats, grid_lons, use_parallel=False
    )
    if chunk_data is not None and not chunk_data.empty:
        # Round values to 3 decimal digits to reduce file size
        chunk_data = chunk_data.round(3)
        
        # Create mapping for this chunk
        mapping = create_chunk_mapping(chunk_indices, grid_lats, grid_lons, output_dir, var_name, chunk_idx, format="parquet")
        
        # Save chunk to parquet
        var_output_dir = os.path.join(output_dir, var_name)
        os.makedirs(var_output_dir, exist_ok=True)
        chunk_filename = f"{var_name}_chunk_{chunk_idx:04d}.parquet"
        chunk_path = os.path.join(var_output_dir, chunk_filename)
        chunk_data.to_parquet(
            chunk_path,
            compression=compression,
            engine='pyarrow'
        )
        file_size_mb = os.path.getsize(chunk_path) / (1024 * 1024)
        return {
            'chunk_idx': chunk_idx,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'filename': chunk_filename,
            'file_size_mb': file_size_mb,
            'mapping_file': f"{var_name}_chunk_{chunk_idx:04d}_mapping.json"
        }
    return None


def extract_full_grid_optimized(
    START,
    END,
    DATADIR,
    DEFAULT_HOURS_FORECASTED,
    SELECTORS,
    output_dir="./extracted_grid_data",
    chunk_size=50000,   # SMALLER chunks for better pyarrow compatibility
    compression="snappy",
    use_parallel=True,
    time_period_parallel=True,  # New: parallelize time periods
    num_cpu_workers=4,  # Much more conservative based on actual usage
    num_io_workers=1,   # Reduced I/O workers
    queue_maxsize=8,   # Smaller queue size
    max_file_groups=5000,  # Reduced file group limit
    create_individual_mappings=False,  # Option to create individual mapping files
    parallel_file_writing=True,  # New: parallel file writing
    enable_resume=True,  # New: enable resume functionality
):
    """
    Extract full grid data from GRIB files with timing and numbered column mapping.
    Producer-consumer: CPU workers process chunks, I/O workers write to disk.
    """
    import functools
    import gc
    import glob
    import multiprocessing as mp
    import queue as pyqueue

    # Initialize timing dictionary
    timing = {
        'total_start': time.time(),
        'phases': {},
        'details': {}
    }
    
    def log_timing(phase, start_time=None, details=None):
        """Log timing for a specific phase."""
        if start_time is None:
            start_time = time.time()
        else:
            elapsed = time.time() - start_time
            timing['phases'][phase] = elapsed
            if details:
                timing['details'][phase] = details
            print(f"{phase}: {elapsed:.2f}s")
    
    start_time = time.time()
    print(f"Starting full grid extraction at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Date range: {START} to {END}")
    print(f"Output directory: {output_dir}")
    print(f"Chunk size: {chunk_size:,}")
    print(f"Use parallel: {use_parallel}")
    print(f"Resume enabled: {enable_resume}")
    
    # Phase 1: Setup and directory creation
    setup_start = time.time()
    os.makedirs(output_dir, exist_ok=True)
    mapping_dir = os.path.join(output_dir, "mappings")
    os.makedirs(mapping_dir, exist_ok=True)
    log_timing("Setup", setup_start)
    
    # Phase 1.5: Resume functionality
    resume_start = time.time()
    processed_dates = set()
    date_range = set(pd.date_range(start=START, end=END, freq="1D").date)
    
    if enable_resume:
        print("Checking for existing outputs...")
        
        # Load existing resume metadata
        existing_metadata = load_resume_metadata(output_dir)
        if existing_metadata:
            print(f"Found existing resume metadata from: {existing_metadata['last_updated']}")
            print(f"Previously processed dates: {len(existing_metadata['processed_dates'])}")
        
        # Check for existing outputs
        processed_dates = get_processed_date_range(output_dir, SELECTORS)
        
        if processed_dates:
            print(f"Found {len(processed_dates)} already processed dates:")
            for date in sorted(processed_dates):
                print(f"  - {date}")
            
            # Calculate remaining dates
            remaining_dates = date_range - processed_dates
            print(f"Remaining dates to process: {len(remaining_dates)}")
            
            if not remaining_dates:
                print("All dates already processed! Extraction complete.")
                return {
                    "status": "completed", 
                    "processing_time_seconds": 0,
                    "timing": timing,
                    "resume_info": {
                        "processed_dates": len(processed_dates),
                        "remaining_dates": 0,
                        "resume_used": True
                    }
                }
            
            # Update START to the first remaining date
            if remaining_dates:
                new_start = min(remaining_dates)
                if new_start != START.date():
                    print(f"Resuming from {new_start} (original start: {START.date()})")
                    START = datetime.datetime.combine(new_start, START.time())
        else:
            print("No existing outputs found. Starting fresh extraction.")
    
    log_timing("Resume check", resume_start, {
        'processed_dates': len(processed_dates),
        'remaining_dates': len(date_range - processed_dates)
    })
    
    # Phase 2: Get grid dimensions and metadata
    grid_start = time.time()
    print("Extracting grid metadata...")
    wind_data_lat_long = get_wind_data_lat_long(START, DATADIR)
    grid_lats, grid_lons = wind_data_lat_long[0], wind_data_lat_long[1]
    
    n_lats, n_lons = grid_lats.shape
    total_grid_points = n_lats * n_lons
    
    print(f"Grid dimensions: {n_lats} x {n_lons} = {total_grid_points:,} total points")
    log_timing("Grid metadata extraction", grid_start)
    
    # Phase 3: Create global mapping file (using fast Parquet format)
    mapping_start = time.time()
    global_mapping = create_global_grid_mapping(grid_lats, grid_lons, output_dir, format="parquet")
    log_timing("Global mapping creation", mapping_start)
    
    # Phase 4: Find all GRIB files
    file_search_start = time.time()
    print(f"Searching for GRIB files in: {DATADIR}")
    
    # Use remaining dates if resume is enabled, otherwise use full range
    search_date_range = date_range - processed_dates if enable_resume else date_range
    print(f"Searching for dates: {len(search_date_range)} dates")
    
    all_files = []
    for date in sorted(search_date_range):
        date_str = date.strftime("%Y%m%d")
        date_dir = os.path.join(DATADIR, date_str)
        if os.path.exists(date_dir):
            grib_pattern = os.path.join(date_dir, "*.grib2")
            date_files = glob.glob(grib_pattern)
            # Filter out subset files which appear to be corrupted
            valid_files = [f for f in date_files if "subset_" not in os.path.basename(f)]
            all_files.extend(valid_files)
            print(f"  Found {len(date_files)} total files, using {len(valid_files)} valid files in {date_str}")
        else:
            print(f"  No directory found for {date_str}")
    if not all_files:
        print(f"No GRIB files found in date range {min(search_date_range)} to {max(search_date_range)}")
        return {}
    print(f"Found {len(all_files)} total GRIB files")
    files = all_files
    print(f"Using {len(files)} GRIB files for extraction")
    log_timing("File discovery", file_search_start, {
        'total_files': len(files),
        'search_dates': len(search_date_range),
        'processed_dates_skipped': len(processed_dates) if enable_resume else 0
    })
    
    # Phase 5: Calculate chunking strategy
    chunk_calc_start = time.time()
    n_chunks = (total_grid_points + chunk_size - 1) // chunk_size
    print(f"Processing in {n_chunks} chunks")
    log_timing("Chunk calculation", chunk_calc_start, {'n_chunks': n_chunks})
    
    # Phase 6: Memory-aware parallel processing setup
    memory_setup_start = time.time()
    available_memory_gb = get_available_memory()
    safe_worker_count = calculate_safe_worker_count(available_memory_gb, chunk_size, len(SELECTORS))
    
    print(f"Available memory: {available_memory_gb:.1f} GB")
    print(f"Safe worker count: {safe_worker_count}")
    print(f"Requested worker count: {num_cpu_workers}")
    
    # Use the smaller of safe_worker_count and num_cpu_workers
    actual_workers = min(safe_worker_count, num_cpu_workers)
    print(f"Using {actual_workers} workers")
    
    # Monitor initial memory state
    print(f"Initial memory usage: {get_memory_usage():.1f} MB")
    monitor_memory_usage()
    log_timing("Memory setup", memory_setup_start, {'actual_workers': actual_workers})
    
    # Phase 7: Prepare tasks
    task_prep_start = time.time()
    tasks = []
    for var_name, var_selector in SELECTORS.items():
        for chunk_idx in range(n_chunks):
            tasks.append((chunk_idx, var_name, var_selector))
    print(f"Total tasks: {len(tasks)} (variables × chunks)")
    log_timing("Task preparation", task_prep_start, {'total_tasks': len(tasks)})
    
    # Phase 8: Process chunks in parallel
    processing_start = time.time()
    cpu_func = functools.partial(
        cpu_worker,
        files=files,
        grid_lats=grid_lats,
        grid_lons=grid_lons,
        chunk_size=chunk_size,
        total_grid_points=total_grid_points,
        output_dir=output_dir,
        max_file_groups=max_file_groups,
        create_individual_mappings=create_individual_mappings,
    )
    
    print("Processing chunks in parallel...")
    print(f"Memory usage before processing: {get_memory_usage():.1f} MB")
    
    # Process in smaller batches to avoid memory buildup
    batch_size = min(actual_workers * 2, len(tasks))  # Process 2x workers at a time
    print(f"Processing in batches of {batch_size} tasks")
    
    all_results = []
    batch_times = []
    
    for i in range(0, len(tasks), batch_size):
        # Check for shutdown request
        if check_shutdown_requested():
            print("Shutdown requested. Stopping processing...")
            break
            
        batch_start = time.time()
        batch_tasks = tasks[i:i+batch_size]
        batch_num = i//batch_size + 1
        total_batches = (len(tasks) + batch_size - 1)//batch_size
        print(f"Processing batch {batch_num}/{total_batches}")
        
        with mp.Pool(actual_workers) as pool:
            batch_results = pool.map(cpu_func, batch_tasks)
        
        all_results.extend([r for r in batch_results if r is not None])
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        
        # Monitor memory after each batch
        current_memory = get_memory_usage()
        print(f"Batch {batch_num} completed in {batch_time:.2f}s, memory: {current_memory:.1f} MB")
        monitor_memory_usage()
        
        # Force garbage collection between batches
        gc.collect()
    
    # Check if we were interrupted
    if check_shutdown_requested():
        print("Processing was interrupted. Saving partial results...")
        # Update resume metadata with what we've processed so far
        if enable_resume:
            partial_processed_dates = get_processed_date_range(output_dir, SELECTORS)
            create_resume_metadata(output_dir, START, END, SELECTORS, partial_processed_dates)
            print(f"Partial progress saved. Processed dates: {len(partial_processed_dates)}")
        
        return {
            "status": "interrupted",
            "processing_time_seconds": time.time() - start_time,
            "timing": timing,
            "resume_info": {
                "processed_dates": len(processed_dates),
                "remaining_dates": len(date_range - processed_dates),
                "resume_used": enable_resume and len(processed_dates) > 0,
                "interrupted": True
            }
        }
    
    processing_time = time.time() - processing_start
    log_timing("Parallel processing", processing_start, {
        'total_batches': len(batch_times),
        'avg_batch_time': sum(batch_times)/len(batch_times) if batch_times else 0,
        'total_results': len(all_results)
    })
    
    print(f"Memory usage after processing: {get_memory_usage():.1f} MB")
    
    # Force garbage collection after processing
    gc.collect()
    print(f"Memory usage after cleanup: {get_memory_usage():.1f} MB")
    
    # Phase 9: Write results to files organized by date
    writing_start = time.time()
    
    if parallel_file_writing:
        print("Writing results to files organized by date (PARALLEL)...")
        
        # Prepare file writing tasks
        file_writing_tasks = []
        for result in all_results:
            if result is not None:
                chunk_data, mapping, chunk_idx, var_name = result
                file_writing_tasks.append((
                    chunk_data, mapping, chunk_idx, var_name, 
                    output_dir, compression, create_individual_mappings
                ))
        
        print(f"Prepared {len(file_writing_tasks)} file writing tasks")
        
        # Use parallel file writing
        if file_writing_tasks:
            # Use a reasonable number of I/O workers (not too many to avoid disk contention)
            io_workers = min(num_io_workers, len(file_writing_tasks), 8)  # Cap at 8 to avoid disk contention
            print(f"Using {io_workers} I/O workers for parallel file writing")
            
            with mp.Pool(io_workers) as pool:
                file_writing_results = list(tqdm(
                    pool.imap(file_writing_worker, file_writing_tasks),
                    total=len(file_writing_tasks),
                    desc="Writing files in parallel"
                ))
            
            # Aggregate write statistics
            total_files_written = 0
            total_size_mb = 0
            total_write_time = 0
            write_times = []
            
            for result in file_writing_results:
                if result is not None:
                    total_files_written += result['files_written']
                    total_size_mb += result['total_size_mb']
                    total_write_time += result['write_time']
                    write_times.append(result['write_time'])
            
            print(f"Parallel file writing completed:")
            print(f"  Files written: {total_files_written}")
            print(f"  Total size: {total_size_mb:.1f} MB")
            print(f"  Total write time: {total_write_time:.2f}s")
            if write_times:
                print(f"  Average write time per task: {sum(write_times)/len(write_times):.2f}s")
            
            log_timing("File writing (parallel)", writing_start, {
                'files_written': total_files_written,
                'total_size_mb': total_size_mb,
                'avg_write_time': sum(write_times)/len(write_times) if write_times else 0,
                'io_workers': io_workers
            })
        else:
            print("No files to write")
            log_timing("File writing (parallel)", writing_start, {
                'files_written': 0,
                'total_size_mb': 0,
                'avg_write_time': 0
            })
    
    else:
        # Sequential file writing (original approach)
        print("Writing results to files organized by date (SEQUENTIAL)...")
        
        write_stats = {
            'files_written': 0,
            'total_size_mb': 0,
            'write_times': []
        }
        
        for result in all_results:
            if result is not None:
                chunk_data, mapping, chunk_idx, var_name = result
                
                # Group data by date
                chunk_data['date'] = chunk_data.index.date
                date_groups = chunk_data.groupby('date')
                
                for date, date_data in date_groups:
                    file_write_start = time.time()
                    
                    # Remove the date column and keep only the data
                    date_data = date_data.drop('date', axis=1)
                    date_str = date.strftime('%Y%m%d')
                    
                    # Create date-specific directories
                    var_output_dir = os.path.join(output_dir, var_name, date_str)
                    os.makedirs(var_output_dir, exist_ok=True)
                    
                    # Write parquet file for this date
                    chunk_filename = f"{var_name}_chunk_{chunk_idx:04d}.parquet"
                    chunk_path = os.path.join(var_output_dir, chunk_filename)
                    
                    # Time the actual file write
                    write_start = time.time()
                    
                    # 🔍 NaN DETECTION: Check for missing data
                    nan_check_start = time.time()
                    nan_counts = date_data.isna().sum()
                    total_nans = nan_counts.sum()
                    
                    if total_nans > 0:
                        print(f"  WARNING: Found {total_nans} NaN values in {chunk_filename}")
                        print(f"     Variable: {var_name}")
                        print(f"     Date: {date_str}")
                        print(f"     NaN breakdown by column:")
                        for col, nan_count in nan_counts.items():
                            if nan_count > 0:
                                print(f"       {col}: {nan_count} NaN values")
                        
                        # Log to file for detailed analysis
                        nan_log_file = os.path.join(output_dir, "nan_detection.log")
                        with open(nan_log_file, 'a') as f:
                            f.write(f"{datetime.datetime.now().isoformat()} | {var_name} | {date_str} | {chunk_filename} | Total NaNs: {total_nans}\n")
                            for col, nan_count in nan_counts.items():
                                if nan_count > 0:
                                    f.write(f"  {col}: {nan_count} NaN values\n")
                    
                                # Ensure data types are compatible with pyarrow
            date_data = date_data.astype('float32')  # Use float32 for better compatibility
            
            # Try pyarrow first, fallback to fastparquet if it fails
            try:
                date_data.to_parquet(chunk_path, compression=compression, engine='pyarrow')
                engine_used = 'pyarrow'
            except Exception as e:
                print(f"  pyarrow failed for {chunk_filename}, trying fastparquet: {e}")
                try:
                    date_data.to_parquet(chunk_path, compression=compression, engine='fastparquet')
                    engine_used = 'fastparquet'
                    print(f"  fastparquet succeeded for {chunk_filename}")
                except Exception as e2:
                    print(f"  Both pyarrow and fastparquet failed for {chunk_filename}: {e2}")
                    raise e2
            
            write_time = time.time() - write_start
            
            file_size_mb = os.path.getsize(chunk_path) / (1024 * 1024)
            write_stats['files_written'] += 1
            write_stats['total_size_mb'] += file_size_mb
            write_stats['write_times'].append(write_time)
            
            print(f"  Wrote {chunk_filename} ({file_size_mb:.1f} MB) in {write_time:.2f}s")
                    
            # Optionally write individual mapping files if requested
            if create_individual_mappings:
                mapping_dir = os.path.join(output_dir, "mappings", var_name, date_str)
                os.makedirs(mapping_dir, exist_ok=True)
                mapping_filename = f"{var_name}_chunk_{chunk_idx:04d}_mapping.json"
                mapping_path = os.path.join(mapping_dir, mapping_filename)
                with open(mapping_path, 'w') as f:
                    json.dump(mapping, f, indent=2)
        
        log_timing("File writing (sequential)", writing_start, {
            'files_written': write_stats['files_written'],
            'total_size_mb': write_stats['total_size_mb'],
            'avg_write_time': sum(write_stats['write_times'])/len(write_stats['write_times']) if write_stats['write_times'] else 0
        })
    
    print("\nParallel processing complete! All chunks processed and written.")
    
    # Phase 10: Post-process to add wind speed calculations
    post_process_start = time.time()
    print("Calculating wind speed from U and V components...")
    add_wind_speed_calculations(output_dir)
    log_timing("Post-processing", post_process_start)
    
    # Phase 11: Final timing summary
    total_time = time.time() - start_time
    timing['total_time'] = total_time
    
    print(f"\n{'='*60}")
    print(f"EXTRACTION COMPLETED")
    print(f"{'='*60}")
    print(f"Total processing time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"Output directory: {output_dir}")
    print(f"Mapping files saved in: {mapping_dir}")
    
    # Detailed timing breakdown
    print(f"\nTIMING BREAKDOWN:")
    print(f"{'='*40}")
    for phase, elapsed in timing['phases'].items():
        percentage = (elapsed / total_time) * 100
        print(f"{phase:25s}: {elapsed:6.2f}s ({percentage:5.1f}%)")
    
    # Identify bottlenecks
    if timing['phases']:
        max_phase = max(timing['phases'].items(), key=lambda x: x[1])
        print(f"\nMAIN BOTTLENECK: {max_phase[0]} ({max_phase[1]:.2f}s, {(max_phase[1]/total_time)*100:.1f}%)")
        
        if max_phase[0] == "File writing":
            print("Suggestion: Consider using faster storage or reducing compression")
        elif max_phase[0] == "Parallel processing":
            print("Suggestion: Consider increasing workers or reducing chunk size")
        elif max_phase[0] == "File discovery":
            print("Suggestion: Consider using SSD storage for GRIB files")
    
    print(f"{'='*60}")
    
    # Update resume metadata
    if enable_resume:
        # Get all processed dates after completion
        final_processed_dates = get_processed_date_range(output_dir, SELECTORS)
        create_resume_metadata(output_dir, START, END, SELECTORS, final_processed_dates)
        print(f"Resume metadata updated. Total processed dates: {len(final_processed_dates)}")
    
    return {
        "status": "completed", 
        "processing_time_seconds": total_time,
        "timing": timing,
        "resume_info": {
            "processed_dates": len(processed_dates),
            "remaining_dates": len(date_range - processed_dates),
            "resume_used": enable_resume and len(processed_dates) > 0
        }
    }


def extract_grid_chunk(
    chunk_indices, files, var_selector, grid_lats, grid_lons, use_parallel=True, max_file_groups=20000
):
    """Extract data for a specific chunk of grid points, with 15-min resolution."""
    import gc  # For garbage collection
    import re
    from functools import partial

    import pygrib
    n_lats, n_lons = grid_lats.shape

    chunk_start = time.time()
    
    # Group files by hour, separating f00 and f01
    grouping_start = time.time()
    file_groups = {}
    for file_path in files:
        # Extract hour and forecast type (f00 or f01) from HRRR naming convention
        # Format: data/YYYYMMDD/hrrr.t{HH}z.wrfsubhf{XX}.grib2
        # or just: hrrr.t{HH}z.wrfsubhf{XX}.grib2
        filename = os.path.basename(file_path)
        match = re.search(r'hrrr\.t(\d{2})z\.wrfsubhf(\d{2})\.grib2', filename)
        if not match:
            continue
        hour_str, forecast_str = match.groups()
        fxx = f"f{forecast_str}"  # Add 'f' prefix back for consistency
        
        # Extract date from directory path if available
        dir_path = os.path.dirname(file_path)
        date_match = re.search(r'(\d{8})', dir_path)  # Look for YYYYMMDD in path
        if date_match:
            date_str = date_match.group(1)
            key = f"{date_str}_{hour_str}"
        else:
            # If no date in path, use just hour (we'll get date from GRIB file)
            key = f"hour_{hour_str}"
            
        if key not in file_groups:
            file_groups[key] = {}
        file_groups[key][fxx] = file_path

    # Initialize chunk_data dictionary to store extracted data
    chunk_data = {}
    
    # Initialize chunk_data dictionary to store extracted data
    
    # Limit the number of file groups to process to prevent memory issues
    # For a full year, we need to handle ~8760 file groups (365 days × 24 hours)
    if max_file_groups is not None and len(file_groups) > max_file_groups:
        print(f"Warning: {len(file_groups)} file groups found, processing only first {max_file_groups}")
        print("This may truncate your data. Consider processing in smaller time periods if needed.")
        # Sort by key to ensure consistent processing
        sorted_keys = sorted(file_groups.keys())
        file_groups = {k: file_groups[k] for k in sorted_keys[:max_file_groups]}
    else:
        # Only print this once per chunk to reduce noise
        if not hasattr(extract_grid_chunk, '_printed_file_groups'):
            print(f"Processing {len(file_groups)} file groups (full dataset)")
            extract_grid_chunk._printed_file_groups = True
    
    grouping_time = time.time() - grouping_start
    
    def extract_values_memory_efficient(grb, chunk_indices):
        """Extract values for chunk indices without loading entire grid into memory."""
        try:
            # Convert all chunk indices to 2D indices at once (more efficient)
            lat_indices, lon_indices = np.unravel_index(chunk_indices, (n_lats, n_lons))
            
            # Extract values using advanced indexing (more memory efficient)
            values_2d = grb.values
            chunk_values = values_2d[lat_indices, lon_indices]
            
            # Convert to float32 to save memory
            return chunk_values.astype(np.float32)
        except MemoryError as e:
            print(f"Memory error in extraction: {e}")
            # Fallback: process in smaller batches
            try:
                batch_size = 10000  # Process 10k points at a time
                chunk_values = []
                for i in range(0, len(chunk_indices), batch_size):
                    batch_indices = chunk_indices[i:i+batch_size]
                    lat_indices, lon_indices = np.unravel_index(batch_indices, (n_lats, n_lons))
                    values_2d = grb.values
                    batch_values = values_2d[lat_indices, lon_indices]
                    chunk_values.extend(batch_values)
                    del values_2d  # Explicitly delete to free memory
                    gc.collect()
                return np.array(chunk_values, dtype=np.float32)
            except Exception as e2:
                print(f"Fallback extraction also failed: {e2}")
                return None
        except Exception as e:
            print(f"Error in memory-efficient extraction: {e}")
            return None
    
    # Time the file processing
    file_processing_start = time.time()
    files_processed = 0
    grib_read_time = 0
    data_extraction_time = 0
    
    for key in sorted(file_groups.keys()):
        group = file_groups[key]
        
        # :00 from f00
        if 'f00' in group:
            try:
                grib_read_start = time.time()
                with pygrib.open(group['f00']) as grbs:
                    variable_found = False
                    for grb in grbs:
                        try:
                            # Use exact matching like the original function
                            if grb.name == var_selector:
                                
                                # Get timestamp from GRIB message
                                timestamp = pd.Timestamp(
                                    year=grb.year, month=grb.month, day=grb.day,
                                    hour=grb.hour, minute=grb.minute
                                )
                                
                                # Use memory-efficient extraction
                                extraction_start = time.time()
                                values = extract_values_memory_efficient(grb, chunk_indices)
                                data_extraction_time += time.time() - extraction_start
                                
                                if values is not None:
                                    chunk_columns = [f"grid_{i:06d}" for i in range(len(chunk_indices))]
                                    chunk_data[timestamp] = dict(zip(chunk_columns, values))
                                    variable_found = True
                                break
                        except MemoryError as me:
                            print(f"Memory error processing GRIB message in {group['f00']}: {me}")
                            gc.collect()
                            continue
                        except Exception as e:
                            print(f"Error processing GRIB message in {group['f00']}: {e}")
                            continue
                        finally:
                            # Force garbage collection after each GRIB message
                            del grb
                            gc.collect()
                    
                    # Variable not found, continue silently
                        
            except Exception as e:
                print(f"Error reading f00 {group['f00']}: {e}")
                gc.collect()  # Clean up memory on error
            
            grib_read_time += time.time() - grib_read_start
            files_processed += 1
        
        # :15, :30, :45 from f01
        if 'f01' in group:
            try:
                grib_read_start = time.time()
                with pygrib.open(group['f01']) as grbs:
                    f01_processed = False
                    for grb in grbs:
                        try:
                            # Use exact matching like the original function
                            if grb.name == var_selector:
                                
                                # Look for time offset in grb (e.g., '15 mins', '30 mins', '45 mins')
                                grb_str = str(grb)
                                
                                # Process ALL time offsets found in this message
                                for offset, minute in [(15, 15), (30, 30), (45, 45)]:
                                    # Look for both formats: "15m mins" and "15 mins"
                                    if f"{offset}m mins" in grb_str or f"{offset} mins" in grb_str:
                                        # Get base timestamp from GRIB message
                                        base_timestamp = pd.Timestamp(
                                            year=grb.year, month=grb.month, day=grb.day,
                                            hour=grb.hour, minute=grb.minute
                                        )
                                        dt = base_timestamp + pd.Timedelta(minutes=minute)
                                        
                                        # Use memory-efficient extraction
                                        extraction_start = time.time()
                                        values = extract_values_memory_efficient(grb, chunk_indices)
                                        data_extraction_time += time.time() - extraction_start
                                        
                                        if values is not None:
                                            chunk_columns = [f"grid_{i:06d}" for i in range(len(chunk_indices))]
                                            chunk_data[dt] = dict(zip(chunk_columns, values))
                                            f01_processed = True
                                
                                # Don't break here - continue processing other messages for the same variable
                                # This allows us to find multiple time offsets in the same file
                                
                        except MemoryError as me:
                            print(f"Memory error processing GRIB message in {group['f01']}: {me}")
                            gc.collect()
                            continue
                        except Exception as e:
                            print(f"Error processing GRIB message in {group['f01']}: {e}")
                            continue
                        finally:
                            # Force garbage collection after each GRIB message
                            del grb
                            gc.collect()
                        
                grib_read_time += time.time() - grib_read_start
                files_processed += 1
                        
            except Exception as e:
                print(f"Error reading f01 {group['f01']}: {e}")
                gc.collect()  # Clean up memory on error
        
        # Force garbage collection after processing each file group
        gc.collect()
    
    file_processing_time = time.time() - file_processing_start
    
    if not chunk_data:
        return pd.DataFrame()
    
    # Time the DataFrame conversion
    df_conversion_start = time.time()
    # Convert to DataFrame, sort by time, round to 3 decimals
    df = pd.DataFrame.from_dict(chunk_data, orient='index').sort_index()
    df.index.name = 'time'
    # Round to exactly 3 decimal places by multiplying by 1000, rounding, then dividing
    df = (df * 1000).round().astype('int32') / 1000.0
    
    # Final garbage collection
    del chunk_data
    gc.collect()
    
    df_conversion_time = time.time() - df_conversion_start
    total_chunk_time = time.time() - chunk_start
    
    print(f"Chunk timing: grouping={grouping_time:.2f}s, file_processing={file_processing_time:.2f}s (grib_read={grib_read_time:.2f}s, data_extraction={data_extraction_time:.2f}s), df_conversion={df_conversion_time:.2f}s, total={total_chunk_time:.2f}s")
    
    return df


def process_single_file_worker(args):
    """Worker function for single file processing - must be at module level and pickleable."""
    file_path, var_selector, chunk_indices, grid_lats, grid_lons = args
    try:
        file_data = {}
        with pygrib.open(file_path) as grbs:
            # For wind speed calculation, we need both U and V components
            if var_selector in ["WindSpeed10", "WindSpeed80"]:
                u_values = None
                v_values = None
                
                # Read all messages to find U and V components
                for grb in grbs:
                    if var_selector == "WindSpeed10":
                        if grb.name == "10 metre U wind component":
                            u_values = grb.values.flatten()[chunk_indices]
                        elif grb.name == "10 metre V wind component":
                            v_values = grb.values.flatten()[chunk_indices]
                    elif var_selector == "WindSpeed80":
                        if grb.name == "U component of wind":
                            u_values = grb.values.flatten()[chunk_indices]
                        elif grb.name == "V component of wind":
                            v_values = grb.values.flatten()[chunk_indices]
                    
                    # If we have both U and V, calculate wind speed
                    if u_values is not None and v_values is not None:
                        # Calculate wind speed magnitude: sqrt(u^2 + v^2)
                        wind_speed_values = np.sqrt(u_values**2 + v_values**2)
                        
                        # Get timestamp from the last processed message
                        timestamp = pd.Timestamp(
                            year=grb.year, month=grb.month, day=grb.day,
                            hour=grb.hour, minute=grb.minute
                        )
                        
                        # Create numbered column names for this chunk
                        chunk_columns = [f"grid_{i:06d}" for i in range(len(chunk_indices))]
                        file_data[timestamp] = dict(zip(chunk_columns, wind_speed_values))
                        break
            else:
                # For non-wind variables, process normally
                for grb in grbs:
                    if grb.name == var_selector:
                        # Get timestamp
                        timestamp = pd.Timestamp(
                            year=grb.year, month=grb.month, day=grb.day,
                            hour=grb.hour, minute=grb.minute
                        )
                        # Extract values for chunk indices
                        values = grb.values.flatten()
                        chunk_values = values[chunk_indices]
                        
                        # Create numbered column names for this chunk (more compact)
                        chunk_columns = [f"grid_{i:06d}" for i in range(len(chunk_indices))]
                        file_data[timestamp] = dict(zip(chunk_columns, chunk_values))
                        break  # Found the variable, move to next file
        return file_data
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return {}


def extract_points_from_full_grid(
    points, START, END, grid_data_dir="./extracted_grid_data"
):
    """Extract data for specific points from pre-extracted full grid data.
    
    This function reads the full grid parquet files and extracts only the
    data for the specified points, providing fast access to any location.
    
    Args:
        points (pd.DataFrame): DataFrame with lat/lon points
        START (datetime): Start datetime
        END (datetime): End datetime
        grid_data_dir (str): Directory containing full grid parquet files
        
    Returns:
        dict: Dictionary of DataFrames containing extracted data
    """
    import json
    import os

    from powersimdata.utility.distance import ll2uv
    from scipy.spatial import KDTree
    
    print(f"Extracting points from full grid data...")
    
    # 1. Load extraction metadata
    metadata_file = os.path.join(grid_data_dir, "extraction_metadata.json")
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Grid data not found. Run extract_full_grid_optimized first.")
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # 2. Find closest grid points for each location
    grid_lats = metadata['grid_dimensions']['n_lats']
    grid_lons = metadata['grid_dimensions']['n_lons']
    
    # Reconstruct grid coordinates (approximate - you might want to store actual coords)
    lat_coords = np.linspace(20, 50, grid_lats)  # Approximate HRRR lat range
    lon_coords = np.linspace(-140, -60, grid_lons)  # Approximate HRRR lon range
    
    # Create grid point coordinates
    grid_lat_lon_unit_vectors = []
    grid_coords = []
    for i, lat in enumerate(lat_coords):
        for j, lon in enumerate(lon_coords):
            grid_lat_lon_unit_vectors.append(ll2uv(lon, lat))
            grid_coords.append((lat, lon))
    
    tree = KDTree(grid_lat_lon_unit_vectors)
    
    # Find closest grid points for each location
    wind_farm_unit_vectors = [ll2uv(lon, lat) for lat, lon in zip(points.lat.values, points.lon.values)]
    _, indices = tree.query(wind_farm_unit_vectors)
    
    # Create mapping of point IDs to grid coordinates
    point_to_grid = {}
    for i, (pid, grid_idx) in enumerate(zip(points.pid, indices)):
        lat_idx, lon_idx = np.unravel_index(grid_idx, (grid_lats, grid_lons))
        lat = lat_coords[lat_idx]
        lon = lon_coords[lon_idx]
        col_name = f"lat_{lat:.3f}_lon_{lon:.3f}"
        point_to_grid[str(pid)] = col_name
    
    # 3. Extract data for each variable
    output = {}
    
    for var_name, var_metadata in metadata['variables'].items():
        print(f"Processing variable: {var_name}")
        
        var_data_dir = os.path.join(grid_data_dir, var_name)
        if not os.path.exists(var_data_dir):
            print(f"Warning: Variable {var_name} not found in grid data")
            continue
        
        # Read all chunks for this variable
        var_data = []
        for chunk_info in var_metadata['chunks']:
            chunk_file = os.path.join(var_data_dir, chunk_info['filename'])
            if os.path.exists(chunk_file):
                chunk_df = pd.read_parquet(chunk_file)
                var_data.append(chunk_df)
        
        if not var_data:
            continue
        
        # Combine all chunks
        full_var_data = pd.concat(var_data, axis=1)
        
        # Extract only the columns we need
        needed_columns = list(point_to_grid.values())
        available_columns = [col for col in needed_columns if col in full_var_data.columns]
        
        if available_columns:
            extracted_data = full_var_data[available_columns].copy()
            
            # Rename columns to point IDs
            reverse_mapping = {v: k for k, v in point_to_grid.items()}
            extracted_data.columns = [reverse_mapping.get(col, col) for col in extracted_data.columns]
            
            # Filter by time range
            mask = (extracted_data.index >= START) & (extracted_data.index <= END)
            extracted_data = extracted_data.loc[mask]
            
            output[var_name] = extracted_data
            
            # Save to parquet
            fn = f"{var_name}_extracted_{START.year}_{START.month}_to_{END.year}_{END.month}.parquet"
            extracted_data.to_parquet(fn)
    
    return output


def cpu_worker(task, files, grid_lats, grid_lons, chunk_size, total_grid_points, output_dir, max_file_groups=5000, create_individual_mappings=False):
    """CPU worker for producer-consumer pattern."""
    import gc
    
    chunk_idx, var_name, var_selector = task
    start_idx = chunk_idx * chunk_size
    end_idx = min(start_idx + chunk_size, total_grid_points)
    chunk_indices = list(range(start_idx, end_idx))
    
    worker_start = time.time()
    
    # Monitor memory before processing
    memory_before = get_memory_usage()
    monitor_memory_usage()  # Check system memory
    
    # Force garbage collection before processing
    gc.collect()
    
    # Time the actual GRIB extraction
    extraction_start = time.time()
    chunk_data = extract_grid_chunk(chunk_indices, files, var_selector, grid_lats, grid_lons, use_parallel=False, max_file_groups=max_file_groups)
    extraction_time = time.time() - extraction_start
    
    if chunk_data is not None and not chunk_data.empty:
        # Time the data processing
        processing_start = time.time()
        
        # Round to exactly 3 decimal places by multiplying by 1000, rounding, then dividing
        chunk_data = (chunk_data * 1000).round().astype('int32') / 1000.0
        
        # Create chunk mapping (but don't write individual files - use global mapping)
        mapping = create_chunk_mapping(chunk_indices, grid_lats, grid_lons, output_dir, var_name, chunk_idx, create_individual_mappings=create_individual_mappings, format="parquet")
        
        processing_time = time.time() - processing_start
        
        # Monitor memory after processing
        memory_after = get_memory_usage()
        if memory_after - memory_before > 25:  # If memory increased by more than 25MB
            print(f"Memory spike in {var_name} chunk {chunk_idx}: {memory_before:.1f} -> {memory_after:.1f} MB")
        
        # Force aggressive garbage collection
        gc.collect()
        
        total_worker_time = time.time() - worker_start
        print(f"Worker {var_name} chunk {chunk_idx}: extraction={extraction_time:.2f}s, processing={processing_time:.2f}s, total={total_worker_time:.2f}s")
        
        return (chunk_data, mapping, chunk_idx, var_name)
    
    # Force garbage collection even if no data
    gc.collect()
    total_worker_time = time.time() - worker_start
    print(f"Worker {var_name} chunk {chunk_idx}: no data, total={total_worker_time:.2f}s")
    return None

def add_wind_speed_calculations(output_dir):
    """Add wind speed calculations from U and V components."""
    import glob
    import re

    # Process each variable directory
    for var_name in ['UWind80', 'VWind80', 'UWind10', 'VWind10']:
        var_dir = os.path.join(output_dir, var_name)
        if not os.path.exists(var_dir):
            continue
            
        # Get all date directories for this variable
        date_dirs = [d for d in os.listdir(var_dir) if os.path.isdir(os.path.join(var_dir, d)) and d.isdigit()]
        
        for date_str in date_dirs:
            date_dir = os.path.join(var_dir, date_str)
            
            # Get all chunk files for this variable and date
            chunk_files = glob.glob(os.path.join(date_dir, f"{var_name}_chunk_*.parquet"))
            
            for chunk_file in chunk_files:
                # Extract chunk number
                chunk_match = re.search(r'chunk_(\d+)\.parquet', chunk_file)
                if not chunk_match:
                    continue
                chunk_num = chunk_match.group(1)
                
                # Read the chunk data
                df = pd.read_parquet(chunk_file)
                
                # Determine the corresponding U and V files
                u_var = None
                v_var = None
                
                if var_name == 'UWind80':
                    v_var = 'VWind80'
                    u_var = 'UWind80'
                elif var_name == 'VWind80':
                    u_var = 'UWind80'
                    v_var = 'VWind80'
                elif var_name == 'UWind10':
                    v_var = 'VWind10'
                    u_var = 'UWind10'
                elif var_name == 'VWind10':
                    u_var = 'UWind10'
                    v_var = 'VWind10'
                else:
                    continue
                    
                # Try to read the corresponding component from the same date directory
                u_file = os.path.join(output_dir, u_var, date_str, f"{u_var}_chunk_{chunk_num}.parquet")
                v_file = os.path.join(output_dir, v_var, date_str, f"{v_var}_chunk_{chunk_num}.parquet")
                
                if os.path.exists(u_file) and os.path.exists(v_file):
                    u_df = pd.read_parquet(u_file)
                    v_df = pd.read_parquet(v_file)

                    # Vectorized wind speed over common columns to avoid fragmented DataFrames
                    common_cols = [c for c in u_df.columns if c in v_df.columns]
                    if common_cols:
                        wind_vals = np.sqrt(u_df[common_cols].to_numpy(dtype=float) ** 2 + v_df[common_cols].to_numpy(dtype=float) ** 2)
                        wind_speed_df = pd.DataFrame(wind_vals, index=u_df.index, columns=common_cols)
                        # Round to exactly 3 decimals
                        wind_speed_df = (wind_speed_df * 1000).round().astype('int32') / 1000.0
                    else:
                        continue
                    
                    # Save wind speed data in the same date directory structure
                    wind_speed_var = 'WindSpeed80' if '80' in var_name else 'WindSpeed10'
                    wind_speed_dir = os.path.join(output_dir, wind_speed_var, date_str)
                    os.makedirs(wind_speed_dir, exist_ok=True)
                    wind_speed_file = os.path.join(wind_speed_dir, f"{wind_speed_var}_chunk_{chunk_num}.parquet")
                    wind_speed_df.to_parquet(wind_speed_file, compression='snappy', engine='pyarrow')
                    
                    print(f"Calculated wind speed for chunk {chunk_num} at {wind_speed_var} for date {date_str}")

def io_worker(output_queue, output_dir, compression):
    """I/O worker for producer-consumer pattern."""
    import queue as pyqueue
    while True:
        try:
            item = output_queue.get(timeout=5)
        except pyqueue.Empty:
            continue
        if item is None:
            break
        chunk_data, mapping, chunk_idx, var_name = item
        # Write mapping
        mapping_filename = f"{var_name}_chunk_{chunk_idx:04d}_mapping.json"
        mapping_path = os.path.join(output_dir, "mappings", mapping_filename)
        with open(mapping_path, 'w') as f:
            json.dump(mapping, f, indent=2)
        # Write parquet
        var_output_dir = os.path.join(output_dir, var_name)
        os.makedirs(var_output_dir, exist_ok=True)
        chunk_filename = f"{var_name}_chunk_{chunk_idx:04d}.parquet"
        chunk_path = os.path.join(var_output_dir, chunk_filename)
        chunk_data.to_parquet(chunk_path, compression=compression, engine='pyarrow')

def check_grid_data_availability(START, END, grid_data_dir="./extracted_grid_data"):
    """Check if full grid data is available for the specified time period.
    
    Args:
        START (datetime): Start datetime
        END (datetime): End datetime
        grid_data_dir (str): Directory containing full grid parquet files
        
    Returns:
        bool: True if data is available, False otherwise
    """
    import json
    import os
    
    metadata_file = os.path.join(grid_data_dir, "extraction_metadata.json")
    if not os.path.exists(metadata_file):
        return False
    
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        grid_start = pd.Timestamp(metadata['start_date'])
        grid_end = pd.Timestamp(metadata['end_date'])
        
        # Check if requested period is within available data
        return (START >= grid_start) and (END <= grid_end)
        
    except Exception:
        return False


def get_memory_usage():
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024  # Convert to MB
    except ImportError:
        return 0

def get_available_memory():
    """Get available system memory in GB."""
    try:
        import psutil
        memory = psutil.virtual_memory()
        return memory.available / 1024 / 1024 / 1024  # Convert to GB
    except ImportError:
        return 256  # Default assumption of 256GB if psutil not available

def calculate_safe_worker_count(available_memory_gb, chunk_size, num_variables):
    """Calculate safe number of workers based on available memory.
    
    For a 256GB system, we want to use only a small fraction to be safe.
    """
    # Estimate memory per worker: chunk_size * num_variables * 8 bytes (float64) * safety_factor
    # Based on actual observed usage, each worker uses much more memory than calculated
    memory_per_worker_mb = (chunk_size * num_variables * 8 * 10) / (1024 * 1024)  # 10x safety factor based on actual usage
    
    # For high-performance systems (256GB+), use more memory
    # For smaller systems, be more conservative
    if available_memory_gb >= 200:
        safe_memory_mb = available_memory_gb * 1024 * 0.70  # 70% of available memory for high-performance systems
    else:
        safe_memory_mb = available_memory_gb * 1024 * 0.20  # 20% of available memory for smaller systems
    
    max_workers = int(safe_memory_mb / memory_per_worker_mb)
    
    # Ensure reasonable bounds - be conservative based on actual usage
    # For high-performance systems (256GB+), allow more workers
    if available_memory_gb >= 200:
        max_workers = max(1, min(max_workers, 36))  # Allow up to 36 workers for high-performance systems
    else:
        max_workers = max(1, min(max_workers, 4))  # Cap at 4 workers for smaller systems
    
    memory_percentage = 70 if available_memory_gb >= 200 else 20
    print(f"Memory calculation:")
    print(f"  Available memory: {available_memory_gb:.1f} GB")
    print(f"  Safe memory limit: {safe_memory_mb:.1f} MB ({memory_percentage}% of available)")
    print(f"  Memory per worker: {memory_per_worker_mb:.1f} MB")
    print(f"  Calculated max workers: {max_workers}")
    
    return max_workers

def configure_parallel_workers(total_cpus=None, file_worker_ratio=0.3, chunk_worker_ratio=0.7):
    """Configure parallel workers to efficiently utilize available cores."""
    if total_cpus is None:
        total_cpus = os.cpu_count()
    
    # Optimized worker allocation for high-core systems
    file_workers = max(1, min(16, int(total_cpus * file_worker_ratio)))  # Cap at 16 for file I/O
    chunk_workers = max(1, min(32, int(total_cpus * chunk_worker_ratio)))  # Cap at 32 for chunk processing
    
    print(f"Optimized parallel configuration for {total_cpus} CPUs:")
    print(f"  File processing workers: {file_workers}")
    print(f"  Chunk processing workers: {chunk_workers}")
    print(f"  Total workers: {file_workers + chunk_workers}")
    
    return {
        'file_workers': file_workers,
        'chunk_workers': chunk_workers,
        'total_cpus': total_cpus
    }

# Use more conservative parallel configuration
PARALLEL_CONFIG = configure_parallel_workers()

def calculate_optimal_chunk_size_for_target_file_size(target_size_gb=1.0, grid_points=3750000):
    """Calculate optimal chunk size to achieve target file size.
    
    Args:
        target_size_gb (float): Target file size in GB
        grid_points (int): Total number of grid points
        
    Returns:
        int: Optimal chunk size in grid points
    """
    # Calculate for 1 day of data (24 hours)
    daily_data_points = grid_points * 24
    target_size_bytes = target_size_gb * 1024**3
    chunk_size = int(target_size_bytes / (daily_data_points * 4))  # float32
    
    # Ensure reasonable bounds
    chunk_size = max(1000, min(chunk_size, grid_points // 10))
    
    return chunk_size

# Pre-calculated optimal chunk sizes
OPTIMAL_CHUNK_SIZES = {
    "1GB_files": calculate_optimal_chunk_size_for_target_file_size(1.0),
    "2GB_files": calculate_optimal_chunk_size_for_target_file_size(2.0),
    "5GB_files": calculate_optimal_chunk_size_for_target_file_size(5.0),
}

def monitor_memory_usage(warning_threshold_gb=20):
    """Monitor memory usage and warn if it gets too high."""
    try:
        import psutil
        memory = psutil.virtual_memory()
        used_gb = memory.used / 1024 / 1024 / 1024
        total_gb = memory.total / 1024 / 1024 / 1024
        percent_used = memory.percent
        
        # For high-performance systems, use higher thresholds
        if total_gb >= 200:
            warning_threshold_gb = 100  # 100GB warning for 256GB systems
            critical_threshold = 85     # 85% critical for high-performance systems
        else:
            critical_threshold = 85     # 85% critical for smaller systems
        
        # Only warn if memory usage is truly critical (>95%) to reduce noise
        if percent_used > 95:
            print(f"🚨 CRITICAL: System memory usage is very high ({percent_used:.1f}%)")
            print(f"   Recommended: Use only 1-2 workers and smaller chunk size")
        elif percent_used > 90 and not hasattr(monitor_memory_usage, '_warned_90'):
            print(f"⚠️  WARNING: High memory usage: {used_gb:.1f} GB ({percent_used:.1f}%)")
            print(f"   Total system memory: {total_gb:.1f} GB")
            monitor_memory_usage._warned_90 = True
        
        return used_gb, percent_used
    except ImportError:
        return 0, 0

def calculate_optimal_settings_for_256gb_system():
    """Calculate optimal settings for a 256GB system to prevent memory issues."""
    print("Calculating optimal settings for 256GB system...")
    
    # Check current memory usage
    try:
        import psutil
        memory = psutil.virtual_memory()
        percent_used = memory.percent
        
        if percent_used > 85:
            print(f"⚠️  System memory usage is very high ({percent_used:.1f}%)")
            print(f"   Using ultra-conservative settings")
            # Ultra-conservative settings for high memory usage
            safe_memory_gb = 256 * 0.05  # Only 5% of total memory
            optimal_workers = 1
            chunk_size = 50000
        elif percent_used > 70:
            print(f"⚠️  System memory usage is high ({percent_used:.1f}%)")
            print(f"   Using conservative settings")
            # Conservative settings for moderate memory usage
            safe_memory_gb = 256 * 0.1  # 10% of total memory
            optimal_workers = 2
            chunk_size = 100000
        else:
            print(f"✅ System memory usage is acceptable ({percent_used:.1f}%)")
            # Conservative settings based on actual observed usage
            safe_memory_gb = 256 * 0.2  # 20% of total memory (based on actual usage)
            optimal_workers = 4
            chunk_size = 150000
    except ImportError:
        # Default conservative settings if psutil not available
        safe_memory_gb = 256 * 0.2
        optimal_workers = 4
        chunk_size = 150000
    
    # Estimate memory per task based on actual observed usage
    memory_per_task_mb = (chunk_size * 3 * 4 * 10) / (1024 * 1024)  # 10x safety factor based on actual usage
    
    # Calculate safe number of concurrent tasks
    safe_concurrent_tasks = int((safe_memory_gb * 1024) / memory_per_task_mb)
    
    print(f"Optimal settings for 256GB system:")
    print(f"  Safe memory limit: {safe_memory_gb:.1f} GB")
    print(f"  Memory per task: {memory_per_task_mb:.1f} MB")
    print(f"  Safe concurrent tasks: {safe_concurrent_tasks}")
    print(f"  Recommended workers: {optimal_workers}")
    print(f"  Recommended chunk size: {chunk_size:,}")
    print(f"  Recommended batch size: {optimal_workers * 2}")
    
    return {
        'workers': optimal_workers,
        'chunk_size': chunk_size,
        'batch_size': optimal_workers * 2,
        'max_file_groups': 5000
    }

# Use optimal settings for 256GB system
OPTIMAL_256GB_SETTINGS = calculate_optimal_settings_for_256gb_system()

def create_chunk_mapping(chunk_indices, grid_lats, grid_lons, output_dir, var_name, chunk_idx, create_individual_mappings=False, format="parquet"):
    """Create mapping for a specific chunk - optionally creates individual mapping files.
    
    Args:
        chunk_indices: Indices for this chunk
        grid_lats: 2D array of latitudes
        grid_lons: 2D array of longitudes
        output_dir: Output directory
        var_name: Variable name
        chunk_idx: Chunk index
        create_individual_mappings: Whether to create individual mapping files
        format: "parquet" (fast) or "json" (human readable)
    """
    n_lats, n_lons = grid_lats.shape
    
    # Always create a dict for compatibility with existing code
    mapping = {}
    for i, idx in enumerate(chunk_indices):
        lat_idx, lon_idx = np.unravel_index(idx, (n_lats, n_lons))
        lat = float(grid_lats[lat_idx, lon_idx])
        lon = float(grid_lons[lat_idx, lon_idx])
        mapping[f"grid_{i:06d}"] = {
            'lat': lat,
            'lon': lon,
            'grid_index': int(idx),
            'lat_index': int(lat_idx),
            'lon_index': int(lon_idx)
        }
    
    # Only create individual mapping files if requested
    if create_individual_mappings:
        if format.lower() == "parquet":
            # Create DataFrame for fast Parquet storage
            import pandas as pd
            mapping_data = []
            for i, idx in enumerate(chunk_indices):
                lat_idx, lon_idx = np.unravel_index(idx, (n_lats, n_lons))
                lat = float(grid_lats[lat_idx, lon_idx])
                lon = float(grid_lons[lat_idx, lon_idx])
                
                mapping_data.append({
                    'grid_id': f"grid_{i:06d}",
                    'lat': lat,
                    'lon': lon,
                    'grid_index': int(idx),
                    'lat_index': int(lat_idx),
                    'lon_index': int(lon_idx)
                })
            
            mapping_df = pd.DataFrame(mapping_data)
            mapping_filename = f"{var_name}_chunk_{chunk_idx:04d}_mapping.parquet"
            mapping_path = os.path.join(output_dir, "mappings", mapping_filename)
            
            # Ensure the mappings directory exists
            os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
            
            # Time the write operation
            write_start = time.time()
            mapping_df.to_parquet(mapping_path, compression="snappy", engine="pyarrow")
            write_time = time.time() - write_start
            
            file_size_mb = os.path.getsize(mapping_path) / (1024 * 1024)
            print(f"Chunk mapping saved: {mapping_filename} ({file_size_mb:.1f} MB) in {write_time:.2f}s")
        else:
            mapping_filename = f"{var_name}_chunk_{chunk_idx:04d}_mapping.json"
            mapping_path = os.path.join(output_dir, "mappings", mapping_filename)
            
            # Ensure the mappings directory exists
            os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
            
            with open(mapping_path, 'w') as f:
                json.dump(mapping, f, indent=2)
    
    return mapping

def file_writing_worker(args):
    """Worker function for parallel file writing."""
    chunk_data, mapping, chunk_idx, var_name, output_dir, compression, create_individual_mappings = args
    
    write_start = time.time()
    write_stats = {
        'files_written': 0,
        'total_size_mb': 0,
        'write_time': 0,
        'write_times': []  # Add missing write_times list
    }
    
    try:
        # Group data by date
        chunk_data['date'] = chunk_data.index.date
        date_groups = chunk_data.groupby('date')
        
        for date, date_data in date_groups:
            # Remove the date column and keep only the data
            date_data = date_data.drop('date', axis=1)
            date_str = date.strftime('%Y%m%d')
            
            # Create date-specific directories
            var_output_dir = os.path.join(output_dir, var_name, date_str)
            os.makedirs(var_output_dir, exist_ok=True)
            
            # Write parquet file (single-file-per-day per variable if enabled)
            single_file_mode = os.getenv('HRRR_SINGLE_FILE_PER_DAY', '1') in ('1', 'true', 'True')
            if single_file_mode:
                # Append-as-row-group approach: write/append one file per day/variable
                day_file = os.path.join(var_output_dir, f"{date_str}.parquet")
                # Use a simple append by reading existing (if any) and concatenating; safe for small day batches
                if os.path.exists(day_file):
                    try:
                        existing = pd.read_parquet(day_file)
                        date_data = pd.concat([existing, date_data])
                    except Exception:
                        pass
                chunk_path = day_file
            else:
                chunk_filename = f"{var_name}_chunk_{chunk_idx:04d}.parquet"
                chunk_path = os.path.join(var_output_dir, chunk_filename)
            
            # Time the actual file write
            file_write_start = time.time()
            
            # 🔍 NaN DETECTION: Check for missing data
            nan_check_start = time.time()
            nan_counts = date_data.isna().sum()
            total_nans = nan_counts.sum()
            
            if total_nans > 0:
                print(f"  🔴 WARNING: Found {total_nans} NaN values in {chunk_filename}")
                print(f"     Variable: {var_name}")
                print(f"     Date: {date_str}")
                print(f"     NaN breakdown by column:")
                for col, nan_count in nan_counts.items():
                    if nan_count > 0:
                        print(f"       {col}: {nan_count} NaN values")
                
                # Log to file for detailed analysis
                nan_log_file = os.path.join(output_dir, "nan_detection.log")
                with open(nan_log_file, 'a') as f:
                    f.write(f"{datetime.datetime.now().isoformat()} | {var_name} | {date_str} | {chunk_filename} | Total NaNs: {total_nans}\n")
                    for col, nan_count in nan_counts.items():
                        if nan_count > 0:
                            f.write(f"  {col}: {nan_count} NaN values\n")
            
            # Ensure data types and apply optional scaling per variable
            wind_vars = {"UWind80", "VWind80", "UWind10", "VWind10", "WindSpeed80", "WindSpeed10"}
            rad_vars = {"rad", "vbd", "vdd"}
            temp_vars = {"2tmp"}

            scale_wind = os.getenv('HRRR_SCALE_WIND_INT16', '1') in ('1', 'true', 'True')
            wind_scale = int(os.getenv('HRRR_WIND_SCALE', '100'))  # 0.01 m/s by default

            if var_name in wind_vars and scale_wind:
                # Scale to int16 with configured precision
                date_data = (date_data * wind_scale).round().astype('int16')
            elif var_name in rad_vars:
                # Radiation: 0.1 precision → int16
                date_data = (date_data * 10).round().astype('int16')
            elif var_name in temp_vars:
                # Temperature: (T - 250) * 100 as int16 (0.01 K)
                date_data = ((date_data - 250.0) * 100).round().astype('int16')
            else:
                # Default float32
                date_data = date_data.astype('float32')
            
            # Try pyarrow first, fallback to fastparquet if it fails
            try:
                date_data.to_parquet(chunk_path, compression=compression, engine='pyarrow')
                engine_used = 'pyarrow'
            except Exception as e:
                print(f"  ⚠️  pyarrow failed for {chunk_filename}, trying fastparquet: {e}")
                try:
                    date_data.to_parquet(chunk_path, compression=compression, engine='fastparquet')
                    engine_used = 'fastparquet'
                    print(f"  ✅ fastparquet succeeded for {chunk_filename}")
                except Exception as e2:
                    print(f"  ❌ Both pyarrow and fastparquet failed for {chunk_filename}: {e2}")
                    raise e2
            
            file_write_time = time.time() - file_write_start
            
            file_size_mb = os.path.getsize(chunk_path) / (1024 * 1024)
            write_stats['files_written'] += 1
            write_stats['total_size_mb'] += file_size_mb
            write_stats['write_times'].append(file_write_time)
            
            if single_file_mode:
                print(f"  Wrote {var_name}/{date_str}.parquet ({file_size_mb:.1f} MB) in {file_write_time:.2f}s")
            else:
                print(f"  Wrote {chunk_filename} ({file_size_mb:.1f} MB) in {file_write_time:.2f}s")
            
            # Optionally write individual mapping files if requested
            if create_individual_mappings:
                mapping_dir = os.path.join(output_dir, "mappings", var_name, date_str)
                os.makedirs(mapping_dir, exist_ok=True)
                mapping_filename = f"{var_name}_chunk_{chunk_idx:04d}_mapping.json"
                mapping_path = os.path.join(mapping_dir, mapping_filename)
                with open(mapping_path, 'w') as f:
                    json.dump(mapping, f, indent=2)
        
        write_stats['write_time'] = time.time() - write_start
        return write_stats
        
    except Exception as e:
        print(f"Error in file writing worker for {var_name} chunk {chunk_idx}: {e}")
        return None

def check_existing_outputs(output_dir, var_name, date_range):
    """Check which dates have already been processed for a variable."""
    existing_dates = set()
    var_output_dir = os.path.join(output_dir, var_name)
    
    if os.path.exists(var_output_dir):
        # Check for date files (YYYYMMDD.parquet)
        for item in os.listdir(var_output_dir):
            if item.endswith('.parquet'):
                # Extract date from filename (YYYYMMDD.parquet)
                try:
                    date_str = item.replace('.parquet', '')
                    date = datetime.datetime.strptime(date_str, '%Y%m%d').date()
                    if date in date_range:
                        existing_dates.add(date)
                except ValueError:
                    continue
    
    return existing_dates

def get_processed_date_range(output_dir, SELECTORS):
    """Get the range of dates that have been processed for all variables."""
    if not os.path.exists(output_dir):
        return set()
    
    # Get all processed dates across all variables
    all_processed_dates = set()
    
    for var_name in SELECTORS.keys():
        var_output_dir = os.path.join(output_dir, var_name)
        if os.path.exists(var_output_dir):
            for item in os.listdir(var_output_dir):
                if item.endswith('.parquet'):
                    # Extract date from filename (YYYYMMDD.parquet)
                    try:
                        date_str = item.replace('.parquet', '')
                        date = datetime.datetime.strptime(date_str, '%Y%m%d').date()
                        all_processed_dates.add(date)
                    except ValueError:
                        continue
    
    return all_processed_dates

def create_resume_metadata(output_dir, START, END, SELECTORS, processed_dates):
    """Create metadata file for resume functionality."""
    metadata = {
        'start_date': START.isoformat(),
        'end_date': END.isoformat(),
        'selectors': SELECTORS,
        'processed_dates': [d.isoformat() for d in sorted(processed_dates)],
        'last_updated': datetime.datetime.now().isoformat(),
        'status': 'in_progress'
    }
    
    metadata_file = os.path.join(output_dir, 'resume_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata_file

def load_resume_metadata(output_dir):
    """Load existing resume metadata if available."""
    metadata_file = os.path.join(output_dir, 'resume_metadata.json')
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            return metadata
        except Exception as e:
            print(f"Warning: Could not load resume metadata: {e}")
    return None

def extract_full_grid_day_by_day(
    START,
    END,
    DATADIR,
    DEFAULT_HOURS_FORECASTED,
    SELECTORS,
    output_dir="./extracted_grid_data",
    chunk_size=None,  # Auto-detect based on system
    compression="snappy",
    use_parallel=True,
    num_cpu_workers=None,  # Auto-detect based on system
    num_io_workers=None,   # Auto-detect based on system
    max_file_groups=None,  # Auto-detect based on system
    create_individual_mappings=False,
    parallel_file_writing=True,
    enable_resume=True,
    day_output_dir_format="flat",  # "daily" or "flat"
    use_aggressive_settings=True,  # Use aggressive settings for high-performance systems
):
    """
    Extract full grid data day by day to prevent memory issues.
    
    This function processes one day at a time, which:
    - Prevents memory overflow
    - Allows for easy interruption and resume
    - Provides better progress tracking
    - Reduces risk of data loss
    - Enables aggressive parallelization (with day-by-day safety)
    
    Args:
        START (datetime): Start datetime
        END (datetime): End datetime
        DATADIR (str): Directory containing GRIB files
        DEFAULT_HOURS_FORECASTED (list): List of forecast hours to process
        SELECTORS (dict): Dictionary of variables to extract
        output_dir (str): Base output directory
        chunk_size (int): Number of grid points per chunk (auto-detect if None)
        compression (str): Parquet compression
        use_parallel (bool): Whether to use parallel processing
        num_cpu_workers (int): Number of CPU workers (auto-detect if None)
        num_io_workers (int): Number of I/O workers (auto-detect if None)
        max_file_groups (int): Maximum file groups to process (auto-detect if None)
        create_individual_mappings (bool): Whether to create individual mapping files
        parallel_file_writing (bool): Whether to use parallel file writing
        enable_resume (bool): Whether to enable resume functionality
        day_output_dir_format (str): Output directory format ("daily" or "flat")
        use_aggressive_settings (bool): Use aggressive settings for high-performance systems
        
    Returns:
        dict: Summary of processing results
    """
    import datetime
    
    print("🚀 Starting Day-by-Day Full Grid Extraction")
    print("=" * 60)
    print(f"Date range: {START.date()} to {END.date()}")
    print(f"Total days: {(END.date() - START.date()).days + 1}")
    print(f"Output directory: {output_dir}")
    print(f"Day output format: {day_output_dir_format}")
    print(f"Resume enabled: {enable_resume}")
    print(f"Aggressive settings: {use_aggressive_settings}")
    print()
    
    # Auto-detect optimal settings based on system capabilities
    if use_aggressive_settings:
        if use_aggressive_settings:
            settings = get_aggressive_parallel_settings()
        else:
            settings = get_optimized_settings_for_high_performance_system()
        
        # Override with user-provided values if specified
        if chunk_size is None:
            chunk_size = settings['chunk_size']
        if num_cpu_workers is None:
            num_cpu_workers = settings['num_cpu_workers']
        if num_io_workers is None:
            num_io_workers = settings['num_io_workers']
        if max_file_groups is None:
            max_file_groups = settings['max_file_groups']
        
        print(f"🎯 Using auto-detected settings:")
        print(f"   Chunk size: {chunk_size:,}")
        print(f"   CPU workers: {num_cpu_workers}")
        print(f"   I/O workers: {num_io_workers}")
        print(f"   Max file groups: {max_file_groups:,}")
        print()
    else:
        # Use conservative defaults if aggressive settings disabled
        if chunk_size is None:
            chunk_size = 150000
        if num_cpu_workers is None:
            num_cpu_workers = 8
        if num_io_workers is None:
            num_io_workers = 4
        if max_file_groups is None:
            max_file_groups = 5000
    
    # Create base output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 🚀 OPTIMIZATION: Extract grid metadata ONCE at the beginning
    print("📊 Extracting grid metadata (once for all days)...")
    grid_start = time.time()
    wind_data_lat_long = get_wind_data_lat_long(START, DATADIR)
    grid_lats, grid_lons = wind_data_lat_long[0], wind_data_lat_long[1]
    
    n_lats, n_lons = grid_lats.shape
    total_grid_points = n_lats * n_lons
    
    print(f"Grid dimensions: {n_lats} x {n_lons} = {total_grid_points:,} total points")
    grid_time = time.time() - grid_start
    print(f"Grid metadata extracted in {grid_time:.2f}s")
    
    # 🚀 OPTIMIZATION: Create global mapping ONCE at the beginning
    print("📊 Creating global grid mapping (once for all days)...")
    mapping_start = time.time()
    global_mapping = create_global_grid_mapping(grid_lats, grid_lons, output_dir, format="parquet")
    mapping_time = time.time() - mapping_start
    print(f"Global mapping created in {mapping_time:.2f}s")
    
    # Initialize tracking
    total_start_time = time.time()
    successful_days = []
    failed_days = []
    skipped_days = []
    
    # Generate list of days to process
    date_range = pd.date_range(start=START.date(), end=END.date(), freq="1D")
    
    # Check for already processed days if resume is enabled
    if enable_resume:
        print("Checking for already processed days...")
        processed_dates = get_processed_date_range(output_dir, SELECTORS)
        remaining_dates = [d for d in date_range if d.date() not in processed_dates]
        
        if processed_dates:
            print(f"Found {len(processed_dates)} already processed days:")
            for date in sorted(processed_dates):
                print(f"  - {date}")
            print(f"Remaining days to process: {len(remaining_dates)}")
        else:
            print("No previously processed days found.")
            remaining_dates = date_range
    else:
        remaining_dates = date_range
    
    if len(remaining_dates) == 0:
        print("✅ All days already processed! Extraction complete.")
        return {
            "status": "completed",
            "total_days": len(date_range),
            "successful_days": len(successful_days),
            "failed_days": len(failed_days),
            "skipped_days": len(skipped_days),
            "processing_time_seconds": 0,
            "resume_used": True
        }
    
    print(f"Processing {len(remaining_dates)} days...")
    print()
    
    # Process each day
    for day_idx, current_date in enumerate(remaining_dates, 1):
        day_start_time = time.time()
        
        # Check for shutdown request
        if check_shutdown_requested():
            print("🛑 Shutdown requested. Saving progress...")
            break
        
        print(f"📅 Processing day {day_idx}/{len(remaining_dates)}: {current_date.date()}")
        print(f"   Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Variables: {list(SELECTORS.keys())}")
        
        # Set day-specific start and end times
        day_start = datetime.datetime.combine(current_date.date(), START.time())
        day_end = datetime.datetime.combine(current_date.date(), START.time()) + datetime.timedelta(days=1)
        
        # Determine output directory for this day
        if day_output_dir_format == "daily":
            # Each day gets its own subdirectory
            day_output_dir = os.path.join(output_dir, current_date.strftime("%Y%m%d"))
        else:
            # Flat structure - all days in same directory
            day_output_dir = output_dir
        
        print(f"   Output: {day_output_dir}")
        
        try:
            # 🚀 OPTIMIZATION: Pass pre-extracted grid data to avoid re-extraction
            day_result = extract_full_grid_optimized_with_preloaded_data(
                START=day_start,
                END=day_end,
                DATADIR=DATADIR,
                DEFAULT_HOURS_FORECASTED=DEFAULT_HOURS_FORECASTED,
                SELECTORS=SELECTORS,
                output_dir=day_output_dir,
                chunk_size=chunk_size,
                compression=compression,
                use_parallel=use_parallel,
                num_cpu_workers=num_cpu_workers,
                num_io_workers=num_io_workers,
                max_file_groups=max_file_groups,
                create_individual_mappings=create_individual_mappings,
                parallel_file_writing=parallel_file_writing,
                enable_resume=False,  # Don't use resume within day processing
                # 🚀 OPTIMIZATION: Pass pre-loaded data
                grid_lats=grid_lats,
                grid_lons=grid_lons,
                global_mapping=global_mapping,
            )
            
            day_time = time.time() - day_start_time
            
            if day_result and day_result.get("status") == "completed":
                successful_days.append(current_date.date())
                print(f"   ✅ Completed in {day_time:.1f}s ({day_time/60:.1f} minutes)")
                print(f"   📁 Files written to: {day_output_dir}")
            else:
                failed_days.append(current_date.date())
                print(f"   ❌ Failed after {day_time:.1f}s ({day_time/60:.1f} minutes)")
                
        except Exception as e:
            day_time = time.time() - day_start_time
            failed_days.append(current_date.date())
            print(f"   ❌ Error after {day_time:.1f}s: {e}")
            
            # Continue with next day instead of stopping
            print(f"   Continuing with next day...")
            continue
        
        # Progress update
        total_time_so_far = time.time() - total_start_time
        avg_time_per_day = total_time_so_far / day_idx
        remaining_days = len(remaining_dates) - day_idx
        estimated_remaining_time = remaining_days * avg_time_per_day
        
        print(f"   📊 Progress: {day_idx}/{len(remaining_dates)} days")
        print(f"   ✅ Success rate: {len(successful_days)}/{day_idx} ({len(successful_days)/day_idx*100:.1f}%)")
        print(f"   ⏱️  Estimated time remaining: {estimated_remaining_time/3600:.1f} hours")
        print(f"   🗓️  Next day: {remaining_dates[day_idx].date() if day_idx < len(remaining_dates) else 'COMPLETE'}")
        print()
        
        # Force garbage collection between days
        import gc
        gc.collect()
        
        # Small delay to allow system to stabilize
        time.sleep(1)
    
    # Final summary
    total_time = time.time() - total_start_time
    
    # 🔍 NaN SUMMARY: Analyze NaN detection logs
    nan_summary_file = os.path.join(output_dir, "nan_summary.txt")
    try:
        if os.path.exists(os.path.join(output_dir, "nan_detection.log")):
            print("\n🔍 NAN DETECTION SUMMARY:")
            print("=" * 40)
            
            # Read and analyze nan detection log
            nan_log_file = os.path.join(output_dir, "nan_detection.log")
            if os.path.exists(nan_log_file):
                with open(nan_log_file, 'r') as f:
                    nan_lines = f.readlines()
                
                if nan_lines:
                    print(f"📋 Found {len(nan_lines)} NaN detection entries")
                    print("📊 NaN patterns by variable:")
                    
                    # Group by variable
                    var_nan_counts = {}
                    for line in nan_lines:
                        if '|' in line:
                            parts = line.strip().split(' | ')
                            if len(parts) >= 4:
                                var_name = parts[1]
                                date_str = parts[2]
                                nan_count = parts[4].split(': ')[1] if ': ' in parts[4] else 'unknown'
                                
                                if var_name not in var_nan_counts:
                                    var_nan_counts[var_name] = []
                                var_nan_counts[var_name].append((date_str, nan_count))
                    
                    for var_name, entries in var_nan_counts.items():
                        print(f"  {var_name}: {len(entries)} files with NaN values")
                        for date_str, nan_count in entries[:5]:  # Show first 5
                            print(f"    {date_str}: {nan_count} NaNs")
                        if len(entries) > 5:
                            print(f"    ... and {len(entries) - 5} more files")
                    
                    # Write summary to file
                    with open(nan_summary_file, 'w') as f:
                        f.write(f"NaN Detection Summary - {datetime.datetime.now().isoformat()}\n")
                        f.write("=" * 50 + "\n")
                        f.write(f"Total NaN entries: {len(nan_lines)}\n")
                        f.write(f"Variables affected: {list(var_nan_counts.keys())}\n\n")
                        
                        for var_name, entries in var_nan_counts.items():
                            f.write(f"{var_name}:\n")
                            for date_str, nan_count in entries:
                                f.write(f"  {date_str}: {nan_count}\n")
                            f.write("\n")
                else:
                    print("✅ No NaN values detected in any files!")
            else:
                print("✅ No NaN detection log found - no NaN issues detected!")
        else:
            print("✅ No NaN detection log found - no NaN issues detected!")
    except Exception as e:
        print(f"⚠️  Error analyzing NaN summary: {e}")
    
    print("=" * 60)
    print("📊 DAY-BY-DAY EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Total processing time: {total_time:.1f}s ({total_time/3600:.1f} hours)")
    print(f"Total days requested: {len(date_range)}")
    print(f"Successful days: {len(successful_days)}")
    print(f"Failed days: {len(failed_days)}")
    print(f"Skipped days: {len(skipped_days)}")
    print(f"Success rate: {len(successful_days)/len(date_range)*100:.1f}%")
    
    if successful_days:
        print(f"First successful day: {min(successful_days)}")
        print(f"Last successful day: {max(successful_days)}")
    
    if failed_days:
        print(f"Failed days: {sorted(failed_days)}")
    
    # Create final resume metadata
    if enable_resume:
        final_processed_dates = get_processed_date_range(output_dir, SELECTORS)
        create_resume_metadata(output_dir, START, END, SELECTORS, final_processed_dates)
        print(f"Resume metadata updated. Total processed dates: {len(final_processed_dates)}")
    
    return {
        "status": "completed" if len(failed_days) == 0 else "partial",
        "total_days": len(date_range),
        "successful_days": len(successful_days),
        "failed_days": len(failed_days),
        "skipped_days": len(skipped_days),
        "processing_time_seconds": total_time,
        "resume_used": enable_resume and len(processed_dates) > 0 if 'processed_dates' in locals() else False,
        "successful_dates": [d.isoformat() for d in successful_days],
        "failed_dates": [d.isoformat() for d in failed_days],
        "skipped_dates": [d.isoformat() for d in skipped_days],
        "grid_metadata_time": grid_time,
        "mapping_creation_time": mapping_time,
    }

def get_optimized_settings_for_high_performance_system():
    """
    Get optimized settings for high-performance systems (36+ CPUs, 256GB+ RAM).
    
    With day-by-day processing, we can be more aggressive since memory is cleaned up
    between days, allowing us to fully utilize the available resources.
    """
    import psutil

    # Get system specs
    cpu_count = os.cpu_count()
    total_memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # Force detection of more CPUs if available (Windows sometimes under-reports)
    # Check if we're on a high-performance system and force higher CPU count
    if cpu_count >= 16 and total_memory_gb >= 200:
        # Likely a high-performance system, use more aggressive CPU count
        cpu_count = min(36, cpu_count * 2)  # Assume we can use more CPUs
        print(f"🚀 High-performance system detected, using {cpu_count} CPUs")
    
    # Check if we're on a high-performance system (override detection if needed)
    # For Linux servers, we might need to check environment variables or other indicators
    is_high_performance = False
    
    # Manual override for known high-performance systems
    # Set this to True if you know you have 36+ CPUs
    MANUAL_HIGH_PERFORMANCE_OVERRIDE = True
    
    if MANUAL_HIGH_PERFORMANCE_OVERRIDE:
        is_high_performance = True
        cpu_count = 36  # Force 36 CPUs for your system
        print(f"🚀 Manual override: Using {cpu_count} CPUs for high-performance system")
    elif cpu_count >= 32 or total_memory_gb >= 200:
        is_high_performance = True
    elif cpu_count >= 16 and total_memory_gb >= 100:
        # Medium-performance system
        pass
    else:
        # Conservative settings for smaller systems
        pass
    
    print(f"🔧 Detected system: {cpu_count} CPUs, {total_memory_gb:.1f} GB RAM")
    
    if is_high_performance:
        # High-performance system (36+ CPUs, 256GB+ RAM)
        print("🚀 Using HIGH-PERFORMANCE settings")
        
        # Use ALL available CPUs, not just 36
        max_cpu_workers = min(cpu_count, 50)  # Cap at 50 for safety
        
        # Use ALL available CPUs for maximum performance
        settings = {
            'chunk_size': 200000,        # Reasonable chunks for 36 CPUs
            'num_cpu_workers': max_cpu_workers,  # Use all available CPUs (up to 50)
            'num_io_workers': max(4, cpu_count // 8),  # Scale I/O workers with CPU count
            'max_file_groups': 10000,    # More file groups for parallel processing
            'queue_maxsize': 32,         # Larger queue for more workers
            'batch_size_multiplier': 2,  # Process more batches in parallel
            'memory_safety_factor': 0.4, # Use 40% of available memory
            'compression': 'snappy',     # Fast compression
            'parallel_file_writing': True,
            'enable_resume': True,
            'day_output_dir_format': 'flat'
        }
        
    elif cpu_count >= 16 and total_memory_gb >= 100:
        # Medium-performance system (16+ CPUs, 100GB+ RAM)
        print("⚡ Using MEDIUM-PERFORMANCE settings")
        
        settings = {
            'chunk_size': 300000,        # Medium chunks
            'num_cpu_workers': 12,       # Use 2/3 of CPUs
            'num_io_workers': 6,         # Medium I/O workers
            'max_file_groups': 10000,    # Medium file groups
            'queue_maxsize': 24,         # Medium queue
            'batch_size_multiplier': 2,  # Process more batches
            'memory_safety_factor': 0.3, # Use 30% of available memory
            'compression': 'snappy',
            'parallel_file_writing': True,
            'enable_resume': True,
            'day_output_dir_format': 'flat'
        }
        
    else:
        # Conservative settings for smaller systems
        print("🛡️ Using CONSERVATIVE settings")
        
        settings = {
            'chunk_size': 150000,        # Conservative chunks
            'num_cpu_workers': 8,        # Conservative workers
            'num_io_workers': 4,         # Conservative I/O
            'max_file_groups': 5000,     # Conservative file groups
            'queue_maxsize': 16,         # Conservative queue
            'batch_size_multiplier': 1,  # Standard batch processing
            'memory_safety_factor': 0.2, # Use 20% of available memory
            'compression': 'snappy',
            'parallel_file_writing': True,
            'enable_resume': True,
            'day_output_dir_format': 'flat'
        }
    
    print(f"📊 Optimized settings:")
    print(f"   Chunk size: {settings['chunk_size']:,}")
    print(f"   CPU workers: {settings['num_cpu_workers']}")
    print(f"   I/O workers: {settings['num_io_workers']}")
    print(f"   Max file groups: {settings['max_file_groups']:,}")
    print(f"   Memory safety factor: {settings['memory_safety_factor']*100:.0f}%")
    
    return settings

def calculate_optimal_chunk_size_for_memory(memory_gb, num_workers, safety_factor=0.4):
    """Calculate optimal chunk size based on available memory and workers."""
    # Estimate memory per chunk: chunk_size * variables * 8 bytes * safety_factor
    # For 3 variables, 8 bytes per value, with safety factor
    memory_per_chunk_mb = (memory_gb * 1024 * safety_factor) / (num_workers * 3 * 8 / (1024 * 1024))
    
    # Ensure reasonable bounds
    min_chunk_size = 50000
    max_chunk_size = 1000000
    optimal_chunk_size = int(memory_per_chunk_mb)
    
    return max(min_chunk_size, min(optimal_chunk_size, max_chunk_size))

def get_aggressive_parallel_settings():
    """Get aggressive parallel settings for day-by-day processing."""
    settings = get_optimized_settings_for_high_performance_system()
    
    # For day-by-day processing, we can be more aggressive
    # since memory is cleaned up between days
    if settings['num_cpu_workers'] >= 20:
        # ULTRA-AGGRESSIVE settings for high-performance systems (36 CPUs, 256GB RAM)
        # Optimized for maximum throughput
        aggressive_settings = {
            'chunk_size': 200000,        # LARGER chunks for better efficiency (was 75000)
            'num_cpu_workers': 36,       # Use ALL CPUs (was 32)
            'num_io_workers': 20,        # More I/O workers (was 12)
            'max_file_groups': 50000,    # More file groups (was 20000)
            'queue_maxsize': 128,        # Larger queue (was 64)
            'batch_size_multiplier': 8,  # Process many batches (was 4)
            'memory_safety_factor': 0.7, # Use 70% of memory (was 50%)
            'compression': 'snappy',
            'parallel_file_writing': True,
            'enable_resume': True,
            'day_output_dir_format': 'flat',
            'aggressive_gc': True,       # Aggressive garbage collection
            'memory_monitoring': True,   # Continuous memory monitoring
            'use_fastparquet': True,     # Use fastparquet for better compatibility
            'optimize_chunk_processing': True,  # New: optimize chunk processing
        }
        
        print("🚀 Using ULTRA-AGGRESSIVE parallel settings for maximum performance")
        print(f"   Chunk size: {aggressive_settings['chunk_size']:,} (larger for efficiency)")
        print(f"   CPU workers: {aggressive_settings['num_cpu_workers']} (ALL CPUs)")
        print(f"   I/O workers: {aggressive_settings['num_io_workers']} (more I/O)")
        print(f"   Memory usage: {aggressive_settings['memory_safety_factor']*100:.0f}% (70%)")
        print(f"   Expected file size: ~{aggressive_settings['chunk_size'] * 24 * 4 / (1024**2):.1f} MB per day")
        print(f"   Queue size: {aggressive_settings['queue_maxsize']} (larger queue)")
        
        return aggressive_settings
    
    return settings

def extract_all_variables_from_grib_file(file_path, var_selectors, chunk_indices, grid_lats, grid_lons):
    """
    Extract ALL variables from a single GRIB file in one pass.
    
    This is much more efficient than opening the file multiple times.
    
    Args:
        file_path: Path to the GRIB file
        var_selectors: Dictionary of {var_name: var_selector} pairs
        chunk_indices: Indices to extract for this chunk
        grid_lats: Grid latitude array
        grid_lons: Grid longitude array
        
    Returns:
        dict: {var_name: {timestamp: {grid_id: value}}} for each variable found
    """
    import gc
    import os
    
    n_lats, n_lons = grid_lats.shape
    results = {var_name: {} for var_name in var_selectors.keys()}
    
    def extract_values_memory_efficient(grb, chunk_indices):
        """Extract values for chunk indices without loading entire grid into memory."""
        try:
            # Convert all chunk indices to 2D indices at once (more efficient)
            lat_indices, lon_indices = np.unravel_index(chunk_indices, (n_lats, n_lons))
            
            # Extract values using advanced indexing (more memory efficient)
            values_2d = grb.values
            chunk_values = values_2d[lat_indices, lon_indices]
            
            # Convert to float32 to save memory
            return chunk_values.astype(np.float32)
        except MemoryError as e:
            print(f"Memory error in extraction: {e}")
            # Fallback: process in smaller batches
            try:
                batch_size = 10000  # Process 10k points at a time
                chunk_values = []
                for i in range(0, len(chunk_indices), batch_size):
                    batch_indices = chunk_indices[i:i+batch_size]
                    lat_indices, lon_indices = np.unravel_index(batch_indices, (n_lats, n_lons))
                    values_2d = grb.values
                    batch_values = values_2d[lat_indices, lon_indices]
                    chunk_values.extend(batch_values)
                    del values_2d  # Explicitly delete to free memory
                    gc.collect()
                return np.array(chunk_values, dtype=np.float32)
            except Exception as e2:
                print(f"Fallback extraction also failed: {e2}")
                return None
        except Exception as e:
            print(f"Error in memory-efficient extraction: {e}")
            return None
    
    try:
        with pygrib.open(file_path) as grbs:
            # Track which variables we've found for each time offset
            found_variables = {var_name: set() for var_name in var_selectors.keys()}
            
            # Single pass through all GRIB messages
            for grb in grbs:
                grb_name = grb.name
                
                # Check if this message contains any of our target variables
                for var_name, var_selector in var_selectors.items():
                    if grb_name == var_selector:
                        # Get base timestamp from GRIB message
                        base_timestamp = pd.Timestamp(
                            year=grb.year, month=grb.month, day=grb.day,
                            hour=grb.hour, minute=grb.minute
                        )
                        
                        # Check for time offsets in this message
                        grb_str = str(grb)
                        time_offsets_found = []
                        
                        # Look for 15-minute intervals only (00 and 15 minutes)
                        for offset, minute in [(0, 0), (15, 15)]:
                            if offset == 0:
                                # For 00 offset, check if it's the base time or if no offset is mentioned
                                if "0 mins" in grb_str or "0m mins" in grb_str or "mins" not in grb_str:
                                    time_offsets_found.append((offset, minute))
                            else:
                                # For 15-minute offset, look for specific mentions
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
                            
                            # Extract values for this variable at this time
                            values = extract_values_memory_efficient(grb, chunk_indices)
                            
                            if values is not None:
                                # 🔍 NaN DETECTION: Check for missing data during extraction
                                nan_count = np.isnan(values).sum()
                                if nan_count > 0:
                                    print(f"  🔴 WARNING: Found {nan_count} NaN values in {var_name} at {dt}")
                                    
                                    # Log to file for detailed analysis
                                    nan_log_file = os.path.join(os.path.dirname(file_path), "nan_detection_extraction.log")
                                    with open(nan_log_file, 'a') as f:
                                        f.write(f"{datetime.datetime.now().isoformat()} | {var_name} | {dt} | {os.path.basename(file_path)} | NaNs: {nan_count}\n")
                                
                                # Create numbered column names for this chunk
                                chunk_columns = [f"grid_{i:06d}" for i in range(len(chunk_indices))]
                                results[var_name][dt] = dict(zip(chunk_columns, values))
                                found_variables[var_name].add(dt)
                        
                        # Don't break - continue processing other messages for the same variable
                        # This allows us to find multiple time offsets in the same file
                
                # Force garbage collection after each message
                del grb
                gc.collect()
                
                # Early exit if we've found all variables
                if len(found_variables) == len(var_selectors):
                    break
            
            # Report missing variables
            found_var_names = {var_name for var_name, timestamps in found_variables.items() if timestamps}
            missing_vars = set(var_selectors.keys()) - found_var_names
            if missing_vars:
                print(f"  Missing variables in {os.path.basename(file_path)}: {missing_vars}")
                
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        gc.collect()
    
    return results

def extract_grid_chunk_optimized(
    chunk_indices, files, var_selectors, grid_lats, grid_lons, use_parallel=True, max_file_groups=20000
):
    """
    Optimized version that extracts ALL variables from each GRIB file in one pass.
    
    This is much more efficient than the original approach which opened each file
    multiple times (once per variable).
    """
    import gc
    import re
    import time
    from collections import defaultdict
    
    chunk_start = time.time()
    n_lats, n_lons = grid_lats.shape
    
    print(f"Processing chunk with {len(chunk_indices):,} grid points")
    print(f"Variables to extract: {list(var_selectors.keys())}")
    
    # Group files by date/time for efficient processing
    grouping_start = time.time()
    file_groups = defaultdict(dict)
    
    for file_path in files:
        if not os.path.exists(file_path):
            continue
            
        # Extract date and forecast hour from filename
        filename = os.path.basename(file_path)
        
        # Parse date from filename (e.g., "hrrr.t21z.wrfsubhf00.grib2")
        try:
            # Extract date components
            if 'wrfsubhf00' in filename:
                # f00 file - hourly data
                forecast_type = 'f00'
            elif 'wrfsubhf01' in filename:
                # f01 file - quarter-hourly data
                forecast_type = 'f01'
            else:
                continue  # Skip unknown file types
                
            # Create a key for grouping (date + hour)
            # Extract time from filename (e.g., "t21z" -> hour 21)
            time_match = re.search(r't(\d{2})z', filename)
            if time_match:
                hour = int(time_match.group(1))
                # Create a key that groups f00 and f01 files for the same hour
                group_key = f"{hour:02d}"
                file_groups[group_key][forecast_type] = file_path
        except Exception as e:
            print(f"Error parsing filename {filename}: {e}")
            continue
    
    # Limit the number of file groups to prevent memory issues
    if len(file_groups) > max_file_groups:
        print(f"Warning: Limiting file groups from {len(file_groups)} to {max_file_groups}")
        # Keep the most recent file groups
        sorted_keys = sorted(file_groups.keys())
        file_groups = {k: file_groups[k] for k in sorted_keys[-max_file_groups:]}
    
    grouping_time = time.time() - grouping_start
    print(f"Grouped {len(files)} files into {len(file_groups)} groups")
    
    # Initialize results for all variables
    all_results = {var_name: {} for var_name in var_selectors.keys()}
    
    # Process each file group
    file_processing_start = time.time()
    files_processed = 0
    grib_read_time = 0
    
    for key in sorted(file_groups.keys()):
        group = file_groups[key]
        
        # Process f00 file (hourly data)
        if 'f00' in group:
            grib_read_start = time.time()
            f00_results = extract_all_variables_from_grib_file(
                group['f00'], var_selectors, chunk_indices, grid_lats, grid_lons
            )
            grib_read_time += time.time() - grib_read_start
            
            # Merge results
            for var_name, var_data in f00_results.items():
                all_results[var_name].update(var_data)
            
            files_processed += 1
        
        # Process f01 file (quarter-hourly data)
        if 'f01' in group:
            grib_read_start = time.time()
            f01_results = extract_all_variables_from_grib_file(
                group['f01'], var_selectors, chunk_indices, grid_lats, grid_lons
            )
            grib_read_time += time.time() - grib_read_start
            
            # For f01 files, the timestamps are already correct - no need to artificially create offsets
            for var_name, var_data in f01_results.items():
                for timestamp, values in var_data.items():
                    # Use the actual timestamp from the GRIB message
                    all_results[var_name][timestamp] = values
            
            files_processed += 1
        
        # Force garbage collection after processing each file group
        gc.collect()
    
    file_processing_time = time.time() - file_processing_start
    
    # Convert results to DataFrames for each variable
    df_conversion_start = time.time()
    final_results = {}
    
    for var_name, var_data in all_results.items():
        if var_data:
            # Convert to DataFrame, sort by time, round to 3 decimals
            df = pd.DataFrame.from_dict(var_data, orient='index').sort_index()
            df.index.name = 'time'
            # Round to exactly 3 decimal places
            df = (df * 1000).round().astype('int32') / 1000.0
            final_results[var_name] = df
        else:
            print(f"Warning: No data found for variable {var_name}")
            final_results[var_name] = pd.DataFrame()
    
    # Clean up memory
    del all_results
    gc.collect()
    
    df_conversion_time = time.time() - df_conversion_start
    total_chunk_time = time.time() - chunk_start
    
    print(f"Chunk timing: grouping={grouping_time:.2f}s, file_processing={file_processing_time:.2f}s (grib_read={grib_read_time:.2f}s), df_conversion={df_conversion_time:.2f}s, total={total_chunk_time:.2f}s")
    
    return final_results

def extract_full_grid_optimized_with_preloaded_data(
    START,
    END,
    DATADIR,
    DEFAULT_HOURS_FORECASTED,
    SELECTORS,
    output_dir="./extracted_grid_data",
    chunk_size=50000,   # SMALLER chunks for better pyarrow compatibility
    compression="snappy",
    use_parallel=True,
    time_period_parallel=True,  # New: parallelize time periods
    num_cpu_workers=4,  # Much more conservative based on actual usage
    num_io_workers=1,   # Reduced I/O workers
    queue_maxsize=8,   # Smaller queue size
    max_file_groups=5000,  # Reduced file group limit
    create_individual_mappings=False,  # Option to create individual mapping files
    parallel_file_writing=True,  # New: parallel file writing
    enable_resume=True,  # New: enable resume functionality
    # 🚀 OPTIMIZATION: Pre-loaded data parameters
    grid_lats=None,
    grid_lons=None,
    global_mapping=None,
):
    """
    Optimized version of extract_full_grid_optimized that accepts pre-loaded grid data.
    
    This avoids re-extracting grid metadata and re-creating mappings for each day.
    
    Args:
        ... (same as extract_full_grid_optimized)
        grid_lats: Pre-loaded grid latitudes (if None, will extract)
        grid_lons: Pre-loaded grid longitudes (if None, will extract)
        global_mapping: Pre-loaded global mapping (if None, will create)
    """
    import functools
    import gc
    import glob
    import multiprocessing as mp
    import queue as pyqueue

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(cleanup_on_exit)
    
    def log_timing(phase, start_time=None, details=None):
        """Log timing information for a phase."""
        if start_time is not None:
            elapsed = time.time() - start_time
            print(f"⏱️  {phase}: {elapsed:.2f}s")
            if details:
                for key, value in details.items():
                    if isinstance(value, (int, float)):
                        if value > 1000:
                            print(f"   {key}: {value:,}")
                        else:
                            print(f"   {key}: {value}")
                    else:
                        print(f"   {key}: {value}")
    
    print(f"🚀 Starting optimized full grid extraction")
    print(f"Date range: {START} to {END}")
    print(f"Variables: {list(SELECTORS.keys())}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Phase 1: Resume functionality (if enabled)
    resume_start = time.time()
    processed_dates = set()
    date_range = set(pd.date_range(start=START, end=END, freq="1D").date)
    
    if enable_resume:
        print("Phase 1.5: Resume functionality...")
        resume_metadata = load_resume_metadata(output_dir)
        if resume_metadata:
            processed_dates = set(resume_metadata.get('processed_dates', []))
            print(f"Found {len(processed_dates)} already processed dates")
        else:
            print("No resume metadata found")
    
    # Phase 2: Grid metadata extraction (use pre-loaded if available)
    grid_start = time.time()
    if grid_lats is None or grid_lons is None:
        print("Extracting grid metadata...")
        wind_data_lat_long = get_wind_data_lat_long(START, DATADIR)
        grid_lats, grid_lons = wind_data_lat_long[0], wind_data_lat_long[1]
    else:
        print("Using pre-loaded grid metadata...")
    
    n_lats, n_lons = grid_lats.shape
    total_grid_points = n_lats * n_lons
    
    print(f"Grid dimensions: {n_lats} x {n_lons} = {total_grid_points:,} total points")
    log_timing("Grid metadata extraction", grid_start)
    
    # Phase 3: Create global mapping file (use pre-loaded if available)
    mapping_start = time.time()
    if global_mapping is None:
        print("Creating global grid mapping...")
        global_mapping = create_global_grid_mapping(grid_lats, grid_lons, output_dir, format="parquet")
    else:
        print("Using pre-loaded global mapping...")
    log_timing("Global mapping creation", mapping_start)
    
    # Phase 4: Find GRIB files for specific hours only
    file_search_start = time.time()
    print(f"Searching for GRIB files in: {DATADIR}")
    
    # Use remaining dates if resume is enabled, otherwise use full range
    search_date_range = date_range - processed_dates if enable_resume else date_range
    print(f"Searching for dates: {len(search_date_range)} dates")
    
    # Calculate the specific hours we need based on START and END
    start_hour = START.hour
    end_hour = END.hour
    target_hours = list(range(start_hour, end_hour + 1))
    print(f"Target hours: {target_hours}")
    
    all_files = []
    for date in sorted(search_date_range):
        date_str = date.strftime("%Y%m%d")
        date_dir = os.path.join(DATADIR, date_str)
        if os.path.exists(date_dir):
            # Find files for specific hours only
            date_files = []
            for hour in target_hours:
                # Look for both f00 and f01 files for each hour
                for forecast_type in ["00", "01"]:
                    # Pattern: hrrr.t{HH}z.wrfsubhf{XX}.grib2
                    hour_str = f"{hour:02d}"
                    pattern = f"hrrr.t{hour_str}z.wrfsubhf{forecast_type}.grib2"
                    file_path = os.path.join(date_dir, pattern)
                    if os.path.exists(file_path):
                        date_files.append(file_path)
                        print(f"    Found: {pattern}")
                    else:
                        print(f"    Missing: {pattern}")
            
            # Filter out subset files which appear to be corrupted
            valid_files = [f for f in date_files if "subset_" not in os.path.basename(f)]
            all_files.extend(valid_files)
            print(f"  Found {len(date_files)} total files, using {len(valid_files)} valid files in {date_str}")
        else:
            print(f"  No directory found for {date_str}")
    if not all_files:
        print(f"No GRIB files found in date range {min(search_date_range)} to {max(search_date_range)}")
        return {}
    print(f"Found {len(all_files)} total GRIB files")
    files = all_files
    print(f"Using {len(files)} GRIB files for extraction")
    log_timing("File discovery", file_search_start, {
        'total_files': len(files),
        'search_dates': len(search_date_range),
        'processed_dates_skipped': len(processed_dates) if enable_resume else 0,
        'target_hours': target_hours
    })
    
    # Phase 5: Calculate chunking strategy
    chunk_calc_start = time.time()
    n_chunks = (total_grid_points + chunk_size - 1) // chunk_size
    print(f"Processing in {n_chunks} chunks")
    log_timing("Chunk calculation", chunk_calc_start, {'n_chunks': n_chunks})
    
    # Phase 6: Memory-aware parallel processing setup
    memory_setup_start = time.time()
    available_memory_gb = get_available_memory()
    safe_worker_count = calculate_safe_worker_count(available_memory_gb, chunk_size, len(SELECTORS))
    
    print(f"Available memory: {available_memory_gb:.1f} GB")
    print(f"Safe worker count: {safe_worker_count}")
    print(f"Requested worker count: {num_cpu_workers}")
    
    # Use the smaller of safe_worker_count and num_cpu_workers
    actual_workers = min(safe_worker_count, num_cpu_workers)
    print(f"Using {actual_workers} workers")
    
    # Monitor initial memory state
    print(f"Initial memory usage: {get_memory_usage():.1f} MB")
    monitor_memory_usage()
    log_timing("Memory setup", memory_setup_start, {'actual_workers': actual_workers})
    
    # Phase 7: Prepare tasks
    task_prep_start = time.time()
    tasks = []
    for var_name, var_selector in SELECTORS.items():
        for chunk_idx in range(n_chunks):
            tasks.append((chunk_idx, var_name, var_selector))
    print(f"Total tasks: {len(tasks)} (variables × chunks)")
    log_timing("Task preparation", task_prep_start, {'total_tasks': len(tasks)})
    
    # Phase 8: Process chunks in parallel
    processing_start = time.time()
    cpu_func = functools.partial(
        cpu_worker,
        files=files,
        grid_lats=grid_lats,
        grid_lons=grid_lons,
        chunk_size=chunk_size,
        total_grid_points=total_grid_points,
        output_dir=output_dir,
        max_file_groups=max_file_groups,
        create_individual_mappings=create_individual_mappings,
    )
    
    print("Processing chunks in parallel...")
    print(f"Memory usage before processing: {get_memory_usage():.1f} MB")
    
    # Process in smaller batches to avoid memory buildup
    batch_size = min(actual_workers * 2, len(tasks))  # Process 2x workers at a time
    print(f"Processing in batches of {batch_size} tasks")
    
    all_results = []
    batch_times = []
    
    for batch_start in range(0, len(tasks), batch_size):
        batch_end = min(batch_start + batch_size, len(tasks))
        batch = tasks[batch_start:batch_end]
        
        # Check for shutdown request before each batch
        if check_shutdown_requested():
            print("🛑 Shutdown requested during processing. Saving partial results...")
            break
        
        batch_start_time = time.time()
        print(f"Processing batch {batch_start//batch_size + 1}/{(len(tasks) + batch_size - 1)//batch_size}")
        
        try:
            with mp.Pool(processes=actual_workers) as pool:
                batch_results = list(tqdm(
                    pool.imap(cpu_func, batch),
                    total=len(batch),
                    desc=f"Batch {batch_start//batch_size + 1}"
                ))
            all_results.extend(batch_results)
            
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            print(f"Batch completed in {batch_time:.2f}s")
            
            # Monitor memory after each batch
            memory_usage = get_memory_usage()
            print(f"Memory usage after batch: {memory_usage:.1f} MB")
            
            # Force garbage collection between batches
            gc.collect()
            
        except Exception as e:
            print(f"Error in batch processing: {e}")
            continue
    
    processing_time = time.time() - processing_start
    log_timing("Parallel processing", processing_start, {
        'total_tasks': len(tasks),
        'completed_tasks': len([r for r in all_results if r is not None]),
        'failed_tasks': len([r for r in all_results if r is None]),
        'avg_batch_time': sum(batch_times) / len(batch_times) if batch_times else 0
    })
    
    # Phase 9: File writing (parallel if enabled)
    writing_start = time.time()
    if parallel_file_writing and all_results:
        print("Phase 9: Parallel file writing...")
        
        # Prepare file writing tasks
        file_writing_tasks = []
        for result in all_results:
            if result is not None:
                chunk_data, mapping, chunk_idx, var_name = result
                file_writing_tasks.append((
                    chunk_data, mapping, chunk_idx, var_name, 
                    output_dir, compression, create_individual_mappings
                ))
        
        if file_writing_tasks:
            print(f"Writing {len(file_writing_tasks)} chunks in parallel...")
            
            try:
                with mp.Pool(processes=num_io_workers) as pool:
                    file_results = list(tqdm(
                        pool.imap(file_writing_worker, file_writing_tasks),
                        total=len(file_writing_tasks),
                        desc="File writing"
                    ))
                
                # Calculate total file size
                total_size_mb = sum(r.get('total_size_mb', 0) for r in file_results if r)
                print(f"File writing completed. Total size: {total_size_mb:.1f} MB")
                
            except Exception as e:
                print(f"Error in parallel file writing: {e}")
                print("Falling back to sequential file writing...")
                
                # Fallback to sequential writing
                for task in tqdm(file_writing_tasks, desc="Sequential file writing"):
                    try:
                        file_writing_worker(task)
                    except Exception as e:
                        print(f"Error writing file: {e}")
        else:
            print("No data to write")
    else:
        print("Phase 9: Sequential file writing...")
        for result in tqdm(all_results, desc="File writing"):
            if result is not None:
                chunk_data, mapping, chunk_idx, var_name = result
                try:
                    file_writing_worker((
                        chunk_data, mapping, chunk_idx, var_name,
                        output_dir, compression, create_individual_mappings
                    ))
                except Exception as e:
                    print(f"Error writing file: {e}")
    
    log_timing("File writing", writing_start)
    
    # Phase 10: Post-processing
    post_start = time.time()
    print("Phase 10: Post-processing...")
    
    # Add wind speed calculations if U and V components are present
    drop_uv = os.getenv('HRRR_DROP_UV_AFTER_WS', '1') in ('1', 'true', 'True')
    if any(var in SELECTORS for var in ['UWind80', 'VWind80', 'UWind10', 'VWind10']):
        print("Adding wind speed calculations...")
        add_wind_speed_calculations(output_dir)
        if drop_uv:
            # Remove U/V directories to save space
            for uv_var in ['UWind80', 'VWind80', 'UWind10', 'VWind10']:
                uv_dir = os.path.join(output_dir, uv_var)
                if os.path.exists(uv_dir):
                    try:
                        import shutil
                        shutil.rmtree(uv_dir)
                        print(f"Dropped U/V data: {uv_dir}")
                    except Exception as e:
                        print(f"Warning: could not remove {uv_dir}: {e}")
    
    # Create final resume metadata
    if enable_resume:
        final_processed_dates = get_processed_date_range(output_dir, SELECTORS)
        create_resume_metadata(output_dir, START, END, SELECTORS, final_processed_dates)
        print(f"Resume metadata updated. Total processed dates: {len(final_processed_dates)}")
    
    log_timing("Post-processing", post_start)
    
    # Final timing summary
    total_time = time.time() - resume_start
    print("=" * 60)
    print("📊 EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Total processing time: {total_time:.1f}s ({total_time/3600:.1f} hours)")
    print(f"Grid points processed: {total_grid_points:,}")
    print(f"Chunks processed: {n_chunks}")
    print(f"Files processed: {len(files)}")
    print(f"Variables extracted: {list(SELECTORS.keys())}")
    print(f"Output directory: {output_dir}")
    
    return {
        "status": "completed" if not check_shutdown_requested() else "interrupted",
        "total_time_seconds": total_time,
        "grid_points": total_grid_points,
        "chunks": n_chunks,
        "files_processed": len(files),
        "variables": list(SELECTORS.keys()),
        "output_directory": output_dir,
        "resume_used": enable_resume and len(processed_dates) > 0,
        "processed_dates": list(processed_dates),
    }

def extract_full_grid_aggressive_optimized(
    START,
    END,
    DATADIR,
    DEFAULT_HOURS_FORECASTED,
    SELECTORS,
    output_dir="./extracted_grid_data",
    use_aggressive_settings=True,  # Use aggressive settings for 36 CPUs, 256GB RAM
):
    """
    🚀 AGGRESSIVE OPTIMIZATION: Extract full grid data with maximum parallelization.
    
    This function implements the recommended aggressive parallelization strategy:
    - Uses ALL 36 CPU cores
    - Leverages 256GB RAM efficiently
    - Creates single global mapping (no redundancy)
    - Optimized for maximum throughput
    
    Args:
        START: Start datetime
        END: End datetime  
        DATADIR: Data directory
        DEFAULT_HOURS_FORECASTED: Forecast hours
        SELECTORS: Variable selectors
        output_dir: Output directory
        use_aggressive_settings: Use aggressive settings for high-performance systems
    """
    import functools
    import gc
    import glob
    import multiprocessing as mp

    # Get aggressive settings for your 36 CPU, 256GB system
    if use_aggressive_settings:
        settings = get_aggressive_parallel_settings()
        print("🚀 Using AGGRESSIVE settings for maximum performance:")
        print(f"   Chunk size: {settings['chunk_size']:,}")
        print(f"   CPU workers: {settings['num_cpu_workers']}")
        print(f"   I/O workers: {settings['num_io_workers']}")
        print(f"   Memory usage: {settings['memory_safety_factor']*100:.0f}%")
        
        chunk_size = settings['chunk_size']
        num_cpu_workers = settings['num_cpu_workers']
        num_io_workers = settings['num_io_workers']
        max_file_groups = settings['max_file_groups']
        compression = settings['compression']
    else:
        # Conservative settings
        chunk_size = 50000
        num_cpu_workers = 8
        num_io_workers = 2
        max_file_groups = 5000
        compression = "snappy"
    
    start_time = time.time()
    print(f"🚀 Starting AGGRESSIVE full grid extraction at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Date range: {START} to {END}")
    print(f"Output directory: {output_dir}")
    print(f"Chunk size: {chunk_size:,}")
    print(f"CPU workers: {num_cpu_workers}")
    print(f"I/O workers: {num_io_workers}")
    
    # Phase 1: Setup and directory creation
    os.makedirs(output_dir, exist_ok=True)
    mapping_dir = os.path.join(output_dir, "mappings")
    os.makedirs(mapping_dir, exist_ok=True)
    
    # Phase 2: Get grid dimensions and metadata
    print("Extracting grid metadata...")
    wind_data_lat_long = get_wind_data_lat_long(START, DATADIR)
    grid_lats, grid_lons = wind_data_lat_long[0], wind_data_lat_long[1]
    n_lats, n_lons = grid_lats.shape
    total_grid_points = n_lats * n_lons
    print(f"Grid dimensions: {n_lats} x {n_lons} = {total_grid_points:,} total points")
    
    # Phase 3: Create SINGLE global mapping (no redundancy!)
    print("Creating SINGLE global grid mapping...")
    global_mapping = create_global_grid_mapping(grid_lats, grid_lons, output_dir, format="parquet")
    print("✅ Global mapping created - all variables will use this single mapping")
    
    # Phase 4: Find all GRIB files
    print(f"Searching for GRIB files in: {DATADIR}")
    date_range = pd.date_range(start=START.date(), end=END.date(), freq="1D")
    all_files = []
    for date in date_range:
        date_str = date.strftime("%Y%m%d")
        date_dir = os.path.join(DATADIR, date_str)
        if os.path.exists(date_dir):
            grib_pattern = os.path.join(date_dir, "*.grib2")
            date_files = glob.glob(grib_pattern)
            # Filter out subset files which appear to be corrupted
            valid_files = [f for f in date_files if "subset_" not in os.path.basename(f)]
            all_files.extend(valid_files)
            print(f"  Found {len(date_files)} total files, using {len(valid_files)} valid files in {date_str}")
        else:
            print(f"  No directory found for {date_str}")
    
    if not all_files:
        print(f"No GRIB files found in date range {START.date()} to {END.date()}")
        return {}
    
    print(f"Found {len(all_files)} total GRIB files")
    files = all_files
    
    # Phase 5: Calculate chunking strategy
    n_chunks = (total_grid_points + chunk_size - 1) // chunk_size
    print(f"Processing in {n_chunks} chunks")
    
    # Phase 6: Memory-aware parallel processing setup
    available_memory_gb = get_available_memory()
    print(f"Available memory: {available_memory_gb:.1f} GB")
    print(f"Using {num_cpu_workers} CPU workers (aggressive)")
    
    # Prepare tasks (for all variables)
    tasks = []
    for var_name, var_selector in SELECTORS.items():
        for chunk_idx in range(n_chunks):
            tasks.append((chunk_idx, var_name, var_selector))
    print(f"Total tasks: {len(tasks)} (variables × chunks)")
    
    # Process chunks in parallel using aggressive settings
    cpu_func = functools.partial(
        cpu_worker_aggressive,  # New aggressive worker
        files=files,
        grid_lats=grid_lats,
        grid_lons=grid_lons,
        chunk_size=chunk_size,
        total_grid_points=total_grid_points,
        output_dir=output_dir,
        max_file_groups=max_file_groups,
        global_mapping=global_mapping,  # Pass global mapping
    )
    
    print("🚀 Processing chunks with AGGRESSIVE parallelization...")
    print(f"Memory usage before processing: {get_memory_usage():.1f} MB")
    
    with mp.Pool(num_cpu_workers) as pool:
        results = pool.map(cpu_func, tasks)
    
    print(f"Memory usage after processing: {get_memory_usage():.1f} MB")
    
    # Force garbage collection after processing
    gc.collect()
    print(f"Memory usage after cleanup: {get_memory_usage():.1f} MB")
    
    # Write results to files organized by date
    print("Writing results to files organized by date...")
    for result in results:
        if result is not None:
            chunk_data, chunk_idx, var_name = result  # No mapping needed - using global mapping
            
            # Group data by date
            chunk_data['date'] = chunk_data.index.date
            date_groups = chunk_data.groupby('date')
            
            for date, date_data in date_groups:
                # Remove the date column and keep only the data
                date_data = date_data.drop('date', axis=1)
                date_str = date.strftime('%Y%m%d')
                
                # Create date-specific directories
                var_output_dir = os.path.join(output_dir, var_name, date_str)
                os.makedirs(var_output_dir, exist_ok=True)
                
                # Write parquet file for this date
                chunk_filename = f"{var_name}_chunk_{chunk_idx:04d}.parquet"
                chunk_path = os.path.join(var_output_dir, chunk_filename)
                date_data.to_parquet(chunk_path, compression=compression, engine='pyarrow')
    
    print("\n✅ AGGRESSIVE parallel processing complete! All chunks processed and written.")
    
    # Post-process to add wind speed calculations
    print("Calculating wind speed from U and V components...")
    add_wind_speed_calculations(output_dir)
    
    # Add final timing information
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"🚀 AGGRESSIVE EXTRACTION COMPLETED")
    print(f"{'='*60}")
    print(f"Total processing time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"Output directory: {output_dir}")
    print(f"Global mapping saved in: {mapping_dir}")
    print(f"CPU utilization: {num_cpu_workers} workers")
    print(f"Memory usage: {get_memory_usage():.1f} MB")
    print(f"{'='*60}")
    
    return {
        "status": "completed", 
        "processing_time_seconds": total_time,
        "cpu_workers_used": num_cpu_workers,
        "memory_usage_mb": get_memory_usage(),
        "global_mapping_used": True
    }

def cpu_worker_aggressive(task, files, grid_lats, grid_lons, chunk_size, total_grid_points, output_dir, max_file_groups=50000, global_mapping=None):
    """🚀 AGGRESSIVE CPU worker that uses global mapping (no individual mappings)."""
    import gc
    
    chunk_idx, var_name, var_selector = task
    start_idx = chunk_idx * chunk_size
    end_idx = min(start_idx + chunk_size, total_grid_points)
    chunk_indices = list(range(start_idx, end_idx))
    
    # Monitor memory before processing
    memory_before = get_memory_usage()
    
    chunk_data = extract_grid_chunk(
        chunk_indices, files, var_selector, grid_lats, grid_lons, 
        use_parallel=False, max_file_groups=max_file_groups
    )
    
    if chunk_data is not None and not chunk_data.empty:
        # Round to exactly 3 decimal places by multiplying by 1000, rounding, then dividing
        chunk_data = (chunk_data * 1000).round().astype('int32') / 1000.0
        
        # Monitor memory after processing
        memory_after = get_memory_usage()
        if memory_after - memory_before > 100:  # If memory increased by more than 100MB
            print(f"Memory spike in {var_name} chunk {chunk_idx}: {memory_before:.1f} -> {memory_after:.1f} MB")
        
        # Force garbage collection
        gc.collect()
        
        # Return only data and metadata - NO individual mapping needed!
        return (chunk_data, chunk_idx, var_name)
    
    # Force garbage collection even if no data
    gc.collect()
    return None

def extract_specific_locations_day_by_day(
    points,
    START,
    END,
    DATADIR,
    DEFAULT_HOURS_FORECASTED,
    SELECTORS,
    output_dir="./extracted_specific_locations",
    compression="snappy",
    use_parallel=True,
    num_cpu_workers=None,  # Auto-detect based on system
    num_io_workers=None,   # Auto-detect based on system
    max_file_groups=None,  # Auto-detect based on system
    parallel_file_writing=True,
    enable_resume=True,
    day_output_dir_format="flat",  # "daily" or "flat"
    use_aggressive_settings=True,  # Use aggressive settings for high-performance systems
):
    """
    Extract HRRR data for specific lat/lon locations using day-by-day processing.
    
    This function combines the efficiency of day-by-day processing with the precision
    of closest point calculation for specific locations.
    
    Args:
        points: DataFrame with columns ['lat', 'lon', 'pid'] for target locations
        START: Start datetime
        END: End datetime
        DATADIR: Directory containing GRIB files
        DEFAULT_HOURS_FORECASTED: List of forecast hours to process
        SELECTORS: Dictionary mapping variable names to GRIB variable names
        output_dir: Output directory for extracted data
        compression: Parquet compression method
        use_parallel: Whether to use parallel processing
        num_cpu_workers: Number of CPU workers (auto-detect if None)
        num_io_workers: Number of I/O workers (auto-detect if None)
        max_file_groups: Maximum file groups to process (auto-detect if None)
        parallel_file_writing: Whether to write files in parallel
        enable_resume: Whether to enable resume functionality
        day_output_dir_format: "daily" for subdirectories, "flat" for single directory
        use_aggressive_settings: Whether to use aggressive settings for high-performance systems
    
    Returns:
        Dictionary with extraction results and metadata
    """
    import datetime
    import os
    # Signal handling for graceful shutdown
    import signal
    import time
    from collections import defaultdict

    import numpy as np
    import pandas as pd
    from scipy.spatial import KDTree
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("🚀 Starting Specific Locations Day-by-Day Extraction")
    print("=" * 60)
    print(f"📍 Target locations: {len(points)} points")
    print(f"📅 Date range: {START.date()} to {END.date()}")
    print(f"📊 Variables: {list(SELECTORS.keys())}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🔄 Resume enabled: {enable_resume}")
    print(f"⚡ Aggressive settings: {use_aggressive_settings}")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Auto-detect system settings if not provided
    if use_aggressive_settings:
        settings = get_aggressive_parallel_settings()
        if num_cpu_workers is None:
            num_cpu_workers = settings['num_cpu_workers']
        if num_io_workers is None:
            num_io_workers = settings['num_io_workers']
        if max_file_groups is None:
            max_file_groups = settings['max_file_groups']
    else:
        settings = get_optimized_settings_for_high_performance_system()
        if num_cpu_workers is None:
            num_cpu_workers = settings['num_cpu_workers']
        if num_io_workers is None:
            num_io_workers = settings['num_io_workers']
        if max_file_groups is None:
            max_file_groups = settings['max_file_groups']
    
    print(f"🔧 Detected system: {os.cpu_count()} CPUs, {get_available_memory():.1f} GB RAM")
    print(f"⚙️  Using settings:")
    print(f"   CPU workers: {num_cpu_workers}")
    print(f"   I/O workers: {num_io_workers}")
    print(f"   Max file groups: {max_file_groups}")
    print()
    
    # Generate date range
    date_range = pd.date_range(start=START, end=END, freq='D')
    total_days = len(date_range)
    
    # Resume functionality
    successful_days = []
    failed_days = []
    skipped_days = []
    
    if enable_resume:
        # Load existing progress
        resume_file = os.path.join(output_dir, "resume_metadata.json")
        if os.path.exists(resume_file):
            try:
                import json
                with open(resume_file, 'r') as f:
                    resume_data = json.load(f)
                
                # Check if this is the same extraction
                if (resume_data.get('start_date') == START.isoformat() and
                    resume_data.get('end_date') == END.isoformat() and
                    resume_data.get('selectors') == SELECTORS):
                    
                    processed_dates = [pd.Timestamp(d) for d in resume_data.get('processed_dates', [])]
                    successful_days = [d.date() for d in processed_dates]
                    
                    # Filter out already processed days
                    remaining_dates = [d for d in date_range if d.date() not in [pd.Timestamp(d).date() for d in resume_data.get('processed_dates', [])]]
                    
                    print(f"📋 Found {len(successful_days)} already processed days:")
                    for date in successful_days[:5]:  # Show first 5
                        print(f"   - {date}")
                    if len(successful_days) > 5:
                        print(f"   ... and {len(successful_days) - 5} more")
                    print(f"📅 Remaining days to process: {len(remaining_dates)}")
                else:
                    print("📋 Resume metadata found but parameters changed. Starting fresh.")
                    remaining_dates = date_range
            except Exception as e:
                print(f"⚠️  Error loading resume metadata: {e}")
                remaining_dates = date_range
        else:
            print("📋 No previously processed days found.")
            remaining_dates = date_range
    else:
        remaining_dates = date_range
    
    if len(remaining_dates) == 0:
        print("✅ All days already processed! Extraction complete.")
        return {
            "status": "completed",
            "total_days": len(date_range),
            "successful_days": len(successful_days),
            "failed_days": len(failed_days),
            "skipped_days": len(skipped_days),
            "processing_time_seconds": 0,
            "resume_used": True
        }
    
    print(f"Processing {len(remaining_dates)} days...")
    print()
    
    # Get grid metadata once (for closest point calculation)
    print("📊 Extracting grid metadata (once for all days)...")
    grid_start_time = time.time()
    
    # Get grid data from first available GRIB file
    wind_data_lat_long = get_wind_data_lat_long(START, DATADIR)
    grid_lats, grid_lons = wind_data_lat_long
    
    print(f"Grid dimensions: {grid_lats.shape[0]} x {grid_lats.shape[1]} = {grid_lats.size:,} total points")
    print(f"Grid metadata extracted in {time.time() - grid_start_time:.2f}s")
    print()
    
    # Calculate closest grid points for target locations
    print("📍 Calculating closest grid points for target locations...")
    closest_start_time = time.time()
    
    # Use the existing find_closest_wind_grids function
    closest_indices = find_closest_wind_grids(points, wind_data_lat_long)
    
    print(f"✅ Found closest points for {len(points)} locations in {time.time() - closest_start_time:.2f}s")
    print(f"📍 Target locations: {len(points)}")
    print(f"🎯 Closest grid points: {len(closest_indices)}")
    print()
    
    # Process each day
    total_start_time = time.time()
    
    for day_idx, current_date in enumerate(remaining_dates, 1):
        day_start_time = time.time()
        
        # Check for shutdown request
        if check_shutdown_requested():
            print("🛑 Shutdown requested. Saving progress...")
            break
        
        print(f"📅 Processing day {day_idx}/{len(remaining_dates)}: {current_date.date()}")
        print(f"   Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Variables: {list(SELECTORS.keys())}")
        
        # Set day-specific start and end times
        day_start = datetime.datetime.combine(current_date.date(), START.time())
        day_end = datetime.datetime.combine(current_date.date(), START.time()) + datetime.timedelta(days=1)
        
        # Determine output directory for this day
        if day_output_dir_format == "daily":
            # Each day gets its own subdirectory
            day_output_dir = os.path.join(output_dir, current_date.strftime("%Y%m%d"))
        else:
            # Flat structure - all days in same directory
            day_output_dir = output_dir
        
        print(f"   Output: {day_output_dir}")
        
        try:
            # Process this day for specific locations
            day_result = process_day_for_specific_locations(
                day_start=day_start,
                day_end=day_end,
                DATADIR=DATADIR,
                DEFAULT_HOURS_FORECASTED=DEFAULT_HOURS_FORECASTED,
                SELECTORS=SELECTORS,
                closest_indices=closest_indices,
                points=points,
                output_dir=day_output_dir,
                compression=compression,
                use_parallel=use_parallel,
                num_cpu_workers=num_cpu_workers,
                num_io_workers=num_io_workers,
                max_file_groups=max_file_groups,
                parallel_file_writing=parallel_file_writing,
            )
            
            day_time = time.time() - day_start_time
            
            if day_result and day_result.get("status") == "completed":
                successful_days.append(current_date.date())
                print(f"   ✅ Completed in {day_time:.1f}s ({day_time/60:.1f} minutes)")
                print(f"   📁 Files written to: {day_output_dir}")
            else:
                failed_days.append(current_date.date())
                print(f"   ❌ Failed after {day_time:.1f}s ({day_time/60:.1f} minutes)")
                
        except Exception as e:
            day_time = time.time() - day_start_time
            failed_days.append(current_date.date())
            print(f"   ❌ Error after {day_time:.1f}s: {e}")
            
            # Continue with next day instead of stopping
            print(f"   Continuing with next day...")
            continue
        
        # Progress update
        total_time_so_far = time.time() - total_start_time
        avg_time_per_day = total_time_so_far / day_idx
        remaining_days = len(remaining_dates) - day_idx
        estimated_remaining_time = remaining_days * avg_time_per_day
        
        print(f"   📊 Progress: {day_idx}/{len(remaining_dates)} days")
        print(f"   ✅ Success rate: {len(successful_days)}/{day_idx} ({len(successful_days)/day_idx*100:.1f}%)")
        print(f"   ⏱️  Estimated time remaining: {estimated_remaining_time/3600:.1f} hours")
        print(f"   🗓️  Next day: {remaining_dates[day_idx].date() if day_idx < len(remaining_dates) else 'COMPLETE'}")
        print()
        
        # Save progress after each day
        if enable_resume:
            try:
                resume_data = {
                    'start_date': START.isoformat(),
                    'end_date': END.isoformat(),
                    'selectors': SELECTORS,
                    'processed_dates': [d.isoformat() for d in successful_days],
                    'last_updated': datetime.datetime.now().isoformat()
                }
                with open(os.path.join(output_dir, "resume_metadata.json"), 'w') as f:
                    import json
                    json.dump(resume_data, f, indent=2)
            except Exception as e:
                print(f"⚠️  Warning: Could not save resume metadata: {e}")
        
        # Force garbage collection between days
        import gc
        gc.collect()
        
        # Small delay to allow system to stabilize
        time.sleep(1)
    
    # Final summary
    total_time = time.time() - total_start_time
    
    print("🎉 SPECIFIC LOCATIONS EXTRACTION COMPLETE!")
    print("=" * 60)
    print(f"📊 Summary:")
    print(f"   Total days: {len(date_range)}")
    print(f"   Successful: {len(successful_days)}")
    print(f"   Failed: {len(failed_days)}")
    print(f"   Skipped: {len(skipped_days)}")
    print(f"   Total time: {total_time/3600:.1f} hours")
    print(f"   Average time per day: {total_time/len(date_range)/60:.1f} minutes")
    print(f"   Target locations: {len(points)}")
    print(f"   Variables: {list(SELECTORS.keys())}")
    print(f"   Output directory: {output_dir}")
    
    return {
        "status": "completed" if len(failed_days) == 0 else "completed_with_errors",
        "total_days": len(date_range),
        "successful_days": len(successful_days),
        "failed_days": len(failed_days),
        "skipped_days": len(skipped_days),
        "processing_time_seconds": total_time,
        "target_locations": len(points),
        "variables": list(SELECTORS.keys()),
        "output_directory": output_dir,
        "resume_used": enable_resume
    }


def process_day_for_specific_locations(
    day_start,
    day_end,
    DATADIR,
    DEFAULT_HOURS_FORECASTED,
    SELECTORS,
    closest_indices,
    points,
    output_dir,
    compression="snappy",
    use_parallel=True,
    num_cpu_workers=4,
    num_io_workers=1,
    max_file_groups=5000,
    parallel_file_writing=True,
):
    """
    Process a single day for specific locations.
    
    Args:
        day_start: Start datetime for this day
        day_end: End datetime for this day
        DATADIR: Directory containing GRIB files
        DEFAULT_HOURS_FORECASTED: List of forecast hours to process
        SELECTORS: Dictionary mapping variable names to GRIB variable names
        closest_indices: Array of closest grid point indices for target locations
        points: DataFrame with target locations
        output_dir: Output directory for this day
        compression: Parquet compression method
        use_parallel: Whether to use parallel processing
        num_cpu_workers: Number of CPU workers
        num_io_workers: Number of I/O workers
        max_file_groups: Maximum file groups to process
        parallel_file_writing: Whether to write files in parallel
    
    Returns:
        Dictionary with processing results
    """
    import datetime
    import os
    import time
    from collections import defaultdict

    import numpy as np
    import pandas as pd
    
    print(f"   🔍 Processing {day_start.date()} ({day_start.time()} to {day_end.time()})")
    
    # Find GRIB files for this day
    day_files = []
    
    # Try the expected naming pattern first
    for dt in pd.date_range(day_start, day_end - datetime.timedelta(hours=1), freq='h'):
        for hour in DEFAULT_HOURS_FORECASTED:
            # Convert hour to integer for formatting
            hour_int = int(hour)
            filename = f"hrrr.t{dt.hour:02d}z.wrfsubhf{hour_int:02d}.grib2"  # Fixed: wrfsubhf instead of wrfprsf
            filepath = os.path.join(DATADIR, dt.strftime('%Y%m%d'), filename)
            if os.path.exists(filepath):
                day_files.append((filepath, dt))
    
    # If no files found with expected pattern, try alternative patterns
    if not day_files:
        day_str = day_start.strftime('%Y%m%d')
        day_dir = os.path.join(DATADIR, day_str)
        
        if os.path.exists(day_dir):
            # Look for any .grib2 files in the day directory
            grib_files = [f for f in os.listdir(day_dir) if f.endswith('.grib2')]
            for filename in grib_files:
                filepath = os.path.join(day_dir, filename)
                # Try to extract datetime from filename
                try:
                    # Extract hour from filename (e.g., "hrrr.t00z.wrfprsf00.grib2" -> hour 0)
                    if 't' in filename and 'z' in filename:
                        hour_part = filename.split('t')[1].split('z')[0]
                        dt = day_start.replace(hour=int(hour_part))
                    else:
                        dt = day_start  # Default to start of day
                    day_files.append((filepath, dt))
                except:
                    day_files.append((filepath, day_start))
    
    # If still no files, try recursive search
    if not day_files:
        import glob
        pattern = os.path.join(DATADIR, day_start.strftime('%Y%m%d'), "*.grib2")
        grib_files = glob.glob(pattern)
        for filepath in grib_files:
            day_files.append((filepath, day_start))
    
    if not day_files:
        return {'success': False, 'error': f'No GRIB files found for {day_start.date()}'}
    
    print(f"   📁 Found {len(day_files)} GRIB files for {day_start.date()}")
    
    # Initialize data storage
    wind_data = defaultdict(list)
    solar_data = defaultdict(list)
    timestamps = []
    
    # Process each GRIB file
    for filepath, dt in day_files:
        try:
            # Extract forecast hour from filename to create unique timestamp
            filename = os.path.basename(filepath)
            if 'wrfsubhf' in filename:
                forecast_hour = filename.split('wrfsubhf')[1].split('.')[0]
                # Create unique timestamp that includes forecast hour
                unique_dt = dt.replace(minute=int(forecast_hour))
            else:
                unique_dt = dt
            
            grb = pygrib.open(filepath)
            
            # Read ALL variables in single pass
            all_variables = {}
            
            # Wind variables
            for var_name in ['UWind80', 'VWind80']:
                if var_name in SELECTORS:
                    try:
                        msg = grb.select(name=SELECTORS[var_name])[0]
                        values = msg.values
                        all_variables[var_name] = values
                    except Exception as e:
                        print(f"   ⚠️  Warning: Could not read {var_name}: {e}")
            
            # Solar variables
            for var_name in ['rad', 'vbd', 'vdd', '2tmp', 'UWind10', 'VWind10']:
                if var_name in SELECTORS:
                    try:
                        msg = grb.select(name=SELECTORS[var_name])[0]
                        values = msg.values
                        all_variables[var_name] = values
                    except Exception as e:
                        print(f"   ⚠️  Warning: Could not read {var_name}: {e}")
            
            grb.close()
            
            # Extract data for wind locations
            for i, (idx, point) in enumerate(zip(wind_closest_indices, wind_points.iterrows())):
                point_data = {}
                for var_name in ['UWind80', 'VWind80']:
                    if var_name in all_variables:
                        # Convert 2D index to 1D
                        row, col = np.unravel_index(idx, all_variables[var_name].shape)
                        point_data[var_name] = all_variables[var_name][row, col]
                
                if point_data:
                    wind_data[unique_dt].append({point[1]['pid']: point_data})
            
            # Extract data for solar locations
            for i, (idx, point) in enumerate(zip(solar_closest_indices, solar_points.iterrows())):
                point_data = {}
                for var_name in ['rad', 'vbd', 'vdd', '2tmp', 'UWind10', 'VWind10']:
                    if var_name in all_variables:
                        # Convert 2D index to 1D
                        row, col = np.unravel_index(idx, all_variables[var_name].shape)
                        point_data[var_name] = all_variables[var_name][row, col]
                
                if point_data:
                    solar_data[unique_dt].append({point[1]['pid']: point_data})
            
            timestamps.append(unique_dt)
            
        except Exception as e:
            print(f"   ❌ Error processing {filepath}: {e}")
            continue
    
    # Convert to DataFrames and save
    if wind_data:
        # Create proper DataFrame structure
        wind_records = []
        for timestamp, point_data_list in wind_data.items():
            for point_data in point_data_list:
                for pid, var_data in point_data.items():
                    record = {'timestamp': timestamp, 'pid': pid}
                    for var_name in ['UWind80', 'VWind80']:
                        record[var_name] = var_data.get(var_name, np.nan)
                    wind_records.append(record)
        
        if wind_records:
            wind_df = pd.DataFrame(wind_records)
            # Pivot to get variables as columns and pids as rows
            wind_pivot = wind_df.pivot(index='timestamp', columns='pid', values=['UWind80', 'VWind80'])
            
            day_str = day_start.strftime('%Y%m%d')
            wind_day_dir = os.path.join(wind_output_dir, day_str)
            os.makedirs(wind_day_dir, exist_ok=True)
            
            # Save each variable separately
            for var_name in ['UWind80', 'VWind80']:
                if var_name in SELECTORS:
                    var_df = wind_pivot[var_name]
                    var_file = os.path.join(wind_day_dir, f"{var_name}.parquet")
                    var_df.to_parquet(var_file, compression=compression, engine='pyarrow')
    
    if solar_data:
        # Create proper DataFrame structure
        solar_records = []
        for timestamp, point_data_list in solar_data.items():
            for point_data in point_data_list:
                for pid, var_data in point_data.items():
                    record = {'timestamp': timestamp, 'pid': pid}
                    for var_name in ['rad', 'vbd', 'vdd', '2tmp', 'UWind10', 'VWind10']:
                        record[var_name] = var_data.get(var_name, np.nan)
                    solar_records.append(record)
        
        if solar_records:
            solar_df = pd.DataFrame(solar_records)
            # Pivot to get variables as columns and pids as rows
            solar_pivot = solar_df.pivot(index='timestamp', columns='pid', values=['rad', 'vbd', 'vdd', '2tmp', 'UWind10', 'VWind10'])
            
            day_str = day_start.strftime('%Y%m%d')
            solar_day_dir = os.path.join(solar_output_dir, day_str)
            os.makedirs(solar_day_dir, exist_ok=True)
            
            # Save each variable separately - organize by variable name first
            for var_name in ['rad', 'vbd', 'vdd', '2tmp', 'UWind10', 'VWind10']:
                if var_name in SELECTORS:
                    var_df = solar_pivot[var_name]
                    # Create variable directory
                    var_dir = os.path.join(solar_output_dir, var_name)
                    os.makedirs(var_dir, exist_ok=True)
                    # Save with date in filename
                    day_str = day_start.strftime('%Y%m%d')
                    var_file = os.path.join(var_dir, f"{day_str}.parquet")
                    var_df.to_parquet(var_file, compression=compression, engine='pyarrow')
    
    processing_time = time.time() - day_start_time
    
    return {
        'success': True,
        'processing_time': processing_time,
        'files_processed': len(day_files),
        'wind_records': len(wind_data),
        'solar_records': len(solar_data)
    }


def process_grib_file_for_specific_locations(args):
    """
    Process a single GRIB file for specific locations.
    
    Args:
        args: Tuple of (filepath, closest_indices, SELECTORS, points)
    
    Returns:
        Dictionary with extracted data by variable
    """
    filepath, closest_indices, SELECTORS, points = args
    
    try:
        data = defaultdict(list)
        
        with pygrib.open(filepath) as grbs:
            for grb in grbs:
                for var_name, grib_name in SELECTORS.items():
                    if grb.name == grib_name:
                        # Extract values for closest grid points
                        all_values = grb.values.flatten()
                        location_values = all_values[closest_indices]
                        
                        # Create timestamp
                        timestamp = pd.Timestamp(
                            year=grb.year, month=grb.month, day=grb.day,
                            hour=grb.hour, minute=grb.minute
                        )
                        
                        # Store data
                        data[var_name].append((timestamp, location_values))
        
        return dict(data)
        
    except Exception as e:
        print(f"   ❌ Error processing {filepath}: {e}")
        return {}

def process_single_file_sequential(filepath, dt, wind_closest_indices, solar_closest_indices, wind_points, solar_points, SELECTORS):
    """Process a single GRIB file - sequential version."""
    try:
        # Extract forecast hour from filename to create unique timestamp
        filename = os.path.basename(filepath)
        
        # Determine file type (f00 = top of hour, f01 = three quarters)
        if 'wrfsubhf00' in filename:
            file_type = 'f00'  # Top of hour data (00 minutes)
        elif 'wrfsubhf01' in filename:
            file_type = 'f01'  # Three quarters data (15, 30, 45 minutes)
        else:
            return {
                'success': False,
                'error': f'Unknown file type: {filename}',
                'filepath': filepath
            }
        
        # Use the same extraction function as the full grid function
        # This ensures we get the same time offset detection
        var_selectors = SELECTORS  # Use the actual SELECTORS mapping
        
        # Create dummy chunk indices for the specific locations
        # We'll extract all grid points and then select the ones we need
        grid_metadata = extract_grid_metadata_from_file(filepath)
        grid_lats = grid_metadata['lats']
        grid_lons = grid_metadata['lons']
        
        # Create chunk indices for all grid points (we'll filter later)
        n_lats, n_lons = grid_lats.shape
        all_indices = np.arange(n_lats * n_lons)
        
        # Extract all variables using the same function as full grid
        results = extract_all_variables_from_grib_file(
            filepath, var_selectors, all_indices, grid_lats, grid_lons
        )
        
        # Process the results to extract only the specific locations we need
        all_timestamps = []
        all_wind_results = []
        all_solar_results = []
        
        for var_name, var_data in results.items():
            for timestamp, grid_values in var_data.items():
                # Convert grid_values back to 2D array
                values_2d = grid_values.reshape(n_lats, n_lons)
                
                # Extract data for wind locations
                wind_results = []
                for i, (idx, point) in enumerate(zip(wind_closest_indices, wind_points.iterrows())):
                    point_data = {}
                    for wind_var in ['UWind80', 'VWind80']:
                        if wind_var == var_name and wind_var in SELECTORS:
                            # Convert 2D index to 1D
                            row, col = np.unravel_index(idx, values_2d.shape)
                            point_data[wind_var] = values_2d[row, col]
                    
                    if point_data:
                        wind_results.append({point[1]['pid']: point_data})
                
                # Extract data for solar locations
                solar_results = []
                for i, (idx, point) in enumerate(zip(solar_closest_indices, solar_points.iterrows())):
                    point_data = {}
                    for solar_var in ['rad', 'vbd', 'vdd', '2tmp', 'UWind10', 'VWind10']:
                        if solar_var == var_name and solar_var in SELECTORS:
                            # Convert 2D index to 1D
                            row, col = np.unravel_index(idx, values_2d.shape)
                            point_data[solar_var] = values_2d[row, col]
                    
                    if point_data:
                        solar_results.append({point[1]['pid']: point_data})
                
                # Add results for this timestamp
                if wind_results or solar_results:
                    all_wind_results.extend(wind_results)
                    all_solar_results.extend(solar_results)
                    if timestamp not in all_timestamps:
                        all_timestamps.append(timestamp)
        
        # 🧹 MEMORY CLEANUP: Clear large arrays
        del results
        del grid_lats
        del grid_lons
        
        # Return results for all timestamps found
        if all_timestamps:
            return {
                'success': True,
                'unique_dt': all_timestamps[0],  # Return first timestamp for compatibility
                'wind_results': all_wind_results,
                'solar_results': all_solar_results,
                'filepath': filepath,
                'all_timestamps': all_timestamps,  # Include all timestamps found
                'file_type': file_type  # Include file type for debugging
            }
        else:
            return {
                'success': False,
                'error': 'No valid data found',
                'filepath': filepath
            }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'filepath': filepath
        }

def extract_grid_metadata_from_file(filepath):
    """Extract grid metadata from a single GRIB file."""
    import pygrib
    
    grb = pygrib.open(filepath)
    
    # Try different variable names to find one that exists
    variable_names = ['2 metre temperature', 'Temperature', 'TMP', '2t']
    lats, lons = None, None
    
    for var_name in variable_names:
        try:
            msg = grb.select(name=var_name)[0]
            lats, lons = msg.latlons()
            break
        except:
            continue
    
    if lats is None or lons is None:
        # Fallback: use first message
        try:
            msg = grb[1]  # First message
            lats, lons = msg.latlons()
        except Exception as e:
            grb.close()
            raise ValueError(f"Could not extract grid metadata from {filepath}: {e}")
    
    grb.close()
    
    return {
        'lats': lats,
        'lons': lons,
        'grid_shape': lats.shape
    }

def process_file_worker(args):
    """Worker function for processing files in parallel."""
    filepath, dt, wind_closest_indices, solar_closest_indices, wind_points, solar_points, SELECTORS = args
    return process_single_file_sequential(filepath, dt, wind_closest_indices, solar_closest_indices, wind_points, solar_points, SELECTORS)


# OLD FUNCTION - RENAMED TO AVOID CONFLICT
def extract_all_locations_single_pass_OLD(
    wind_points,
    solar_points,
    START,
    END,
    DATADIR,
    DEFAULT_HOURS_FORECASTED,
    SELECTORS,
    wind_output_dir="./wind_extracted",
    solar_output_dir="./solar_extracted",
    compression="snappy",
    use_parallel=True,
    num_cpu_workers=None,
    num_io_workers=None,
    max_file_groups=None,
    parallel_file_writing=True,
    enable_resume=True,
    day_output_dir_format="flat",
    use_aggressive_settings=True,
):
    """
    🚀 OPTIMIZED: Extract HRRR data for wind and solar locations in a SINGLE PASS.
    
    This function reads each GRIB file only ONCE and extracts ALL variables,
    then separates the data into wind and solar outputs. This is ~50% more efficient
    than the previous approach of reading files twice.
    
    Args:
        wind_points: DataFrame with wind location data ['lat', 'lon', 'pid']
        solar_points: DataFrame with solar location data ['lat', 'lon', 'pid']
        START: Start datetime
        END: End datetime
        DATADIR: Directory containing GRIB files
        DEFAULT_HOURS_FORECASTED: List of forecast hours to process
        SELECTORS: Dictionary mapping variable names to GRIB variable names
        wind_output_dir: Output directory for wind data
        solar_output_dir: Output directory for solar data
        compression: Parquet compression method
        use_parallel: Whether to use parallel processing
        num_cpu_workers: Number of CPU workers (auto-detect if None)
        num_io_workers: Number of I/O workers (auto-detect if None)
        max_file_groups: Maximum file groups to process (auto-detect if None)
        parallel_file_writing: Whether to write files in parallel
        enable_resume: Whether to enable resume functionality
        day_output_dir_format: "daily" for subdirectories, "flat" for single directory
        use_aggressive_settings: Whether to use aggressive settings for high-performance systems
    
    Returns:
        Dictionary with extraction results and metadata
    """
    import datetime
    import gc
    import os
    # Signal handling for graceful shutdown
    import signal
    import time
    from collections import defaultdict

    import numpy as np
    import pandas as pd
    from scipy.spatial import KDTree
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    def extract_grid_metadata(DATADIR, DEFAULT_HOURS_FORECASTED):
        """Extract grid metadata from GRIB files."""
        # Find a sample GRIB file to extract grid info
        sample_files = []
        
        # Try multiple date ranges to find files
        test_dates = [
            pd.date_range(START, START + datetime.timedelta(days=1), freq='h'),
            pd.date_range(START + datetime.timedelta(days=1), START + datetime.timedelta(days=2), freq='h'),
            pd.date_range(START + datetime.timedelta(days=2), START + datetime.timedelta(days=3), freq='h'),
        ]
        
        for date_range in test_dates:
            for dt in date_range:
                for hour in DEFAULT_HOURS_FORECASTED:
                    # Convert hour to integer for formatting
                    hour_int = int(hour)
                    filename = f"hrrr.t{dt.hour:02d}z.wrfsubhf{hour_int:02d}.grib2"
                    filepath = os.path.join(DATADIR, dt.strftime('%Y%m%d'), filename)
                    if os.path.exists(filepath):
                        sample_files.append(filepath)
                        print(f"✅ Found sample GRIB file: {filepath}")
                        break
                if sample_files:
                    break
            if sample_files:
                break
        
        # If still no files found, try alternative naming patterns
        if not sample_files:
            print(f"🔍 Searching for GRIB files in {DATADIR}...")
            
            # List all subdirectories
            if os.path.exists(DATADIR):
                subdirs = [d for d in os.listdir(DATADIR) if os.path.isdir(os.path.join(DATADIR, d))]
                print(f"📁 Found subdirectories: {subdirs[:10]}...")  # Show first 10
                
                # Look for any .grib2 files
                for subdir in subdirs[:5]:  # Check first 5 subdirectories
                    subdir_path = os.path.join(DATADIR, subdir)
                    grib_files = [f for f in os.listdir(subdir_path) if f.endswith('.grib2')]
                    if grib_files:
                        sample_file = os.path.join(subdir_path, grib_files[0])
                        sample_files.append(sample_file)
                        print(f"✅ Found GRIB file: {sample_file}")
                        break
        
        if not sample_files:
            # Try to find any .grib2 file recursively
            import glob
            grib_files = glob.glob(os.path.join(DATADIR, "**/*.grib2"), recursive=True)
            if grib_files:
                sample_files.append(grib_files[0])
                print(f"✅ Found GRIB file via glob: {grib_files[0]}")
        
        if not sample_files:
            raise FileNotFoundError(f"No GRIB files found in {DATADIR}. Please check the path and file naming convention.")
        
        # Extract grid info from first file
        import pygrib
        grb = pygrib.open(sample_files[0])
        
        # Try different variable names to find one that exists
        variable_names = ['2 metre temperature', 'Temperature', 'TMP', '2t']
        lats, lons = None, None
        
        for var_name in variable_names:
            try:
                msg = grb.select(name=var_name)[0]
                lats, lons = msg.latlons()
                print(f"✅ Successfully extracted grid metadata using variable: {var_name}")
                break
            except:
                continue
        
        if lats is None or lons is None:
            # Fallback: use first message
            try:
                msg = grb[1]  # First message
                lats, lons = msg.latlons()
                print(f"✅ Successfully extracted grid metadata using first message")
            except Exception as e:
                grb.close()
                raise ValueError(f"Could not extract grid metadata from {sample_files[0]}: {e}")
        
        grb.close()
        
        return {
            'lats': lats,
            'lons': lons,
            'grid_shape': lats.shape
        }
    
    def find_closest_grid_points(points, grid_lats, grid_lons):
        """Find closest grid points for given lat/lon coordinates."""
        # Handle case where grid_lats/grid_lons might be dictionaries
        if isinstance(grid_lats, dict):
            grid_lats = grid_lats['lats']
        if isinstance(grid_lons, dict):
            grid_lons = grid_lons['lons']
        
        # Ensure we have numpy arrays
        if not isinstance(grid_lats, np.ndarray):
            grid_lats = np.array(grid_lats)
        if not isinstance(grid_lons, np.ndarray):
            grid_lons = np.array(grid_lons)
        
        # Flatten grid coordinates
        grid_points = np.column_stack([grid_lats.flatten(), grid_lons.flatten()])
        
        # Create KDTree for efficient nearest neighbor search
        tree = KDTree(grid_points)
        
        # Find closest points
        query_points = points[['lat', 'lon']].values
        distances, indices = tree.query(query_points)
        
        return indices
    
    def process_day_single_pass(
        day_start, day_end, DATADIR, DEFAULT_HOURS_FORECASTED, SELECTORS,
        wind_closest_indices, solar_closest_indices, wind_points, solar_points,
        wind_output_dir, solar_output_dir, compression, use_parallel,
        num_cpu_workers, num_io_workers, max_file_groups, parallel_file_writing
    ):
        """Process a single day with single-pass GRIB reading."""
        import multiprocessing as mp
        from multiprocessing import Pool, cpu_count

        import pygrib
        
        day_start_time = time.time()
        
        # Find GRIB files for this day
        day_files = []
        
        # Try the expected naming pattern first - ONLY for the specific hours we want
        # Instead of all hours in the day, only look for the hours specified in START and END
        start_hour = day_start.hour
        end_hour = day_end.hour
        start_hour = 0
        end_hour = 1
        
        # Create a list of hours we actually want to process
        target_hours = list(range(start_hour, end_hour))
        print(f"   🎯 Looking for files for hours: {target_hours}")
        
        for hour in target_hours:
            for forecast_hour in DEFAULT_HOURS_FORECASTED:
                # Convert forecast hour to integer for formatting
                forecast_int = int(forecast_hour)
                filename = f"hrrr.t{hour:02d}z.wrfsubhf{forecast_int:02d}.grib2"
                filepath = os.path.join(DATADIR, day_start.strftime('%Y%m%d'), filename)
                if os.path.exists(filepath):
                    dt = day_start.replace(hour=hour)
                    day_files.append((filepath, dt))
                    print(f"   ✅ Found: {filename}")
                else:
                    print(f"   ❌ Missing: {filename}")
        
        # If no files found with expected pattern, try alternative patterns
        if not day_files:
            day_str = day_start.strftime('%Y%m%d')
            day_dir = os.path.join(DATADIR, day_str)
            
            if os.path.exists(day_dir):
                # Look for any .grib2 files in the day directory
                grib_files = [f for f in os.listdir(day_dir) if f.endswith('.grib2')]
                for filename in grib_files:
                    filepath = os.path.join(day_dir, filename)
                    # Try to extract datetime from filename
                    try:
                        # Extract hour from filename (e.g., "hrrr.t00z.wrfprsf00.grib2" -> hour 0)
                        if 't' in filename and 'z' in filename:
                            hour_part = filename.split('t')[1].split('z')[0]
                            dt = day_start.replace(hour=int(hour_part))
                        else:
                            dt = day_start  # Default to start of day
                        day_files.append((filepath, dt))
                    except:
                        day_files.append((filepath, day_start))
        
        # If still no files, try recursive search
        if not day_files:
            import glob
            pattern = os.path.join(DATADIR, day_start.strftime('%Y%m%d'), "*.grib2")
            grib_files = glob.glob(pattern)
            for filepath in grib_files:
                day_files.append((filepath, day_start))
        
        if not day_files:
            return {'success': False, 'error': f'No GRIB files found for {day_start.date()}'}
        
        print(f"   📁 Found {len(day_files)} GRIB files for {day_start.date()}")
        
        # Initialize data storage
        wind_data = defaultdict(list)
        solar_data = defaultdict(list)
        timestamps = []
        
        # Process files in parallel or sequentially
        if use_parallel and len(day_files) > 1:
            print(f"   🚀 Processing {len(day_files)} files in parallel with {num_cpu_workers} workers")
            
            # Use multiprocessing to process files in parallel
            # Prepare arguments for each file
            worker_args = [(filepath, dt, wind_closest_indices, solar_closest_indices, wind_points, solar_points, SELECTORS) 
                          for filepath, dt in day_files]
            
            with Pool(processes=num_cpu_workers) as pool:
                results = list(tqdm(
                    pool.imap(process_file_worker, worker_args),
                    total=len(day_files),
                    desc="Processing files"
                ))
            
            # Collect results
            for result in results:
                if result['success']:
                    # Handle multiple timestamps from each file
                    if 'all_timestamps' in result:
                        # Multiple timestamps found (15-minute data)
                        # For now, just use the first timestamp to avoid duplicates
                        # TODO: Implement proper 15-minute data handling
                        unique_dt = result['all_timestamps'][0]
                        wind_data[unique_dt].extend(result['wind_results'])
                        solar_data[unique_dt].extend(result['solar_results'])
                        if unique_dt not in timestamps:  # Avoid duplicates
                            timestamps.append(unique_dt)
                    else:
                        # Single timestamp (backward compatibility)
                        unique_dt = result['unique_dt']
                        wind_data[unique_dt].extend(result['wind_results'])
                        solar_data[unique_dt].extend(result['solar_results'])
                        if unique_dt not in timestamps:  # Avoid duplicates
                            timestamps.append(unique_dt)
                else:
                    print(f"   ❌ Error processing {result.get('filepath', 'unknown')}: {result.get('error', 'unknown error')}")
        else:
            print(f"   📊 Processing {len(day_files)} files sequentially")
            # Fallback sequential processing
            for filepath, dt in day_files:
                result = process_single_file_sequential(filepath, dt, wind_closest_indices, solar_closest_indices, wind_points, solar_points, SELECTORS)
                if result['success']:
                    # Handle multiple timestamps from each file
                    if 'all_timestamps' in result:
                        # Multiple timestamps found (15-minute data)
                        # For now, just use the first timestamp to avoid duplicates
                        # TODO: Implement proper 15-minute data handling
                        unique_dt = result['all_timestamps'][0]
                        wind_data[unique_dt].extend(result['wind_results'])
                        solar_data[unique_dt].extend(result['solar_results'])
                        if unique_dt not in timestamps:  # Avoid duplicates
                            timestamps.append(unique_dt)
                    else:
                        # Single timestamp (backward compatibility)
                        unique_dt = result['unique_dt']
                        wind_data[unique_dt].extend(result['wind_results'])
                        solar_data[unique_dt].extend(result['solar_results'])
                        if unique_dt not in timestamps:  # Avoid duplicates
                            timestamps.append(unique_dt)
                else:
                    print(f"   ❌ Error processing {filepath}: {result['error']}")
                    continue
        
        # Convert to DataFrames and save
        if wind_data:
            # Create proper DataFrame structure with duplicate removal
            wind_records = []
            seen_wind_keys = set()  # Track (timestamp, pid) combinations
            
            for timestamp, point_data_list in wind_data.items():
                for point_data in point_data_list:
                    for pid, var_data in point_data.items():
                        # Check for duplicates
                        key = (timestamp, pid)
                        if key in seen_wind_keys:
                            continue  # Skip duplicate
                        seen_wind_keys.add(key)
                        
                        record = {'timestamp': timestamp, 'pid': pid}
                        for var_name in ['UWind80', 'VWind80']:
                            record[var_name] = var_data.get(var_name, np.nan)
                        wind_records.append(record)
            
            if wind_records:
                wind_df = pd.DataFrame(wind_records)
                # Pivot to get variables as columns and pids as rows
                wind_pivot = wind_df.pivot(index='timestamp', columns='pid', values=['UWind80', 'VWind80'])
                
                # Save each variable separately - organize by variable name first
                for var_name in ['UWind80', 'VWind80']:
                    if var_name in SELECTORS:
                        var_df = wind_pivot[var_name]
                        # Create variable directory
                        var_dir = os.path.join(wind_output_dir, var_name)
                        os.makedirs(var_dir, exist_ok=True)
                        # Save with date in filename
                        day_str = day_start.strftime('%Y%m%d')
                        var_file = os.path.join(var_dir, f"{day_str}.parquet")
                        var_df.to_parquet(var_file, compression=compression, engine='pyarrow')
        
        if solar_data:
            # Create proper DataFrame structure with duplicate removal
            solar_records = []
            seen_solar_keys = set()  # Track (timestamp, pid) combinations
            
            for timestamp, point_data_list in solar_data.items():
                for point_data in point_data_list:
                    for pid, var_data in point_data.items():
                        # Check for duplicates
                        key = (timestamp, pid)
                        if key in seen_solar_keys:
                            continue  # Skip duplicate
                        seen_solar_keys.add(key)
                        
                        record = {'timestamp': timestamp, 'pid': pid}
                        for var_name in ['rad', 'vbd', 'vdd', '2tmp', 'UWind10', 'VWind10']:
                            record[var_name] = var_data.get(var_name, np.nan)
                        solar_records.append(record)
            
            if solar_records:
                solar_df = pd.DataFrame(solar_records)
                # Pivot to get variables as columns and pids as rows
                solar_pivot = solar_df.pivot(index='timestamp', columns='pid', values=['rad', 'vbd', 'vdd', '2tmp', 'UWind10', 'VWind10'])
                
                # Save each variable separately - organize by variable name first
                for var_name in ['rad', 'vbd', 'vdd', '2tmp', 'UWind10', 'VWind10']:
                    if var_name in SELECTORS:
                        var_df = solar_pivot[var_name]
                        # Create variable directory
                        var_dir = os.path.join(solar_output_dir, var_name)
                        os.makedirs(var_dir, exist_ok=True)
                        # Save with date in filename
                        day_str = day_start.strftime('%Y%m%d')
                        var_file = os.path.join(var_dir, f"{day_str}.parquet")
                        var_df.to_parquet(var_file, compression=compression, engine='pyarrow')
        
        processing_time = time.time() - day_start_time
        
        # 🧹 MEMORY CLEANUP: Clear large data structures
        wind_data.clear()
        solar_data.clear()
        timestamps.clear()
        
        # Force garbage collection to free memory
        import gc
        gc.collect()
        
        # Small delay to allow system to stabilize
        time.sleep(0.5)
        
        return {
            'success': True,
            'processing_time': processing_time,
            'files_processed': len(day_files),
            'wind_records': len(wind_data),
            'solar_records': len(solar_data)
        }
    
    def discover_grib_variables(filepath):
        """Discover what variables are available in a GRIB file."""
        import pygrib
        try:
            grb = pygrib.open(filepath)
            variables = []
            for msg in grb:
                variables.append(msg.name)
            grb.close()
            return list(set(variables))  # Remove duplicates
        except Exception as e:
            print(f"   ⚠️  Could not read variables from {filepath}: {e}")
            return []
    
    def find_matching_variable(desired_name, available_variables):
        """Find the best matching variable name from available variables."""
        # Try exact match first
        if desired_name in available_variables:
            return desired_name
        
        # Try partial matches
        for var in available_variables:
            if desired_name.lower() in var.lower():
                return var
        
        # Try common variations
        variations = {
            "U component of wind at 80 m above ground": ["U component of wind", "U wind", "U-component"],
            "V component of wind at 80 m above ground": ["V component of wind", "V wind", "V-component"],
            "U component of wind at 10 m above ground": ["U component of wind", "U wind", "U-component"],
            "V component of wind at 10 m above ground": ["V component of wind", "V wind", "V-component"],
            "2 metre temperature": ["2 metre temperature", "2m temperature", "Temperature"],
            "Downward solar radiation flux": ["Downward solar radiation flux", "Solar radiation", "Radiation"]
        }
        
        if desired_name in variations:
            for variation in variations[desired_name]:
                for var in available_variables:
                    if variation.lower() in var.lower():
                        return var
        
        return None

    # 🚀 MAIN EXECUTION LOGIC
    print("🚀 OPTIMIZED SINGLE-PASS EXTRACTION")
    print("=" * 50)
    print("📊 Using single-pass GRIB reading for maximum efficiency")
    print("   - Each GRIB file read only ONCE")
    print("   - All variables extracted simultaneously")
    print("   - ~50% faster than previous approach")
    print()

    # Get optimized settings
    if use_aggressive_settings:
        settings = get_optimized_settings_for_high_performance_system()
        if num_cpu_workers is None:
            num_cpu_workers = settings['num_cpu_workers']
        if num_io_workers is None:
            num_io_workers = settings['num_io_workers']
        if max_file_groups is None:
            max_file_groups = settings['max_file_groups']
    
    print(f"🔧 Detected system: {os.cpu_count()} CPUs, {get_available_memory():.1f} GB RAM")
    print(f"⚙️  Using settings:")
    print(f"   CPU workers: {num_cpu_workers}")
    print(f"   I/O workers: {num_io_workers}")
    print(f"   Max file groups: {max_file_groups}")
    print()

    # Extract grid metadata once
    print("📊 Extracting grid metadata (once for all days)...")
    grid_start_time = time.time()
    
    grid_metadata = extract_grid_metadata(DATADIR, DEFAULT_HOURS_FORECASTED)
    grid_lats = grid_metadata['lats']
    grid_lons = grid_metadata['lons']
    
    print(f"Grid dimensions: {grid_lats.shape[0]} x {grid_lats.shape[1]} = {grid_lats.size:,} total points")
    print(f"Grid metadata extracted in {time.time() - grid_start_time:.2f}s")
    print()

    # Calculate closest grid points for target locations
    print("📍 Calculating closest grid points for target locations...")
    closest_start_time = time.time()
    
    # Find closest points for wind and solar locations separately
    wind_closest_indices = find_closest_grid_points(wind_points, grid_lats, grid_lons)
    solar_closest_indices = find_closest_grid_points(solar_points, grid_lats, grid_lons)
    
    __import__('ipdb').set_trace()
    print(f"✅ Found closest points for {len(wind_points)} wind locations and {len(solar_points)} solar locations in {time.time() - closest_start_time:.2f}s")
    print(f"📍 Wind locations: {len(wind_points)}")
    print(f"📍 Solar locations: {len(solar_points)}")
    print(f"🎯 Wind closest grid points: {len(wind_closest_indices)}")
    print(f"🎯 Solar closest grid points: {len(solar_closest_indices)}")
    print()

    # Generate date range
    date_range = pd.date_range(start=START, end=END, freq='D')
    total_days = len(date_range)
    
    print(f"📅 Processing {total_days} days from {START.date()} to {END.date()}")
    print()

    # Process each day
    total_start_time = time.time()
    successful_days = []
    failed_days = []
    
    for day_idx, current_date in enumerate(date_range, 1):
        day_start_time = time.time()
        
        # Check for shutdown request
        if check_shutdown_requested():
            print("🛑 Shutdown requested. Saving progress...")
            break
        
        print(f"📅 Processing day {day_idx}/{total_days}: {current_date.date()}")
        print(f"   Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Variables: {list(SELECTORS.keys())}")
        
        # Set day-specific start and end times
        day_start = datetime.datetime.combine(current_date.date(), START.time())
        day_end = datetime.datetime.combine(current_date.date(), START.time()) + datetime.timedelta(days=1)
        
        # __import__('ipdb').set_trace()
        try:
            # Process this day with single-pass extraction
            day_result = process_day_single_pass(
                day_start=day_start,
                day_end=day_end,
                DATADIR=DATADIR,
                DEFAULT_HOURS_FORECASTED=DEFAULT_HOURS_FORECASTED,
                SELECTORS=SELECTORS,
                wind_closest_indices=wind_closest_indices,
                solar_closest_indices=solar_closest_indices,
                wind_points=wind_points,
                solar_points=solar_points,
                wind_output_dir=wind_output_dir,
                solar_output_dir=solar_output_dir,
                compression=compression,
                use_parallel=use_parallel,
                num_cpu_workers=num_cpu_workers,
                num_io_workers=num_io_workers,
                max_file_groups=max_file_groups,
                parallel_file_writing=parallel_file_writing,
            )
            
            day_time = time.time() - day_start_time
            
            if day_result and day_result.get("success"):
                successful_days.append(current_date.date())
                print(f"   ✅ Completed in {day_time:.1f}s ({day_time/60:.1f} minutes)")
                print(f"   📁 Files written to: {wind_output_dir} and {solar_output_dir}")
            else:
                failed_days.append(current_date.date())
                print(f"   ❌ Failed after {day_time:.1f}s ({day_time/60:.1f} minutes)")
                
        except Exception as e:
            day_time = time.time() - day_start_time
            failed_days.append(current_date.date())
            print(f"   ❌ Error after {day_time:.1f}s: {e}")
            
            # Continue with next day instead of stopping
            print(f"   Continuing with next day...")
            continue
        
        # Progress update
        total_time_so_far = time.time() - total_start_time
        avg_time_per_day = total_time_so_far / day_idx
        remaining_days = total_days - day_idx
        estimated_remaining_time = remaining_days * avg_time_per_day
        
        print(f"   📊 Progress: {day_idx}/{total_days} days")
        print(f"   ✅ Success rate: {len(successful_days)}/{day_idx} ({len(successful_days)/day_idx*100:.1f}%)")
        print(f"   ⏱️  Estimated time remaining: {estimated_remaining_time/3600:.1f} hours")
        print(f"   🗓️  Next day: {date_range[day_idx].date() if day_idx < total_days else 'COMPLETE'}")
        print()
        
        # 🧹 MEMORY CLEANUP: Force garbage collection between days
        import gc
        gc.collect()
        
        # Small delay to allow system to stabilize
        time.sleep(0.5)
    
    # Final summary
    total_time = time.time() - total_start_time
    
    print("✅ Single-pass extraction completed!")
    print()
    
    return {
        "status": "completed",
        "total_days": total_days,
        "successful_days": len(successful_days),
        "failed_days": len(failed_days),
        "processing_time_seconds": total_time,
        "wind_locations": len(wind_points),
        "solar_locations": len(solar_points),
        "wind_output_dir": wind_output_dir,
        "solar_output_dir": solar_output_dir
    }

def extract_all_locations_single_pass(
    wind_points,
    solar_points,
    START,
    END,
    DATADIR,
    DEFAULT_HOURS_FORECASTED,
    SELECTORS,
    wind_output_dir="./wind_extracted",
    solar_output_dir="./solar_extracted",
    compression="snappy",
    use_parallel=True,
    num_cpu_workers=None,
    num_io_workers=None,
    max_file_groups=None,
    parallel_file_writing=True,
    enable_resume=True,
    day_output_dir_format="flat",
    use_aggressive_settings=True,
):
    """
    🚀 OPTIMIZED: Extract HRRR data for wind and solar locations using the working full grid pattern.
    
    This function follows the same pattern as the working full grid extraction but adapted for specific locations.
    It uses the proven extract_full_grid_optimized_with_preloaded_data function internally.
    
    Args:
        wind_points: DataFrame with wind location data ['lat', 'lon', 'pid']
        solar_points: DataFrame with solar location data ['lat', 'lon', 'pid']
        START: Start datetime
        END: End datetime
        DATADIR: Directory containing GRIB files
        DEFAULT_HOURS_FORECASTED: List of forecast hours to process
        SELECTORS: Dictionary mapping variable names to GRIB variable names
        wind_output_dir: Output directory for wind data
        solar_output_dir: Output directory for solar data
        compression: Parquet compression method
        use_parallel: Whether to use parallel processing
        num_cpu_workers: Number of CPU workers (auto-detect if None)
        num_io_workers: Number of I/O workers (auto-detect if None)
        max_file_groups: Maximum file groups to process (auto-detect if None)
        parallel_file_writing: Whether to write files in parallel
        enable_resume: Whether to enable resume functionality
        day_output_dir_format: "daily" for subdirectories, "flat" for single directory
        use_aggressive_settings: Whether to use aggressive settings for high-performance systems
    
    Returns:
        Dictionary with extraction results and metadata
    """
    import datetime
    import gc
    import os
    # Signal handling for graceful shutdown
    import signal
    import time
    from collections import defaultdict

    import numpy as np
    import pandas as pd
    from scipy.spatial import KDTree
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("🚀 OPTIMIZED SINGLE-PASS EXTRACTION")
    print("=" * 50)
    print("📊 Using single-pass GRIB reading for maximum efficiency")
    print("   - Each GRIB file read only ONCE")
    print("   - All variables extracted simultaneously")
    print("   - ~50% faster than previous approach")
    print()
    
    # Auto-detect optimal settings based on system capabilities
    if use_aggressive_settings:
        settings = get_aggressive_parallel_settings()
        
        # Override with user-provided values if specified
        if num_cpu_workers is None:
            num_cpu_workers = settings['num_cpu_workers']
        if num_io_workers is None:
            num_io_workers = settings['num_io_workers']
        if max_file_groups is None:
            max_file_groups = settings['max_file_groups']
        
        print(f"🔧 Detected system: {os.cpu_count()} CPUs, {get_available_memory():.1f} GB RAM")
        print(f"⚙️  Using settings:")
        print(f"   CPU workers: {num_cpu_workers}")
        print(f"   I/O workers: {num_io_workers}")
        print(f"   Max file groups: {max_file_groups}")
        print()
    else:
        # Use conservative defaults if aggressive settings disabled
        if num_cpu_workers is None:
            num_cpu_workers = 8
        if num_io_workers is None:
            num_io_workers = 4
        if max_file_groups is None:
            max_file_groups = 5000
    
    # Create output directories
    os.makedirs(wind_output_dir, exist_ok=True)
    os.makedirs(solar_output_dir, exist_ok=True)
    
    # 🚀 OPTIMIZATION: Extract grid metadata ONCE at the beginning
    print("📊 Extracting grid metadata (once for all days)...")
    grid_start = time.time()
    wind_data_lat_long = get_wind_data_lat_long(START, DATADIR)
    grid_lats, grid_lons = wind_data_lat_long[0], wind_data_lat_long[1]
    
    n_lats, n_lons = grid_lats.shape
    total_grid_points = n_lats * n_lons
    
    print(f"Grid dimensions: {n_lats} x {n_lons} = {total_grid_points:,} total points")
    grid_time = time.time() - grid_start
    print(f"Grid metadata extracted in {grid_time:.2f}s")
    
    # 🚀 OPTIMIZATION: Calculate closest grid points for target locations
    print("📍 Calculating closest grid points for target locations...")
    closest_start = time.time()
    
    # Find closest grid points for wind locations
    wind_closest_indices = find_closest_grid_points(wind_points, grid_lats, grid_lons)
    
    # Find closest grid points for solar locations  
    solar_closest_indices = find_closest_grid_points(solar_points, grid_lats, grid_lons)
    
    closest_time = time.time() - closest_start
    print(f"✅ Found closest points for {len(wind_points)} wind locations and {len(solar_points)} solar locations in {closest_time:.2f}s")
    print(f"📍 Wind locations: {len(wind_points)}")
    print(f"📍 Solar locations: {len(solar_points)}")
    print(f"🎯 Wind closest grid points: {len(wind_closest_indices)}")
    print(f"🎯 Solar closest grid points: {len(solar_closest_indices)}")
    print()
    
    # Generate list of hours to process (for 2-hour test)
    if (END - START).total_seconds() <= 24 * 3600:  # If less than 24 hours, process by hour
        # For 2-hour test, process the exact time range
        print(f"📅 Processing 2-hour period from {START} to {END}")
        print()
        
        # Process the specific 2-hour period
        successful_days = []
        failed_days = []
        total_start_time = time.time()
        
        day_start_time = time.time()
        
        # Check for shutdown request
        if check_shutdown_requested():
            print("🛑 Shutdown requested. Saving progress...")
            return {
                "status": "interrupted",
                "total_days": 1,
                "successful_days": len(successful_days),
                "failed_days": len(failed_days),
                "processing_time_seconds": time.time() - total_start_time,
                "wind_locations": len(wind_points),
                "solar_locations": len(solar_points),
                "wind_output_dir": wind_output_dir,
                "solar_output_dir": solar_output_dir
            }
        
        print(f"📅 Processing 2-hour period: {START} to {END}")
        print(f"   Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Variables: {list(SELECTORS.keys())}")
        
        # Use the exact START and END times
        day_start = START
        day_end = END
        
        # Determine output directory for this period
        if day_output_dir_format == "daily":
            # Each period gets its own subdirectory
            period_wind_output_dir = os.path.join(wind_output_dir, START.strftime("%Y%m%d_%H%M"))
            period_solar_output_dir = os.path.join(solar_output_dir, START.strftime("%Y%m%d_%H%M"))
        else:
            # Flat structure - all periods in same directory
            period_wind_output_dir = wind_output_dir
            period_solar_output_dir = solar_output_dir
        
        print(f"   Output: {period_wind_output_dir} and {period_solar_output_dir}")
        
        try:
            # 🚀 OPTIMIZATION: Use the working full grid function but extract only specific points
            # First, extract full grid data for this 2-hour period
            temp_output_dir = f"/tmp/hrrr_temp_{START.strftime('%Y%m%d_%H%M')}"
            
            day_result = extract_full_grid_optimized_with_preloaded_data(
                START=day_start,
                END=day_end,
                DATADIR=DATADIR,
                DEFAULT_HOURS_FORECASTED=DEFAULT_HOURS_FORECASTED,
                SELECTORS=SELECTORS,
                output_dir=temp_output_dir,
                chunk_size=50000,  # Use smaller chunks for efficiency
                compression=compression,
                use_parallel=use_parallel,
                num_cpu_workers=num_cpu_workers,
                num_io_workers=num_io_workers,
                max_file_groups=max_file_groups,
                create_individual_mappings=False,
                parallel_file_writing=parallel_file_writing,
                enable_resume=False,  # Don't use resume within period processing
                # 🚀 OPTIMIZATION: Pass pre-loaded data
                grid_lats=grid_lats,
                grid_lons=grid_lons,
                global_mapping=None,  # Let the function create its own mapping
            )
            
            if day_result and day_result.get("status") == "completed":
                # Now extract the specific points from the full grid data
                print("   📊 Extracting specific location data from full grid...")
                
                # Extract wind data for specific points
                wind_data = extract_points_from_full_grid(
                    points=wind_points,
                    START=day_start,
                    END=day_end,
                    grid_data_dir=temp_output_dir
                )
                
                # Extract solar data for specific points
                solar_data = extract_points_from_full_grid(
                    points=solar_points,
                    START=day_start,
                    END=day_end,
                    grid_data_dir=temp_output_dir
                )
                
                # Save the extracted data
                if wind_data is not None and not wind_data.empty:
                    os.makedirs(period_wind_output_dir, exist_ok=True)
                    wind_filename = f"wind_data_{START.strftime('%Y%m%d_%H%M')}_to_{END.strftime('%H%M')}.parquet"
                    wind_path = os.path.join(period_wind_output_dir, wind_filename)
                    wind_data.to_parquet(wind_path, compression=compression)
                    print(f"   ✅ Saved wind data: {wind_path}")
                
                if solar_data is not None and not solar_data.empty:
                    os.makedirs(period_solar_output_dir, exist_ok=True)
                    solar_filename = f"solar_data_{START.strftime('%Y%m%d_%H%M')}_to_{END.strftime('%H%M')}.parquet"
                    solar_path = os.path.join(period_solar_output_dir, solar_filename)
                    solar_data.to_parquet(solar_path, compression=compression)
                    print(f"   ✅ Saved solar data: {solar_path}")
                
                # Clean up temporary files
                import shutil
                if os.path.exists(temp_output_dir):
                    shutil.rmtree(temp_output_dir)
                
                successful_days.append(START.date())
                day_time = time.time() - day_start_time
                print(f"   ✅ Completed in {day_time:.1f}s ({day_time/60:.1f} minutes)")
                print(f"   📁 Files written to: {period_wind_output_dir} and {period_solar_output_dir}")
            else:
                failed_days.append(START.date())
                day_time = time.time() - day_start_time
                print(f"   ❌ Failed after {day_time:.1f}s ({day_time/60:.1f} minutes)")
                
        except Exception as e:
            day_time = time.time() - day_start_time
            failed_days.append(START.date())
            print(f"   ❌ Error after {day_time:.1f}s: {e}")
            
            # Continue instead of stopping
            print(f"   Continuing...")
        
        # Progress update for single period
        total_time_so_far = time.time() - total_start_time
        
        print(f"   📊 Progress: 1/1 period")
        print(f"   ✅ Success rate: {len(successful_days)}/1 ({len(successful_days)*100:.1f}%)")
        print(f"   ⏱️  Total time: {total_time_so_far:.1f}s ({total_time_so_far/60:.1f} minutes)")
        print()
        
        # Force garbage collection
        gc.collect()
        
        # Small delay to allow system to stabilize
        time.sleep(1)
    
    else:
        # For longer periods (more than 24 hours), process by days
        # Generate list of days to process
        date_range = pd.date_range(start=START.date(), end=END.date(), freq="1D")
        print(f"📅 Processing {len(date_range)} days from {START.date()} to {END.date()}")
        print()
        
        # Process each day using the working full grid function
        successful_days = []
        failed_days = []
        total_start_time = time.time()
        
        for day_idx, current_date in enumerate(date_range, 1):
            day_start_time = time.time()
            
            # Check for shutdown request
            if check_shutdown_requested():
                print("🛑 Shutdown requested. Saving progress...")
                break
            
            print(f"📅 Processing day {day_idx}/{len(date_range)}: {current_date.date()}")
            print(f"   Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Variables: {list(SELECTORS.keys())}")
            
            # Set day-specific start and end times
            day_start = datetime.datetime.combine(current_date.date(), START.time())
            day_end = datetime.datetime.combine(current_date.date(), START.time()) + datetime.timedelta(days=1)
            
            # Determine output directory for this day
            if day_output_dir_format == "daily":
                # Each day gets its own subdirectory
                day_wind_output_dir = os.path.join(wind_output_dir, current_date.strftime("%Y%m%d"))
                day_solar_output_dir = os.path.join(solar_output_dir, current_date.strftime("%Y%m%d"))
            else:
                # Flat structure - all days in same directory
                day_wind_output_dir = wind_output_dir
                day_solar_output_dir = solar_output_dir
            
            print(f"   Output: {day_wind_output_dir} and {day_solar_output_dir}")
            
            try:
                # 🚀 OPTIMIZATION: Use the working full grid function but extract only specific points
                # First, extract full grid data for this day
                temp_output_dir = f"/tmp/hrrr_temp_{current_date.strftime('%Y%m%d')}"
                
                day_result = extract_full_grid_optimized_with_preloaded_data(
                    START=day_start,
                    END=day_end,
                    DATADIR=DATADIR,
                    DEFAULT_HOURS_FORECASTED=DEFAULT_HOURS_FORECASTED,
                    SELECTORS=SELECTORS,
                    output_dir=temp_output_dir,
                    chunk_size=50000,  # Use smaller chunks for efficiency
                    compression=compression,
                    use_parallel=use_parallel,
                    num_cpu_workers=num_cpu_workers,
                    num_io_workers=num_io_workers,
                    max_file_groups=max_file_groups,
                    create_individual_mappings=False,
                    parallel_file_writing=parallel_file_writing,
                    enable_resume=False,  # Don't use resume within day processing
                    # 🚀 OPTIMIZATION: Pass pre-loaded data
                    grid_lats=grid_lats,
                    grid_lons=grid_lons,
                    global_mapping=None,  # Let the function create its own mapping
                )
                
                if day_result and day_result.get("status") == "completed":
                    # Now extract the specific points from the full grid data
                    print("   📊 Extracting specific location data from full grid...")
                    
                    # Extract wind data for specific points
                    wind_data = extract_points_from_full_grid(
                        points=wind_points,
                        START=day_start,
                        END=day_end,
                        grid_data_dir=temp_output_dir
                    )
                    
                    # Extract solar data for specific points
                    solar_data = extract_points_from_full_grid(
                        points=solar_points,
                        START=day_start,
                        END=day_end,
                        grid_data_dir=temp_output_dir
                    )
                    
                    # Save the extracted data
                    if wind_data is not None and not wind_data.empty:
                        os.makedirs(day_wind_output_dir, exist_ok=True)
                        wind_filename = f"wind_data_{current_date.strftime('%Y%m%d')}.parquet"
                        wind_path = os.path.join(day_wind_output_dir, wind_filename)
                        wind_data.to_parquet(wind_path, compression=compression)
                        print(f"   ✅ Saved wind data: {wind_path}")
                    
                    if solar_data is not None and not solar_data.empty:
                        os.makedirs(day_solar_output_dir, exist_ok=True)
                        solar_filename = f"solar_data_{current_date.strftime('%Y%m%d')}.parquet"
                        solar_path = os.path.join(day_solar_output_dir, solar_filename)
                        solar_data.to_parquet(solar_path, compression=compression)
                        print(f"   ✅ Saved solar data: {solar_path}")
                    
                    # Clean up temporary files
                    import shutil
                    if os.path.exists(temp_output_dir):
                        shutil.rmtree(temp_output_dir)
                    
                    successful_days.append(current_date.date())
                    day_time = time.time() - day_start_time
                    print(f"   ✅ Completed in {day_time:.1f}s ({day_time/60:.1f} minutes)")
                    print(f"   📁 Files written to: {day_wind_output_dir} and {day_solar_output_dir}")
                else:
                    failed_days.append(current_date.date())
                    day_time = time.time() - day_start_time
                    print(f"   ❌ Failed after {day_time:.1f}s ({day_time/60:.1f} minutes)")
                    
            except Exception as e:
                day_time = time.time() - day_start_time
                failed_days.append(current_date.date())
                print(f"   ❌ Error after {day_time:.1f}s: {e}")
                
                # Continue with next day instead of stopping
                print(f"   Continuing with next day...")
                continue
            
            # Progress update
            total_time_so_far = time.time() - total_start_time
            avg_time_per_day = total_time_so_far / day_idx
            remaining_days = len(date_range) - day_idx
            estimated_remaining_time = remaining_days * avg_time_per_day
            
            print(f"   📊 Progress: {day_idx}/{len(date_range)} days")
            print(f"   ✅ Success rate: {len(successful_days)}/{day_idx} ({len(successful_days)/day_idx*100:.1f}%)")
            print(f"   ⏱️  Estimated time remaining: {estimated_remaining_time/3600:.1f} hours")
            print(f"   🗓️  Next day: {date_range[day_idx].date() if day_idx < len(date_range) else 'COMPLETE'}")
            print()
            
            # Force garbage collection between days
            gc.collect()
            
            # Small delay to allow system to stabilize
            time.sleep(1)
    
    # Final summary
    total_time = time.time() - total_start_time
    
    print(f"\n{'='*60}")
    print(f"EXTRACTION COMPLETED")
    print(f"{'='*60}")
    print(f"Total processing time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"Wind output directory: {wind_output_dir}")
    print(f"Solar output directory: {solar_output_dir}")
    
    # Detailed summary
    print(f"\n📁 EXTRACTION SUMMARY")
    print(f"{'='*40}")
    print(f"📊 Processing Summary:")
    print(f"   Total days: {len(date_range)}")
    print(f"   Successful days: {len(successful_days)}")
    print(f"   Failed days: {len(failed_days)}")
    print(f"   Processing time: {total_time/3600:.1f} hours")
    print(f"   Wind locations: {len(wind_points)}")
    print(f"   Solar locations: {len(solar_points)}")
    
    if successful_days:
        print(f"\n✅ Extraction completed successfully!")
    else:
        print(f"\n❌ No days were processed successfully!")
    
    # Cleanup
    print(f"\n🧹 Performing final cleanup...")
    gc.collect()
    print(f"✅ Cleanup completed.")
    
    return {
        "status": "completed" if successful_days else "failed",
        "total_days": len(date_range),
        "successful_days": len(successful_days),
        "failed_days": len(failed_days),
        "processing_time_seconds": total_time,
        "wind_locations": len(wind_points),
        "solar_locations": len(solar_points),
        "wind_output_dir": wind_output_dir,
        "solar_output_dir": solar_output_dir
    }


def find_closest_grid_points(points, grid_lats, grid_lons):
    """Find closest grid points for given lat/lon coordinates."""
    # Handle case where grid_lats/grid_lons might be dictionaries
    if isinstance(grid_lats, dict):
        grid_lats = grid_lats['lats']
    if isinstance(grid_lons, dict):
        grid_lons = grid_lons['lons']
    
    # Ensure we have numpy arrays
    if not isinstance(grid_lats, np.ndarray):
        grid_lats = np.array(grid_lats)
    if not isinstance(grid_lons, np.ndarray):
        grid_lons = np.array(grid_lons)
    
    # Flatten grid coordinates
    grid_points = np.column_stack([grid_lats.flatten(), grid_lons.flatten()])
    
    # Create KDTree for efficient nearest neighbor search
    tree = KDTree(grid_points)
    
    # Find closest points
    query_points = points[['lat', 'lon']].values
    distances, indices = tree.query(query_points)
    
    return indices


