from functools import partial
import concurrent.futures
from tqdm import tqdm
import pandas as pd
from collections import defaultdict

import datetime
import functools
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pygrib
from powersimdata.utility.distance import ll2uv
from scipy.spatial import KDTree
from tqdm import tqdm  # Assuming you're using tqdm for progress tracking

from prereise.gather.winddata import const
from prereise.gather.const import (
    DATADIR,
    END,
    SEARCHSTRING,
    SELECTORS,
    START,
    TZ,
    YEAR,
)
from prereise.gather.winddata.hrrr.helpers import formatted_filename
from prereise.gather.winddata.impute import linear
from prereise.gather.winddata.power_curves import (
    get_power,
    get_state_power_curves,
    get_turbine_power_curves,
    shift_turbine_curve,
)


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
