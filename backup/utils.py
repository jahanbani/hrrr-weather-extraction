import requests
from tqdm import tqdm
import time

import numpy as np
import pandas as pd
import PySAM.Pvwattsv8 as PVWatts
import PySAM.PySSC as pssc  # noqa: N813
from herbie.fast import FastHerbie

from prereise.gather.winddata.hrrr.calculations import (
    calculate_pout_individual_vectorized,
)


# ##############################################################


def process_plant(inx, dfsd, pv_dict):
    """Process a single solar plant's data."""
    # Calculate power for the plant
    power = calculate_solar_power(dfsd, pv_dict)
    df_power = pd.DataFrame(power).rename(columns={0: inx})

    if df_power.loc[df_power[inx] > 0].empty:
        print(f"Output of {inx} is all zeros")
    if df_power.loc[df_power[inx] > 0].max().values > 1:
        print(f"Output of {inx} is greater than one")

    return df_power


def get_elevation_open(lat, lon):
    url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            results = response.json()["results"]
            return results[0]["elevation"] if results else None
        else:
            print(f"API returned status code {response.status_code} for {lat}, {lon}")
            return None
    except Exception as e:
        print(f"Error getting elevation for {lat}, {lon}: {e}")
        return None


def _fallback_elevation_calls(batch, elevations, successful_calls):
    """Helper function to handle fallback elevation calls for a batch"""
    for lat, lon in batch:
        try:
            elevation = get_elevation_open(lat, lon)
            elevations[(lat, lon)] = elevation
            if elevation is not None:
                successful_calls += 1
        except Exception as e2:
            print(f"Error getting elevation for {lat}, {lon}: {e2}")
            elevations[(lat, lon)] = None
    return successful_calls


def get_elevations_batch(lat_lon_pairs, batch_size=100):
    """Get elevations for multiple lat/lon pairs in batches"""
    elevations = {}
    successful_calls = 0
    total_calls = len(lat_lon_pairs)

    print(
        f"Getting elevations for {total_calls} locations in batches of {batch_size}..."
    )

    for i in range(0, len(lat_lon_pairs), batch_size):
        batch = lat_lon_pairs[i : i + batch_size]
        locations = "|".join([f"{lat},{lon}" for lat, lon in batch])
        url = f"https://api.open-elevation.com/api/v1/lookup?locations={locations}"

        try:
            response = requests.get(url, timeout=30)  # Add timeout
            if response.status_code == 200:
                results = response.json()["results"]

                for j, result in enumerate(results):
                    idx = i + j
                    if idx < len(lat_lon_pairs):
                        elevation = result.get("elevation")
                        elevations[lat_lon_pairs[idx]] = elevation
                        if elevation is not None:
                            successful_calls += 1
            else:
                print(f"API returned status code {response.status_code} for batch {i}")
                successful_calls = _fallback_elevation_calls(
                    batch, elevations, successful_calls
                )

        except Exception as e:
            print(f"Error getting elevations for batch {i}: {e}")
            successful_calls = _fallback_elevation_calls(
                batch, elevations, successful_calls
            )

        # Add a small delay to avoid rate limiting
        time.sleep(0.1)

    print(
        f"Successfully retrieved {successful_calls}/{total_calls} elevations ({successful_calls / total_calls * 100:.1f}%)"
    )

    # Check if we got enough elevations
    if successful_calls < total_calls * 0.8:  # Less than 80% success rate
        print(
            "WARNING: Many elevation API calls failed. Consider using a different elevation source."
        )

    return elevations


def prepare_data_vectorized(df):
    """Pre-process all data at once instead of per plant"""
    # The DataFrame has a MultiIndex with data type and timestamp
    # We need to work with the timestamp level (level 1)
    timestamp_index = df.index.get_level_values(1)

    # Filter by year once - use the correct axis
    # year_mask = timestamp_index.year == YEAR
    # df_filtered = df.loc[year_mask, :].copy()
    df_filtered = df

    # Clip values vectorized (only for 'df' and 'dn' data types)
    if "df" in df_filtered.index.get_level_values(0):
        df_data = df_filtered.loc["df"]
        df_data[df_data > 1000] = 1000
        df_filtered.loc["df"] = df_data

    if "dn" in df_filtered.index.get_level_values(0):
        dn_data = df_filtered.loc["dn"]
        dn_data[dn_data > 1000] = 1000
        df_filtered.loc["dn"] = dn_data

    # Forward fill once
    df_filtered = df_filtered.ffill()

    return df_filtered


def prepare_calculate_solar_power(
    solar_wind_speed_data,
    solar_tmp_data,
    solar_vdd_data,
    solar_vbd_data,
    solar_plant,
    TZ,
):
    # PVWatts parameters (these are model-specific, not plant-specific)
    default_pv_parameters = {
        "adjust_constant": 0,
        "gcr": 0.4,
        "inv_eff": 97,
        "losses": 14,
    }

    # Replace negative values with zero
    solar_wind_speed_data[solar_wind_speed_data.columns] = solar_wind_speed_data[
        solar_wind_speed_data.columns
    ].clip(lower=0)
    solar_vdd_data[solar_vdd_data.columns] = solar_vdd_data[
        solar_vdd_data.columns
    ].clip(lower=0)
    solar_vbd_data[solar_vbd_data.columns] = solar_vbd_data[
        solar_vbd_data.columns
    ].clip(lower=0)
    solar_tmp_data[solar_tmp_data.columns] = solar_tmp_data[
        solar_tmp_data.columns
    ].clip(lower=0)

    # solar_wind_speed_data.loc[solar_wind_speed_data < 0] = 0
    # solar_vdd_data.loc[solar_vdd_data < 0] = 0
    # solar_vbd_data.loc[solar_vbd_data < 0] = 0
    # solar_tmp_data.loc[solar_tmp_data < 0] = 0

    solar_data_all = {
        "wspd": solar_wind_speed_data,
        "df": solar_vdd_data,
        "dn": solar_vbd_data,
        "tdry": solar_tmp_data,
    }
    df = pd.concat(solar_data_all.values(), keys=solar_data_all.keys())

    latlon = solar_plant.loc[:, ["pid", "lat", "lon"]]
    # drop lat lon from solar_plant
    solar_plant = solar_plant.drop(columns=["lat", "lon"])
    # Don't set index to avoid MultiIndex issues
    # solar_plant = solar_plant.set_index(["pid"])
    # convert solar_plant to dictionary, where pid is the key
    solar_plant_dict = solar_plant.set_index("pid").to_dict(orient="index")
    # XXX FIXME how about ALBDO?

    # Prepare data for each plant
    dfs_dict = {}

    # Pre-process data vectorized
    print("Pre-processing data vectorized...")
    print(f"Original DataFrame shape: {df.shape}")
    print(f"DataFrame index levels: {df.index.names}")
    print(f"DataFrame index length: {len(df.index)}")
    print(f"DataFrame columns: {list(df.columns[:5])}")
    print(f"DataFrame column type: {type(df.columns[0])}")

    df_processed = prepare_data_vectorized(df)
    print(f"Filtered DataFrame shape: {df_processed.shape}")

    # Process plants efficiently
    for inx, lat, lon in tqdm(latlon.values, desc="Preparing data"):
        dfs_inx = df_processed[[inx]].unstack(0)[inx]

        # Add time components to the DataFrame - use the actual DataFrame index
        dfs_inx["year"] = dfs_inx.index.year
        dfs_inx["month"] = dfs_inx.index.month
        dfs_inx["day"] = dfs_inx.index.day
        dfs_inx["hour"] = dfs_inx.index.hour
        dfs_inx["minute"] = dfs_inx.index.minute
        dfs_inx = dfs_inx.reset_index(drop=True)

        dfsd = dfs_inx.to_dict(orient="list")
        dfsd["lat"] = lat
        dfsd["lon"] = lon
        # dfsd["tz"] = -1 * TZ
        dfsd["tz"] = 0
        # Get elevation from the solar_plant DataFrame (already calculated in get_points)
        plant_elevation = (
            solar_plant.loc[solar_plant["pid"] == inx, "elevation"].iloc[0]
            if inx in solar_plant["pid"].values
            else None
        )
        dfsd["elev"] = plant_elevation
        dfs_dict[inx] = dfsd

        if dfs_inx.isnull().values.any():
            print(f"{lat} and {lon} at {inx} has NaN")

    # Process plants to calculate power
    results = []
    for inx in dfs_dict.keys():
        print(inx)
        result = process_plant(
            inx, dfs_dict[inx], {**default_pv_parameters, **solar_plant_dict[inx]}
        )
        results.append(result)

    # Concatenate the results
    df_power_all = pd.concat(results, axis=1)
    df_power_all.index = solar_wind_speed_data.index

    # Apply timezone shift at the end before returning
    df_power_all.index = df_power_all.index - pd.Timedelta(hours=TZ)

    # XXX FIXME: Address the leap year issue
    # Create date range index
    if False:
        dfindex = pd.date_range(
            start=f"{YEAR}-01-01 00:00",
            end=f"{YEAR + 1}-01-01 00:00",
            freq="15T",
            inclusive="left",
        )

        # Handle leap year data
        leapdfindex = dfindex[(dfindex.month == 2) & (dfindex.day == 29)]
        dfindex = dfindex[~((dfindex.month == 2) & (dfindex.day == 29))]
        df_power_all.index = dfindex

        # Add data for Feb 29th if it's a leap year
        if YEAR == 2020:
            leapdata = df_power_all.loc[
                (df_power_all.index.month == 2) & (df_power_all.index.day == 28)
            ]
            leapdata.index = leapdfindex
            df_power_all = pd.concat([df_power_all, leapdata], axis=0).sort_index(
                axis=0
            )

    # Convert the columns to strings
    df_power_all.columns = df_power_all.columns.astype(str)
    df_power_all.to_parquet(
        f"solar_output_power_{solar_wind_speed_data.index[0].year}.parquet"
    )
    return df_power_all


def calculate_wind_pout(wind_speed_data, wind_farms, START, END, TZ):
    # Wind farms data should already be prepared with defaults from get_points()

    wind_power = calculate_pout_individual_vectorized(
        wind_speed_data,
        wind_farms,
        start_dt=START,
        end_dt=END,
        # hours_forecasted=DEFAULT_HOURS_FORECASTED,
    )

    # wind_power.to_excel("wind_output" + ".xlsx")
    # convert the columns of wind power to string
    wind_power.columns = wind_power.columns.astype(str)
    wind_power.to_parquet(
        f"wind_output_{str(START.year)}_{str(START.month)}_to_{str(END.year)}_{str(END.month)}.parquet"
    )

    # Apply timezone shift at the end before returning
    wind_power.index = wind_power.index - pd.Timedelta(hours=TZ)

    return wind_power


def calculate_solar_power(solar_data, pv_dict):
    """Use PVWatts to translate weather data into power.

    :param dict solar_data: weather data as returned by :meth:`Psm3Data.to_dict`.
    :param dict pv_dict: solar plant attributes.
    :return: (*numpy.array*) hourly power output.
    """
    # Azimuth and tilt angle must not be negative NOTE
    print("*" * 80)
    print(pv_dict)
    print("*" * 80)
    pv_dat = pssc.dict_to_ssc_table(pv_dict, "pvwattsv8")
    pv = PVWatts.wrap(pv_dat)
    pv.SolarResource.assign({"solar_resource_data": solar_data})
    pv.execute(1)
    return np.array(pv.Outputs.gen)


def check_columns(df, tech, solcol, windcol):
    if tech == "solar":
        # remove any columns that are not in the list of columns
        df = df.loc[:, df.columns.isin(solcol)]

        if not all([col in df.columns for col in solcol]):
            raise ValueError(f"Columns {solcol} not found in {tech} data frame.")
    elif tech == "wind":
        df = df.loc[:, df.columns.isin(windcol)]
        if not all([col in df.columns for col in windcol]):
            raise ValueError(f"Columns {windcol} not found in {tech} data frame.")
    return df


def ensure_column_types(df, tech):
    """Ensure proper data types for solar and wind plant columns"""
    if tech == "solar":
        # Solar plant column types
        type_mapping = {
            "pid": "string",  # Plant ID
            "lat": "float64",  # Latitude
            "lon": "float64",  # Longitude
            "Azimuth Angle": "float64",  # Degrees
            "Tilt Angle": "float64",  # Degrees
            "DC Net Capacity (MW)": "float64",  # Megawatts
            "array_type": "int64",  # Array type code
            "module_type": "int64",  # Module type code
            "dc_ac_ratio": "float64",  # DC/AC ratio
            "elevation": "float64",  # Elevation in meters
        }
    elif tech == "wind":
        # Wind farm column types
        type_mapping = {
            "pid": "string",  # Plant ID
            "lat": "float64",  # Latitude
            "lon": "float64",  # Longitude
            "Predominant Turbine Manufacturer": "string",  # Manufacturer name
            "Predominant Turbine Model Number": "string",  # Model number
            "Turbine Hub Height (Feet)": "float64",  # Height in feet
            "elevation": "float64",  # Elevation in meters
        }
    else:
        return df

    # Apply type conversions with error handling
    for col, dtype in type_mapping.items():
        if col in df.columns:
            try:
                if dtype == "string":
                    df[col] = df[col].astype("string")
                elif dtype == "float64":
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
                elif dtype == "int64":
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype(
                        "Int64"
                    )  # nullable integer
            except Exception as e:
                print(f"Warning: Could not convert column '{col}' to {dtype}: {e}")
                # Keep original type if conversion fails

    return df


def get_points(FNS):
    """
    Read latitude, longitude, and plant type information from files and
    return DataFrames containing prepared data for wind farms and solar plants.

    This function:
    1. Reads plant data from various file formats (CSV, Excel, Parquet)
    2. Ensures proper data types for all columns
    3. Calculates elevations for all unique locations
    4. Applies default values for missing solar plant parameters
    5. Applies default values for missing wind farm parameters
    6. Handles offshore wind farm specifications
    7. Renames columns to match PVWatts requirements
    8. Handles duplicate lat/lon coordinates
    9. Returns prepared data ready for power calculations.

    :param dict FNS: Dictionary with 'solar' and 'wind' keys containing lists of file paths
    :return: tuple -- (latlon_data, wind_farms, solar_plants, mapping)
        - latlon_data: DataFrame with unique lat/lon points
        - wind_farms: DataFrame with prepared wind farm data (defaults applied)
        - solar_plants: DataFrame with prepared solar plant data (defaults applied)
        - mapping: Dictionary mapping duplicate coordinates
    """
    # Read data from the CSV file
    solcol = [
        "pid",
        "lat",
        "lon",
        "Azimuth Angle",
        "Tilt Angle",
        "DC Net Capacity (MW)",
        "array_type",
        "module_type",
        "dc_ac_ratio",
    ]
    windcol = [
        "pid",
        "Predominant Turbine Manufacturer",
        "Predominant Turbine Model Number",
        "Turbine Hub Height (Feet)",
        "lat",
        "lon",
    ]

    latlondict = {}
    for t, FNL in FNS.items():
        dfs = []
        for inx, FN in enumerate(FNL):
            if FN.endswith(".xlsx"):
                dft = pd.read_excel(FN)
            elif FN.endswith(".parquet"):
                dft = pd.read_parquet(FN)
            elif FN.endswith(".csv"):
                dft = pd.read_csv(FN)
            else:
                raise ValueError("File name must end with csv, .xlsx, or .parquet")
            if "latitude" in dft.columns:
                dft = dft.rename(columns={"latitude": "lat"})
            if "longitude" in dft.columns:
                dft = dft.rename(columns={"longitude": "lon"})
            dft = check_columns(dft, t, solcol, windcol)

            # Ensure proper data types for each column
            dft = ensure_column_types(dft, t)

            dfs.append(
                dft.dropna(subset=["lat", "lon"])
                .drop_duplicates(subset=["lat", "lon"])
                .reset_index(drop=True)
            )
        latlondict[t] = pd.concat(dfs, ignore_index=True)

    latlon = {
        key: df.assign(**{k: int(k == key) for k in latlondict})
        for key, df in latlondict.items()
    }

    df = pd.concat(latlon.values(), ignore_index=True)

    # Get elevations for all unique locations
    print("Getting elevations for all unique locations...")
    unique_locations = df[["lat", "lon"]].drop_duplicates()
    lat_lon_pairs = [(lat, lon) for lat, lon in unique_locations.values]
    elevations = get_elevations_batch(lat_lon_pairs)

    # Add elevation to the main DataFrame
    df["elevation"] = df.apply(
        lambda row: elevations.get((row["lat"], row["lon"])), axis=1
    )

    if "wind" in df.columns:
        wind_farms = df.loc[df["wind"] == 1, windcol + ["elevation"]]

        # Apply wind farm defaults
        wind_defaults = {
            "Predominant Turbine Manufacturer": "IEC",
            "Predominant Turbine Model Number": "Class 2",
            "Turbine Hub Height (Feet)": 328.084,
        }
        wind_farms = wind_farms.fillna(value=wind_defaults)

        # Handle offshore wind farms (if Offshore column exists)
        if "Offshore" in wind_farms.columns:
            wind_farms.loc[
                wind_farms["Offshore"] == 1, "Predominant Turbine Model Number"
            ] = "V236"
            wind_farms.loc[
                wind_farms["Offshore"] == 1, "Predominant Turbine Manufacturer"
            ] = "Vestas"

        # Ensure hub height column exists
        if "Turbine Hub Height (Feet)" not in wind_farms.columns:
            wind_farms["Turbine Hub Height (Feet)"] = 262.467

        df.loc[df["wind"] == 1, windcol + ["elevation"]].to_excel(
            "wind_farms.xlsx", index=False
        )
    else:
        wind_farms = pd.DataFrame(columns=windcol + ["elevation"])

    if "solar" in df.columns:
        solar_plants = df.loc[df["solar"] == 1, solcol + ["elevation"]]

        # Apply solar plant defaults
        solar_defaults = {
            "dc_ac_ratio": 1.2,
            "module_type": 2,
            "array_type": 0,
            "DC Net Capacity (MW)": 1,
            "Tilt Angle": 20,
            "Azimuth Angle": 180,
        }
        solar_plants = solar_plants.fillna(value=solar_defaults)

        # Rename columns to match PVWatts requirements
        solar_plants = solar_plants.rename(
            columns={
                "DC Net Capacity (MW)": "system_capacity",
                "array_type": "array_type",
                "module_type": "module_type",
                "dc_ac_ratio": "dc_ac_ratio",
                "Azimuth Angle": "azimuth",
                "Tilt Angle": "tilt",
            }
        )
        # Set DC net capacity to 1 MW for everyone
        solar_plants["system_capacity"] = 1

        df.loc[df["solar"] == 1, solcol + ["elevation"]].to_excel(
            "solar_plants.xlsx", index=False
        )
    else:
        solar_plants = pd.DataFrame(columns=solcol + ["elevation"])

    # identifying repetitive lat lon because we combine wind and solar
    # Create a DataFrame with duplicates removed
    latlon_data = df.drop_duplicates(subset=["lat", "lon"])
    # Create a dictionary to store the mapping
    mapping = {}
    # Iterate over the unique DataFrame
    for index, row in latlon_data.iterrows():
        # Find the indices in the original DataFrame that match the current unique row
        # matching_indices = df[(df['lat'] == row['lat']) & (df['lon'] == row['lon'])].index.tolist()
        matching_indices = df[(df["lat"] == row["lat"]) & (df["lon"] == row["lon"])][
            "pid"
        ].tolist()
        # Remove the current index from the list of matching indices
        matching_indices.remove(row["pid"])
        # If there are any matching indices left, add them to the mapping
        if matching_indices:
            mapping[row["pid"]] = matching_indices

    return (
        latlon_data,
        wind_farms,
        solar_plants,
        mapping,
    )


def download_data(START, END, DATADIR, SEARCHSTRING):
    # Create a range of dates
    FHDATES = pd.date_range(
        start=START,
        end=END,
        freq="1h",
    )

    print("Create a range of forecast lead times")
    fxx = [0, 1]
    FH = FastHerbie(
        FHDATES,
        model="hrrr",
        fxx=fxx,
        product="subh",
        max_threads=100,
        save_dir=DATADIR,
    )
    print("downloading")
    FH.download(
        search=SEARCHSTRING,
        save_dir=DATADIR,
        max_threads=100,
        verbose=True,
    )
