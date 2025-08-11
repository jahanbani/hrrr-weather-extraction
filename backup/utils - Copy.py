import requests
import multiprocessing
from joblib import Parallel, delayed
import concurrent.futures
from tqdm import tqdm
import time

# import numba
import numpy as np
import pandas as pd
import PySAM.Pvwattsv8 as PVWatts
import PySAM.PySSC as pssc  # noqa: N813
import PySAM.ResourceTools as RT
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Bing, Photon
from herbie.fast import FastHerbie

# from prereise.gather.solardata.nsrdb.nrel_api import ipdb
from prereise.gather.winddata.hrrr.calculations import (
    calculate_pout_individual_vectorized,
)

geolocator = Photon(user_agent="geoapiExercises")


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
    response = requests.get(url)
    results = response.json()["results"]
    return results[0]["elevation"] if results else None


def get_elevations_batch(lat_lon_pairs, batch_size=100):
    """Get elevations for multiple lat/lon pairs in batches"""
    elevations = {}
    for i in range(0, len(lat_lon_pairs), batch_size):
        batch = lat_lon_pairs[i:i+batch_size]
        locations = "|".join([f"{lat},{lon}" for lat, lon in batch])
        url = f"https://api.open-elevation.com/api/v1/lookup?locations={locations}"
        try:
            response = requests.get(url)
            results = response.json()["results"]
            
            for j, result in enumerate(results):
                idx = i + j
                if idx < len(lat_lon_pairs):
                    elevations[lat_lon_pairs[idx]] = result.get("elevation")
        except Exception as e:
            print(f"Error getting elevations for batch {i}: {e}")
            # Fallback to individual calls for this batch
            for lat, lon in batch:
                try:
                    elevations[(lat, lon)] = get_elevation_open(lat, lon)
                except Exception as e2:
                    print(f"Error getting elevation for {lat}, {lon}: {e2}")
                    elevations[(lat, lon)] = None
    
    return elevations


def prepare_data_vectorized(df, YEAR):
    """Pre-process all data at once instead of per plant"""
    # Filter by year once
    df_filtered = df[df.index.year == YEAR].copy()
    
    # Add time components vectorized
    df_filtered["year"] = df_filtered.index.year
    df_filtered["month"] = df_filtered.index.month
    df_filtered["day"] = df_filtered.index.day
    df_filtered["hour"] = df_filtered.index.hour
    df_filtered["minute"] = df_filtered.index.minute
    
    # Clip values vectorized
    df_filtered.loc[df_filtered["df"] > 1000, "df"] = 1000
    df_filtered.loc[df_filtered["dn"] > 1000, "dn"] = 1000
    
    # Forward fill once
    df_filtered = df_filtered.ffill()
    
    return df_filtered


def get_elevation_with_retry(lat, lon, max_retries=3):
    """Get elevation with retry logic for rate limiting"""
    for attempt in range(max_retries):
        try:
            return get_elevation_open(lat, lon)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to get elevation for {lat}, {lon}: {e}")
                return None
            time.sleep(1)  # Rate limiting


def prepare_calculate_solar_power(
    solar_wind_speed_data,
    solar_tmp_data,
    solar_vdd_data,
    solar_vbd_data,
    solar_plant,
    YEAR,
    TZ,
    SOLARSCEN,
):
    default_pv_parameters = {
        "adjust:constant": 0,
        "gcr": 0.4,
        "inv_eff": 97,
        "losses": 14,
    }
    defaults = {
        "dc_ac_ratio": 1.2,
        "module_type": 2,
        "array_type": 0,
        "DC Net Capacity (MW)": 1,
        "Tilt Angle": 20,
        "Azimuth Angle": 180,
    }
    solar_plant = solar_plant.fillna(value=defaults)
    solar_plant = solar_plant.rename(
        columns={
            "DC Net Capacity (MW)": "system_capacity",
            "array_type": "array_type",
            "module_type": "module_type",
            "dc_ac_ratio": "dc_ac_ratio",
            "Azimuth Angle": "azimuth",
            "Tilt Angle": "tilt",
        }
    )
    # set dc net capacity to 1 mw for everyone
    solar_plant["system_capacity"] = 1

    # Replace negative values with zero
    solar_wind_speed_data[solar_wind_speed_data < 0] = 0
    solar_vdd_data[solar_vdd_data < 0] = 0
    solar_vbd_data[solar_vbd_data < 0] = 0
    solar_tmp_data[solar_tmp_data < 0] = 0

    solar_data_all = {
        "wspd": solar_wind_speed_data,
        "df": solar_vdd_data,
        "dn": solar_vbd_data,
        "tdry": solar_tmp_data - 273.15,
    }
    df = pd.concat(solar_data_all.values(), keys=solar_data_all.keys())

    latlon = solar_plant.loc[:, ["pid", "lat", "lon"]]
    # drop lat lon from solar_plant
    solar_plant = solar_plant.drop(columns=["lat", "lon"]).set_index(["pid"])
    # convert solar_plant to dictionary, where index is the key
    solar_plant_dict = solar_plant.to_dict(orient="index")
    # XXX FIXME how about ALBDO?

    # Prepare data for each plant
    dfs_dict = {}
    test = {}
    
    # OPTIMIZED VERSION - Get all elevations at once
    print("Getting elevations for all plants...")
    lat_lon_pairs = [(lat, lon) for _, lat, lon in latlon.values]
    elevations = get_elevations_batch(lat_lon_pairs)
    
    # Pre-process data once
    print("Pre-processing data vectorized...")
    df_processed = prepare_data_vectorized(df, YEAR)
    
    # Process plants more efficiently
    for inx, lat, lon in tqdm(latlon.values, desc="Preparing data"):
        dfs_inx = df_processed[[inx]].unstack(0)[inx].reset_index(drop=True)
        
        dfsd = dfs_inx.to_dict(orient="list")
        dfsd["lat"] = lat
        dfsd["lon"] = lon
        # dfsd["tz"] = -1 * TZ
        dfsd["tz"] = 0
        dfsd["elev"] = elevations.get((lat, lon))
        dfs_dict[inx] = dfsd

        if dfs_inx.isnull().values.any():
            print(f"{lat} and {lon} at {inx} has NaN")
    
    # ORIGINAL VERSION (COMMENTED OUT) - Uncomment to use original approach
    # for inx, lat, lon in tqdm(latlon.values, desc="Preparing data"):
    #     dfs_inx = df[[inx]].unstack(0)[inx]
    #     dfs_inx["year"] = dfs_inx.index.year
    #     dfs_inx["month"] = dfs_inx.index.month
    #     dfs_inx["day"] = dfs_inx.index.day
    #     dfs_inx["hour"] = dfs_inx.index.hour
    #     dfs_inx["minute"] = dfs_inx.index.minute
    #     dfs_inx = dfs_inx[dfs_inx.index.year == YEAR]
    #     dfs_inx.loc[dfs_inx["df"] > 1000, "df"] = 1000
    #     dfs_inx.loc[dfs_inx["dn"] > 1000, "dn"] = 1000
    #     dfs_inx = dfs_inx.reset_index(drop=True).ffill()
    #     dfsd = dfs_inx.to_dict(orient="list")
    #     dfsd["lat"] = lat
    #     dfsd["lon"] = lon
    #     # dfsd["tz"] = -1 * TZ
    #     dfsd["tz"] = 0
    #     dfsd["elev"] = get_elevation_open(lat, lon)
    #     dfs_dict[inx] = dfsd

    #     if dfs_inx.isnull().values.any():
    #         print(f"{lat} and {lon} at {inx} has NaN")

    # Remove the large DataFrame from memory
    # del df

    # results = [
    #     process_plant(
    #         inx, dfs_dict[inx], {**default_pv_parameters, **solar_plant_dict[inx]}
    #     )
    #     for inx in tqdm(dfs_dict.keys(), desc="Processing plants")
    # ]
    results = []
    for inx in dfs_dict.keys():
        print(inx)
        result = process_plant(
            inx, dfs_dict[inx], {**default_pv_parameters, **solar_plant_dict[inx]}
        )
        results.append(result)

    print("start of parallel processing")
    # results = Parallel(n_jobs=-1, max_nbytes=None, backend="loky", verbose=10)(
    #     delayed(process_plant)(inx, dfs_dict[inx], pv_dict)
    #     for inx in tqdm(dfs_dict.keys(), desc="Processing plants")
    # )
    print("end of parallel processing")

    # Concatenate the results
    df_power_all = pd.concat(results, axis=1)
    df_power_all.index = solar_wind_speed_data.index - pd.Timedelta(hours=TZ)

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
    df_power_all.to_parquet(f"solar_output_power_{SOLARSCEN}_{YEAR}.parquet")
    return df_power_all


def calculate_wind_pout(wind_speed_data, wind_farms, START, END, TZ, WINDSCEN):
    defaults = {
        "Predominant Turbine Manufacturer": "IEC",
        "Predominant Turbine Model Number": "Class 2",
        "Turbine Hub Height (Feet)": 328.084,
    }
    wind_farms = wind_farms.fillna(value=defaults)

    wind_speed_data.index = wind_speed_data.index - pd.Timedelta(hours=TZ)

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
        f"wind_output_{WINDSCEN}_{str(START.year)}_{str(START.month)}_to_{str(END.year)}_{str(END.month)}.parquet"
    )

    return wind_power


def calculate_solar_power(solar_data, pv_dict):
    """Use PVWatts to translate weather data into power.

    :param dict solar_data: weather data as returned by :meth:`Psm3Data.to_dict`.
    :param dict pv_dict: solar plant attributes.
    :return: (*numpy.array*) hourly power output.
    """
    pv_dat = pssc.dict_to_ssc_table(pv_dict, "pvwattsv8")
    pv = PVWatts.wrap(pv_dat)
    pv.SolarResource.assign({"solar_resource_data": solar_data})
    pv.execute(1)
    return np.array(pv.Outputs.gen)


def get_states(df):
    """takes lat lon and returns state
    XXX this seems to not work XXX FIXME
    """
    # Replace YOUR_API_KEY with your own Bing Maps API key
    geolocator = Bing(
        api_key="Aqmzwd34w1kATW-28eBQ1vRrdW3p8A2xAqeo3uokayav_LQqN-LqCrDPBtP_YSoM"
    )
    geocode = RateLimiter(geolocator.reverse, min_delay_seconds=1)

    def get_location_by_coordinates(lat, lon):
        location = geocode((lat, lon))
        return location.raw["address"].get("adminDistrict", "")

    def process_df(df):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            try:
                states = list(
                    executor.map(get_location_by_coordinates, df["lat"], df["lon"])
                )
                df["state"] = states
            except Exception as e:
                print(e)
                df["state"] = "offshore"
        return df

    # read the In_windsolarlocations.xlsx
    # df = pd.read_excel("In_windsolarlocations.xlsx")[["lat", "lon"]]
    df = process_df(df)

    return df


def read_data(points, START, END, DATADIR, SELECTORS):
    dataall = {}
    for inx, DEFAULT_HOURS_FORECASTED in enumerate(["0", "1"]):
        data = extract_data(
            points,
            START,
            END,
            DATADIR,
            DEFAULT_HOURS_FORECASTED,
            SELECTORS,
        )
        dataall[inx] = data

    # for each SELK merge the data to get the best data
    fns = {}
    for SELK, SELV in data.items():
        dataall[1][SELK].loc[dataall[1][SELK].index.minute == 0, :] = dataall[0][
            SELK
        ].loc[dataall[0][SELK].index.minute == 0, :]
        data[SELK] = dataall[1][SELK]
        # write them in parquet format
        # first we need ot convert columns to string
        data[SELK].columns = data[SELK].columns.astype(str)
        fn = (
            SELK
            + f"_{str(START.year)}_{str(START.month)}_to_{str(END.year)}_{str(END.month)}.parquet"
        )
        data[SELK].to_parquet(fn)
        fns[SELK] = fn

    return data, fns


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


def get_points(FNS):
    """
    Read latitude, longitude, and plant type information from a CSV file and
    return a DataFrame containing unique points for wind farms and solar plants.

    :param str csv_filepath: File path of the CSV containing point information.
    :return: (*pandas.DataFrame*) -- Data frame containing unique points for
        wind farms and solar plants, including columns 'pid', 'lat', and 'lon'.
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

    if "wind" in df.columns:
        wind_farms = df.loc[df["wind"] == 1, windcol]
        df.loc[df["wind"] == 1, :].to_excel("wind_farms.xlsx", index=False)

    else:
        wind_farms = pd.DataFrame(columns=windcol)
    if "solar" in df.columns:
        solar_plants = df.loc[df["solar"] == 1, solcol]
        df.loc[df["solar"] == 1, solcol].to_excel("solar_plants.xlsx", index=False)
    else:
        solar_plants = pd.DataFrame(columns=solcol)

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


def osw_model(wind_farms):
    if "Offshore" in wind_farms.columns:
        """add wind turbine manufacturer and model number for the offshore"""

        wind_farms.loc[
            wind_farms["Offshore"] == 1, "Predominant Turbine Model Number"
        ] = "V236"
        wind_farms.loc[
            wind_farms["Offshore"] == 1, "Predominant Turbine Manufacturer"
        ] = "Vestas"

    return wind_farms


def prepare_wind(wind_farms):
    """prepare the wind data to be used in the power calculation"""

    wind_farms = osw_model(wind_farms)
    if "Turbine Hub Height (Feet)" not in wind_farms.columns:
        # height in feet why?
        wind_farms.loc[:, "Turbine Hub Height (Feet)"] = 262.467

    return wind_farms


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
