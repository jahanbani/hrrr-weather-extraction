"""
Wind and Solar Power Calculations
Reads parquet files from hrrr_enhanced.py and calculates wind/solar power outputs
"""

import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import functools
from tqdm import tqdm
import PySAM.Pvwattsv8 as PVWatts
import PySAM.PySSC as pssc

# Import constants from config_unified
from config_unified import DEFAULT_CONFIG, get_constants

import utils

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Define calculation flags
CALCWIND = True
CALCSOL = True


def verify_data_quality(df, var_name):
    """
    Verify data quality for a given DataFrame
    """
    logger.info(f"🔍 Verifying data quality for {var_name}")

    issues = []

    # 1. Check for missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        issues.append(f"❌ {missing_count} missing values found")

    # 2. Check for infinite values
    inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    if inf_count > 0:
        issues.append(f"❌ {inf_count} infinite values found")

    # 3. Check time index consistency (15-minute intervals)
    if df.index.is_monotonic_increasing:
        time_diffs = df.index.to_series().diff().dropna()
        expected_diff = pd.Timedelta(minutes=15)

        # Check if all intervals are 15 minutes (with small tolerance)
        tolerance = pd.Timedelta(seconds=30)  # 30 second tolerance
        irregular_intervals = time_diffs[abs(time_diffs - expected_diff) > tolerance]

        if len(irregular_intervals) > 0:
            issues.append(
                f"❌ {len(irregular_intervals)} irregular time intervals found"
            )
            logger.warning(
                f"   Sample irregular intervals: {irregular_intervals.head()}"
            )

    # 4. Variable-specific checks
    if var_name == "2tmp":
        # Temperature should be in reasonable range (-100 to +100 Celsius)
        temp_range = df.values.flatten()
        temp_range = temp_range[~np.isnan(temp_range)]  # Remove NaN

        if len(temp_range) > 0:
            min_temp = np.min(temp_range)
            max_temp = np.max(temp_range)

            if min_temp < -100 or max_temp > 100:
                issues.append(
                    f"❌ Temperature out of reasonable range: {min_temp:.2f} to {max_temp:.2f}°C"
                )

            # Check for negative temperatures (should be converted to Celsius)
            if min_temp > 0:
                issues.append(
                    f"⚠️ Temperature appears to be in Kelvin (min: {min_temp:.2f}K)"
                )

    elif var_name in ["vdd", "vbd", "rad"]:
        # Solar radiation should be positive
        negative_count = (df < 0).sum().sum()
        if negative_count > 0:
            issues.append(f"❌ {negative_count} negative solar radiation values found")

        # Solar radiation should be reasonable (0-1500 W/m²)
        max_rad = df.max().max()
        if max_rad > 1500:
            issues.append(f"⚠️ Unusually high solar radiation: {max_rad:.2f} W/m²")

    elif var_name.startswith("WindSpeed"):
        # Wind speed should be positive
        negative_count = (df < 0).sum().sum()
        if negative_count > 0:
            issues.append(f"❌ {negative_count} negative wind speed values found")

        # Wind speed should be reasonable (0-100 m/s)
        max_wind = df.max().max()
        if max_wind > 100:
            issues.append(f"⚠️ Unusually high wind speed: {max_wind:.2f} m/s")

    # 5. Check data completeness
    expected_hours = 24 * 365  # Approximate hours in a year
    actual_hours = len(df) / 4  # 4 readings per hour (15-minute intervals)
    completeness = actual_hours / expected_hours

    if completeness < 0.95:
        issues.append(f"⚠️ Data completeness: {completeness:.1%} (expected >95%)")

    # Report results
    if issues:
        logger.warning(f"   Issues found for {var_name}:")
        for issue in issues:
            logger.warning(f"     {issue}")
        return False
    else:
        logger.info(f"   ✅ {var_name} data quality verified")
        return True


def convert_temperature_to_celsius(df):
    """
    Convert temperature from Kelvin to Celsius
    """
    logger.info("🌡️ Converting temperature from Kelvin to Celsius")

    # Check if temperature is in Kelvin (typically > 200K)
    if df.max().max() > 200:
        logger.info("   Converting from Kelvin to Celsius")
        df_celsius = df - 273.15
        return df_celsius
    else:
        logger.info("   Temperature already appears to be in Celsius")
        return df


def check_consolidated_files_exist():
    """
    Check if consolidated parquet files already exist
    Returns: Dictionary with file info if they exist, None otherwise
    """
    output_dir = "consolidated_data"
    if not os.path.exists(output_dir):
        return None

    # Check for expected consolidated files
    expected_vars = ["WindSpeed80", "WindSpeed10", "2tmp", "vdd", "vbd", "rad"]
    existing_files = {}

    for var in expected_vars:
        filename = f"{var}_consolidated.parquet"
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
            existing_files[var] = {
                "filename": filename,
                "filepath": filepath,
                "size_mb": file_size,
            }

    return existing_files if existing_files else None


def read_consolidated_files():
    """
    Read existing consolidated parquet files
    Returns: Dictionary of DataFrames with weather data
    """
    logger.info("📖 Reading existing consolidated parquet files...")

    existing_files = check_consolidated_files_exist()
    if not existing_files:
        logger.warning("⚠️ No consolidated files found")
        return {}

    weather_data = {}

    for var, file_info in existing_files.items():
        try:
            df = pd.read_parquet(file_info["filepath"])
            weather_data[var] = df

            logger.info(
                f"✅ Loaded {var}: {df.shape[0]} rows, {df.shape[1]} columns, {file_info['size_mb']:.2f} MB"
            )
            logger.info(f"   Time range: {df.index.min()} to {df.index.max()}")

        except Exception as e:
            logger.error(f"❌ Error reading {var}: {e}")

    return weather_data


def verify_and_write_consolidated_files(weather_data):
    """
    Verify data quality and write consolidated files (write even if verification fails)
    """
    logger.info("🔍 Verifying data quality and writing consolidated files...")

    # Create output directory
    output_dir = "consolidated_data"
    os.makedirs(output_dir, exist_ok=True)

    saved_files = []
    verification_results = {}

    for var_name, df in weather_data.items():
        # Verify data quality
        verification_passed = verify_data_quality(df, var_name)
        verification_results[var_name] = verification_passed

        # Write consolidated file regardless of verification result
        try:
            filename = f"{var_name}_consolidated.parquet"
            filepath = os.path.join(output_dir, filename)

            # Save to parquet
            df.to_parquet(filepath, compression="snappy")

            # Get file size
            file_size = os.path.getsize(filepath) / (1024 * 1024)  # Convert to MB

            status = "✅" if verification_passed else "⚠️"
            logger.info(
                f"{status} Saved {filename}: {df.shape[0]} rows, {df.shape[1]} columns, {file_size:.2f} MB"
            )
            logger.info(f"   Time range: {df.index.min()} to {df.index.max()}")

            saved_files.append(
                {
                    "variable": var_name,
                    "filename": filename,
                    "rows": df.shape[0],
                    "columns": df.shape[1],
                    "size_mb": file_size,
                    "time_range": f"{df.index.min()} to {df.index.max()}",
                    "verification_passed": verification_passed,
                }
            )

        except Exception as e:
            logger.error(f"❌ Error saving {var_name}: {e}")

    # Summary of verification results
    passed_count = sum(verification_results.values())
    total_count = len(verification_results)

    logger.info(
        f"📊 Verification summary: {passed_count}/{total_count} variables passed quality checks"
    )

    if passed_count < total_count:
        logger.warning(
            "⚠️ Some data quality issues found, but consolidated files were still written"
        )

    return saved_files


def read_parquet_files():
    """
    Read parquet files from /research/alij/extracted_data/
    Returns: Dictionary of DataFrames with weather data organized by variable type
    """
    logger.info("📖 Reading parquet files from /research/alij/extracted_data/")

    base_dir = "/research/alij/extracted_data"

    # Only read WindSpeed variables (we have calculated wind speeds)
    wind_variables = ["WindSpeed80", "WindSpeed10"]
    solar_variables = ["rad", "vbd", "vdd", "2tmp"]

    data = {}

    # Read wind data
    wind_dir = os.path.join(base_dir, "wind")
    if os.path.exists(wind_dir):
        logger.info(f"📁 Reading wind data from {wind_dir}")

        for var in wind_variables:
            var_dir = os.path.join(wind_dir, var)
            if os.path.exists(var_dir):
                logger.info(f"📂 Reading {var} data...")

                # Get all parquet files for this variable
                parquet_files = [
                    f for f in os.listdir(var_dir) if f.endswith(".parquet")
                ]
                parquet_files.sort()  # Sort by date

                if parquet_files:
                    logger.info(f"   Found {len(parquet_files)} files for {var}")

                    # Read and concatenate all files for this variable
                    dfs = []
                    for file in parquet_files:
                        try:
                            file_path = os.path.join(var_dir, file)
                            df = pd.read_parquet(file_path)
                            dfs.append(df)
                            logger.info(
                                f"   ✅ Loaded {file}: {df.shape[0]} rows, {df.shape[1]} columns"
                            )
                        except Exception as e:
                            logger.error(f"   ❌ Error reading {file}: {e}")

                    if dfs:
                        # Concatenate all dataframes
                        combined_df = pd.concat(dfs, axis=0)
                        combined_df = combined_df.sort_index()  # Sort by time

                        # Convert temperature to Celsius if needed
                        if var == "2tmp":
                            combined_df = convert_temperature_to_celsius(combined_df)

                        data[var] = combined_df
                        logger.info(
                            f"   ✅ Combined {var}: {combined_df.shape[0]} rows, {combined_df.shape[1]} columns"
                        )
                        logger.info(
                            f"   Time range: {combined_df.index.min()} to {combined_df.index.max()}"
                        )

    # Read solar data (if available)
    solar_dir = os.path.join(base_dir, "solar")
    if os.path.exists(solar_dir):
        logger.info(f"📁 Reading solar data from {solar_dir}")

        # Check what solar variables are available
        available_solar_vars = [
            d
            for d in os.listdir(solar_dir)
            if os.path.isdir(os.path.join(solar_dir, d))
        ]
        logger.info(f"   Available solar variables: {available_solar_vars}")

        for var in available_solar_vars:
            var_dir = os.path.join(solar_dir, var)
            logger.info(f"📂 Reading {var} data...")

            # Get all parquet files for this variable
            parquet_files = [f for f in os.listdir(var_dir) if f.endswith(".parquet")]
            parquet_files.sort()  # Sort by date

            if parquet_files:
                logger.info(f"   Found {len(parquet_files)} files for {var}")

                # Read and concatenate all files for this variable
                dfs = []
                for file in parquet_files:
                    try:
                        file_path = os.path.join(var_dir, file)
                        df = pd.read_parquet(file_path)
                        dfs.append(df)
                        logger.info(
                            f"   ✅ Loaded {file}: {df.shape[0]} rows, {df.shape[1]} columns"
                        )
                    except Exception as e:
                        logger.error(f"   ❌ Error reading {file}: {e}")

                if dfs:
                    # Concatenate all dataframes
                    combined_df = pd.concat(dfs, axis=0)
                    combined_df = combined_df.sort_index()  # Sort by time

                    # Convert temperature to Celsius if needed
                    if var == "2tmp":
                        combined_df = convert_temperature_to_celsius(combined_df)

                    data[var] = combined_df
                    logger.info(
                        f"   ✅ Combined {var}: {combined_df.shape[0]} rows, {combined_df.shape[1]} columns"
                    )
                    logger.info(
                        f"   Time range: {combined_df.index.min()} to {combined_df.index.max()}"
                    )

    if not data:
        logger.warning("⚠️ No parquet files found in extracted_data directory")

    return data


def save_consolidated_parquet_files(weather_data):
    """
    Save each variable as a single consolidated parquet file
    """
    logger.info("💾 Saving consolidated parquet files...")

    # Create output directory
    output_dir = "consolidated_data"
    os.makedirs(output_dir, exist_ok=True)

    saved_files = []

    for var_name, df in weather_data.items():
        try:
            # Create filename
            filename = f"{var_name}_consolidated.parquet"
            filepath = os.path.join(output_dir, filename)

            # Save to parquet
            df.to_parquet(filepath, compression="snappy")

            # Get file size
            file_size = os.path.getsize(filepath) / (1024 * 1024)  # Convert to MB

            logger.info(
                f"✅ Saved {filename}: {df.shape[0]} rows, {df.shape[1]} columns, {file_size:.2f} MB"
            )
            logger.info(f"   Time range: {df.index.min()} to {df.index.max()}")

            saved_files.append(
                {
                    "variable": var_name,
                    "filename": filename,
                    "rows": df.shape[0],
                    "columns": df.shape[1],
                    "size_mb": file_size,
                    "time_range": f"{df.index.min()} to {df.index.max()}",
                }
            )

        except Exception as e:
            logger.error(f"❌ Error saving {var_name}: {e}")

    logger.info(
        f"🎉 Successfully saved {len(saved_files)} consolidated parquet files to {output_dir}/"
    )

    return saved_files


def main():
    """
    Main function to orchestrate the power calculation process
    """
    # Get constants
    constants = get_constants()

    logger.info("🚀 Starting power calculations")
    logger.info(f"📅 Processing period: {constants['START']} to {constants['END']}")

    # Step 1: Read wind and solar farm data using get_points function
    latlon_data, wind_farms, solar_plants, mapping = utils.get_points(
        {
            "wind": ["backup/wind.csv"],
            "solar": ["backup/solar.csv"],
        }
    )

    logger.info(
        f"📊 Loaded {len(wind_farms)} wind farms and {len(solar_plants)} solar plants"
    )

    # Step 2: Check if consolidated files exist
    existing_files = check_consolidated_files_exist()

    if existing_files:
        logger.info("📁 Found existing consolidated files, reading them...")
        weather_data = read_consolidated_files()

        if weather_data:
            logger.info("✅ Successfully loaded consolidated files")
            # Run verification on existing files
            # verify_and_write_consolidated_files(weather_data)
        else:
            logger.warning(
                "⚠️ Failed to read consolidated files, will read individual files"
            )
            weather_data = read_parquet_files()
            saved_files = verify_and_write_consolidated_files(weather_data)
    else:
        logger.info("📁 No consolidated files found, reading individual files...")
        weather_data = read_parquet_files()
        saved_files = verify_and_write_consolidated_files(weather_data)

    logger.info("✅ Data loading and verification completed")

    # Initialize output variables
    wind_output_power = None
    solar_output_power = None

    if CALCWIND:
        logger.info("🌪️ Starting wind power calculations...")
        # find the wind related files; there is Wind80 in their names
        windfns = [fn for fn in weather_data.keys() if fn.startswith("WindSpeed80")]
        if windfns:
            logger.info(f"📁 Found wind data files: {windfns}")
            try:
                wind_output_power = utils.calculate_wind_pout(
                    weather_data[windfns[0]][
                        list(
                            set(weather_data[windfns[0]].columns).intersection(
                                set(wind_farms["pid"])
                            )
                        )
                    ],
                    wind_farms,
                    constants["START"],
                    constants["END"],
                    constants["TZ"],
                ).round(2)
                logger.info("✅ Wind power calculations completed successfully")
            except Exception as e:
                logger.error(f"❌ Error in wind power calculations: {e}")
        else:
            logger.warning("⚠️ WindSpeed80 files not found in weather_data.")
            logger.info(f"Available weather data keys: {list(weather_data.keys())}")

    if CALCSOL:
        logger.info("☀️ Starting solar power calculations...")
        # find the solar related files
        solw = [fn for fn in weather_data.keys() if fn.startswith("WindSpeed10")]
        sol2tmp = [fn for fn in weather_data.keys() if fn.startswith("2tmp")]
        solvdd = [fn for fn in weather_data.keys() if fn.startswith("vdd")]
        solvbd = [fn for fn in weather_data.keys() if fn.startswith("vbd")]

        logger.info(f"📁 Found solar data files:")
        logger.info(f"   Wind10: {solw}")
        logger.info(f"   Temperature: {sol2tmp}")
        logger.info(f"   Direct radiation: {solvdd}")
        logger.info(f"   Diffuse radiation: {solvbd}")

        if all([solw, sol2tmp, solvdd, solvbd]):
            try:
                solar_output_power = utils.prepare_calculate_solar_power(
                    weather_data[solw[0]][
                        list(
                            set(weather_data[solw[0]].columns).intersection(
                                set(solar_plants["pid"])
                            )
                        )
                    ],
                    weather_data[sol2tmp[0]][
                        list(
                            set(weather_data[solw[0]].columns).intersection(
                                set(solar_plants["pid"])
                            )
                        )
                    ],
                    weather_data[solvdd[0]][
                        list(
                            set(weather_data[solw[0]].columns).intersection(
                                set(solar_plants["pid"])
                            )
                        )
                    ],
                    weather_data[solvbd[0]][
                        list(
                            set(weather_data[solw[0]].columns).intersection(
                                set(solar_plants["pid"])
                            )
                        )
                    ],
                    solar_plants,
                    constants["TZ"],
                )
                logger.info("✅ Solar power calculations completed successfully")
            except Exception as e:
                logger.error(f"❌ Error in solar power calculations: {e}")
        else:
            logger.warning("⚠️ Missing required solar data files.")
            logger.info(f"Available weather data keys: {list(weather_data.keys())}")

    logger.info("🎉 Power calculations completed!")

    # For now, just return the data for further processing
    return (
        wind_farms,
        solar_plants,
        weather_data,
        existing_files if existing_files else saved_files,
        wind_output_power,
        solar_output_power,
    )


if __name__ == "__main__":
    result = main()
    # Handle different return types
    if len(result) == 6:
        (
            wind_farms,
            solar_plants,
            weather_data,
            saved_files,
            wind_output_power,
            solar_output_power,
        ) = result
    else:
        wind_farms, solar_plants, weather_data = result
        saved_files = []
        wind_output_power = None
        solar_output_power = None

    # Print summary
    print("\n" + "=" * 50)
    print("DATA LOADING SUMMARY")
    print("=" * 50)
    print(f"Wind farms: {len(wind_farms)}")
    print(f"Solar plants: {len(solar_plants)}")
    print(f"Weather data files: {len(weather_data)}")

    if weather_data:
        print("\nWeather data files:")
        for key, df in weather_data.items():
            print(f"  {key}: {df.shape[0]} rows, {df.shape[1]} columns")
            print(f"    Time range: {df.index.min()} to {df.index.max()}")
            print(f"    Sample locations: {list(df.columns[:3])}")
            print()

    print("\n" + "=" * 50)
    print("POWER CALCULATION SUMMARY")
    print("=" * 50)

    if wind_output_power is not None:
        print(f"✅ Wind power calculated successfully")
        print(f"   Output shape: {wind_output_power.shape}")
        print(
            f"   Time range: {wind_output_power.index.min()} to {wind_output_power.index.max()}"
        )
        print(f"   Wind farms: {len(wind_output_power.columns)}")
        print(f"   Max power: {wind_output_power.max().max():.2f} MW")
        print(f"   Mean power: {wind_output_power.mean().mean():.2f} MW")
    else:
        print("❌ Wind power calculation failed or not attempted")

    if solar_output_power is not None:
        print(f"✅ Solar power calculated successfully")
        print(f"   Output shape: {solar_output_power.shape}")
        print(
            f"   Time range: {solar_output_power.index.min()} to {solar_output_power.index.max()}"
        )
        print(f"   Solar plants: {len(solar_output_power.columns)}")
        print(f"   Max power: {solar_output_power.max().max():.2f} MW")
        print(f"   Mean power: {solar_output_power.mean().mean():.2f} MW")
    else:
        print("❌ Solar power calculation failed or not attempted")

    print("\n" + "=" * 50)
    print("CONSOLIDATED FILES SUMMARY")
    print("=" * 50)

    if isinstance(saved_files, list) and saved_files:
        for file_info in saved_files:
            if isinstance(file_info, dict):
                print(f"📁 {file_info.get('filename', 'Unknown')}")
                print(f"   Variable: {file_info.get('variable', 'Unknown')}")
                print(f"   Size: {file_info.get('size_mb', 0):.2f} MB")
                print(
                    f"   Dimensions: {file_info.get('rows', 0)} rows × {file_info.get('columns', 0)} columns"
                )
                print(f"   Time range: {file_info.get('time_range', 'Unknown')}")
                print()
    else:
        print("No consolidated files to display")
        print()
