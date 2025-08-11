# Expected Output Format

This document shows the expected output format for the HRRR data extraction.

## 📊 Output Structure

After running the extraction, you'll get **2 main files**:

```
extracted_data/
├── wind/
│   └── wind_data_20230101_to_20230101.parquet
└── solar/
    └── solar_data_20230101_to_20230101.parquet
```

## 📈 Data Format

### Wind Data File (`wind_data_20230101_to_20230101.parquet`)

**Rows**: Timestamps at 15-minute intervals
**Columns**: All wind variables for all wind locations

| time                | wind_001_UWind80 | wind_001_VWind80 | wind_001_WindSpeed80 | wind_002_UWind80 | wind_002_VWind80 | wind_002_WindSpeed80 |
|---------------------|------------------|------------------|---------------------|------------------|------------------|---------------------|
| 2023-01-01 00:00:00| 5.2              | -3.1             | 6.1                | 4.8              | 2.3              | 5.3                |
| 2023-01-01 00:15:00| 5.5              | -2.9             | 6.2                | 4.9              | 2.1              | 5.3                |
| 2023-01-01 00:30:00| 5.8              | -3.3             | 6.7                | 5.1              | 2.5              | 5.6                |
| 2023-01-01 00:45:00| 6.1              | -3.0             | 6.8                | 5.3              | 2.2              | 5.7                |
| 2023-01-01 01:00:00| 6.3              | -2.8             | 6.9                | 5.5              | 2.0              | 5.9                |

### Solar Data File (`solar_data_20230101_to_20230101.parquet`)

**Rows**: Timestamps at 15-minute intervals  
**Columns**: All solar variables for all solar locations

| time                | solar_001_rad | solar_001_vbd | solar_001_vdd | solar_001_2tmp | solar_002_rad | solar_002_vbd | solar_002_vdd | solar_002_2tmp |
|---------------------|---------------|---------------|---------------|----------------|---------------|---------------|---------------|----------------|
| 2023-01-01 00:00:00| 0.0            | 0.0            | 0.0            | 2.1            | 0.0            | 0.0            | 0.0            | 1.8            |
| 2023-01-01 00:15:00| 0.0            | 0.0            | 0.0            | 2.0            | 0.0            | 0.0            | 0.0            | 1.7            |
| 2023-01-01 00:30:00| 0.0            | 0.0            | 0.0            | 1.9            | 0.0            | 0.0            | 0.0            | 1.6            |
| 2023-01-01 00:45:00| 0.0            | 0.0            | 0.0            | 1.8            | 0.0            | 0.0            | 0.0            | 1.5            |
| 2023-01-01 01:00:00| 0.0            | 0.0            | 0.0            | 1.7            | 0.0            | 0.0            | 0.0            | 1.4            |

## 🏷️ Column Naming Convention

### Wind Variables
- `{location}_{variable}` format
- **UWind80**: U-component of wind at 80m height
- **VWind80**: V-component of wind at 80m height  
- **WindSpeed80**: Calculated wind speed at 80m height (√(U² + V²))
- **UWind10**: U-component of wind at 10m height
- **VWind10**: V-component of wind at 10m height
- **WindSpeed10**: Calculated wind speed at 10m height

### Solar Variables
- `{location}_{variable}` format
- **rad**: Downward short-wave radiation flux
- **vbd**: Visible Beam Downward Solar Flux
- **vdd**: Visible Diffuse Downward Solar Flux
- **2tmp**: 2 metre temperature

## ⏰ Temporal Resolution

- **f00 files**: Top of the hour data (:00)
- **f01 files**: Subhourly data (:15, :30, :45)
- **Total**: 4 timestamps per hour, 96 timestamps per day

## 📍 Location Mapping

The column names correspond to your CSV file locations:

**wind.csv:**
```csv
pid,lat,lon
wind_001,40.7128,-74.0060
wind_002,34.0522,-118.2437
```

**solar.csv:**
```csv
pid,lat,lon
solar_001,36.7783,-119.4179
solar_002,31.9686,-99.9018
```

## 🔍 Reading the Data

```python
import pandas as pd

# Read wind data
wind_df = pd.read_parquet("extracted_data/wind/wind_data_20230101_to_20230101.parquet")
print(f"Wind data shape: {wind_df.shape}")
print(f"Wind data columns: {list(wind_df.columns)}")

# Read solar data  
solar_df = pd.read_parquet("extracted_data/solar/solar_data_20230101_to_20230101.parquet")
print(f"Solar data shape: {solar_df.shape}")
print(f"Solar data columns: {list(solar_df.columns)}")

# Get data for specific location
wind_001_speed = wind_df['wind_001_WindSpeed80']
solar_001_radiation = solar_df['solar_001_rad']
```

## ✅ Benefits of This Format

1. **Single File per Data Type**: Easy to work with
2. **All Variables Together**: No need to merge multiple files
3. **Time Series Ready**: Perfect for analysis and plotting
4. **Location-Based Columns**: Easy to filter by location
5. **Variable-Based Columns**: Easy to filter by variable type 