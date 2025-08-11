# Expected Output Format

This document describes the expected output format for the `extract_specific_points_optimized` function.

## Folder Structure

The function creates the following directory structure:

```
extracted_data/
├── wind/
│   ├── UWind80/
│   │   └── 20230101_to_20230101.parquet
│   ├── VWind80/
│   │   └── 20230101_to_20230101.parquet
│   ├── UWind10/
│   │   └── 20230101_to_20230101.parquet
│   ├── VWind10/
│   │   └── 20230101_to_20230101.parquet
│   ├── WindSpeed80/
│   │   └── 20230101_to_20230101.parquet
│   └── WindSpeed10/
│       └── 20230101_to_20230101.parquet
└── solar/
    ├── rad/
    │   └── 20230101_to_20230101.parquet
    ├── vbd/
    │   └── 20230101_to_20230101.parquet
    ├── vdd/
    │   └── 20230101_to_20230101.parquet
    ├── 2tmp/
    │   └── 20230101_to_20230101.parquet
    ├── UWind10/
    │   └── 20230101_to_20230101.parquet
    └── VWind10/
        └── 20230101_to_20230101.parquet
```

## Data Format

Each Parquet file contains:
- **Index**: Timestamp (15-minute resolution: :00, :15, :30, :45)
- **Columns**: Location data with format `{type}_{pid}` (e.g., `wind_001`, `solar_002`)
- **Values**: Variable values for each location at each timestamp

## Example Data Tables

### Wind Variable Files

**UWind80_20230101_to_20230101.parquet:**
```
                     wind_001  wind_002  wind_003
2023-01-01 00:00:00    5.2      3.8      4.1
2023-01-01 00:15:00    5.5      4.0      4.3
2023-01-01 00:30:00    5.8      4.2      4.5
2023-01-01 00:45:00    6.1      4.4      4.7
2023-01-01 01:00:00    6.4      4.6      4.9
```

**VWind80_20230101_to_20230101.parquet:**
```
                     wind_001  wind_002  wind_003
2023-01-01 00:00:00    2.1      1.8      2.3
2023-01-01 00:15:00    2.3      2.0      2.5
2023-01-01 00:30:00    2.5      2.2      2.7
2023-01-01 00:45:00    2.7      2.4      2.9
2023-01-01 01:00:00    2.9      2.6      3.1
```

**WindSpeed80_20230101_to_20230101.parquet:**
```
                     wind_001  wind_002  wind_003
2023-01-01 00:00:00    5.6      4.2      4.7
2023-01-01 00:15:00    6.0      4.5      5.0
2023-01-01 00:30:00    6.3      4.7      5.3
2023-01-01 00:45:00    6.7      5.0      5.6
2023-01-01 01:00:00    7.0      5.3      5.9
```

### Solar Variable Files

**rad_20230101_to_20230101.parquet:**
```
                     solar_001  solar_002
2023-01-01 00:00:00      0.0       0.0
2023-01-01 00:15:00      0.0       0.0
2023-01-01 00:30:00      0.0       0.0
2023-01-01 00:45:00      0.0       0.0
2023-01-01 01:00:00      0.0       0.0
```

**2tmp_20230101_to_20230101.parquet:**
```
                     solar_001  solar_002
2023-01-01 00:00:00     15.2      14.8
2023-01-01 00:15:00     15.4      15.0
2023-01-01 00:30:00     15.6      15.2
2023-01-01 00:45:00     15.8      15.4
2023-01-01 01:00:00     16.0      15.6
```

## File Naming Convention

### Wind Variables
- `UWind80_YYYYMMDD_to_YYYYMMDD.parquet` - U component of wind at 80m
- `VWind80_YYYYMMDD_to_YYYYMMDD.parquet` - V component of wind at 80m
- `UWind10_YYYYMMDD_to_YYYYMMDD.parquet` - U component of wind at 10m
- `VWind10_YYYYMMDD_to_YYYYMMDD.parquet` - V component of wind at 10m
- `WindSpeed80_YYYYMMDD_to_YYYYMMDD.parquet` - Calculated wind speed at 80m
- `WindSpeed10_YYYYMMDD_to_YYYYMMDD.parquet` - Calculated wind speed at 10m

### Solar Variables
- `rad_YYYYMMDD_to_YYYYMMDD.parquet` - Downward short-wave radiation flux
- `vbd_YYYYMMDD_to_YYYYMMDD.parquet` - Visible Beam Downward Solar Flux
- `vdd_YYYYMMDD_to_YYYYMMDD.parquet` - Visible Diffuse Downward Solar Flux
- `2tmp_YYYYMMDD_to_YYYYMMDD.parquet` - 2 metre temperature
- `UWind10_YYYYMMDD_to_YYYYMMDD.parquet` - 10 metre U wind component
- `VWind10_YYYYMMDD_to_YYYYMMDD.parquet` - 10 metre V wind component

## Temporal Resolution

- **f00 files**: Extract top-of-the-hour data (:00)
- **f01 files**: Extract subhourly data (:15, :30, :45)
- **Total resolution**: 15-minute intervals

## Reading the Data

```python
import pandas as pd

# Read a specific variable file
wind_speed_data = pd.read_parquet('extracted_data/wind/WindSpeed80_20230101_to_20230101.parquet')
solar_rad_data = pd.read_parquet('extracted_data/solar/rad_20230101_to_20230101.parquet')

# Access data for a specific location
wind_speed_location_1 = wind_speed_data['wind_001']
solar_rad_location_1 = solar_rad_data['solar_001']

# Access data for a specific timestamp
wind_speed_at_time = wind_speed_data.loc['2023-01-01 00:30:00']
solar_rad_at_time = solar_rad_data.loc['2023-01-01 00:30:00']
```

## Key Features

1. **Separate files per variable**: Each weather variable is saved in its own file
2. **Time as rows**: Each row represents a timestamp (15-minute intervals)
3. **Locations as columns**: Each column represents a specific location
4. **Consistent naming**: Files follow `{variable}_{start_date}_to_{end_date}.parquet` format
5. **Derived variables**: Wind speeds are calculated and saved as separate files 