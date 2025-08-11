# Optimized Specific Points Extraction

This module provides an optimized function to extract HRRR (High-Resolution Rapid Refresh) weather data for specific wind and solar locations. Unlike the full grid extraction, this function only extracts data for the closest grid points to your specified locations, significantly reducing processing time and memory usage.

## Key Features

- **Efficient Point Extraction**: Only extracts data for the closest grid points to your wind and solar locations
- **Memory Optimized**: Uses float32 data types and batch processing to minimize memory usage
- **Parallel Processing**: Supports multi-threaded processing for faster extraction
- **Resume Capability**: Can resume from interruptions
- **15-minute Resolution**: Handles both f00 and f01 GRIB files for 15-minute temporal resolution
- **Subhourly Extraction**: Extracts data at :00 (f00), :15, :30, :45 (f01) timestamps
- **Auto-Detection**: Automatically detects data directory based on operating system
- **Derived Calculations**: Automatically calculates wind speeds from U and V components
- **Flexible Output**: Separate output directories for wind and solar data

## Data Directory Auto-Detection

The function automatically detects the appropriate data directory based on your operating system:

- **Windows**: `data/YYYYMMDD`
- **Linux**: `/research/alij/hrrr/YYYYMMDD`

## File Structure

```
├── extract_specific_points_optimized.py  # Main extraction function
├── test_specific_points.py               # Test script
├── example_wind.csv                      # Example wind points CSV
├── example_solar.csv                     # Example solar points CSV
└── README_specific_points.md            # This file
```

## CSV File Format

Your wind and solar CSV files should have the following format:

```csv
pid,lat,lon
wind_001,40.7128,-74.0060
wind_002,34.0522,-118.2437
wind_003,41.8781,-87.6298
...
```

**Required columns:**
- `pid`: Unique identifier for each point
- `lat`: Latitude in decimal degrees
- `lon`: Longitude in decimal degrees

## Usage

### Basic Usage

```python
from extract_specific_points_optimized import extract_specific_points_optimized
import datetime

# Define parameters
wind_csv_path = "wind.csv"
solar_csv_path = "solar.csv"
START = datetime.datetime(2023, 1, 1, 0, 0, 0)
END = datetime.datetime(2023, 1, 1, 23, 0, 0)

# Run extraction (auto-detects data directory)
result = extract_specific_points_optimized(
    wind_csv_path=wind_csv_path,
    solar_csv_path=solar_csv_path,
    START=START,
    END=END,
    DATADIR=None,  # Auto-detect based on OS
    wind_output_dir="./wind_extracted",
    solar_output_dir="./solar_extracted"
)

if result:
    print(f"✅ Extraction completed in {result['processing_time_seconds']:.1f} seconds")
    print(f"Wind points: {result['wind_points']}")
    print(f"Solar points: {result['solar_points']}")
```

### Advanced Usage with Custom Variables

```python
# Custom wind variables
wind_selectors = {
    "UWind80": "U component of wind",
    "VWind80": "V component of wind",
    "UWind10": "10 metre U wind component",
    "VWind10": "10 metre V wind component",
}

# Custom solar variables
solar_selectors = {
    "rad": "Downward short-wave radiation flux",
    "vbd": "Visible Beam Downward Solar Flux",
    "vdd": "Visible Diffuse Downward Solar Flux",
    "2tmp": "2 metre temperature",
}

result = extract_specific_points_optimized(
    wind_csv_path=wind_csv_path,
    solar_csv_path=solar_csv_path,
    START=START,
    END=END,
    DATADIR=None,  # Auto-detect based on OS
    wind_selectors=wind_selectors,
    solar_selectors=solar_selectors,
    wind_output_dir="./wind_extracted",
    solar_output_dir="./solar_extracted",
    use_parallel=True,
    num_workers=8,
    max_file_groups=10000,
    enable_resume=True
)
```

## Function Parameters

### Required Parameters

- `wind_csv_path` (str): Path to wind points CSV file
- `solar_csv_path` (str): Path to solar points CSV file
- `START` (datetime): Start datetime for extraction
- `END` (datetime): End datetime for extraction

### Optional Parameters

- `DATADIR` (str): Directory containing GRIB files (auto-detect if `None`)
- `DEFAULT_HOURS_FORECASTED` (list): Forecast hours to process (default: `["0", "1"]` for f00 and f01)
- `wind_selectors` (dict): Wind variables to extract (default: standard wind variables)
- `solar_selectors` (dict): Solar variables to extract (default: standard solar variables)
- `wind_output_dir` (str): Output directory for wind data (default: `"./wind_extracted"`)
- `solar_output_dir` (str): Output directory for solar data (default: `"./solar_extracted"`)
- `compression` (str): Parquet compression (default: `"snappy"`)
- `use_parallel` (bool): Enable parallel processing (default: `True`)
- `num_workers` (int): Number of workers (auto-detect if `None`)
- `max_file_groups` (int): Maximum file groups to process (default: `10000`)
- `enable_resume` (bool): Enable resume functionality (default: `True`)

## Data Extraction Details

### Temporal Resolution

The function extracts data at 15-minute intervals:

- **f00 files**: Top of the hour data (:00)
- **f01 files**: Subhourly data (:15, :30, :45)

This provides 4 timestamps per hour, or 96 timestamps per day.

### Expected Data Structure

For each hour, you should expect:
- 1 timestamp from f00 files (top of the hour)
- 3 timestamps from f01 files (15, 30, 45 minutes past the hour)

## Output Structure

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

### File Naming Convention

**Wind variable files:**
- `UWind80_YYYYMMDD_to_YYYYMMDD.parquet` - U component of wind at 80m
- `VWind80_YYYYMMDD_to_YYYYMMDD.parquet` - V component of wind at 80m
- `UWind10_YYYYMMDD_to_YYYYMMDD.parquet` - U component of wind at 10m
- `VWind10_YYYYMMDD_to_YYYYMMDD.parquet` - V component of wind at 10m
- `WindSpeed80_YYYYMMDD_to_YYYYMMDD.parquet` - Calculated wind speed at 80m
- `WindSpeed10_YYYYMMDD_to_YYYYMMDD.parquet` - Calculated wind speed at 10m

**Solar variable files:**
- `rad_YYYYMMDD_to_YYYYMMDD.parquet` - Downward short-wave radiation flux
- `vbd_YYYYMMDD_to_YYYYMMDD.parquet` - Visible Beam Downward Solar Flux
- `vdd_YYYYMMDD_to_YYYYMMDD.parquet` - Visible Diffuse Downward Solar Flux
- `2tmp_YYYYMMDD_to_YYYYMMDD.parquet` - 2 metre temperature
- `UWind10_YYYYMMDD_to_YYYYMMDD.parquet` - 10 metre U wind component
- `VWind10_YYYYMMDD_to_YYYYMMDD.parquet` - 10 metre V wind component

## Performance Optimizations

### Memory Management
- Uses `float32` instead of `float64` to reduce memory usage by 50%
- Implements batch processing for large point sets
- Aggressive garbage collection between operations

### Processing Efficiency
- Only reads the closest grid points to your locations
- Processes variables sequentially to avoid memory conflicts
- Uses KDTree for efficient nearest neighbor search

### Parallel Processing
- Auto-detects optimal worker count based on system capabilities
- Processes file groups in parallel
- Conservative defaults to prevent memory issues

## Comparison with Full Grid Extraction

| Feature | Full Grid | Specific Points |
|---------|-----------|-----------------|
| Memory Usage | High (3.75M points) | Low (only your points) |
| Processing Time | Hours/Days | Minutes/Hours |
| Storage Size | GBs/TBs | MBs/GBs |
| Use Case | Research/Archive | Production/Analysis |

## Error Handling

The function includes comprehensive error handling:
- Validates CSV file format and required columns
- Checks GRIB file availability
- Handles memory errors with fallback processing
- Continues processing even if individual files fail
- Provides detailed progress and error reporting

## Testing

Run the test script to verify the function works with your setup:

```bash
python test_specific_points.py
```

The test script will:
1. Create sample CSV files if they don't exist
2. Run a small extraction test (2 hours)
3. Report results and timing
4. Verify data extraction from f00 and f01 files

## Dependencies

Required Python packages:
- `numpy`
- `pandas`
- `pygrib`
- `scipy`
- `tqdm`
- `powersimdata`

Install with:
```bash
pip install numpy pandas pygrib scipy tqdm powersimdata
```

## Troubleshooting

### Common Issues

1. **CSV file not found**: Ensure your CSV files exist and have the correct format
2. **GRIB data not found**: The function will auto-detect the data directory, but ensure your data follows the expected structure
3. **Memory errors**: Reduce `num_workers` or `max_file_groups`
4. **Slow processing**: Increase `num_workers` if you have sufficient memory

### Performance Tips

1. **For large datasets**: Process in smaller time periods
2. **For many points**: Use batch processing (function handles this automatically)
3. **For high-performance systems**: Increase `num_workers` and `max_file_groups`
4. **For memory-constrained systems**: Use conservative settings

## Example Results

```
🚀 OPTIMIZED SPECIFIC POINTS EXTRACTION
============================================================
Date range: 2023-01-01 to 2023-01-01
Wind CSV: wind.csv
Solar CSV: solar.csv
Data directory: /research/alij/hrrr
Output directories: ./wind_extracted, ./solar_extracted
Forecast hours: ['0', '1'] (f00 and f01 only)

🎯 Using settings:
   Workers: 8
   Max file groups: 10000
   Parallel processing: True
   Resume enabled: True

📁 Loading point data from CSV files...
✅ Loaded 100 wind points from wind.csv
✅ Loaded 50 solar points from solar.csv
Point data loaded in 0.02s

📊 Extracting grid metadata...
Grid dimensions: 1059 x 1799 = 1,905,141 total points
Grid metadata extracted in 1.23s

🎯 Finding closest grid points...
Wind points: 100 indices
Solar points: 50 indices
Closest points found in 0.15s

📋 Variables to extract:
   Wind variables: ['UWind80', 'VWind80', 'UWind10', 'VWind10']
   Solar variables: ['rad', 'vbd', 'vdd', '2tmp']
   Total variables: 8

🔍 Discovering GRIB files...
  Found 48 valid files in 20230101
Found 48 total GRIB files
File discovery completed in 0.05s

📦 Grouping files by time...
Processing 24 file groups
File grouping completed in 0.01s

📁 Creating output directories...

🚀 Processing variables...
Processing variable: UWind80
Processing variable: VWind80
Processing variable: UWind10
Processing variable: VWind10
Processing variable: rad
Processing variable: vbd
Processing variable: vdd
Processing variable: 2tmp
Variable processing completed in 45.67s

💾 Saving results...
  Saved wind UWind80: 96 timestamps, 100 points
  Saved wind VWind80: 96 timestamps, 100 points
  Saved wind UWind10: 96 timestamps, 100 points
  Saved wind VWind10: 96 timestamps, 100 points
  Saved solar rad: 96 timestamps, 50 points
  Saved solar vbd: 96 timestamps, 50 points
  Saved solar vdd: 96 timestamps, 50 points
  Saved solar 2tmp: 96 timestamps, 50 points
Results saved in 2.34s

🌪️  Calculating derived wind speeds...
  Calculated WindSpeed80
  Calculated WindSpeed10
Derived calculations completed in 0.12s

============================================================
📊 EXTRACTION SUMMARY
============================================================
Total processing time: 48.1s (0.8 minutes)
Wind points processed: 100
Solar points processed: 50
Variables extracted: 8
Wind variables: ['UWind80', 'VWind80', 'UWind10', 'VWind10']
Solar variables: ['rad', 'vbd', 'vdd', '2tmp']
Derived variables: ['WindSpeed80', 'WindSpeed10']
Output directories:
  Wind: ./wind_extracted
  Solar: ./solar_extracted
Files created:
  Wind files: 6
  Solar files: 4
```

This optimized function provides a much more efficient way to extract HRRR data for specific locations compared to processing the entire grid, with automatic data directory detection and proper subhourly extraction from f00 and f01 files. 