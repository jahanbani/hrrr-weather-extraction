# Configuration Guide

This guide explains how to configure the HRRR data extraction system using the centralized configuration file.

## 📁 Configuration File

All settings are now centralized in `config.py`. This makes it easy to change output directories, processing parameters, and other settings in one place.

## 🔧 Quick Configuration

### Output Directories

To change where your data is saved, modify these lines in `config.py`:

```python
# Base output directory for all extracted data
OUTPUT_BASE_DIR = "./extracted_data"

# Subdirectories for different types of data
WIND_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "wind")
SOLAR_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "solar")
FULL_GRID_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "full_grid")
```

**Examples:**
- `OUTPUT_BASE_DIR = "./my_data"` - Save to `./my_data/`
- `OUTPUT_BASE_DIR = "/path/to/data"` - Save to absolute path
- `OUTPUT_BASE_DIR = "C:/Users/username/data"` - Windows path

### Processing Parameters

Adjust processing settings:

```python
# Default number of workers for parallel processing
DEFAULT_NUM_WORKERS = None  # Auto-detect if None

# Default maximum file groups to process
DEFAULT_MAX_FILE_GROUPS = 10000

# Enable parallel processing by default
DEFAULT_USE_PARALLEL = True

# Enable resume functionality by default
DEFAULT_ENABLE_RESUME = True
```

### Variables to Extract

Modify which variables are extracted:

```python
# Default wind variables to extract
DEFAULT_WIND_SELECTORS = {
    "UWind80": "U component of wind",
    "VWind80": "V component of wind",
}

# Default solar variables to extract
DEFAULT_SOLAR_SELECTORS = {
    "rad": "Downward short-wave radiation flux",
    "vbd": "Visible Beam Downward Solar Flux",
    "vdd": "Visible Diffuse Downward Solar Flux",
    "2tmp": "2 metre temperature",
    "UWind10": "10 metre U wind component",
    "VWind10": "10 metre V wind component",
}
```

## 📊 Output Structure

With the default configuration, your data will be organized as:

```
extracted_data/
├── wind/
│   └── wind_data_YYYYMMDD_to_YYYYMMDD.parquet
├── solar/
│   └── solar_data_YYYYMMDD_to_YYYYMMDD.parquet
└── full_grid/
    └── [full grid data files]
```

## 🚀 Usage Examples

### Change Output Directory

```python
# In config.py
OUTPUT_BASE_DIR = "./my_custom_data"
```

### Change Processing Parameters

```python
# In config.py
DEFAULT_NUM_WORKERS = 8  # Use exactly 8 workers
DEFAULT_MAX_FILE_GROUPS = 5000  # Process fewer files
```

### Add New Variables

```python
# In config.py
DEFAULT_WIND_SELECTORS = {
    "UWind80": "U component of wind",
    "VWind80": "V component of wind",
    "UWind10": "10 metre U wind component",  # Added
    "VWind10": "10 metre V wind component",  # Added
}
```

## 🔄 Files Using Configuration

The following files now use the centralized configuration:

- `hrrr.py` - Main extraction script
- `extract_specific_points_optimized.py` - Optimized extraction function
- `test_specific_points.py` - Test script

## ✅ Benefits

1. **Single Point of Control**: Change settings in one place
2. **Consistency**: All scripts use the same settings
3. **Easy Maintenance**: No need to update multiple files
4. **Flexibility**: Easy to switch between different configurations

## 🎯 Quick Start

1. Edit `config.py` to set your desired output directory
2. Run your extraction scripts
3. Data will be saved to your configured location

## 📝 Example Configuration

Here's a complete example configuration:

```python
# Output directories
OUTPUT_BASE_DIR = "./my_hrrr_data"
WIND_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "wind")
SOLAR_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "solar")

# Processing settings
DEFAULT_NUM_WORKERS = 4
DEFAULT_MAX_FILE_GROUPS = 5000
DEFAULT_USE_PARALLEL = True

# Variables to extract
DEFAULT_WIND_SELECTORS = {
    "UWind80": "U component of wind",
    "VWind80": "V component of wind",
}

DEFAULT_SOLAR_SELECTORS = {
    "rad": "Downward short-wave radiation flux",
    "2tmp": "2 metre temperature",
}
```

This configuration will:
- Save data to `./my_hrrr_data/`
- Use 4 workers for processing
- Extract only wind U/V components and solar radiation/temperature
- Process up to 5000 file groups 