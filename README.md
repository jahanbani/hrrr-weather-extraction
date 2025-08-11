# 🌤️ HRRR Weather Data Extraction System

A high-performance, parallelized system for extracting HRRR (High-Resolution Rapid Refresh) weather data for specific locations, regions, and full grids. Optimized for 36+ CPU cores and 256+ GB memory systems.

## 🚀 Features

### **Core Capabilities**
- **Specific Point Extraction**: Extract data for individual wind/solar locations
- **Region Extraction**: Extract data for defined geographic regions
- **Full Grid Extraction**: Process entire HRRR grids efficiently
- **Multi-Region Processing**: Handle multiple regions simultaneously

### **Performance Optimizations**
- **36+ CPU Core Support**: Automatic worker optimization
- **256+ GB Memory Support**: Intelligent memory management
- **Parallel Processing**: Multiprocessing for CPU-intensive operations
- **I/O Optimization**: Single-pass GRIB file reading
- **Memory Safety**: Automatic fallback and resource monitoring

### **Data Formats**
- **Input**: GRIB2 files (HRRR format)
- **Output**: Parquet files with compression
- **Variables**: Wind (U/V components), Solar radiation, Temperature, Pressure

## 🏗️ Architecture

### **Consolidated Modules**
```
extraction_core.py          # Core extraction functions
extraction_utils.py         # Utility functions
full_grid_extraction.py     # Full grid processing
prereise_essentials.py      # Essential prereise functions
config_unified.py           # Unified configuration
hrrr_enhanced.py           # Main execution script
```

### **Key Functions**
- `extract_specific_points_daily_single_pass()` - Point extraction with multiprocessing
- `extract_region_data_quarterly()` - Region extraction
- `extract_multiple_regions_quarterly()` - Multi-region processing
- `extract_full_grid_day_by_day()` - Full grid extraction

## 🚀 Quick Start

### **1. Installation**
```bash
# Clone the repository
git clone <your-repo-url>
cd Weather

# Install dependencies
pip install -r requirements.txt
```

### **2. Configuration**
```python
# Update config_unified.py with your paths
DATADIR = "/path/to/your/grib/files"
output_base_dir = "/path/to/output"
```

### **3. Run Extraction**
```python
# Run the main script
python hrrr_enhanced.py
```

## 📊 Usage Examples

### **Specific Point Extraction**
```python
from extraction_core import extract_specific_points_daily_single_pass

result = extract_specific_points_daily_single_pass(
    wind_csv_path="wind.csv",
    solar_csv_path="solar.csv",
    START=datetime(2019, 1, 1),
    END=datetime(2019, 1, 31),
    DATADIR="/path/to/grib/data",
    use_parallel=True,        # Enable multiprocessing
    num_workers=None,         # Auto-optimize for your system
    compression="snappy"
)
```

### **Region Extraction**
```python
from extraction_core import extract_region_data_quarterly

result = extract_region_data_quarterly(
    region_bounds={
        "lat_min": 30.0,
        "lat_max": 47.0,
        "lon_min": -106.0,
        "lon_max": -88.0
    },
    START=datetime(2019, 1, 1),
    END=datetime(2019, 1, 31),
    DATADIR="/path/to/grib/data",
    output_dir="/path/to/output"
)
```

## ⚙️ System Requirements

### **Hardware**
- **CPU**: 16+ cores (optimized for 36+ cores)
- **Memory**: 64+ GB (optimized for 256+ GB)
- **Storage**: Fast SSD/NVMe for I/O operations

### **Software**
- **Python**: 3.8+
- **OS**: Linux (tested on RHEL/CentOS 9)
- **Dependencies**: See requirements.txt

## 🔧 Configuration

### **Environment Variables**
```bash
export HRRR_DATA_DIR="/path/to/grib/files"
export OUTPUT_BASE_DIR="/path/to/output"
export NUM_WORKERS=32  # Optional: override auto-detection
```

### **Configuration File**
```python
# config_unified.py
class HRRRConfig:
    DATADIR = "/research/alij/hrrr"
    output_base_dir = "/research/alij/extracted_data"
    num_workers = None  # Auto-detect
    max_memory_gb = 200.0  # Use 200GB out of 256GB
```

## 📈 Performance

### **Expected Speedup**
| System Type | CPUs | Memory | Workers | Speedup |
|-------------|------|--------|---------|---------|
| **High-Performance** | 36+ | 200+ GB | 32 | **~32x** |
| Medium | 24+ | 150+ GB | 20 | ~20x |
| Standard | 16+ | 100+ GB | 12 | ~12x |
| Conservative | Any | Any | 8 | ~8x |

### **Memory Management**
- **Per Worker**: 8 GB (256 GB ÷ 32 workers)
- **Safety Margin**: 4 CPUs reserved for system
- **Auto-Optimization**: Intelligent resource allocation

## 🐛 Troubleshooting

### **Common Issues**

#### **1. No GRIB Files Found**
```bash
# Check data directory structure
ls -la /path/to/grib/files/
# Should see: 20190101/, 20190102/, etc.
```

#### **2. Memory Issues**
```python
# Reduce worker count
num_workers = 16  # Instead of auto-detection
```

#### **3. Import Errors**
```bash
# Check dependencies
pip install -r requirements.txt
```

### **Debug Mode**
```python
# Enable verbose logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 API Reference

### **Core Functions**

#### **`extract_specific_points_daily_single_pass()`**
Extract data for specific wind and solar locations.

**Parameters:**
- `wind_csv_path`: Path to wind locations CSV
- `solar_csv_path`: Path to solar locations CSV
- `START`: Start datetime
- `END`: End datetime
- `DATADIR`: GRIB data directory
- `use_parallel`: Enable multiprocessing
- `num_workers`: Number of workers (None = auto-detect)

**Returns:**
- Dictionary with extraction results and statistics

#### **`extract_region_data_quarterly()`**
Extract data for a defined geographic region.

**Parameters:**
- `region_bounds`: Dictionary with lat/lon bounds
- `START`: Start datetime
- `END`: End datetime
- `DATADIR`: GRIB data directory
- `output_dir`: Output directory path

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **HRRR Data**: NOAA/ESRL for providing HRRR data
- **Prereise**: Original extraction framework
- **Community**: Contributors and users of the system

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review existing issues
3. Create a new issue with detailed information

---

**Built with ❤️ for high-performance weather data processing**
