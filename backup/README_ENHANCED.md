# Enhanced HRRR Data Extraction

This enhanced version of the HRRR data extraction package provides improved error handling, monitoring, validation, and user experience while maintaining full backward compatibility with your existing code.

## 🚀 **New Features**

### **1. Unified Configuration System**
- **Type-safe configuration** with validation
- **Auto-detection** of optimal settings
- **JSON/YAML support** for configuration files
- **Backward compatibility** with existing config.py

### **2. Enhanced Error Handling**
- **Comprehensive logging** with file and console output
- **Graceful error recovery** with detailed error messages
- **Input validation** for all files and parameters
- **System resource monitoring** (memory, disk space)

### **3. Performance Monitoring**
- **Real-time metrics** tracking
- **Memory usage monitoring** with automatic cleanup
- **Throughput measurement** (files/second, points/second)
- **CPU utilization tracking**

### **4. Command-Line Interface**
- **Easy-to-use CLI** for all operations
- **Interactive prompts** for confirmation
- **Configuration management** commands
- **Validation and testing** commands

## 📁 **New Files Created**

```
├── config_unified.py          # Unified configuration system
├── utils_enhanced.py          # Enhanced utility functions
├── hrrr_enhanced.py          # Enhanced main extraction script
├── cli.py                    # Command-line interface
├── test_enhanced.py          # Test suite for new features
├── requirements_enhanced.txt  # Enhanced dependencies
└── README_ENHANCED.md        # This documentation
```

## 🛠️ **Installation**

### **1. Install Enhanced Dependencies**
```bash
pip install -r requirements_enhanced.txt
```

### **2. Test Enhanced Features**
```bash
python test_enhanced.py
```

## 📖 **Usage**

### **Option 1: Command-Line Interface (Recommended)**

#### **Extract Specific Locations**
```bash
# Basic usage
python cli.py extract-specific --start-date 2023-01-01 --end-date 2023-01-02

# With custom options
python cli.py extract-specific \
    --wind-csv wind.csv \
    --solar-csv solar.csv \
    --start-date 2023-01-01 \
    --end-date 2023-01-02 \
    --output-dir ./my_output \
    --workers 16 \
    --chunk-size 200
```

#### **Extract Full Grid**
```bash
python cli.py extract-full-grid \
    --start-date 2023-01-01 \
    --end-date 2023-12-31 \
    --output-dir ./full_grid_output \
    --workers 32
```

#### **Validate System and Inputs**
```bash
python cli.py validate --wind-csv wind.csv --solar-csv solar.csv
```

#### **Show System Information**
```bash
python cli.py info
```

#### **Create Configuration File**
```bash
python cli.py create-config --output my_config.json
```

### **Option 2: Python Scripts**

#### **Enhanced Main Script**
```python
# Use the enhanced version instead of hrrr.py
from hrrr_enhanced import main_enhanced

# Run enhanced extraction
result = main_enhanced()
```

#### **Custom Configuration**
```python
from config_unified import HRRRConfig

# Create custom configuration
config = HRRRConfig(
    wind_csv_path="my_wind.csv",
    solar_csv_path="my_solar.csv",
    output_base_dir="./custom_output",
    num_workers=16,
    chunk_size=200
)

# Validate configuration
config.validate()

# Use in extraction
from hrrr_enhanced import extract_specific_locations_enhanced
result = extract_specific_locations_enhanced(config)
```

## 🔧 **Configuration**

### **Configuration File Format (JSON)**
```json
{
  "wind_csv_path": "wind.csv",
  "solar_csv_path": "solar.csv",
  "output_base_dir": "/research/alij/extracted_data",
  "num_workers": 16,
  "chunk_size": 100,
  "use_parallel": true,
  "enable_resume": true,
  "max_memory_gb": 50.0,
  "wind_selectors": {
    "UWind80": "u",
    "VWind80": "v"
  },
  "solar_selectors": {
    "rad": "dswrf",
    "vbd": "vbdsf",
    "vdd": "vddsf",
    "2tmp": "2t",
    "UWind10": "10u",
    "VWind10": "10v"
  }
}
```

### **Environment Variables**
```bash
export HRRR_WIND_CSV=wind.csv
export HRRR_SOLAR_CSV=solar.csv
export HRRR_OUTPUT_DIR=./output
export HRRR_WORKERS=16
```

## 📊 **Monitoring and Logging**

### **Log Files**
- `hrrr_extraction_enhanced.log` - Main extraction log
- `hrrr_extraction.log` - Original extraction log (preserved)

### **Performance Metrics**
The enhanced version tracks:
- **Processing time** per operation
- **Throughput** (files/second, points/second)
- **Memory usage** with automatic cleanup
- **CPU utilization** during processing
- **Error rates** and recovery statistics

### **Example Performance Output**
```
📈 Performance Summary:
   Total duration: 1h 23m 45s
   Avg throughput: 5.2 files/s
   Peak memory: 45.3 GB
   Avg CPU: 92.1%
```

## 🔍 **Validation Features**

### **Input Validation**
- **CSV file existence** and format checking
- **Required column validation** (pid, lat, lon)
- **Coordinate range validation** (lat: -90 to 90, lon: -180 to 180)
- **Date range validation** (start < end)

### **System Validation**
- **Disk space checking** before extraction
- **Memory availability** monitoring
- **Data directory validation** (GRIB files present)
- **System resource** availability

### **Example Validation Output**
```
🔍 Validating inputs...
✅ Validated wind.csv: 150 rows, 3 columns
✅ Validated solar.csv: 75 rows, 3 columns
✅ All input files validated successfully
💾 Checking disk space...
✅ Sufficient disk space: 500.2 GB available
```

## 🚨 **Error Handling**

### **Graceful Error Recovery**
- **File not found** errors with helpful suggestions
- **Memory pressure** handling with automatic cleanup
- **Network timeouts** with retry logic
- **Corrupted data** detection and reporting

### **Detailed Error Messages**
```
❌ Error during GRIB file reading on data/20230101/hrrr.t00z.wrfsubhf00.grib2
   Reason: File appears to be corrupted or incomplete
   Action: Skipping file and continuing with next
   Impact: 1 file skipped, 99.8% success rate maintained
```

## 🔄 **Backward Compatibility**

### **Existing Code Compatibility**
- **All existing scripts** continue to work unchanged
- **Original config.py** values are automatically imported
- **Existing output formats** are preserved
- **Original function signatures** remain unchanged

### **Migration Path**
```python
# Old way (still works)
from hrrr import extract_specific_locations
result = extract_specific_locations()

# New enhanced way (recommended)
from hrrr_enhanced import extract_specific_locations_enhanced
result = extract_specific_locations_enhanced()
```

## 🧪 **Testing**

### **Run All Tests**
```bash
python test_enhanced.py
```

### **Test Individual Components**
```python
# Test configuration
from config_unified import HRRRConfig
config = HRRRConfig()
config.validate()

# Test utilities
from utils_enhanced import check_memory_usage, format_duration
check_memory_usage()
format_duration(3661)  # Returns "1h 1m"

# Test CLI
import subprocess
subprocess.run(["python", "cli.py", "--help"])
```

## 📈 **Performance Improvements**

### **Memory Management**
- **Automatic garbage collection** when memory usage > 80%
- **Chunked processing** to prevent memory overflow
- **Memory monitoring** with real-time alerts

### **Parallel Processing**
- **Optimal worker count** auto-detection
- **Improved error handling** in parallel operations
- **Progress tracking** for long-running operations

### **I/O Optimization**
- **Batch file operations** for better throughput
- **Resume capability** for interrupted operations
- **Caching** of frequently accessed data

## 🎯 **Best Practices**

### **1. Use the CLI for Simple Operations**
```bash
# Quick extraction
python cli.py extract-specific --start-date 2023-01-01 --end-date 2023-01-02
```

### **2. Use Configuration Files for Complex Setups**
```bash
# Create config
python cli.py create-config --output my_config.json

# Edit config file
# Run with config
python cli.py --config my_config.json extract-specific --start-date 2023-01-01 --end-date 2023-01-02
```

### **3. Monitor Performance**
```python
from utils_enhanced import performance_monitor
summary = performance_monitor.get_summary()
print(f"Peak memory: {summary['peak_memory_gb']:.1f} GB")
```

### **4. Validate Before Running**
```bash
# Always validate first
python cli.py validate
```

## 🆘 **Troubleshooting**

### **Common Issues**

#### **1. Memory Errors**
```
❌ High memory usage: 85.2%
```
**Solution**: Reduce chunk size or number of workers
```bash
python cli.py extract-specific --chunk-size 50 --workers 8
```

#### **2. File Not Found**
```
❌ Wind CSV file not found: wind.csv
```
**Solution**: Check file path and create if needed
```bash
python cli.py info  # Check file status
```

#### **3. Disk Space**
```
❌ Low disk space: 5.2 GB available, 10.0 GB required
```
**Solution**: Free up disk space or change output directory
```bash
python cli.py extract-specific --output-dir /path/with/more/space
```

### **Getting Help**
```bash
# Show all available commands
python cli.py --help

# Show help for specific command
python cli.py extract-specific --help
```

## 📝 **Migration Guide**

### **From Original to Enhanced**

1. **Install enhanced dependencies**
   ```bash
   pip install -r requirements_enhanced.txt
   ```

2. **Test enhanced features**
   ```bash
   python test_enhanced.py
   ```

3. **Try the CLI**
   ```bash
   python cli.py info
   python cli.py validate
   ```

4. **Run enhanced extraction**
   ```bash
   python cli.py extract-specific --start-date 2023-01-01 --end-date 2023-01-02
   ```

5. **Monitor performance**
   - Check log files for detailed metrics
   - Use performance monitoring features
   - Validate inputs before running

## 🎉 **Benefits**

### **For Users**
- **Easier to use** with CLI interface
- **Better error messages** and troubleshooting
- **Performance monitoring** and optimization
- **Input validation** prevents common errors

### **For Developers**
- **Type-safe configuration** with validation
- **Comprehensive logging** for debugging
- **Modular design** for easy extension
- **Backward compatibility** with existing code

### **For Operations**
- **Resume capability** for long-running jobs
- **Resource monitoring** prevents system overload
- **Detailed metrics** for performance tuning
- **Robust error handling** reduces manual intervention

---

**Note**: All existing files remain unchanged. The enhanced features are provided through new files that can be used alongside or instead of the original code. 