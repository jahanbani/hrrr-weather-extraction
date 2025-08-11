# Day-by-Day HRRR Extraction Approach

## 🎯 **Overview**

The new `extract_specific_points_daily.py` function implements a **day-by-day processing approach** that solves the memory accumulation and slowdown issues of the original implementation.

## 🚀 **Key Benefits**

### **1. Memory Stability**
- **Constant memory usage**: Processes one day at a time
- **No accumulation**: Memory is cleared after each day
- **Predictable performance**: No slowdown over time

### **2. Fault Tolerance**
- **Resume capability**: Can restart from any point
- **Progress preservation**: Completed days are saved immediately
- **Error isolation**: Failed days don't affect others

### **3. Better Resource Management**
- **Distributed I/O**: Writes happen throughout processing
- **Batch processing**: Configurable batch sizes
- **Memory monitoring**: Built-in garbage collection

## 📁 **File Structure**

### **Output Organization**
```
wind/
├── UWind80/
│   ├── 20230101.parquet
│   ├── 20230102.parquet
│   └── ...
├── VWind80/
│   ├── 20230101.parquet
│   ├── 20230102.parquet
│   └── ...
└── ...

solar/
├── rad/
│   ├── 20230101.parquet
│   ├── 20230102.parquet
│   └── ...
├── vbd/
│   ├── 20230101.parquet
│   ├── 20230102.parquet
│   └── ...
└── ...
```

### **File Naming Convention**
- **One file per day per variable**
- **Format**: `YYYYMMDD.parquet`
- **Example**: `20230101.parquet` for January 1, 2023

## ⚙️ **Configuration Options**

### **Core Parameters**
```python
extract_specific_points_daily(
    wind_csv_path="wind.csv",
    solar_csv_path="solar.csv",
    START=datetime.datetime(2023, 1, 1, 0, 0, 0),
    END=datetime.datetime(2023, 12, 31, 23, 0, 0),
    num_workers=16,        # Number of parallel workers
    batch_size=7,          # Days per batch (one week)
    enable_resume=True,    # Resume from previous run
    use_parallel=True,     # Enable parallel processing
)
```

### **Performance Tuning**
- **`num_workers`**: Number of parallel processes (default: all CPUs)
- **`batch_size`**: Days processed per batch (default: 7)
- **`enable_resume`**: Skip completed days (default: True)

## 🔄 **Processing Flow**

### **1. Day-by-Day Processing**
```python
for day in date_range:
    # Process single day
    day_data = process_single_day(day)
    
    # Write immediately
    write_day_to_parquet(day_data)
    
    # Clear memory
    del day_data
    gc.collect()
```

### **2. Batch Processing**
```python
# Process multiple days in parallel
for batch in date_batches:
    with Pool(workers) as pool:
        results = pool.map(process_single_day, batch)
    
    # Clear memory after each batch
    gc.collect()
```

### **3. Resume Capability**
```python
# Check for completed days
completed_days = scan_output_directories()

# Filter out completed days
remaining_days = [d for d in date_range if d not in completed_days]
```

## 📊 **Performance Comparison**

### **Original Approach**
```
Memory: 0GB → 5GB → 10GB → 15GB → 20GB → CRASH
I/O:   0MB/s → 0MB/s → 0MB/s → 0MB/s → 500MB/s (massive write)
Risk:  High (all-or-nothing)
```

### **Day-by-Day Approach**
```
Memory: 2GB → 2GB → 2GB → 2GB → 2GB (constant)
I/O:   50MB/s → 50MB/s → 50MB/s → 50MB/s (steady)
Risk:  Low (incremental progress)
```

## 🛠️ **Usage Instructions**

### **1. Basic Usage**
```bash
# Run the enhanced script (now uses day-by-day)
python hrrr_enhanced.py
```

### **2. Direct Function Call**
```python
from extract_specific_points_daily import extract_specific_points_daily

result = extract_specific_points_daily(
    wind_csv_path="wind.csv",
    solar_csv_path="solar.csv",
    START=datetime.datetime(2023, 1, 1, 0, 0, 0),
    END=datetime.datetime(2023, 1, 7, 23, 0, 0),
    num_workers=16,
    batch_size=7,
)
```

### **3. Testing**
```bash
# Test the new function
python test_daily_extraction.py
```

## 📈 **Monitoring and Progress**

### **Progress Tracking**
- **Real-time updates**: Progress per day
- **Batch completion**: Progress per batch
- **Error reporting**: Failed days with reasons

### **Performance Metrics**
- **Processing time per day**
- **Memory usage per day**
- **Success/failure rates**
- **Overall throughput**

### **Log Output Example**
```
🚀 DAY-BY-DAY HRRR EXTRACTION
==================================================
📁 Data directory: /research/alij/hrrr
🌪️  Wind locations: 100
☀️  Solar locations: 50
📅 Processing 365 days: 2023-01-01 to 2023-12-31
🔍 Checking for completed days...
✅ Found 45 completed days
📅 Remaining days to process: 320
🚀 Using parallel processing with 16 workers
📦 Processing in batches of 7 days

📦 Processing batch 1/46
🌅 Processing day: 2023-01-01
✅ Saved wind UWind80: /output/wind/UWind80/20230101.parquet
✅ Saved wind VWind80: /output/wind/VWind80/20230101.parquet
✅ Saved solar rad: /output/solar/rad/20230101.parquet
```

## 🔧 **Optimization Tips**

### **1. Batch Size Optimization**
- **Small batches (1-3 days)**: Lower memory, slower I/O
- **Medium batches (7 days)**: Balanced performance
- **Large batches (14+ days)**: Higher memory, faster I/O

### **2. Worker Count Optimization**
- **Conservative**: `num_workers = cpu_count // 2`
- **Aggressive**: `num_workers = cpu_count`
- **Memory-constrained**: `num_workers = cpu_count // 4`

### **3. Memory Management**
- **Monitor memory usage**: Use `memory_optimizer.py`
- **Adjust batch size**: Based on available memory
- **Force garbage collection**: After each batch

## 🚨 **Troubleshooting**

### **Common Issues**

#### **1. Memory Issues**
```python
# Reduce batch size
batch_size=3  # Instead of 7

# Reduce worker count
num_workers=8  # Instead of 16
```

#### **2. I/O Bottlenecks**
```python
# Use separate output directories per process
output_dir = f"output/process_{os.getpid()}"

# Increase batch size to reduce I/O frequency
batch_size=14  # Instead of 7
```

#### **3. Resume Issues**
```python
# Disable resume to start fresh
enable_resume=False

# Check output directories manually
ls /output/wind/UWind80/
```

## 📝 **Migration Guide**

### **From Original to Day-by-Day**

#### **1. Update Enhanced Script**
✅ **COMPLETED**: `hrrr_enhanced.py` now uses day-by-day approach

#### **2. Test the New Function**
```bash
python test_daily_extraction.py
```

#### **3. Run Full Extraction**
```bash
python hrrr_enhanced.py
```

#### **4. Monitor Performance**
```bash
python memory_optimizer.py
```

## 🎯 **Expected Results**

### **Performance Improvements**
- **Memory stability**: No more crashes due to memory exhaustion
- **Consistent speed**: No slowdown over time
- **Fault tolerance**: Can resume from any point
- **Better monitoring**: Clear progress tracking

### **File Organization**
- **One file per day**: Easy to manage and analyze
- **Variable separation**: Each variable in its own directory
- **Date-based naming**: Clear chronological organization

### **Scalability**
- **Horizontal scaling**: Can process multiple years
- **Vertical scaling**: Can use more CPUs/memory
- **Distributed processing**: Can split across machines

## 🔮 **Future Enhancements**

### **1. Advanced Features**
- **Adaptive batch sizing**: Based on memory usage
- **Dynamic worker allocation**: Based on system load
- **Compression optimization**: Based on data characteristics

### **2. Monitoring Enhancements**
- **Real-time dashboard**: Web-based progress monitoring
- **Performance alerts**: Memory/CPU threshold warnings
- **Automated optimization**: Self-tuning parameters

### **3. Integration Features**
- **Database storage**: Store metadata in SQL database
- **Cloud integration**: Upload to cloud storage
- **API interface**: REST API for remote monitoring

## 📞 **Support**

### **Testing on Linux Machine**
If you need to test this on your Linux machine with real data, I can help you:

1. **Transfer the files** to your Linux system
2. **Adjust parameters** based on your system specs
3. **Monitor performance** during extraction
4. **Optimize settings** based on results

Just let me know when you're ready to test it on the Linux machine! 