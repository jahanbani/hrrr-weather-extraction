# Date Range Configuration Guide

## 🕐 **Where to Set Extraction Hours**

The extraction time range is configured in **`hrrr.py`** in the `extract_specific_locations()` and `extract_full_grid()` functions.

## 📅 **Current Configuration**

You are currently set to run for **1 full year** (2023):
- **Start:** January 1, 2023, 00:00:00
- **End:** December 31, 2023, 23:00:00

## 🔧 **How to Change the Time Range**

### **Location 1: Specific Locations Extraction**
In `hrrr.py`, around **line 85-95**:

```python
# ============================================================================
# DATE RANGE CONFIGURATION - MODIFY THESE LINES TO CHANGE EXTRACTION PERIOD
# ============================================================================

# Option 1: Full year extraction (2023) - CURRENTLY ACTIVE
START = datetime.datetime(2023, 1, 1, 0, 0, 0)  # Start: January 1, 2023
END = datetime.datetime(2023, 12, 31, 23, 0, 0)  # End: December 31, 2023

# Option 2: Month extraction (January 2023)
# START = datetime.datetime(2023, 1, 1, 0, 0, 0)  # Start: January 1, 2023
# END = datetime.datetime(2023, 1, 31, 23, 0, 0)  # End: January 31, 2023

# Option 3: Week extraction (first week of 2023)
# START = datetime.datetime(2023, 1, 1, 0, 0, 0)  # Start: January 1, 2023
# END = datetime.datetime(2023, 1, 7, 23, 0, 0)  # End: January 7, 2023

# Option 4: Day extraction (January 1, 2023)
# START = datetime.datetime(2023, 1, 1, 0, 0, 0)  # Start: January 1, 2023
# END = datetime.datetime(2023, 1, 1, 23, 0, 0)  # End: January 1, 2023

# Option 5: Test extraction (2 hours only)
# START = datetime.datetime(2023, 1, 1, 0, 0, 0)  # Start: January 1, 2023, 00:00
# END = datetime.datetime(2023, 1, 1, 2, 0, 0)  # End: January 1, 2023, 02:00
```

### **Location 2: Full Grid Extraction**
In `hrrr.py`, around **line 220-230**:

```python
# ============================================================================
# DATE RANGE CONFIGURATION - MODIFY THESE LINES TO CHANGE EXTRACTION PERIOD
# ============================================================================

# Option 1: Full year extraction (2023) - DEFAULT
START = datetime.datetime(2023, 1, 1, 0, 0, 0)  # Start: January 1, 2023
END = datetime.datetime(2023, 12, 31, 23, 0, 0)  # End: December 31, 2023

# Option 2: Month extraction (January 2023)
# START = datetime.datetime(2023, 1, 1, 0, 0, 0)  # Start: January 1, 2023
# END = datetime.datetime(2023, 1, 31, 23, 0, 0)  # End: January 31, 2023
```

## 🎯 **Quick Configuration Examples**

### **For Testing (2 hours):**
```python
START = datetime.datetime(2023, 1, 1, 0, 0, 0)   # 00:00
END = datetime.datetime(2023, 1, 1, 2, 0, 0)     # 02:00
```

### **For One Day:**
```python
START = datetime.datetime(2023, 1, 1, 0, 0, 0)   # 00:00
END = datetime.datetime(2023, 1, 1, 23, 0, 0)    # 23:00
```

### **For One Week:**
```python
START = datetime.datetime(2023, 1, 1, 0, 0, 0)   # January 1, 00:00
END = datetime.datetime(2023, 1, 7, 23, 0, 0)    # January 7, 23:00
```

### **For One Month:**
```python
START = datetime.datetime(2023, 1, 1, 0, 0, 0)   # January 1, 00:00
END = datetime.datetime(2023, 1, 31, 23, 0, 0)   # January 31, 23:00
```

### **For Full Year:**
```python
START = datetime.datetime(2023, 1, 1, 0, 0, 0)   # January 1, 00:00
END = datetime.datetime(2023, 12, 31, 23, 0, 0)  # December 31, 23:00
```

## ⚠️ **Important Notes**

1. **Time Format:** Use `datetime.datetime(year, month, day, hour, minute, second)`
2. **Hour Range:** 0-23 (24-hour format)
3. **Date Range:** Make sure END is after START
4. **Processing Time:** Longer ranges = more processing time
5. **Storage:** Longer ranges = more output files

## 📊 **Processing Time Estimates**

| Time Range | Estimated Processing Time | Output Size |
|------------|--------------------------|-------------|
| 2 hours    | ~5-10 minutes           | ~50 MB      |
| 1 day      | ~1-2 hours              | ~600 MB     |
| 1 week     | ~8-12 hours             | ~4 GB       |
| 1 month    | ~2-3 days               | ~15 GB      |
| 1 year     | ~2-3 weeks              | ~180 GB     |

## 🔄 **How to Switch Between Options**

1. **Comment out** the current active option
2. **Uncomment** the option you want to use
3. **Save** the file
4. **Run** the extraction

Example:
```python
# Comment out current option
# START = datetime.datetime(2023, 1, 1, 0, 0, 0)
# END = datetime.datetime(2023, 12, 31, 23, 0, 0)

# Uncomment desired option
START = datetime.datetime(2023, 1, 1, 0, 0, 0)  # Test: 2 hours
END = datetime.datetime(2023, 1, 1, 2, 0, 0)    # Test: 2 hours
```

## 🚀 **Current Status**

You are currently configured for **FULL YEAR EXTRACTION** (2023), which will process:
- **8,760 hours** of data (365 days × 24 hours)
- **Estimated time:** 2-3 weeks
- **Estimated output:** ~180 GB

To change to a shorter period, modify the START and END dates in `hrrr.py`. 