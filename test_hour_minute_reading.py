#!/usr/bin/env python3
"""
Test script to demonstrate reading hour and minute information from parquet files.
"""

import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_hour_minute_from_index(df):
    """
    Extract hour and minute information from DatetimeIndex, similar to how read_data function works.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        print("⚠️ DataFrame does not have DatetimeIndex, cannot extract hour/minute")
        return df
    
    # Extract hour and minute from index (like read_data function)
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    
    # Add quarter hour for convenience
    df['quarter_hour'] = df.index.minute // 15
    
    # Add forecast hour information (f00 vs f01)
    import numpy as np
    df['forecast_hour'] = np.where(df.index.minute == 0, "0", "1")
    
    print(f"   📊 Extracted hour/minute: {df['hour'].nunique()} unique hours, {df['minute'].nunique()} unique minutes")
    print(f"   📊 Forecast hours: f00 (hourly) = {(df['forecast_hour'] == '0').sum()} records, f01 (quarter-hourly) = {(df['forecast_hour'] == '1').sum()} records")
    
    return df

def test_hour_minute_reading():
    """Test reading hour and minute information from parquet files."""
    
    print("🧪 Testing Hour/Minute Information Reading")
    print("=" * 50)
    
    # Test with a sample parquet file
    test_file = "./backup/rad_special_special_2023_1_to_2023_1.parquet"
    
    try:
        # Read the parquet file
        print(f"📖 Reading test file: {test_file}")
        df = pd.read_parquet(test_file)
        
        print(f"📊 Original DataFrame:")
        print(f"   Shape: {df.shape}")
        print(f"   Index type: {type(df.index)}")
        print(f"   First 5 timestamps: {df.index[:5].tolist()}")
        print(f"   Columns: {list(df.columns[:5])}")
        
        # Extract hour and minute information
        print(f"\n🔍 Extracting hour and minute information...")
        df_with_hour_minute = extract_hour_minute_from_index(df)
        
        print(f"📊 DataFrame with hour/minute info:")
        print(f"   Shape: {df_with_hour_minute.shape}")
        print(f"   New columns: {[col for col in df_with_hour_minute.columns if col in ['hour', 'minute', 'quarter_hour', 'forecast_hour']]}")
        
        # Show sample of hour/minute data
        print(f"\n📅 Sample hour/minute data:")
        sample_data = df_with_hour_minute[['hour', 'minute', 'quarter_hour', 'forecast_hour']].head(10)
        print(sample_data)
        
        # Test filtering by hour and minute
        print(f"\n🔍 Testing hour/minute filtering:")
        
        # Filter by hour
        hour_12_data = df_with_hour_minute[df_with_hour_minute['hour'] == 12]
        print(f"   Records at hour 12: {len(hour_12_data)}")
        
        # Filter by minute
        minute_0_data = df_with_hour_minute[df_with_hour_minute['minute'] == 0]
        print(f"   Records at minute 0 (:00): {len(minute_0_data)}")
        
        minute_15_data = df_with_hour_minute[df_with_hour_minute['minute'] == 15]
        print(f"   Records at minute 15 (:15): {len(minute_15_data)}")
        
        # Filter by quarter hour
        quarter_0_data = df_with_hour_minute[df_with_hour_minute['quarter_hour'] == 0]
        print(f"   Records at quarter hour 0 (:00): {len(quarter_0_data)}")
        
        quarter_1_data = df_with_hour_minute[df_with_hour_minute['quarter_hour'] == 1]
        print(f"   Records at quarter hour 1 (:15): {len(quarter_1_data)}")
        
        # Filter by forecast hour
        f00_data = df_with_hour_minute[df_with_hour_minute['forecast_hour'] == "0"]
        print(f"   Records with f00 (hourly): {len(f00_data)}")
        
        f01_data = df_with_hour_minute[df_with_hour_minute['forecast_hour'] == "1"]
        print(f"   Records with f01 (quarter-hourly): {len(f01_data)}")
        
        # Show the mapping
        print(f"\n📋 Hour/Minute Mapping:")
        print(f"   Hour (0-23): Extracted from index.hour")
        print(f"   Minute (0,15,30,45): Extracted from index.minute")
        print(f"   Quarter hour (0-3):")
        print(f"     - 0 = :00 (f00 - hourly data)")
        print(f"     - 1 = :15 (f01 - quarter-hourly data)")
        print(f"     - 2 = :30 (f01 - quarter-hourly data)")
        print(f"     - 3 = :45 (f01 - quarter-hourly data)")
        print(f"   Forecast hour:")
        print(f"     - '0' = f00 (hourly data from :00 timestamps)")
        print(f"     - '1' = f01 (quarter-hourly data from :15, :30, :45 timestamps)")
        
        print(f"\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()

def demonstrate_usage():
    """Demonstrate how to use the hour/minute reading functions."""
    
    print("\n" + "=" * 50)
    print("📚 USAGE EXAMPLES")
    print("=" * 50)
    
    print("""
# Example 1: Read parquet file and extract hour/minute info
import pandas as pd
from power_calculations import extract_hour_minute_from_index

# Read your parquet file
df = pd.read_parquet('your_file.parquet')

# Extract hour and minute information (like read_data function)
df_with_hour_minute = extract_hour_minute_from_index(df)

# Now you have these new columns:
# - 'hour': 0-23 (hour of day)
# - 'minute': 0, 15, 30, 45 (minute of hour)
# - 'quarter_hour': 0-3 (0=:00, 1=:15, 2=:30, 3=:45)
# - 'forecast_hour': '0' or '1' (f00 or f01)

# Example 2: Filter by hour and minute criteria
from power_calculations import filter_by_hour_minute

# Get only hourly data (f00)
hourly_data = filter_by_hour_minute(df_with_hour_minute, forecast_hour="0")

# Get only quarter-hourly data (f01)
quarter_hourly_data = filter_by_hour_minute(df_with_hour_minute, forecast_hour="1")

# Get data for specific hour
hour_12_data = filter_by_hour_minute(df_with_hour_minute, hour=12)

# Get data for specific minute
minute_15_data = filter_by_hour_minute(df_with_hour_minute, minute=15)  # :15

# Get data for specific quarter hour
quarter_1_data = filter_by_hour_minute(df_with_hour_minute, quarter_hour=1)  # :15

# Example 3: Get hour/minute statistics
from power_calculations import get_hour_minute_statistics

stats = get_hour_minute_statistics(df_with_hour_minute, "variable_name")
print(f"Total records: {stats['total_records']}")
print(f"Unique hours: {stats['unique_hours']}")
print(f"Unique minutes: {stats['unique_minutes']}")
print(f"Hourly records: {stats['hourly_records']}")
print(f"Quarter-hourly records: {stats['quarter_hourly_records']}")

# Example 4: Use the enhanced reading function
from power_calculations import read_parquet_files_with_hour_minute

# Read all parquet files with hour/minute info
weather_data = read_parquet_files_with_hour_minute()

# Each DataFrame in weather_data now has hour/minute columns
for var_name, df in weather_data.items():
    print(f"{var_name}: {df['hour'].nunique()} hours, {df['minute'].nunique()} minutes")
""")

if __name__ == "__main__":
    test_hour_minute_reading()
    demonstrate_usage()
