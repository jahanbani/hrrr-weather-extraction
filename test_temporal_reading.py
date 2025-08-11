#!/usr/bin/env python3
"""
Test script to demonstrate reading hour and quarter hour information from parquet files.
"""

import pandas as pd
import logging
from power_calculations import extract_temporal_info_from_index, get_temporal_statistics, filter_by_temporal_criteria

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_temporal_reading():
    """Test reading temporal information from parquet files."""
    
    print("🧪 Testing Temporal Information Reading")
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
        
        # Extract temporal information
        print(f"\n🔍 Extracting temporal information...")
        df_with_temporal = extract_temporal_info_from_index(df)
        
        print(f"📊 DataFrame with temporal info:")
        print(f"   Shape: {df_with_temporal.shape}")
        print(f"   New columns: {[col for col in df_with_temporal.columns if col in ['hour', 'quarter_hour', 'forecast_hour', 'minute']]}")
        
        # Show sample of temporal data
        print(f"\n📅 Sample temporal data:")
        sample_data = df_with_temporal[['hour', 'quarter_hour', 'forecast_hour', 'minute']].head(10)
        print(sample_data)
        
        # Get temporal statistics
        print(f"\n📈 Temporal statistics:")
        stats = get_temporal_statistics(df_with_temporal, "rad")
        
        # Test filtering
        print(f"\n🔍 Testing temporal filtering:")
        
        # Filter by hour
        hour_12_data = filter_by_temporal_criteria(df_with_temporal, hour=12)
        print(f"   Records at hour 12: {len(hour_12_data)}")
        
        # Filter by quarter hour (0 = :00, 1 = :15, 2 = :30, 3 = :45)
        quarter_0_data = filter_by_temporal_criteria(df_with_temporal, quarter_hour=0)
        print(f"   Records at quarter hour 0 (:00): {len(quarter_0_data)}")
        
        quarter_1_data = filter_by_temporal_criteria(df_with_temporal, quarter_hour=1)
        print(f"   Records at quarter hour 1 (:15): {len(quarter_1_data)}")
        
        # Filter by forecast hour
        f00_data = filter_by_temporal_criteria(df_with_temporal, forecast_hour="0")
        print(f"   Records with f00 (hourly): {len(f00_data)}")
        
        f01_data = filter_by_temporal_criteria(df_with_temporal, forecast_hour="1")
        print(f"   Records with f01 (quarter-hourly): {len(f01_data)}")
        
        # Show the mapping
        print(f"\n📋 Temporal Mapping:")
        print(f"   Hour (0-23): Extracted from index.hour")
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
    """Demonstrate how to use the temporal reading functions."""
    
    print("\n" + "=" * 50)
    print("📚 USAGE EXAMPLES")
    print("=" * 50)
    
    print("""
# Example 1: Read parquet file and extract temporal info
import pandas as pd
from power_calculations import extract_temporal_info_from_index

# Read your parquet file
df = pd.read_parquet('your_file.parquet')

# Extract temporal information
df_with_temporal = extract_temporal_info_from_index(df)

# Now you have these new columns:
# - 'hour': 0-23 (hour of day)
# - 'quarter_hour': 0-3 (0=:00, 1=:15, 2=:30, 3=:45)
# - 'forecast_hour': '0' or '1' (f00 or f01)
# - 'minute': 0, 15, 30, 45

# Example 2: Filter by temporal criteria
from power_calculations import filter_by_temporal_criteria

# Get only hourly data (f00)
hourly_data = filter_by_temporal_criteria(df_with_temporal, forecast_hour="0")

# Get only quarter-hourly data (f01)
quarter_hourly_data = filter_by_temporal_criteria(df_with_temporal, forecast_hour="1")

# Get data for specific hour
hour_12_data = filter_by_temporal_criteria(df_with_temporal, hour=12)

# Get data for specific quarter hour
quarter_1_data = filter_by_temporal_criteria(df_with_temporal, quarter_hour=1)  # :15

# Example 3: Get temporal statistics
from power_calculations import get_temporal_statistics

stats = get_temporal_statistics(df_with_temporal, "variable_name")
print(f"Total records: {stats['total_records']}")
print(f"Hourly records: {stats['hourly_records']}")
print(f"Quarter-hourly records: {stats['quarter_hourly_records']}")

# Example 4: Use the enhanced reading function
from power_calculations import read_parquet_files_with_temporal_info

# Read all parquet files with temporal info
weather_data = read_parquet_files_with_temporal_info()

# Each DataFrame in weather_data now has temporal columns
for var_name, df in weather_data.items():
    print(f"{var_name}: {df['hour'].nunique()} hours, {df['quarter_hour'].nunique()} quarter hours")
""")

if __name__ == "__main__":
    test_temporal_reading()
    demonstrate_usage()
