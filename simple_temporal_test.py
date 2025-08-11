#!/usr/bin/env python3
"""
Simple test script to demonstrate reading hour and quarter hour information from parquet files.
"""

import pandas as pd

def extract_temporal_info_from_index(df):
    """
    Extract hour and quarter hour information from the DatetimeIndex of a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with DatetimeIndex
        
    Returns:
        pd.DataFrame: DataFrame with added 'hour' and 'quarter_hour' columns
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        print("⚠️ DataFrame does not have DatetimeIndex, cannot extract temporal info")
        return df
    
    # Extract hour from index
    df['hour'] = df.index.hour
    
    # Extract quarter hour based on minutes
    # 00 min = f00 (hourly) -> quarter_hour = 0
    # 15,30,45 min = f01 (quarter-hourly) -> quarter_hour = 1,2,3
    df['quarter_hour'] = df.index.minute // 15
    
    # Add forecast hour information (f00 vs f01)
    # Use numpy where for better performance
    import numpy as np
    df['forecast_hour'] = np.where(df.index.minute == 0, "0", "1")
    
    # Add minute information for reference
    df['minute'] = df.index.minute
    
    print(f"   📊 Temporal info extracted: {df['hour'].nunique()} hours, {df['quarter_hour'].nunique()} quarter hours")
    print(f"   📊 Forecast hours: f00 (hourly) = {(df['forecast_hour'] == '0').sum()} records, f01 (quarter-hourly) = {(df['forecast_hour'] == '1').sum()} records")
    
    return df

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
        
        # Test filtering
        print(f"\n🔍 Testing temporal filtering:")
        
        # Filter by hour
        hour_12_data = df_with_temporal[df_with_temporal['hour'] == 12]
        print(f"   Records at hour 12: {len(hour_12_data)}")
        
        # Filter by quarter hour (0 = :00, 1 = :15, 2 = :30, 3 = :45)
        quarter_0_data = df_with_temporal[df_with_temporal['quarter_hour'] == 0]
        print(f"   Records at quarter hour 0 (:00): {len(quarter_0_data)}")
        
        quarter_1_data = df_with_temporal[df_with_temporal['quarter_hour'] == 1]
        print(f"   Records at quarter hour 1 (:15): {len(quarter_1_data)}")
        
        # Filter by forecast hour
        f00_data = df_with_temporal[df_with_temporal['forecast_hour'] == "0"]
        print(f"   Records with f00 (hourly): {len(f00_data)}")
        
        f01_data = df_with_temporal[df_with_temporal['forecast_hour'] == "1"]
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

if __name__ == "__main__":
    test_temporal_reading()
