#!/usr/bin/env python3
"""
Quick test to verify the variable scope fix works.
"""

import logging
from datetime import datetime
from extraction_core import _extract_single_day

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_variable_scope():
    """Test that the variable scope issue is fixed."""
    
    logger.info("🧪 Testing Variable Scope Fix...")
    
    # Create a mock task (similar to what the main function creates)
    mock_task = (
        datetime(2019, 1, 1),  # date
        None,  # wind_locations (will be None for this test)
        None,  # solar_locations (will be None for this test)
        "/research/alij/hrrr",  # datadir
        ["0", "1"],  # hours_forecasted
        {"UWind80": "u", "VWind80": "v"},  # wind_selectors
        {"rad": "dswrf", "vbd": "vbdsf"},  # solar_selectors
        "/tmp/wind",  # wind_output_dir
        "/tmp/solar",  # solar_output_dir
        "snappy"  # compression
    )
    
    try:
        # This should not crash with "wind_indices not defined"
        result = _extract_single_day(mock_task)
        logger.info(f"✅ Test passed! Result: {result['status']}")
        logger.info(f"   Error (expected): {result.get('error', 'None')}")
        return True
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_variable_scope()
    if success:
        print("🎉 Variable scope fix is working!")
    else:
        print("💥 Variable scope fix failed!")
