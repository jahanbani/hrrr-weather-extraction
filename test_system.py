#!/usr/bin/env python3
"""
Test script to verify the HRRR extraction system works correctly.
"""

import sys
import os
from datetime import datetime

def test_imports():
    """Test if all modules can be imported."""
    try:
        import extraction_core
        print("✅ extraction_core imported successfully")
        
        import config_unified
        print("✅ config_unified imported successfully")
        
        import prereise_essentials
        print("✅ prereise_essentials imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_grib_discovery():
    """Test GRIB file discovery."""
    try:
        from extraction_core import _find_grib_files_for_day
        
        # Test with available data
        test_date = datetime(2019, 1, 1)
        datadir = "/research/alij/hrrr"
        hours_forecasted = ["0", "1"]
        
        files = _find_grib_files_for_day(test_date, datadir, hours_forecasted)
        print(f"✅ Found {len(files)} GRIB files for {test_date.date()}")
        
        if files:
            sample_files = [os.path.basename(f) for f in files[:3]]
            print(f"📋 Sample files: {sample_files}")
        
        return True
    except Exception as e:
        print(f"❌ GRIB discovery error: {e}")
        return False

def test_optimization():
    """Test auto-optimization for 36 CPU system."""
    try:
        import extraction_core
        
        workers = extraction_core._optimize_workers_for_36cpu_256gb()
        print(f"✅ Auto-optimized to {workers} workers")
        
        return True
    except Exception as e:
        print(f"❌ Optimization error: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing HRRR Extraction System")
    print("=" * 50)
    
    tests = [
        ("Module Imports", test_imports),
        ("GRIB Discovery", test_grib_discovery),
        ("Auto-Optimization", test_optimization),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing: {test_name}")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
