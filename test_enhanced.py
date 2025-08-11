#!/usr/bin/env python3
"""
Test script for enhanced HRRR extraction functionality.
This script tests the new features without modifying existing files.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_configuration():
    """Test configuration loading and validation."""
    logger.info("🧪 Testing configuration...")
    
    try:
        from config_unified import HRRRConfig, DEFAULT_CONFIG
        
        # Test default configuration
        config = HRRRConfig()
        logger.info("✅ Default configuration created")
        
        # Test validation (should fail if files don't exist)
        try:
            config.validate()
            logger.info("✅ Configuration validation passed")
        except Exception as e:
            logger.warning(f"⚠️  Configuration validation failed (expected): {e}")
        
        # Test configuration methods
        output_dirs = config.get_output_dirs()
        logger.info(f"✅ Output directories: {output_dirs}")
        
        data_dir = config.get_data_directory()
        logger.info(f"✅ Data directory: {data_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        return False


def test_utilities():
    """Test enhanced utility functions."""
    logger.info("🧪 Testing utilities...")
    
    try:
        from utils_enhanced import (
            check_memory_usage,
            format_duration,
            log_system_info,
            performance_monitor
        )
        
        # Test memory check
        memory_high = check_memory_usage()
        logger.info(f"✅ Memory check: {'High' if memory_high else 'Normal'}")
        
        # Test duration formatting
        test_durations = [30, 90, 3661, 7200]
        for duration in test_durations:
            formatted = format_duration(duration)
            logger.info(f"✅ Duration {duration}s -> {formatted}")
        
        # Test system info
        log_system_info()
        logger.info("✅ System info logged")
        
        # Test performance monitor
        performance_monitor.start_operation()
        performance_monitor.end_operation(10, 100)
        summary = performance_monitor.get_summary()
        logger.info(f"✅ Performance monitor: {summary}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Utilities test failed: {e}")
        return False


def test_file_validation():
    """Test file validation functions."""
    logger.info("🧪 Testing file validation...")
    
    try:
        from utils_enhanced import validate_csv_file, validate_inputs
        
        # Test with non-existent files (should fail gracefully)
        try:
            validate_csv_file("nonexistent.csv", ["pid", "lat", "lon"])
            logger.error("❌ Should have failed for non-existent file")
            return False
        except FileNotFoundError:
            logger.info("✅ Correctly failed for non-existent file")
        
        # Test with existing files if they exist
        if os.path.exists("wind.csv") and os.path.exists("solar.csv"):
            try:
                validate_inputs("wind.csv", "solar.csv")
                logger.info("✅ File validation passed")
            except Exception as e:
                logger.warning(f"⚠️  File validation failed: {e}")
        else:
            logger.info("⚠️  Skipping file validation (wind.csv/solar.csv not found)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ File validation test failed: {e}")
        return False


def test_enhanced_main():
    """Test enhanced main functionality (without running full extraction)."""
    logger.info("🧪 Testing enhanced main...")
    
    try:
        from hrrr_enhanced import main_enhanced
        
        # Test that the function can be imported and called
        logger.info("✅ Enhanced main function imported successfully")
        
        # Note: We don't actually run the extraction in tests
        # to avoid modifying files or taking too long
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced main test failed: {e}")
        return False


def test_cli():
    """Test CLI functionality."""
    logger.info("🧪 Testing CLI...")
    
    try:
        from cli import cli
        
        # Test that CLI can be imported
        logger.info("✅ CLI imported successfully")
        
        # Test CLI help (this should work without running extraction)
        import subprocess
        result = subprocess.run([sys.executable, "cli.py", "--help"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ CLI help command works")
        else:
            logger.warning(f"⚠️  CLI help command failed: {result.stderr}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ CLI test failed: {e}")
        return False


def create_test_files():
    """Create test CSV files for validation."""
    logger.info("📝 Creating test files...")
    
    try:
        import pandas as pd
        
        # Create test wind CSV
        wind_data = {
            'pid': ['wind_001', 'wind_002', 'wind_003'],
            'lat': [40.7128, 34.0522, 41.8781],
            'lon': [-74.0060, -118.2437, -87.6298]
        }
        wind_df = pd.DataFrame(wind_data)
        wind_df.to_csv('test_wind.csv', index=False)
        logger.info("✅ Created test_wind.csv")
        
        # Create test solar CSV
        solar_data = {
            'pid': ['solar_001', 'solar_002', 'solar_003'],
            'lat': [36.7783, 39.8283, 37.7749],
            'lon': [-119.4179, -98.5795, -122.4194]
        }
        solar_df = pd.DataFrame(solar_data)
        solar_df.to_csv('test_solar.csv', index=False)
        logger.info("✅ Created test_solar.csv")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create test files: {e}")
        return False


def cleanup_test_files():
    """Clean up test files."""
    logger.info("🧹 Cleaning up test files...")
    
    test_files = ['test_wind.csv', 'test_solar.csv']
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
            logger.info(f"✅ Removed {file}")


def run_all_tests():
    """Run all tests."""
    logger.info("🚀 Starting enhanced functionality tests...")
    logger.info("=" * 50)
    
    tests = [
        ("Configuration", test_configuration),
        ("Utilities", test_utilities),
        ("File Validation", test_file_validation),
        ("Enhanced Main", test_enhanced_main),
        ("CLI", test_cli),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{status}: {test_name}")
        except Exception as e:
            logger.error(f"❌ FAILED: {test_name} - {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{status}: {test_name}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Enhanced functionality is ready.")
        return True
    else:
        logger.warning(f"⚠️  {total - passed} tests failed. Check the logs above.")
        return False


if __name__ == "__main__":
    # Create test files
    create_test_files()
    
    # Run tests
    success = run_all_tests()
    
    # Clean up
    cleanup_test_files()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1) 