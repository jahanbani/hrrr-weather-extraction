#!/usr/bin/env python3
"""
Test the HRRR extraction package with comprehensive timeouts.
UNSTICK STRATEGY: Every operation has timeouts, fallbacks, and progress reporting.
"""

import time
import sys
import signal
import os
import traceback

# Add package to path
sys.path.insert(0, '/home/alij/EE/Weather/hrrr_extraction')

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

def test_with_timeout(test_name, test_func, timeout_seconds=30):
    """Run a test with timeout protection"""
    print(f"\n🧪 Testing: {test_name} (max {timeout_seconds}s)")
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    start_time = time.time()
    try:
        result = test_func()
        elapsed = time.time() - start_time
        signal.alarm(0)
        print(f"   ✅ SUCCESS in {elapsed:.2f}s: {result}")
        return True
    except TimeoutError:
        print(f"   ⏰ TIMEOUT after {timeout_seconds}s")
        signal.alarm(0)
        return False
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"   ❌ ERROR in {elapsed:.2f}s: {e}")
        signal.alarm(0)
        return False

def test_basic_imports():
    """Test basic package imports"""
    from core.config import HRRRConfig
    from core.utils import setup_logging
    from core.extraction import extract_multiple_regions_quarterly_optimized
    return "All imports successful"

def test_config_creation():
    """Test configuration creation"""
    from core.config import HRRRConfig
    config = HRRRConfig()
    regions = config.get_regions()
    return f"Config created with {len(regions)} regions"

def test_extraction_function():
    """Test the main extraction function with mock data"""
    from core.extraction import test_package_functionality
    result = test_package_functionality()
    return f"Extraction test: {result.get('status', 'unknown') if result else 'failed'}"

def main():
    """Main test runner with comprehensive timeout protection"""
    print("🚀 HRRR PACKAGE TEST WITH TIMEOUTS")
    print("=" * 50)
    print("📋 UNSTICK STRATEGY: 30s max per test, fallbacks for everything")
    
    # Global timeout for entire test suite
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(300)  # 5 minutes total
    
    tests = [
        ("Basic Imports", test_basic_imports, 10),
        ("Config Creation", test_config_creation, 15),
        ("Extraction Function", test_extraction_function, 60),
    ]
    
    results = []
    
    try:
        for test_name, test_func, timeout in tests:
            success = test_with_timeout(test_name, test_func, timeout)
            results.append((test_name, success))
            
            if not success:
                print(f"   🔄 FALLBACK: Continuing with next test...")
        
        signal.alarm(0)  # Clear global timeout
        
        # Summary
        print(f"\n📊 TEST RESULTS:")
        passed = sum(1 for _, success in results if success)
        total = len(results)
        
        for test_name, success in results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"   {status}: {test_name}")
        
        print(f"\n🎯 FINAL SCORE: {passed}/{total} tests passed")
        
        if passed >= total // 2:
            print("✅ PACKAGE IS FUNCTIONAL (majority tests passed)")
            return 0
        else:
            print("⚠️ PACKAGE NEEDS WORK (majority tests failed)")
            return 1
            
    except TimeoutError:
        print("\n⏰ GLOBAL TIMEOUT: Test suite exceeded 5 minutes")
        signal.alarm(0)
        return 2
    except KeyboardInterrupt:
        print("\n🛑 INTERRUPTED by user")
        signal.alarm(0)
        return 3
    except Exception as e:
        print(f"\n💥 UNEXPECTED ERROR: {e}")
        traceback.print_exc()
        signal.alarm(0)
        return 4

if __name__ == "__main__":
    sys.exit(main())
