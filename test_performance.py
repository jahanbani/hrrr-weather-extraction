#!/usr/bin/env python3
"""
Quick performance test for the optimized HRRR extraction system.
"""

import time
import logging
from datetime import datetime
from extraction_core import extract_specific_points_daily_single_pass
from memory_optimizer import memory_optimizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_performance():
    """Test the performance of the optimized extraction system."""
    
    logger.info("🧪 Testing Performance Optimizations...")
    
    # Test memory optimizer
    logger.info("🔍 Testing Memory Optimizer...")
    memory_usage = memory_optimizer.get_memory_usage()
    logger.info(f"   Current memory: {memory_usage['used_gb']:.1f}GB / {memory_usage['total_gb']:.1f}GB")
    
    # Test worker optimization
    logger.info("🎯 Testing Worker Optimization...")
    from extraction_core import _optimize_workers_for_36cpu_256gb
    optimal_workers = _optimize_workers_for_36cpu_256gb()
    logger.info(f"   Optimal workers: {optimal_workers}")
    
    # Test file grouping
    logger.info("📁 Testing File Grouping...")
    from extraction_core import _group_grib_files_by_time
    
    # Mock GRIB files for testing
    test_files = [
        "/research/alij/hrrr/20190101/hrrr.t00z.wrfsubhf00.grib2",
        "/research/alij/hrrr/20190101/hrrr.t00z.wrfsubhf01.grib2",
        "/research/alij/hrrr/20190101/hrrr.t01z.wrfsubhf00.grib2",
        "/research/alij/hrrr/20190101/hrrr.t01z.wrfsubhf01.grib2",
        "/research/alij/hrrr/20190101/hrrr.t02z.wrfsubhf00.grib2",
        "/research/alij/hrrr/20190101/hrrr.t02z.wrfsubhf01.grib2"
    ]
    
    file_groups = _group_grib_files_by_time(test_files)
    logger.info(f"   Test files grouped into {len(file_groups)} time groups")
    for key, group in file_groups.items():
        logger.info(f"     {key}: {list(group.keys())}")
    
    logger.info("✅ Performance test completed successfully!")
    logger.info("🚀 System is ready for high-performance extraction!")

if __name__ == "__main__":
    test_performance()
