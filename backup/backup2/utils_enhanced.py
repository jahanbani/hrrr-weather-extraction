#!/usr/bin/env python3
"""
Enhanced utility functions for HRRR data extraction.
These functions provide improved error handling, validation, and monitoring.
"""

import os
import time
import gc
import logging
from typing import Dict, List, Optional, Any, Tuple
from contextlib import contextmanager
from dataclasses import dataclass
import pandas as pd
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for extraction operations"""
    start_time: float
    end_time: float
    files_processed: int
    data_points_extracted: int
    memory_peak_gb: float
    cpu_utilization: float
    
    @property
    def duration_seconds(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def throughput_files_per_second(self) -> float:
        return self.files_processed / self.duration_seconds if self.duration_seconds > 0 else 0
    
    @property
    def throughput_points_per_second(self) -> float:
        return self.data_points_extracted / self.duration_seconds if self.duration_seconds > 0 else 0


class PerformanceMonitor:
    """Monitor and track performance metrics"""
    
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.current_start = None
    
    def start_operation(self):
        """Start timing an operation"""
        self.current_start = time.time()
        logger.info("Performance monitoring started")
    
    def end_operation(self, files_processed: int, points_extracted: int):
        """End timing and record metrics"""
        if self.current_start is None:
            return
        
        end_time = time.time()
        
        # Get system metrics
        try:
            import psutil
            memory_gb = psutil.virtual_memory().used / (1024**3)
            cpu_percent = psutil.cpu_percent()
        except ImportError:
            memory_gb = 0.0
            cpu_percent = 0.0
        
        metrics = PerformanceMetrics(
            start_time=self.current_start,
            end_time=end_time,
            files_processed=files_processed,
            data_points_extracted=points_extracted,
            memory_peak_gb=memory_gb,
            cpu_utilization=cpu_percent
        )
        
        self.metrics.append(metrics)
        self.current_start = None
        
        logger.info(f"Operation completed: {metrics.duration_seconds:.1f}s, "
                   f"{metrics.throughput_files_per_second:.2f} files/s, "
                   f"{metrics.throughput_points_per_second:.2f} points/s")
    
    def get_summary(self) -> Dict[str, float]:
        """Get performance summary"""
        if not self.metrics:
            return {}
        
        return {
            'total_duration': sum(m.duration_seconds for m in self.metrics),
            'avg_throughput_files_per_sec': sum(m.throughput_files_per_second for m in self.metrics) / len(self.metrics),
            'avg_throughput_points_per_sec': sum(m.throughput_points_per_second for m in self.metrics) / len(self.metrics),
            'peak_memory_gb': max(m.memory_peak_gb for m in self.metrics),
            'avg_cpu_utilization': sum(m.cpu_utilization for m in self.metrics) / len(self.metrics)
        }


def check_memory_usage(warning_threshold: float = 80.0) -> bool:
    """Check memory usage and warn if high"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        if memory.percent > warning_threshold:
            logger.warning(f"High memory usage: {memory.percent:.1f}%")
            gc.collect()
            return True
        return False
    except ImportError:
        logger.debug("psutil not available - memory monitoring disabled")
        return False


def validate_csv_file(file_path: str, required_columns: List[str]) -> bool:
    """Validate CSV file exists and has required columns"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")
        
        logger.info(f"✅ Validated {file_path}: {len(df)} rows, {len(df.columns)} columns")
        return True
    except Exception as e:
        logger.error(f"❌ Error validating {file_path}: {e}")
        raise


def validate_inputs(wind_csv_path: str, solar_csv_path: str) -> bool:
    """Validate input files exist and have correct format"""
    required_columns = ["pid", "lat", "lon"]
    
    logger.info("🔍 Validating input files...")
    
    # Validate wind CSV
    validate_csv_file(wind_csv_path, required_columns)
    
    # Validate solar CSV
    validate_csv_file(solar_csv_path, required_columns)
    
    logger.info("✅ All input files validated successfully")
    return True


@contextmanager
def error_context(operation: str):
    """Context manager for error handling"""
    try:
        yield
    except Exception as e:
        logger.error(f"Error during {operation}: {e}")
        raise


def safe_file_operation(file_path: str, operation: str) -> Optional[Any]:
    """Safely perform file operations with error handling"""
    with error_context(f"File {operation} on {file_path}"):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Perform the operation (placeholder)
        return True


def create_output_directories(output_dirs: Dict[str, str]) -> bool:
    """Create output directories if they don't exist"""
    for name, path in output_dirs.items():
        try:
            os.makedirs(path, exist_ok=True)
            logger.info(f"✅ Created output directory: {path}")
        except Exception as e:
            logger.error(f"❌ Error creating directory {path}: {e}")
            raise
    
    return True


def get_file_size_mb(file_path: str) -> float:
    """Get file size in MB"""
    try:
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)
    except OSError:
        return 0.0


def estimate_processing_time(num_files: int, avg_file_size_mb: float, 
                           workers: int = 4) -> float:
    """Estimate processing time in seconds"""
    # Rough estimate: 1 second per MB per file, divided by workers
    base_time = num_files * avg_file_size_mb / workers
    return max(base_time, 1.0)  # Minimum 1 second


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def log_system_info():
    """Log system information for debugging"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        cpu_count = psutil.cpu_count()
        
        logger.info(f"System Info:")
        logger.info(f"  CPU cores: {cpu_count}")
        logger.info(f"  Memory: {memory.total / (1024**3):.1f} GB total")
        logger.info(f"  Memory available: {memory.available / (1024**3):.1f} GB")
        logger.info(f"  Platform: {os.name}")
        
    except ImportError:
        logger.warning("psutil not available - system info not logged")


def check_disk_space(path: str, required_gb: float = 10.0) -> bool:
    """Check if sufficient disk space is available"""
    try:
        import shutil
        total, used, free = shutil.disk_usage(path)
        free_gb = free / (1024**3)
        
        if free_gb < required_gb:
            logger.warning(f"Low disk space: {free_gb:.1f} GB available, "
                         f"{required_gb:.1f} GB required")
            return False
        
        logger.info(f"✅ Sufficient disk space: {free_gb:.1f} GB available")
        return True
        
    except Exception as e:
        logger.warning(f"Could not check disk space: {e}")
        return True  # Assume OK if we can't check


def validate_data_directory(data_dir: str) -> bool:
    """Validate that data directory exists and contains GRIB files"""
    if not os.path.exists(data_dir):
        logger.error(f"Data directory not found: {data_dir}")
        return False
    
    # Check for GRIB files
    grib_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.grib2'):
                grib_files.append(os.path.join(root, file))
    
    if not grib_files:
        logger.warning(f"No GRIB files found in {data_dir}")
        return False
    
    logger.info(f"✅ Found {len(grib_files)} GRIB files in {data_dir}")
    return True


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


if __name__ == "__main__":
    # Test utilities
    print("🧪 Testing enhanced utilities...")
    
    # Test memory check
    check_memory_usage()
    
    # Test system info
    log_system_info()
    
    # Test duration formatting
    print(f"Duration formatting: {format_duration(3661.5)}")
    
    print("✅ Enhanced utilities test completed") 