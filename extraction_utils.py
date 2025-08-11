"""
Extraction Utilities - Helper functions and utilities for HRRR data extraction.

This module contains utility functions that support the main extraction operations,
including data processing, file handling, and optimization helpers.
"""

import logging
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import warnings

logger = logging.getLogger(__name__)


def count_csv_rows(csv_path: str) -> int:
    """Count data rows in a CSV file (excluding header) efficiently."""
    try:
        with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
            # Subtract 1 for header if present
            return max(sum(1 for _ in f) - 1, 0)
    except FileNotFoundError:
        return 0


def validate_region_bounds(region_bounds: Dict[str, float]) -> bool:
    """Validate that region bounds are properly formatted and logical."""
    required_keys = ["lat_min", "lat_max", "lon_min", "lon_max"]
    
    # Check all required keys exist
    if not all(key in region_bounds for key in required_keys):
        raise ValueError(f"Region bounds must contain: {required_keys}")
    
    # Check logical bounds
    if region_bounds["lat_min"] >= region_bounds["lat_max"]:
        raise ValueError("lat_min must be less than lat_max")
    if region_bounds["lon_min"] >= region_bounds["lon_max"]:
        raise ValueError("lon_min must be less than lon_max")
    
    # Check reasonable ranges
    if not (-90 <= region_bounds["lat_min"] <= 90):
        raise ValueError("lat_min must be between -90 and 90")
    if not (-90 <= region_bounds["lat_max"] <= 90):
        raise ValueError("lat_max must be between -90 and 90")
    if not (-180 <= region_bounds["lon_min"] <= 180):
        raise ValueError("lon_min must be between -180 and 180")
    if not (-180 <= region_bounds["lon_max"] <= 180):
        raise ValueError("lon_max must be between -180 and 180")
    
    return True


def estimate_grid_points(region_bounds: Dict[str, float]) -> int:
    """Estimate the number of grid points in a region."""
    # Rough estimation based on region size
    lat_range = region_bounds["lat_max"] - region_bounds["lat_min"]
    lon_range = region_bounds["lon_max"] - region_bounds["lon_min"]
    
    # HRRR grid is approximately 3km resolution
    # Convert to approximate grid points
    lat_points = int(lat_range * 111 / 3)  # 111 km per degree latitude
    lon_points = int(lon_range * 111 * np.cos(np.radians(region_bounds["lat_min"])) / 3)
    
    return max(lat_points * lon_points, 1)  # Ensure at least 1 point


def calculate_memory_requirements(
    regions: Dict[str, Dict[str, float]], 
    days: int, 
    variables: int
) -> Dict[str, float]:
    """Calculate estimated memory requirements for extraction operations."""
    total_grid_points = sum(estimate_grid_points(bounds) for bounds in regions.values())
    
    # Rough memory estimates per grid point per variable per time step
    # Assuming float32 data and some overhead
    bytes_per_point_per_var = 4  # float32
    overhead_factor = 1.5  # Include some overhead for processing
    
    # Calculate total memory needed
    total_memory_bytes = (
        total_grid_points * 
        days * 
        24 *  # 24 hours per day
        variables * 
        bytes_per_point_per_var * 
        overhead_factor
    )
    
    # Convert to GB
    total_memory_gb = total_memory_bytes / (1024**3)
    
    return {
        "total_memory_gb": total_memory_gb,
        "total_grid_points": total_grid_points,
        "estimated_days": days,
        "estimated_variables": variables
    }


def optimize_worker_count(
    total_grid_points: int, 
    available_memory_gb: float,
    available_cpus: int
) -> Dict[str, int]:
    """Optimize the number of workers based on system resources."""
    # Memory-based optimization
    memory_per_worker_gb = 2.0  # Conservative estimate per worker
    max_workers_by_memory = int(available_memory_gb / memory_per_worker_gb)
    
    # CPU-based optimization
    max_workers_by_cpu = available_cpus - 1  # Leave one CPU for system
    
    # Grid size optimization
    if total_grid_points < 10000:
        # Small grids: fewer workers
        max_workers_by_grid = 2
    elif total_grid_points < 100000:
        # Medium grids: moderate workers
        max_workers_by_grid = 4
    else:
        # Large grids: more workers
        max_workers_by_grid = 8
    
    # Take the minimum of all constraints
    optimal_workers = min(max_workers_by_memory, max_workers_by_cpu, max_workers_by_grid)
    
    return {
        "optimal_workers": max(optimal_workers, 1),  # Ensure at least 1 worker
        "max_workers_by_memory": max_workers_by_memory,
        "max_workers_by_cpu": max_workers_by_cpu,
        "max_workers_by_grid": max_workers_by_grid
    }


def create_output_structure(base_dir: str, regions: Dict[str, Dict[str, float]]) -> Dict[str, str]:
    """Create the output directory structure for extraction results."""
    output_dirs = {}
    
    # Create base directory
    os.makedirs(base_dir, exist_ok=True)
    
    # Create region-specific directories
    for region_name in regions.keys():
        region_dir = os.path.join(base_dir, region_name)
        os.makedirs(region_dir, exist_ok=True)
        output_dirs[region_name] = region_dir
        
        # Create subdirectories for different data types
        wind_dir = os.path.join(region_dir, "wind")
        solar_dir = os.path.join(region_dir, "solar")
        
        os.makedirs(wind_dir, exist_ok=True)
        os.makedirs(solar_dir, exist_ok=True)
        
        output_dirs[f"{region_name}_wind"] = wind_dir
        output_dirs[f"{region_name}_solar"] = solar_dir
    
    return output_dirs


def format_extraction_summary(results: Dict[str, Any]) -> str:
    """Format extraction results into a readable summary."""
    summary_lines = []
    
    if results.get("status") == "completed":
        summary_lines.append("✅ EXTRACTION COMPLETED SUCCESSFULLY")
        summary_lines.append("=" * 50)
        
        if "total_days" in results:
            summary_lines.append(f"📅 Total Days: {results['total_days']}")
            summary_lines.append(f"✅ Successful Days: {results['successful_days']}")
            summary_lines.append(f"❌ Failed Days: {results['failed_days']}")
        
        if "total_regions" in results:
            summary_lines.append(f"🌍 Total Regions: {results['total_regions']}")
            summary_lines.append(f"✅ Successful Regions: {results['successful_regions']}")
            summary_lines.append(f"❌ Failed Regions: {results['failed_regions']}")
        
        if "total_grid_points" in results:
            summary_lines.append(f"📊 Total Grid Points: {results['total_grid_points']:,}")
        
        if "processing_time_seconds" in results:
            hours = results['processing_time_seconds'] / 3600
            summary_lines.append(f"⏱️  Processing Time: {hours:.2f} hours")
        
        if "files_processed" in results:
            summary_lines.append(f"📁 Files Processed: {results['files_processed']}")
            
    else:
        summary_lines.append("❌ EXTRACTION FAILED")
        summary_lines.append("=" * 30)
        summary_lines.append(f"Error: {results.get('error', 'Unknown error')}")
    
    return "\n".join(summary_lines)


def check_extraction_prerequisites(
    wind_csv_path: str = None,
    solar_csv_path: str = None,
    regions: Dict[str, Dict[str, float]] = None,
    output_dir: str = None
) -> Dict[str, bool]:
    """Check if all prerequisites are met for extraction operations."""
    prerequisites = {}
    
    # Check CSV files if provided
    if wind_csv_path:
        prerequisites["wind_csv_exists"] = os.path.exists(wind_csv_path)
        if prerequisites["wind_csv_exists"]:
            prerequisites["wind_csv_readable"] = os.access(wind_csv_path, os.R_OK)
    
    if solar_csv_path:
        prerequisites["solar_csv_exists"] = os.path.exists(solar_csv_path)
        if prerequisites["solar_csv_exists"]:
            prerequisites["solar_csv_readable"] = os.access(solar_csv_path, os.R_OK)
    
    # Check regions if provided
    if regions:
        prerequisites["regions_valid"] = all(
            validate_region_bounds(bounds) for bounds in regions.values()
        )
    
    # Check output directory
    if output_dir:
        prerequisites["output_dir_writable"] = os.access(
            os.path.dirname(output_dir) or ".", os.W_OK
        )
    
    # Overall status
    prerequisites["all_prerequisites_met"] = all(prerequisites.values())
    
    return prerequisites


def get_extraction_statistics(output_dir: str) -> Dict[str, Any]:
    """Get statistics about extracted data files."""
    if not os.path.exists(output_dir):
        return {"error": "Output directory does not exist"}
    
    stats = {
        "total_files": 0,
        "total_size_bytes": 0,
        "file_types": {},
        "subdirectories": []
    }
    
    try:
        for root, dirs, files in os.walk(output_dir):
            # Count subdirectories
            if root != output_dir:
                stats["subdirectories"].append(os.path.relpath(root, output_dir))
            
            # Count files and sizes
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                
                stats["total_files"] += 1
                stats["total_size_bytes"] += file_size
                
                # Count file types
                file_ext = os.path.splitext(file)[1].lower()
                stats["file_types"][file_ext] = stats["file_types"].get(file_ext, 0) + 1
        
        # Convert to more readable units
        stats["total_size_mb"] = stats["total_size_bytes"] / (1024 * 1024)
        stats["total_size_gb"] = stats["total_size_mb"] / 1024
        
    except Exception as e:
        stats["error"] = str(e)
    
    return stats
