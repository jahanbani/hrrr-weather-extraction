#!/usr/bin/env python3
"""
Examples demonstrating complex geometry support for HRRR data extraction.

This script shows how to use:
1. Polygon coordinates (irregular shapes)
2. Point buffers (circular regions)
3. GeoJSON files
4. Shapefiles
5. Mixed geometry types

Run this script to test the enhanced geometry support.
"""

import sys
import os
from datetime import datetime, timedelta

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config_unified import HRRRConfig
from region_extraction import extract_multiple_regions_quarterly_optimized
from geometry_support import EXAMPLE_REGIONS, check_geometry_dependencies

def example_complex_geometries():
    """
    Demonstrate complex geometry support with various region types.
    """
    
    print("🗺️  COMPLEX GEOMETRY SUPPORT EXAMPLES")
    print("=" * 50)
    
    # Check if geometry support is available
    try:
        check_geometry_dependencies()
        print("✅ Advanced geometry support available (shapely + geopandas)")
    except ImportError as e:
        print(f"❌ {e}")
        print("🔄 Falling back to rectangular regions only")
        
    # Load configuration
    config = HRRRConfig()
    
    # Set a shorter date range for testing (1 day)
    START = datetime(2023, 1, 1, 0, 0)
    END = datetime(2023, 1, 1, 23, 0)
    
    print(f"\n📅 Test period: {START} to {END}")
    
    # Define various geometry types for testing
    test_regions = {
        # 1. Traditional rectangle (backward compatibility)
        "texas_rectangle": {
            "lat_min": 25.8, "lat_max": 36.5,
            "lon_min": -106.5, "lon_max": -93.5
        },
        
        # 2. Irregular polygon (more precise state boundary)
        "texas_polygon": {
            "type": "polygon",
            "coordinates": [
                [-106.5, 25.8],   # SW corner
                [-93.5, 25.8],    # SE corner
                [-93.5, 29.0],    # East coast bend
                [-94.0, 29.8],    # Louisiana border
                [-94.2, 32.0],    # East border
                [-94.0, 36.5],    # NE corner
                [-103.0, 36.5],   # North border
                [-106.5, 32.0],   # West border
                [-106.5, 25.8]    # Close polygon
            ]
        },
        
        # 3. Circular buffer around Houston
        "houston_metro": {
            "type": "point_buffer",
            "lon": -95.3698,  # Houston longitude
            "lat": 29.7604,   # Houston latitude  
            "radius_km": 100.0  # 100 km radius
        },
        
        # 4. Circular buffer around Dallas
        "dallas_metro": {
            "type": "point_buffer", 
            "lon": -96.7970,  # Dallas longitude
            "lat": 32.7767,   # Dallas latitude
            "radius_km": 75.0   # 75 km radius
        },
        
        # 5. Custom California coastal region
        "california_coast": {
            "type": "polygon",
            "coordinates": [
                [-124.0, 32.5],   # San Diego area
                [-117.0, 32.5],   # Inland from San Diego
                [-117.0, 34.0],   # Los Angeles inland
                [-118.5, 34.5],   # Los Angeles area
                [-120.0, 35.5],   # Central coast
                [-122.0, 37.0],   # San Francisco inland
                [-124.0, 37.5],   # San Francisco coast
                [-124.5, 39.0],   # Northern coast
                [-124.5, 32.5],   # Back to start
                [-124.0, 32.5]    # Close polygon
            ]
        }
    }
    
    print(f"\n🌍 Testing {len(test_regions)} different geometry types:")
    for name, region_def in test_regions.items():
        region_type = region_def.get("type", "rectangle")
        print(f"   • {name}: {region_type}")
    
    # Get GRIB data path
    try:
        from prereise.gather.const import get_grib_data_path
        DATADIR = get_grib_data_path()
        
        if DATADIR is None:
            print("❌ No GRIB data path found!")
            return
            
        print(f"\n📁 Using GRIB data from: {DATADIR}")
        
        # Test the extraction
        print("\n🚀 Starting complex geometry extraction...")
        results = extract_multiple_regions_quarterly_optimized(
            regions=test_regions,
            START=START,
            END=END, 
            DATADIR=DATADIR,
            DEFAULT_HOURS_FORECASTED=[0, 1],  # f00 and f01 files
            wind_selectors=config.wind_selectors,
            solar_selectors=config.solar_selectors,
            base_output_dir="./test_complex_geometries",
            compression="snappy",
            use_parallel=True,
            num_workers=4,
            enable_resume=True
        )
        
        # Print results summary
        print("\n📊 EXTRACTION RESULTS:")
        print("=" * 50)
        
        for region_name, result in results.items():
            if result and result.get('status') == 'completed':
                grid_points = result.get('grid_points', 0)
                variables = len(result.get('variables_saved', set()))
                processing_time = result.get('processing_time_seconds', 0)
                
                print(f"✅ {region_name}:")
                print(f"   Grid points: {grid_points:,}")
                print(f"   Variables: {variables}")
                print(f"   Time: {processing_time:.1f}s")
                
                # Show geometry-specific info
                if 'geometry_info' in result:
                    geo_info = result['geometry_info']
                    print(f"   Geometry: {geo_info.get('type', 'unknown')}")
                    if 'bounds' in geo_info:
                        bounds = geo_info['bounds']
                        print(f"   Bounds: ({bounds['lat_min']:.2f}, {bounds['lon_min']:.2f}) to ({bounds['lat_max']:.2f}, {bounds['lon_max']:.2f})")
            else:
                print(f"❌ {region_name}: Failed or no data")
                
        print("\n✅ Complex geometry testing completed!")
        print("\nOutput files saved to: ./test_complex_geometries/")
        print("Each region has its own subdirectory with variable-specific data.")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


def example_file_based_geometries():
    """
    Example of using GeoJSON and Shapefile geometries.
    
    Note: This requires actual geometry files to exist.
    """
    print("\n🗂️  FILE-BASED GEOMETRY EXAMPLES")
    print("=" * 50)
    
    # Example region definitions using files
    file_regions = {
        # GeoJSON example
        "state_from_geojson": {
            "type": "geojson",
            "file_path": "texas.geojson",
            "feature_index": 0  # Use first feature in file
        },
        
        # Shapefile example
        "county_from_shapefile": {
            "type": "shapefile", 
            "file_path": "harris_county.shp",
            "feature_index": 0  # Use first feature in file
        },
        
        # Multiple features example
        "multiple_counties": {
            "type": "geojson",
            "file_path": "texas_counties.geojson",
            "feature_index": 45  # Specific county
        }
    }
    
    print("File-based geometry definitions:")
    for name, region_def in file_regions.items():
        print(f"   • {name}:")
        print(f"     Type: {region_def['type']}")
        print(f"     File: {region_def['file_path']}")
        print(f"     Feature: {region_def.get('feature_index', 0)}")
    
    print("\n💡 To use file-based geometries:")
    print("   1. Install dependencies: pip install shapely geopandas")
    print("   2. Download or create GeoJSON/Shapefile for your region")
    print("   3. Update the file_path in the region definition")
    print("   4. Run the extraction as normal")
    
    print("\n🌐 Good sources for geometry files:")
    print("   • Natural Earth: https://www.naturalearthdata.com/")
    print("   • US Census Bureau: https://www.census.gov/geographies/")
    print("   • OpenStreetMap: https://www.openstreetmap.org/")
    print("   • State/local government GIS portals")


def create_sample_geojson():
    """
    Create a sample GeoJSON file for testing.
    """
    
    sample_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "name": "Sample Texas Region",
                    "area": "custom"
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-106.5, 25.8],   # SW corner
                        [-93.5, 25.8],    # SE corner
                        [-93.5, 29.0],    # East coast bend
                        [-94.0, 29.8],    # Louisiana border  
                        [-94.2, 32.0],    # East border
                        [-94.0, 36.5],    # NE corner
                        [-103.0, 36.5],   # North border
                        [-106.5, 32.0],   # West border
                        [-106.5, 25.8]    # Close polygon
                    ]]
                }
            }
        ]
    }
    
    import json
    with open("sample_texas.geojson", "w") as f:
        json.dump(sample_geojson, f, indent=2)
    
    print("✅ Created sample_texas.geojson for testing")
    return "sample_texas.geojson"


if __name__ == "__main__":
    print("🗺️  HRRR COMPLEX GEOMETRY SUPPORT")
    print("=" * 60)
    
    # Test basic complex geometries
    example_complex_geometries()
    
    # Show file-based examples
    example_file_based_geometries()
    
    # Create sample file
    print(f"\n📄 Creating sample GeoJSON file...")
    sample_file = create_sample_geojson()
    
    print(f"\n🎯 QUICK START:")
    print(f"   1. Install: pip install shapely geopandas")
    print(f"   2. Run: python example_complex_geometries.py")
    print(f"   3. Check output: ./test_complex_geometries/")
    
    print(f"\n✨ Advanced geometry support is now available!")
    print(f"   • Supports polygons, circles, GeoJSON, and shapefiles")
    print(f"   • Backward compatible with rectangular regions") 
    print(f"   • Automatically falls back if geometry libraries unavailable")
