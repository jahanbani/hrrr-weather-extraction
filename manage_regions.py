#!/usr/bin/env python3
"""
Region management utility for HRRR data extraction.

This script demonstrates how to manage regions through the configuration file.
You can add, remove, list, and test different geometry types.
"""

import sys
import os
from datetime import datetime
from typing import Dict, Any, List

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config_unified import HRRRConfig, save_config_to_file

def list_regions(config: HRRRConfig):
    """List all configured regions grouped by type."""
    print("🗺️  CONFIGURED REGIONS")
    print("=" * 50)
    
    region_types = config.list_region_types()
    
    for geometry_type, region_names in region_types.items():
        print(f"\n📐 {geometry_type.upper()} regions ({len(region_names)}):")
        for name in region_names:
            region_def = config.regions[name]
            
            if geometry_type == "rectangle":
                print(f"   • {name}: ({region_def['lat_min']:.1f}, {region_def['lon_min']:.1f}) to ({region_def['lat_max']:.1f}, {region_def['lon_max']:.1f})")
            elif geometry_type == "point_buffer":
                print(f"   • {name}: Circle at ({region_def['lat']:.2f}, {region_def['lon']:.2f}), radius {region_def['radius_km']}km")
            elif geometry_type == "polygon":
                coords = region_def['coordinates']
                print(f"   • {name}: Polygon with {len(coords)} vertices")
            elif geometry_type in ["geojson", "shapefile"]:
                file_path = region_def.get('file_path', 'unknown')
                print(f"   • {name}: From {file_path}")
            else:
                print(f"   • {name}: {region_def}")
    
    print(f"\n📊 Total: {len(config.regions)} regions")


def add_custom_region(config: HRRRConfig):
    """Interactive function to add a custom region."""
    print("\n➕ ADD CUSTOM REGION")
    print("=" * 30)
    
    name = input("Region name: ").strip()
    if not name:
        print("❌ Name cannot be empty")
        return
        
    if name in config.regions:
        overwrite = input(f"Region '{name}' exists. Overwrite? (y/N): ").strip().lower()
        if overwrite != 'y':
            print("❌ Cancelled")
            return
    
    print("\nGeometry type:")
    print("1. Rectangle (lat/lon bounds)")
    print("2. Point buffer (circle around coordinates)")
    print("3. Polygon (list of coordinates)")
    print("4. GeoJSON file")
    print("5. Shapefile")
    
    choice = input("Choose type (1-5): ").strip()
    
    if choice == "1":
        # Rectangle
        try:
            lat_min = float(input("Minimum latitude: "))
            lat_max = float(input("Maximum latitude: "))
            lon_min = float(input("Minimum longitude: "))
            lon_max = float(input("Maximum longitude: "))
            
            region_def = {
                "lat_min": lat_min, "lat_max": lat_max,
                "lon_min": lon_min, "lon_max": lon_max
            }
        except ValueError:
            print("❌ Invalid coordinates")
            return
            
    elif choice == "2":
        # Point buffer
        try:
            lat = float(input("Center latitude: "))
            lon = float(input("Center longitude: "))
            radius = float(input("Radius (km): "))
            
            region_def = {
                "type": "point_buffer",
                "lat": lat, "lon": lon,
                "radius_km": radius
            }
        except ValueError:
            print("❌ Invalid values")
            return
            
    elif choice == "3":
        # Polygon
        print("Enter polygon coordinates as lat,lon pairs (press Enter when done):")
        coordinates = []
        while True:
            coord_input = input(f"Point {len(coordinates)+1} (lat,lon): ").strip()
            if not coord_input:
                break
            try:
                lat, lon = map(float, coord_input.split(','))
                coordinates.append([lon, lat])  # Note: GeoJSON uses [lon, lat] order
            except:
                print("❌ Invalid format. Use: lat,lon")
                continue
        
        if len(coordinates) < 3:
            print("❌ Polygon needs at least 3 points")
            return
            
        # Close polygon if needed
        if coordinates[0] != coordinates[-1]:
            coordinates.append(coordinates[0])
            
        region_def = {
            "type": "polygon",
            "coordinates": coordinates
        }
        
    elif choice == "4":
        # GeoJSON file
        file_path = input("GeoJSON file path: ").strip()
        try:
            feature_index = int(input("Feature index (0 for first): ") or "0")
        except ValueError:
            feature_index = 0
            
        region_def = {
            "type": "geojson",
            "file_path": file_path,
            "feature_index": feature_index
        }
        
    elif choice == "5":
        # Shapefile
        file_path = input("Shapefile path (.shp): ").strip()
        try:
            feature_index = int(input("Feature index (0 for first): ") or "0")
        except ValueError:
            feature_index = 0
            
        region_def = {
            "type": "shapefile",
            "file_path": file_path,
            "feature_index": feature_index
        }
        
    else:
        print("❌ Invalid choice")
        return
    
    # Add description
    description = input("Description (optional): ").strip()
    if description:
        region_def["description"] = description
    
    # Add region to config
    config.add_region(name, region_def)
    print(f"✅ Added region '{name}'")


def remove_region(config: HRRRConfig):
    """Remove a region from config."""
    print("\n🗑️  REMOVE REGION")
    print("=" * 30)
    
    if not config.regions:
        print("No regions to remove")
        return
    
    print("Available regions:")
    for i, name in enumerate(config.regions.keys(), 1):
        print(f"  {i}. {name}")
    
    try:
        choice = int(input("\nSelect region to remove (number): "))
        region_names = list(config.regions.keys())
        if 1 <= choice <= len(region_names):
            name = region_names[choice - 1]
            confirm = input(f"Remove '{name}'? (y/N): ").strip().lower()
            if confirm == 'y':
                config.remove_region(name)
                print(f"✅ Removed region '{name}'")
            else:
                print("❌ Cancelled")
        else:
            print("❌ Invalid selection")
    except ValueError:
        print("❌ Invalid input")


def save_config(config: HRRRConfig):
    """Save configuration to file."""
    filename = input("\nSave config to file (default: custom_regions.json): ").strip()
    if not filename:
        filename = "custom_regions.json"
    
    try:
        save_config_to_file(config, filename)
        print(f"✅ Configuration saved to {filename}")
    except Exception as e:
        print(f"❌ Error saving config: {e}")


def test_regions(config: HRRRConfig):
    """Test region extraction with selected regions."""
    print("\n🧪 TEST REGION EXTRACTION")
    print("=" * 40)
    
    if not config.regions:
        print("No regions configured")
        return
    
    print("Available regions:")
    region_names = list(config.regions.keys())
    for i, name in enumerate(region_names, 1):
        region_def = config.regions[name]
        region_type = region_def.get("type", "rectangle")
        print(f"  {i}. {name} ({region_type})")
    
    try:
        choice = input("\nSelect regions to test (comma-separated numbers, or 'all'): ").strip()
        
        if choice.lower() == 'all':
            selected_regions = {name: config.regions[name] for name in region_names}
        else:
            indices = [int(x.strip()) for x in choice.split(',')]
            selected_regions = {}
            for idx in indices:
                if 1 <= idx <= len(region_names):
                    name = region_names[idx - 1]
                    selected_regions[name] = config.regions[name]
        
        if not selected_regions:
            print("❌ No valid regions selected")
            return
        
        print(f"\n🚀 Testing {len(selected_regions)} regions...")
        print("Selected regions:")
        for name in selected_regions.keys():
            print(f"   • {name}")
        
        # Set a short test period
        config.start_date = datetime(2023, 1, 1, 0, 0)
        config.end_date = datetime(2023, 1, 1, 3, 0)  # Just 4 hours for testing
        
        # Import and run test
        from extraction_core import extract_multiple_regions_optimized
        from prereise_essentials import get_grib_data_path
        
        grib_path = get_grib_data_path()
        if not grib_path:
            print("❌ GRIB data path not found")
            return
        
        results = extract_multiple_regions_quarterly_optimized(
            regions=selected_regions,
            START=config.start_date,
            END=config.end_date,
            DATADIR=grib_path,
            DEFAULT_HOURS_FORECASTED=[0, 1],
            wind_selectors=config.wind_selectors,
            solar_selectors=config.solar_selectors,
            base_output_dir="./test_custom_regions",
            compression="snappy",
            use_parallel=True,
            num_workers=min(4, len(selected_regions)),
            enable_resume=True
        )
        
        print("\n📊 TEST RESULTS:")
        for name, result in results.items():
            if result and result.get('status') == 'completed':
                grid_points = result.get('grid_points', 0)
                print(f"✅ {name}: {grid_points:,} grid points processed")
            else:
                print(f"❌ {name}: Failed")
        
    except ValueError:
        print("❌ Invalid input")
    except Exception as e:
        print(f"❌ Error during testing: {e}")


def main():
    """Main menu for region management."""
    print("🗺️  HRRR REGION MANAGEMENT")
    print("=" * 50)
    
    # Load configuration
    config = HRRRConfig()
    
    while True:
        print(f"\n📍 Current regions: {len(config.regions)}")
        print("\nOptions:")
        print("1. List all regions")
        print("2. Add custom region")
        print("3. Remove region")
        print("4. Save configuration")
        print("5. Test regions")
        print("6. Quit")
        
        choice = input("\nSelect option (1-6): ").strip()
        
        if choice == "1":
            list_regions(config)
        elif choice == "2":
            add_custom_region(config)
        elif choice == "3":
            remove_region(config)
        elif choice == "4":
            save_config(config)
        elif choice == "5":
            test_regions(config)
        elif choice == "6":
            break
        else:
            print("❌ Invalid choice")
    
    print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()
