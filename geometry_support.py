"""
Enhanced geometry support for HRRR data extraction.

This module provides support for complex geometries beyond simple rectangular bounds:
- Polygon coordinates (list of lat/lon points)
- GeoJSON files
- Shapefiles
- Multiple geometry types (Point, Polygon, MultiPolygon)

Dependencies:
    pip install shapely geopandas

Usage Examples:
    # Polygon from coordinates
    texas_polygon = [
        [-106.5, 25.8],  # SW corner
        [-93.5, 25.8],   # SE corner  
        [-93.5, 36.5],   # NE corner
        [-106.5, 36.5],  # NW corner
        [-106.5, 25.8]   # Close polygon
    ]
    
    # GeoJSON file
    geometry = load_geometry_from_geojson("texas.geojson")
    
    # Shapefile
    geometry = load_geometry_from_shapefile("texas.shp")
"""

import os
import logging
from typing import Dict, List, Tuple, Union, Any, Optional
import numpy as np
import pandas as pd

try:
    from shapely.geometry import Point, Polygon, MultiPolygon, shape
    from shapely.ops import unary_union
    import geopandas as gpd
    GEOMETRY_SUPPORT = True
except ImportError:
    GEOMETRY_SUPPORT = False
    Point = Polygon = MultiPolygon = shape = unary_union = gpd = None

logger = logging.getLogger(__name__)


class GeometryType:
    """Supported geometry types."""
    RECTANGLE = "rectangle"
    POLYGON = "polygon" 
    GEOJSON = "geojson"
    SHAPEFILE = "shapefile"
    POINT_BUFFER = "point_buffer"


def check_geometry_dependencies():
    """Check if geometry dependencies are available."""
    if not GEOMETRY_SUPPORT:
        raise ImportError(
            "Geometry support requires additional dependencies. Install with:\n"
            "pip install shapely geopandas"
        )
    return True


def load_geometry_from_coordinates(coords: List[List[float]]) -> Polygon:
    """
    Load geometry from a list of coordinate pairs.
    
    Args:
        coords: List of [lon, lat] coordinate pairs defining polygon vertices
        
    Returns:
        Polygon: Shapely polygon object
        
    Example:
        texas_coords = [
            [-106.5, 25.8],  # SW corner
            [-93.5, 25.8],   # SE corner
            [-93.5, 36.5],   # NE corner  
            [-106.5, 36.5],  # NW corner
            [-106.5, 25.8]   # Close polygon
        ]
        polygon = load_geometry_from_coordinates(texas_coords)
    """
    check_geometry_dependencies()
    
    if len(coords) < 3:
        raise ValueError("Polygon must have at least 3 coordinates")
    
    # Ensure polygon is closed
    if coords[0] != coords[-1]:
        coords = coords + [coords[0]]
    
    # Create Shapely polygon (note: Shapely uses (lon, lat) order)
    return Polygon(coords)


def load_geometry_from_geojson(file_path: str, feature_index: int = 0) -> Union[Polygon, MultiPolygon]:
    """
    Load geometry from a GeoJSON file.
    
    Args:
        file_path: Path to GeoJSON file
        feature_index: Index of feature to use (if multiple features exist)
        
    Returns:
        Union[Polygon, MultiPolygon]: Shapely geometry object
    """
    check_geometry_dependencies()
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"GeoJSON file not found: {file_path}")
    
    # Load with geopandas
    gdf = gpd.read_file(file_path)
    
    if len(gdf) == 0:
        raise ValueError(f"No features found in GeoJSON file: {file_path}")
    
    if feature_index >= len(gdf):
        raise ValueError(f"Feature index {feature_index} out of range. File has {len(gdf)} features.")
    
    geometry = gdf.geometry.iloc[feature_index]
    
    # Convert to WGS84 if needed
    if gdf.crs and gdf.crs != 'EPSG:4326':
        logger.info(f"Converting from {gdf.crs} to WGS84")
        gdf = gdf.to_crs('EPSG:4326')
        geometry = gdf.geometry.iloc[feature_index]
    
    return geometry


def load_geometry_from_shapefile(file_path: str, feature_index: int = 0) -> Union[Polygon, MultiPolygon]:
    """
    Load geometry from a shapefile.
    
    Args:
        file_path: Path to shapefile (.shp)
        feature_index: Index of feature to use (if multiple features exist)
        
    Returns:
        Union[Polygon, MultiPolygon]: Shapely geometry object
    """
    check_geometry_dependencies()
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Shapefile not found: {file_path}")
    
    # Load with geopandas
    gdf = gpd.read_file(file_path)
    
    if len(gdf) == 0:
        raise ValueError(f"No features found in shapefile: {file_path}")
    
    if feature_index >= len(gdf):
        raise ValueError(f"Feature index {feature_index} out of range. File has {len(gdf)} features.")
    
    geometry = gdf.geometry.iloc[feature_index]
    
    # Convert to WGS84 if needed
    if gdf.crs and gdf.crs != 'EPSG:4326':
        logger.info(f"Converting from {gdf.crs} to WGS84")
        gdf = gdf.to_crs('EPSG:4326')
        geometry = gdf.geometry.iloc[feature_index]
    
    return geometry


def create_point_buffer(lon: float, lat: float, radius_km: float) -> Polygon:
    """
    Create a circular buffer around a point.
    
    Args:
        lon: Longitude of center point
        lat: Latitude of center point
        radius_km: Radius in kilometers
        
    Returns:
        Polygon: Circular buffer polygon
    """
    check_geometry_dependencies()
    
    # Create point
    point = Point(lon, lat)
    
    # Convert km to degrees (approximate)
    # 1 degree ≈ 111 km at equator
    radius_deg = radius_km / 111.0
    
    # Create buffer
    return point.buffer(radius_deg)


def filter_grid_points_by_geometry(
    grid_lats: np.ndarray,
    grid_lons: np.ndarray, 
    geometry: Union[Polygon, MultiPolygon]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter grid points that fall within a complex geometry.
    
    Args:
        grid_lats: 2D array of latitude values
        grid_lons: 2D array of longitude values
        geometry: Shapely geometry object
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (lat_indices, lon_indices) of points within geometry
    """
    check_geometry_dependencies()
    
    logger.info(f"Filtering grid points using {type(geometry).__name__}...")
    
    # Get grid shape
    n_lats, n_lons = grid_lats.shape
    
    # Flatten grids for vectorized operations
    flat_lats = grid_lats.flatten()
    flat_lons = grid_lons.flatten()
    
    # Create points and test containment vectorized
    points = [Point(lon, lat) for lon, lat in zip(flat_lons, flat_lats)]
    
    if isinstance(geometry, MultiPolygon):
        # For MultiPolygon, check if point is in any of the polygons
        mask = np.array([any(geom.contains(point) for geom in geometry.geoms) for point in points])
    else:
        # For single Polygon
        mask = np.array([geometry.contains(point) for point in points])
    
    # Convert back to 2D indices
    valid_indices = np.where(mask)[0]
    lat_indices = valid_indices // n_lons
    lon_indices = valid_indices % n_lons
    
    logger.info(f"Found {len(lat_indices)} grid points within geometry")
    
    return lat_indices, lon_indices


def parse_region_definition(region_def: Dict[str, Any]) -> Union[Polygon, MultiPolygon]:
    """
    Parse a region definition and return a Shapely geometry.
    
    Args:
        region_def: Dictionary defining the region geometry
        
    Supported formats:
        # Rectangle (existing format)
        {
            "type": "rectangle", 
            "lat_min": 25.8, "lat_max": 36.5,
            "lon_min": -106.5, "lon_max": -93.5
        }
        
        # Polygon coordinates  
        {
            "type": "polygon",
            "coordinates": [[-106.5, 25.8], [-93.5, 25.8], [-93.5, 36.5], [-106.5, 36.5], [-106.5, 25.8]]
        }
        
        # GeoJSON file
        {
            "type": "geojson",
            "file_path": "texas.geojson",
            "feature_index": 0  # optional
        }
        
        # Shapefile
        {
            "type": "shapefile", 
            "file_path": "texas.shp",
            "feature_index": 0  # optional
        }
        
        # Point buffer
        {
            "type": "point_buffer",
            "lon": -100.0, "lat": 31.0,
            "radius_km": 50.0
        }
    
    Returns:
        Union[Polygon, MultiPolygon]: Shapely geometry object
    """
    
    geometry_type = region_def.get("type", "rectangle")
    
    if geometry_type == GeometryType.RECTANGLE:
        # Convert rectangle to polygon for consistent handling
        coords = [
            [region_def["lon_min"], region_def["lat_min"]],  # SW
            [region_def["lon_max"], region_def["lat_min"]],  # SE
            [region_def["lon_max"], region_def["lat_max"]],  # NE
            [region_def["lon_min"], region_def["lat_max"]],  # NW
            [region_def["lon_min"], region_def["lat_min"]]   # Close
        ]
        return load_geometry_from_coordinates(coords)
        
    elif geometry_type == GeometryType.POLYGON:
        return load_geometry_from_coordinates(region_def["coordinates"])
        
    elif geometry_type == GeometryType.GEOJSON:
        feature_index = region_def.get("feature_index", 0)
        return load_geometry_from_geojson(region_def["file_path"], feature_index)
        
    elif geometry_type == GeometryType.SHAPEFILE:
        feature_index = region_def.get("feature_index", 0)
        return load_geometry_from_shapefile(region_def["file_path"], feature_index)
        
    elif geometry_type == GeometryType.POINT_BUFFER:
        return create_point_buffer(
            region_def["lon"], region_def["lat"], region_def["radius_km"]
        )
        
    else:
        raise ValueError(f"Unsupported geometry type: {geometry_type}")


def get_geometry_bounds(geometry: Union[Polygon, MultiPolygon]) -> Dict[str, float]:
    """
    Get bounding box of a geometry.
    
    Args:
        geometry: Shapely geometry object
        
    Returns:
        Dict[str, float]: Bounding box with lat/lon min/max
    """
    minx, miny, maxx, maxy = geometry.bounds
    
    return {
        "lon_min": minx,
        "lat_min": miny, 
        "lon_max": maxx,
        "lat_max": maxy
    }


def validate_geometry(geometry: Union[Polygon, MultiPolygon]) -> bool:
    """
    Validate that a geometry is suitable for grid filtering.
    
    Args:
        geometry: Shapely geometry object
        
    Returns:
        bool: True if geometry is valid
    """
    if not geometry.is_valid:
        logger.error("Geometry is not valid")
        return False
        
    if geometry.is_empty:
        logger.error("Geometry is empty")
        return False
    
    bounds = get_geometry_bounds(geometry)
    
    # Check if bounds are reasonable for Earth coordinates
    if not (-180 <= bounds["lon_min"] <= 180 and -180 <= bounds["lon_max"] <= 180):
        logger.error(f"Invalid longitude bounds: {bounds['lon_min']} to {bounds['lon_max']}")
        return False
        
    if not (-90 <= bounds["lat_min"] <= 90 and -90 <= bounds["lat_max"] <= 90):
        logger.error(f"Invalid latitude bounds: {bounds['lat_min']} to {bounds['lat_max']}")
        return False
        
    logger.info(f"Geometry validation passed. Bounds: {bounds}")
    return True


# Example region definitions for testing
EXAMPLE_REGIONS = {
    "texas_rectangle": {
        "type": "rectangle",
        "lat_min": 25.8, "lat_max": 36.5,
        "lon_min": -106.5, "lon_max": -93.5
    },
    
    "texas_polygon": {
        "type": "polygon", 
        "coordinates": [
            [-106.5, 25.8],  # SW corner
            [-93.5, 25.8],   # SE corner
            [-93.5, 36.5],   # NE corner
            [-106.5, 36.5],  # NW corner
            [-106.5, 25.8]   # Close polygon
        ]
    },
    
    "houston_buffer": {
        "type": "point_buffer",
        "lon": -95.3698, "lat": 29.7604,  # Houston coordinates
        "radius_km": 100.0
    },
    
    # Example for file-based geometries (files not included)
    "california_geojson": {
        "type": "geojson",
        "file_path": "california.geojson",
        "feature_index": 0
    },
    
    "florida_shapefile": {
        "type": "shapefile", 
        "file_path": "florida.shp",
        "feature_index": 0
    }
}


if __name__ == "__main__":
    # Test geometry support
    print("Testing geometry support...")
    
    try:
        check_geometry_dependencies()
        print("✅ Geometry dependencies available")
        
        # Test polygon creation
        texas_coords = EXAMPLE_REGIONS["texas_polygon"]["coordinates"]
        polygon = load_geometry_from_coordinates(texas_coords)
        print(f"✅ Created Texas polygon: {polygon.area:.2f} square degrees")
        
        # Test point buffer
        houston_buffer = create_point_buffer(-95.3698, 29.7604, 100.0)
        print(f"✅ Created Houston buffer: {houston_buffer.area:.2f} square degrees")
        
        # Test region parsing
        for name, region_def in EXAMPLE_REGIONS.items():
            if region_def["type"] in ["geojson", "shapefile"]:
                continue  # Skip file-based tests without actual files
                
            try:
                geometry = parse_region_definition(region_def)
                bounds = get_geometry_bounds(geometry)
                print(f"✅ {name}: {bounds}")
            except Exception as e:
                print(f"⚠️  {name}: {e}")
                
    except ImportError as e:
        print(f"❌ {e}")
