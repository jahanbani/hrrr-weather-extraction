#!/usr/bin/env python3
"""
Unified configuration for HRRR data extraction.
This consolidates all settings from config.py and adds new features.
"""

import logging
import os
import platform
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("hrrr.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class HRRRConfig:
    """Unified configuration for HRRR extraction with validation and defaults"""

    # Paths
    wind_csv_path: str = "wind.csv"
    solar_csv_path: str = "solar.csv"
    # output_base_dir: str = "./extracted_data"  # Use local directory for Windows
    output_base_dir: str = (
        "/research/alij/extracted_data"  # Use local directory for Windows
    )

    # Processing
    num_workers: Optional[int] = None  # Auto-detect - will use ALL CPUs
    chunk_size: int = 50000  # Optimized for 36 CPUs
    use_parallel: bool = True
    enable_resume: bool = True
    max_memory_gb: float = 200.0  # Use 200GB out of 256GB (leave some for OS)

    # Variables
    wind_selectors: Dict[str, str] = field(
        default_factory=lambda: {
            "UWind80": "U component of wind",  # U component of wind at 80m
            "VWind80": "V component of wind",  # V component of wind at 80m
        }
    )

    # Solar variables to extract
    solar_selectors: Dict[str, str] = field(
        default_factory=lambda: {
            "UWind10": "10 metre U wind component",  # 10 metre U wind component
            "VWind10": "10 metre V wind component",  # 10 metre V wind component
            # Mean surface downward short-wave radiation flux
            "rad": "Mean surface downward short-wave radiation flux",
            "vbd": "Visible Beam Downward Solar Flux",  # Visible Beam Downward Solar Flux
            # Visible Diffuse Downward Solar Flux
            "vdd": "Visible Diffuse Downward Solar Flux",
            "2tmp": "2 metre temperature",  # 2 metre temperature
        }
    )

    # Date ranges
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

    # Constants from prereise.gather.const
    TZ: int = 8  # Timezone offset for data processing
    DATADIR: str = "/research/alij/hrrr"
    SEARCHSTRING: str = "V[B,D]DSF|DSWRF|TMP:2 m|(?:U|V)GRD:(?:10|80) m"
    SELECTORS: Dict[str, str] = field(
        default_factory=lambda: {
            "UWind80": "u",
            "VWind80": "v",
            "UWind10": "10u",
            "VWind10": "10v",
            "rad": "dswrf",
            "vbd": "vbdsf",
            "vdd": "vddsf",
            "2tmp": "2t",
        }
    )

    # Performance
    enable_monitoring: bool = True
    enable_progress_tracking: bool = True
    enable_memory_checks: bool = True

    # Region definitions for complex geometry support
    regions: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {})

    # Active regions - specify which regions to actually use
    active_regions: Optional[List[str]] = (
        None  # None = use all, or specify list like ["spp_all"]
    )

    def __post_init__(self):
        """Initialize defaults after object creation"""
        # Auto-detect optimal number of workers - use ALL available CPUs
        if self.num_workers is None:
            self.num_workers = os.cpu_count() or 40  # Use all CPUs, no artificial limit

        # Set default wind selectors (using full GRIB variable names)
        if not self.wind_selectors:
            self.wind_selectors = {
                "UWind80": "U component of wind",
                "VWind80": "V component of wind",
            }

        # Set default solar selectors (using full GRIB variable names)
        if not self.solar_selectors:
            self.solar_selectors = {
                "rad": "Downward short-wave radiation flux",
                "vbd": "Visible Beam Downward Solar Flux",
                "vdd": "Visible Diffuse Downward Solar Flux",
                "2tmp": "2 metre temperature",
                "UWind10": "10 metre U wind component",
                "VWind10": "10 metre V wind component",
            }

        # Set default dates if not provided
        if self.start_date is None:
            self.start_date = datetime(2022, 12, 31, 0, 0, 0)
        if self.end_date is None:
            self.end_date = datetime(2023, 1, 1, 0, 0, 0)

        # Set default regions if not provided
        if not self.regions:
            self.regions = self._get_default_regions()

        # Set default active regions to SPP only for production use
        if self.active_regions is None:
            self.active_regions = ["spp_all"]  # Default to SPP only

    def validate(self) -> bool:
        """Validate configuration settings"""
        errors = []

        # Check file paths
        if not os.path.exists(self.wind_csv_path):
            errors.append(f"Wind CSV file not found: {self.wind_csv_path}")

        if not os.path.exists(self.solar_csv_path):
            errors.append(f"Solar CSV file not found: {self.solar_csv_path}")

        # Check date range
        if self.start_date and self.end_date and self.start_date >= self.end_date:
            errors.append("Start date must be before end date")

        # Check memory settings
        if self.max_memory_gb <= 0:
            errors.append("Max memory must be positive")

        if errors:
            raise ValueError(f"Configuration errors: {'; '.join(errors)}")

        return True

    def _get_default_regions(self) -> Dict[str, Dict[str, Any]]:
        """Get default region definitions with various geometry types"""
        return {
            # Traditional rectangular regions (backward compatibility)
            "texas": {
                "lat_min": 25.8,
                "lat_max": 36.5,
                "lon_min": -106.5,
                "lon_max": -93.5,
            },
            "california": {
                "lat_min": 32.5,
                "lat_max": 42.0,
                "lon_min": -124.5,
                "lon_max": -114.0,
            },
            "florida": {
                "lat_min": 24.5,
                "lat_max": 31.0,
                "lon_min": -87.5,
                "lon_max": -80.0,
            },
            # SPP (Southwest Power Pool) footprint
            "spp_all": {
                "lat_min": 30.0,  # catches southernmost WEIS/RTO nodes in TX/LA
                "lat_max": 47.0,  # covers northern SPP in the Dakotas
                "lon_min": -106.0,  # covers western WEIS edge in NM/CO
                "lon_max": -88.0,  # ensures easternmost RTO points in AR/MO/TN are in
                "description": "Southwest Power Pool complete footprint including WEIS and RTO regions",
            },
            # Complex geometry examples (require shapely/geopandas)
            "texas_polygon": {
                "type": "polygon",
                "coordinates": [
                    [-106.5, 25.8],  # SW corner
                    [-93.5, 25.8],  # SE corner
                    [-93.5, 29.0],  # East coast bend
                    [-94.0, 29.8],  # Louisiana border
                    [-94.2, 32.0],  # East border
                    [-94.0, 36.5],  # NE corner
                    [-103.0, 36.5],  # North border
                    [-106.5, 32.0],  # West border
                    [-106.5, 25.8],  # Close polygon
                ],
            },
            "houston_metro": {
                "type": "point_buffer",
                "lon": -95.3698,  # Houston coordinates
                "lat": 29.7604,
                "radius_km": 100.0,  # 100 km radius
            },
            "dallas_metro": {
                "type": "point_buffer",
                "lon": -96.7970,  # Dallas coordinates
                "lat": 32.7767,
                "radius_km": 75.0,  # 75 km radius
            },
            "california_coast": {
                "type": "polygon",
                "coordinates": [
                    [-124.0, 32.5],  # San Diego area
                    [-117.0, 32.5],  # Inland from San Diego
                    [-117.0, 34.0],  # Los Angeles inland
                    [-118.5, 34.5],  # Los Angeles area
                    [-120.0, 35.5],  # Central coast
                    [-122.0, 37.0],  # San Francisco inland
                    [-124.0, 37.5],  # San Francisco coast
                    [-124.5, 39.0],  # Northern coast
                    [-124.5, 32.5],  # Back to start
                    [-124.0, 32.5],  # Close polygon
                ],
            },
            # File-based geometry examples (files need to exist)
            "state_from_geojson": {
                "type": "geojson",
                "file_path": "geometries/texas.geojson",
                "feature_index": 0,
                "description": "Load Texas boundary from GeoJSON file",
            },
            "county_from_shapefile": {
                "type": "shapefile",
                "file_path": "geometries/harris_county.shp",
                "feature_index": 0,
                "description": "Load Harris County from shapefile",
            },
        }

    def get_output_dirs(self) -> Dict[str, str]:
        """Get output directory paths"""
        return {
            "wind": os.path.join(self.output_base_dir, "wind"),
            "solar": os.path.join(self.output_base_dir, "solar"),
            "full_grid": os.path.join(self.output_base_dir, "full_grid"),
        }

    def get_data_directory(self) -> str:
        """Get the appropriate data directory based on OS"""
        system = platform.system()
        if system == "Windows":
            return "data"
        elif system == "Linux":
            return "/research/alij/hrrr"
        else:
            return "/research/alij/hrrr"  # Default to Linux path

    def get_regions(
        self, region_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get region definitions, respecting active_regions setting.

        Args:
            region_names: List of region names to return. If None, uses active_regions.

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of region definitions
        """
        # Determine which regions to return
        if region_names is None:
            # Use active_regions if set, otherwise all regions
            target_regions = (
                self.active_regions
                if self.active_regions
                else list(self.regions.keys())
            )
        else:
            target_regions = region_names

        return {
            name: self.regions[name] for name in target_regions if name in self.regions
        }

    def add_region(self, name: str, region_def: Dict[str, Any]) -> None:
        """
        Add a new region definition.

        Args:
            name: Region name
            region_def: Region definition dictionary
        """
        self.regions[name] = region_def

    def remove_region(self, name: str) -> bool:
        """
        Remove a region definition.

        Args:
            name: Region name to remove

        Returns:
            bool: True if region was removed, False if not found
        """
        if name in self.regions:
            del self.regions[name]
            return True
        return False

    def list_region_types(self) -> Dict[str, List[str]]:
        """
        List regions grouped by their geometry type.

        Returns:
            Dict[str, List[str]]: Dictionary mapping geometry types to region names
        """
        types = {}
        for name, region_def in self.regions.items():
            region_type = region_def.get("type", "rectangle")
            if region_type not in types:
                types[region_type] = []
            types[region_type].append(name)
        return types

    def set_active_regions(self, region_names: List[str]) -> None:
        """
        Set which regions are active for extraction.

        Args:
            region_names: List of region names to activate
        """
        # Validate that all requested regions exist
        missing = [name for name in region_names if name not in self.regions]
        if missing:
            raise ValueError(
                f"Unknown regions: {missing}. Available: {list(self.regions.keys())}"
            )

        self.active_regions = region_names

    def get_active_regions(self) -> List[str]:
        """Get list of currently active region names."""
        return self.active_regions or list(self.regions.keys())

    def use_spp_only(self) -> None:
        """Convenience method to set active regions to SPP only."""
        self.set_active_regions(["spp_all"])

    def use_all_regions(self) -> None:
        """Convenience method to activate all available regions."""
        self.active_regions = None  # None means use all


# Default configuration instance
DEFAULT_CONFIG = HRRRConfig()


def load_config_from_file(config_path: str) -> HRRRConfig:
    """Load configuration from JSON/YAML file"""
    import json

    import yaml

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        if config_path.endswith(".json"):
            config_data = json.load(f)
        elif config_path.endswith(".yaml") or config_path.endswith(".yml"):
            config_data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path}")

    return HRRRConfig(**config_data)


def save_config_to_file(config: HRRRConfig, config_path: str):
    """Save configuration to JSON file"""
    import json

    config_dict = {
        "wind_csv_path": config.wind_csv_path,
        "solar_csv_path": config.solar_csv_path,
        "output_base_dir": config.output_base_dir,
        "num_workers": config.num_workers,
        "chunk_size": config.chunk_size,
        "use_parallel": config.use_parallel,
        "enable_resume": config.enable_resume,
        "max_memory_gb": config.max_memory_gb,
        "wind_selectors": config.wind_selectors,
        "solar_selectors": config.solar_selectors,
        "enable_monitoring": config.enable_monitoring,
        "enable_progress_tracking": config.enable_progress_tracking,
        "enable_memory_checks": config.enable_memory_checks,
        "regions": config.regions,
    }

    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2, default=str)


def get_constants():
    """Get constants from the default configuration"""
    return {
        "START": DEFAULT_CONFIG.start_date,
        "END": DEFAULT_CONFIG.end_date,
        "TZ": DEFAULT_CONFIG.TZ,
        "DATADIR": DEFAULT_CONFIG.DATADIR,
        "SEARCHSTRING": DEFAULT_CONFIG.SEARCHSTRING,
        "SELECTORS": DEFAULT_CONFIG.SELECTORS,
    }


# Backward compatibility - import existing config values
try:
    from config import (
        DEFAULT_COMPRESSION,
        DEFAULT_HOURS_FORECASTED,
        DEFAULT_SOLAR_SELECTORS,
        DEFAULT_WIND_SELECTORS,
        FULL_GRID_OUTPUT_DIR,
        SOLAR_OUTPUT_DIR,
        WIND_OUTPUT_DIR,
    )

    # Update default config with existing values
    DEFAULT_CONFIG.wind_selectors.update(DEFAULT_WIND_SELECTORS)
    DEFAULT_CONFIG.solar_selectors.update(DEFAULT_SOLAR_SELECTORS)

except ImportError:
    logger.warning("Could not import existing config.py - using defaults")


if __name__ == "__main__":
    # Test configuration
    config = HRRRConfig()
    print("✅ Configuration loaded successfully")
    print(f"Wind CSV: {config.wind_csv_path}")
    print(f"Solar CSV: {config.solar_csv_path}")
    print(f"Workers: {config.num_workers}")
    print(f"Wind selectors: {config.wind_selectors}")
    print(f"Solar selectors: {config.solar_selectors}")
