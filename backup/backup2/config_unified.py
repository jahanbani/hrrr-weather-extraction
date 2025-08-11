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
    handlers=[logging.FileHandler("hrrr_extraction.log"), logging.StreamHandler()],
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
            self.end_date = datetime(2024, 1, 1, 23, 0, 0)

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
