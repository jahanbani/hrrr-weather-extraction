# HRRR Package Structure Recommendation

## 🏗️ **Proposed Package Structure**

```
hrrr_extraction/
├── README.md                           # Main documentation
├── setup.py                           # Package installation
├── requirements.txt                   # Dependencies
├── pyproject.toml                    # Modern Python packaging
├── .gitignore                        # Git ignore rules
├── .env.example                      # Environment variables template
├── Makefile                          # Build and development commands
│
├── src/
│   └── hrrr_extraction/
│       ├── __init__.py               # Package initialization
│       ├── core/                     # Core extraction logic
│       │   ├── __init__.py
│       │   ├── extractor.py          # Main extraction engine
│       │   ├── parallel.py           # Parallel processing
│       │   ├── grid.py               # Grid operations
│       │   ├── grib_reader.py        # GRIB file handling
│       │   └── memory_manager.py     # Memory management
│       │
│       ├── config/                   # Configuration management
│       │   ├── __init__.py
│       │   ├── settings.py           # Main settings
│       │   ├── paths.py              # Path detection
│       │   ├── variables.py          # Variable definitions
│       │   └── validation.py         # Config validation
│       │
│       ├── utils/                    # Utility functions
│       │   ├── __init__.py
│       │   ├── data_utils.py         # Data manipulation
│       │   ├── file_utils.py         # File operations
│       │   ├── validation.py         # Input validation
│       │   ├── monitoring.py         # Performance monitoring
│       │   └── logging.py            # Logging setup
│       │
│       ├── cli/                      # Command-line interface
│       │   ├── __init__.py
│       │   ├── main.py               # CLI entry point
│       │   ├── commands.py           # CLI commands
│       │   └── helpers.py            # CLI utilities
│       │
│       ├── models/                   # Data models
│       │   ├── __init__.py
│       │   ├── data_models.py        # Pydantic models
│       │   └── schemas.py            # Data schemas
│       │
│       └── exceptions/               # Custom exceptions
│           ├── __init__.py
│           ├── base.py               # Base exceptions
│           └── specific.py           # Specific error types
│
├── tests/                            # Test suite
│   ├── __init__.py
│   ├── conftest.py                   # Test configuration
│   ├── unit/                         # Unit tests
│   │   ├── __init__.py
│   │   ├── test_extractor.py
│   │   ├── test_parallel.py
│   │   ├── test_config.py
│   │   ├── test_grid.py
│   │   └── test_grib_reader.py
│   ├── integration/                  # Integration tests
│   │   ├── __init__.py
│   │   ├── test_full_workflow.py
│   │   ├── test_performance.py
│   │   └── test_memory_usage.py
│   ├── fixtures/                     # Test data
│   │   ├── __init__.py
│   │   ├── sample_data/
│   │   │   ├── wind.csv
│   │   │   ├── solar.csv
│   │   │   └── test_grib/
│   │   └── mock_files/
│   └── benchmarks/                   # Performance benchmarks
│       ├── __init__.py
│       ├── test_memory_benchmark.py
│       └── test_speed_benchmark.py
│
├── docs/                             # Documentation
│   ├── api.md                        # API documentation
│   ├── installation.md               # Installation guide
│   ├── usage.md                      # Usage examples
│   ├── performance.md                # Performance tuning
│   ├── troubleshooting.md            # Common issues
│   ├── configuration.md              # Configuration guide
│   └── development.md                # Development guide
│
├── examples/                         # Usage examples
│   ├── basic_extraction.py
│   ├── parallel_extraction.py
│   ├── custom_variables.py
│   ├── performance_comparison.py
│   ├── memory_optimization.py
│   └── cli_usage.py
│
├── scripts/                          # Utility scripts
│   ├── install_dependencies.sh
│   ├── setup_environment.sh
│   ├── run_tests.sh
│   ├── benchmark.sh
│   └── cleanup.sh
│
├── data/                             # Data files
│   ├── sample/                       # Sample data
│   │   ├── wind.csv
│   │   ├── solar.csv
│   │   └── config/
│   │       ├── default.yaml
│   │       └── high_performance.yaml
│   └── output/                       # Output directory
│       ├── wind/
│       ├── solar/
│       └── full_grid/
│
└── config/                           # Configuration files
    ├── default.yaml                  # Default configuration
    ├── development.yaml              # Development settings
    ├── production.yaml               # Production settings
    └── test.yaml                     # Test settings
```

## 🔧 **Key Improvements**

### **1. Modular Architecture**
```python
# src/hrrr_extraction/core/extractor.py
from typing import Dict, Any, Optional
from datetime import datetime
from ..config.settings import ExtractionConfig
from ..utils.monitoring import PerformanceMonitor
from ..exceptions.base import HRRRError

class HRRRExtractor:
    """Main extraction engine with unified interface"""
    
    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.monitor = PerformanceMonitor()
        self.memory_manager = MemoryManager(config.max_memory_gb)
    
    def extract_specific_points(self) -> Dict[str, Any]:
        """Extract data for specific wind and solar locations"""
        try:
            self.monitor.start_operation()
            # Implementation here
            return result
        except Exception as e:
            raise HRRRError(f"Extraction failed: {e}")
        finally:
            self.monitor.end_operation()
```

### **2. Configuration Management**
```python
# src/hrrr_extraction/config/settings.py
from dataclasses import dataclass
from typing import Dict, Optional
from datetime import datetime
import os

@dataclass
class ExtractionConfig:
    """Unified configuration with validation"""
    
    # Required parameters
    wind_csv_path: str
    solar_csv_path: str
    start_date: datetime
    end_date: datetime
    
    # Optional parameters with defaults
    num_workers: Optional[int] = None
    max_memory_gb: float = 50.0
    chunk_size: int = 100
    use_parallel: bool = True
    enable_resume: bool = True
    
    # Output settings
    output_base_dir: str = "/research/alij/extracted_data"
    compression: str = "snappy"
    
    # Variable selectors
    wind_selectors: Dict[str, str] = None
    solar_selectors: Dict[str, str] = None
    
    def __post_init__(self):
        """Initialize defaults and validate"""
        self._set_defaults()
        self._validate()
    
    def _set_defaults(self):
        """Set default values"""
        if self.num_workers is None:
            self.num_workers = min(os.cpu_count() or 4, 32)
        
        if self.wind_selectors is None:
            self.wind_selectors = {
                "UWind80": "u",
                "VWind80": "v",
            }
        
        if self.solar_selectors is None:
            self.solar_selectors = {
                "rad": "dswrf",
                "vbd": "vbdsf",
                "vdd": "vddsf",
                "2tmp": "2t",
                "UWind10": "10u",
                "VWind10": "10v",
            }
    
    def _validate(self):
        """Validate configuration settings"""
        errors = []
        
        # File validation
        if not os.path.exists(self.wind_csv_path):
            errors.append(f"Wind CSV not found: {self.wind_csv_path}")
        
        if not os.path.exists(self.solar_csv_path):
            errors.append(f"Solar CSV not found: {self.solar_csv_path}")
        
        # Date validation
        if self.start_date >= self.end_date:
            errors.append("Start date must be before end date")
        
        # Memory validation
        if self.max_memory_gb <= 0:
            errors.append("Max memory must be positive")
        
        if errors:
            raise ValueError(f"Configuration errors: {'; '.join(errors)}")
```

### **3. Error Handling**
```python
# src/hrrr_extraction/exceptions/base.py
class HRRRError(Exception):
    """Base exception for HRRR extraction"""
    pass

class ConfigurationError(HRRRError):
    """Configuration related errors"""
    pass

class GRIBFileError(HRRRError):
    """GRIB file related errors"""
    pass

class MemoryError(HRRRError):
    """Memory related errors"""
    pass

class ValidationError(HRRRError):
    """Validation related errors"""
    pass
```

### **4. Performance Monitoring**
```python
# src/hrrr_extraction/utils/monitoring.py
import time
import psutil
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class PerformanceMetrics:
    """Performance metrics for operations"""
    duration_seconds: float
    files_processed: int
    data_points_extracted: int
    memory_peak_gb: float
    cpu_utilization: float
    
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
    
    def end_operation(self, files_processed: int, points_extracted: int):
        """End timing and record metrics"""
        if self.current_start is None:
            return
        
        end_time = time.time()
        duration = end_time - self.current_start
        
        # Get system metrics
        memory_gb = psutil.virtual_memory().used / (1024**3)
        cpu_percent = psutil.cpu_percent()
        
        metrics = PerformanceMetrics(
            duration_seconds=duration,
            files_processed=files_processed,
            data_points_extracted=points_extracted,
            memory_peak_gb=memory_gb,
            cpu_utilization=cpu_percent
        )
        
        self.metrics.append(metrics)
        self.current_start = None
```

### **5. CLI Interface**
```python
# src/hrrr_extraction/cli/main.py
import click
from typing import Optional
from ..config.settings import ExtractionConfig
from ..core.extractor import HRRRExtractor

@click.group()
@click.option('--config', '-c', help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.pass_context
def cli(ctx, config: Optional[str], verbose: bool):
    """HRRR Data Extraction Tool"""
    ctx.ensure_object(dict)
    ctx.obj['config'] = config
    ctx.obj['verbose'] = verbose

@cli.command()
@click.option('--wind-csv', required=True, help='Wind locations CSV file')
@click.option('--solar-csv', required=True, help='Solar locations CSV file')
@click.option('--start-date', required=True, help='Start date (YYYY-MM-DD)')
@click.option('--end-date', required=True, help='End date (YYYY-MM-DD)')
@click.option('--workers', default=0, help='Number of workers (0=auto)')
@click.option('--memory-gb', default=50.0, help='Max memory usage in GB')
@click.pass_context
def extract(ctx, wind_csv: str, solar_csv: str, start_date: str, 
           end_date: str, workers: int, memory_gb: float):
    """Extract HRRR data for specific locations"""
    
    # Parse dates
    from datetime import datetime
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Create configuration
    config = ExtractionConfig(
        wind_csv_path=wind_csv,
        solar_csv_path=solar_csv,
        start_date=start_dt,
        end_date=end_dt,
        num_workers=workers if workers > 0 else None,
        max_memory_gb=memory_gb
    )
    
    # Run extraction
    extractor = HRRRExtractor(config)
    result = extractor.extract_specific_points()
    
    click.echo(f"✅ Extraction completed successfully!")
    click.echo(f"📊 Processed {result['files_processed']} files")
    click.echo(f"⏱️  Duration: {result['duration_seconds']:.1f} seconds")

@cli.command()
@click.option('--output-dir', required=True, help='Output directory')
@click.pass_context
def validate(ctx, output_dir: str):
    """Validate extracted data"""
    # Implementation here
    pass

@cli.command()
@click.option('--config-file', required=True, help='Configuration file')
@click.pass_context
def benchmark(ctx, config_file: str):
    """Run performance benchmarks"""
    # Implementation here
    pass
```

## 📦 **Package Installation**

### **setup.py**
```python
from setuptools import setup, find_packages

setup(
    name="hrrr-extraction",
    version="1.0.0",
    description="High-performance HRRR data extraction tool",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "pygrib>=2.1.0",
        "scipy>=1.7.0",
        "tqdm>=4.62.0",
        "psutil>=5.8.0",
        "click>=8.0.0",
        "pyyaml>=6.0",
        "python-dotenv>=0.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "hrrr-extract=hrrr_extraction.cli.main:cli",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
```

### **pyproject.toml**
```toml
[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "hrrr-extraction"
version = "1.0.0"
description = "High-performance HRRR data extraction tool"
readme = "README.md"
requires-python = ">=3.8"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "your.email@example.com"},
]
keywords = ["weather", "data", "extraction", "hrrr", "grib"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
]

dependencies = [
    "numpy>=1.21.0",
    "pandas>=1.3.0",
    "pygrib>=2.1.0",
    "scipy>=1.7.0",
    "tqdm>=4.62.0",
    "psutil>=5.8.0",
    "click>=8.0.0",
    "pyyaml>=6.0",
    "python-dotenv>=0.19.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=6.0.0",
    "pytest-cov>=2.12.0",
    "black>=21.0.0",
    "flake8>=3.9.0",
    "mypy>=0.910",
]
docs = [
    "sphinx>=4.0.0",
    "sphinx-rtd-theme>=1.0.0",
]

[project.scripts]
hrrr-extract = "hrrr_extraction.cli.main:cli"

[tool.black]
line-length = 88
target-version = ['py38']

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--strict-markers",
    "--strict-config",
    "--cov=src/hrrr_extraction",
    "--cov-report=term-missing",
    "--cov-report=html",
]
```

## 🚀 **Migration Strategy**

### **Phase 1: Foundation (Week 1-2)**
1. Create new package structure
2. Move core functionality to new modules
3. Implement unified configuration
4. Add basic error handling

### **Phase 2: Enhancement (Week 3-4)**
1. Add performance monitoring
2. Implement CLI interface
3. Add comprehensive testing
4. Create documentation

### **Phase 3: Optimization (Week 5-6)**
1. Add memory management
2. Implement chunked processing
3. Add benchmarking tools
4. Performance tuning

### **Phase 4: Deployment (Week 7-8)**
1. Package distribution
2. CI/CD pipeline
3. Documentation website
4. User training

## 📊 **Benefits of New Structure**

### **1. Maintainability**
- Clear separation of concerns
- Modular design for easy testing
- Consistent coding standards
- Comprehensive error handling

### **2. Usability**
- Intuitive CLI interface
- Configuration file support
- Environment-based settings
- Comprehensive documentation

### **3. Performance**
- Memory-aware processing
- Parallel processing optimization
- Performance monitoring
- Benchmarking tools

### **4. Extensibility**
- Plugin architecture ready
- Easy to add new features
- Version control friendly
- CI/CD ready

## 🎯 **Success Metrics**

### **Code Quality**
- [ ] 90% test coverage
- [ ] Zero critical security issues
- [ ] All linting checks pass
- [ ] Type checking complete

### **Performance**
- [ ] 50% reduction in memory usage
- [ ] 30% improvement in speed
- [ ] 99% uptime during extractions
- [ ] < 1% error rate

### **Usability**
- [ ] Complete API documentation
- [ ] User-friendly error messages
- [ ] Intuitive CLI interface
- [ ] Comprehensive examples 