# Optimization Suggestions

## 🚀 **PERFORMANCE OPTIMIZATIONS**

### **1. Memory Management**
```python
# Current: Basic memory management
gc.collect()

# Improved: Advanced memory management
class MemoryManager:
    def __init__(self, max_memory_gb=50):
        self.max_memory = max_memory_gb * 1024**3
        self.current_usage = 0
    
    def check_memory(self):
        """Monitor memory usage and trigger cleanup if needed"""
        import psutil
        memory = psutil.virtual_memory()
        if memory.percent > 80:
            self.force_cleanup()
    
    def force_cleanup(self):
        """Aggressive memory cleanup"""
        gc.collect()
        import weakref
        weakref.ref(lambda: None)()  # Force weak reference cleanup
```

### **2. Chunked Processing**
```python
# Current: Process all files at once
for file in all_files:
    process_file(file)

# Improved: Chunked processing with memory limits
def process_in_chunks(files, chunk_size=100, max_memory_gb=50):
    """Process files in memory-efficient chunks"""
    for i in range(0, len(files), chunk_size):
        chunk = files[i:i + chunk_size]
        process_chunk(chunk)
        
        # Memory check between chunks
        if memory_usage() > max_memory_gb:
            force_cleanup()
```

### **3. Async I/O for File Operations**
```python
import asyncio
import aiofiles

async def async_grib_reader(file_path):
    """Async GRIB file reading for better I/O performance"""
    async with aiofiles.open(file_path, 'rb') as f:
        content = await f.read()
    return pygrib.open(content)

async def process_files_async(file_paths):
    """Process multiple GRIB files asynchronously"""
    tasks = [async_grib_reader(path) for path in file_paths]
    return await asyncio.gather(*tasks)
```

### **4. Caching and Memoization**
```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def cached_grid_coordinates(date_str):
    """Cache grid coordinates for each date"""
    return get_grid_coordinates(date_str)

def file_hash(file_path):
    """Generate hash for file to detect changes"""
    return hashlib.md5(open(file_path, 'rb').read()).hexdigest()

class FileCache:
    def __init__(self):
        self.cache = {}
    
    def get_or_compute(self, file_path, compute_func):
        """Get cached result or compute and cache"""
        file_hash = self.file_hash(file_path)
        if file_hash not in self.cache:
            self.cache[file_hash] = compute_func(file_path)
        return self.cache[file_hash]
```

### **5. Vectorized Operations**
```python
# Current: Loop-based processing
for i, point in enumerate(points):
    result[i] = process_point(point)

# Improved: Vectorized processing
def vectorized_point_processing(points, grid_data):
    """Vectorized point processing using NumPy"""
    # Convert to numpy arrays for vectorized operations
    points_array = np.array(points)
    grid_array = np.array(grid_data)
    
    # Vectorized nearest neighbor search
    distances = np.linalg.norm(points_array[:, None] - grid_array, axis=2)
    nearest_indices = np.argmin(distances, axis=1)
    
    return nearest_indices
```

### **6. Parallel Processing Improvements**
```python
# Current: Basic multiprocessing
with mp.Pool(processes=num_workers) as pool:
    results = pool.map(process_file, files)

# Improved: Advanced parallel processing
class AdvancedParallelProcessor:
    def __init__(self, max_workers=None, chunk_size=100):
        self.max_workers = max_workers or mp.cpu_count()
        self.chunk_size = chunk_size
        self.executor = None
    
    def __enter__(self):
        self.executor = ProcessPoolExecutor(
            max_workers=self.max_workers,
            mp_context=mp.get_context('spawn')  # More stable on Windows
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.executor:
            self.executor.shutdown(wait=True)
    
    def process_with_progress(self, items, process_func):
        """Process items with progress tracking and error handling"""
        futures = []
        for item in items:
            future = self.executor.submit(process_func, item)
            futures.append(future)
        
        results = []
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                result = future.result(timeout=300)  # 5-minute timeout
                results.append(result)
            except Exception as e:
                logger.error(f"Task failed: {e}")
                results.append(None)
        
        return results
```

## 🔧 **CODE QUALITY IMPROVEMENTS**

### **1. Type Hints and Validation**
```python
from typing import Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field, validator
import numpy as np
import pandas as pd

class ExtractionConfig(BaseModel):
    """Configuration for HRRR data extraction"""
    start_date: datetime
    end_date: datetime
    wind_csv_path: str = Field(..., description="Path to wind locations CSV")
    solar_csv_path: str = Field(..., description="Path to solar locations CSV")
    output_dir: str = Field(default="./output", description="Output directory")
    num_workers: Optional[int] = Field(default=None, description="Number of workers")
    chunk_size: int = Field(default=100, description="Processing chunk size")
    
    @validator('end_date')
    def end_date_after_start(cls, v, values):
        if 'start_date' in values and v <= values['start_date']:
            raise ValueError('end_date must be after start_date')
        return v

class PointData(BaseModel):
    """Data model for point locations"""
    pid: str
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    
    class Config:
        extra = "forbid"  # No additional fields allowed

def extract_data(config: ExtractionConfig) -> Dict[str, pd.DataFrame]:
    """Type-safe data extraction function"""
    # Validate inputs
    config.validate()
    
    # Process with type safety
    wind_points = load_points(config.wind_csv_path, PointData)
    solar_points = load_points(config.solar_csv_path, PointData)
    
    return process_extraction(config, wind_points, solar_points)
```

### **2. Error Handling and Logging**
```python
import logging
from contextlib import contextmanager
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hrrr_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ExtractionError(Exception):
    """Custom exception for extraction errors"""
    pass

@contextmanager
def error_context(operation: str):
    """Context manager for error handling"""
    try:
        yield
    except Exception as e:
        logger.error(f"Error during {operation}: {e}")
        raise ExtractionError(f"{operation} failed: {e}")

def safe_grib_operation(file_path: str, operation: str) -> Optional[Any]:
    """Safely perform GRIB operations with error handling"""
    with error_context(f"GRIB {operation} on {file_path}"):
        with pygrib.open(file_path) as grbs:
            return perform_operation(grbs, operation)
```

### **3. Configuration Management**
```python
from dataclasses import dataclass
from pathlib import Path
import os
from typing import Dict, Any

@dataclass
class SystemConfig:
    """System-specific configuration"""
    cpu_count: int
    memory_gb: float
    platform: str
    
    @classmethod
    def auto_detect(cls) -> 'SystemConfig':
        """Auto-detect system configuration"""
        import psutil
        return cls(
            cpu_count=os.cpu_count(),
            memory_gb=psutil.virtual_memory().total / (1024**3),
            platform=platform.system()
        )

class ConfigManager:
    """Centralized configuration management"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self.system_config = SystemConfig.auto_detect()
        self.settings = self.load_settings()
    
    def load_settings(self) -> Dict[str, Any]:
        """Load settings from file or use defaults"""
        if self.config_file and Path(self.config_file).exists():
            return self.load_from_file(self.config_file)
        return self.get_default_settings()
    
    def get_optimal_workers(self) -> int:
        """Calculate optimal number of workers based on system"""
        # Use 75% of CPU cores, but not more than 32
        optimal = min(int(self.system_config.cpu_count * 0.75), 32)
        return max(optimal, 1)  # At least 1 worker
    
    def get_chunk_size(self) -> int:
        """Calculate optimal chunk size based on memory"""
        memory_gb = self.system_config.memory_gb
        # Conservative chunk size: 1GB per chunk
        return max(50, int(memory_gb * 0.1))
```

## 📊 **MONITORING AND METRICS**

### **1. Performance Monitoring**
```python
import time
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class PerformanceMetrics:
    """Performance metrics for extraction"""
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
        return self.files_processed / self.duration_seconds
    
    @property
    def throughput_points_per_second(self) -> float:
        return self.data_points_extracted / self.duration_seconds

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
        import psutil
        
        metrics = PerformanceMetrics(
            start_time=self.current_start,
            end_time=end_time,
            files_processed=files_processed,
            data_points_extracted=points_extracted,
            memory_peak_gb=psutil.virtual_memory().used / (1024**3),
            cpu_utilization=psutil.cpu_percent()
        )
        
        self.metrics.append(metrics)
        self.current_start = None
    
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
```

### **2. Progress Tracking**
```python
from tqdm import tqdm
import threading
from typing import Optional

class ProgressTracker:
    """Advanced progress tracking with multiple bars"""
    
    def __init__(self, total_files: int, total_points: int):
        self.file_bar = tqdm(total=total_files, desc="Files", position=0)
        self.point_bar = tqdm(total=total_points, desc="Points", position=1)
        self.lock = threading.Lock()
    
    def update_files(self, count: int = 1):
        """Update file progress"""
        with self.lock:
            self.file_bar.update(count)
    
    def update_points(self, count: int = 1):
        """Update point progress"""
        with self.lock:
            self.point_bar.update(count)
    
    def close(self):
        """Close progress bars"""
        self.file_bar.close()
        self.point_bar.close()
```

## 🔄 **RESUME AND RECOVERY**

### **1. Checkpoint System**
```python
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional

class CheckpointManager:
    """Manage extraction checkpoints for resume functionality"""
    
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def save_checkpoint(self, extraction_id: str, state: Dict[str, Any]):
        """Save extraction state to checkpoint"""
        checkpoint_file = self.checkpoint_dir / f"{extraction_id}.pkl"
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(state, f)
        
        # Also save as JSON for human readability
        json_file = self.checkpoint_dir / f"{extraction_id}.json"
        with open(json_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    def load_checkpoint(self, extraction_id: str) -> Optional[Dict[str, Any]]:
        """Load extraction state from checkpoint"""
        checkpoint_file = self.checkpoint_dir / f"{extraction_id}.pkl"
        if checkpoint_file.exists():
            with open(checkpoint_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def list_checkpoints(self) -> List[str]:
        """List available checkpoints"""
        return [f.stem for f in self.checkpoint_dir.glob("*.pkl")]
    
    def delete_checkpoint(self, extraction_id: str):
        """Delete a checkpoint"""
        checkpoint_file = self.checkpoint_dir / f"{extraction_id}.pkl"
        json_file = self.checkpoint_dir / f"{extraction_id}.json"
        
        if checkpoint_file.exists():
            checkpoint_file.unlink()
        if json_file.exists():
            json_file.unlink()
```

### **2. Incremental Processing**
```python
class IncrementalProcessor:
    """Process data incrementally with resume capability"""
    
    def __init__(self, checkpoint_manager: CheckpointManager):
        self.checkpoint_manager = checkpoint_manager
    
    def process_incremental(self, extraction_id: str, files: List[str], 
                          process_func, batch_size: int = 100):
        """Process files incrementally with checkpointing"""
        
        # Load previous state
        state = self.checkpoint_manager.load_checkpoint(extraction_id)
        if state:
            processed_files = set(state.get('processed_files', []))
            results = state.get('results', {})
            start_index = state.get('last_index', 0)
        else:
            processed_files = set()
            results = {}
            start_index = 0
        
        # Process remaining files
        for i in range(start_index, len(files), batch_size):
            batch = files[i:i + batch_size]
            
            for file_path in batch:
                if file_path not in processed_files:
                    try:
                        result = process_func(file_path)
                        results[file_path] = result
                        processed_files.add(file_path)
                        
                        # Save checkpoint every batch
                        self.checkpoint_manager.save_checkpoint(extraction_id, {
                            'processed_files': list(processed_files),
                            'results': results,
                            'last_index': i + len(batch)
                        })
                        
                    except Exception as e:
                        logger.error(f"Error processing {file_path}: {e}")
                        continue
        
        return results
``` 