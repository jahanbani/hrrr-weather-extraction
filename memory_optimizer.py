#!/usr/bin/env python3
"""
Advanced Memory Management and Optimization for HRRR Extraction
Based on analysis of backup optimization files.
"""

import gc
import time
import weakref
import logging
from contextlib import contextmanager
from typing import Dict, Any, Optional, Callable
import psutil
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class MemoryOptimizer:
    """Advanced memory management and optimization for HRRR extraction."""
    
    def __init__(self, max_memory_gb: float = 200.0, warning_threshold: float = 0.8):
        self.max_memory = max_memory_gb * 1024**3  # Convert to bytes
        self.warning_threshold = warning_threshold
        self.optimization_history = []
        
        # Get system info
        self.total_memory = psutil.virtual_memory().total
        self.available_memory = psutil.virtual_memory().available
        
        logger.info(f"🔍 Memory Optimizer initialized:")
        logger.info(f"   Total system memory: {self.total_memory / (1024**3):.1f} GB")
        logger.info(f"   Available memory: {self.available_memory / (1024**3):.1f} GB")
        logger.info(f"   Max allowed usage: {max_memory_gb:.1f} GB")
        logger.info(f"   Warning threshold: {warning_threshold * 100:.0f}%")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'percent': memory.percent,
            'free_gb': memory.free / (1024**3)
        }
    
    def check_memory(self) -> bool:
        """Check if memory usage is acceptable."""
        memory = psutil.virtual_memory()
        return memory.percent < (self.warning_threshold * 100)
    
    def force_cleanup(self, aggressive: bool = False):
        """Force aggressive memory cleanup."""
        logger.info("🧹 Forcing memory cleanup...")
        
        # Standard cleanup
        collected = gc.collect()
        logger.info(f"   Garbage collected: {collected} objects")
        
        if aggressive:
            # Aggressive cleanup
            gc.collect(2)  # Full collection
            
            # Force weak reference cleanup
            weakref.ref(lambda: None)()
            
            # Clear Python cache
            import sys
            if hasattr(sys, 'intern'):
                sys.intern.clear()
            
            logger.info("   Aggressive cleanup completed")
        
        # Check result
        memory_after = psutil.virtual_memory()
        freed_mb = (memory_after.available - self.available_memory) / (1024**2)
        logger.info(f"   Memory freed: {freed_mb:.1f} MB")
        
        return freed_mb
    
    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage."""
        if df.empty:
            return df
        
        original_memory = df.memory_usage(deep=True).sum()
        
        # Convert numeric columns to float32 where possible
        for col in df.select_dtypes(include=[np.float64]).columns:
            if df[col].notna().all():  # No NaN values
                df[col] = df[col].astype('float32')
        
        # Convert integer columns to smaller types
        for col in df.select_dtypes(include=[np.int64]).columns:
            col_min, col_max = df[col].min(), df[col].max()
            if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype('int8')
            elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype('int16')
            elif col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype('int32')
        
        # Convert object columns to category where beneficial
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique values
                df[col] = df[col].astype('category')
        
        optimized_memory = df.memory_usage(deep=True).sum()
        memory_saved = original_memory - optimized_memory
        savings_percent = (memory_saved / original_memory) * 100
        
        logger.info(f"📊 DataFrame optimized:")
        logger.info(f"   Original memory: {original_memory / (1024**2):.1f} MB")
        logger.info(f"   Optimized memory: {optimized_memory / (1024**2):.1f} MB")
        logger.info(f"   Memory saved: {memory_saved / (1024**2):.1f} MB ({savings_percent:.1f}%)")
        
        return df
    
    @contextmanager
    def memory_monitor(self, operation_name: str, threshold_gb: Optional[float] = None):
        """Context manager for monitoring memory during operations."""
        threshold = threshold_gb or (self.max_memory / (1024**3))
        
        start_time = time.time()
        start_memory = psutil.virtual_memory()
        
        logger.info(f"🔍 Starting memory monitoring for: {operation_name}")
        logger.info(f"   Initial memory: {start_memory.used / (1024**3):.1f} GB")
        
        try:
            yield
            
            # Check final memory state
            end_memory = psutil.virtual_memory()
            end_time = time.time()
            
            memory_used = end_memory.used - start_memory.used
            duration = end_time - start_time
            
            logger.info(f"✅ Operation completed: {operation_name}")
            logger.info(f"   Duration: {duration:.2f}s")
            logger.info(f"   Memory change: {memory_used / (1024**3):+.2f} GB")
            logger.info(f"   Final memory: {end_memory.used / (1024**3):.1f} GB")
            
            # Record optimization
            self.optimization_history.append({
                'operation': operation_name,
                'duration': duration,
                'memory_change_gb': memory_used / (1024**3),
                'timestamp': time.time()
            })
            
        except Exception as e:
            logger.error(f"❌ Operation failed: {operation_name} - {e}")
            raise
        finally:
            # Force cleanup if memory usage is high
            if not self.check_memory():
                logger.warning(f"⚠️  High memory usage detected, forcing cleanup")
                self.force_cleanup(aggressive=True)
    
    def get_optimal_chunk_size(self, memory_per_chunk_mb: float = 100) -> int:
        """Calculate optimal chunk size based on available memory."""
        available_gb = psutil.virtual_memory().available / (1024**3)
        safe_memory_gb = available_gb * 0.7  # Use 70% of available memory
        
        optimal_chunks = int((safe_memory_gb * 1024) / memory_per_chunk_mb)
        
        logger.info(f"🎯 Optimal chunk size calculation:")
        logger.info(f"   Available memory: {available_gb:.1f} GB")
        logger.info(f"   Safe memory usage: {safe_memory_gb:.1f} GB")
        logger.info(f"   Memory per chunk: {memory_per_chunk_mb:.1f} MB")
        logger.info(f"   Optimal chunks: {optimal_chunks}")
        
        return max(1, optimal_chunks)
    
    def monitor_performance(self, interval: float = 5.0):
        """Monitor system performance continuously."""
        logger.info(f"📊 Starting performance monitoring (interval: {interval}s)")
        
        try:
            while True:
                memory = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent(interval=1)
                
                logger.info(f"📈 Performance snapshot:")
                logger.info(f"   CPU: {cpu_percent:.1f}%")
                logger.info(f"   Memory: {memory.percent:.1f}% ({memory.used / (1024**3):.1f} GB)")
                logger.info(f"   Available: {memory.available / (1024**3):.1f} GB")
                
                # Check if cleanup is needed
                if memory.percent > (self.warning_threshold * 100):
                    logger.warning(f"⚠️  High memory usage detected, forcing cleanup")
                    self.force_cleanup(aggressive=True)
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("📊 Performance monitoring stopped")
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization history."""
        if not self.optimization_history:
            return {"message": "No operations recorded"}
        
        total_operations = len(self.optimization_history)
        total_duration = sum(op['duration'] for op in self.optimization_history)
        total_memory_change = sum(op['memory_change_gb'] for op in self.optimization_history)
        
        return {
            'total_operations': total_operations,
            'total_duration_seconds': total_duration,
            'total_memory_change_gb': total_memory_change,
            'average_duration': total_duration / total_operations,
            'average_memory_change': total_memory_change / total_operations,
            'operations': self.optimization_history[-10:]  # Last 10 operations
        }
    
    def clear_history(self):
        """Clear optimization history."""
        self.optimization_history.clear()
        logger.info("🗑️  Optimization history cleared")


# Global instance for easy access
memory_optimizer = MemoryOptimizer()


def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience function to optimize DataFrame memory."""
    return memory_optimizer.optimize_dataframe(df)


@contextmanager
def memory_monitor(operation_name: str, threshold_gb: Optional[float] = None):
    """Convenience context manager for memory monitoring."""
    with memory_optimizer.memory_monitor(operation_name, threshold_gb):
        yield


def force_memory_cleanup(aggressive: bool = False):
    """Convenience function to force memory cleanup."""
    return memory_optimizer.force_cleanup(aggressive)


if __name__ == "__main__":
    # Test the memory optimizer
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testing Memory Optimizer...")
    
    # Create test data
    test_df = pd.DataFrame({
        'float64_col': np.random.randn(10000).astype('float64'),
        'int64_col': np.random.randint(0, 1000, 10000).astype('int64'),
        'object_col': ['category_' + str(i % 10) for i in range(10000)]
    })
    
    print(f"Original DataFrame memory: {test_df.memory_usage(deep=True).sum() / (1024**2):.1f} MB")
    
    # Test optimization
    with memory_monitor("DataFrame optimization"):
        optimized_df = optimize_dataframe_memory(test_df)
    
    print(f"Optimized DataFrame memory: {optimized_df.memory_usage(deep=True).sum() / (1024**2):.1f} MB")
    
    # Test memory cleanup
    force_memory_cleanup()
    
    # Show summary
    summary = memory_optimizer.get_optimization_summary()
    print(f"Optimization summary: {summary}")
