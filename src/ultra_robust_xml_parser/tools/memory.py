"""Memory management features for Ultra Robust XML Parser.

Provides comprehensive memory management including object pools, memory monitoring,
streaming optimization, garbage collection tuning, and memory leak detection for
production-ready, high-throughput XML processing.
"""

import gc
import sys
import threading
import time
import weakref
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union

from ultra_robust_xml_parser.shared.logging import get_logger

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class MemoryLevel(Enum):
    """Memory usage level indicators."""
    
    LOW = "low"
    NORMAL = "normal" 
    HIGH = "high"
    CRITICAL = "critical"


class PoolType(Enum):
    """Types of object pools."""
    
    STRING_POOL = "string_pool"
    TOKEN_POOL = "token_pool"
    ELEMENT_POOL = "element_pool"
    BUFFER_POOL = "buffer_pool"


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    
    # Process memory information
    resident_memory_mb: float = 0.0
    virtual_memory_mb: float = 0.0
    memory_percent: float = 0.0
    
    # Python heap information
    heap_objects: int = 0
    heap_size_mb: float = 0.0
    gc_collections: Dict[int, int] = field(default_factory=dict)
    
    # Pool usage statistics
    pool_statistics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Alert information
    alerts: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory stats to dictionary representation."""
        return {
            "resident_memory_mb": self.resident_memory_mb,
            "virtual_memory_mb": self.virtual_memory_mb,
            "memory_percent": self.memory_percent,
            "heap_objects": self.heap_objects,
            "heap_size_mb": self.heap_size_mb,
            "gc_collections": self.gc_collections,
            "pool_statistics": self.pool_statistics,
            "alerts": self.alerts,
            "timestamp": self.timestamp
        }


@dataclass
class MemoryAlert:
    """Memory usage alert."""
    
    alert_type: str
    level: MemoryLevel
    message: str
    current_value: float
    threshold: float
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary representation."""
        return {
            "alert_type": self.alert_type,
            "level": self.level.value,
            "message": self.message,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "timestamp": self.timestamp
        }


class ObjectPool:
    """Generic object pool for memory-efficient object reuse.
    
    Provides object pooling to reduce allocation overhead in high-throughput
    scenarios by reusing objects instead of constantly creating and destroying them.
    
    Examples:
        Basic string pool usage:
        >>> pool = ObjectPool("strings", str, max_size=100)
        >>> obj = pool.get()
        >>> pool.return_object(obj)
        
        Custom object pool:
        >>> class Token: pass
        >>> token_pool = ObjectPool("tokens", Token, max_size=50)
        >>> token = token_pool.get()
        >>> token_pool.return_object(token)
    """
    
    def __init__(
        self,
        name: str,
        object_type: Type,
        max_size: int = 100,
        factory: Optional[Callable[[], Any]] = None
    ):
        """Initialize object pool.
        
        Args:
            name: Pool name for identification
            object_type: Type of objects to pool
            max_size: Maximum pool size
            factory: Optional factory function for object creation
        """
        self.name = name
        self.object_type = object_type
        self.max_size = max_size
        self.factory = factory or object_type
        
        self._pool: List[Any] = []
        self._lock = threading.RLock()
        
        # Statistics
        self.created_count = 0
        self.reused_count = 0
        self.returned_count = 0
        
        self.logger = get_logger(__name__, None, f"object_pool_{name}")
    
    def get(self) -> Any:
        """Get object from pool or create new one.
        
        Returns:
            Object from pool or newly created object
        """
        with self._lock:
            if self._pool:
                obj = self._pool.pop()
                self.reused_count += 1
                self.logger.debug(
                    "Object retrieved from pool",
                    extra={
                        "pool_name": self.name,
                        "pool_size": len(self._pool),
                        "reused_count": self.reused_count
                    }
                )
                return obj
            else:
                obj = self.factory()
                self.created_count += 1
                self.logger.debug(
                    "New object created",
                    extra={
                        "pool_name": self.name,
                        "created_count": self.created_count
                    }
                )
                return obj
    
    def return_object(self, obj: Any) -> bool:
        """Return object to pool.
        
        Args:
            obj: Object to return to pool
            
        Returns:
            True if object was returned to pool, False if pool is full
        """
        if not isinstance(obj, self.object_type):
            self.logger.warning(
                "Invalid object type returned to pool",
                extra={
                    "pool_name": self.name,
                    "expected_type": str(self.object_type),
                    "actual_type": str(type(obj))
                }
            )
            return False
        
        with self._lock:
            if len(self._pool) < self.max_size:
                # Reset object state if it has a reset method
                if hasattr(obj, 'reset') and callable(getattr(obj, 'reset')):
                    try:
                        obj.reset()
                    except Exception as e:
                        self.logger.warning(f"Failed to reset object: {e}")
                        return False
                
                self._pool.append(obj)
                self.returned_count += 1
                
                self.logger.debug(
                    "Object returned to pool",
                    extra={
                        "pool_name": self.name,
                        "pool_size": len(self._pool),
                        "returned_count": self.returned_count
                    }
                )
                return True
            else:
                self.logger.debug(
                    "Pool is full, discarding object",
                    extra={"pool_name": self.name, "max_size": self.max_size}
                )
                return False
    
    def clear(self) -> None:
        """Clear all objects from pool."""
        with self._lock:
            cleared_count = len(self._pool)
            self._pool.clear()
            
            self.logger.info(
                "Pool cleared",
                extra={
                    "pool_name": self.name,
                    "cleared_objects": cleared_count
                }
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pool usage statistics.
        
        Returns:
            Dictionary containing pool statistics
        """
        with self._lock:
            return {
                "name": self.name,
                "object_type": str(self.object_type),
                "max_size": self.max_size,
                "current_size": len(self._pool),
                "created_count": self.created_count,
                "reused_count": self.reused_count,
                "returned_count": self.returned_count,
                "efficiency": self.reused_count / max(1, self.created_count + self.reused_count)
            }


class StreamingBuffer:
    """Memory-efficient streaming buffer with automatic sizing.
    
    Provides efficient buffer management for streaming XML processing with
    automatic growth and shrinkage based on usage patterns.
    
    Examples:
        Basic buffer usage:
        >>> buffer = StreamingBuffer(initial_size=1024)
        >>> buffer.write(b"<xml>data</xml>")
        >>> data = buffer.read(4)
        >>> buffer.clear()
        
        Auto-sizing buffer:
        >>> buffer = StreamingBuffer(auto_size=True)
        >>> buffer.write(large_xml_data)
        >>> # Buffer automatically adjusts size
    """
    
    def __init__(
        self,
        initial_size: int = 8192,
        max_size: int = 1024 * 1024,  # 1MB
        auto_size: bool = True
    ):
        """Initialize streaming buffer.
        
        Args:
            initial_size: Initial buffer size in bytes
            max_size: Maximum buffer size in bytes
            auto_size: Whether to automatically adjust buffer size
        """
        self.initial_size = initial_size
        self.max_size = max_size
        self.auto_size = auto_size
        
        self._buffer = bytearray(initial_size)
        self._position = 0
        self._length = 0
        
        # Statistics
        self.total_reads = 0
        self.total_writes = 0
        self.total_bytes_read = 0
        self.total_bytes_written = 0
        self.resize_count = 0
        
        self.logger = get_logger(__name__, None, "streaming_buffer")
    
    def write(self, data: Union[bytes, bytearray]) -> int:
        """Write data to buffer.
        
        Args:
            data: Data to write
            
        Returns:
            Number of bytes written
        """
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("Data must be bytes or bytearray")
        
        data_len = len(data)
        required_size = self._length + data_len
        
        # Resize buffer if needed
        if required_size > len(self._buffer):
            if self.auto_size and required_size <= self.max_size:
                new_size = min(required_size * 2, self.max_size)
                self._resize(new_size)
            elif required_size > self.max_size:
                raise MemoryError(f"Buffer size would exceed maximum {self.max_size}")
        
        # Write data
        self._buffer[self._length:self._length + data_len] = data
        self._length += data_len
        
        # Update statistics
        self.total_writes += 1
        self.total_bytes_written += data_len
        
        return data_len
    
    def read(self, size: int = -1) -> bytes:
        """Read data from buffer.
        
        Args:
            size: Number of bytes to read (-1 for all)
            
        Returns:
            Read data as bytes
        """
        if size == -1:
            size = self._length - self._position
        
        size = min(size, self._length - self._position)
        
        if size <= 0:
            return b""
        
        data = bytes(self._buffer[self._position:self._position + size])
        self._position += size
        
        # Update statistics
        self.total_reads += 1
        self.total_bytes_read += size
        
        return data
    
    def clear(self) -> None:
        """Clear buffer contents."""
        self._position = 0
        self._length = 0
        
        # Resize to initial size if auto-sizing is enabled
        if self.auto_size and len(self._buffer) > self.initial_size * 2:
            self._resize(self.initial_size)
    
    def _resize(self, new_size: int) -> None:
        """Resize buffer to new size.
        
        Args:
            new_size: New buffer size
        """
        old_size = len(self._buffer)
        new_buffer = bytearray(new_size)
        
        # Copy existing data
        copy_size = min(self._length, new_size)
        new_buffer[:copy_size] = self._buffer[:copy_size]
        
        self._buffer = new_buffer
        self._length = min(self._length, new_size)
        self._position = min(self._position, self._length)
        
        self.resize_count += 1
        
        self.logger.debug(
            "Buffer resized",
            extra={
                "old_size": old_size,
                "new_size": new_size,
                "resize_count": self.resize_count
            }
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics.
        
        Returns:
            Dictionary containing buffer statistics
        """
        return {
            "initial_size": self.initial_size,
            "max_size": self.max_size,
            "current_size": len(self._buffer),
            "used_size": self._length,
            "position": self._position,
            "auto_size": self.auto_size,
            "total_reads": self.total_reads,
            "total_writes": self.total_writes,
            "total_bytes_read": self.total_bytes_read,
            "total_bytes_written": self.total_bytes_written,
            "resize_count": self.resize_count
        }


class MemoryManager:
    """Comprehensive memory management system for Ultra Robust XML Parser.
    
    Provides object pooling, memory monitoring, streaming optimization, 
    garbage collection tuning, and memory leak detection for production
    environments with high-throughput XML processing requirements.
    
    Examples:
        Basic memory management:
        >>> memory_manager = MemoryManager()
        >>> memory_manager.start_monitoring()
        >>> stats = memory_manager.get_memory_stats()
        >>> print(f"Memory usage: {stats.memory_percent}%")
        
        Object pooling:
        >>> memory_manager.create_pool("strings", str, max_size=100)
        >>> obj = memory_manager.get_from_pool("strings")
        >>> memory_manager.return_to_pool("strings", obj)
        
        Memory alerts:
        >>> def alert_handler(alert):
        ...     print(f"Alert: {alert.message}")
        >>> memory_manager.add_alert_handler(alert_handler)
    """
    
    def __init__(
        self,
        memory_threshold_mb: float = 500.0,
        critical_threshold_mb: float = 1000.0,
        monitoring_interval: float = 5.0
    ):
        """Initialize memory manager.
        
        Args:
            memory_threshold_mb: Memory usage threshold in MB for alerts
            critical_threshold_mb: Critical memory threshold in MB
            monitoring_interval: Monitoring interval in seconds
        """
        self.memory_threshold_mb = memory_threshold_mb
        self.critical_threshold_mb = critical_threshold_mb
        self.monitoring_interval = monitoring_interval
        
        # Object pools
        self._pools: Dict[str, ObjectPool] = {}
        self._pool_lock = threading.RLock()
        
        # Monitoring
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._alert_handlers: List[Callable[[MemoryAlert], None]] = []
        
        # Statistics
        self._stats_history: List[MemoryStats] = []
        self._max_history_size = 1000
        
        # Leak detection
        self._tracked_objects: Set[weakref.ref] = set()
        self._last_gc_stats = {i: gc.get_count()[i] for i in range(3)}
        
        self.logger = get_logger(__name__, None, "memory_manager")
        
        # Initialize default pools
        self._create_default_pools()
    
    def _create_default_pools(self) -> None:
        """Create default object pools."""
        default_pools = [
            ("strings", str, 200),
            ("lists", list, 100),
            ("dicts", dict, 100),
            ("bytes", bytes, 150),
        ]
        
        for name, obj_type, size in default_pools:
            self.create_pool(name, obj_type, size)
    
    def create_pool(
        self,
        name: str,
        object_type: Type,
        max_size: int = 100,
        factory: Optional[Callable[[], Any]] = None
    ) -> ObjectPool:
        """Create new object pool.
        
        Args:
            name: Pool name
            object_type: Type of objects to pool
            max_size: Maximum pool size
            factory: Optional factory function
            
        Returns:
            Created object pool
        """
        with self._pool_lock:
            if name in self._pools:
                self.logger.warning(f"Pool {name} already exists")
                return self._pools[name]
            
            pool = ObjectPool(name, object_type, max_size, factory)
            self._pools[name] = pool
            
            self.logger.info(
                "Object pool created",
                extra={
                    "pool_name": name,
                    "object_type": str(object_type),
                    "max_size": max_size
                }
            )
            
            return pool
    
    def get_from_pool(self, pool_name: str) -> Any:
        """Get object from specified pool.
        
        Args:
            pool_name: Name of pool
            
        Returns:
            Object from pool
            
        Raises:
            KeyError: If pool doesn't exist
        """
        with self._pool_lock:
            if pool_name not in self._pools:
                raise KeyError(f"Pool {pool_name} does not exist")
            return self._pools[pool_name].get()
    
    def return_to_pool(self, pool_name: str, obj: Any) -> bool:
        """Return object to specified pool.
        
        Args:
            pool_name: Name of pool
            obj: Object to return
            
        Returns:
            True if object was returned successfully
            
        Raises:
            KeyError: If pool doesn't exist
        """
        with self._pool_lock:
            if pool_name not in self._pools:
                raise KeyError(f"Pool {pool_name} does not exist")
            return self._pools[pool_name].return_object(obj)
    
    def clear_pool(self, pool_name: str) -> None:
        """Clear specified pool.
        
        Args:
            pool_name: Name of pool to clear
            
        Raises:
            KeyError: If pool doesn't exist
        """
        with self._pool_lock:
            if pool_name not in self._pools:
                raise KeyError(f"Pool {pool_name} does not exist")
            self._pools[pool_name].clear()
    
    def clear_all_pools(self) -> None:
        """Clear all object pools."""
        with self._pool_lock:
            for pool in self._pools.values():
                pool.clear()
            
            self.logger.info("All pools cleared")
    
    def get_pool_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all pools.
        
        Returns:
            Dictionary containing statistics for each pool
        """
        with self._pool_lock:
            return {name: pool.get_statistics() for name, pool in self._pools.items()}
    
    def start_monitoring(self) -> None:
        """Start memory monitoring."""
        if self._monitoring:
            self.logger.warning("Memory monitoring is already running")
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="MemoryMonitor"
        )
        self._monitor_thread.start()
        
        self.logger.info("Memory monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop memory monitoring."""
        self._monitoring = False
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=1.0)
        
        self.logger.info("Memory monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Memory monitoring loop."""
        while self._monitoring:
            try:
                stats = self.get_memory_stats()
                
                # Store stats history
                self._stats_history.append(stats)
                if len(self._stats_history) > self._max_history_size:
                    self._stats_history.pop(0)
                
                # Check for alerts
                self._check_memory_alerts(stats)
                
                # Sleep until next monitoring cycle
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in memory monitoring: {e}")
                time.sleep(self.monitoring_interval)
    
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics.
        
        Returns:
            Current memory statistics
        """
        stats = MemoryStats()
        
        # Process memory information
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                memory_info = process.memory_info()
                
                stats.resident_memory_mb = memory_info.rss / (1024 * 1024)
                stats.virtual_memory_mb = memory_info.vms / (1024 * 1024)
                stats.memory_percent = process.memory_percent()
                
            except Exception as e:
                self.logger.warning(f"Failed to get process memory info: {e}")
        
        # Python heap information
        stats.heap_objects = len(gc.get_objects())
        stats.heap_size_mb = sys.getsizeof(gc.get_objects()) / (1024 * 1024)
        stats.gc_collections = {i: gc.get_count()[i] for i in range(3)}
        
        # Pool statistics
        stats.pool_statistics = self.get_pool_statistics()
        
        return stats
    
    def _check_memory_alerts(self, stats: MemoryStats) -> None:
        """Check for memory usage alerts.
        
        Args:
            stats: Current memory statistics
        """
        alerts = []
        
        # Check resident memory
        if stats.resident_memory_mb > self.critical_threshold_mb:
            alerts.append(MemoryAlert(
                "memory_usage",
                MemoryLevel.CRITICAL,
                f"Critical memory usage: {stats.resident_memory_mb:.1f}MB",
                stats.resident_memory_mb,
                self.critical_threshold_mb
            ))
        elif stats.resident_memory_mb > self.memory_threshold_mb:
            alerts.append(MemoryAlert(
                "memory_usage",
                MemoryLevel.HIGH,
                f"High memory usage: {stats.resident_memory_mb:.1f}MB",
                stats.resident_memory_mb,
                self.memory_threshold_mb
            ))
        
        # Check heap object count (potential memory leak indicator)
        if stats.heap_objects > 100000:
            alerts.append(MemoryAlert(
                "heap_objects",
                MemoryLevel.HIGH,
                f"High heap object count: {stats.heap_objects}",
                float(stats.heap_objects),
                100000.0
            ))
        
        # Process alerts
        for alert in alerts:
            stats.alerts.append(alert.message)
            self._trigger_alert(alert)
    
    def _trigger_alert(self, alert: MemoryAlert) -> None:
        """Trigger memory alert to registered handlers.
        
        Args:
            alert: Memory alert to trigger
        """
        self.logger.warning(
            "Memory alert triggered",
            extra={
                "alert_type": alert.alert_type,
                "level": alert.level.value,
                "alert_message": alert.message,
                "current_value": alert.current_value,
                "threshold": alert.threshold
            }
        )
        
        # Call alert handlers
        for handler in self._alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Error in alert handler: {e}")
    
    def add_alert_handler(self, handler: Callable[[MemoryAlert], None]) -> None:
        """Add memory alert handler.
        
        Args:
            handler: Function to call when alert is triggered
        """
        self._alert_handlers.append(handler)
    
    def remove_alert_handler(self, handler: Callable[[MemoryAlert], None]) -> None:
        """Remove memory alert handler.
        
        Args:
            handler: Handler function to remove
        """
        if handler in self._alert_handlers:
            self._alert_handlers.remove(handler)
    
    def optimize_gc(self) -> Dict[str, Any]:
        """Optimize garbage collection settings.
        
        Returns:
            Dictionary containing GC optimization results
        """
        # Get current GC settings
        original_thresholds = gc.get_threshold()
        
        # Optimize thresholds for XML processing workload
        # Increase gen0 threshold to reduce frequent collections
        # Adjust gen1/gen2 thresholds for better performance
        new_thresholds = (1000, 15, 15)  # Default: (700, 10, 10)
        
        gc.set_threshold(*new_thresholds)
        
        # Force collection to start fresh
        collected = gc.collect()
        
        self.logger.info(
            "GC optimization applied",
            extra={
                "original_thresholds": original_thresholds,
                "new_thresholds": new_thresholds,
                "collected_objects": collected
            }
        )
        
        return {
            "original_thresholds": original_thresholds,
            "new_thresholds": new_thresholds,
            "collected_objects": collected,
            "current_counts": gc.get_count()
        }
    
    def force_gc(self) -> Dict[str, Any]:
        """Force garbage collection and return statistics.
        
        Returns:
            Dictionary containing GC results
        """
        before_counts = gc.get_count()
        before_objects = len(gc.get_objects())
        
        collected = gc.collect()
        
        after_counts = gc.get_count()
        after_objects = len(gc.get_objects())
        
        result = {
            "before_counts": before_counts,
            "after_counts": after_counts,
            "before_objects": before_objects,
            "after_objects": after_objects,
            "collected_objects": collected,
            "objects_freed": before_objects - after_objects
        }
        
        self.logger.info(
            "Forced garbage collection",
            extra=result
        )
        
        return result
    
    def get_stats_history(self, limit: int = 100) -> List[MemoryStats]:
        """Get memory statistics history.
        
        Args:
            limit: Maximum number of stats to return
            
        Returns:
            List of historical memory statistics
        """
        return self._stats_history[-limit:] if self._stats_history else []
    
    def export_stats(self, output_path: Path, format_type: str = "json") -> None:
        """Export memory statistics to file.
        
        Args:
            output_path: Path to write statistics
            format_type: Export format ('json', 'csv')
        """
        if format_type == "json":
            import json
            
            export_data = {
                "memory_manager_config": {
                    "memory_threshold_mb": self.memory_threshold_mb,
                    "critical_threshold_mb": self.critical_threshold_mb,
                    "monitoring_interval": self.monitoring_interval
                },
                "current_stats": self.get_memory_stats().to_dict(),
                "pool_statistics": self.get_pool_statistics(),
                "stats_history": [stats.to_dict() for stats in self._stats_history]
            }
            
            output_path.write_text(json.dumps(export_data, indent=2))
            
        elif format_type == "csv":
            import csv
            
            with output_path.open('w', newline='') as csvfile:
                fieldnames = [
                    'timestamp', 'resident_memory_mb', 'virtual_memory_mb',
                    'memory_percent', 'heap_objects', 'heap_size_mb', 'alerts'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for stats in self._stats_history:
                    writer.writerow({
                        'timestamp': stats.timestamp,
                        'resident_memory_mb': stats.resident_memory_mb,
                        'virtual_memory_mb': stats.virtual_memory_mb,
                        'memory_percent': stats.memory_percent,
                        'heap_objects': stats.heap_objects,
                        'heap_size_mb': stats.heap_size_mb,
                        'alerts': '; '.join(stats.alerts)
                    })
        
        self.logger.info(
            "Memory statistics exported",
            extra={
                "output_path": str(output_path),
                "format": format_type,
                "stats_count": len(self._stats_history)
            }
        )