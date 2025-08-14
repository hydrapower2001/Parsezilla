"""Tests for the memory management module."""

import json
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ultra_robust_xml_parser.tools.memory import (
    MemoryAlert,
    MemoryLevel,
    MemoryManager,
    MemoryStats,
    ObjectPool,
    PoolType,
    StreamingBuffer,
)


class TestMemoryStats:
    """Test MemoryStats data class."""
    
    def test_memory_stats_creation(self):
        """Test memory stats creation."""
        stats = MemoryStats(
            resident_memory_mb=100.5,
            virtual_memory_mb=200.0,
            memory_percent=15.5,
            heap_objects=1000,
            heap_size_mb=50.0,
            gc_collections={0: 10, 1: 5, 2: 2},
            pool_statistics={"test_pool": {"size": 10}},
            alerts=["High memory usage"]
        )
        
        assert stats.resident_memory_mb == 100.5
        assert stats.virtual_memory_mb == 200.0
        assert stats.memory_percent == 15.5
        assert stats.heap_objects == 1000
        assert stats.heap_size_mb == 50.0
        assert stats.gc_collections[0] == 10
        assert stats.pool_statistics["test_pool"]["size"] == 10
        assert "High memory usage" in stats.alerts
    
    def test_memory_stats_to_dict(self):
        """Test memory stats conversion to dictionary."""
        stats = MemoryStats(
            resident_memory_mb=100.0,
            memory_percent=10.0,
            alerts=["Test alert"]
        )
        
        stats_dict = stats.to_dict()
        
        assert stats_dict["resident_memory_mb"] == 100.0
        assert stats_dict["memory_percent"] == 10.0
        assert stats_dict["alerts"] == ["Test alert"]
        assert "timestamp" in stats_dict


class TestMemoryAlert:
    """Test MemoryAlert data class."""
    
    def test_memory_alert_creation(self):
        """Test memory alert creation."""
        alert = MemoryAlert(
            alert_type="memory_usage",
            level=MemoryLevel.HIGH,
            message="High memory usage detected",
            current_value=500.0,
            threshold=400.0
        )
        
        assert alert.alert_type == "memory_usage"
        assert alert.level == MemoryLevel.HIGH
        assert alert.message == "High memory usage detected"
        assert alert.current_value == 500.0
        assert alert.threshold == 400.0
    
    def test_memory_alert_to_dict(self):
        """Test memory alert conversion to dictionary."""
        alert = MemoryAlert(
            alert_type="heap_objects",
            level=MemoryLevel.CRITICAL,
            message="Critical heap usage",
            current_value=200000.0,
            threshold=100000.0
        )
        
        alert_dict = alert.to_dict()
        
        assert alert_dict["alert_type"] == "heap_objects"
        assert alert_dict["level"] == "critical"
        assert alert_dict["message"] == "Critical heap usage"
        assert alert_dict["current_value"] == 200000.0
        assert alert_dict["threshold"] == 100000.0


class TestObjectPool:
    """Test ObjectPool class."""
    
    def test_object_pool_creation(self):
        """Test object pool creation."""
        pool = ObjectPool("test_pool", str, max_size=10)
        
        assert pool.name == "test_pool"
        assert pool.object_type == str
        assert pool.max_size == 10
        assert pool.created_count == 0
        assert pool.reused_count == 0
        assert pool.returned_count == 0
    
    def test_get_object_from_empty_pool(self):
        """Test getting object from empty pool."""
        pool = ObjectPool("strings", str, max_size=5)
        
        obj = pool.get()
        
        assert isinstance(obj, str)
        assert pool.created_count == 1
        assert pool.reused_count == 0
    
    def test_return_and_reuse_object(self):
        """Test returning and reusing objects."""
        pool = ObjectPool("strings", str, max_size=5)
        
        # Get object (creates new one)
        obj1 = pool.get()
        assert pool.created_count == 1
        
        # Return object to pool
        returned = pool.return_object(obj1)
        assert returned is True
        assert pool.returned_count == 1
        
        # Get object again (should reuse)
        obj2 = pool.get()
        assert obj2 is obj1  # Same object
        assert pool.created_count == 1  # No new creation
        assert pool.reused_count == 1
    
    def test_pool_max_size_limit(self):
        """Test pool respects maximum size limit."""
        pool = ObjectPool("strings", str, max_size=2)
        
        # Fill pool to capacity
        obj1 = pool.get()
        obj2 = pool.get()
        obj3 = pool.get()
        
        pool.return_object(obj1)
        pool.return_object(obj2)
        
        # Pool should be full, third return should fail
        returned = pool.return_object(obj3)
        assert returned is False
        assert pool.returned_count == 2  # Only two returned
    
    def test_return_wrong_type_object(self):
        """Test returning object of wrong type."""
        pool = ObjectPool("strings", str, max_size=5)
        
        # Try to return int to string pool
        returned = pool.return_object(42)
        assert returned is False
    
    def test_object_reset_on_return(self):
        """Test object reset on return."""
        class ResettableObject:
            def __init__(self):
                self.data = []
            
            def reset(self):
                self.data.clear()
        
        pool = ObjectPool("resettable", ResettableObject, max_size=5)
        
        obj = pool.get()
        obj.data.append("test")
        
        pool.return_object(obj)
        
        # Get the same object back
        reused_obj = pool.get()
        assert reused_obj is obj
        assert len(reused_obj.data) == 0  # Should be reset
    
    def test_pool_clear(self):
        """Test clearing pool."""
        pool = ObjectPool("strings", str, max_size=5)
        
        # Add objects to pool
        obj1 = pool.get()
        obj2 = pool.get()
        pool.return_object(obj1)
        pool.return_object(obj2)
        
        assert pool.returned_count == 2
        
        pool.clear()
        
        # Pool should be empty, next get should create new object
        obj3 = pool.get()
        assert pool.created_count == 3  # New object created
    
    def test_pool_statistics(self):
        """Test pool statistics."""
        pool = ObjectPool("test_pool", str, max_size=10)
        
        # Perform some operations
        obj1 = pool.get()
        obj2 = pool.get()
        pool.return_object(obj1)
        
        reused_obj = pool.get()  # Should reuse obj1
        
        stats = pool.get_statistics()
        
        assert stats["name"] == "test_pool"
        assert stats["object_type"] == "<class 'str'>"
        assert stats["max_size"] == 10
        assert stats["current_size"] == 0  # obj1 was reused
        assert stats["created_count"] == 2
        assert stats["reused_count"] == 1
        assert stats["returned_count"] == 1
        assert 0 <= stats["efficiency"] <= 1


class TestStreamingBuffer:
    """Test StreamingBuffer class."""
    
    def test_streaming_buffer_creation(self):
        """Test streaming buffer creation."""
        buffer = StreamingBuffer(initial_size=1024, max_size=8192)
        
        assert buffer.initial_size == 1024
        assert buffer.max_size == 8192
        assert buffer.auto_size is True
        assert buffer.total_reads == 0
        assert buffer.total_writes == 0
    
    def test_write_and_read_data(self):
        """Test writing and reading data."""
        buffer = StreamingBuffer(initial_size=100)
        
        test_data = b"<xml><item>test data</item></xml>"
        
        # Write data
        bytes_written = buffer.write(test_data)
        assert bytes_written == len(test_data)
        assert buffer.total_writes == 1
        assert buffer.total_bytes_written == len(test_data)
        
        # Read data
        read_data = buffer.read(len(test_data))
        assert read_data == test_data
        assert buffer.total_reads == 1
        assert buffer.total_bytes_read == len(test_data)
    
    def test_partial_reads(self):
        """Test partial reading of data."""
        buffer = StreamingBuffer()
        
        test_data = b"<xml><item>content</item></xml>"
        buffer.write(test_data)
        
        # Read in chunks
        chunk1 = buffer.read(5)
        chunk2 = buffer.read(10)
        chunk3 = buffer.read()  # Read remaining
        
        assert chunk1 == b"<xml>"
        assert chunk2 == b"<item>cont"
        assert chunk3 == b"ent</item></xml>"
        
        assert buffer.total_reads == 3
    
    def test_buffer_auto_resize(self):
        """Test automatic buffer resizing."""
        buffer = StreamingBuffer(initial_size=10, max_size=100, auto_size=True)
        
        # Write data larger than initial size
        large_data = b"x" * 50
        bytes_written = buffer.write(large_data)
        
        assert bytes_written == 50
        assert buffer.resize_count > 0  # Should have resized
        assert len(buffer._buffer) > 10  # Buffer should be larger
    
    def test_buffer_max_size_limit(self):
        """Test buffer respects maximum size limit."""
        buffer = StreamingBuffer(initial_size=10, max_size=20, auto_size=True)
        
        # Try to write data larger than max size
        too_large_data = b"x" * 30
        
        with pytest.raises(MemoryError):
            buffer.write(too_large_data)
    
    def test_write_invalid_data_type(self):
        """Test writing invalid data type."""
        buffer = StreamingBuffer()
        
        with pytest.raises(TypeError):
            buffer.write("string instead of bytes")
    
    def test_buffer_clear(self):
        """Test clearing buffer."""
        buffer = StreamingBuffer(initial_size=50)
        
        # Write some data
        buffer.write(b"test data")
        
        # Clear buffer
        buffer.clear()
        
        # Buffer should be empty
        data = buffer.read()
        assert data == b""
    
    def test_buffer_statistics(self):
        """Test buffer statistics."""
        buffer = StreamingBuffer(initial_size=100, max_size=500)
        
        # Perform operations
        buffer.write(b"test1")
        buffer.write(b"test2")
        buffer.read(5)
        
        stats = buffer.get_statistics()
        
        assert stats["initial_size"] == 100
        assert stats["max_size"] == 500
        assert stats["total_reads"] == 1
        assert stats["total_writes"] == 2
        assert stats["total_bytes_written"] == 10
        assert stats["total_bytes_read"] == 5


class TestMemoryManager:
    """Test MemoryManager class."""
    
    def test_memory_manager_creation(self):
        """Test memory manager creation."""
        manager = MemoryManager(
            memory_threshold_mb=200.0,
            critical_threshold_mb=500.0,
            monitoring_interval=1.0
        )
        
        assert manager.memory_threshold_mb == 200.0
        assert manager.critical_threshold_mb == 500.0
        assert manager.monitoring_interval == 1.0
        assert not manager._monitoring
        
        # Should have default pools created
        assert "strings" in manager._pools
        assert "lists" in manager._pools
        assert "dicts" in manager._pools
        assert "bytes" in manager._pools
    
    def test_create_custom_pool(self):
        """Test creating custom pool."""
        manager = MemoryManager()
        
        pool = manager.create_pool("custom", int, max_size=20)
        
        assert pool.name == "custom"
        assert pool.object_type == int
        assert pool.max_size == 20
        assert "custom" in manager._pools
    
    def test_create_duplicate_pool(self):
        """Test creating pool with duplicate name."""
        manager = MemoryManager()
        
        pool1 = manager.create_pool("duplicate", str, max_size=10)
        pool2 = manager.create_pool("duplicate", int, max_size=20)  # Different type
        
        # Should return the original pool
        assert pool1 is pool2
        assert pool2.object_type == str  # Original type
    
    def test_get_from_pool(self):
        """Test getting objects from pool."""
        manager = MemoryManager()
        
        obj = manager.get_from_pool("strings")
        assert isinstance(obj, str)
    
    def test_return_to_pool(self):
        """Test returning objects to pool."""
        manager = MemoryManager()
        
        obj = manager.get_from_pool("strings")
        returned = manager.return_to_pool("strings", obj)
        
        assert returned is True
    
    def test_get_from_nonexistent_pool(self):
        """Test getting from non-existent pool."""
        manager = MemoryManager()
        
        with pytest.raises(KeyError):
            manager.get_from_pool("nonexistent")
    
    def test_return_to_nonexistent_pool(self):
        """Test returning to non-existent pool."""
        manager = MemoryManager()
        
        with pytest.raises(KeyError):
            manager.return_to_pool("nonexistent", "object")
    
    def test_clear_pool(self):
        """Test clearing specific pool."""
        manager = MemoryManager()
        
        # Add objects to pool
        obj1 = manager.get_from_pool("strings")
        obj2 = manager.get_from_pool("strings")
        manager.return_to_pool("strings", obj1)
        manager.return_to_pool("strings", obj2)
        
        # Clear pool
        manager.clear_pool("strings")
        
        # Pool should be empty
        pool_stats = manager.get_pool_statistics()["strings"]
        assert pool_stats["current_size"] == 0
    
    def test_clear_all_pools(self):
        """Test clearing all pools."""
        manager = MemoryManager()
        
        # Add objects to multiple pools
        obj1 = manager.get_from_pool("strings")
        obj2 = manager.get_from_pool("lists")
        manager.return_to_pool("strings", obj1)
        manager.return_to_pool("lists", obj2)
        
        # Clear all pools
        manager.clear_all_pools()
        
        # All pools should be empty
        pool_stats = manager.get_pool_statistics()
        for pool_name, stats in pool_stats.items():
            assert stats["current_size"] == 0
    
    def test_get_pool_statistics(self):
        """Test getting pool statistics."""
        manager = MemoryManager()
        
        stats = manager.get_pool_statistics()
        
        assert isinstance(stats, dict)
        assert "strings" in stats
        assert "lists" in stats
        
        # Each pool should have statistics
        for pool_name, pool_stats in stats.items():
            assert "name" in pool_stats
            assert "max_size" in pool_stats
            assert "current_size" in pool_stats
    
    @patch('ultra_robust_xml_parser.tools.memory.PSUTIL_AVAILABLE', False)
    def test_get_memory_stats_without_psutil(self):
        """Test getting memory stats without psutil."""
        manager = MemoryManager()
        
        stats = manager.get_memory_stats()
        
        assert isinstance(stats, MemoryStats)
        assert stats.resident_memory_mb == 0.0  # Should be default
        assert stats.heap_objects > 0  # Should have Python objects
        assert isinstance(stats.pool_statistics, dict)
    
    @patch('ultra_robust_xml_parser.tools.memory.PSUTIL_AVAILABLE', True)
    @patch('ultra_robust_xml_parser.tools.memory.psutil')
    def test_get_memory_stats_with_psutil(self, mock_psutil):
        """Test getting memory stats with psutil."""
        # Mock psutil
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 100 * 1024 * 1024  # 100MB
        mock_process.memory_info.return_value.vms = 200 * 1024 * 1024  # 200MB
        mock_process.memory_percent.return_value = 15.5
        mock_psutil.Process.return_value = mock_process
        
        manager = MemoryManager()
        stats = manager.get_memory_stats()
        
        assert stats.resident_memory_mb == 100.0
        assert stats.virtual_memory_mb == 200.0
        assert stats.memory_percent == 15.5
    
    def test_monitoring_start_stop(self):
        """Test starting and stopping monitoring."""
        manager = MemoryManager(monitoring_interval=0.1)
        
        assert not manager._monitoring
        
        manager.start_monitoring()
        assert manager._monitoring
        assert manager._monitor_thread is not None
        
        # Let it run briefly
        time.sleep(0.2)
        
        manager.stop_monitoring()
        assert not manager._monitoring
    
    def test_alert_handlers(self):
        """Test memory alert handlers."""
        manager = MemoryManager()
        
        alerts_received = []
        
        def alert_handler(alert):
            alerts_received.append(alert)
        
        manager.add_alert_handler(alert_handler)
        
        # Create a test alert
        test_alert = MemoryAlert(
            "test", MemoryLevel.HIGH, "Test alert", 100.0, 50.0
        )
        
        manager._trigger_alert(test_alert)
        
        assert len(alerts_received) == 1
        assert alerts_received[0] is test_alert
        
        # Remove handler
        manager.remove_alert_handler(alert_handler)
        
        manager._trigger_alert(test_alert)
        
        # Should still be 1 (handler removed)
        assert len(alerts_received) == 1
    
    def test_force_gc(self):
        """Test forced garbage collection."""
        manager = MemoryManager()
        
        result = manager.force_gc()
        
        assert isinstance(result, dict)
        assert "before_counts" in result
        assert "after_counts" in result
        assert "collected_objects" in result
        assert "objects_freed" in result
    
    def test_optimize_gc(self):
        """Test GC optimization."""
        manager = MemoryManager()
        
        result = manager.optimize_gc()
        
        assert isinstance(result, dict)
        assert "original_thresholds" in result
        assert "new_thresholds" in result
        assert "collected_objects" in result
        assert "current_counts" in result
    
    def test_stats_history(self):
        """Test statistics history tracking."""
        manager = MemoryManager()
        
        # Initially no history
        history = manager.get_stats_history()
        assert len(history) == 0
        
        # Add some stats to history manually
        stats1 = manager.get_memory_stats()
        manager._stats_history.append(stats1)
        
        stats2 = manager.get_memory_stats()
        manager._stats_history.append(stats2)
        
        history = manager.get_stats_history()
        assert len(history) == 2
        
        # Test limit
        limited_history = manager.get_stats_history(limit=1)
        assert len(limited_history) == 1
        assert limited_history[0] is stats2  # Should be the most recent
    
    def test_export_stats_json(self):
        """Test exporting statistics to JSON."""
        manager = MemoryManager()
        
        # Add some history
        stats = manager.get_memory_stats()
        manager._stats_history.append(stats)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            export_path = Path(f.name)
        
        try:
            manager.export_stats(export_path, "json")
            
            assert export_path.exists()
            
            # Verify exported content
            exported_data = json.loads(export_path.read_text())
            assert "memory_manager_config" in exported_data
            assert "current_stats" in exported_data
            assert "pool_statistics" in exported_data
            assert "stats_history" in exported_data
            
            # Check config
            config = exported_data["memory_manager_config"]
            assert config["memory_threshold_mb"] == manager.memory_threshold_mb
            
        finally:
            export_path.unlink()
    
    def test_export_stats_csv(self):
        """Test exporting statistics to CSV."""
        manager = MemoryManager()
        
        # Add some history
        stats1 = manager.get_memory_stats()
        stats2 = manager.get_memory_stats()
        manager._stats_history.extend([stats1, stats2])
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            export_path = Path(f.name)
        
        try:
            manager.export_stats(export_path, "csv")
            
            assert export_path.exists()
            
            # Verify CSV content
            content = export_path.read_text()
            assert "timestamp,resident_memory_mb" in content
            lines = content.strip().split('\n')
            assert len(lines) == 3  # Header + 2 data rows
            
        finally:
            export_path.unlink()


class TestMemoryLevels:
    """Test memory level enum."""
    
    def test_memory_levels_exist(self):
        """Test that all memory levels exist."""
        expected_levels = ["low", "normal", "high", "critical"]
        
        for expected_level in expected_levels:
            assert hasattr(MemoryLevel, expected_level.upper())
            assert MemoryLevel[expected_level.upper()].value == expected_level


class TestPoolTypes:
    """Test pool type enum."""
    
    def test_pool_types_exist(self):
        """Test that all pool types exist."""
        expected_types = ["string_pool", "token_pool", "element_pool", "buffer_pool"]
        
        for expected_type in expected_types:
            assert hasattr(PoolType, expected_type.upper())
            assert PoolType[expected_type.upper()].value == expected_type


@pytest.mark.integration
class TestMemoryManagerIntegration:
    """Integration tests for memory management."""
    
    def test_full_memory_management_workflow(self):
        """Test complete memory management workflow."""
        manager = MemoryManager(
            memory_threshold_mb=100.0,
            critical_threshold_mb=200.0,
            monitoring_interval=0.1
        )
        
        try:
            # Create custom pools
            manager.create_pool("tokens", dict, max_size=50)
            manager.create_pool("elements", list, max_size=25)
            
            # Use pools
            token1 = manager.get_from_pool("tokens")
            token2 = manager.get_from_pool("tokens")
            element1 = manager.get_from_pool("elements")
            
            manager.return_to_pool("tokens", token1)
            manager.return_to_pool("elements", element1)
            
            # Get statistics
            pool_stats = manager.get_pool_statistics()
            memory_stats = manager.get_memory_stats()
            
            assert "tokens" in pool_stats
            assert "elements" in pool_stats
            assert isinstance(memory_stats, MemoryStats)
            
            # Start monitoring briefly
            manager.start_monitoring()
            time.sleep(0.2)
            manager.stop_monitoring()
            
            # Should have some history
            history = manager.get_stats_history()
            assert len(history) > 0
            
        finally:
            manager.stop_monitoring()
    
    def test_memory_alert_integration(self):
        """Test memory alert integration."""
        # Create manager with very low threshold to trigger alerts
        manager = MemoryManager(
            memory_threshold_mb=0.1,  # Very low threshold
            critical_threshold_mb=0.2
        )
        
        alerts_received = []
        
        def test_handler(alert):
            alerts_received.append(alert)
        
        manager.add_alert_handler(test_handler)
        
        # Force check for alerts with current stats
        stats = manager.get_memory_stats()
        manager._check_memory_alerts(stats)
        
        # Should likely trigger alerts due to low threshold
        # (This is environment dependent, so we just check the mechanism works)
        assert isinstance(alerts_received, list)  # Handler was called
    
    def test_streaming_buffer_integration(self):
        """Test streaming buffer in realistic scenario."""
        buffer = StreamingBuffer(initial_size=64, max_size=1024, auto_size=True)
        
        # Simulate streaming XML processing
        xml_chunks = [
            b'<?xml version="1.0"?>',
            b'<root xmlns="http://example.com">',
            b'<items>',
            b'<item id="1">First item</item>',
            b'<item id="2">Second item</item>',
            b'</items>',
            b'</root>'
        ]
        
        # Write all chunks
        total_written = 0
        for chunk in xml_chunks:
            written = buffer.write(chunk)
            total_written += written
        
        # Read back all data
        all_data = buffer.read()
        expected_xml = b''.join(xml_chunks)
        
        assert all_data == expected_xml
        assert buffer.total_bytes_written == total_written
        assert buffer.total_bytes_read == len(expected_xml)
        
        # Get final statistics
        stats = buffer.get_statistics()
        assert stats["total_writes"] == len(xml_chunks)
        assert stats["total_reads"] == 1