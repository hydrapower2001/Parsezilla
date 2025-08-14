"""Tests for the performance profiling module."""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ultra_robust_xml_parser.tools.profiling import (
    HAS_PSUTIL,
    LayerPerformance,
    PerformanceProfiler,
    PerformanceReport,
    ProfilingSession,
    benchmark_parser_configurations,
)


class TestLayerPerformance:
    """Test LayerPerformance data class."""
    
    def test_layer_performance_creation(self):
        """Test layer performance object creation."""
        layer_perf = LayerPerformance(
            layer_name="test_layer",
            start_time=1000.0,
            end_time=1001.0,
            memory_start=1024,
            memory_end=2048,
            cpu_percent=25.0,
            operations_count=100
        )
        
        assert layer_perf.layer_name == "test_layer"
        assert layer_perf.duration_ms == 1000.0  # 1 second = 1000ms
        assert layer_perf.memory_delta == 1024  # 2048 - 1024
        assert layer_perf.ops_per_second == 100.0  # 100 ops / 1 second
    
    def test_layer_performance_zero_duration(self):
        """Test layer performance with zero duration."""
        layer_perf = LayerPerformance(
            layer_name="instant",
            start_time=1000.0,
            end_time=1000.0,
            memory_start=1024,
            memory_end=1024,
            cpu_percent=0.0,
            operations_count=50
        )
        
        assert layer_perf.duration_ms == 0.0
        assert layer_perf.memory_delta == 0
        assert layer_perf.ops_per_second == 0.0  # Avoid division by zero


class TestProfilingSession:
    """Test ProfilingSession data class."""
    
    def test_profiling_session_creation(self):
        """Test profiling session creation."""
        session = ProfilingSession(
            session_id="test_session",
            start_time=1000.0,
            end_time=1002.0,
            input_size=1024
        )
        
        assert session.session_id == "test_session"
        assert session.total_duration_ms == 2000.0  # 2 seconds = 2000ms
        # 1024 bytes / 2 seconds = 512 bytes/s = 512/(1024*1024) MB/s ≈ 0.00048828125
        assert abs(session.throughput_mb_per_s - 0.00048828125) < 0.0001
    
    def test_profiling_session_zero_duration(self):
        """Test profiling session with zero duration."""
        session = ProfilingSession(
            session_id="instant",
            start_time=1000.0,
            end_time=1000.0,
            input_size=1024
        )
        
        assert session.total_duration_ms == 0.0
        assert session.throughput_mb_per_s == 0.0


class TestPerformanceReport:
    """Test PerformanceReport data class."""
    
    def test_performance_report_empty(self):
        """Test empty performance report."""
        report = PerformanceReport(sessions=[], generation_time=time.time())
        
        assert report.session_count == 0
        assert report.average_duration_ms == 0.0
        assert report.average_throughput_mb_per_s == 0.0
    
    def test_performance_report_with_sessions(self):
        """Test performance report with sessions."""
        sessions = [
            ProfilingSession("s1", 1000.0, 1001.0, 1024),
            ProfilingSession("s2", 2000.0, 2002.0, 2048)
        ]
        
        report = PerformanceReport(sessions=sessions, generation_time=time.time())
        
        assert report.session_count == 2
        assert report.average_duration_ms == 1500.0  # (1000 + 2000) / 2
        # Average throughput calculation
        s1_throughput = (1024 / (1024 * 1024)) / 1.0  # ~0.001 MB/s
        s2_throughput = (2048 / (1024 * 1024)) / 2.0  # ~0.001 MB/s
        expected_avg = (s1_throughput + s2_throughput) / 2
        assert abs(report.average_throughput_mb_per_s - expected_avg) < 0.0001


class TestPerformanceProfiler:
    """Test PerformanceProfiler class."""
    
    def test_profiler_initialization(self):
        """Test profiler initialization."""
        profiler = PerformanceProfiler(enable_memory_tracking=True)
        
        assert profiler.enable_memory_tracking == HAS_PSUTIL  # Should be False if psutil not available
        assert len(profiler.sessions) == 0
        assert profiler.current_session is None
    
    def test_profiler_initialization_memory_disabled(self):
        """Test profiler initialization with memory tracking disabled."""
        profiler = PerformanceProfiler(enable_memory_tracking=False)
        
        assert profiler.enable_memory_tracking is False
        assert len(profiler.sessions) == 0
    
    def test_start_and_end_session(self):
        """Test starting and ending profiling sessions."""
        profiler = PerformanceProfiler()
        
        # Start session
        session = profiler.start_session("test_session", input_size=1024)
        
        assert session.session_id == "test_session"
        assert session.input_size == 1024
        assert session.start_time > 0
        assert session.end_time == 0.0
        assert profiler.current_session == session
        
        # Small delay to ensure measurable duration
        time.sleep(0.001)
        
        # End session
        profiler.end_session(session)
        
        assert session.end_time > session.start_time
        assert profiler.current_session is None
        assert len(profiler.sessions) == 1
        assert profiler.sessions[0] == session
    
    def test_add_layer_performance(self):
        """Test adding layer performance data."""
        profiler = PerformanceProfiler()
        session = profiler.start_session("test", input_size=100)
        
        layer_perf = LayerPerformance(
            layer_name="test_layer",
            start_time=1000.0,
            end_time=1001.0,
            memory_start=1024,
            memory_end=2048,
            cpu_percent=50.0
        )
        
        profiler.add_layer_performance(session, layer_perf)
        
        assert len(session.layers) == 1
        assert session.layers[0] == layer_perf
    
    def test_generate_report(self):
        """Test report generation."""
        profiler = PerformanceProfiler()
        
        # Add some test sessions
        session1 = profiler.start_session("session1", 1024)
        time.sleep(0.001)
        profiler.end_session(session1)
        
        session2 = profiler.start_session("session2", 2048)
        time.sleep(0.001)
        profiler.end_session(session2)
        
        report = profiler.generate_report()
        
        assert isinstance(report, PerformanceReport)
        assert report.session_count == 2
        assert len(report.sessions) == 2
        assert report.generation_time > 0
    
    def test_save_report(self):
        """Test saving report to file."""
        profiler = PerformanceProfiler()
        session = profiler.start_session("test", 100)
        profiler.end_session(session)
        
        report = profiler.generate_report()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = Path(f.name)
        
        try:
            profiler.save_report(report, output_path)
            
            assert output_path.exists()
            
            # Verify the saved content
            saved_data = json.loads(output_path.read_text())
            assert "generation_time" in saved_data
            assert "summary" in saved_data
            assert "sessions" in saved_data
            assert len(saved_data["sessions"]) == 1
            
        finally:
            output_path.unlink()
    
    def test_optimization_recommendations_empty(self):
        """Test optimization recommendations with empty report."""
        profiler = PerformanceProfiler()
        report = profiler.generate_report()
        
        recommendations = profiler.get_optimization_recommendations(report)
        
        assert len(recommendations) == 1
        assert "No profiling data available" in recommendations[0]
    
    def test_optimization_recommendations_slow_performance(self):
        """Test optimization recommendations for slow performance."""
        profiler = PerformanceProfiler()
        
        # Create a session with slow performance
        session = ProfilingSession(
            session_id="slow",
            start_time=1000.0,
            end_time=2500.0,  # 1.5 seconds (> 1 second threshold)
            input_size=1024
        )
        profiler.sessions.append(session)
        
        report = profiler.generate_report()
        recommendations = profiler.get_optimization_recommendations(report)
        
        # Should recommend streaming for long processing times
        assert any("streaming mode" in rec for rec in recommendations)
    
    def test_optimization_recommendations_low_throughput(self):
        """Test optimization recommendations for low throughput."""
        profiler = PerformanceProfiler()
        
        # Create a session with low throughput
        session = ProfilingSession(
            session_id="low_throughput",
            start_time=1000.0,
            end_time=3000.0,  # 2 seconds
            input_size=1024 * 1024  # 1 MB, but takes 2 seconds = 0.5 MB/s (< 1 MB/s threshold)
        )
        profiler.sessions.append(session)
        
        report = profiler.generate_report()
        recommendations = profiler.get_optimization_recommendations(report)
        
        # Should recommend performance-optimized configuration
        assert any("performance-optimized" in rec for rec in recommendations)
    
    def test_optimization_recommendations_bottleneck_layer(self):
        """Test optimization recommendations for bottleneck layers."""
        profiler = PerformanceProfiler()
        
        session = ProfilingSession(
            session_id="bottleneck_test",
            start_time=1000.0,
            end_time=1100.0,  # 100ms total
            input_size=1024
        )
        
        # Add a layer that takes > 40% of total time
        slow_layer = LayerPerformance(
            layer_name="slow_layer",
            start_time=1000.0,
            end_time=1050.0,  # 50ms out of 100ms total (50% > 40% threshold)
            memory_start=1024,
            memory_end=1024,
            cpu_percent=90.0
        )
        session.layers.append(slow_layer)
        
        profiler.sessions.append(session)
        
        report = profiler.generate_report()
        recommendations = profiler.get_optimization_recommendations(report)
        
        # Should identify the slow layer as a bottleneck
        assert any("slow_layer" in rec and "bottleneck" in rec for rec in recommendations)
    
    def test_clear_sessions(self):
        """Test clearing profiling sessions."""
        profiler = PerformanceProfiler()
        
        # Add some sessions
        session1 = profiler.start_session("s1", 100)
        profiler.end_session(session1)
        session2 = profiler.start_session("s2", 200)
        profiler.end_session(session2)
        
        assert len(profiler.sessions) == 2
        
        profiler.clear_sessions()
        
        assert len(profiler.sessions) == 0
        assert profiler.current_session is None


class TestLayerProfiler:
    """Test LayerProfiler context manager."""
    
    def test_layer_profiler_context_manager(self):
        """Test layer profiler as context manager."""
        profiler = PerformanceProfiler()
        session = profiler.start_session("test", 100)
        
        with profiler.profile_layer(session, "test_layer") as layer_perf:
            assert isinstance(layer_perf, LayerPerformance)
            assert layer_perf.layer_name == "test_layer"
            assert layer_perf.start_time > 0
            
            # Simulate some work
            time.sleep(0.001)
        
        # After context exit, layer should be added to session
        assert len(session.layers) == 1
        layer = session.layers[0]
        assert layer.layer_name == "test_layer"
        assert layer.end_time > layer.start_time


class TestParsingProfiler:
    """Test ParsingProfiler context manager."""
    
    def test_parsing_profiler_context_manager(self):
        """Test parsing profiler as context manager."""
        profiler = PerformanceProfiler()
        
        with profiler.profile_parsing("test_parse") as session:
            assert isinstance(session, ProfilingSession)
            assert session.session_id == "test_parse"
            
            # Simulate parsing work
            time.sleep(0.001)
        
        # After context exit, session should be completed and stored
        assert len(profiler.sessions) == 1
        stored_session = profiler.sessions[0]
        assert stored_session.session_id == "test_parse"
        assert stored_session.end_time > stored_session.start_time


class TestBenchmarkFunction:
    """Test benchmark_parser_configurations function."""
    
    @patch('ultra_robust_xml_parser.tools.profiling.UltraRobustXMLParser')
    def test_benchmark_parser_configurations(self, mock_parser_class):
        """Test benchmarking different parser configurations."""
        # Mock parser instance
        mock_parser = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.confidence = 0.95
        mock_parser.parse.return_value = mock_result
        mock_parser_class.return_value = mock_parser
        
        xml_content = "<root><item>test</item></root>"
        
        results = benchmark_parser_configurations(xml_content, iterations=2)
        
        # Should return results for all three configurations
        assert len(results) == 3
        assert "balanced" in results
        assert "aggressive" in results
        assert "conservative" in results
        
        # Each configuration should have 2 sessions (iterations=2)
        for config_name, report in results.items():
            assert report.session_count == 2
            for session in report.sessions:
                assert session.metadata["configuration"] == config_name
                assert "iteration" in session.metadata


@pytest.mark.skipif(not HAS_PSUTIL, reason="psutil not available")
class TestPerformanceProfilerWithPsutil:
    """Test PerformanceProfiler with psutil available."""
    
    def test_memory_tracking_enabled(self):
        """Test memory tracking when psutil is available."""
        profiler = PerformanceProfiler(enable_memory_tracking=True)
        session = profiler.start_session("memory_test", 100)
        
        with profiler.profile_layer(session, "memory_layer") as layer_perf:
            # Memory tracking should be enabled
            assert layer_perf.memory_start > 0
            time.sleep(0.001)
        
        # Memory end should be recorded
        layer = session.layers[0]
        assert layer.memory_end > 0


@pytest.mark.skipif(HAS_PSUTIL, reason="psutil is available")
class TestPerformanceProfilerWithoutPsutil:
    """Test PerformanceProfiler when psutil is not available."""
    
    def test_memory_tracking_disabled(self):
        """Test memory tracking when psutil is not available."""
        profiler = PerformanceProfiler(enable_memory_tracking=True)
        
        # Should automatically disable memory tracking
        assert profiler.enable_memory_tracking is False