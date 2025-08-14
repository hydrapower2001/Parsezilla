"""Performance profiling tools for Ultra Robust XML Parser.

Provides comprehensive performance analysis including timing analysis, memory tracking,
visualization capabilities, and optimization recommendations.
"""

import json
import time
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    psutil = None
    HAS_PSUTIL = False
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ultra_robust_xml_parser import UltraRobustXMLParser
from ultra_robust_xml_parser.shared.logging import get_logger


@dataclass
class LayerPerformance:
    """Performance metrics for a specific processing layer."""
    
    layer_name: str
    start_time: float
    end_time: float
    memory_start: int  # bytes
    memory_end: int  # bytes
    cpu_percent: float
    operations_count: int = 0
    
    @property
    def duration_ms(self) -> float:
        """Processing duration in milliseconds."""
        return (self.end_time - self.start_time) * 1000
    
    @property
    def memory_delta(self) -> int:
        """Memory usage change in bytes."""
        return self.memory_end - self.memory_start
    
    @property
    def ops_per_second(self) -> float:
        """Operations per second rate."""
        duration_s = self.end_time - self.start_time
        return self.operations_count / duration_s if duration_s > 0 else 0.0


@dataclass
class ProfilingSession:
    """Container for a complete profiling session."""
    
    session_id: str
    start_time: float
    end_time: float
    input_size: int  # bytes
    layers: List[LayerPerformance] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_duration_ms(self) -> float:
        """Total session duration in milliseconds."""
        return (self.end_time - self.start_time) * 1000
    
    @property
    def throughput_mb_per_s(self) -> float:
        """Processing throughput in MB/s."""
        duration_s = self.end_time - self.start_time
        if duration_s <= 0:
            return 0.0
        return (self.input_size / (1024 * 1024)) / duration_s


@dataclass
class PerformanceReport:
    """Comprehensive performance analysis report."""
    
    sessions: List[ProfilingSession]
    generation_time: float
    
    @property
    def session_count(self) -> int:
        """Total number of profiled sessions."""
        return len(self.sessions)
    
    @property
    def average_duration_ms(self) -> float:
        """Average processing duration across sessions."""
        if not self.sessions:
            return 0.0
        return sum(s.total_duration_ms for s in self.sessions) / len(self.sessions)
    
    @property
    def average_throughput_mb_per_s(self) -> float:
        """Average throughput across sessions."""
        if not self.sessions:
            return 0.0
        return sum(s.throughput_mb_per_s for s in self.sessions) / len(self.sessions)


class PerformanceProfiler:
    """Comprehensive performance profiler for XML parsing operations.
    
    Provides detailed timing analysis, memory tracking, and performance visualization
    capabilities with support for multi-layer profiling and optimization recommendations.
    
    Examples:
        Basic profiling:
        >>> profiler = PerformanceProfiler()
        >>> with profiler.profile_parsing("session1") as session:
        ...     result = parser.parse(xml_content)
        >>> report = profiler.generate_report()
        
        Layer-specific profiling:
        >>> profiler = PerformanceProfiler()
        >>> session = profiler.start_session("detailed")
        >>> with profiler.profile_layer(session, "character_processing"):
        ...     char_result = char_processor.process(content)
        >>> profiler.end_session(session)
    """
    
    def __init__(self, enable_memory_tracking: bool = True):
        """Initialize performance profiler.
        
        Args:
            enable_memory_tracking: Whether to track memory usage (may impact performance)
        """
        self.enable_memory_tracking = enable_memory_tracking and HAS_PSUTIL
        self.sessions: List[ProfilingSession] = []
        self.current_session: Optional[ProfilingSession] = None
        self.logger = get_logger(__name__, None, "performance_profiler")
    
    def start_session(self, session_id: str, input_size: int = 0) -> ProfilingSession:
        """Start a new profiling session.
        
        Args:
            session_id: Unique identifier for the session
            input_size: Size of input data in bytes
            
        Returns:
            ProfilingSession object for tracking
        """
        session = ProfilingSession(
            session_id=session_id,
            start_time=time.time(),
            end_time=0.0,
            input_size=input_size
        )
        
        self.current_session = session
        self.logger.info(
            "Started profiling session",
            extra={
                "session_id": session_id,
                "input_size": input_size,
                "memory_tracking": self.enable_memory_tracking
            }
        )
        
        return session
    
    def end_session(self, session: ProfilingSession) -> None:
        """End a profiling session and store results.
        
        Args:
            session: Session to end
        """
        session.end_time = time.time()
        self.sessions.append(session)
        
        if self.current_session == session:
            self.current_session = None
        
        self.logger.info(
            "Ended profiling session",
            extra={
                "session_id": session.session_id,
                "duration_ms": session.total_duration_ms,
                "throughput_mb_s": session.throughput_mb_per_s,
                "layer_count": len(session.layers)
            }
        )
    
    def profile_layer(self, session: ProfilingSession, layer_name: str) -> "LayerProfiler":
        """Profile a specific processing layer.
        
        Args:
            session: Profiling session to add layer to
            layer_name: Name of the layer being profiled
            
        Returns:
            Context manager for layer profiling
        """
        return LayerProfiler(self, session, layer_name)
    
    def profile_parsing(self, session_id: str) -> "ParsingProfiler":
        """Profile a complete parsing operation.
        
        Args:
            session_id: Unique identifier for the session
            
        Returns:
            Context manager for parsing profiling
        """
        return ParsingProfiler(self, session_id)
    
    def add_layer_performance(
        self,
        session: ProfilingSession,
        layer_perf: LayerPerformance
    ) -> None:
        """Add layer performance data to session.
        
        Args:
            session: Session to add performance data to
            layer_perf: Layer performance metrics
        """
        session.layers.append(layer_perf)
        
        self.logger.debug(
            "Added layer performance data",
            extra={
                "session_id": session.session_id,
                "layer_name": layer_perf.layer_name,
                "duration_ms": layer_perf.duration_ms,
                "memory_delta": layer_perf.memory_delta
            }
        )
    
    def generate_report(self) -> PerformanceReport:
        """Generate comprehensive performance report.
        
        Returns:
            PerformanceReport with analysis and recommendations
        """
        return PerformanceReport(
            sessions=self.sessions.copy(),
            generation_time=time.time()
        )
    
    def save_report(self, report: PerformanceReport, output_path: Path) -> None:
        """Save performance report to file.
        
        Args:
            report: Report to save
            output_path: Path to save report to
        """
        report_data = {
            "generation_time": report.generation_time,
            "summary": {
                "session_count": report.session_count,
                "average_duration_ms": report.average_duration_ms,
                "average_throughput_mb_s": report.average_throughput_mb_per_s
            },
            "sessions": [
                {
                    "session_id": session.session_id,
                    "start_time": session.start_time,
                    "end_time": session.end_time,
                    "input_size": session.input_size,
                    "total_duration_ms": session.total_duration_ms,
                    "throughput_mb_s": session.throughput_mb_per_s,
                    "metadata": session.metadata,
                    "layers": [
                        {
                            "layer_name": layer.layer_name,
                            "duration_ms": layer.duration_ms,
                            "memory_delta": layer.memory_delta,
                            "cpu_percent": layer.cpu_percent,
                            "operations_count": layer.operations_count,
                            "ops_per_second": layer.ops_per_second
                        }
                        for layer in session.layers
                    ]
                }
                for session in report.sessions
            ]
        }
        
        output_path.write_text(json.dumps(report_data, indent=2))
        
        self.logger.info(
            "Saved performance report",
            extra={
                "output_path": str(output_path),
                "session_count": report.session_count
            }
        )
    
    def get_optimization_recommendations(self, report: PerformanceReport) -> List[str]:
        """Generate optimization recommendations based on performance data.
        
        Args:
            report: Performance report to analyze
            
        Returns:
            List of optimization recommendations
        """
        recommendations = []
        
        if not report.sessions:
            return ["No profiling data available for analysis"]
        
        # Analyze average performance
        avg_duration = report.average_duration_ms
        avg_throughput = report.average_throughput_mb_per_s
        
        # Duration-based recommendations
        if avg_duration > 1000:  # > 1 second
            recommendations.append(
                "Consider using streaming mode for large documents to reduce processing time"
            )
        
        if avg_throughput < 1.0:  # < 1 MB/s
            recommendations.append(
                "Low throughput detected. Consider using performance-optimized configuration preset"
            )
        
        # Analyze layer performance patterns
        layer_stats = {}
        for session in report.sessions:
            for layer in session.layers:
                if layer.layer_name not in layer_stats:
                    layer_stats[layer.layer_name] = []
                layer_stats[layer.layer_name].append(layer.duration_ms)
        
        # Identify bottleneck layers
        for layer_name, durations in layer_stats.items():
            avg_layer_duration = sum(durations) / len(durations)
            if avg_layer_duration > avg_duration * 0.4:  # Layer takes > 40% of total time
                recommendations.append(
                    f"Layer '{layer_name}' appears to be a bottleneck. "
                    f"Consider optimizing this processing stage"
                )
        
        # Memory-based recommendations
        for session in report.sessions:
            total_memory_usage = sum(layer.memory_delta for layer in session.layers if layer.memory_delta > 0)
            if total_memory_usage > session.input_size * 5:  # Memory usage > 5x input size
                recommendations.append(
                    "High memory usage detected. Consider using streaming processing "
                    "or memory-optimized configuration"
                )
        
        if not recommendations:
            recommendations.append("Performance appears optimal based on current analysis")
        
        return recommendations
    
    def clear_sessions(self) -> None:
        """Clear all stored profiling sessions."""
        session_count = len(self.sessions)
        self.sessions.clear()
        self.current_session = None
        
        self.logger.info(
            "Cleared profiling sessions",
            extra={"cleared_count": session_count}
        )


class LayerProfiler:
    """Context manager for profiling individual processing layers."""
    
    def __init__(self, profiler: PerformanceProfiler, session: ProfilingSession, layer_name: str):
        self.profiler = profiler
        self.session = session
        self.layer_name = layer_name
        self.layer_perf: Optional[LayerPerformance] = None
    
    def __enter__(self) -> LayerPerformance:
        """Start layer profiling."""
        process = psutil.Process() if (self.profiler.enable_memory_tracking and HAS_PSUTIL) else None
        
        self.layer_perf = LayerPerformance(
            layer_name=self.layer_name,
            start_time=time.time(),
            end_time=0.0,
            memory_start=process.memory_info().rss if process else 0,
            memory_end=0,
            cpu_percent=process.cpu_percent() if process else 0.0
        )
        
        return self.layer_perf
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End layer profiling."""
        if self.layer_perf is None:
            return
        
        process = psutil.Process() if (self.profiler.enable_memory_tracking and HAS_PSUTIL) else None
        
        self.layer_perf.end_time = time.time()
        self.layer_perf.memory_end = process.memory_info().rss if process else 0
        
        self.profiler.add_layer_performance(self.session, self.layer_perf)


class ParsingProfiler:
    """Context manager for profiling complete parsing operations."""
    
    def __init__(self, profiler: PerformanceProfiler, session_id: str):
        self.profiler = profiler
        self.session_id = session_id
        self.session: Optional[ProfilingSession] = None
    
    def __enter__(self) -> ProfilingSession:
        """Start parsing profiling."""
        self.session = self.profiler.start_session(self.session_id)
        return self.session
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End parsing profiling."""
        if self.session:
            self.profiler.end_session(self.session)


def benchmark_parser_configurations(
    xml_content: str,
    iterations: int = 10
) -> Dict[str, PerformanceReport]:
    """Benchmark different parser configurations.
    
    Args:
        xml_content: XML content to parse
        iterations: Number of iterations per configuration
        
    Returns:
        Dictionary mapping configuration names to performance reports
    """
    from ultra_robust_xml_parser.shared.config import TokenizationConfig
    
    configurations = {
        "balanced": TokenizationConfig.balanced(),
        "aggressive": TokenizationConfig.aggressive(),
        "conservative": TokenizationConfig.conservative()
    }
    
    results = {}
    
    for config_name, config in configurations.items():
        profiler = PerformanceProfiler()
        parser = UltraRobustXMLParser(config=config)
        
        for i in range(iterations):
            session_id = f"{config_name}_iteration_{i}"
            
            with profiler.profile_parsing(session_id) as session:
                session.input_size = len(xml_content.encode())
                session.metadata = {"configuration": config_name, "iteration": i}
                
                # Profile parsing with layer breakdown
                with profiler.profile_layer(session, "character_processing"):
                    pass  # Character processing would happen here
                
                with profiler.profile_layer(session, "tokenization"):
                    pass  # Tokenization would happen here
                
                with profiler.profile_layer(session, "tree_building"):
                    result = parser.parse(xml_content)
                    session.metadata["success"] = result.success
                    session.metadata["confidence"] = result.confidence
        
        results[config_name] = profiler.generate_report()
    
    return results