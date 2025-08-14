"""Performance benchmarking for XML tokenization.

This module provides comprehensive benchmarking capabilities to compare
tokenization performance against existing XML parsers and track performance
regression over time.
"""

import gc
import psutil
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ultra_robust_xml_parser.character import CharacterStreamResult
from ultra_robust_xml_parser.character.encoding import EncodingResult, DetectionMethod
from ultra_robust_xml_parser.shared import get_logger

from .api import EnhancedXMLTokenizer, TokenizationConfig


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    
    parser_name: str
    test_case: str
    processing_time_ms: float
    memory_used_mb: float
    characters_processed: int
    tokens_generated: int
    success: bool
    error_message: Optional[str] = None
    
    @property
    def characters_per_second(self) -> float:
        """Calculate characters processed per second."""
        if self.processing_time_ms <= 0:
            return 0.0
        return (self.characters_processed * 1000.0) / self.processing_time_ms
    
    @property
    def tokens_per_second(self) -> float:
        """Calculate tokens generated per second."""
        if self.processing_time_ms <= 0:
            return 0.0
        return (self.tokens_generated * 1000.0) / self.processing_time_ms
    
    @property
    def memory_per_character(self) -> float:
        """Calculate memory usage per character."""
        if self.characters_processed <= 0:
            return 0.0
        return (self.memory_used_mb * 1024 * 1024) / self.characters_processed


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results with statistical analysis."""
    
    results: List[BenchmarkResult] = field(default_factory=list)
    suite_name: str = "Tokenization Benchmark"
    timestamp: float = field(default_factory=time.time)
    
    def add_result(self, result: BenchmarkResult) -> None:
        """Add a benchmark result to the suite."""
        self.results.append(result)
    
    def get_results_by_parser(self, parser_name: str) -> List[BenchmarkResult]:
        """Get all results for a specific parser."""
        return [r for r in self.results if r.parser_name == parser_name]
    
    def get_results_by_test_case(self, test_case: str) -> List[BenchmarkResult]:
        """Get all results for a specific test case."""
        return [r for r in self.results if r.test_case == test_case]
    
    def get_statistics(self, parser_name: str, metric: str) -> Dict[str, float]:
        """Get statistical analysis for a parser and metric."""
        parser_results = self.get_results_by_parser(parser_name)
        if not parser_results:
            return {}
        
        values = []
        for result in parser_results:
            if metric == "processing_time_ms":
                values.append(result.processing_time_ms)
            elif metric == "memory_used_mb":
                values.append(result.memory_used_mb)
            elif metric == "characters_per_second":
                values.append(result.characters_per_second)
            elif metric == "tokens_per_second":
                values.append(result.tokens_per_second)
            elif metric == "memory_per_character":
                values.append(result.memory_per_character)
        
        if not values:
            return {}
        
        return {
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "count": len(values)
        }
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        parsers = list(set(r.parser_name for r in self.results))
        test_cases = list(set(r.test_case for r in self.results))
        
        report = {
            "suite_name": self.suite_name,
            "timestamp": self.timestamp,
            "total_results": len(self.results),
            "parsers": parsers,
            "test_cases": test_cases,
            "summary": {},
            "detailed_results": {}
        }
        
        # Summary statistics
        for parser in parsers:
            parser_results = self.get_results_by_parser(parser)
            successful_results = [r for r in parser_results if r.success]
            
            report["summary"][parser] = {
                "total_runs": len(parser_results),
                "successful_runs": len(successful_results),
                "success_rate": len(successful_results) / len(parser_results) if parser_results else 0.0,
                "performance": self.get_statistics(parser, "characters_per_second"),
                "memory": self.get_statistics(parser, "memory_used_mb")
            }
        
        # Detailed results by test case
        for test_case in test_cases:
            case_results = self.get_results_by_test_case(test_case)
            report["detailed_results"][test_case] = {}
            
            for parser in parsers:
                parser_case_results = [r for r in case_results if r.parser_name == parser]
                if parser_case_results:
                    result = parser_case_results[0]  # Take first result for this case/parser
                    report["detailed_results"][test_case][parser] = {
                        "processing_time_ms": result.processing_time_ms,
                        "memory_used_mb": result.memory_used_mb,
                        "characters_per_second": result.characters_per_second,
                        "tokens_per_second": result.tokens_per_second,
                        "success": result.success,
                        "error": result.error_message
                    }
        
        return report


class TokenizationBenchmark:
    """Comprehensive tokenization performance benchmark."""
    
    def __init__(
        self,
        correlation_id: Optional[str] = None,
        warmup_runs: int = 3,
        benchmark_runs: int = 10
    ) -> None:
        """Initialize benchmark.
        
        Args:
            correlation_id: Optional correlation ID for tracking
            warmup_runs: Number of warmup runs before benchmarking
            benchmark_runs: Number of benchmark runs to average
        """
        self.correlation_id = correlation_id
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.logger = get_logger(__name__, correlation_id, "benchmark")
        
        # Initialize tokenizer
        self.tokenizer = EnhancedXMLTokenizer(
            config=TokenizationConfig.performance_optimized(),
            correlation_id=correlation_id
        )
        
        # Test data
        self.test_cases = self._create_test_cases()
    
    def _create_test_cases(self) -> Dict[str, str]:
        """Create test cases for benchmarking."""
        return {
            "small_well_formed": '''<?xml version="1.0" encoding="UTF-8"?>
<root>
    <element attr="value">Content</element>
    <empty/>
    <!-- Comment -->
</root>''',
            
            "medium_well_formed": '''<?xml version="1.0" encoding="UTF-8"?>
<document>
    <metadata>
        <title>Test Document</title>
        <author>Benchmark Suite</author>
        <version>1.0</version>
    </metadata>
    <content>
        <section id="intro">
            <heading>Introduction</heading>
            <paragraph>This is a test document with multiple elements.</paragraph>
            <list>
                <item>First item</item>
                <item>Second item</item>
                <item>Third item</item>
            </list>
        </section>
        <section id="details">
            <heading>Details</heading>
            <paragraph>More detailed content with <emphasis>nested</emphasis> elements.</paragraph>
            <data value1="123" value2="456" value3="789"/>
        </section>
    </content>
</document>''',
            
            "large_well_formed": self._generate_large_xml(),
            
            "malformed_recoverable": '''<?xml version="1.0"?>
<root>
    <unclosed_tag>
    <element attr=unquoted>Content
    <element attr="unclosed>More content</element>
    <!-- Unclosed comment
    <![CDATA[ Unclosed CDATA
    <empty
</root>''',
            
            "highly_malformed": '''<root><tag1><tag2>content<tag3 attr="value>more<tag4</tag2><tag5/>text</tag1>
<malformed attr=value attr2>content</wrong_close>
<< >> <> ><
<element attr==value attr2=val1=val2>text
<!-- comment <!-- nested comment --> still in comment
<![CDATA[ cdata ]]> more ]]> content
<?pi instruction ? > content</root>''',
        }
    
    def _generate_large_xml(self) -> str:
        """Generate large XML document for benchmarking."""
        elements = []
        elements.append('<?xml version="1.0" encoding="UTF-8"?>')
        elements.append('<large_document>')
        
        for i in range(1000):
            elements.append(f'''
    <item id="{i}" category="test" priority="{i % 10}">
        <title>Item {i}</title>
        <description>This is a description for item {i} with some content.</description>
        <metadata>
            <created>2025-01-01T00:00:00Z</created>
            <modified>2025-01-01T00:00:00Z</modified>
            <tags>
                <tag>benchmark</tag>
                <tag>performance</tag>
                <tag>xml</tag>
            </tags>
        </metadata>
        <content>
            <data value1="{i}" value2="{i*2}" value3="{i*3}"/>
            <text>Some longer text content for item {i} that includes multiple words and sentences.</text>
        </content>
    </item>''')
        
        elements.append('</large_document>')
        return '\n'.join(elements)
    
    def _measure_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def _benchmark_ultra_robust_parser(
        self, 
        test_case: str, 
        xml_content: str
    ) -> BenchmarkResult:
        """Benchmark our ultra-robust parser."""
        # Create character stream result
        encoding_result = EncodingResult(
            encoding="utf-8",
            confidence=1.0,
            method=DetectionMethod.FALLBACK,
            issues=[]
        )
        
        char_stream = CharacterStreamResult(
            text=xml_content,
            encoding=encoding_result,
            confidence=1.0,
            processing_time=0.0,
            diagnostics=[]
        )
        
        # Measure memory before
        gc.collect()
        memory_before = self._measure_memory_usage()
        
        # Benchmark tokenization
        start_time = time.time()
        
        try:
            result = self.tokenizer.tokenize(char_stream)
            success = result.success
            error_message = None
            tokens_generated = len(result.tokens)
        except Exception as e:
            success = False
            error_message = str(e)
            tokens_generated = 0
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Measure memory after
        memory_after = self._measure_memory_usage()
        memory_used = max(0, memory_after - memory_before)
        
        return BenchmarkResult(
            parser_name="ultra_robust_xml_parser",
            test_case=test_case,
            processing_time_ms=processing_time,
            memory_used_mb=memory_used,
            characters_processed=len(xml_content),
            tokens_generated=tokens_generated,
            success=success,
            error_message=error_message
        )
    
    def _benchmark_external_parser(
        self, 
        parser_name: str,
        test_case: str,
        xml_content: str
    ) -> Optional[BenchmarkResult]:
        """Benchmark external XML parser for comparison."""
        try:
            if parser_name == "xml.etree.ElementTree":
                import xml.etree.ElementTree as ET
                
                gc.collect()
                memory_before = self._measure_memory_usage()
                start_time = time.time()
                
                try:
                    root = ET.fromstring(xml_content)
                    # Count elements as rough token equivalent
                    tokens_generated = len(list(root.iter()))
                    success = True
                    error_message = None
                except Exception as e:
                    success = False
                    error_message = str(e)
                    tokens_generated = 0
                
                processing_time = (time.time() - start_time) * 1000
                memory_after = self._measure_memory_usage()
                memory_used = max(0, memory_after - memory_before)
                
                return BenchmarkResult(
                    parser_name=parser_name,
                    test_case=test_case,
                    processing_time_ms=processing_time,
                    memory_used_mb=memory_used,
                    characters_processed=len(xml_content),
                    tokens_generated=tokens_generated,
                    success=success,
                    error_message=error_message
                )
            
            elif parser_name == "lxml":
                try:
                    from lxml import etree
                    
                    gc.collect()
                    memory_before = self._measure_memory_usage()
                    start_time = time.time()
                    
                    try:
                        root = etree.fromstring(xml_content.encode('utf-8'))
                        tokens_generated = len(list(root.iter()))
                        success = True
                        error_message = None
                    except Exception as e:
                        success = False
                        error_message = str(e)
                        tokens_generated = 0
                    
                    processing_time = (time.time() - start_time) * 1000
                    memory_after = self._measure_memory_usage()
                    memory_used = max(0, memory_after - memory_before)
                    
                    return BenchmarkResult(
                        parser_name=parser_name,
                        test_case=test_case,
                        processing_time_ms=processing_time,
                        memory_used_mb=memory_used,
                        characters_processed=len(xml_content),
                        tokens_generated=tokens_generated,
                        success=success,
                        error_message=error_message
                    )
                
                except ImportError:
                    self.logger.warning("lxml not available for benchmarking")
                    return None
            
        except Exception as e:
            self.logger.error(f"Error benchmarking {parser_name}", extra={"error": str(e)})
            return None
        
        return None
    
    def run_benchmark(self, include_external_parsers: bool = True) -> BenchmarkSuite:
        """Run comprehensive benchmark suite.
        
        Args:
            include_external_parsers: Whether to include external parsers for comparison
            
        Returns:
            BenchmarkSuite with all results
        """
        suite = BenchmarkSuite(suite_name="Tokenization Performance Benchmark")
        
        parsers_to_test = ["ultra_robust_xml_parser"]
        if include_external_parsers:
            parsers_to_test.extend(["xml.etree.ElementTree", "lxml"])
        
        self.logger.info(
            "Starting benchmark suite",
            extra={
                "test_cases": len(self.test_cases),
                "parsers": parsers_to_test,
                "warmup_runs": self.warmup_runs,
                "benchmark_runs": self.benchmark_runs
            }
        )
        
        # Run benchmarks for each test case and parser
        for test_case, xml_content in self.test_cases.items():
            self.logger.info(f"Benchmarking test case: {test_case}")
            
            for parser_name in parsers_to_test:
                self.logger.debug(f"Testing parser: {parser_name}")
                
                # Warmup runs
                for _ in range(self.warmup_runs):
                    if parser_name == "ultra_robust_xml_parser":
                        self._benchmark_ultra_robust_parser(test_case, xml_content)
                    else:
                        self._benchmark_external_parser(parser_name, test_case, xml_content)
                
                # Benchmark runs
                run_results = []
                for run in range(self.benchmark_runs):
                    if parser_name == "ultra_robust_xml_parser":
                        result = self._benchmark_ultra_robust_parser(test_case, xml_content)
                    else:
                        result = self._benchmark_external_parser(parser_name, test_case, xml_content)
                    
                    if result:
                        run_results.append(result)
                
                # Calculate average results
                if run_results:
                    successful_runs = [r for r in run_results if r.success]
                    if successful_runs:
                        avg_result = BenchmarkResult(
                            parser_name=parser_name,
                            test_case=test_case,
                            processing_time_ms=statistics.mean([r.processing_time_ms for r in successful_runs]),
                            memory_used_mb=statistics.mean([r.memory_used_mb for r in successful_runs]),
                            characters_processed=successful_runs[0].characters_processed,
                            tokens_generated=int(statistics.mean([r.tokens_generated for r in successful_runs])),
                            success=True
                        )
                        suite.add_result(avg_result)
                    elif run_results:  # All runs failed
                        failed_result = BenchmarkResult(
                            parser_name=parser_name,
                            test_case=test_case,
                            processing_time_ms=0.0,
                            memory_used_mb=0.0,
                            characters_processed=len(xml_content),
                            tokens_generated=0,
                            success=False,
                            error_message=run_results[0].error_message
                        )
                        suite.add_result(failed_result)
        
        self.logger.info(
            "Benchmark suite completed",
            extra={
                "total_results": len(suite.results),
                "suite_duration_minutes": (time.time() - suite.timestamp) / 60
            }
        )
        
        return suite
    
    def compare_performance(
        self, 
        baseline_suite: BenchmarkSuite,
        current_suite: BenchmarkSuite
    ) -> Dict[str, Any]:
        """Compare performance between two benchmark suites.
        
        Args:
            baseline_suite: Baseline benchmark results
            current_suite: Current benchmark results
            
        Returns:
            Performance comparison report
        """
        comparison = {
            "baseline_timestamp": baseline_suite.timestamp,
            "current_timestamp": current_suite.timestamp,
            "improvements": {},
            "regressions": {},
            "summary": {}
        }
        
        # Compare each test case
        for test_case in self.test_cases.keys():
            baseline_results = baseline_suite.get_results_by_test_case(test_case)
            current_results = current_suite.get_results_by_test_case(test_case)
            
            for baseline_result in baseline_results:
                parser_name = baseline_result.parser_name
                current_result = next(
                    (r for r in current_results if r.parser_name == parser_name), None
                )
                
                if current_result and baseline_result.success and current_result.success:
                    # Calculate performance change
                    time_change = (
                        (current_result.processing_time_ms - baseline_result.processing_time_ms) 
                        / baseline_result.processing_time_ms
                    )
                    
                    memory_change = (
                        (current_result.memory_used_mb - baseline_result.memory_used_mb)
                        / baseline_result.memory_used_mb
                        if baseline_result.memory_used_mb > 0 else 0
                    )
                    
                    key = f"{parser_name}_{test_case}"
                    
                    if time_change < -0.05:  # 5% improvement threshold
                        comparison["improvements"][key] = {
                            "time_improvement_percent": abs(time_change) * 100,
                            "memory_change_percent": memory_change * 100,
                            "baseline_time_ms": baseline_result.processing_time_ms,
                            "current_time_ms": current_result.processing_time_ms
                        }
                    elif time_change > 0.05:  # 5% regression threshold
                        comparison["regressions"][key] = {
                            "time_regression_percent": time_change * 100,
                            "memory_change_percent": memory_change * 100,
                            "baseline_time_ms": baseline_result.processing_time_ms,
                            "current_time_ms": current_result.processing_time_ms
                        }
        
        comparison["summary"] = {
            "total_improvements": len(comparison["improvements"]),
            "total_regressions": len(comparison["regressions"]),
            "has_regressions": len(comparison["regressions"]) > 0
        }
        
        return comparison