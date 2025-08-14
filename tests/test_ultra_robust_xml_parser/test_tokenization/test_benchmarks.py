"""Tests for tokenization performance benchmarking.

This module tests the performance benchmarking system for tokenization,
including comparison with external parsers and performance regression detection.
"""

import pytest
import time
from unittest.mock import Mock, patch

from ultra_robust_xml_parser.tokenization.benchmarks import (
    BenchmarkResult,
    BenchmarkSuite,
    TokenizationBenchmark,
)


class TestBenchmarkResult:
    """Test benchmark result data structure."""
    
    def test_benchmark_result_creation(self):
        """Test basic benchmark result creation."""
        result = BenchmarkResult(
            parser_name="test_parser",
            test_case="small_xml",
            processing_time_ms=10.5,
            memory_used_mb=2.3,
            characters_processed=1000,
            tokens_generated=50,
            success=True
        )
        
        assert result.parser_name == "test_parser"
        assert result.test_case == "small_xml"
        assert result.processing_time_ms == 10.5
        assert result.memory_used_mb == 2.3
        assert result.characters_processed == 1000
        assert result.tokens_generated == 50
        assert result.success is True
        assert result.error_message is None
    
    def test_performance_metrics_calculation(self):
        """Test calculated performance metrics."""
        result = BenchmarkResult(
            parser_name="test",
            test_case="test",
            processing_time_ms=100.0,  # 100ms
            memory_used_mb=2.0,
            characters_processed=1000,
            tokens_generated=50,
            success=True
        )
        
        # 1000 chars / 0.1 seconds = 10000 chars/sec
        assert result.characters_per_second == 10000.0
        
        # 50 tokens / 0.1 seconds = 500 tokens/sec
        assert result.tokens_per_second == 500.0
        
        # 2MB / 1000 chars = 2048 bytes/char
        assert result.memory_per_character == 2048.0
    
    def test_zero_division_handling(self):
        """Test handling of zero division in calculations."""
        result = BenchmarkResult(
            parser_name="test",
            test_case="test",
            processing_time_ms=0.0,
            memory_used_mb=1.0,
            characters_processed=0,
            tokens_generated=0,
            success=True
        )
        
        assert result.characters_per_second == 0.0
        assert result.tokens_per_second == 0.0
        assert result.memory_per_character == 0.0


class TestBenchmarkSuite:
    """Test benchmark suite management."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.suite = BenchmarkSuite("Test Suite")
        
        # Add sample results
        self.results = [
            BenchmarkResult(
                parser_name="parser_a",
                test_case="case_1",
                processing_time_ms=10.0,
                memory_used_mb=1.0,
                characters_processed=100,
                tokens_generated=10,
                success=True
            ),
            BenchmarkResult(
                parser_name="parser_a",
                test_case="case_2", 
                processing_time_ms=20.0,
                memory_used_mb=2.0,
                characters_processed=200,
                tokens_generated=20,
                success=True
            ),
            BenchmarkResult(
                parser_name="parser_b",
                test_case="case_1",
                processing_time_ms=15.0,
                memory_used_mb=1.5,
                characters_processed=100,
                tokens_generated=10,
                success=True
            ),
            BenchmarkResult(
                parser_name="parser_b",
                test_case="case_2",
                processing_time_ms=0.0,
                memory_used_mb=0.0,
                characters_processed=200,
                tokens_generated=0,
                success=False,
                error_message="Parse error"
            ),
        ]
        
        for result in self.results:
            self.suite.add_result(result)
    
    def test_suite_initialization(self):
        """Test suite initialization."""
        suite = BenchmarkSuite("Custom Suite")
        
        assert suite.suite_name == "Custom Suite"
        assert len(suite.results) == 0
        assert suite.timestamp > 0
    
    def test_result_filtering(self):
        """Test filtering results by parser and test case."""
        # Filter by parser
        parser_a_results = self.suite.get_results_by_parser("parser_a")
        assert len(parser_a_results) == 2
        assert all(r.parser_name == "parser_a" for r in parser_a_results)
        
        parser_b_results = self.suite.get_results_by_parser("parser_b")
        assert len(parser_b_results) == 2
        
        # Filter by test case
        case_1_results = self.suite.get_results_by_test_case("case_1")
        assert len(case_1_results) == 2
        assert all(r.test_case == "case_1" for r in case_1_results)
    
    def test_statistics_calculation(self):
        """Test statistical analysis."""
        stats = self.suite.get_statistics("parser_a", "processing_time_ms")
        
        assert stats["min"] == 10.0
        assert stats["max"] == 20.0
        assert stats["mean"] == 15.0
        assert stats["median"] == 15.0
        assert stats["count"] == 2
        assert stats["stdev"] > 0  # Standard deviation should be > 0
    
    def test_empty_statistics(self):
        """Test statistics for non-existent parser."""
        stats = self.suite.get_statistics("nonexistent", "processing_time_ms")
        assert stats == {}
    
    def test_report_generation(self):
        """Test comprehensive report generation."""
        report = self.suite.generate_report()
        
        assert report["suite_name"] == "Test Suite"
        assert report["total_results"] == 4
        assert "parser_a" in report["parsers"]
        assert "parser_b" in report["parsers"]
        assert "case_1" in report["test_cases"]
        assert "case_2" in report["test_cases"]
        
        # Check summary statistics
        assert "summary" in report
        assert "parser_a" in report["summary"]
        assert "parser_b" in report["summary"]
        
        parser_a_summary = report["summary"]["parser_a"]
        assert parser_a_summary["total_runs"] == 2
        assert parser_a_summary["successful_runs"] == 2
        assert parser_a_summary["success_rate"] == 1.0
        
        parser_b_summary = report["summary"]["parser_b"]
        assert parser_b_summary["total_runs"] == 2
        assert parser_b_summary["successful_runs"] == 1
        assert parser_b_summary["success_rate"] == 0.5
        
        # Check detailed results
        assert "detailed_results" in report
        assert "case_1" in report["detailed_results"]
        assert "parser_a" in report["detailed_results"]["case_1"]
        assert "parser_b" in report["detailed_results"]["case_1"]


class TestTokenizationBenchmark:
    """Test tokenization benchmark execution."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.benchmark = TokenizationBenchmark(
            correlation_id="test-123",
            warmup_runs=1,
            benchmark_runs=2
        )
    
    def test_benchmark_initialization(self):
        """Test benchmark initialization."""
        assert self.benchmark.correlation_id == "test-123"
        assert self.benchmark.warmup_runs == 1
        assert self.benchmark.benchmark_runs == 2
        assert len(self.benchmark.test_cases) > 0
    
    def test_test_cases_creation(self):
        """Test that test cases are created properly."""
        test_cases = self.benchmark.test_cases
        
        # Should have various test case types
        expected_cases = [
            "small_well_formed",
            "medium_well_formed", 
            "large_well_formed",
            "malformed_recoverable",
            "highly_malformed"
        ]
        
        for case in expected_cases:
            assert case in test_cases
            assert len(test_cases[case]) > 0  # Not empty
    
    def test_large_xml_generation(self):
        """Test large XML generation."""
        large_xml = self.benchmark._generate_large_xml()
        
        assert len(large_xml) > 10000  # Should be substantial
        assert "<?xml version" in large_xml
        assert "<large_document>" in large_xml
        assert "</large_document>" in large_xml
        assert large_xml.count("<item") == 1000  # 1000 items
    
    @patch('psutil.Process')
    def test_memory_measurement(self, mock_process):
        """Test memory usage measurement."""
        # Mock memory info
        mock_memory = Mock()
        mock_memory.rss = 1024 * 1024 * 50  # 50MB in bytes
        mock_process.return_value.memory_info.return_value = mock_memory
        
        memory_mb = self.benchmark._measure_memory_usage()
        
        assert memory_mb == 50.0  # 50MB
    
    def test_ultra_robust_parser_benchmark(self):
        """Test benchmarking our ultra-robust parser."""
        xml_content = '''<?xml version="1.0"?>
<root>
    <element>Content</element>
</root>'''
        
        with patch.object(self.benchmark, '_measure_memory_usage', side_effect=[10.0, 12.0]):
            result = self.benchmark._benchmark_ultra_robust_parser("test_case", xml_content)
        
        assert isinstance(result, BenchmarkResult)
        assert result.parser_name == "ultra_robust_xml_parser"
        assert result.test_case == "test_case"
        assert result.processing_time_ms > 0
        assert result.memory_used_mb >= 0
        assert result.characters_processed == len(xml_content)
        assert result.success is True
        assert result.tokens_generated > 0
    
    @patch('xml.etree.ElementTree.fromstring')
    def test_external_parser_benchmark_elementtree(self, mock_fromstring):
        """Test benchmarking ElementTree parser."""
        xml_content = "<root><element>Content</element></root>"
        
        # Mock ElementTree parsing
        mock_root = Mock()
        mock_root.iter.return_value = [Mock(), Mock(), Mock()]  # 3 elements
        mock_fromstring.return_value = mock_root
        
        with patch.object(self.benchmark, '_measure_memory_usage', side_effect=[5.0, 6.0]):
            result = self.benchmark._benchmark_external_parser(
                "xml.etree.ElementTree", "test_case", xml_content
            )
        
        assert result is not None
        assert result.parser_name == "xml.etree.ElementTree"
        assert result.success is True
        assert result.tokens_generated == 3
        assert result.memory_used_mb == 1.0  # 6.0 - 5.0
    
    @patch('xml.etree.ElementTree.fromstring')
    def test_external_parser_error_handling(self, mock_fromstring):
        """Test error handling in external parser benchmarking."""
        xml_content = "<malformed><element>Content"
        
        # Mock parsing error
        mock_fromstring.side_effect = Exception("Parse error")
        
        with patch.object(self.benchmark, '_measure_memory_usage', side_effect=[5.0, 6.0]):
            result = self.benchmark._benchmark_external_parser(
                "xml.etree.ElementTree", "test_case", xml_content
            )
        
        assert result is not None
        assert result.success is False
        assert result.error_message == "Parse error"
        assert result.tokens_generated == 0
    
    @patch('ultra_robust_xml_parser.tokenization.benchmarks.TokenizationBenchmark._benchmark_ultra_robust_parser')
    @patch('ultra_robust_xml_parser.tokenization.benchmarks.TokenizationBenchmark._benchmark_external_parser')
    def test_full_benchmark_run(self, mock_external, mock_internal):
        """Test full benchmark execution."""
        # Mock benchmark results
        mock_internal.return_value = BenchmarkResult(
            parser_name="ultra_robust_xml_parser",
            test_case="test_case",
            processing_time_ms=10.0,
            memory_used_mb=1.0,
            characters_processed=100,
            tokens_generated=10,
            success=True
        )
        
        mock_external.return_value = BenchmarkResult(
            parser_name="xml.etree.ElementTree",
            test_case="test_case", 
            processing_time_ms=15.0,
            memory_used_mb=1.5,
            characters_processed=100,
            tokens_generated=8,
            success=True
        )
        
        # Reduce test cases for faster test
        original_test_cases = self.benchmark.test_cases
        self.benchmark.test_cases = {"small_test": "<root></root>"}
        
        try:
            suite = self.benchmark.run_benchmark(include_external_parsers=True)
            
            assert isinstance(suite, BenchmarkSuite)
            assert len(suite.results) > 0
            
            # Check that warmup + benchmark runs were called
            expected_calls_per_parser = self.benchmark.warmup_runs + self.benchmark.benchmark_runs
            assert mock_internal.call_count >= expected_calls_per_parser
            
        finally:
            # Restore original test cases
            self.benchmark.test_cases = original_test_cases
    
    def test_performance_comparison(self):
        """Test performance comparison between benchmark suites."""
        # Create baseline suite
        baseline = BenchmarkSuite("Baseline")
        baseline.add_result(BenchmarkResult(
            parser_name="parser",
            test_case="case",
            processing_time_ms=100.0,
            memory_used_mb=2.0,
            characters_processed=1000,
            tokens_generated=50,
            success=True
        ))
        
        # Create current suite (improved performance)
        current = BenchmarkSuite("Current")
        current.add_result(BenchmarkResult(
            parser_name="parser", 
            test_case="case",
            processing_time_ms=80.0,  # 20% improvement
            memory_used_mb=1.8,
            characters_processed=1000,
            tokens_generated=50,
            success=True
        ))
        
        comparison = self.benchmark.compare_performance(baseline, current)
        
        assert "baseline_timestamp" in comparison
        assert "current_timestamp" in comparison
        assert "improvements" in comparison
        assert "regressions" in comparison
        assert "summary" in comparison
        
        # Should detect improvement
        assert len(comparison["improvements"]) > 0
        assert len(comparison["regressions"]) == 0
        assert comparison["summary"]["has_regressions"] is False
        
        # Check improvement details
        improvement_key = "parser_case"
        assert improvement_key in comparison["improvements"]
        improvement = comparison["improvements"][improvement_key]
        assert improvement["time_improvement_percent"] == 20.0
        assert improvement["baseline_time_ms"] == 100.0
        assert improvement["current_time_ms"] == 80.0
    
    def test_performance_regression_detection(self):
        """Test detection of performance regressions."""
        # Create baseline suite
        baseline = BenchmarkSuite("Baseline")
        baseline.add_result(BenchmarkResult(
            parser_name="parser",
            test_case="case", 
            processing_time_ms=100.0,
            memory_used_mb=2.0,
            characters_processed=1000,
            tokens_generated=50,
            success=True
        ))
        
        # Create current suite (worse performance)
        current = BenchmarkSuite("Current")
        current.add_result(BenchmarkResult(
            parser_name="parser",
            test_case="case",
            processing_time_ms=130.0,  # 30% regression
            memory_used_mb=2.5,
            characters_processed=1000,
            tokens_generated=50,
            success=True
        ))
        
        comparison = self.benchmark.compare_performance(baseline, current)
        
        # Should detect regression
        assert len(comparison["improvements"]) == 0
        assert len(comparison["regressions"]) > 0
        assert comparison["summary"]["has_regressions"] is True
        
        # Check regression details
        regression_key = "parser_case"
        assert regression_key in comparison["regressions"]
        regression = comparison["regressions"][regression_key]
        assert regression["time_regression_percent"] == 30.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])