"""Performance benchmarks for character stream processing.

This module provides performance tests and benchmarks to ensure the character
stream processing meets performance requirements and to detect regressions.
"""

import io
import time
import pytest
from typing import List, Tuple

from ultra_robust_xml_parser.character.stream import (
    CharacterStreamProcessor,
    StreamProcessingPresets,
    StreamingResult,
)


class TestPerformanceBenchmarks:
    """Performance benchmarks for character stream processing."""
    
    def create_test_xml(self, item_count: int, with_issues: bool = False) -> bytes:
        """Create test XML data of specified size.
        
        Args:
            item_count: Number of items to include
            with_issues: Whether to include character issues
            
        Returns:
            Generated XML as bytes
        """
        xml_parts = [b'<?xml version="1.0" encoding="UTF-8"?><items>']
        
        for i in range(item_count):
            if with_issues and i % 100 == 0:
                # Add some problematic characters occasionally
                xml_parts.append(f'<item id="{i}">Content\x00{i}\x01</item>'.encode('utf-8'))
            else:
                xml_parts.append(f'<item id="{i}">Content {i}</item>'.encode('utf-8'))
        
        xml_parts.append(b'</items>')
        return b''.join(xml_parts)
    
    def benchmark_processing(self, processor: CharacterStreamProcessor, data: bytes) -> dict:
        """Benchmark processing performance.
        
        Args:
            processor: Processor to benchmark
            data: Data to process
            
        Returns:
            Performance metrics
        """
        start_time = time.time()
        result = processor.process(data)
        end_time = time.time()
        
        return {
            'processing_time': end_time - start_time,
            'input_size': len(data),
            'output_size': len(result.text),
            'confidence': result.confidence,
            'diagnostics_count': len(result.diagnostics),
            'throughput_mb_per_sec': (len(data) / (1024 * 1024)) / (end_time - start_time) if end_time > start_time else 0
        }
    
    def benchmark_streaming(self, processor: CharacterStreamProcessor, data: bytes) -> dict:
        """Benchmark streaming performance.
        
        Args:
            processor: Processor to benchmark
            data: Data to process
            
        Returns:
            Performance metrics
        """
        start_time = time.time()
        stream_result = processor.process_stream(data)
        
        chunk_count = 0
        total_output_size = 0
        for chunk in stream_result.chunks:
            chunk_count += 1
            total_output_size += len(chunk)
        
        end_time = time.time()
        
        return {
            'processing_time': end_time - start_time,
            'input_size': len(data),
            'output_size': total_output_size,
            'chunk_count': chunk_count,
            'throughput_mb_per_sec': (len(data) / (1024 * 1024)) / (end_time - start_time) if end_time > start_time else 0
        }
    
    @pytest.mark.performance
    def test_small_document_performance(self):
        """Benchmark performance with small documents (< 10KB)."""
        processor = CharacterStreamProcessor()
        test_data = self.create_test_xml(100)  # ~10KB
        
        metrics = self.benchmark_processing(processor, test_data)
        
        # Performance assertions
        assert metrics['processing_time'] < 0.1  # Should process in < 100ms
        assert metrics['throughput_mb_per_sec'] > 0.1  # At least 0.1 MB/s
        assert metrics['confidence'] > 0.9  # High confidence for clean data
        
        print(f"Small document performance:")
        print(f"  Processing time: {metrics['processing_time']:.3f}s")
        print(f"  Throughput: {metrics['throughput_mb_per_sec']:.2f} MB/s")
    
    @pytest.mark.performance
    def test_medium_document_performance(self):
        """Benchmark performance with medium documents (100KB - 1MB)."""
        processor = CharacterStreamProcessor()
        test_data = self.create_test_xml(10000)  # ~1MB
        
        metrics = self.benchmark_processing(processor, test_data)
        
        # Performance assertions
        assert metrics['processing_time'] < 1.0  # Should process in < 1 second
        assert metrics['throughput_mb_per_sec'] > 1.0  # At least 1 MB/s
        
        print(f"Medium document performance:")
        print(f"  Input size: {metrics['input_size']:,} bytes")
        print(f"  Processing time: {metrics['processing_time']:.3f}s")
        print(f"  Throughput: {metrics['throughput_mb_per_sec']:.2f} MB/s")
    
    @pytest.mark.performance
    def test_large_document_performance(self):
        """Benchmark performance with large documents (> 1MB)."""
        processor = CharacterStreamProcessor()
        test_data = self.create_test_xml(50000)  # ~5MB
        
        metrics = self.benchmark_processing(processor, test_data)
        
        # Performance assertions
        assert metrics['processing_time'] < 5.0  # Should process in < 5 seconds
        assert metrics['throughput_mb_per_sec'] > 1.0  # At least 1 MB/s
        
        print(f"Large document performance:")
        print(f"  Input size: {metrics['input_size']:,} bytes")
        print(f"  Processing time: {metrics['processing_time']:.3f}s")
        print(f"  Throughput: {metrics['throughput_mb_per_sec']:.2f} MB/s")
    
    @pytest.mark.performance
    def test_streaming_vs_normal_performance(self):
        """Compare streaming vs normal processing performance."""
        test_data = self.create_test_xml(20000)  # ~2MB
        
        # Normal processing
        normal_processor = CharacterStreamProcessor()
        normal_metrics = self.benchmark_processing(normal_processor, test_data)
        
        # Streaming processing
        streaming_processor = CharacterStreamProcessor(
            StreamProcessingPresets.web_scraping(buffer_size=8192)
        )
        streaming_metrics = self.benchmark_streaming(streaming_processor, test_data)
        
        # Both should complete reasonably quickly
        assert normal_metrics['processing_time'] < 2.0
        assert streaming_metrics['processing_time'] < 2.0
        
        # Output sizes should match
        assert normal_metrics['output_size'] == streaming_metrics['output_size']
        
        print(f"Performance comparison:")
        print(f"  Normal: {normal_metrics['processing_time']:.3f}s, "
              f"{normal_metrics['throughput_mb_per_sec']:.2f} MB/s")
        print(f"  Streaming: {streaming_metrics['processing_time']:.3f}s, "
              f"{streaming_metrics['throughput_mb_per_sec']:.2f} MB/s, "
              f"{streaming_metrics['chunk_count']} chunks")
    
    @pytest.mark.performance
    def test_problematic_content_performance(self):
        """Benchmark performance with problematic content requiring transformations."""
        processor = CharacterStreamProcessor(StreamProcessingPresets.data_recovery())
        test_data = self.create_test_xml(5000, with_issues=True)  # ~500KB with issues
        
        metrics = self.benchmark_processing(processor, test_data)
        
        # Should still maintain reasonable performance despite transformations
        assert metrics['processing_time'] < 2.0  # Should process in < 2 seconds
        assert metrics['throughput_mb_per_sec'] > 0.5  # At least 0.5 MB/s
        
        print(f"Problematic content performance:")
        print(f"  Processing time: {metrics['processing_time']:.3f}s")
        print(f"  Throughput: {metrics['throughput_mb_per_sec']:.2f} MB/s")
        print(f"  Diagnostics: {metrics['diagnostics_count']}")
    
    @pytest.mark.performance
    def test_different_buffer_sizes_performance(self):
        """Benchmark performance with different buffer sizes."""
        test_data = self.create_test_xml(10000)  # ~1MB
        buffer_sizes = [1024, 4096, 8192, 16384, 32768]
        
        results = []
        
        for buffer_size in buffer_sizes:
            config = StreamProcessingPresets.web_scraping(buffer_size=buffer_size)
            processor = CharacterStreamProcessor(config)
            
            metrics = self.benchmark_streaming(processor, test_data)
            metrics['buffer_size'] = buffer_size
            results.append(metrics)
            
            # Basic performance assertion
            assert metrics['processing_time'] < 2.0
            assert metrics['throughput_mb_per_sec'] > 0.5
        
        print(f"Buffer size performance comparison:")
        for result in results:
            print(f"  {result['buffer_size']:5d} bytes: {result['processing_time']:.3f}s, "
                  f"{result['throughput_mb_per_sec']:.2f} MB/s, {result['chunk_count']} chunks")
    
    @pytest.mark.performance
    def test_memory_usage_baseline(self):
        """Baseline memory usage test for streaming vs normal processing."""
        import gc
        import sys
        
        # This is a basic memory usage test - more sophisticated profiling
        # would require memory_profiler or similar tools
        
        test_data = self.create_test_xml(20000)  # ~2MB
        
        # Measure normal processing
        gc.collect()
        normal_processor = CharacterStreamProcessor()
        start_mem = sys.getsizeof(test_data)
        result = normal_processor.process(test_data)
        normal_result_size = sys.getsizeof(result.text)
        
        # Measure streaming processing
        gc.collect()
        streaming_processor = CharacterStreamProcessor(
            StreamProcessingPresets.web_scraping(buffer_size=4096)
        )
        stream_result = streaming_processor.process_stream(test_data)
        
        # Process stream in chunks to simulate real usage
        total_chunks_size = 0
        chunk_count = 0
        for chunk in stream_result.chunks:
            chunk_count += 1
            total_chunks_size += len(chunk)
            # In real usage, chunks would be processed and discarded
        
        # Basic assertions
        assert normal_result_size > 0
        assert total_chunks_size == normal_result_size  # Same content
        assert chunk_count > 1  # Multiple chunks processed
        
        print(f"Memory usage baseline:")
        print(f"  Input size: {len(test_data):,} bytes")
        print(f"  Normal result size: {normal_result_size:,} bytes")
        print(f"  Streaming chunks: {chunk_count}")
        print(f"  Total streaming output: {total_chunks_size:,} bytes")


class TestPerformanceRegression:
    """Tests to detect performance regressions."""
    
    def test_baseline_performance_regression(self):
        """Baseline performance test to detect regressions."""
        processor = CharacterStreamProcessor()
        
        # Standard test case
        test_data = b'<?xml version="1.0"?><root>' + b'<item>test</item>' * 1000 + b'</root>'
        
        # Performance measurement
        start_time = time.time()
        result = processor.process(test_data)
        processing_time = time.time() - start_time
        
        # Basic regression checks
        assert processing_time < 0.5  # Should complete in < 500ms
        assert len(result.text) > 0  # Should produce output
        assert result.confidence > 0.8  # Should have high confidence
        
        # Calculate throughput
        throughput = len(test_data) / (processing_time * 1024 * 1024)  # MB/s
        assert throughput > 0.1  # At least 0.1 MB/s
        
        print(f"Baseline performance: {processing_time:.3f}s, {throughput:.2f} MB/s")
    
    def test_streaming_performance_regression(self):
        """Streaming performance regression test."""
        processor = CharacterStreamProcessor(
            StreamProcessingPresets.web_scraping(buffer_size=4096)
        )
        
        test_data = b'<items>' + b'<item>data</item>' * 5000 + b'</items>'
        
        start_time = time.time()
        stream_result = processor.process_stream(test_data)
        
        chunk_count = 0
        for chunk in stream_result.chunks:
            chunk_count += 1
            # Simulate processing delay
            time.sleep(0.0001)  # 0.1ms per chunk
        
        processing_time = time.time() - start_time
        
        # Regression checks
        assert processing_time < 2.0  # Should complete in < 2s including delays
        assert chunk_count > 1  # Should produce multiple chunks
        
        throughput = len(test_data) / ((processing_time - chunk_count * 0.0001) * 1024 * 1024)
        assert throughput > 0.1  # At least 0.1 MB/s excluding artificial delays
        
        print(f"Streaming performance: {processing_time:.3f}s, {chunk_count} chunks")


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "-m", "performance", "-s"])