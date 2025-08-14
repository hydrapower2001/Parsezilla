#!/usr/bin/env python3
"""
Performance benchmarking demonstration for the ultra-robust XML parser.

This example shows how to use the performance benchmarking system to measure
and compare the performance of validation, optimization, and formatting operations.
"""

import sys
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ultra_robust_xml_parser.tree.builder import XMLDocument, XMLElement
from ultra_robust_xml_parser.tree.validation import (
    PerformanceBenchmark, ValidationLevel, OutputFormat
)


def create_benchmark_documents():
    """Create various test documents for benchmarking."""
    documents = []
    
    # 1. Simple document
    simple_root = XMLElement(tag="simple", text="Simple document")
    documents.append(XMLDocument(root=simple_root))
    
    # 2. Medium complexity document
    medium_root = XMLElement(tag="catalog")
    for i in range(10):
        product = XMLElement(
            tag="product",
            attributes={"id": str(i), "category": "electronics"},
            children=[
                XMLElement(tag="name", text=f"Product {i}"),
                XMLElement(tag="price", text=f"{19.99 + i * 5:.2f}"),
                XMLElement(tag="description", text=f"  Description   with   spaces  {i}  ")
            ]
        )
        # Set parent relationships
        for child in product.children:
            child.parent = product
        medium_root.children.append(product)
        product.parent = medium_root
    
    documents.append(XMLDocument(root=medium_root))
    
    # 3. Complex nested document
    complex_root = XMLElement(tag="complex")
    current = complex_root
    
    # Create nested structure
    for level in range(5):
        for i in range(3):
            element = XMLElement(
                tag=f"level{level}_item{i}",
                attributes={"level": str(level), "item": str(i), "empty_attr": ""},
                text=f"Content at level {level}, item {i}" if i % 2 == 0 else None
            )
            current.children.append(element)
            element.parent = current
            
            if level < 4:  # Go deeper
                current = element
    
    documents.append(XMLDocument(root=complex_root))
    
    # 4. Document with optimization opportunities
    optimization_root = XMLElement(tag="optimization_test")
    
    # Add elements with whitespace issues
    for i in range(5):
        element = XMLElement(
            tag="whitespace_element",
            text=f"   Text   with   lots   of   spaces   {i}   ",
            attributes={"empty1": "", "valid": f"value{i}", "empty2": ""}
        )
        optimization_root.children.append(element)
        element.parent = optimization_root
    
    # Add redundant empty elements
    for i in range(3):
        empty = XMLElement(tag=f"empty{i}")
        optimization_root.children.append(empty)
        empty.parent = optimization_root
    
    documents.append(XMLDocument(root=optimization_root))
    
    print(f"📄 Created {len(documents)} benchmark documents")
    for i, doc in enumerate(documents, 1):
        print(f"  Document {i}: {doc.total_elements} elements, depth {doc.max_depth}")
    
    return documents


def demonstrate_validation_benchmark(benchmark: PerformanceBenchmark, documents):
    """Demonstrate validation benchmarking."""
    print("\n🔍 VALIDATION BENCHMARK")
    print("=" * 40)
    
    # Benchmark different validation levels
    validation_levels = [ValidationLevel.MINIMAL, ValidationLevel.STANDARD, ValidationLevel.STRICT]
    suite = benchmark.benchmark_validation(documents, validation_levels)
    
    print(f"✅ Validation benchmark completed:")
    print(f"  Total tests: {len(suite.results)}")
    print(f"  Success rate: {suite.success_rate:.1f}%")
    print(f"  Total time: {suite.total_time_ms:.2f}ms")
    print(f"  Average time: {suite.average_processing_time_ms:.2f}ms")
    print(f"  Elements processed: {suite.total_elements_processed}")
    
    # Show best results by different metrics
    fastest = suite.get_best_result("processing_time_ms")
    if fastest:
        print(f"\n⚡ Fastest test:")
        print(f"  Test: {fastest.test_name}")
        print(f"  Time: {fastest.processing_time_ms:.2f}ms")
        print(f"  Level: {fastest.details.get('validation_level')}")
    
    highest_throughput = suite.get_best_result("throughput_elements_per_second")
    if highest_throughput:
        print(f"\n📈 Highest throughput:")
        print(f"  Test: {highest_throughput.test_name}")
        print(f"  Throughput: {highest_throughput.throughput_elements_per_second:.0f} elements/sec")
    
    return suite


def demonstrate_optimization_benchmark(benchmark: PerformanceBenchmark, documents):
    """Demonstrate optimization benchmarking."""
    print("\n⚡ OPTIMIZATION BENCHMARK")
    print("=" * 40)
    
    suite = benchmark.benchmark_optimization(documents)
    
    print(f"✅ Optimization benchmark completed:")
    print(f"  Total tests: {len(suite.results)}")
    print(f"  Success rate: {suite.success_rate:.1f}%")
    print(f"  Total time: {suite.total_time_ms:.2f}ms")
    print(f"  Average time: {suite.average_processing_time_ms:.2f}ms")
    
    # Show optimization effectiveness
    total_elements_removed = 0
    total_memory_saved = 0
    
    for result in suite.results:
        if result.success and 'elements_removed' in result.details:
            total_elements_removed += result.details['elements_removed']
        if result.success and 'memory_saved_bytes' in result.details:
            total_memory_saved += result.details['memory_saved_bytes']
    
    print(f"\n🔧 Optimization effectiveness:")
    print(f"  Total elements removed: {total_elements_removed}")
    print(f"  Total memory saved: {total_memory_saved} bytes")
    
    # Show most effective optimization
    most_effective = None
    max_elements_removed = 0
    
    for result in suite.results:
        if result.success and 'elements_removed' in result.details:
            removed = result.details['elements_removed']
            if removed > max_elements_removed:
                max_elements_removed = removed
                most_effective = result
    
    if most_effective:
        print(f"\n🏆 Most effective optimization:")
        print(f"  Test: {most_effective.test_name}")
        print(f"  Elements removed: {most_effective.details['elements_removed']}")
        print(f"  Actions performed: {most_effective.details['actions_performed']}")
    
    return suite


def demonstrate_formatting_benchmark(benchmark: PerformanceBenchmark, documents):
    """Demonstrate formatting benchmarking."""
    print("\n🔄 FORMATTING BENCHMARK")
    print("=" * 40)
    
    # Test common output formats
    formats = [OutputFormat.XML_STRING, OutputFormat.XML_PRETTY, OutputFormat.JSON, OutputFormat.DICTIONARY]
    suite = benchmark.benchmark_formatting(documents, formats)
    
    print(f"✅ Formatting benchmark completed:")
    print(f"  Total tests: {len(suite.results)}")
    print(f"  Success rate: {suite.success_rate:.1f}%")
    print(f"  Total time: {suite.total_time_ms:.2f}ms")
    print(f"  Average time: {suite.average_processing_time_ms:.2f}ms")
    
    # Analyze output format performance
    format_stats = {}
    for result in suite.results:
        if result.success and 'output_format' in result.details:
            format_name = result.details['output_format']
            if format_name not in format_stats:
                format_stats[format_name] = {'times': [], 'sizes': []}
            
            format_stats[format_name]['times'].append(result.processing_time_ms)
            if 'output_size_bytes' in result.details:
                format_stats[format_name]['sizes'].append(result.details['output_size_bytes'])
    
    print(f"\n📊 Performance by format:")
    for format_name, stats in format_stats.items():
        avg_time = sum(stats['times']) / len(stats['times'])
        avg_size = sum(stats['sizes']) / len(stats['sizes']) if stats['sizes'] else 0
        print(f"  {format_name}:")
        print(f"    Average time: {avg_time:.2f}ms")
        print(f"    Average size: {avg_size:.0f} bytes")
    
    return suite


def demonstrate_comprehensive_report(benchmark: PerformanceBenchmark):
    """Demonstrate comprehensive benchmark reporting."""
    print("\n📊 COMPREHENSIVE BENCHMARK REPORT")
    print("=" * 45)
    
    # Generate detailed report
    report = benchmark.generate_benchmark_report(include_details=True)
    
    print(f"📋 Benchmark Summary:")
    summary = report['benchmark_summary']
    print(f"  Total suites: {summary['total_suites']}")
    print(f"  Generated at: {summary['generated_at']}")
    if summary.get('correlation_id'):
        print(f"  Correlation ID: {summary['correlation_id']}")
    
    print(f"\n📈 Suite Results:")
    for suite_data in report['suites']:
        print(f"\n  {suite_data['name']}:")
        print(f"    Tests: {suite_data['total_tests']}")
        print(f"    Success rate: {suite_data['success_rate']:.1f}%")
        print(f"    Total time: {suite_data['total_time_ms']:.2f}ms")
        print(f"    Average time: {suite_data['average_time_ms']:.2f}ms")
        print(f"    Elements processed: {suite_data['total_elements']}")
        
        # Show statistics
        stats = suite_data.get('summary_stats', {})
        if stats:
            print(f"    Statistics:")
            print(f"      Min time: {stats['min_time_ms']:.2f}ms")
            print(f"      Max time: {stats['max_time_ms']:.2f}ms")
            print(f"      Median time: {stats['median_time_ms']:.2f}ms")
            if stats['avg_throughput'] > 0:
                print(f"      Avg throughput: {stats['avg_throughput']:.0f} elem/sec")
    
    # Export report to JSON for external analysis
    report_file = Path(__file__).parent / "benchmark_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Full report exported to: {report_file}")
    
    return report


def demonstrate_performance_comparison():
    """Demonstrate performance comparison between configurations."""
    print("\n⚖️  PERFORMANCE COMPARISON")
    print("=" * 40)
    
    # Create test document
    root = XMLElement(tag="comparison_test")
    for i in range(20):
        element = XMLElement(
            tag="item",
            attributes={"id": str(i), "empty": ""},
            text=f"   Item   {i}   content   "
        )
        root.children.append(element)
        element.parent = root
    
    document = XMLDocument(root=root)
    
    print(f"📄 Test document: {document.total_elements} elements")
    
    # Compare validation levels
    print(f"\n🔍 Validation Level Comparison:")
    benchmark = PerformanceBenchmark(correlation_id="comparison-demo")
    
    levels = [ValidationLevel.MINIMAL, ValidationLevel.STANDARD, ValidationLevel.STRICT, ValidationLevel.PEDANTIC]
    
    for level in levels:
        suite = benchmark.benchmark_validation([document], [level])
        if suite.results:
            result = suite.results[0]
            print(f"  {level.name:10}: {result.processing_time_ms:6.2f}ms, "
                  f"{result.throughput_elements_per_second:6.0f} elem/sec, "
                  f"confidence: {result.details.get('confidence', 0):.3f}")
    
    # Compare output formats
    print(f"\n🔄 Output Format Comparison:")
    formats = [OutputFormat.XML_STRING, OutputFormat.XML_PRETTY, OutputFormat.JSON, OutputFormat.DICTIONARY]
    
    format_suite = benchmark.benchmark_formatting([document], formats)
    
    for format_type in formats:
        # Find result for this format
        format_results = [r for r in format_suite.results if r.details.get('output_format') == format_type.value]
        if format_results:
            result = format_results[0]
            output_size = result.details.get('output_size_bytes', 0)
            print(f"  {format_type.value:12}: {result.processing_time_ms:6.2f}ms, "
                  f"{output_size:5d} bytes, "
                  f"{result.throughput_elements_per_second:6.0f} elem/sec")


def main():
    """Main demonstration function."""
    print("🚀 ULTRA-ROBUST XML PARSER - PERFORMANCE BENCHMARK DEMO")
    print("=" * 65)
    
    try:
        # Create benchmark documents
        documents = create_benchmark_documents()
        
        # Initialize benchmark system
        benchmark = PerformanceBenchmark(correlation_id="demo-benchmark")
        
        # Run comprehensive benchmarks
        validation_suite = demonstrate_validation_benchmark(benchmark, documents)
        optimization_suite = demonstrate_optimization_benchmark(benchmark, documents)
        formatting_suite = demonstrate_formatting_benchmark(benchmark, documents)
        
        # Generate comprehensive report
        report = demonstrate_comprehensive_report(benchmark)
        
        # Show performance comparisons
        demonstrate_performance_comparison()
        
        print(f"\n🎉 BENCHMARK DEMONSTRATION COMPLETE!")
        print("✅ All benchmarking features demonstrated successfully")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ BENCHMARK FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())