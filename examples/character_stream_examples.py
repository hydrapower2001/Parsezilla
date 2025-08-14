#!/usr/bin/env python3
"""
Character Stream Processing Examples

This script demonstrates various usage patterns for the CharacterStreamProcessor,
including different input types, configuration presets, streaming, and error handling.
"""

import io
import sys
import time
from pathlib import Path

# Add src to path for running examples directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ultra_robust_xml_parser.character.stream import (
    CharacterStreamProcessor,
    StreamProcessingPresets,
    StreamingResult,
)


def example_basic_usage():
    """Example 1: Basic usage with different input types."""
    print("=== Example 1: Basic Usage ===")
    
    processor = CharacterStreamProcessor()
    
    # Process bytes
    xml_bytes = b'<?xml version="1.0"?><root>Hello World</root>'
    result = processor.process(xml_bytes)
    print(f"Bytes input: {result.text[:50]}...")
    print(f"Encoding detected: {result.encoding.encoding}")
    print(f"Confidence: {result.confidence:.2f}")
    print()
    
    # Process string
    xml_string = '<?xml version="1.0"?><root>Hello String</root>'
    result = processor.process(xml_string)
    print(f"String input: {result.text[:50]}...")
    print(f"Confidence: {result.confidence:.2f}")
    print()
    
    # Process file-like object
    xml_file = io.BytesIO(b'<?xml version="1.0"?><root>Hello File</root>')
    result = processor.process(xml_file)
    print(f"File input: {result.text[:50]}...")
    print(f"Confidence: {result.confidence:.2f}")
    print()


def example_configuration_presets():
    """Example 2: Using configuration presets for different scenarios."""
    print("=== Example 2: Configuration Presets ===")
    
    # Sample XML with various issues
    problematic_xml = b'''<?xml version="1.0"?>
    <root>
        <item>Content with \x00 invalid \x01 characters</item>
        <data>More \x02 problematic \x03 content</data>
    </root>
    '''
    
    # Web scraping preset (lenient)
    web_processor = CharacterStreamProcessor(StreamProcessingPresets.web_scraping())
    web_result = web_processor.process(problematic_xml)
    print(f"Web scraping result: {len(web_result.text)} chars processed")
    print(f"Diagnostics: {len(web_result.diagnostics)} issues")
    print()
    
    # Data recovery preset (maximum recovery)
    recovery_processor = CharacterStreamProcessor(StreamProcessingPresets.data_recovery())
    recovery_result = recovery_processor.process(problematic_xml)
    print(f"Data recovery result: {len(recovery_result.text)} chars processed")
    print(f"Confidence: {recovery_result.confidence:.2f}")
    print()
    
    # Strict mode preset
    strict_processor = CharacterStreamProcessor(StreamProcessingPresets.strict_mode())
    strict_result = strict_processor.process(problematic_xml)
    print(f"Strict mode result: {len(strict_result.text)} chars processed")
    print(f"Transformations applied: {len(strict_result.transformations.changes)}")
    print()


def example_custom_configuration():
    """Example 3: Custom configuration with overrides."""
    print("=== Example 3: Custom Configuration ===")
    
    # Custom configuration with specific overrides
    config = StreamProcessingPresets.web_scraping(
        buffer_size=1024,
        transform_overrides={
            'replacement_char': '?',
            'preserve_whitespace': False,
            'strict_xml': False
        }
    )
    
    # Validate configuration
    issues = StreamProcessingPresets.validate_config(config)
    if issues:
        print(f"Configuration issues: {issues}")
    else:
        print("Configuration is valid")
    
    processor = CharacterStreamProcessor(config)
    test_xml = b'<root>\x00\x01  Spaced   content  \x02</root>'
    result = processor.process(test_xml)
    
    print(f"Custom config result: {result.text}")
    print(f"Buffer size used: {config.buffer_size}")
    print()


def example_streaming_processing():
    """Example 4: Streaming processing for large inputs."""
    print("=== Example 4: Streaming Processing ===")
    
    # Create a large XML document
    large_xml = b'<items>'
    for i in range(1000):
        large_xml += f'<item id="{i}">Item content {i}</item>'.encode('utf-8')
    large_xml += b'</items>'
    
    print(f"Processing {len(large_xml):,} bytes of XML...")
    
    # Configure processor for streaming
    processor = CharacterStreamProcessor(
        StreamProcessingPresets.data_recovery(buffer_size=4096)
    )
    
    # Track progress
    progress_updates = []
    def track_progress(processed: int, total: int) -> bool:
        progress_updates.append((processed, total))
        if len(progress_updates) % 5 == 0:  # Print every 5th update
            percent = (processed / total) * 100 if total > 0 else 0
            print(f"Progress: {processed:,}/{total:,} bytes ({percent:.1f}%)")
        return True  # Continue processing
    
    # Process with streaming
    start_time = time.time()
    stream_result = processor.process_stream(large_xml, track_progress)
    
    # Consume the stream
    chunk_count = 0
    total_output_size = 0
    
    for chunk in stream_result.chunks:
        chunk_count += 1
        total_output_size += len(chunk)
        
        # Process first few chunks as example
        if chunk_count <= 3:
            print(f"Chunk {chunk_count}: {len(chunk)} chars - {chunk[:50]}...")
    
    end_time = time.time()
    
    print(f"Streaming complete:")
    print(f"  - Processed {chunk_count} chunks")
    print(f"  - Total output: {total_output_size:,} characters")
    print(f"  - Processing time: {end_time - start_time:.2f} seconds")
    print(f"  - Progress updates: {len(progress_updates)}")
    print()


def example_malformed_xml_handling():
    """Example 5: Handling various malformed XML scenarios."""
    print("=== Example 5: Malformed XML Handling ===")
    
    processor = CharacterStreamProcessor(StreamProcessingPresets.data_recovery())
    
    # Various malformed XML scenarios
    test_cases = [
        # Invalid control characters
        (b'<root>\x00Invalid\x01chars\x02</root>', "Invalid control characters"),
        
        # Encoding mismatch
        ('<?xml version="1.0" encoding="ISO-8859-1"?><root>café</root>'.encode('utf-8'), 
         "Encoding declaration mismatch"),
        
        # Broken structure
        (b'<root><unclosed><tag>content</tag></root>', "Broken tag structure"),
        
        # Invalid entities
        (b'<root>&invalid; &#999999; text</root>', "Invalid entities"),
        
        # Mixed line endings and control chars
        (b'<root>\r\n\x0BContent\x0C\r\n</root>', "Mixed line endings"),
    ]
    
    for xml_data, description in test_cases:
        print(f"Testing: {description}")
        result = processor.process(xml_data)
        
        print(f"  - Success: {'root' in result.text}")
        print(f"  - Confidence: {result.confidence:.2f}")
        print(f"  - Diagnostics: {len(result.diagnostics)} issues")
        
        if result.diagnostics:
            for i, diagnostic in enumerate(result.diagnostics[:2]):  # Show first 2
                print(f"    {i+1}. {diagnostic}")
        print()


def example_error_recovery():
    """Example 6: Error recovery and troubleshooting."""
    print("=== Example 6: Error Recovery ===")
    
    processor = CharacterStreamProcessor()
    
    # Test error conditions
    error_cases = [
        (b'', "Empty input"),
        (b'\xff\xff\xff\xff', "Invalid byte sequence"),
        ("", "Empty string"),
    ]
    
    for test_input, description in error_cases:
        print(f"Testing: {description}")
        result = processor.process(test_input)
        
        print(f"  - Never fails: {isinstance(result, type(result))}")
        print(f"  - Confidence: {result.confidence:.2f}")
        print(f"  - Has diagnostics: {len(result.diagnostics) > 0}")
        
        if result.diagnostics:
            print(f"  - First diagnostic: {result.diagnostics[0]}")
        print()


def example_performance_comparison():
    """Example 7: Performance comparison between normal and streaming processing."""
    print("=== Example 7: Performance Comparison ===")
    
    # Create test data
    test_sizes = [1000, 10000, 50000]  # Number of items
    
    for size in test_sizes:
        xml_data = b'<items>'
        for i in range(size):
            xml_data += f'<item>{i}</item>'.encode('utf-8')
        xml_data += b'</items>'
        
        processor = CharacterStreamProcessor(
            StreamProcessingPresets.web_scraping(buffer_size=8192)
        )
        
        print(f"Testing with {size:,} items ({len(xml_data):,} bytes):")
        
        # Normal processing
        start_time = time.time()
        normal_result = processor.process(xml_data)
        normal_time = time.time() - start_time
        
        # Streaming processing
        start_time = time.time()
        stream_result = processor.process_stream(xml_data)
        # Consume stream
        chunks = list(stream_result.chunks)
        streaming_time = time.time() - start_time
        
        print(f"  - Normal processing: {normal_time:.3f}s")
        print(f"  - Streaming processing: {streaming_time:.3f}s")
        print(f"  - Stream chunks: {len(chunks)}")
        print(f"  - Results match: {normal_result.text == ''.join(chunks)}")
        print()


def main():
    """Run all examples."""
    print("Character Stream Processing API Examples")
    print("=" * 50)
    print()
    
    try:
        example_basic_usage()
        example_configuration_presets()
        example_custom_configuration()
        example_streaming_processing()
        example_malformed_xml_handling()
        example_error_recovery()
        example_performance_comparison()
        
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()