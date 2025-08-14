#!/usr/bin/env python3
"""Core Parser API Progressive Disclosure Demo.

This example demonstrates the progressive disclosure API from simple functions
to advanced configuration, showcasing all levels of the API architecture.
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import ultra_robust_xml_parser as urxp


def demo_level_1_simple_api():
    """Demonstrate Level 1: Simple parsing functions."""
    print("=" * 60)
    print("LEVEL 1: Simple API Functions")
    print("=" * 60)
    
    # Simple string parsing
    print("\n1. Simple string parsing:")
    xml_content = '''
    <library>
        <book id="1" author="Jane Smith">
            <title>XML Processing Guide</title>
            <year>2023</year>
        </book>
        <book id="2" author="John Doe">
            <title>Advanced Parsing</title>
            <year>2024</year>
        </book>
    </library>
    '''
    
    result = urxp.parse_string(xml_content)
    print(f"Success: {result.success}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Elements: {result.tree.total_elements}")
    print(f"Root tag: {result.tree.root.tag}")
    
    # Find books
    books = result.tree.find_all('book')
    for book in books:
        title = book.find('title').text if book.find('title') else 'Unknown'
        author = book.get_attribute('author', 'Unknown')
        print(f"  - {title} by {author}")
    
    # Demonstrate malformed XML handling
    print("\n2. Malformed XML handling:")
    malformed_xml = '''
    <root>
        <unclosed_tag>content
        <missing_end>more content
        <valid>properly closed</valid>
    </root>
    '''
    
    result = urxp.parse_string(malformed_xml)
    print(f"Success: {result.success}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Repairs applied: {result.repair_count}")
    print(f"Elements recovered: {result.tree.total_elements}")
    
    # Demonstrate automatic input type detection
    print("\n3. Automatic input type detection:")
    
    # Bytes input
    bytes_xml = b'<?xml version="1.0" encoding="UTF-8"?><root><item>value</item></root>'
    result = urxp.parse(bytes_xml)
    print(f"Bytes input - Success: {result.success}, Elements: {result.tree.total_elements}")
    
    # String input
    string_xml = '<root><item>string value</item></root>'
    result = urxp.parse(string_xml)
    print(f"String input - Success: {result.success}, Elements: {result.tree.total_elements}")


def demo_level_2_advanced_api():
    """Demonstrate Level 2: Advanced configuration with UltraRobustXMLParser."""
    print("\n" + "=" * 60)
    print("LEVEL 2: Advanced Configuration API")
    print("=" * 60)
    
    # Different configuration strategies
    configs = {
        "conservative": urxp.TokenizationConfig.conservative(),
        "balanced": urxp.TokenizationConfig.balanced(),
        "aggressive": urxp.TokenizationConfig.aggressive(),
    }
    
    malformed_xml = '''
    <document>
        <section>
            <paragraph>Some text
            <bold>unclosed bold
            <italic>nested italic</italic>
            <title>Section Title
        </section>
        <footer>End of document
    '''
    
    print("\n1. Different configuration strategies:")
    for config_name, config in configs.items():
        parser = urxp.UltraRobustXMLParser(config=config)
        result = parser.parse(malformed_xml)
        
        print(f"\n{config_name.upper()} Configuration:")
        print(f"  Success: {result.success}")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Repairs: {result.repair_count}")
        print(f"  Elements: {result.tree.total_elements}")
        
        # Show parser statistics
        stats = parser.statistics
        print(f"  Parser stats: {stats['total_parses']} parses, "
              f"{stats['success_rate']:.1%} success rate")
    
    # Parser reuse demonstration
    print("\n2. Parser reuse for performance:")
    parser = urxp.UltraRobustXMLParser()
    
    xml_documents = [
        '<doc1><title>First Document</title></doc1>',
        '<doc2><title>Second Document</title></doc2>',
        '<doc3><title>Third Document</title></doc3>',
        '<malformed><unclosed>content</malformed>',
    ]
    
    for i, xml in enumerate(xml_documents, 1):
        result = parser.parse(xml)
        print(f"  Document {i}: Success={result.success}, "
              f"Confidence={result.confidence:.3f}")
    
    final_stats = parser.statistics
    print(f"\nFinal parser statistics:")
    print(f"  Total parses: {final_stats['total_parses']}")
    print(f"  Success rate: {final_stats['success_rate']:.1%}")
    print(f"  Average time: {final_stats['average_processing_time_ms']:.2f}ms")
    
    # Configuration override demonstration
    print("\n3. Per-parse configuration override:")
    parser = urxp.UltraRobustXMLParser(config=urxp.TokenizationConfig.balanced())
    
    # Parse with default config
    result1 = parser.parse('<root><item>default config</item></root>')
    print(f"  Default config - Confidence: {result1.confidence:.3f}")
    
    # Parse with conservative override
    conservative_config = urxp.TokenizationConfig.conservative()
    result2 = parser.parse('<root><item>conservative override</item></root>', 
                          config_override=conservative_config)
    print(f"  Conservative override - Confidence: {result2.confidence:.3f}")


def demo_progressive_result_api():
    """Demonstrate progressive result object API."""
    print("\n" + "=" * 60)
    print("Progressive Result Object API")
    print("=" * 60)
    
    xml_content = '''
    <catalog xmlns:book="http://example.com/book">
        <book:item id="1" category="fiction">
            <book:title>The Great Adventure</book:title>
            <book:author>Alice Johnson</book:author>
            <book:price currency="USD">29.99</book:price>
            <book:description>
                An exciting tale of discovery and friendship.
            </book:description>
        </book:item>
        <book:item id="2" category="non-fiction">
            <book:title>Learning XML Processing</book:title>
            <book:author>Bob Wilson</book:author>
            <book:price currency="EUR">34.50</book:price>
        </book:item>
    </catalog>
    '''
    
    result = urxp.parse_string(xml_content)
    
    print("\n1. Basic result access:")
    print(f"Success: {result.success}")
    print(f"Confidence: {result.confidence:.3f}")
    
    print("\n2. Tree navigation (.tree property):")
    print(f"Root element: {result.tree.root.tag}")
    print(f"Total elements: {result.tree.total_elements}")
    
    # Navigate the tree
    items = result.tree.find_all('item')
    print(f"Found {len(items)} book items:")
    
    for item in items:
        # Handle namespaced elements
        title_elem = None
        author_elem = None
        price_elem = None
        
        # Find child elements (handling namespace)
        for child in item.children:
            if child.local_name == 'title':
                title_elem = child
            elif child.local_name == 'author':
                author_elem = child
            elif child.local_name == 'price':
                price_elem = child
        
        title = title_elem.text if title_elem else 'Unknown'
        author = author_elem.text if author_elem else 'Unknown'
        price = price_elem.text if price_elem else '0'
        currency = price_elem.get_attribute('currency', 'USD') if price_elem else 'USD'
        category = item.get_attribute('category', 'unknown')
        
        print(f"  - {title} by {author} ({price} {currency}) [{category}]")
    
    print("\n3. Comprehensive metadata (.metadata property):")
    metadata = result.metadata
    
    print(f"Processing time: {metadata['processing_time_ms']:.2f}ms")
    print(f"Element count: {metadata['element_count']}")
    print(f"Has repairs: {metadata['has_repairs']}")
    
    # Confidence breakdown
    confidence_info = metadata['confidence_breakdown']
    print(f"Confidence breakdown:")
    for source, confidence in confidence_info.items():
        if confidence > 0:
            print(f"  {source}: {confidence:.3f}")
    
    # Parsing statistics
    parsing_stats = metadata['parsing_statistics']
    print(f"Parsing statistics:")
    print(f"  Characters processed: {parsing_stats['characters_processed']}")
    print(f"  Tokens generated: {parsing_stats['tokens_generated']}")
    print(f"  Memory used: {parsing_stats['memory_used_bytes']} bytes")


def demo_error_handling():
    """Demonstrate comprehensive error handling."""
    print("\n" + "=" * 60)
    print("Never-Fail Error Handling")
    print("=" * 60)
    
    test_cases = [
        ("Empty string", ""),
        ("Invalid XML", "<<>>invalid<<>>"),
        ("Severely malformed", "<a><b><c>no closing tags ever"),
        ("Mixed content", "Some text <tag>with xml</tag> and more text"),
        ("Invalid encoding", b"\xff\xfe\x00\x00invalid\x00\x00"),
        ("Binary garbage", b"\x89PNG\r\n\x1a\n\x00\x00\x00"),
    ]
    
    for name, test_input in test_cases:
        print(f"\n{name}:")
        try:
            result = urxp.parse(test_input)
            print(f"  Success: {result.success}")
            print(f"  Confidence: {result.confidence:.3f}")
            print(f"  Elements: {result.tree.total_elements}")
            print(f"  Diagnostics: {len(result.diagnostics)}")
            
            if result.diagnostics:
                print(f"  First diagnostic: {result.diagnostics[0].message[:60]}...")
                
        except Exception as e:
            print(f"  UNEXPECTED EXCEPTION: {e}")
            print("  (This should never happen with never-fail philosophy!)")


def main():
    """Run all demonstrations."""
    print("Ultra-Robust XML Parser - Core API Demo")
    print("Progressive Disclosure from Simple to Advanced")
    
    demo_level_1_simple_api()
    demo_level_2_advanced_api()
    demo_progressive_result_api()
    demo_error_handling()
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()