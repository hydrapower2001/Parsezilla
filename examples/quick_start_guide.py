#!/usr/bin/env python3
"""
Quick Start Guide for the Ultra-Robust XML Parser Tree Finalization Features.

This example provides a simple introduction to the key features of the
tree finalization and validation system.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ultra_robust_xml_parser.tree.builder import XMLDocument, XMLElement
from ultra_robust_xml_parser.tree.validation import (
    TreeValidator, ValidationLevel, TreeOptimizer,
    OutputFormatter, OutputFormat, OutputConfiguration
)


def quick_start_example():
    """Quick start example showing basic usage."""
    
    print("🚀 QUICK START - Ultra-Robust XML Parser")
    print("=" * 45)
    
    # Step 1: Create a simple XML document
    print("\n📄 Step 1: Creating XML Document")
    print("-" * 30)
    
    # Create elements
    title = XMLElement(tag="title", text="My Book")
    author = XMLElement(tag="author", text="John Doe")
    price = XMLElement(tag="price", attributes={"currency": "USD"}, text="19.99")
    
    # Create book element
    book = XMLElement(
        tag="book", 
        attributes={"id": "123", "genre": "fiction"},
        children=[title, author, price]
    )
    
    # Set parent relationships
    for child in book.children:
        child.parent = book
    
    # Create document
    document = XMLDocument(root=book, version="1.0", encoding="utf-8")
    
    print(f"✅ Created document with {document.total_elements} elements")
    print(f"📏 Document depth: {document.max_depth}")
    
    
    # Step 2: Validate the document
    print("\n🔍 Step 2: Document Validation")
    print("-" * 30)
    
    validator = TreeValidator(ValidationLevel.STANDARD)
    validation_result = validator.validate(document)
    
    print(f"✅ Validation success: {validation_result.success}")
    print(f"📊 Confidence: {validation_result.confidence:.3f}")
    print(f"⚠️  Issues found: {len(validation_result.issues)}")
    
    if validation_result.issues:
        for issue in validation_result.issues[:2]:  # Show first 2 issues
            print(f"  - {issue.severity.name}: {issue.message}")
    
    
    # Step 3: Optimize the document
    print("\n⚡ Step 3: Document Optimization")
    print("-" * 30)
    
    optimizer = TreeOptimizer()
    optimization_result = optimizer.optimize(document)
    
    print(f"✅ Optimization success: {optimization_result.success}")
    print(f"🔧 Actions performed: {optimization_result.total_actions}")
    print(f"💾 Memory saved: {optimization_result.total_memory_saved_bytes} bytes")
    
    
    # Step 4: Convert to different formats
    print("\n🔄 Step 4: Output Formatting")
    print("-" * 30)
    
    formatter = OutputFormatter()
    
    # XML output
    xml_result = formatter.format(document, OutputFormat.XML_PRETTY)
    print("📋 Pretty XML:")
    print(xml_result.formatted_output)
    
    # JSON output
    json_result = formatter.format(document, OutputFormat.JSON_PRETTY)
    print("\n📋 JSON Format:")
    print(json_result.formatted_output)
    
    
    # Step 5: Navigate the document
    print("\n🧭 Step 5: Document Navigation")
    print("-" * 30)
    
    # Find elements
    title_elem = document.root.find("title")
    author_elem = document.root.find("author")
    price_elem = document.root.find("price")
    
    if title_elem and author_elem and price_elem:
        book_title = title_elem.text
        book_author = author_elem.text
        book_price = price_elem.text
        currency = price_elem.attributes.get("currency", "")
        
        print(f"📖 Book: '{book_title}' by {book_author}")
        print(f"💰 Price: {currency} {book_price}")
    
    # Get document statistics
    print(f"\n📊 Document Statistics:")
    print(f"  Elements: {document.total_elements}")
    print(f"  Attributes: {document.total_attributes}")
    print(f"  Max depth: {document.max_depth}")
    
    print(f"\n🎉 Quick start complete!")


def validation_levels_example():
    """Example showing different validation levels."""
    
    print("\n\n🔍 VALIDATION LEVELS EXAMPLE")
    print("=" * 40)
    
    # Create document with potential issues
    problem_element = XMLElement(
        tag="ns:element",  # Namespace without declaration
        attributes={"empty": ""},  # Empty attribute
        text="  lots   of   whitespace  "
    )
    
    document = XMLDocument(root=problem_element)
    
    levels = [
        (ValidationLevel.MINIMAL, "Basic checks only"),
        (ValidationLevel.STANDARD, "XML well-formedness"),
        (ValidationLevel.STRICT, "Full XML compliance"),
        (ValidationLevel.PEDANTIC, "Strictest validation")
    ]
    
    for level, description in levels:
        print(f"\n📋 {level.name} ({description}):")
        
        validator = TreeValidator(level)
        result = validator.validate(document)
        
        print(f"  Success: {result.success}")
        print(f"  Issues: {len(result.issues)}")
        print(f"  Confidence: {result.confidence:.3f}")


def output_formats_example():
    """Example showing different output formats."""
    
    print("\n\n🔄 OUTPUT FORMATS EXAMPLE")
    print("=" * 35)
    
    # Create sample document
    item = XMLElement(tag="item", attributes={"id": "1"}, text="Sample text")
    root = XMLElement(tag="items", children=[item])
    item.parent = root
    
    document = XMLDocument(root=root)
    formatter = OutputFormatter()
    
    formats = [
        (OutputFormat.XML_STRING, "Standard XML"),
        (OutputFormat.XML_PRETTY, "Pretty XML"),
        (OutputFormat.XML_MINIFIED, "Minified XML"),
        (OutputFormat.JSON, "JSON"),
        (OutputFormat.DICTIONARY, "Dictionary")
    ]
    
    for format_type, description in formats:
        print(f"\n📋 {description}:")
        result = formatter.format(document, format_type)
        
        if result.success:
            output = result.formatted_output
            if len(output) > 100:
                output = output[:100] + "..."
            print(f"  {output}")
        else:
            print(f"  ❌ Failed: {result.issues}")


def main():
    """Main function."""
    try:
        quick_start_example()
        validation_levels_example()
        output_formats_example()
        
        print(f"\n✅ All examples completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())