#!/usr/bin/env python3
"""
Comprehensive demonstration of XML tree finalization and validation capabilities.

This example showcases all major features of the ultra-robust XML parser's
tree finalization system including validation, optimization, and output formatting.
"""

import sys
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ultra_robust_xml_parser.tree.builder import XMLDocument, XMLElement, ParseResult
from ultra_robust_xml_parser.tree.validation import (
    TreeValidator, ValidationLevel, TreeOptimizer,
    OutputFormatter, OutputFormat, OutputConfiguration
)
from ultra_robust_xml_parser.shared import DiagnosticSeverity


def create_sample_document() -> XMLDocument:
    """Create a sample XML document for demonstration."""
    print("📄 Creating sample XML document...")
    
    # Create document metadata
    metadata = XMLElement(
        tag="metadata",
        attributes={
            "created": "2024-01-15",
            "author": "XML Demo",
            "version": "1.0"
        }
    )
    
    # Create product catalog
    products = []
    for i in range(1, 4):
        product = XMLElement(
            tag="product",
            attributes={
                "id": str(i),
                "category": "electronics" if i % 2 == 1 else "books"
            }
        )
        
        # Add product details
        name = XMLElement(tag="name", text=f"Product {i}")
        price = XMLElement(tag="price", text=f"{19.99 + i * 10:.2f}")
        description = XMLElement(
            tag="description", 
            text=f"  This   is   product   {i}   with   extra   whitespace  "
        )
        
        product.children = [name, price, description]
        # Set parent relationships
        for child in product.children:
            child.parent = product
            
        products.append(product)
    
    # Create empty elements for optimization demo
    empty1 = XMLElement(tag="empty_element")  # Will be removed by optimizer
    empty2 = XMLElement(tag="another_empty")  # Will be removed by optimizer
    
    # Create catalog container
    catalog = XMLElement(tag="catalog", children=products + [empty1, empty2])
    for child in catalog.children:
        child.parent = catalog
    
    # Create root document
    root = XMLElement(
        tag="store",
        attributes={
            "xmlns": "http://example.com/store",
            "empty_attr": "",  # Will be optimized away
            "valid_attr": "keep_this"
        },
        children=[metadata, catalog]
    )
    
    # Set parent relationships
    for child in root.children:
        child.parent = root
    
    # Create document
    document = XMLDocument(
        root=root,
        version="1.0",
        encoding="utf-8",
        standalone=True
    )
    
    print(f"✅ Created document with {document.total_elements} elements")
    return document


def demonstrate_tree_validation(document: XMLDocument) -> None:
    """Demonstrate tree validation capabilities."""
    print("\n🔍 TREE VALIDATION DEMONSTRATION")
    print("=" * 50)
    
    # Test different validation levels
    validation_levels = [
        (ValidationLevel.MINIMAL, "Basic structural integrity"),
        (ValidationLevel.STANDARD, "XML well-formedness"),
        (ValidationLevel.STRICT, "Strict XML compliance with namespaces"),
        (ValidationLevel.PEDANTIC, "Pedantic validation (all rules)")
    ]
    
    for level, description in validation_levels:
        print(f"\n📋 Validating with {level.name} level ({description}):")
        
        validator = TreeValidator(validation_level=level, correlation_id="demo-validation")
        result = validator.validate(document)
        
        print(f"  ✅ Success: {result.success}")
        print(f"  📊 Confidence: {result.confidence:.3f}")
        print(f"  🔢 Elements validated: {result.elements_validated}")
        print(f"  🏷️  Attributes validated: {result.attributes_validated}")
        print(f"  ⏱️  Processing time: {result.processing_time_ms:.2f}ms")
        print(f"  ⚠️  Issues found: {len(result.issues)}")
        
        if result.issues:
            print("  📝 Issues:")
            for i, issue in enumerate(result.issues[:3], 1):  # Show first 3 issues
                print(f"    {i}. {issue.severity.name}: {issue.message}")
                if issue.suggested_fix:
                    print(f"       💡 Suggestion: {issue.suggested_fix}")
            if len(result.issues) > 3:
                print(f"    ... and {len(result.issues) - 3} more issues")


def demonstrate_tree_optimization(document: XMLDocument) -> None:
    """Demonstrate tree optimization capabilities."""
    print("\n⚡ TREE OPTIMIZATION DEMONSTRATION")
    print("=" * 50)
    
    # Count elements before optimization
    elements_before = document.total_elements
    print(f"📊 Elements before optimization: {elements_before}")
    
    # Create optimizer and optimize
    optimizer = TreeOptimizer(correlation_id="demo-optimization")
    result = optimizer.optimize(document)
    
    elements_after = document.total_elements
    print(f"📊 Elements after optimization: {elements_after}")
    
    print(f"\n✅ Optimization Success: {result.success}")
    print(f"📊 Confidence: {result.confidence:.3f}")
    print(f"🔢 Total actions performed: {result.total_actions}")
    print(f"📝 Elements affected: {result.elements_affected}")
    print(f"💾 Memory saved: {result.total_memory_saved_bytes} bytes")
    print(f"⏱️ Processing time: {result.processing_time_ms:.2f}ms")
    
    if result.actions:
        print("\n🔧 Optimization Actions Performed:")
        for action in result.actions:
            print(f"  • {action.optimization_type.value}: {action.description}")
            if action.elements_affected > 0:
                print(f"    - Elements affected: {action.elements_affected}")
            if action.memory_saved_bytes > 0:
                print(f"    - Memory saved: {action.memory_saved_bytes} bytes")


def demonstrate_output_formatting(document: XMLDocument) -> None:
    """Demonstrate multiple output format capabilities."""
    print("\n🔄 OUTPUT FORMATTING DEMONSTRATION")
    print("=" * 50)
    
    formatter = OutputFormatter(correlation_id="demo-formatting")
    
    # Demonstrate different output formats
    formats = [
        (OutputFormat.XML_STRING, "Standard XML"),
        (OutputFormat.XML_PRETTY, "Pretty-printed XML"),
        (OutputFormat.XML_MINIFIED, "Minified XML"),
        (OutputFormat.JSON, "JSON format"),
        (OutputFormat.JSON_PRETTY, "Pretty JSON"),
        (OutputFormat.DICTIONARY, "Dictionary format")
    ]
    
    for output_format, description in formats:
        print(f"\n📋 {description} ({output_format.value}):")
        
        # Create specific configuration for each format
        if output_format == OutputFormat.XML_PRETTY:
            config = OutputConfiguration(xml_indent="  ")
        elif output_format == OutputFormat.XML_MINIFIED:
            config = OutputConfiguration(exclude_empty_attributes=True)
        elif output_format in (OutputFormat.JSON, OutputFormat.JSON_PRETTY):
            config = OutputConfiguration(include_metadata=True)
        else:
            config = OutputConfiguration()
        
        result = formatter.format(document, output_format, config)
        
        print(f"  ✅ Success: {result.success}")
        print(f"  📏 Output size: {result.output_size_bytes} bytes")
        print(f"  ⏱️  Processing time: {result.processing_time_ms:.2f}ms")
        
        if result.issues:
            print(f"  ⚠️  Issues: {len(result.issues)}")
            for issue in result.issues:
                print(f"    - {issue}")
        
        # Show sample of output (truncated for readability)
        output_sample = result.formatted_output
        if len(output_sample) > 200:
            output_sample = output_sample[:200] + "..."
        
        print(f"  📄 Output sample:")
        for line in output_sample.split('\n')[:5]:  # First 5 lines
            print(f"    {line}")
        if output_sample.count('\n') > 5:
            print("    ...")


def demonstrate_comprehensive_parseresult(document: XMLDocument) -> None:
    """Demonstrate enhanced ParseResult with comprehensive metadata."""
    print("\n📊 COMPREHENSIVE PARSERESULT DEMONSTRATION")
    print("=" * 50)
    
    # Create a ParseResult with validation and optimization
    validator = TreeValidator(ValidationLevel.STANDARD)
    optimizer = TreeOptimizer()
    
    validation_result = validator.validate(document)
    optimization_result = optimizer.optimize(document)
    
    # Create comprehensive ParseResult
    parse_result = ParseResult(
        document=document,
        success=True,
        confidence=0.95,
        validation_result=validation_result,
        optimization_result=optimization_result
    )
    
    # Add some diagnostics for demonstration
    parse_result.add_diagnostic(DiagnosticSeverity.INFO, "Document processed successfully", "demo")
    parse_result.add_diagnostic(DiagnosticSeverity.WARNING, "Minor formatting issue detected", "demo")
    
    # Get comprehensive summary
    summary = parse_result.summary()
    
    print("📋 Parse Result Summary:")
    print(f"  ✅ Success: {summary['success']}")
    print(f"  📊 Confidence: {summary['confidence']:.3f}")
    print(f"  📄 Document well-formed: {summary['document_well_formed']}")
    
    print("\n📈 Parsing Statistics:")
    stats = summary['parsing_statistics']
    print(f"  🔢 Elements: {stats['element_count']}")
    print(f"  🏷️  Attributes: {stats['attribute_count']}")
    print(f"  📏 Max depth: {stats['max_depth']}")
    print(f"  🔧 Repairs: {stats['repair_count']}")
    print(f"  ⏱️  Processing time: {stats['processing_time_ms']:.2f}ms")
    
    print("\n🔍 Validation Summary:")
    validation = summary['validation']
    if validation['validation_performed']:
        print(f"  ✅ Success: {validation['validation_success']}")
        print(f"  📊 Confidence: {validation['validation_confidence']:.3f}")
        print(f"  ⚠️  Errors: {validation['validation_error_count']}")
        print(f"  ⚠️  Warnings: {validation['validation_warning_count']}")
        print(f"  📋 Level: {validation['validation_level']}")
    else:
        print("  ❌ No validation performed")
    
    print("\n⚡ Optimization Summary:")
    optimization = summary['optimization']
    if optimization['optimization_performed']:
        print(f"  ✅ Success: {optimization['optimization_success']}")
        print(f"  📊 Confidence: {optimization['optimization_confidence']:.3f}")
        print(f"  🔧 Actions: {optimization['total_optimizations']}")
        print(f"  📝 Elements affected: {optimization['elements_affected']}")
        print(f"  💾 Memory saved: {optimization['memory_saved_bytes']} bytes")
    else:
        print("  ❌ No optimization performed")
    
    print("\n🎯 Confidence Breakdown:")
    confidence_breakdown = summary['confidence_breakdown']
    for key, value in confidence_breakdown.items():
        if isinstance(value, float):
            print(f"  {key.replace('_', ' ').title()}: {value:.3f}")
    
    print("\n📝 Diagnostics Summary:")
    diagnostics = summary['diagnostics_summary']
    print(f"  📋 Total: {diagnostics['total_diagnostics']}")
    print(f"  ⚠️  Has errors: {diagnostics['has_errors']}")
    if diagnostics['diagnostics_by_severity']:
        print("  📊 By severity:")
        for severity, count in diagnostics['diagnostics_by_severity'].items():
            print(f"    {severity}: {count}")


def demonstrate_tree_navigation(document: XMLDocument) -> None:
    """Demonstrate tree navigation and content extraction."""
    print("\n🧭 TREE NAVIGATION DEMONSTRATION")
    print("=" * 50)
    
    root = document.root
    if not root:
        print("❌ No root element found")
        return
    
    print(f"📋 Root element: <{root.tag}>")
    print(f"🏷️  Root attributes: {root.attributes}")
    
    # Find all products
    print(f"\n🔍 Finding all products...")
    products = root.find_all("product")
    print(f"📋 Found {len(products)} products:")
    
    for product in products:
        product_id = product.attributes.get("id", "unknown")
        category = product.attributes.get("category", "unknown")
        
        # Navigate to product details
        name_elem = product.find("name")
        price_elem = product.find("price")
        
        name = name_elem.text if name_elem else "No name"
        price = price_elem.text if price_elem else "No price"
        
        print(f"  📦 Product {product_id} ({category}): {name} - ${price}")
    
    # Demonstrate path navigation
    print(f"\n🛤️  Path Navigation Examples:")
    
    # Find element by path
    metadata = root.find("metadata")
    if metadata:
        print(f"  📋 Metadata path: {metadata.get_path()}")
        print(f"  🏷️  Metadata attributes: {metadata.attributes}")
    
    # Find nested elements
    catalog = root.find("catalog")
    if catalog:
        print(f"  📦 Catalog contains {len(catalog.children)} items")
        print(f"  📏 Catalog depth: {catalog.get_depth()}")
    
    # Demonstrate tree iteration
    print(f"\n🔄 Tree Iteration:")
    element_count = 0
    text_content_count = 0
    
    for element in document.iter_elements():
        element_count += 1
        if element.text and element.text.strip():
            text_content_count += 1
    
    print(f"  🔢 Total elements: {element_count}")
    print(f"  📝 Elements with text: {text_content_count}")


def main():
    """Main demonstration function."""
    print("🚀 ULTRA-ROBUST XML PARSER - TREE FINALIZATION DEMO")
    print("=" * 60)
    
    try:
        # Create sample document
        document = create_sample_document()
        
        # Demonstrate all capabilities
        demonstrate_tree_validation(document)
        demonstrate_tree_optimization(document)
        demonstrate_output_formatting(document)
        demonstrate_comprehensive_parseresult(document)
        demonstrate_tree_navigation(document)
        
        print(f"\n🎉 DEMONSTRATION COMPLETE!")
        print("✅ All tree finalization features demonstrated successfully")
        
    except Exception as e:
        print(f"\n❌ DEMONSTRATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())