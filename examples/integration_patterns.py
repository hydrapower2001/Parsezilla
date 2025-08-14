#!/usr/bin/env python3
"""
Integration patterns and practical usage examples for the ultra-robust XML parser.

This example demonstrates how to integrate the parser with existing workflows,
handle various XML scenarios, and extract meaningful data from documents.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ultra_robust_xml_parser.tree.builder import XMLDocument, XMLElement
from ultra_robust_xml_parser.tree.validation import (
    TreeValidator, ValidationLevel, TreeOptimizer,
    OutputFormatter, OutputFormat, OutputConfiguration
)


class XMLDataExtractor:
    """Example data extraction class using the ultra-robust parser."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validator = TreeValidator(validation_level)
        self.optimizer = TreeOptimizer()
        self.formatter = OutputFormatter()
    
    def extract_product_data(self, document: XMLDocument) -> List[Dict[str, Any]]:
        """Extract product information from XML document."""
        print("📦 Extracting product data...")
        
        # Validate document first
        validation_result = self.validator.validate(document)
        if not validation_result.success:
            print(f"⚠️  Document validation issues: {len(validation_result.issues)} issues found")
            # Continue anyway - this is ultra-robust parsing!
        
        products = []
        if document.root:
            product_elements = document.root.find_all("product")
            
            for product_elem in product_elements:
                product_data = {
                    "id": product_elem.attributes.get("id"),
                    "category": product_elem.attributes.get("category"),
                }
                
                # Extract nested data safely
                name_elem = product_elem.find("name")
                if name_elem and name_elem.text:
                    product_data["name"] = name_elem.text.strip()
                
                price_elem = product_elem.find("price")
                if price_elem and price_elem.text:
                    try:
                        product_data["price"] = float(price_elem.text.strip().replace("$", ""))
                    except ValueError:
                        product_data["price"] = None
                
                description_elem = product_elem.find("description")
                if description_elem and description_elem.text:
                    product_data["description"] = description_elem.text.strip()
                
                products.append(product_data)
        
        print(f"✅ Extracted {len(products)} products")
        return products
    
    def convert_to_json_api_format(self, document: XMLDocument) -> str:
        """Convert XML document to JSON API format."""
        print("🔄 Converting to JSON API format...")
        
        # Configure output for API use
        config = OutputConfiguration(
            include_metadata=False,
            exclude_empty_elements=True,
            exclude_empty_attributes=True,
            dict_attribute_prefix="",  # No prefix for cleaner JSON
            dict_text_key="value"
        )
        
        result = self.formatter.format(document, OutputFormat.JSON_COMPACT, config)
        
        if result.success:
            print(f"✅ JSON conversion successful ({result.output_size_bytes} bytes)")
            return result.formatted_output
        else:
            print(f"❌ JSON conversion failed: {result.issues}")
            return "{}"
    
    def optimize_for_storage(self, document: XMLDocument) -> XMLDocument:
        """Optimize document for efficient storage."""
        print("💾 Optimizing document for storage...")
        
        # Apply optimizations
        optimization_result = self.optimizer.optimize(document)
        
        if optimization_result.success:
            print(f"✅ Optimization successful:")
            print(f"  🔧 Actions performed: {optimization_result.total_actions}")
            print(f"  💾 Memory saved: {optimization_result.total_memory_saved_bytes} bytes")
            print(f"  📝 Elements affected: {optimization_result.elements_affected}")
        else:
            print("⚠️  Optimization had issues but document is still usable")
        
        return document


def create_real_world_xml_samples() -> Dict[str, XMLDocument]:
    """Create various real-world XML document samples."""
    print("📄 Creating real-world XML samples...")
    
    samples = {}
    
    # 1. E-commerce product catalog
    samples["ecommerce"] = create_ecommerce_sample()
    
    # 2. Configuration file
    samples["config"] = create_config_sample()
    
    # 3. RSS feed
    samples["rss"] = create_rss_sample()
    
    # 4. Malformed XML (for robustness testing)
    samples["malformed"] = create_malformed_sample()
    
    print(f"✅ Created {len(samples)} sample documents")
    return samples


def create_ecommerce_sample() -> XMLDocument:
    """Create e-commerce product catalog XML."""
    categories = XMLElement(tag="categories")
    
    # Electronics category
    electronics = XMLElement(tag="category", attributes={"id": "electronics", "name": "Electronics"})
    
    laptop = XMLElement(tag="product", attributes={"id": "1", "sku": "LAP001"})
    laptop.children = [
        XMLElement(tag="name", text="Gaming Laptop"),
        XMLElement(tag="price", text="1299.99"),
        XMLElement(tag="currency", text="USD"),
        XMLElement(tag="stock", text="15"),
        XMLElement(tag="description", text="High-performance gaming laptop with RGB lighting"),
    ]
    for child in laptop.children:
        child.parent = laptop
    
    phone = XMLElement(tag="product", attributes={"id": "2", "sku": "PHN001"})
    phone.children = [
        XMLElement(tag="name", text="Smartphone"),
        XMLElement(tag="price", text="699.99"),
        XMLElement(tag="currency", text="USD"),
        XMLElement(tag="stock", text="0"),  # Out of stock
        XMLElement(tag="description", text="Latest smartphone with advanced camera"),
    ]
    for child in phone.children:
        child.parent = phone
    
    electronics.children = [laptop, phone]
    for child in electronics.children:
        child.parent = electronics
    
    categories.children = [electronics]
    electronics.parent = categories
    
    # Root catalog
    root = XMLElement(tag="catalog", attributes={
        "version": "2.0",
        "generated": "2024-01-15T10:30:00Z",
        "xmlns": "http://example.com/catalog"
    })
    root.children = [categories]
    categories.parent = root
    
    return XMLDocument(root=root, version="1.0", encoding="utf-8")


def create_config_sample() -> XMLDocument:
    """Create application configuration XML."""
    # Database configuration
    database = XMLElement(tag="database")
    database.children = [
        XMLElement(tag="host", text="localhost"),
        XMLElement(tag="port", text="5432"),
        XMLElement(tag="name", text="myapp"),
        XMLElement(tag="user", text="dbuser"),
        XMLElement(tag="ssl", text="true"),
    ]
    for child in database.children:
        child.parent = database
    
    # Logging configuration
    logging = XMLElement(tag="logging")
    logging.children = [
        XMLElement(tag="level", text="INFO"),
        XMLElement(tag="file", text="/var/log/myapp.log"),
        XMLElement(tag="max_size", text="100MB"),
        XMLElement(tag="rotate", text="daily"),
    ]
    for child in logging.children:
        child.parent = logging
    
    # Features configuration
    features = XMLElement(tag="features")
    feature1 = XMLElement(tag="feature", attributes={"name": "analytics", "enabled": "true"})
    feature2 = XMLElement(tag="feature", attributes={"name": "beta_ui", "enabled": "false"})
    features.children = [feature1, feature2]
    for child in features.children:
        child.parent = features
    
    # Root configuration
    root = XMLElement(tag="configuration", attributes={"version": "1.0"})
    root.children = [database, logging, features]
    for child in root.children:
        child.parent = root
    
    return XMLDocument(root=root, version="1.0", encoding="utf-8")


def create_rss_sample() -> XMLDocument:
    """Create RSS feed XML."""
    # Channel info
    channel = XMLElement(tag="channel")
    channel.children = [
        XMLElement(tag="title", text="Tech News Feed"),
        XMLElement(tag="link", text="https://example.com/news"),
        XMLElement(tag="description", text="Latest technology news and updates"),
        XMLElement(tag="language", text="en-US"),
    ]
    
    # News items
    item1 = XMLElement(tag="item")
    item1.children = [
        XMLElement(tag="title", text="AI Breakthrough in Natural Language Processing"),
        XMLElement(tag="link", text="https://example.com/news/ai-breakthrough"),
        XMLElement(tag="description", text="Researchers achieve new milestone in NLP"),
        XMLElement(tag="pubDate", text="Mon, 15 Jan 2024 10:00:00 GMT"),
        XMLElement(tag="guid", text="ai-breakthrough-2024-01-15"),
    ]
    
    item2 = XMLElement(tag="item")
    item2.children = [
        XMLElement(tag="title", text="New Programming Language Released"),
        XMLElement(tag="link", text="https://example.com/news/new-language"),
        XMLElement(tag="description", text="Developer community excited about new language features"),
        XMLElement(tag="pubDate", text="Sun, 14 Jan 2024 15:30:00 GMT"),
        XMLElement(tag="guid", text="new-language-2024-01-14"),
    ]
    
    # Set up parent relationships
    for item in [item1, item2]:
        for child in item.children:
            child.parent = item
    
    channel.children.extend([item1, item2])
    for child in channel.children:
        child.parent = channel
    
    # Root RSS
    root = XMLElement(tag="rss", attributes={"version": "2.0"})
    root.children = [channel]
    channel.parent = root
    
    return XMLDocument(root=root, version="1.0", encoding="utf-8")


def create_malformed_sample() -> XMLDocument:
    """Create intentionally malformed XML for robustness testing."""
    # Create elements that might have issues
    root = XMLElement(tag="document")
    
    # Element with problematic content
    problematic = XMLElement(tag="problematic", text="Content with <unescaped> & characters")
    
    # Element with empty attributes
    with_empty_attrs = XMLElement(
        tag="element",
        attributes={"empty": "", "valid": "value", "also_empty": ""}
    )
    
    # Deeply nested structure
    current = XMLElement(tag="level1")
    for i in range(2, 10):  # Create deep nesting
        child = XMLElement(tag=f"level{i}")
        current.children = [child]
        child.parent = current
        current = child
    
    # Add some redundant empty elements
    empty1 = XMLElement(tag="empty1")
    empty2 = XMLElement(tag="empty2")
    empty3 = XMLElement(tag="empty3")
    
    # Build document structure
    root.children = [problematic, with_empty_attrs, root.children[0] if root.children else current, empty1, empty2, empty3]
    if not root.children:
        root.children = [problematic, with_empty_attrs, current, empty1, empty2, empty3]
    
    for child in root.children:
        child.parent = root
    
    return XMLDocument(root=root, version="1.0", encoding="utf-8")


def demonstrate_data_extraction_pattern(samples: Dict[str, XMLDocument]) -> None:
    """Demonstrate practical data extraction patterns."""
    print("\n📊 DATA EXTRACTION PATTERNS")
    print("=" * 50)
    
    extractor = XMLDataExtractor()
    
    # Extract e-commerce data
    print("\n🛒 E-commerce Data Extraction:")
    ecommerce_doc = samples["ecommerce"]
    products = extractor.extract_product_data(ecommerce_doc)
    
    for product in products:
        print(f"  📦 {product.get('name', 'Unknown')} (ID: {product.get('id', 'N/A')})")
        print(f"    💰 Price: ${product.get('price', 'N/A')}")
        print(f"    📋 Category: {product.get('category', 'N/A')}")
    
    # Extract configuration data
    print("\n⚙️  Configuration Data Extraction:")
    config_doc = samples["config"]
    if config_doc.root:
        db_config = config_doc.root.find("database")
        if db_config:
            print("  🗄️  Database Configuration:")
            for child in db_config.children:
                print(f"    {child.tag}: {child.text}")
        
        features = config_doc.root.find_all("feature")
        if features:
            print("  🔧 Features:")
            for feature in features:
                name = feature.attributes.get("name", "unknown")
                enabled = feature.attributes.get("enabled", "false")
                status = "✅ enabled" if enabled.lower() == "true" else "❌ disabled"
                print(f"    {name}: {status}")


def demonstrate_format_conversion_pattern(samples: Dict[str, XMLDocument]) -> None:
    """Demonstrate format conversion patterns."""
    print("\n🔄 FORMAT CONVERSION PATTERNS")
    print("=" * 50)
    
    extractor = XMLDataExtractor()
    
    for name, document in samples.items():
        print(f"\n📄 Converting {name} sample:")
        
        # Convert to JSON API format
        json_output = extractor.convert_to_json_api_format(document)
        json_size = len(json_output.encode('utf-8'))
        print(f"  📏 JSON size: {json_size} bytes")
        
        # Show sample of JSON output
        try:
            json_data = json.loads(json_output)
            json_pretty = json.dumps(json_data, indent=2)[:300] + "..."
            print(f"  📋 JSON sample:")
            for line in json_pretty.split('\n')[:8]:
                print(f"    {line}")
        except json.JSONDecodeError:
            print(f"  ⚠️  JSON output is not valid JSON")


def demonstrate_robustness_pattern(samples: Dict[str, XMLDocument]) -> None:
    """Demonstrate robustness handling patterns."""
    print("\n🛡️  ROBUSTNESS PATTERNS")
    print("=" * 50)
    
    malformed_doc = samples["malformed"]
    extractor = XMLDataExtractor(ValidationLevel.STRICT)
    
    print("🔍 Processing intentionally malformed document with strict validation:")
    
    # Validate with strict level
    validation_result = extractor.validator.validate(malformed_doc)
    print(f"  ✅ Validation completed (success: {validation_result.success})")
    print(f"  📊 Confidence: {validation_result.confidence:.3f}")
    print(f"  ⚠️  Issues found: {len(validation_result.issues)}")
    
    if validation_result.issues:
        print("  🔍 Top validation issues:")
        for i, issue in enumerate(validation_result.issues[:3], 1):
            print(f"    {i}. {issue.severity.name}: {issue.message}")
            if issue.suggested_fix:
                print(f"       💡 Fix: {issue.suggested_fix}")
    
    # Optimize despite issues
    print("\n⚡ Optimizing malformed document:")
    optimized_doc = extractor.optimize_for_storage(malformed_doc)
    
    # Try to extract data anyway
    print("\n📊 Attempting data extraction despite issues:")
    try:
        if optimized_doc.root:
            elements_found = len(list(optimized_doc.iter_elements()))
            print(f"  ✅ Successfully processed {elements_found} elements")
            
            # Try to find any meaningful content
            text_elements = [elem for elem in optimized_doc.iter_elements() if elem.text and elem.text.strip()]
            print(f"  📝 Found {len(text_elements)} elements with text content")
            
            if text_elements:
                print("  📄 Sample text content:")
                for elem in text_elements[:3]:
                    content = elem.text.strip()[:50] + "..." if len(elem.text.strip()) > 50 else elem.text.strip()
                    print(f"    {elem.tag}: {content}")
        else:
            print("  ❌ No root element found")
    except Exception as e:
        print(f"  ❌ Extraction failed: {e}")
        # Even with failures, we can often get partial data
        print("  🛡️  Ultra-robust parser continues to provide what it can")


def demonstrate_performance_pattern(samples: Dict[str, XMLDocument]) -> None:
    """Demonstrate performance optimization patterns."""
    print("\n⚡ PERFORMANCE OPTIMIZATION PATTERNS")
    print("=" * 50)
    
    extractor = XMLDataExtractor()
    
    for name, document in samples.items():
        print(f"\n📄 Optimizing {name} sample:")
        
        # Measure before optimization
        elements_before = document.total_elements
        
        # Apply optimizations
        optimization_result = extractor.optimizer.optimize(document)
        
        elements_after = document.total_elements
        
        print(f"  📊 Elements: {elements_before} → {elements_after}")
        print(f"  🔧 Actions: {optimization_result.total_actions}")
        print(f"  💾 Memory saved: {optimization_result.total_memory_saved_bytes} bytes")
        print(f"  ⏱️  Time: {optimization_result.processing_time_ms:.2f}ms")
        print(f"  📈 Confidence: {optimization_result.confidence:.3f}")


def main():
    """Main demonstration function."""
    print("🚀 ULTRA-ROBUST XML PARSER - INTEGRATION PATTERNS")
    print("=" * 60)
    
    try:
        # Create sample documents
        samples = create_real_world_xml_samples()
        
        # Demonstrate various integration patterns
        demonstrate_data_extraction_pattern(samples)
        demonstrate_format_conversion_pattern(samples)
        demonstrate_robustness_pattern(samples)
        demonstrate_performance_pattern(samples)
        
        print(f"\n🎉 INTEGRATION PATTERNS DEMONSTRATION COMPLETE!")
        print("✅ All integration patterns demonstrated successfully")
        
    except Exception as e:
        print(f"\n❌ DEMONSTRATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())