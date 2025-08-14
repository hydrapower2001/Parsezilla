#!/usr/bin/env python3
"""
Comprehensive diagnostic reporting demonstration for the ultra-robust XML parser.

This example shows how to use the diagnostic reporting system to generate
detailed analysis reports for XML processing operations.
"""

import sys
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ultra_robust_xml_parser.tree.builder import XMLDocument, XMLElement, ParseResult
from ultra_robust_xml_parser.tree.validation import (
    TreeValidator, ValidationLevel, TreeOptimizer,
    DiagnosticReporter, DiagnosticReport
)
from ultra_robust_xml_parser.shared import DiagnosticSeverity


def create_demo_documents():
    """Create various demo documents for diagnostic reporting."""
    print("📄 Creating demo documents for diagnostic reporting...")
    
    documents = {}
    
    # 1. Clean, well-formed document
    clean_root = XMLElement(
        tag="catalog",
        attributes={"xmlns": "http://example.com/catalog", "version": "1.0"},
        children=[
            XMLElement(tag="title", text="Product Catalog"),
            XMLElement(
                tag="product",
                attributes={"id": "1", "category": "electronics"},
                children=[
                    XMLElement(tag="name", text="Laptop"),
                    XMLElement(tag="price", text="999.99"),
                    XMLElement(tag="currency", text="USD")
                ]
            )
        ]
    )
    
    # Set up parent relationships
    for child in clean_root.children:
        child.parent = clean_root
        if hasattr(child, 'children') and child.children:
            for grandchild in child.children:
                grandchild.parent = child
    
    documents["clean"] = XMLDocument(root=clean_root, version="1.0", encoding="utf-8")
    
    # 2. Document with validation issues
    problematic_root = XMLElement(
        tag="document",
        attributes={"empty_attr": "", "valid_attr": "value"},  # Empty attribute
        children=[
            XMLElement(tag="ns:element", text="Namespace without declaration"),  # Namespace issue
            XMLElement(tag="element", text="  Extra   whitespace  "),  # Whitespace issue
            XMLElement(tag="empty_element"),  # Empty element
            XMLElement(tag="another_empty")   # Another empty element
        ]
    )
    
    for child in problematic_root.children:
        child.parent = problematic_root
    
    documents["problematic"] = XMLDocument(root=problematic_root)
    
    # 3. Complex nested document with optimization opportunities
    complex_root = XMLElement(tag="complex_document")
    current = complex_root
    
    # Create deep nesting
    for level in range(5):
        for i in range(2):
            element = XMLElement(
                tag=f"level{level}_item{i}",
                attributes={
                    "level": str(level),
                    "item": str(i),
                    "empty1": "",  # Empty attributes for optimization
                    "empty2": "",
                    "valid": f"value_{level}_{i}"
                },
                text=f"   Content   with   excessive   whitespace   {level}-{i}   "
            )
            current.children.append(element)
            element.parent = current
            
            if level < 3:
                current = element  # Go deeper
    
    documents["complex"] = XMLDocument(root=complex_root)
    
    print(f"✅ Created {len(documents)} demo documents")
    return documents


def demonstrate_basic_diagnostic_reporting(documents):
    """Demonstrate basic diagnostic reporting functionality."""
    print("\n🔍 BASIC DIAGNOSTIC REPORTING")
    print("=" * 50)
    
    reporter = DiagnosticReporter(correlation_id="basic-demo")
    
    for doc_name, document in documents.items():
        print(f"\n📄 Analyzing {doc_name} document:")
        print("-" * 30)
        
        # Create a ParseResult for demonstration
        parse_result = ParseResult(
            document=document,
            success=True,
            confidence=0.85 if doc_name == "clean" else 0.65
        )
        
        # Add some sample diagnostics based on document type
        if doc_name == "problematic":
            parse_result.add_diagnostic(DiagnosticSeverity.WARNING, "Empty attribute detected", "validation")
            parse_result.add_diagnostic(DiagnosticSeverity.ERROR, "Namespace without declaration", "namespace")
            parse_result.add_diagnostic(DiagnosticSeverity.INFO, "Whitespace normalization needed", "content")
        elif doc_name == "complex":
            parse_result.add_diagnostic(DiagnosticSeverity.INFO, "Deep nesting detected", "structure")
            parse_result.add_diagnostic(DiagnosticSeverity.WARNING, "Multiple empty attributes found", "optimization")
        
        # Generate diagnostic report
        report = reporter.generate_comprehensive_report(parse_result)
        
        print(f"  📋 Report ID: {report.report_id}")
        print(f"  🏥 Document Health: {report.document_health}")
        print(f"  📊 Overall Confidence: {report.overall_confidence:.3f}")
        print(f"  ⚠️  Total Issues: {report.error_summary.total_errors + report.warning_count + report.info_count}")
        print(f"  🔧 Repair Assessment: {report.repair_summary.repair_impact_assessment}")
        
        if report.recommendations:
            print(f"  💡 Key Recommendations:")
            for i, rec in enumerate(report.recommendations[:2], 1):
                print(f"    {i}. {rec}")


def demonstrate_comprehensive_analysis(documents):
    """Demonstrate comprehensive analysis with validation and optimization."""
    print("\n📊 COMPREHENSIVE ANALYSIS WITH VALIDATION & OPTIMIZATION")
    print("=" * 65)
    
    reporter = DiagnosticReporter(correlation_id="comprehensive-demo")
    
    # Use the problematic document for comprehensive analysis
    document = documents["problematic"]
    
    print("🔍 Step 1: Initial Parsing")
    parse_result = ParseResult(
        document=document,
        success=True,
        confidence=0.7
    )
    parse_result.add_diagnostic(DiagnosticSeverity.WARNING, "Document contains potential issues", "general")
    
    print("🔍 Step 2: Validation Analysis")
    validator = TreeValidator(ValidationLevel.STRICT)
    validation_result = validator.validate(document)
    
    print(f"  ✅ Validation Success: {validation_result.success}")
    print(f"  📊 Validation Confidence: {validation_result.confidence:.3f}")
    print(f"  ⚠️  Issues Found: {len(validation_result.issues)}")
    
    print("🔍 Step 3: Optimization Analysis")
    optimizer = TreeOptimizer()
    optimization_result = optimizer.optimize(document)
    
    print(f"  ✅ Optimization Success: {optimization_result.success}")
    print(f"  🔧 Actions Performed: {optimization_result.total_actions}")
    print(f"  💾 Memory Saved: {optimization_result.total_memory_saved_bytes} bytes")
    
    print("🔍 Step 4: Comprehensive Report Generation")
    comprehensive_report = reporter.generate_comprehensive_report(
        parse_result=parse_result,
        validation_result=validation_result,
        optimization_result=optimization_result,
        include_recommendations=True
    )
    
    print(f"\n📋 COMPREHENSIVE ANALYSIS RESULTS:")
    print(f"  🏥 Document Health: {comprehensive_report.document_health}")
    print(f"  📊 Overall Confidence: {comprehensive_report.overall_confidence:.3f}")
    print(f"  📈 Processing Summary:")
    for key, value in comprehensive_report.processing_summary.items():
        print(f"    {key.replace('_', ' ').title()}: {value}")
    
    print(f"\n  🚨 Error Analysis:")
    error_summary = comprehensive_report.error_summary
    print(f"    Total Errors: {error_summary.total_errors}")
    print(f"    Critical Errors: {error_summary.critical_errors}")
    print(f"    Error Rate: {error_summary.error_rate:.4f}")
    
    if error_summary.error_categories:
        print(f"    Error Categories:")
        for category, count in error_summary.error_categories.items():
            print(f"      {category.title()}: {count}")
    
    print(f"\n  🔧 Repair Analysis:")
    repair_summary = comprehensive_report.repair_summary
    print(f"    Total Repairs: {repair_summary.total_repairs}")
    print(f"    Success Rate: {repair_summary.repair_success_rate:.1%}")
    print(f"    Impact: {repair_summary.repair_impact_assessment}")
    
    print(f"\n  📊 Diagnostic Categories:")
    for name, category in comprehensive_report.diagnostic_categories.items():
        if category.issue_count > 0:
            print(f"    {category.name}: {category.issue_count} issues")
    
    if comprehensive_report.recommendations:
        print(f"\n  💡 Recommendations:")
        for i, recommendation in enumerate(comprehensive_report.recommendations, 1):
            print(f"    {i}. {recommendation}")
    
    return comprehensive_report


def demonstrate_report_export_formats(report):
    """Demonstrate different report export formats."""
    print("\n📄 REPORT EXPORT FORMATS")
    print("=" * 40)
    
    reporter = DiagnosticReporter()
    
    # Export formats to demonstrate
    formats = [
        ("text", "Plain Text Report"),
        ("json", "JSON Format"),
        ("xml", "XML Format"),
        ("html", "HTML Format")
    ]
    
    for format_type, description in formats:
        print(f"\n📋 {description}:")
        print("-" * 25)
        
        try:
            exported_report = reporter.export_report(report, format_type)
            
            # Show sample of the exported report
            lines = exported_report.split('\n')
            sample_lines = lines[:10]  # First 10 lines
            
            for line in sample_lines:
                print(f"  {line}")
            
            if len(lines) > 10:
                print(f"  ... ({len(lines) - 10} more lines)")
            
            print(f"  📏 Total size: {len(exported_report)} characters")
            
            # Save to file for external review
            output_file = Path(__file__).parent / f"sample_diagnostic_report.{format_type}"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(exported_report)
            print(f"  💾 Saved to: {output_file}")
            
        except Exception as e:
            print(f"  ❌ Export failed: {e}")


def demonstrate_health_assessment_scenarios():
    """Demonstrate different document health assessment scenarios."""
    print("\n🏥 DOCUMENT HEALTH ASSESSMENT SCENARIOS")
    print("=" * 50)
    
    reporter = DiagnosticReporter(correlation_id="health-demo")
    
    # Scenario 1: Excellent document
    print("\n🌟 Scenario 1: Excellent Document")
    excellent_doc = XMLDocument(root=XMLElement(tag="perfect", text="Perfect document"))
    excellent_result = ParseResult(document=excellent_doc, success=True, confidence=0.99)
    excellent_report = reporter.generate_comprehensive_report(excellent_result)
    
    print(f"  Health: {excellent_report.document_health}")
    print(f"  Confidence: {excellent_report.overall_confidence:.3f}")
    print(f"  Is Healthy: {excellent_report.is_healthy()}")
    
    # Scenario 2: Document with warnings
    print("\n⚠️  Scenario 2: Document with Warnings")
    warning_doc = XMLDocument(root=XMLElement(tag="warning_doc"))
    warning_result = ParseResult(document=warning_doc, success=True, confidence=0.8)
    warning_result.add_diagnostic(DiagnosticSeverity.WARNING, "Minor formatting issue", "format")
    warning_result.add_diagnostic(DiagnosticSeverity.WARNING, "Optimization opportunity", "performance")
    warning_report = reporter.generate_comprehensive_report(warning_result)
    
    print(f"  Health: {warning_report.document_health}")
    print(f"  Confidence: {warning_report.overall_confidence:.3f}")
    print(f"  Warnings: {warning_report.warning_count}")
    
    # Scenario 3: Document with errors
    print("\n❌ Scenario 3: Document with Errors")
    error_doc = XMLDocument(root=XMLElement(tag="error_doc"))
    error_result = ParseResult(document=error_doc, success=True, confidence=0.6)
    error_result.add_diagnostic(DiagnosticSeverity.ERROR, "Structural integrity issue", "structure")
    error_result.add_diagnostic(DiagnosticSeverity.ERROR, "Namespace validation failed", "namespace")
    error_report = reporter.generate_comprehensive_report(error_result)
    
    print(f"  Health: {error_report.document_health}")
    print(f"  Confidence: {error_report.overall_confidence:.3f}")
    print(f"  Errors: {error_report.error_summary.total_errors}")
    
    # Scenario 4: Critical document
    print("\n🚨 Scenario 4: Critical Document")
    critical_doc = XMLDocument(root=XMLElement(tag="critical_doc"))
    critical_result = ParseResult(document=critical_doc, success=True, confidence=0.3)
    critical_result.add_diagnostic(DiagnosticSeverity.CRITICAL, "Severe structural damage", "structure")
    critical_result.add_diagnostic(DiagnosticSeverity.CRITICAL, "Security vulnerability detected", "security")
    critical_report = reporter.generate_comprehensive_report(critical_result)
    
    print(f"  Health: {critical_report.document_health}")
    print(f"  Confidence: {critical_report.overall_confidence:.3f}")
    print(f"  Critical Issues: {critical_report.error_summary.critical_errors}")
    print(f"  Is Healthy: {critical_report.is_healthy()}")
    
    if critical_report.recommendations:
        print(f"  🚨 Critical Recommendations:")
        for rec in critical_report.recommendations:
            if "CRITICAL" in rec or "SECURITY" in rec:
                print(f"    • {rec}")


def demonstrate_diagnostic_categorization():
    """Demonstrate how diagnostics are categorized."""
    print("\n📊 DIAGNOSTIC CATEGORIZATION SYSTEM")
    print("=" * 50)
    
    reporter = DiagnosticReporter(correlation_id="category-demo")
    
    # Create document with various types of issues
    test_doc = XMLDocument(root=XMLElement(tag="test"))
    test_result = ParseResult(document=test_doc, success=True, confidence=0.7)
    
    # Add diagnostics of different types
    diagnostic_samples = [
        (DiagnosticSeverity.ERROR, "Document structure is malformed", "structure"),
        (DiagnosticSeverity.WARNING, "Namespace prefix not declared", "namespaces"),
        (DiagnosticSeverity.WARNING, "Empty attribute value found", "attributes"),
        (DiagnosticSeverity.INFO, "Text content could be normalized", "content"),
        (DiagnosticSeverity.ERROR, "Character encoding issue detected", "encoding"),
        (DiagnosticSeverity.INFO, "XML compliance check passed", "compliance"),
        (DiagnosticSeverity.INFO, "Memory usage can be optimized", "optimization"),
        (DiagnosticSeverity.CRITICAL, "Potential XXE vulnerability", "security")
    ]
    
    print("📝 Adding sample diagnostics:")
    for severity, message, category_hint in diagnostic_samples:
        test_result.add_diagnostic(severity, message, category_hint)
        print(f"  {severity.name}: {message}")
    
    # Generate report to see categorization
    categorized_report = reporter.generate_comprehensive_report(test_result)
    
    print(f"\n📊 Categorization Results:")
    print(f"  Total diagnostics processed: {len(diagnostic_samples)}")
    
    for name, category in categorized_report.diagnostic_categories.items():
        if category.issue_count > 0:
            print(f"\n  📋 {category.name}:")
            print(f"    Description: {category.description}")
            print(f"    Issue Count: {category.issue_count}")
            print(f"    Severity Level: {category.severity_level.name}")
    
    # Show severity breakdown
    print(f"\n📈 Severity Breakdown:")
    severity_breakdown = categorized_report.get_severity_breakdown()
    for severity, count in severity_breakdown.items():
        if count > 0:
            print(f"  {severity}: {count}")


def main():
    """Main demonstration function."""
    print("🚀 ULTRA-ROBUST XML PARSER - DIAGNOSTIC REPORTING DEMO")
    print("=" * 65)
    
    try:
        # Create demo documents
        documents = create_demo_documents()
        
        # Run all demonstrations
        demonstrate_basic_diagnostic_reporting(documents)
        
        comprehensive_report = demonstrate_comprehensive_analysis(documents)
        
        demonstrate_report_export_formats(comprehensive_report)
        
        demonstrate_health_assessment_scenarios()
        
        demonstrate_diagnostic_categorization()
        
        print(f"\n🎉 DIAGNOSTIC REPORTING DEMONSTRATION COMPLETE!")
        print("✅ All diagnostic reporting features demonstrated successfully")
        print("\n📄 Generated sample reports saved to current directory")
        print("🔍 Review the exported reports to see the full diagnostic capabilities")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ DEMONSTRATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())