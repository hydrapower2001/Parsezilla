"""Example migration helper for transitioning from existing XML libraries.

This example demonstrates how to use the migration utilities to analyze
existing code and migrate from lxml, ElementTree, or BeautifulSoup to
ultra_robust_xml_parser.
"""

from ultra_robust_xml_parser.api import (
    analyze_migration_code,
    convert_migration_code,
    create_migration_guide,
    create_legacy_wrapper,
    get_equivalent_adapter
)


def demonstrate_code_analysis():
    """Demonstrate code analysis for migration opportunities."""
    
    print("Migration Helper Example")
    print("=" * 25)
    
    # Example legacy code using different XML libraries
    legacy_code_samples = {
        "lxml_example": '''
import lxml.etree as ET

def process_xml_with_lxml(xml_string):
    """Process XML using lxml."""
    try:
        root = ET.fromstring(xml_string)
        print(f"Root tag: {root.tag}")
        
        # Find all child elements
        for child in root:
            print(f"Child: {child.tag} = {child.text}")
        
        return root
    except ET.XMLSyntaxError as e:
        print(f"XML parsing error: {e}")
        return None
''',
        
        "elementtree_example": '''
import xml.etree.ElementTree as ET

def parse_config_file(filename):
    """Parse configuration XML file."""
    try:
        tree = ET.parse(filename)
        root = tree.getroot()
        
        # Extract configuration values
        config = {}
        for setting in root.findall('.//setting'):
            name = setting.get('name')
            value = setting.text
            config[name] = value
        
        return config
    except ET.ParseError as e:
        print(f"Parse error: {e}")
        return {}

def create_xml_response(data):
    """Create XML response."""
    root = ET.Element("response")
    
    for key, value in data.items():
        elem = ET.SubElement(root, key)
        elem.text = str(value)
    
    return ET.tostring(root, encoding='unicode')
''',
        
        "beautifulsoup_example": '''
from bs4 import BeautifulSoup
import requests

def scrape_xml_data(url):
    """Scrape XML data from a URL."""
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'xml')
    
    # Extract data
    items = []
    for item in soup.find_all('item'):
        title = item.find('title')
        description = item.find('description')
        
        items.append({
            'title': title.text if title else '',
            'description': description.text if description else ''
        })
    
    return items
'''
    }
    
    print("1. Analyzing Legacy Code Samples")
    print("-" * 33)
    
    for example_name, code in legacy_code_samples.items():
        print(f"\n{example_name.replace('_', ' ').title()}:")
        
        # Analyze the code
        analysis = analyze_migration_code(code)
        
        print(f"  Lines of code: {analysis['total_lines']}")
        print(f"  XML imports found: {len(analysis['xml_imports'])}")
        print(f"  Function calls to migrate: {len(analysis['function_calls'])}")
        print(f"  Suggested changes: {len(analysis['suggested_changes'])}")
        
        if analysis['xml_imports']:
            print(f"  Imports: {', '.join(analysis['xml_imports'])}")
        
        # Generate migration guide
        guide = create_migration_guide(code)
        print(f"\n  Migration Guide Preview:")
        print("  " + "\n  ".join(guide.split('\n')[:8]) + "...")  # First 8 lines
    
    return legacy_code_samples


def demonstrate_code_conversion():
    """Demonstrate automatic code conversion."""
    
    print("\n2. Automatic Code Conversion")
    print("-" * 28)
    
    # Simple code sample for conversion
    original_code = '''
import lxml.etree as ET

def parse_simple_xml(xml_string):
    root = ET.fromstring(xml_string)
    return root.tag, root.text
'''
    
    print("Original Code:")
    print("-" * 13)
    print(original_code)
    
    # Convert the code
    converted_code = convert_migration_code(original_code)
    
    print("\nConverted Code:")
    print("-" * 14)
    print(converted_code)
    
    # Analyze the difference
    print("\nConversion Summary:")
    print("• Added ultra_robust_xml_parser import")
    print("• Replaced lxml.etree.fromstring with parse().to_adapter()")
    print("• Maintained original function structure")


def demonstrate_legacy_wrappers():
    """Demonstrate legacy wrapper functions for gradual migration."""
    
    print("\n3. Legacy Wrapper Functions")
    print("-" * 28)
    
    # Create wrapper functions for legacy APIs
    legacy_functions = [
        'lxml.etree.fromstring',
        'xml.etree.ElementTree.fromstring',
        'BeautifulSoup'
    ]
    
    print("Creating wrapper functions for gradual migration:")
    
    for func_name in legacy_functions:
        print(f"\n• {func_name}")
        
        # Get equivalent adapter
        equivalent = get_equivalent_adapter(func_name)
        print(f"  Equivalent adapter: {equivalent}")
        
        # Create wrapper
        wrapper = create_legacy_wrapper(func_name)
        if wrapper:
            print(f"  ✓ Wrapper function created")
            
            # Test the wrapper (simplified)
            try:
                test_xml = "<test>sample</test>"
                if func_name == 'BeautifulSoup':
                    result = wrapper(test_xml, 'xml')
                else:
                    result = wrapper(test_xml)
                
                if result:
                    print(f"  ✓ Wrapper test successful")
                else:
                    print(f"  ⚠ Wrapper test returned None")
                    
            except Exception as e:
                print(f"  ⚠ Wrapper test failed: {e}")
        else:
            print(f"  ✗ Wrapper creation failed")


def demonstrate_migration_strategy():
    """Demonstrate a complete migration strategy."""
    
    print("\n4. Complete Migration Strategy")
    print("-" * 30)
    
    migration_steps = [
        "1. Code Analysis",
        "   • Scan codebase for XML library usage",
        "   • Identify migration opportunities",
        "   • Estimate migration effort",
        "",
        "2. Gradual Migration",
        "   • Install ultra_robust_xml_parser alongside existing libraries",
        "   • Create wrapper functions for compatibility",
        "   • Migrate one module at a time",
        "",
        "3. Testing & Validation",
        "   • Run compatibility tests",
        "   • Compare outputs with original libraries",
        "   • Performance benchmark comparisons",
        "",
        "4. Full Transition",
        "   • Remove legacy library dependencies",
        "   • Update documentation",
        "   • Train team on new API"
    ]
    
    for step in migration_steps:
        print(step)
    
    print(f"\nMigration Benefits:")
    print("• Never-fail XML parsing")
    print("• Unified API across all XML libraries")
    print("• Better error handling and diagnostics")
    print("• Performance monitoring and optimization")
    print("• Stream processing capabilities")


def create_migration_checklist():
    """Create a migration checklist for teams."""
    
    checklist = '''
MIGRATION CHECKLIST
==================

Pre-Migration:
□ Audit existing XML processing code
□ Identify all XML library dependencies
□ Create test cases for current functionality
□ Backup current codebase

Migration Phase 1 - Setup:
□ Install ultra_robust_xml_parser
□ Run code analysis tools
□ Create migration plan with priorities
□ Set up development/testing environment

Migration Phase 2 - Gradual Transition:
□ Create wrapper functions for critical paths
□ Migrate low-risk modules first
□ Update unit tests for migrated code
□ Monitor performance impacts

Migration Phase 3 - Testing:
□ Run compatibility tests
□ Compare outputs with legacy libraries
□ Performance benchmark comparisons
□ Load testing with realistic data

Migration Phase 4 - Completion:
□ Migrate remaining modules
□ Remove legacy library dependencies
□ Update documentation
□ Team training on new API

Post-Migration:
□ Monitor system performance
□ Collect feedback from development team
□ Document lessons learned
□ Plan for ongoing optimization
'''
    
    return checklist


if __name__ == "__main__":
    try:
        # Run demonstrations
        legacy_samples = demonstrate_code_analysis()
        demonstrate_code_conversion()
        demonstrate_legacy_wrappers()
        demonstrate_migration_strategy()
        
        # Create and display migration checklist
        print("\n5. Migration Checklist")
        print("-" * 19)
        checklist = create_migration_checklist()
        print(checklist)
        
        print("\n" + "=" * 50)
        print("Migration Helper Demo Complete!")
        print("\nThis example showed:")
        print("• Automated code analysis for migration opportunities")
        print("• Automatic code conversion capabilities") 
        print("• Legacy wrapper functions for gradual migration")
        print("• Complete migration strategy and checklist")
        
    except Exception as e:
        print(f"Error during migration demo: {e}")
        import traceback
        traceback.print_exc()