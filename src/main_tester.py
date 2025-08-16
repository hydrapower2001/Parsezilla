#!/usr/bin/env python3
"""
XML Parser Tester

Usage:
    python main_tester.py                    # Use built-in example XML content
    python main_tester.py <path/to/file.xml> # Process XML file

When using a file, the expected XML output section is not shown.
"""

import ultra_robust_xml_parser as urxp
import sys
from pathlib import Path
from ultra_robust_xml_parser.tree.validation import OutputFormatter, OutputFormat

# Choose input source
if len(sys.argv) > 1:
    # File path provided as argument
    file_path = sys.argv[1]
    if not Path(file_path).exists():
        print(f"Error: File '{file_path}' not found")
        sys.exit(1)
    
    print(f"Processing file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        xml_content = f.read()
    
    # Parse the file content
    result = urxp.parse_string(xml_content)
    use_example = False
else:
    # Use example XML content
    print("Using example XML content (malformed)")
    xml_content = '''<?xml version="1.0" encoding="utf-8"?>
<root>
  <tag1>Simple unclosed tag
  <tag2 attr=no_quotes>Text content</tag2>
  <tag3>Special chars >> and <<</tag3>
</root>'''

    # This is how the corrected XML should look:
    expected_xml = '''<?xml version="1.0" encoding="utf-8"?>
<root>
  <tag1>Simple unclosed tag</tag1>
  <tag2 attr="no_quotes">Text content</tag2>
  <tag3>Special chars &gt;&gt; and &lt;&lt;</tag3>
</root>'''

    # Parse with conservative configuration for minimal changes
    result = urxp.parse_string(xml_content)
    use_example = True

# 2. Check the parsing results
print(f"Parsing success: {result.success}")
print(f"Confidence level: {result.confidence:.2f}")
print(f"Number of repairs made: {result.repair_count}")

# Print detailed repair information and token analysis
print("\nToken Analysis:")
for token in result.tree.source_tokens:
    print(f"Token: {token.type.name:15} | Value: {token.value:20} | Confidence: {token.confidence:.2f}")
    if token.repairs:
        for repair in token.repairs:
            print(f"  - Repair: {repair.description}")

print("\nRepair Information:")
if result.tree.repairs:
    print("\nRepairs made:")
    for repair in result.tree.repairs:
        print(f"- Type: {repair.repair_type}")
        print(f"  Description: {repair.description}")
        print(f"  Severity: {repair.severity}")
        print(f"  Confidence impact: {repair.confidence_impact}")

# 3. Generate and compare the well-formed XML
if result.success:
    # Get the corrected XML string using the proper formatter
    formatter = OutputFormatter()
    format_result = formatter.format(result.tree, OutputFormat.XML_PRETTY)
    corrected_xml = format_result.formatted_output
    
    print("\nOriginal XML:")
    print("=" * 50)
    print(xml_content)
    print("=" * 50)
    
    print("\nParser's corrected output:")
    print("=" * 50)
    print(corrected_xml)
    print("=" * 50)
    
    # Only show expected XML for the example content
    if use_example:
        print("\nExpected corrected XML:")
        print("=" * 50)
        print(expected_xml)
        print("=" * 50)
    
    # Save to file
    output_file = "/Users/christian/Downloads/xmlconf/xmltest/not-wf/sa/corrected.xml"
    try:
        with open(output_file, 'w') as f:
            f.write(corrected_xml)
        print(f"\nCorrected XML saved to: {output_file}")
    except Exception as e:
        print(f"\nWarning: Could not save to {output_file}: {e}")
else:
    print("\nParsing failed - no corrected output available")