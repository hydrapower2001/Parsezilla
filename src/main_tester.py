import ultra_robust_xml_parser as urxp
from pathlib import Path
from ultra_robust_xml_parser.tree.validation import OutputFormatter, OutputFormat

# Test with a minimal example
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

# The corrections made:
# 1. Added missing </tag1> closing tag
# 2. Added quotes around the attr value in tag2
# 3. Converted < and > to &lt; and &gt; entities in tag3's content
# Note: Whitespace and indentation should be preserved

# Parse with conservative configuration for minimal changes
result = urxp.parse_string(xml_content)

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
    
    print("\nOriginal malformed XML:")
    print("=" * 50)
    print(xml_content)
    print("=" * 50)
    
    print("\nParser's corrected output:")
    print("=" * 50)
    print(corrected_xml)
    print("=" * 50)
    
    print("\nExpected corrected XML:")
    print("=" * 50)
    print(expected_xml)
    print("=" * 50)
    
    # Save to file
    output_file = "/Users/christian/Downloads/xmlconf/xmltest/not-wf/sa/corrected.xml"
    with open(output_file, 'w') as f:
        f.write(corrected_xml)
if result.success:
    output_file = "/Users/christian/Downloads/xmlconf/xmltest/not-wf/sa/corrected.xml"
    # Get the corrected XML string using the proper formatter
    formatter = OutputFormatter()
    format_result = formatter.format(result.tree, OutputFormat.XML_PRETTY)
    corrected_xml = format_result.formatted_output
    
    print("\nWell-formed XML output:")
    print("=" * 50)
    print(corrected_xml)
    print("=" * 50)
    
    # Save to file
    with open(output_file, 'w') as f:
        f.write(corrected_xml)
    output_file = "/Users/christian/Downloads/xmlconf/xmltest/not-wf/sa/corrected.xml"