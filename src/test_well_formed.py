import ultra_robust_xml_parser as urxp
import logging

# Enable debug logging to see fast-path detection
logging.basicConfig(level=logging.DEBUG)

# Test with well-formed XML to verify fast-path works
well_formed_xml = '''<?xml version="1.0" encoding="utf-8"?>
<root>
  <tag1>Simple text content</tag1>
  <tag2 attr="properly_quoted">Text content</tag2>
  <tag3>No special characters here</tag3>
</root>'''

# Parse with well-formed XML
result = urxp.parse_string(well_formed_xml)

print(f"Parsing success: {result.success}")
print(f"Confidence level: {result.confidence:.2f}")
print(f"Number of repairs made: {result.repair_count}")

print("\nToken Analysis:")
for token in result.tree.source_tokens[:10]:  # Show first 10 tokens
    print(f"Token: {token.type.name:15} | Value: {token.value:20} | Confidence: {token.confidence:.2f}")

print(f"\nParsed {len(result.tree.source_tokens)} tokens total")