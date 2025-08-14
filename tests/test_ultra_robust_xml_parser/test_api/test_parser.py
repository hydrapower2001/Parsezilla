"""Tests for the core parser API with progressive disclosure.

Tests the main parsing API functions and UltraRobustXMLParser class,
ensuring never-fail philosophy and progressive disclosure functionality.
"""

import tempfile
from pathlib import Path
from typing import Any

import pytest

from ultra_robust_xml_parser.api.parser import (
    UltraRobustXMLParser,
    parse,
    parse_file,
    parse_string,
)
from ultra_robust_xml_parser.shared.config import TokenizationConfig
from ultra_robust_xml_parser.tree.builder import ParseResult


class TestSimpleParsingFunctions:
    """Test Level 1: Simple module-level parsing functions."""

    def test_parse_string_basic(self):
        """Test basic string parsing functionality."""
        xml = '<root><item id="1">value</item></root>'
        result = parse_string(xml)
        
        assert isinstance(result, ParseResult)
        assert result.success is True
        assert result.confidence > 0.5
        assert result.tree.root.tag == "root"
        assert len(result.tree.find_all("item")) == 1
        assert result.tree.find("item").get_attribute("id") == "1"
        assert result.tree.find("item").text == "value"

    def test_parse_string_malformed(self):
        """Test string parsing with malformed XML."""
        xml = '<root><unclosed>content<missing_end>more</root>'
        result = parse_string(xml)
        
        assert isinstance(result, ParseResult)
        assert result.success is True  # Never-fail philosophy
        assert result.repair_count > 0
        assert result.tree.root.tag == "root"

    def test_parse_string_empty(self):
        """Test string parsing with empty input."""
        result = parse_string("")
        
        assert isinstance(result, ParseResult)
        # Should handle gracefully with diagnostics
        assert len(result.diagnostics) > 0

    def test_parse_universal_string(self):
        """Test universal parse function with string input."""
        xml = '<root><item>test</item></root>'
        result = parse(xml)
        
        assert isinstance(result, ParseResult)
        assert result.success is True
        assert result.tree.root.tag == "root"

    def test_parse_universal_bytes(self):
        """Test universal parse function with bytes input."""
        xml = b'<?xml version="1.0" encoding="UTF-8"?><root><item>test</item></root>'
        result = parse(xml)
        
        assert isinstance(result, ParseResult)
        assert result.success is True
        assert result.tree.root.tag == "root"

    def test_parse_universal_unknown_type(self):
        """Test universal parse function with unknown input type."""
        # Pass an integer (should convert to string)
        result = parse(12345)
        
        assert isinstance(result, ParseResult)
        # May not be successful XML but should not raise exception

    def test_parse_file_basic(self):
        """Test file parsing with temporary file."""
        xml_content = '''<?xml version="1.0"?>
        <document>
            <title>Test Document</title>
            <content>File parsing test</content>
        </document>'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(xml_content)
            temp_path = f.name
        
        try:
            result = parse_file(temp_path)
            assert isinstance(result, ParseResult)
            assert result.success is True
            assert result.tree.root.tag == "document"
            assert result.tree.find("title").text.strip() == "Test Document"
        finally:
            Path(temp_path).unlink()

    def test_parse_file_nonexistent(self):
        """Test file parsing with non-existent file."""
        result = parse_file("/nonexistent/file.xml")
        
        assert isinstance(result, ParseResult)
        assert result.success is False
        assert "not found" in result.diagnostics[0].message.lower()

    def test_parse_file_with_path_object(self):
        """Test file parsing with Path object."""
        xml_content = '<root><item>path object test</item></root>'
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(xml_content)
            temp_path = Path(f.name)
        
        try:
            result = parse_file(temp_path)
            assert isinstance(result, ParseResult)
            assert result.success is True
            assert result.tree.root.tag == "root"
        finally:
            temp_path.unlink()

    def test_parse_file_encoding_override(self):
        """Test file parsing with encoding override."""
        xml_content = '<root><item>encoding test</item></root>'
        
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', 
                                       suffix='.xml', delete=False) as f:
            f.write(xml_content)
            temp_path = f.name
        
        try:
            result = parse_file(temp_path, encoding='utf-8')
            assert isinstance(result, ParseResult)
            assert result.success is True
            assert result.tree.root.tag == "root"
        finally:
            Path(temp_path).unlink()

    def test_parse_with_file_like_object(self):
        """Test parse with file-like object."""
        from io import StringIO
        
        xml_content = '<root><item>file-like test</item></root>'
        file_obj = StringIO(xml_content)
        
        result = parse(file_obj)
        assert isinstance(result, ParseResult)
        assert result.success is True
        assert result.tree.root.tag == "root"

    def test_never_fail_guarantee(self):
        """Test that API functions never raise exceptions."""
        problematic_inputs = [
            None,
            b'\xff\xfe\x00\x00',  # Invalid encoding
            '<><><>invalid><><',   # Severely malformed
            123.456,              # Numeric input
            ['list', 'input'],    # List input
        ]
        
        for inp in problematic_inputs:
            try:
                result = parse(inp)
                assert isinstance(result, ParseResult)
                # May not be successful, but should return a result
            except Exception as e:
                pytest.fail(f"parse() raised exception with input {inp}: {e}")


class TestUltraRobustXMLParser:
    """Test Level 2: Advanced UltraRobustXMLParser class."""

    def test_basic_initialization(self):
        """Test basic parser initialization."""
        parser = UltraRobustXMLParser()
        assert isinstance(parser, UltraRobustXMLParser)
        assert parser.config is not None
        assert parser._parse_count == 0

    def test_initialization_with_config(self):
        """Test parser initialization with custom config."""
        config = TokenizationConfig.conservative()
        parser = UltraRobustXMLParser(config=config)
        
        assert parser.config == config
        assert parser._parse_count == 0

    def test_basic_parsing(self):
        """Test basic parsing with parser class."""
        parser = UltraRobustXMLParser()
        xml = '<root><item>test</item></root>'
        
        result = parser.parse(xml)
        assert isinstance(result, ParseResult)
        assert result.success is True
        assert result.tree.root.tag == "root"

    def test_parser_reuse_statistics(self):
        """Test parser reuse and statistics tracking."""
        parser = UltraRobustXMLParser()
        
        # Initial statistics
        stats = parser.statistics
        assert stats['total_parses'] == 0
        assert stats['successful_parses'] == 0
        assert stats['success_rate'] == 0.0
        
        # Parse multiple documents
        xml_docs = [
            '<doc1><title>First</title></doc1>',
            '<doc2><title>Second</title></doc2>',
            '<malformed><unclosed>content</malformed>',
        ]
        
        for xml in xml_docs:
            result = parser.parse(xml)
            assert isinstance(result, ParseResult)
        
        # Check updated statistics
        final_stats = parser.statistics
        assert final_stats['total_parses'] == 3
        assert final_stats['successful_parses'] >= 2  # At least well-formed ones
        assert final_stats['success_rate'] > 0.5
        assert final_stats['total_processing_time_ms'] > 0

    def test_configuration_override(self):
        """Test per-parse configuration override."""
        parser = UltraRobustXMLParser(config=TokenizationConfig.balanced())
        xml = '<root><item>test</item></root>'
        
        # Parse with default config
        result1 = parser.parse(xml)
        assert isinstance(result1, ParseResult)
        
        # Parse with override config
        conservative_config = TokenizationConfig.conservative()
        result2 = parser.parse(xml, config_override=conservative_config)
        assert isinstance(result2, ParseResult)
        
        # Both should be successful
        assert result1.success is True
        assert result2.success is True

    def test_reconfigure_parser(self):
        """Test parser reconfiguration."""
        parser = UltraRobustXMLParser()
        original_config = parser.config
        
        # Reconfigure with new config
        new_config = TokenizationConfig.aggressive()
        parser.reconfigure(config=new_config)
        
        assert parser.config != original_config
        assert parser.config.recovery.strategy.name == "AGGRESSIVE"

    def test_reset_statistics(self):
        """Test statistics reset functionality."""
        parser = UltraRobustXMLParser()
        
        # Parse some documents
        for i in range(3):
            parser.parse(f'<doc{i}><content>test</content></doc{i}>')
        
        # Verify statistics exist
        stats = parser.statistics
        assert stats['total_parses'] == 3
        
        # Reset and verify
        parser.reset_statistics()
        reset_stats = parser.statistics
        assert reset_stats['total_parses'] == 0
        assert reset_stats['successful_parses'] == 0
        assert reset_stats['total_processing_time_ms'] == 0.0

    def test_correlation_id_tracking(self):
        """Test correlation ID tracking."""
        correlation_id = "test-correlation-123"
        parser = UltraRobustXMLParser(correlation_id=correlation_id)
        
        result = parser.parse('<root><item>test</item></root>')
        assert result.correlation_id == correlation_id

    def test_never_fail_advanced_parser(self):
        """Test never-fail guarantee for advanced parser."""
        parser = UltraRobustXMLParser()
        
        problematic_inputs = [
            b'\xff\xfe\x00\x00invalid',
            '<><><>malformed><><',
            None,
            123,
        ]
        
        for inp in problematic_inputs:
            try:
                result = parser.parse(inp)
                assert isinstance(result, ParseResult)
            except Exception as e:
                pytest.fail(f"UltraRobustXMLParser.parse() raised exception: {e}")


class TestParseResultEnhancements:
    """Test enhanced ParseResult properties for intuitive API."""

    def test_tree_property(self):
        """Test .tree property for direct XMLDocument access."""
        xml = '<root><item id="1">value</item></root>'
        result = parse_string(xml)
        
        # Test tree property
        tree = result.tree
        assert tree is result.document  # Should be same object
        assert tree.root.tag == "root"
        assert len(tree.find_all("item")) == 1

    def test_success_property(self):
        """Test .success property functionality."""
        # Successful parse
        result = parse_string('<root><item>test</item></root>')
        assert result.success is True
        
        # This should still succeed due to never-fail philosophy
        result = parse_string('<malformed><unclosed>content')
        assert result.success is True  # Repairs applied

    def test_confidence_property(self):
        """Test .confidence property functionality."""
        # Well-formed XML should have high confidence
        result = parse_string('<root><item>test</item></root>')
        assert 0.0 <= result.confidence <= 1.0
        assert result.confidence > 0.7
        
        # Malformed XML should have lower confidence
        result = parse_string('<root><unclosed>content<missing>')
        assert 0.0 <= result.confidence <= 1.0

    def test_metadata_property(self):
        """Test .metadata property comprehensive information."""
        xml = '''
        <catalog>
            <book id="1">
                <title>Test Book</title>
                <author>Test Author</author>
            </book>
            <book id="2">
                <title>Another Book</title>
            </book>
        </catalog>
        '''
        
        result = parse_string(xml)
        metadata = result.metadata
        
        # Check required metadata fields
        assert 'element_count' in metadata
        assert 'processing_time_ms' in metadata
        assert 'confidence_breakdown' in metadata
        assert 'parsing_statistics' in metadata
        assert 'diagnostics_summary' in metadata
        
        # Verify data types and ranges
        assert isinstance(metadata['element_count'], int)
        assert metadata['element_count'] > 0
        assert isinstance(metadata['processing_time_ms'], (int, float))
        assert metadata['processing_time_ms'] >= 0
        assert isinstance(metadata['confidence_breakdown'], dict)

    def test_metadata_with_repairs(self):
        """Test metadata information with repairs."""
        malformed_xml = '<root><unclosed>content<missing>more'
        result = parse_string(malformed_xml)
        
        metadata = result.metadata
        assert 'repair_count' in metadata
        assert 'has_repairs' in metadata
        assert 'repair_summary' in metadata
        
        if result.repair_count > 0:
            assert metadata['has_repairs'] is True
            assert 'repair_types' in metadata['repair_summary']


class TestProgressiveDisclosure:
    """Test progressive API disclosure architecture."""

    def test_api_level_progression(self):
        """Test progression from simple to advanced API levels."""
        xml = '<root><item id="1">test</item></root>'
        
        # Level 1: Simple function
        result1 = parse_string(xml)
        assert result1.success
        
        # Level 2: Advanced parser
        parser = UltraRobustXMLParser()
        result2 = parser.parse(xml)
        assert result2.success
        
        # Both should produce equivalent results
        assert result1.tree.root.tag == result2.tree.root.tag
        assert result1.tree.total_elements == result2.tree.total_elements

    def test_configuration_complexity_levels(self):
        """Test different configuration complexity levels."""
        xml = '<root><item>test</item></root>'
        
        # Simple: no configuration
        result_simple = parse_string(xml)
        
        # Intermediate: preset configuration
        parser_balanced = UltraRobustXMLParser(config=TokenizationConfig.balanced())
        result_balanced = parser_balanced.parse(xml)
        
        # Advanced: custom configuration
        custom_config = TokenizationConfig.conservative()
        parser_custom = UltraRobustXMLParser(config=custom_config)
        result_custom = parser_custom.parse(xml)
        
        # All should succeed
        assert result_simple.success
        assert result_balanced.success
        assert result_custom.success

    def test_result_access_patterns(self):
        """Test different result access patterns."""
        xml = '<root><item id="1">value</item></root>'
        result = parse_string(xml)
        
        # Simple access: direct properties
        assert result.success
        assert result.confidence > 0
        
        # Intermediate access: tree navigation
        root_tag = result.tree.root.tag
        item_value = result.tree.find('item').text
        assert root_tag == "root"
        assert item_value == "value"
        
        # Advanced access: comprehensive metadata
        metadata = result.metadata
        stats = metadata['parsing_statistics']
        assert 'characters_processed' in stats
        assert 'tokens_generated' in stats


class TestPythonConventionCompliance:
    """Test Python convention compliance and familiar patterns."""

    def test_function_naming_conventions(self):
        """Test that function names follow Python conventions."""
        # snake_case function names
        assert callable(parse)
        assert callable(parse_string)
        assert callable(parse_file)
        
        # Class names in PascalCase
        parser = UltraRobustXMLParser()
        assert parser.__class__.__name__ == "UltraRobustXMLParser"

    def test_familiar_api_patterns(self):
        """Test API patterns familiar to Python XML library users."""
        xml = '<root><item id="1">value</item></root>'
        result = parse_string(xml)
        
        # Tree-like navigation similar to xml.etree
        root = result.tree.root
        assert root.tag == "root"
        
        # Element finding similar to BeautifulSoup/lxml
        item = result.tree.find('item')
        assert item is not None
        assert item.get_attribute('id') == '1'
        
        # Text access
        assert item.text == 'value'

    def test_docstring_conventions(self):
        """Test that functions have proper docstrings."""
        # Check main functions have docstrings
        assert parse.__doc__ is not None
        assert parse_string.__doc__ is not None
        assert parse_file.__doc__ is not None
        
        # Check class has docstring
        assert UltraRobustXMLParser.__doc__ is not None
        
        # Check docstrings contain examples
        assert "Examples:" in parse.__doc__
        assert "Examples:" in parse_string.__doc__
        assert "Examples:" in UltraRobustXMLParser.__doc__

    def test_return_type_consistency(self):
        """Test consistent return types across API."""
        xml = '<root><item>test</item></root>'
        
        # All parsing functions should return ParseResult
        result1 = parse(xml)
        result2 = parse_string(xml)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(xml)
            temp_path = f.name
        
        try:
            result3 = parse_file(temp_path)
            
            # All should be ParseResult instances
            assert isinstance(result1, ParseResult)
            assert isinstance(result2, ParseResult)
            assert isinstance(result3, ParseResult)
            
        finally:
            Path(temp_path).unlink()

    def test_error_handling_patterns(self):
        """Test error handling follows Python patterns."""
        # Functions should not raise exceptions (never-fail)
        # but should return result objects with error information
        
        try:
            result = parse_string("invalid<<>>xml")
            assert isinstance(result, ParseResult)
            # Should have diagnostics but not raise
        except Exception as e:
            pytest.fail(f"Function raised unexpected exception: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])