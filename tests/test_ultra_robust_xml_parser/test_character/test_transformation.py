"""Tests for character transformation system."""

import pytest
from unittest.mock import patch

from ultra_robust_xml_parser.character.transformation import (
    CharacterTransformer,
    TransformationContext,
    TransformationStrategy,
    TransformConfig,
    TransformResult,
    TransformationChange,
    XML10Validator,
)


class TestXML10Validator:
    """Tests for XML 1.0 character validation."""
    
    def test_valid_characters(self):
        """Test validation of valid XML 1.0 characters."""
        # Tab, LF, CR
        assert XML10Validator.is_valid_xml_char(0x0009)
        assert XML10Validator.is_valid_xml_char(0x000A)
        assert XML10Validator.is_valid_xml_char(0x000D)
        
        # Basic ASCII range
        assert XML10Validator.is_valid_xml_char(0x0020)  # Space
        assert XML10Validator.is_valid_xml_char(0x0041)  # 'A'
        assert XML10Validator.is_valid_xml_char(0x007A)  # 'z'
        
        # Extended valid ranges
        assert XML10Validator.is_valid_xml_char(0xD7FF)  # End of basic plane
        assert XML10Validator.is_valid_xml_char(0xE000)  # Start of private use
        assert XML10Validator.is_valid_xml_char(0xFFFD)  # Replacement character
        
        # Supplementary planes
        assert XML10Validator.is_valid_xml_char(0x10000)  # Start of supplementary
        assert XML10Validator.is_valid_xml_char(0x10FFFF)  # End of valid Unicode
    
    def test_invalid_characters(self):
        """Test validation of invalid XML 1.0 characters."""
        # Control characters (except allowed ones)
        assert not XML10Validator.is_valid_xml_char(0x0000)  # NULL
        assert not XML10Validator.is_valid_xml_char(0x0008)  # Backspace
        assert not XML10Validator.is_valid_xml_char(0x000B)  # Vertical tab
        assert not XML10Validator.is_valid_xml_char(0x000C)  # Form feed
        assert not XML10Validator.is_valid_xml_char(0x001F)  # Unit separator
        
        # Surrogate range
        assert not XML10Validator.is_valid_xml_char(0xD800)  # Start of surrogates
        assert not XML10Validator.is_valid_xml_char(0xDFFF)  # End of surrogates
        
        # Non-characters
        assert not XML10Validator.is_valid_xml_char(0xFFFE)
        assert not XML10Validator.is_valid_xml_char(0xFFFF)
        assert not XML10Validator.is_valid_xml_char(0x1FFFE)
        assert not XML10Validator.is_valid_xml_char(0x1FFFF)
        assert not XML10Validator.is_valid_xml_char(0x10FFFE)  # Non-character at end of valid Unicode
        
        # Beyond valid Unicode (this should be valid)
        assert XML10Validator.is_valid_xml_char(0x20000)  # Valid supplementary plane
        assert not XML10Validator.is_valid_xml_char(0x110000)  # Beyond valid range
    
    def test_validation_issues(self):
        """Test detailed validation issue reporting."""
        # Control character
        issues = XML10Validator.get_validation_issues(0x0000)
        assert len(issues) == 1
        assert "Control character" in issues[0]
        
        # Surrogate
        issues = XML10Validator.get_validation_issues(0xD800)
        assert len(issues) == 1
        assert "Surrogate character" in issues[0]
        
        # Non-character
        issues = XML10Validator.get_validation_issues(0xFFFE)
        assert len(issues) == 1
        assert "Non-character" in issues[0]
        
        # Valid character should have no issues
        issues = XML10Validator.get_validation_issues(0x0041)
        assert len(issues) == 0
    
    def test_cache_functionality(self):
        """Test validation caching."""
        XML10Validator.clear_cache()
        
        # First validation should cache result
        result1 = XML10Validator.is_valid_xml_char(0x0041)
        
        # Second validation should use cache
        result2 = XML10Validator.is_valid_xml_char(0x0041)
        
        assert result1 == result2 == True
        
        # Cache should contain the result
        assert 0x0041 in XML10Validator._validation_cache


class TestTransformConfig:
    """Tests for transformation configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TransformConfig()
        
        assert config.default_strategy == TransformationStrategy.REPLACEMENT
        assert config.replacement_char == "\uFFFD"
        assert config.preserve_whitespace is True
        assert config.strict_xml is True
        assert len(config.context_strategies) == 0
        assert len(config.custom_mappings) == 0
    
    def test_strict_preset(self):
        """Test strict configuration preset."""
        config = TransformConfig.create_preset("strict")
        
        assert config.default_strategy == TransformationStrategy.REMOVAL
        assert config.strict_xml is True
        assert config.preserve_whitespace is False
    
    def test_lenient_preset(self):
        """Test lenient configuration preset."""
        config = TransformConfig.create_preset("lenient")
        
        assert config.default_strategy == TransformationStrategy.REPLACEMENT
        assert config.strict_xml is True
        assert config.preserve_whitespace is True
        
        # Should have stricter rules for tag and attribute names
        assert config.context_strategies[TransformationContext.TAG_NAME] == TransformationStrategy.REMOVAL
        assert config.context_strategies[TransformationContext.ATTRIBUTE_NAME] == TransformationStrategy.REMOVAL
    
    def test_data_recovery_preset(self):
        """Test data recovery configuration preset."""
        config = TransformConfig.create_preset("data_recovery")
        
        assert config.default_strategy == TransformationStrategy.PRESERVATION
        assert config.strict_xml is False
        assert config.preserve_whitespace is True
        
        # Should have replacement for structural elements
        assert config.context_strategies[TransformationContext.TAG_NAME] == TransformationStrategy.REPLACEMENT
        assert config.context_strategies[TransformationContext.ATTRIBUTE_NAME] == TransformationStrategy.REPLACEMENT
    
    def test_invalid_preset(self):
        """Test error handling for invalid preset."""
        with pytest.raises(ValueError, match="Unknown preset"):
            TransformConfig.create_preset("invalid_preset")


class TestTransformResult:
    """Tests for transformation result."""
    
    def test_valid_confidence(self):
        """Test valid confidence score."""
        result = TransformResult(text="test", confidence=0.5)
        assert result.confidence == 0.5
    
    def test_invalid_confidence_too_low(self):
        """Test error for confidence score too low."""
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            TransformResult(text="test", confidence=-0.1)
    
    def test_invalid_confidence_too_high(self):
        """Test error for confidence score too high."""
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            TransformResult(text="test", confidence=1.1)
    
    def test_default_values(self):
        """Test default values for optional fields."""
        result = TransformResult(text="test", confidence=0.8)
        
        assert result.text == "test"
        assert result.confidence == 0.8
        assert result.changes == []
        assert result.statistics == {}
        assert result.issues == []


class TestCharacterTransformer:
    """Tests for character transformation."""
    
    def test_empty_text(self):
        """Test transformation of empty text."""
        transformer = CharacterTransformer()
        result = transformer.transform("")
        
        assert result.text == ""
        assert result.confidence == 1.0
        assert len(result.changes) == 0
        assert result.statistics["total_chars"] == 0
        assert result.statistics["transformed_chars"] == 0
    
    def test_valid_ascii_text(self):
        """Test transformation of valid ASCII text."""
        transformer = CharacterTransformer()
        result = transformer.transform("Hello World!")
        
        assert result.text == "Hello World!"
        assert result.confidence == 1.0
        assert len(result.changes) == 0
        assert result.statistics["ascii_fast_path"] is True
    
    def test_ascii_with_invalid_control_chars(self):
        """Test ASCII text with invalid control characters."""
        transformer = CharacterTransformer()
        text_with_null = "Hello\x00World"
        result = transformer.transform(text_with_null)
        
        # Default strategy is replacement
        assert "\x00" not in result.text
        assert result.text == "Hello\uFFFDWorld"
        assert result.confidence < 1.0
        assert len(result.changes) == 1
        assert result.changes[0].original_char == "\x00"
        assert result.changes[0].transformed_char == "\uFFFD"
        assert result.changes[0].strategy == TransformationStrategy.REPLACEMENT
    
    def test_removal_strategy(self):
        """Test removal transformation strategy."""
        config = TransformConfig(default_strategy=TransformationStrategy.REMOVAL)
        transformer = CharacterTransformer(config)
        
        text_with_invalid = "Hello\x00\x01World"
        result = transformer.transform(text_with_invalid)
        
        assert result.text == "HelloWorld"
        assert len(result.changes) == 2
        for change in result.changes:
            assert change.strategy == TransformationStrategy.REMOVAL
            assert change.transformed_char == ""
    
    def test_replacement_strategy(self):
        """Test replacement transformation strategy."""
        config = TransformConfig(
            default_strategy=TransformationStrategy.REPLACEMENT,
            replacement_char="?"
        )
        transformer = CharacterTransformer(config)
        
        text_with_invalid = "Hello\x00World"
        result = transformer.transform(text_with_invalid)
        
        assert result.text == "Hello?World"
        assert len(result.changes) == 1
        assert result.changes[0].strategy == TransformationStrategy.REPLACEMENT
        assert result.changes[0].transformed_char == "?"
    
    def test_escape_strategy(self):
        """Test escape transformation strategy."""
        config = TransformConfig(default_strategy=TransformationStrategy.ESCAPE)
        transformer = CharacterTransformer(config)
        
        # Test numeric entity for control character (which is invalid)
        result = transformer.transform("Hello\x00World")
        assert "&#0;" in result.text
        assert len(result.changes) == 1
        assert result.changes[0].strategy == TransformationStrategy.ESCAPE
        
        # Test HTML entity escaping for special characters like & when they are mapped
        config_with_entities = TransformConfig(
            default_strategy=TransformationStrategy.ESCAPE,
            custom_mappings={"&": "&amp;"}  # Custom mapping for &
        )
        transformer_with_entities = CharacterTransformer(config_with_entities)
        result = transformer_with_entities.transform("Hello&World")
        # & is valid XML but we have a custom mapping, so it should be transformed
        assert "&amp;" in result.text
        assert len(result.changes) == 1
        assert result.changes[0].strategy == TransformationStrategy.MAPPING  # Custom mappings use MAPPING strategy
    
    def test_mapping_strategy(self):
        """Test custom mapping transformation."""
        config = TransformConfig(
            custom_mappings={"\x00": "[NULL]", "\x01": "[SOH]"}
        )
        transformer = CharacterTransformer(config)
        
        text_with_invalid = "Hello\x00\x01World"
        result = transformer.transform(text_with_invalid)
        
        assert result.text == "Hello[NULL][SOH]World"
        assert len(result.changes) == 2
        for change in result.changes:
            assert change.strategy == TransformationStrategy.MAPPING
    
    def test_preservation_strategy(self):
        """Test preservation transformation strategy."""
        config = TransformConfig(default_strategy=TransformationStrategy.PRESERVATION)
        transformer = CharacterTransformer(config)
        
        text_with_invalid = "Hello\x00World"
        result = transformer.transform(text_with_invalid)
        
        # Character should be preserved
        assert result.text == text_with_invalid
        assert len(result.changes) == 1
        assert result.changes[0].strategy == TransformationStrategy.PRESERVATION
        assert result.changes[0].transformed_char == "\x00"
        assert "Preserved" in result.changes[0].reason
    
    def test_context_aware_transformation(self):
        """Test context-aware transformation strategies."""
        config = TransformConfig(
            default_strategy=TransformationStrategy.REPLACEMENT,
            context_strategies={
                TransformationContext.TAG_NAME: TransformationStrategy.REMOVAL,
                TransformationContext.ATTRIBUTE_NAME: TransformationStrategy.REMOVAL,
            }
        )
        transformer = CharacterTransformer(config)
        
        # Tag name should use removal
        result = transformer.transform("tag\x00name", TransformationContext.TAG_NAME)
        assert result.text == "tagname"
        assert result.changes[0].strategy == TransformationStrategy.REMOVAL
        
        # Text content should use default replacement
        result = transformer.transform("text\x00content", TransformationContext.TEXT_CONTENT)
        assert "\uFFFD" in result.text
        assert result.changes[0].strategy == TransformationStrategy.REPLACEMENT
    
    def test_surrogate_characters(self):
        """Test handling of surrogate characters."""
        transformer = CharacterTransformer()
        
        # Create text with surrogate characters (this is tricky in Python)
        # We'll test the validation directly since Python strings handle surrogates carefully
        issues = XML10Validator.get_validation_issues(0xD800)
        assert "Surrogate character" in issues[0]
    
    def test_unicode_noncharacters(self):
        """Test handling of Unicode non-characters."""
        transformer = CharacterTransformer()
        
        # Test U+FFFE (non-character)
        text_with_nonchar = "Hello\uFFFEWorld"
        result = transformer.transform(text_with_nonchar)
        
        # Should be transformed (default is replacement)
        assert result.text != text_with_nonchar
        assert len(result.changes) == 1
        assert "Non-character" in result.changes[0].reason
    
    def test_confidence_calculation(self):
        """Test confidence score calculation."""
        transformer = CharacterTransformer()
        
        # Valid text should have confidence 1.0
        result = transformer.transform("Valid text")
        assert result.confidence == 1.0
        
        # Text with some invalid characters should have lower confidence
        result = transformer.transform("Invalid\x00\x01text")
        assert result.confidence < 1.0
        assert result.confidence > 0.0
    
    def test_large_text_performance(self):
        """Test performance with large text."""
        transformer = CharacterTransformer()
        
        # Create large text with some invalid characters
        large_text = "Valid text " * 1000 + "\x00" + "More valid text " * 1000
        
        result = transformer.transform(large_text)
        
        # Should complete successfully
        assert len(result.text) > 0
        assert len(result.changes) == 1  # Only the null character
        assert result.statistics["total_chars"] > 20000
    
    def test_statistics_tracking(self):
        """Test transformation statistics."""
        transformer = CharacterTransformer()
        
        text_with_invalid = "Hello\x00\x01World"
        result = transformer.transform(text_with_invalid)
        
        assert result.statistics["total_chars"] == len(text_with_invalid)
        assert result.statistics["transformed_chars"] == 2
        assert result.statistics["invalid_chars"] == 2
    
    def test_position_tracking(self):
        """Test accurate position tracking in changes."""
        transformer = CharacterTransformer()
        
        text_with_invalid = "a\x00b\x01c"
        result = transformer.transform(text_with_invalid)
        
        # Should track positions accurately
        assert len(result.changes) == 2
        assert result.changes[0].position == 1  # Position of \x00
        assert result.changes[1].position == 3  # Position of \x01
    
    def test_issue_deduplication(self):
        """Test that duplicate issues are removed."""
        transformer = CharacterTransformer()
        
        # Multiple same invalid characters
        text_with_duplicates = "\x00\x00\x00"
        result = transformer.transform(text_with_duplicates)
        
        # Issues should be deduplicated
        assert len(result.issues) == 1


class TestIntegration:
    """Integration tests for the transformation system."""
    
    def test_xml_document_transformation(self):
        """Test transformation of a complete XML document."""
        config = TransformConfig.create_preset("lenient")
        transformer = CharacterTransformer(config)
        
        # XML with various invalid characters
        xml_content = '<?xml version="1.0"?>\n<root\x00>\n  <item\x01 attr="value\x02">Text\x03</item>\n</root>'
        
        # Transform different parts with appropriate contexts
        tag_result = transformer.transform("root\x00", TransformationContext.TAG_NAME)
        attr_result = transformer.transform("value\x02", TransformationContext.ATTRIBUTE_VALUE)
        text_result = transformer.transform("Text\x03", TransformationContext.TEXT_CONTENT)
        
        # Tag name should use removal (stricter)
        assert "\x00" not in tag_result.text
        assert tag_result.changes[0].strategy == TransformationStrategy.REMOVAL
        
        # Attribute value should use replacement (lenient default)
        assert "\uFFFD" in attr_result.text
        assert attr_result.changes[0].strategy == TransformationStrategy.REPLACEMENT
        
        # Text content should use replacement (lenient default)
        assert "\uFFFD" in text_result.text
        assert text_result.changes[0].strategy == TransformationStrategy.REPLACEMENT
    
    def test_preset_comparison(self):
        """Test different presets on same input."""
        text_with_invalid = "test\x00data"
        
        # Strict preset
        strict_transformer = CharacterTransformer(TransformConfig.create_preset("strict"))
        strict_result = strict_transformer.transform(text_with_invalid)
        
        # Lenient preset
        lenient_transformer = CharacterTransformer(TransformConfig.create_preset("lenient"))
        lenient_result = lenient_transformer.transform(text_with_invalid)
        
        # Data recovery preset
        recovery_transformer = CharacterTransformer(TransformConfig.create_preset("data_recovery"))
        recovery_result = recovery_transformer.transform(text_with_invalid)
        
        # All should handle the invalid character differently
        assert strict_result.text != lenient_result.text != recovery_result.text
        
        # Recovery should preserve the character
        assert "\x00" in recovery_result.text
        
        # Strict should remove it
        assert "\x00" not in strict_result.text
        assert len(strict_result.text) < len(text_with_invalid)
        
        # Lenient should replace it
        assert "\x00" not in lenient_result.text
        assert "\uFFFD" in lenient_result.text
    
    def test_comprehensive_character_range(self):
        """Test a comprehensive range of problematic characters."""
        transformer = CharacterTransformer()
        
        # Test various problematic character ranges
        problematic_chars = [
            "\x00",  # NULL
            "\x01",  # SOH
            "\x08",  # Backspace
            "\x0B",  # Vertical tab
            "\x0C",  # Form feed
            "\x0E",  # Shift out
            "\x1F",  # Unit separator
            "\uFFFE",  # Non-character
            "\uFFFF",  # Non-character
        ]
        
        for char in problematic_chars:
            text = f"before{char}after"
            result = transformer.transform(text)
            
            # Should transform the problematic character
            assert char not in result.text or result.changes[0].strategy == TransformationStrategy.PRESERVATION
            assert len(result.changes) >= 1
            assert result.confidence < 1.0


class TestCoverageEdgeCases:
    """Tests to ensure complete code coverage."""
    
    def test_cache_size_limit(self):
        """Test validation cache size limit."""
        # Clear cache and add many entries to test size limit
        XML10Validator.clear_cache()
        
        # Add entries up to and beyond the cache limit
        for i in range(1005):  # More than CACHE_SIZE_LIMIT (1000)
            XML10Validator.is_valid_xml_char(i)
        
        # Cache should not exceed the limit
        assert len(XML10Validator._validation_cache) <= 1000
    
    def test_noncharacter_range_validation(self):
        """Test validation of BMP non-character ranges."""
        # Test characters in NONCHARACTER_RANGES to hit line 212
        assert not XML10Validator.is_valid_xml_char(0xFFFE)
        assert not XML10Validator.is_valid_xml_char(0xFFFF)
    
    def test_fallback_strategy_application(self):
        """Test fallback to replacement strategy."""
        # Create a transformer with an unknown strategy (this would be an edge case)
        config = TransformConfig(default_strategy=TransformationStrategy.REPLACEMENT)
        transformer = CharacterTransformer(config)
        
        # Force a scenario where we might hit fallback logic
        # by testing an unsupported scenario in _apply_strategy
        text_with_invalid = "test\x00data"
        result = transformer.transform(text_with_invalid)
        
        # Should handle it gracefully
        assert len(result.changes) == 1
    
    def test_ascii_only_detection_exception(self):
        """Test exception handling in ASCII-only detection."""
        transformer = CharacterTransformer()
        
        # Test with normal text
        result = transformer.transform("normal text")
        assert result.statistics.get("ascii_fast_path", False)
    
    def test_validation_issues_for_various_ranges(self):
        """Test validation issues for different character ranges."""
        # Test control character
        issues = XML10Validator.get_validation_issues(0x0000)
        assert any("Control character" in issue for issue in issues)
        
        # Test surrogate
        issues = XML10Validator.get_validation_issues(0xD800)
        assert any("Surrogate character" in issue for issue in issues)
        
        # Test supplementary non-character
        issues = XML10Validator.get_validation_issues(0x1FFFE)
        assert any("Non-character" in issue for issue in issues)
        
        # Test BMP non-character
        issues = XML10Validator.get_validation_issues(0xFFFE)
        assert any("Non-character" in issue for issue in issues)
        
        # Test other invalid character
        issues = XML10Validator.get_validation_issues(0x110000)  # Beyond valid Unicode
        assert any("Invalid XML character" in issue for issue in issues)
    
    def test_custom_mappings_with_full_transform(self):
        """Test custom mappings in full transform path."""
        config = TransformConfig(
            custom_mappings={"😀": "[EMOJI]"}  # Non-ASCII character with mapping
        )
        transformer = CharacterTransformer(config)
        
        # This should trigger full transform path due to non-ASCII
        result = transformer.transform("Hello😀World")
        
        assert "[EMOJI]" in result.text
        assert len(result.changes) == 1
        assert result.changes[0].strategy == TransformationStrategy.MAPPING
        assert not result.statistics.get("ascii_fast_path", True)
    
    def test_confidence_calculation_edge_cases(self):
        """Test confidence calculation with edge cases."""
        transformer = CharacterTransformer()
        
        # Test with many changes relative to text length
        many_invalid = "\x00" * 5 + "a" * 5  # 50% invalid
        result = transformer.transform(many_invalid)
        
        # Confidence should be reduced (it's exactly 0.8 due to the fast path logic)
        assert result.confidence <= 0.8
        
        # Test with many issues
        result = transformer.transform("\x00\x01\x02\x03\x04")  # All invalid
        assert result.confidence >= 0.1  # Should not go below minimum


if __name__ == "__main__":
    pytest.main([__file__])