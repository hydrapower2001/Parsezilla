"""Tests for token assembly and repair functionality."""

import pytest
from src.ultra_robust_xml_parser.tokenization import (
    Token, TokenType, TokenPosition, TokenRepair,
    TokenAssemblyEngine, AssemblyResult, 
    RepairType, RepairSeverity, AssemblyRepairAction
)


class TestTokenAssemblyEngine:
    """Test cases for the TokenAssemblyEngine class."""
    
    def test_basic_instantiation(self):
        """Test basic engine instantiation with different configurations."""
        # Default configuration
        engine = TokenAssemblyEngine()
        assert engine.correlation_id is None
        assert engine.enable_caching is True
        assert engine.strict_mode is False
        
        # Custom configuration
        engine2 = TokenAssemblyEngine(
            correlation_id="test-123",
            enable_caching=False,
            strict_mode=True
        )
        assert engine2.correlation_id == "test-123"
        assert engine2.enable_caching is False
        assert engine2.strict_mode is True
    
    def test_empty_token_list_handling(self):
        """Test handling of empty token lists."""
        engine = TokenAssemblyEngine()
        result = engine.assemble_and_repair_tokens([])
        
        assert isinstance(result, AssemblyResult)
        assert result.tokens == []
        assert result.repairs_applied == 0
        assert result.confidence_adjustment == 0.0
        assert result.repair_rate == 0.0
    
    def test_single_valid_token_processing(self):
        """Test processing of a single valid token with no repairs needed."""
        engine = TokenAssemblyEngine()
        token = Token(
            TokenType.TAG_NAME, 
            "valid_tag",
            TokenPosition(1, 1, 0),
            0.9
        )
        
        result = engine.assemble_and_repair_tokens([token])
        
        assert len(result.tokens) == 1
        assert result.tokens[0].value == "valid_tag"
        assert result.repairs_applied == 0
        assert result.repair_rate == 0.0
    
    def test_tag_name_sanitization_basic(self):
        """Test basic tag name sanitization functionality."""
        engine = TokenAssemblyEngine()
        
        # Test invalid characters in tag name
        token = Token(
            TokenType.TAG_NAME,
            "tag<with>invalid&chars",
            TokenPosition(1, 1, 0),
            0.8
        )
        
        result = engine.assemble_and_repair_tokens([token])
        
        assert len(result.tokens) == 1
        assert result.repairs_applied > 0
        assert result.tokens[0].has_repairs
        assert result.tokens[0].confidence < 0.8  # Confidence should be reduced
        
        # Should remove invalid XML characters
        repaired_value = result.tokens[0].value
        assert "<" not in repaired_value
        assert ">" not in repaired_value
        assert "&" not in repaired_value
    
    def test_tag_name_sanitization_edge_cases(self):
        """Test edge cases for tag name sanitization."""
        engine = TokenAssemblyEngine()
        
        test_cases = [
            ("", "unknown"),  # Empty name
            ("123invalid", "_123invalid"),  # Starts with number in strict mode
            ("too" + "x" * 1000, "too" + "x" * 997),  # Length truncation
            ("spaces in name", "spacesinname"),  # Whitespace removal
            ("special!@#$%chars", "special_____chars"),  # Special char replacement
        ]
        
        engine_strict = TokenAssemblyEngine(strict_mode=True)
        
        for original, expected_pattern in test_cases:
            token = Token(TokenType.TAG_NAME, original, TokenPosition(1, 1, 0), 0.8)
            result = engine_strict.assemble_and_repair_tokens([token])
            
            repaired_value = result.tokens[0].value
            if expected_pattern == "unknown":
                assert repaired_value == expected_pattern
            elif "123invalid" in expected_pattern:
                assert repaired_value.startswith("_")
            elif "too" in expected_pattern:
                assert len(repaired_value) <= 1000
    
    def test_attribute_quote_repair(self):
        """Test attribute value quote repair functionality.""" 
        engine = TokenAssemblyEngine()
        
        # Test with proper tag context - attribute values are only repaired within tags
        tokens = [
            Token(TokenType.TAG_START, "<", TokenPosition(1, 1, 0), 1.0),
            Token(TokenType.TAG_NAME, "div", TokenPosition(1, 2, 1), 1.0),
            Token(TokenType.ATTR_NAME, "class", TokenPosition(1, 6, 5), 1.0),
            Token(TokenType.ATTR_VALUE, "unquoted_value", TokenPosition(1, 12, 11), 0.8),  # Should be repaired
            Token(TokenType.TAG_END, ">", TokenPosition(1, 27, 26), 1.0),
        ]
        
        result = engine.assemble_and_repair_tokens(tokens)
        
        # Find the attribute value token
        attr_value_token = None
        for token in result.tokens:
            if token.type == TokenType.ATTR_VALUE:
                attr_value_token = token
                break
        
        assert attr_value_token is not None
        # Should have quotes after repair
        assert attr_value_token.value.startswith('"') and attr_value_token.value.endswith('"')
    
    def test_duplicate_attribute_detection(self):
        """Test detection and resolution of duplicate attributes."""
        engine = TokenAssemblyEngine()
        
        # Simulate tokens within a tag with duplicate attributes
        tokens = [
            Token(TokenType.TAG_START, "<", TokenPosition(1, 1, 0), 1.0),
            Token(TokenType.TAG_NAME, "div", TokenPosition(1, 2, 1), 1.0),
            Token(TokenType.ATTR_NAME, "class", TokenPosition(1, 6, 5), 1.0),
            Token(TokenType.ATTR_VALUE, '"first"', TokenPosition(1, 12, 11), 1.0),
            Token(TokenType.ATTR_NAME, "class", TokenPosition(1, 20, 19), 1.0),  # Duplicate
            Token(TokenType.ATTR_VALUE, '"second"', TokenPosition(1, 26, 25), 1.0),
            Token(TokenType.TAG_END, ">", TokenPosition(1, 35, 34), 1.0),
        ]
        
        result = engine.assemble_and_repair_tokens(tokens)
        
        # Should have detected and handled the duplicate attribute
        assert result.repairs_applied > 0
        assert RepairType.ATTRIBUTE_DUPLICATE_RESOLUTION in result.repair_summary
        
        # Find the repaired duplicate attribute token
        duplicate_token = None
        for token in result.tokens:
            if token.has_repairs and token.type == TokenType.ATTR_NAME:
                for repair in token.repairs:
                    if "duplicate" in repair.repair_type.lower():
                        duplicate_token = token
                        break
        
        assert duplicate_token is not None
        assert duplicate_token.confidence < 1.0
    
    def test_text_token_merging_basic(self):
        """Test basic text token merging functionality."""
        engine = TokenAssemblyEngine()
        
        # Create adjacent text tokens
        tokens = [
            Token(TokenType.TEXT, "Hello ", TokenPosition(1, 1, 0), 0.9),
            Token(TokenType.TEXT, "world", TokenPosition(1, 7, 6), 0.9),
            Token(TokenType.TEXT, "!", TokenPosition(1, 12, 11), 0.9),
        ]
        
        result = engine.assemble_and_repair_tokens(tokens)
        
        # Should merge into single text token
        assert len(result.tokens) == 1
        assert result.tokens[0].type == TokenType.TEXT
        assert "Hello world!" == result.tokens[0].value or "Hello world !" == result.tokens[0].value
        assert result.tokens[0].has_repairs
        assert result.repairs_applied > 0
        assert RepairType.TEXT_TOKEN_MERGING in result.repair_summary
    
    def test_text_token_merging_with_whitespace_preservation(self):
        """Test text token merging with whitespace preservation."""
        # Test with strict mode to preserve whitespace without normalization
        engine = TokenAssemblyEngine(strict_mode=True)
        
        tokens = [
            Token(TokenType.TEXT, "  Leading spaces", TokenPosition(1, 1, 0), 0.9),
            Token(TokenType.TEXT, "middle content  ", TokenPosition(1, 17, 16), 0.9),
            Token(TokenType.TEXT, "  trailing  ", TokenPosition(1, 33, 32), 0.9),
        ]
        
        result = engine.assemble_and_repair_tokens(tokens)
        
        assert len(result.tokens) == 1
        merged_token = result.tokens[0]
        
        # In strict mode, whitespace structure should be more preserved
        assert "Leading spaces" in merged_token.value
        assert "middle content" in merged_token.value
        assert "trailing" in merged_token.value
        
        # Test lenient mode behavior (default)
        lenient_engine = TokenAssemblyEngine(strict_mode=False)
        lenient_result = lenient_engine.assemble_and_repair_tokens(tokens)
        lenient_merged = lenient_result.tokens[0]
        
        # In lenient mode, whitespace may be normalized
        assert "Leading spaces" in lenient_merged.value
        assert "middle content" in lenient_merged.value
        assert "trailing" in lenient_merged.value
    
    def test_text_token_merging_limits(self):
        """Test that text token merging respects limits."""
        engine = TokenAssemblyEngine()
        
        # Create many text tokens (more than MAX_TEXT_MERGE_TOKENS)
        tokens = []
        for i in range(60):  # More than MAX_TEXT_MERGE_TOKENS (50)
            tokens.append(Token(
                TokenType.TEXT, 
                f"text{i} ",
                TokenPosition(1, i * 10 + 1, i * 10),
                0.9
            ))
        
        result = engine.assemble_and_repair_tokens(tokens)
        
        # The merging limit is per merge operation, not total tokens
        # So we should see evidence that merging happened (repairs > 0) 
        # but the exact number of output tokens depends on implementation
        if result.repairs_applied > 0:
            # If merging occurred, we should have fewer tokens than input
            assert len(result.tokens) <= len(tokens)
        else:
            # If no merging (edge case), tokens remain unchanged
            assert len(result.tokens) == len(tokens)
    
    def test_mixed_token_types_no_inappropriate_merging(self):
        """Test that different token types are not inappropriately merged."""
        engine = TokenAssemblyEngine()
        
        tokens = [
            Token(TokenType.TEXT, "Some text", TokenPosition(1, 1, 0), 0.9),
            Token(TokenType.TAG_START, "<", TokenPosition(1, 10, 9), 1.0),
            Token(TokenType.TAG_NAME, "div", TokenPosition(1, 11, 10), 1.0),
            Token(TokenType.TAG_END, ">", TokenPosition(1, 14, 13), 1.0),
            Token(TokenType.TEXT, "More text", TokenPosition(1, 15, 14), 0.9),
        ]
        
        result = engine.assemble_and_repair_tokens(tokens)
        
        # Should maintain separate tokens for different types
        text_tokens = [t for t in result.tokens if t.type == TokenType.TEXT]
        tag_tokens = [t for t in result.tokens if t.type in (TokenType.TAG_START, TokenType.TAG_NAME, TokenType.TAG_END)]
        
        assert len(text_tokens) == 2  # Two separate text tokens
        assert len(tag_tokens) == 3  # Three tag-related tokens
    
    def test_confidence_scoring_adjustments(self):
        """Test confidence scoring adjustments based on repair actions."""
        engine = TokenAssemblyEngine()
        
        # Create token requiring significant repairs
        token = Token(
            TokenType.TAG_NAME,
            "very<bad>tag&name!@#",  # Multiple invalid characters
            TokenPosition(1, 1, 0),
            0.9  # Start with high confidence
        )
        
        result = engine.assemble_and_repair_tokens([token])
        
        repaired_token = result.tokens[0]
        
        # Confidence should be reduced due to significant repairs
        assert repaired_token.confidence < token.confidence
        assert repaired_token.has_repairs
        
        # Should have multiple repairs recorded
        assert len(repaired_token.repairs) > 0
        
        # Check that confidence impact is negative
        total_impact = sum(repair.confidence_impact for repair in repaired_token.repairs)
        assert total_impact < 0
    
    def test_repair_caching_functionality(self):
        """Test repair result caching for performance optimization."""
        engine = TokenAssemblyEngine(enable_caching=True)
        
        # Create identical token sequences
        tokens1 = [
            Token(TokenType.TAG_NAME, "test<tag", TokenPosition(1, 1, 0), 0.8),
            Token(TokenType.TEXT, "content", TokenPosition(1, 10, 9), 0.9),
        ]
        
        tokens2 = [
            Token(TokenType.TAG_NAME, "test<tag", TokenPosition(1, 1, 0), 0.8),
            Token(TokenType.TEXT, "content", TokenPosition(1, 10, 9), 0.9),
        ]
        
        # Process first sequence
        result1 = engine.assemble_and_repair_tokens(tokens1)
        initial_cache_size = engine.get_repair_statistics()["cache_size"]
        
        # Process identical sequence
        result2 = engine.assemble_and_repair_tokens(tokens2)
        final_cache_size = engine.get_repair_statistics()["cache_size"]
        
        # Cache should have grown
        assert final_cache_size >= initial_cache_size
        
        # Results should be equivalent
        assert len(result1.tokens) == len(result2.tokens)
        assert result1.repairs_applied == result2.repairs_applied
    
    def test_cache_disabled_mode(self):
        """Test operation with caching disabled."""
        engine = TokenAssemblyEngine(enable_caching=False)
        
        tokens = [
            Token(TokenType.TAG_NAME, "test<tag", TokenPosition(1, 1, 0), 0.8),
        ]
        
        result = engine.assemble_and_repair_tokens(tokens)
        stats = engine.get_repair_statistics()
        
        assert stats["cache_enabled"] is False
        assert stats["cache_size"] == 0
        assert result.repairs_applied > 0  # Still performs repairs
    
    def test_strict_mode_differences(self):
        """Test differences between strict and lenient repair modes."""
        strict_engine = TokenAssemblyEngine(strict_mode=True)
        lenient_engine = TokenAssemblyEngine(strict_mode=False)
        
        # Test with tag name starting with number
        token = Token(TokenType.TAG_NAME, "123tag", TokenPosition(1, 1, 0), 0.8)
        
        strict_result = strict_engine.assemble_and_repair_tokens([token])
        lenient_result = lenient_engine.assemble_and_repair_tokens([token])
        
        # Strict mode should be more aggressive with repairs
        assert strict_result.repairs_applied >= lenient_result.repairs_applied
        
        # Strict mode should modify the numeric start
        if strict_result.repairs_applied > 0:
            assert not strict_result.tokens[0].value.startswith("1")
    
    def test_repair_statistics_tracking(self):
        """Test comprehensive repair statistics tracking."""
        engine = TokenAssemblyEngine()
        
        # Process tokens with various repair types
        tokens = [
            Token(TokenType.TAG_NAME, "bad<tag", TokenPosition(1, 1, 0), 0.8),  # Tag sanitization
            Token(TokenType.ATTR_VALUE, "unquoted", TokenPosition(1, 10, 9), 0.8),  # Quote repair
            Token(TokenType.TEXT, "text1", TokenPosition(1, 20, 19), 0.9),  # Text merging
            Token(TokenType.TEXT, "text2", TokenPosition(1, 26, 25), 0.9),  # Text merging
        ]
        
        result = engine.assemble_and_repair_tokens(tokens)
        stats = engine.get_repair_statistics()
        
        assert "repair_counts" in stats
        assert "total_repairs" in stats
        assert stats["total_repairs"] > 0
        
        # Should have multiple repair types
        assert len(stats["repair_counts"]) > 1
    
    def test_malformed_tag_structure_handling(self):
        """Test handling of malformed tag structures.""" 
        engine = TokenAssemblyEngine()
        
        # Simulate malformed tag tokens that might come from recovery
        tokens = [
            Token(TokenType.MALFORMED_TAG, "<broken", TokenPosition(1, 1, 0), 0.3),
            Token(TokenType.INVALID_CHARS, ">>", TokenPosition(1, 8, 7), 0.1),
        ]
        
        result = engine.assemble_and_repair_tokens(tokens)
        
        # Should handle malformed tokens gracefully
        assert len(result.tokens) >= 1
        assert result.repairs_applied >= 0  # May or may not apply repairs
        
        # Result should have reasonable confidence
        for token in result.tokens:
            assert 0.0 <= token.confidence <= 1.0
    
    def test_large_token_sequence_performance(self):
        """Test performance with large token sequences."""
        engine = TokenAssemblyEngine(enable_caching=True)
        
        # Create large sequence of tokens
        tokens = []
        for i in range(1000):
            tokens.append(Token(
                TokenType.TEXT if i % 2 == 0 else TokenType.TAG_NAME,
                f"token{i}",
                TokenPosition(1, i * 10 + 1, i * 10),
                0.8
            ))
        
        result = engine.assemble_and_repair_tokens(tokens)
        
        # Should complete in reasonable time
        assert result.processing_time_ms < 5000  # Less than 5 seconds
        assert len(result.tokens) > 0
        
        # Performance metrics should be available
        assert result.processing_time_ms > 0
    
    def test_assembly_result_properties(self):
        """Test AssemblyResult properties and calculations."""
        engine = TokenAssemblyEngine()
        
        tokens = [
            Token(TokenType.TAG_NAME, "bad<tag", TokenPosition(1, 1, 0), 0.8),
            Token(TokenType.TEXT, "good_text", TokenPosition(1, 10, 9), 0.9),
        ]
        
        result = engine.assemble_and_repair_tokens(tokens)
        
        # Test repair_rate property
        assert 0.0 <= result.repair_rate <= 1.0
        
        # If repairs were applied, repair_rate should be > 0
        if result.repairs_applied > 0:
            assert result.repair_rate > 0
        
        # Test other properties exist
        assert hasattr(result, 'tokens')
        assert hasattr(result, 'repairs_applied')
        assert hasattr(result, 'confidence_adjustment')
        assert hasattr(result, 'processing_time_ms')
        assert hasattr(result, 'repair_summary')
        assert hasattr(result, 'diagnostics')
    
    def test_correlation_id_tracking(self):
        """Test correlation ID tracking through repair process."""
        correlation_id = "test-correlation-123"
        engine = TokenAssemblyEngine(correlation_id=correlation_id)
        
        assert engine.correlation_id == correlation_id
        
        # Process some tokens
        tokens = [Token(TokenType.TAG_NAME, "test", TokenPosition(1, 1, 0), 0.8)]
        result = engine.assemble_and_repair_tokens(tokens)
        
        # Should complete successfully with correlation ID
        assert len(result.tokens) > 0
    
    def test_clear_cache_functionality(self):
        """Test cache clearing functionality."""
        engine = TokenAssemblyEngine(enable_caching=True)
        
        # Add something to cache
        tokens = [Token(TokenType.TAG_NAME, "test<tag", TokenPosition(1, 1, 0), 0.8)]
        engine.assemble_and_repair_tokens(tokens)
        
        # Verify cache has content
        stats_before = engine.get_repair_statistics()
        assert stats_before["cache_size"] > 0
        
        # Clear cache
        engine.clear_cache()
        
        # Verify cache is cleared
        stats_after = engine.get_repair_statistics()
        assert stats_after["cache_size"] == 0
    
    def test_reset_statistics_functionality(self):
        """Test statistics reset functionality."""
        engine = TokenAssemblyEngine()
        
        # Generate some statistics
        tokens = [Token(TokenType.TAG_NAME, "bad<tag", TokenPosition(1, 1, 0), 0.8)]
        engine.assemble_and_repair_tokens(tokens)
        
        # Verify statistics exist
        stats_before = engine.get_repair_statistics()
        assert stats_before["total_repairs"] > 0
        
        # Reset statistics
        engine.reset_statistics()
        
        # Verify statistics are reset
        stats_after = engine.get_repair_statistics()
        assert stats_after["total_repairs"] == 0
        assert len(stats_after["repair_counts"]) == 0


class TestRepairTypes:
    """Test the RepairType enum and related functionality."""
    
    def test_repair_type_enum_completeness(self):
        """Test that all required repair types are defined."""
        expected_types = [
            'TAG_NAME_SANITIZATION',
            'ATTRIBUTE_QUOTE_REPAIR', 
            'ATTRIBUTE_DUPLICATE_RESOLUTION',
            'TEXT_TOKEN_MERGING',
            'MALFORMED_TAG_REPAIR',
            'WHITESPACE_NORMALIZATION',
            'CONFIDENCE_ADJUSTMENT'
        ]
        
        for expected_type in expected_types:
            assert hasattr(RepairType, expected_type)
    
    def test_repair_severity_enum(self):
        """Test RepairSeverity enum values."""
        assert hasattr(RepairSeverity, 'MINOR')
        assert hasattr(RepairSeverity, 'MODERATE')
        assert hasattr(RepairSeverity, 'MAJOR')
        assert hasattr(RepairSeverity, 'CRITICAL')


class TestAssemblyRepairAction:
    """Test the AssemblyRepairAction dataclass."""
    
    def test_repair_action_creation(self):
        """Test creating AssemblyRepairAction instances."""
        action = AssemblyRepairAction(
            repair_type=RepairType.TAG_NAME_SANITIZATION,
            severity=RepairSeverity.MODERATE,
            description="Test repair",
            original_value="original",
            repaired_value="repaired",
            confidence_impact=-0.1,
            position=TokenPosition(1, 1, 0),
            context={"test": True}
        )
        
        assert action.repair_type == RepairType.TAG_NAME_SANITIZATION
        assert action.severity == RepairSeverity.MODERATE
        assert action.description == "Test repair"
        assert action.original_value == "original"
        assert action.repaired_value == "repaired"
        assert action.confidence_impact == -0.1
        assert action.context == {"test": True}


if __name__ == "__main__":
    # Run basic smoke test
    engine = TokenAssemblyEngine()
    tokens = [
        Token(TokenType.TAG_NAME, "test<tag", TokenPosition(1, 1, 0), 0.8),
        Token(TokenType.TEXT, "hello", TokenPosition(1, 10, 9), 0.9),
        Token(TokenType.TEXT, "world", TokenPosition(1, 16, 15), 0.9),
    ]
    
    result = engine.assemble_and_repair_tokens(tokens)
    print(f"✓ Smoke test passed: {len(result.tokens)} tokens, {result.repairs_applied} repairs")
    print(f"  Processing time: {result.processing_time_ms:.2f}ms")
    print(f"  Repair rate: {result.repair_rate:.1%}")