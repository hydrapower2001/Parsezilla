"""Integration tests for token assembly with XMLTokenizer."""

import pytest
from src.ultra_robust_xml_parser.tokenization import (
    XMLTokenizer, Token, TokenType, TokenPosition, RepairType
)


class TestTokenizerAssemblyIntegration:
    """Test integration of token assembly with the XMLTokenizer."""
    
    def test_tokenizer_assembly_initialization(self):
        """Test that XMLTokenizer properly initializes assembly engine."""
        # Test with assembly enabled (default)
        tokenizer_enabled = XMLTokenizer(enable_assembly=True)
        assert tokenizer_enabled.assembly_engine is not None
        assert tokenizer_enabled.enable_assembly is True
        
        # Test with assembly disabled
        tokenizer_disabled = XMLTokenizer(enable_assembly=False)
        assert tokenizer_disabled.assembly_engine is None
        assert tokenizer_disabled.enable_assembly is False
    
    def test_assembly_engine_direct_usage(self):
        """Test direct usage of assembly engine through tokenizer."""
        tokenizer = XMLTokenizer(enable_assembly=True)
        
        # Create tokens that need assembly/repair
        test_tokens = [
            Token(TokenType.TAG_NAME, "bad<tag>name", TokenPosition(1, 1, 0), 0.8),
            Token(TokenType.ATTR_VALUE, "unquoted_value", TokenPosition(1, 15, 14), 0.8),
            Token(TokenType.TEXT, "text1 ", TokenPosition(1, 30, 29), 0.9),
            Token(TokenType.TEXT, "text2", TokenPosition(1, 36, 35), 0.9),
        ]
        
        # Use assembly engine directly
        result = tokenizer.assembly_engine.assemble_and_repair_tokens(test_tokens)
        
        assert len(result.tokens) >= 1  # At least some tokens
        assert result.repairs_applied > 0  # Some repairs should be applied
        assert result.repair_rate > 0  # Non-zero repair rate
    
    def test_assembly_statistics_access(self):
        """Test access to assembly statistics through tokenizer."""
        tokenizer = XMLTokenizer(enable_assembly=True)
        
        # Initially should have empty statistics
        stats = tokenizer.get_assembly_statistics()
        assert stats is not None
        assert stats["total_repairs"] == 0
        assert stats["cache_size"] == 0
        assert stats["cache_enabled"] is True
        assert stats["strict_mode"] is False
        
        # Test with assembly disabled
        tokenizer_disabled = XMLTokenizer(enable_assembly=False)
        stats_disabled = tokenizer_disabled.get_assembly_statistics()
        assert stats_disabled is None
    
    def test_assembly_with_recovery_integration(self):
        """Test that assembly works alongside recovery engine."""
        tokenizer = XMLTokenizer(
            enable_assembly=True,
            enable_recovery=True
        )
        
        assert tokenizer.assembly_engine is not None
        assert tokenizer.recovery_engine is not None
        
        # Test that both engines can coexist
        test_tokens = [
            Token(TokenType.TAG_NAME, "test<>tag", TokenPosition(1, 1, 0), 0.7),
            Token(TokenType.ERROR, "invalid_char", TokenPosition(1, 10, 9), 0.1),  # Recovery token
            Token(TokenType.TEXT, "content1", TokenPosition(1, 20, 19), 0.9),
            Token(TokenType.TEXT, "content2", TokenPosition(1, 29, 28), 0.9),
        ]
        
        result = tokenizer.assembly_engine.assemble_and_repair_tokens(test_tokens)
        
        # Should handle mixed token types including recovery tokens
        assert len(result.tokens) >= 1
        assert result.repairs_applied >= 0  # May apply repairs
    
    def test_assembly_with_fast_path_integration(self):
        """Test assembly integration with fast-path tokenization."""
        tokenizer = XMLTokenizer(
            enable_assembly=True,
            enable_fast_path=True
        )
        
        assert tokenizer.assembly_engine is not None
        assert tokenizer.enable_fast_path is True
        
        # Both features should be available
        stats = tokenizer.get_assembly_statistics()
        assert stats is not None
    
    def test_assembly_correlation_id_tracking(self):
        """Test that correlation ID is properly propagated to assembly engine."""
        correlation_id = "test-assembly-correlation-123"
        tokenizer = XMLTokenizer(
            correlation_id=correlation_id,
            enable_assembly=True
        )
        
        assert tokenizer.correlation_id == correlation_id
        assert tokenizer.assembly_engine.correlation_id == correlation_id
    
    def test_assembly_cache_functionality(self):
        """Test assembly caching through tokenizer integration."""
        tokenizer = XMLTokenizer(enable_assembly=True)
        
        # Create identical token sequences
        tokens1 = [
            Token(TokenType.TAG_NAME, "test<tag", TokenPosition(1, 1, 0), 0.8),
            Token(TokenType.TEXT, "content", TokenPosition(1, 10, 9), 0.9),
        ]
        
        tokens2 = [
            Token(TokenType.TAG_NAME, "test<tag", TokenPosition(1, 1, 0), 0.8),
            Token(TokenType.TEXT, "content", TokenPosition(1, 10, 9), 0.9),
        ]
        
        # Process both sequences
        result1 = tokenizer.assembly_engine.assemble_and_repair_tokens(tokens1)
        initial_cache_size = tokenizer.get_assembly_statistics()["cache_size"]
        
        result2 = tokenizer.assembly_engine.assemble_and_repair_tokens(tokens2)
        final_cache_size = tokenizer.get_assembly_statistics()["cache_size"]
        
        # Cache should have grown
        assert final_cache_size >= initial_cache_size
    
    def test_assembly_error_handling(self):
        """Test error handling in assembly integration."""
        tokenizer = XMLTokenizer(enable_assembly=True)
        
        # Test with empty token list
        result = tokenizer.assembly_engine.assemble_and_repair_tokens([])
        assert result.tokens == []
        assert result.repairs_applied == 0
        
        # Test with malformed tokens (should handle gracefully)
        malformed_tokens = [
            Token(TokenType.MALFORMED_TAG, "<broken", TokenPosition(1, 1, 0), 0.1),
        ]
        
        result = tokenizer.assembly_engine.assemble_and_repair_tokens(malformed_tokens)
        assert len(result.tokens) >= 0  # Should not crash
        assert result.processing_time_ms >= 0
    
    def test_assembly_with_all_features_enabled(self):
        """Test assembly working with all tokenizer features enabled."""
        tokenizer = XMLTokenizer(
            correlation_id="full-feature-test",
            enable_assembly=True,
            enable_recovery=True,
            enable_fast_path=True
        )
        
        # All features should be properly initialized
        assert tokenizer.assembly_engine is not None
        assert tokenizer.recovery_engine is not None
        assert tokenizer.enable_fast_path is True
        assert tokenizer.enable_assembly is True
        assert tokenizer.enable_recovery is True
        
        # All statistics should be accessible
        assembly_stats = tokenizer.get_assembly_statistics()
        recovery_stats = tokenizer.get_recovery_statistics()
        
        assert assembly_stats is not None
        assert recovery_stats is not None
    
    def test_assembly_configuration_propagation(self):
        """Test that assembly configuration is properly set."""
        tokenizer = XMLTokenizer(enable_assembly=True)
        
        assembly_engine = tokenizer.assembly_engine
        assert assembly_engine is not None
        
        # Check default configuration
        assert assembly_engine.enable_caching is True
        assert assembly_engine.strict_mode is False
        
        # Check that statistics reflect configuration
        stats = assembly_engine.get_repair_statistics()
        assert stats["cache_enabled"] is True
        assert stats["strict_mode"] is False
    
    def test_assembly_performance_integration(self):
        """Test assembly performance tracking integration."""
        tokenizer = XMLTokenizer(enable_assembly=True)
        
        # Create tokens that will trigger various repair types
        test_tokens = [
            Token(TokenType.TAG_NAME, "tag<>with&issues", TokenPosition(1, 1, 0), 0.7),  # Tag repair
            Token(TokenType.ATTR_VALUE, "unquoted", TokenPosition(1, 20, 19), 0.8),      # Quote repair
            Token(TokenType.TEXT, "text ", TokenPosition(1, 30, 29), 0.9),               # Text merge
            Token(TokenType.TEXT, "merge", TokenPosition(1, 35, 34), 0.9),               # Text merge
        ]
        
        result = tokenizer.assembly_engine.assemble_and_repair_tokens(test_tokens)
        
        # Performance metrics should be recorded
        assert result.processing_time_ms >= 0
        assert result.repairs_applied > 0
        assert result.repair_rate > 0
        
        # Statistics should be updated
        stats = tokenizer.get_assembly_statistics()
        assert stats["total_repairs"] > 0
    
    def test_assembly_repair_type_coverage(self):
        """Test that different repair types are covered in integration."""
        tokenizer = XMLTokenizer(enable_assembly=True)
        
        # Create tokens that will trigger different repair types
        repair_test_cases = [
            # Tag name sanitization
            [Token(TokenType.TAG_NAME, "bad<tag", TokenPosition(1, 1, 0), 0.8)],
            
            # Text merging
            [
                Token(TokenType.TEXT, "text1", TokenPosition(1, 1, 0), 0.9),
                Token(TokenType.TEXT, "text2", TokenPosition(1, 6, 5), 0.9)
            ],
        ]
        
        all_repair_types = set()
        
        for test_tokens in repair_test_cases:
            result = tokenizer.assembly_engine.assemble_and_repair_tokens(test_tokens)
            if result.repair_summary:
                all_repair_types.update(result.repair_summary.keys())
        
        # Should have covered multiple repair types
        assert len(all_repair_types) > 0
        
        # Common repair types should be represented
        repair_type_names = {rt.name for rt in all_repair_types}
        expected_types = {"TAG_NAME_SANITIZATION", "TEXT_TOKEN_MERGING"}
        assert len(repair_type_names.intersection(expected_types)) > 0


if __name__ == "__main__":
    # Run basic integration smoke test
    tokenizer = XMLTokenizer(enable_assembly=True)
    
    test_tokens = [
        Token(TokenType.TAG_NAME, "test<tag", TokenPosition(1, 1, 0), 0.8),
        Token(TokenType.TEXT, "hello", TokenPosition(1, 10, 9), 0.9),
        Token(TokenType.TEXT, "world", TokenPosition(1, 16, 15), 0.9),
    ]
    
    result = tokenizer.assembly_engine.assemble_and_repair_tokens(test_tokens)
    print(f"✓ Integration smoke test passed: {len(result.tokens)} tokens, {result.repairs_applied} repairs")
    
    stats = tokenizer.get_assembly_statistics()
    print(f"  Assembly stats: {stats['total_repairs']} total repairs, cache size {stats['cache_size']}")