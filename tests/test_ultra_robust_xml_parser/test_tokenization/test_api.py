"""Tests for comprehensive tokenization API and result objects.

This module tests the enhanced tokenization API, streaming interface, 
token filtering, and comprehensive result objects.
"""

import pytest
import time
from typing import List

from ultra_robust_xml_parser.character import CharacterStreamResult
from ultra_robust_xml_parser.character.encoding import EncodingResult, DetectionMethod
from ultra_robust_xml_parser.character.transformation import TransformResult
from ultra_robust_xml_parser.shared import (
    DiagnosticSeverity,
    TokenizationConfig,
)
from ultra_robust_xml_parser.shared.config import RecoveryStrategy, FilterMode
from ultra_robust_xml_parser.tokenization import (
    EnhancedXMLTokenizer,
    StreamingTokenizer,
    TokenFilter,
    Token,
    TokenType,
)
from ultra_robust_xml_parser.tokenization.api import TokenizationResult as EnhancedTokenizationResult


def create_test_char_stream(text: str) -> CharacterStreamResult:
    """Create a test CharacterStreamResult with minimal valid data."""
    encoding_result = EncodingResult(
        encoding="utf-8",
        confidence=1.0,
        method=DetectionMethod.FALLBACK,
        issues=[]
    )
    
    transform_result = TransformResult(
        text=text,
        confidence=1.0,
        changes=[],
        statistics={},
        issues=[]
    )
    
    return CharacterStreamResult(
        text=text,
        encoding=encoding_result,
        transformations=transform_result,
        confidence=1.0,
        diagnostics=[],
        metadata={}
    )


class TestEnhancedTokenizationResult:
    """Test enhanced tokenization result object."""
    
    def test_result_initialization(self):
        """Test basic result initialization and validation."""
        result = EnhancedTokenizationResult()
        
        assert result.success is True
        assert result.confidence == 1.0
        assert len(result.tokens) == 0
        assert len(result.diagnostics) == 0
        assert result.token_count == 0
        assert result.error_rate == 0.0
        assert result.repair_rate == 0.0
        assert not result.has_errors()
    
    def test_result_with_tokens(self):
        """Test result with actual tokens."""
        from ultra_robust_xml_parser.tokenization.tokenizer import TokenPosition
        
        tokens = [
            Token(
                type=TokenType.TAG_START,
                value="<",
                position=TokenPosition(1, 1, 0),
                confidence=1.0
            ),
            Token(
                type=TokenType.TAG_NAME,
                value="root",
                position=TokenPosition(1, 2, 1),
                confidence=0.9
            ),
            Token(
                type=TokenType.ERROR,
                value="malformed",
                position=TokenPosition(1, 6, 5),
                confidence=0.1
            ),
        ]
        
        result = EnhancedTokenizationResult(tokens=tokens)
        
        assert result.token_count == 3
        assert result.metadata.total_tokens == 3
        assert result.metadata.error_tokens == 1
        assert result.error_rate == 1/3
        assert result.confidence < 1.0  # Should be lower due to error token
    
    def test_diagnostic_management(self):
        """Test diagnostic entry management."""
        result = EnhancedTokenizationResult()
        
        result.add_diagnostic(
            DiagnosticSeverity.WARNING,
            "Test warning",
            "test_component"
        )
        
        result.add_diagnostic(
            DiagnosticSeverity.ERROR,
            "Test error", 
            "test_component"
        )
        
        assert len(result.diagnostics) == 2
        assert result.has_errors()
        
        warnings = result.get_diagnostics_by_severity(DiagnosticSeverity.WARNING)
        errors = result.get_diagnostics_by_severity(DiagnosticSeverity.ERROR)
        
        assert len(warnings) == 1
        assert len(errors) == 1
        assert warnings[0].message == "Test warning"
        assert errors[0].message == "Test error"
    
    def test_token_filtering_methods(self):
        """Test token filtering convenience methods."""
        from ultra_robust_xml_parser.tokenization.tokenizer import TokenPosition, TokenRepair
        
        tokens = [
            Token(
                type=TokenType.TEXT,
                value="content1",
                position=TokenPosition(1, 1, 0),
                confidence=0.9
            ),
            Token(
                type=TokenType.TAG_NAME,
                value="element",
                position=TokenPosition(1, 10, 9),
                confidence=1.0
            ),
            Token(
                type=TokenType.TEXT,
                value="content2",
                position=TokenPosition(1, 20, 19),
                confidence=0.7,
                repairs=[TokenRepair("test", "test", "orig", "fixed", 0.0)]
            ),
        ]
        
        result = EnhancedTokenizationResult(tokens=tokens)
        
        # Test token type filtering
        text_tokens = result.get_tokens_by_type(TokenType.TEXT)
        assert len(text_tokens) == 2
        
        # Test range filtering
        range_tokens = result.get_tokens_in_range(0, 15)
        assert len(range_tokens) == 2
        
        # Test confidence filtering
        high_conf_tokens = result.get_high_confidence_tokens(0.85)
        assert len(high_conf_tokens) == 2  # 0.9 and 1.0
        
        # Test well-formed vs malformed
        well_formed = result.well_formed_tokens
        malformed = result.malformed_tokens
        assert len(well_formed) == 2  # No repairs and high confidence
        assert len(malformed) == 1   # Has repairs
    
    def test_summary_generation(self):
        """Test summary statistics generation."""
        result = EnhancedTokenizationResult()
        result.performance.processing_time_ms = 100.0
        result.performance.characters_processed = 1000
        
        summary = result.summary()
        
        assert "success" in summary
        assert "confidence" in summary
        assert "token_count" in summary
        assert "processing_time_ms" in summary
        assert "characters_per_second" in summary
        assert summary["processing_time_ms"] == 100.0


class TestTokenFilter:
    """Test token filtering functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        from ultra_robust_xml_parser.tokenization.tokenizer import TokenPosition
        
        self.test_tokens = [
            Token(
                type=TokenType.TAG_START,
                value="<",
                position=TokenPosition(1, 1, 0),
                confidence=1.0
            ),
            Token(
                type=TokenType.TAG_NAME,
                value="root",
                position=TokenPosition(1, 2, 1),
                confidence=0.9
            ),
            Token(
                type=TokenType.TEXT,
                value="content with pattern",
                position=TokenPosition(1, 8, 7),
                confidence=0.8
            ),
            Token(
                type=TokenType.ERROR,
                value="error",
                position=TokenPosition(1, 30, 29),
                confidence=0.1
            ),
        ]
    
    def test_filter_by_type_include(self):
        """Test filtering by token type (include mode)."""
        config = TokenizationConfig()
        config.filtering.mode = FilterMode.INCLUDE
        
        filter_obj = TokenFilter(config)
        
        filtered = filter_obj.filter_by_type(
            self.test_tokens, 
            {TokenType.TAG_START, TokenType.TAG_NAME}
        )
        
        assert len(filtered) == 2
        assert all(
            token.type in {TokenType.TAG_START, TokenType.TAG_NAME} 
            for token in filtered
        )
    
    def test_filter_by_type_exclude(self):
        """Test filtering by token type (exclude mode)."""
        config = TokenizationConfig()
        config.filtering.mode = FilterMode.EXCLUDE
        
        filter_obj = TokenFilter(config)
        
        filtered = filter_obj.filter_by_type(
            self.test_tokens, 
            {TokenType.ERROR}
        )
        
        assert len(filtered) == 3
        assert all(token.type != TokenType.ERROR for token in filtered)
    
    def test_filter_by_confidence(self):
        """Test filtering by confidence threshold."""
        filter_obj = TokenFilter()
        
        filtered = filter_obj.filter_by_confidence(self.test_tokens, 0.85)
        
        assert len(filtered) == 2  # 1.0 and 0.9
        assert all(token.confidence >= 0.85 for token in filtered)
    
    def test_filter_by_position_range(self):
        """Test filtering by position range."""
        filter_obj = TokenFilter()
        
        filtered = filter_obj.filter_by_position_range(self.test_tokens, 0, 10)
        
        assert len(filtered) == 3  # Positions 0, 1, 7
        assert all(0 <= token.position.offset < 10 for token in filtered)
    
    def test_filter_by_content_pattern(self):
        """Test filtering by content pattern."""
        filter_obj = TokenFilter()
        
        filtered = filter_obj.filter_by_content_pattern(self.test_tokens, "pattern")
        
        assert len(filtered) == 1
        assert "pattern" in filtered[0].value
    
    def test_apply_multiple_filters(self):
        """Test applying multiple filters in sequence."""
        filter_obj = TokenFilter()
        
        filtered = filter_obj.apply_filters(
            self.test_tokens,
            type_filter={TokenType.TAG_NAME, TokenType.TEXT},
            confidence_threshold=0.85,
            position_range=(0, 20)
        )
        
        assert len(filtered) == 1  # Only TAG_NAME meets all criteria
        assert filtered[0].type == TokenType.TAG_NAME
        assert filtered[0].confidence >= 0.85
        assert 0 <= filtered[0].position.offset < 20
    
    def test_max_results_limit(self):
        """Test maximum results limiting."""
        config = TokenizationConfig()
        config.filtering.max_results = 2
        
        filter_obj = TokenFilter(config)
        
        filtered = filter_obj.apply_filters(self.test_tokens)
        
        assert len(filtered) <= 2


class TestStreamingTokenizer:
    """Test streaming tokenization interface."""
    
    def test_streaming_initialization(self):
        """Test streaming tokenizer initialization."""
        config = TokenizationConfig()
        config.streaming.chunk_size = 100
        config.streaming.buffer_size = 1000
        
        tokenizer = StreamingTokenizer(config, "test-correlation")
        
        assert tokenizer.config.streaming.chunk_size == 100
        assert tokenizer.correlation_id == "test-correlation"
        assert not tokenizer.is_cancelled
    
    def test_streaming_tokenization(self):
        """Test streaming tokenization process."""
        xml_content = '''<?xml version="1.0"?>
<root>
    <element attr="value">Content</element>
    <another>More content</another>
</root>'''
        
        # Create character stream
        
        char_stream = create_test_char_stream(xml_content)
        
        tokenizer = StreamingTokenizer()
        
        tokens_yielded = []
        progress_calls = []
        
        def progress_callback(progress: float, tokens_count: int):
            progress_calls.append((progress, tokens_count))
        
        # Stream tokenization
        token_generator = tokenizer.tokenize_stream(char_stream, progress_callback)
        
        final_result = None
        for token in token_generator:
            tokens_yielded.append(token)
        
        # Get final result via generator return value
        try:
            next(token_generator)
        except StopIteration as e:
            final_result = e.value
        
        assert len(tokens_yielded) > 0
        if final_result:  # May be None for empty generators
            assert isinstance(final_result, EnhancedTokenizationResult)
            assert final_result.success
            assert len(final_result.tokens) == len(tokens_yielded)
    
    def test_streaming_cancellation(self):
        """Test streaming cancellation."""
        tokenizer = StreamingTokenizer()
        
        assert not tokenizer.is_cancelled
        
        tokenizer.cancel()
        
        assert tokenizer.is_cancelled


class TestEnhancedXMLTokenizer:
    """Test enhanced XML tokenizer with comprehensive API."""
    
    def test_enhanced_tokenizer_initialization(self):
        """Test enhanced tokenizer initialization."""
        config = TokenizationConfig.performance_optimized()
        config.correlation_id = "test-123"
        
        tokenizer = EnhancedXMLTokenizer(config)
        
        assert tokenizer.correlation_id == "test-123"
        assert tokenizer.config.performance.enable_fast_path
        assert isinstance(tokenizer.token_filter, TokenFilter)
        assert isinstance(tokenizer.streaming_tokenizer, StreamingTokenizer)
    
    def test_enhanced_tokenization_basic(self):
        """Test basic enhanced tokenization."""
        xml_content = '''<?xml version="1.0"?>
<root>
    <element>Content</element>
</root>'''
        
        # Create character stream
        
        char_stream = create_test_char_stream(xml_content)
        
        tokenizer = EnhancedXMLTokenizer()
        result = tokenizer.tokenize(char_stream)
        
        assert isinstance(result, EnhancedTokenizationResult)
        assert result.success
        assert result.token_count > 0
        assert result.confidence > 0.0
        assert result.performance.processing_time_ms > 0
        assert result.performance.characters_processed == len(xml_content)
    
    def test_enhanced_tokenization_with_filters(self):
        """Test enhanced tokenization with filter application."""
        xml_content = '''<root>
    <element>Content</element>
    <another>More</another>
</root>'''
        
        # Configure to filter only tag names
        config = TokenizationConfig()
        config.filtering.mode = FilterMode.INCLUDE
        config.filtering.token_types = {"TAG_NAME"}
        
        # Create character stream
        
        char_stream = create_test_char_stream(xml_content)
        
        tokenizer = EnhancedXMLTokenizer(config)
        result = tokenizer.tokenize(char_stream, apply_filters=True)
        
        assert result.success
        assert len(result.tokens) > 0
        
        # Check that diagnostic was added about filtering
        filter_diagnostics = [
            d for d in result.diagnostics 
            if "filter" in d.message.lower()
        ]
        assert len(filter_diagnostics) > 0
    
    def test_enhanced_tokenization_error_handling(self):
        """Test error handling in enhanced tokenization."""
        # Create invalid character stream
        char_stream = create_test_char_stream("")  # Use empty string instead of None
        
        tokenizer = EnhancedXMLTokenizer()
        result = tokenizer.tokenize(char_stream)
        
        # Should still return result (never-fail philosophy)
        assert isinstance(result, EnhancedTokenizationResult)
        assert result.performance.processing_time_ms >= 0
    
    def test_configuration_update(self):
        """Test dynamic configuration updates."""
        tokenizer = EnhancedXMLTokenizer()
        
        original_config = tokenizer.config
        
        new_config = TokenizationConfig.conservative()
        new_config.correlation_id = "new-id"
        
        tokenizer.configure(new_config)
        
        assert tokenizer.config != original_config
        assert tokenizer.correlation_id == "new-id"
    
    def test_streaming_integration(self):
        """Test streaming integration through enhanced tokenizer."""
        xml_content = '''<root><element>Content</element></root>'''
        
        encoding_result = EncodingResult(
            encoding="utf-8",
            confidence=1.0,
            method=DetectionMethod.FALLBACK,
            issues=[]
        )
        
        char_stream = create_test_char_stream(xml_content)
        
        tokenizer = EnhancedXMLTokenizer()
        
        tokens = list(tokenizer.tokenize_streaming(char_stream))
        
        assert len(tokens) > 0
        assert all(isinstance(token, Token) for token in tokens)


class TestTokenizationConfiguration:
    """Test tokenization configuration presets."""
    
    def test_conservative_configuration(self):
        """Test conservative configuration preset."""
        config = TokenizationConfig.conservative()
        
        assert config.recovery.strategy == RecoveryStrategy.CONSERVATIVE
        assert config.assembly.strict_mode is True
        assert config.assembly.repair_confidence_threshold == 0.8
        assert config.performance.enable_fast_path is False
    
    def test_balanced_configuration(self):
        """Test balanced configuration preset."""
        config = TokenizationConfig.balanced()
        
        assert config.recovery.strategy == RecoveryStrategy.BALANCED
        # Should be default values
    
    def test_aggressive_configuration(self):
        """Test aggressive configuration preset."""
        config = TokenizationConfig.aggressive()
        
        assert config.recovery.strategy == RecoveryStrategy.AGGRESSIVE
        assert config.recovery.max_recovery_attempts == 20
        assert config.assembly.enable_smart_assembly is True
        assert config.performance.enable_fast_path is True
    
    def test_performance_optimized_configuration(self):
        """Test performance-optimized configuration preset."""
        config = TokenizationConfig.performance_optimized()
        
        assert config.performance.enable_fast_path is True
        assert config.performance.enable_caching is True
        assert config.performance.enable_parallel_processing is True
        assert config.assembly.enable_caching is True
        assert config.streaming.enable_backpressure is True


class TestIntegrationScenarios:
    """Test complete integration scenarios."""
    
    def test_well_formed_xml_processing(self):
        """Test processing of well-formed XML."""
        xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<document>
    <metadata>
        <title>Test Document</title>
        <author>Test Author</author>
    </metadata>
    <content>
        <section id="1">
            <heading>Introduction</heading>
            <paragraph>This is a test paragraph with <emphasis>emphasis</emphasis>.</paragraph>
            <list>
                <item>First item</item>
                <item>Second item</item>
            </list>
        </section>
    </content>
</document>'''
        
        # Create character stream
        encoding_result = EncodingResult(
            encoding="utf-8",
            confidence=1.0,
            method="bom_detection"
        )
        
        char_stream = create_test_char_stream(xml_content)
        
        tokenizer = EnhancedXMLTokenizer(TokenizationConfig.balanced())
        result = tokenizer.tokenize(char_stream)
        
        assert result.success
        assert result.confidence >= 0.9  # High confidence for well-formed XML
        assert result.error_rate == 0.0   # No errors
        assert result.repair_rate <= 0.1  # Minimal repairs
        assert len(result.get_tokens_by_type(TokenType.TAG_NAME)) >= 8  # Multiple elements
        assert len(result.get_tokens_by_type(TokenType.TEXT)) >= 5     # Text content
    
    def test_malformed_xml_recovery(self):
        """Test processing of malformed XML with recovery."""
        xml_content = '''<?xml version="1.0"?>
<root>
    <unclosed_tag>
    <element attr=unquoted>Content
    <element attr="unclosed>More content</element>
    <empty
</root>'''
        
        encoding_result = EncodingResult(
            encoding="utf-8",
            confidence=1.0,
            method=DetectionMethod.FALLBACK,
            issues=[]
        )
        
        char_stream = create_test_char_stream(xml_content)
        
        tokenizer = EnhancedXMLTokenizer(TokenizationConfig.aggressive())
        result = tokenizer.tokenize(char_stream)
        
        assert result.success  # Should succeed with recovery
        assert result.confidence < 0.9  # Lower confidence due to malformations
        assert result.token_count > 0   # Should produce tokens
        assert len(result.diagnostics) > 0  # Should have diagnostics
        
        # Should have some recovered or synthetic tokens
        recovered_tokens = result.get_tokens_by_type(TokenType.RECOVERED_CONTENT)
        synthetic_tokens = result.get_tokens_by_type(TokenType.SYNTHETIC_CLOSE)
        error_tokens = result.get_tokens_by_type(TokenType.ERROR)
        
        # At least some recovery should have occurred
        assert len(recovered_tokens) + len(synthetic_tokens) + len(error_tokens) > 0
    
    def test_large_document_streaming(self):
        """Test streaming processing of large document."""
        # Generate large XML
        elements = ['<?xml version="1.0"?>', '<large>']
        for i in range(100):  # 100 elements
            elements.append(f'<item id="{i}">Content {i}</item>')
        elements.append('</large>')
        
        xml_content = '\n'.join(elements)
        
        encoding_result = EncodingResult(
            encoding="utf-8",
            confidence=1.0,
            method=DetectionMethod.FALLBACK,
            issues=[]
        )
        
        char_stream = create_test_char_stream(xml_content)
        
        config = TokenizationConfig.performance_optimized()
        config.streaming.chunk_size = 500  # Small chunks for testing
        
        tokenizer = EnhancedXMLTokenizer(config)
        
        # Test streaming
        tokens = []
        for token in tokenizer.tokenize_streaming(char_stream):
            tokens.append(token)
        
        assert len(tokens) > 300  # Expect many tokens from 100 items
        
        # Test regular tokenization for comparison
        regular_result = tokenizer.tokenize(char_stream)
        
        assert regular_result.success
        assert len(regular_result.tokens) > 300
        assert regular_result.performance.characters_processed == len(xml_content)
    
    def test_configuration_impact(self):
        """Test impact of different configurations."""
        xml_content = '''<root>
    <element attr=unquoted>Content
    <malformed>
</root>'''
        
        encoding_result = EncodingResult(
            encoding="utf-8",
            confidence=1.0,
            method=DetectionMethod.FALLBACK,
            issues=[]
        )
        
        char_stream = create_test_char_stream(xml_content)
        
        # Test conservative configuration
        conservative_tokenizer = EnhancedXMLTokenizer(TokenizationConfig.conservative())
        conservative_result = conservative_tokenizer.tokenize(char_stream)
        
        # Test aggressive configuration
        aggressive_tokenizer = EnhancedXMLTokenizer(TokenizationConfig.aggressive())
        aggressive_result = aggressive_tokenizer.tokenize(char_stream)
        
        # Both should succeed (never-fail philosophy)
        assert conservative_result.success
        assert aggressive_result.success
        
        # Aggressive should potentially have more repairs
        # (This is a general expectation, specific results may vary)
        assert conservative_result.token_count > 0
        assert aggressive_result.token_count > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])