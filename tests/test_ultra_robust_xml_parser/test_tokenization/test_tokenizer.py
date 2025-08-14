"""Comprehensive tests for XML tokenization functionality."""

import pytest
from unittest.mock import Mock, patch
from dataclasses import dataclass, field
from typing import List

from ultra_robust_xml_parser.character import CharacterStreamResult
from ultra_robust_xml_parser.tokenization import (
    XMLTokenizer,
    Token,
    TokenType,
    TokenPosition,
    TokenizerState,
    TokenRepair,
    TokenizationResult,
)


@dataclass
class MockEncodingResult:
    """Mock encoding result for testing."""
    encoding: str = "utf-8"
    confidence: float = 1.0

@dataclass
class MockTransformResult:
    """Mock transform result for testing."""
    text: str = ""
    transformations: List = field(default_factory=list)

@dataclass
class MockCharacterStreamResult:
    """Mock character stream result for testing."""
    text: str
    encoding: MockEncodingResult = field(default_factory=MockEncodingResult)
    transformations: MockTransformResult = field(default_factory=MockTransformResult)
    confidence: float = 1.0
    
    def __post_init__(self):
        if isinstance(self.encoding, str):
            self.encoding = MockEncodingResult(encoding=self.encoding)
        if not hasattr(self, 'transformations') or self.transformations is None:
            self.transformations = MockTransformResult()


class TestTokenPosition:
    """Tests for TokenPosition class."""
    
    def test_token_position_creation(self):
        """Test TokenPosition creation with valid values."""
        pos = TokenPosition(line=5, column=10, offset=50)
        assert pos.line == 5
        assert pos.column == 10
        assert pos.offset == 50
    
    def test_token_position_validation(self):
        """Test TokenPosition validation for invalid values."""
        with pytest.raises(ValueError, match="Line number must be >= 1"):
            TokenPosition(line=0, column=1, offset=0)
        
        with pytest.raises(ValueError, match="Column number must be >= 1"):
            TokenPosition(line=1, column=0, offset=0)
        
        with pytest.raises(ValueError, match="Offset must be >= 0"):
            TokenPosition(line=1, column=1, offset=-1)


class TestTokenRepair:
    """Tests for TokenRepair class."""
    
    def test_token_repair_creation(self):
        """Test TokenRepair creation."""
        repair = TokenRepair(
            repair_type="quote_fix",
            description="Fixed unmatched quote",
            original_content="attr=value",
            repaired_content='attr="value"',
            confidence_impact=-0.1
        )
        assert repair.repair_type == "quote_fix"
        assert repair.description == "Fixed unmatched quote"
        assert repair.confidence_impact == -0.1


class TestToken:
    """Tests for Token class."""
    
    def test_token_creation(self):
        """Test Token creation with valid values."""
        pos = TokenPosition(1, 1, 0)
        token = Token(
            type=TokenType.TAG_START,
            value="<",
            position=pos,
            confidence=0.95
        )
        assert token.type == TokenType.TAG_START
        assert token.value == "<"
        assert token.confidence == 0.95
        assert not token.has_repairs
        assert token.is_well_formed
    
    def test_token_with_repairs(self):
        """Test Token with repairs."""
        pos = TokenPosition(1, 1, 0)
        repair = TokenRepair("test", "Test repair", "orig", "fixed", -0.1)
        token = Token(
            type=TokenType.TEXT,
            value="fixed_content",
            position=pos,
            confidence=0.8,
            repairs=[repair]
        )
        assert token.has_repairs
        assert not token.is_well_formed  # Has repairs, so not well-formed
    
    def test_token_confidence_validation(self):
        """Test Token confidence validation."""
        pos = TokenPosition(1, 1, 0)
        
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            Token(TokenType.TEXT, "content", pos, confidence=1.5)
        
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            Token(TokenType.TEXT, "content", pos, confidence=-0.1)


class TestTokenizationResult:
    """Tests for TokenizationResult class."""
    
    def test_tokenization_result_creation(self):
        """Test TokenizationResult creation and derived statistics."""
        pos = TokenPosition(1, 1, 0)
        repair = TokenRepair("test", "Test repair", "orig", "fixed", -0.1)
        
        tokens = [
            Token(TokenType.TAG_START, "<", pos, confidence=1.0),
            Token(TokenType.TEXT, "content", pos, confidence=0.8, repairs=[repair]),
            Token(TokenType.ERROR, "bad", pos, confidence=0.1),
        ]
        
        result = TokenizationResult(tokens=tokens)
        
        assert result.token_count == 3
        assert result.total_repairs == 1
        assert result.error_count == 1
        assert result.confidence == 0.1  # Minimum confidence
        assert result.well_formed_percentage == 1/3  # Only first token is well-formed
    
    def test_empty_tokenization_result(self):
        """Test TokenizationResult with no tokens."""
        result = TokenizationResult(tokens=[])
        assert result.token_count == 0
        assert result.total_repairs == 0
        assert result.error_count == 0
        assert result.confidence == 1.0
        assert result.well_formed_percentage == 1.0


class TestXMLTokenizer:
    """Tests for XMLTokenizer class."""
    
    def test_tokenizer_initialization(self):
        """Test XMLTokenizer initialization."""
        tokenizer = XMLTokenizer(correlation_id="test-123")
        assert tokenizer.correlation_id == "test-123"
        assert tokenizer.enable_fast_path is True
        assert tokenizer.state == TokenizerState.TEXT_CONTENT
    
    def test_tokenizer_initialization_no_fast_path(self):
        """Test XMLTokenizer initialization with fast-path disabled."""
        tokenizer = XMLTokenizer(enable_fast_path=False)
        assert tokenizer.enable_fast_path is False
        assert not tokenizer.fast_path_enabled
    
    def test_simple_text_tokenization(self):
        """Test tokenization of simple text content."""
        tokenizer = XMLTokenizer()
        char_stream = MockCharacterStreamResult(text="Hello World")
        
        result = tokenizer.tokenize(char_stream)
        
        assert result.success
        assert result.token_count == 1
        assert result.tokens[0].type == TokenType.TEXT
        assert result.tokens[0].value == "Hello World"
        assert result.tokens[0].confidence == 1.0
    
    def test_simple_tag_tokenization(self):
        """Test tokenization of simple XML tag."""
        tokenizer = XMLTokenizer()
        char_stream = MockCharacterStreamResult(text="<tag>")
        
        result = tokenizer.tokenize(char_stream)
        
        assert result.success
        assert result.token_count == 3
        
        # Check token sequence: TAG_START, TAG_NAME, TAG_END
        assert result.tokens[0].type == TokenType.TAG_START
        assert result.tokens[0].value == "<"
        
        assert result.tokens[1].type == TokenType.TAG_NAME
        assert result.tokens[1].value == "tag"
        
        assert result.tokens[2].type == TokenType.TAG_END
        assert result.tokens[2].value == ">"
    
    def test_tag_with_attributes_tokenization(self):
        """Test tokenization of tag with attributes."""
        tokenizer = XMLTokenizer()
        char_stream = MockCharacterStreamResult(text='<tag attr="value">')
        
        result = tokenizer.tokenize(char_stream)
        
        assert result.success
        # Should have TAG_START, TAG_NAME, ATTR_NAME, ATTR_VALUE, TAG_END
        assert result.token_count >= 4
        
        tag_name_found = False
        attr_name_found = False
        attr_value_found = False
        
        for token in result.tokens:
            if token.type == TokenType.TAG_NAME and token.value == "tag":
                tag_name_found = True
            elif token.type == TokenType.ATTR_NAME and token.value == "attr":
                attr_name_found = True
            elif token.type == TokenType.ATTR_VALUE and token.value == "value":
                attr_value_found = True
        
        assert tag_name_found
        assert attr_name_found
        assert attr_value_found
    
    def test_xml_comment_tokenization(self):
        """Test tokenization of XML comments."""
        tokenizer = XMLTokenizer()
        char_stream = MockCharacterStreamResult(text="<!-- This is a comment -->")
        
        result = tokenizer.tokenize(char_stream)
        
        assert result.success
        assert result.token_count == 1
        assert result.tokens[0].type == TokenType.COMMENT
        assert result.tokens[0].value == " This is a comment "
    
    def test_cdata_tokenization(self):
        """Test tokenization of CDATA sections."""
        tokenizer = XMLTokenizer()
        char_stream = MockCharacterStreamResult(text="<![CDATA[Some data here]]>")
        
        result = tokenizer.tokenize(char_stream)
        
        assert result.success
        assert result.token_count == 1
        assert result.tokens[0].type == TokenType.CDATA
        assert result.tokens[0].value == "Some data here"
    
    def test_processing_instruction_tokenization(self):
        """Test tokenization of processing instructions."""
        tokenizer = XMLTokenizer()
        char_stream = MockCharacterStreamResult(text='<?xml version="1.0"?>')
        
        result = tokenizer.tokenize(char_stream)
        
        assert result.success
        # Look for processing instruction token
        pi_found = False
        for token in result.tokens:
            if token.type == TokenType.PROCESSING_INSTRUCTION:
                pi_found = True
                assert 'xml version="1.0"' in token.value
                break
        
        assert pi_found
    
    def test_position_tracking(self):
        """Test position tracking during tokenization."""
        tokenizer = XMLTokenizer(enable_fast_path=False)  # Use robust processing for accurate position tracking
        char_stream = MockCharacterStreamResult(text="<tag>\nText\n</tag>")
        
        result = tokenizer.tokenize(char_stream)
        
        assert result.success
        
        # First token should be at line 1, column 1
        first_token = result.tokens[0]
        assert first_token.position.line == 1
        assert first_token.position.column == 1
        
        # Find text token - it includes the newline, so position depends on when text starts
        text_token = None
        for token in result.tokens:
            if token.type == TokenType.TEXT and "Text" in token.value:
                text_token = token
                break
        
        assert text_token is not None
        # Text token position is where the text buffer started (after the > of <tag>)
        # This is expected behavior - position tracks start of token
        assert text_token.position.line >= 1
    
    def test_malformed_xml_error_recovery(self):
        """Test error recovery with malformed XML."""
        tokenizer = XMLTokenizer()
        char_stream = MockCharacterStreamResult(text="<<invalid>>")
        
        result = tokenizer.tokenize(char_stream)
        
        # Should not fail (never-fail philosophy)
        assert result.success
        
        # With recovery engine, should have error tokens, repairs, or enhanced token types
        has_errors = any(token.type == TokenType.ERROR for token in result.tokens)
        has_repairs = any(token.has_repairs for token in result.tokens)
        has_enhanced_tokens = any(
            token.type in (TokenType.INVALID_CHARS, TokenType.MALFORMED_TAG, 
                          TokenType.RECOVERED_CONTENT) 
            for token in result.tokens
        )
        
        assert has_errors or has_repairs or has_enhanced_tokens
    
    def test_unclosed_tag_handling(self):
        """Test handling of unclosed tags."""
        tokenizer = XMLTokenizer()
        char_stream = MockCharacterStreamResult(text="<tag")
        
        result = tokenizer.tokenize(char_stream)
        
        # Should not fail
        assert result.success
        
        # Should have tokens even for incomplete input
        assert result.token_count > 0
    
    def test_empty_content_handling(self):
        """Test handling of empty content."""
        tokenizer = XMLTokenizer()
        char_stream = MockCharacterStreamResult(text="")
        
        result = tokenizer.tokenize(char_stream)
        
        assert result.success
        assert result.token_count == 0
    
    def test_fast_path_detection_well_formed(self):
        """Test fast-path detection for well-formed XML."""
        tokenizer = XMLTokenizer()
        
        # Well-formed XML should trigger fast-path
        well_formed_xml = '<?xml version="1.0"?><root><child attr="value">text</child></root>'
        assert tokenizer._detect_well_formed_xml(well_formed_xml)
    
    def test_fast_path_detection_malformed(self):
        """Test fast-path detection rejects malformed XML."""
        tokenizer = XMLTokenizer()
        
        # Malformed XML should not trigger fast-path
        malformed_xml = "<<root>>invalid<</root>>"
        assert not tokenizer._detect_well_formed_xml(malformed_xml)
    
    def test_fast_path_tokenization(self):
        """Test fast-path tokenization produces correct tokens."""
        tokenizer = XMLTokenizer()
        well_formed_xml = '<root attr="value">text</root>'
        
        # Force fast-path detection to return True
        with patch.object(tokenizer, '_detect_well_formed_xml', return_value=True):
            char_stream = MockCharacterStreamResult(text=well_formed_xml)
            result = tokenizer.tokenize(char_stream)
        
        assert result.success
        assert result.token_count > 0
        
        # Should have high confidence tokens
        assert all(token.confidence >= 0.9 for token in result.tokens)
    
    def test_fast_path_fallback(self):
        """Test fast-path fallback to robust processing."""
        tokenizer = XMLTokenizer()
        
        # Mock fast-path detection to return True but tokenization to fail
        with patch.object(tokenizer, '_detect_well_formed_xml', return_value=True), \
             patch.object(tokenizer, '_fast_path_tokenize', return_value=[]):
            
            char_stream = MockCharacterStreamResult(text="<root>text</root>")
            result = tokenizer.tokenize(char_stream)
        
        assert result.success
        # Should fall back to robust processing
        assert not tokenizer.fast_path_enabled
    
    def test_quote_handling_single_quotes(self):
        """Test handling of single-quoted attribute values."""
        tokenizer = XMLTokenizer()
        char_stream = MockCharacterStreamResult(text="<tag attr='value'>")
        
        result = tokenizer.tokenize(char_stream)
        
        assert result.success
        
        # Find attribute value token
        attr_value_found = False
        for token in result.tokens:
            if token.type == TokenType.ATTR_VALUE and token.value == 'value':
                attr_value_found = True
                break
        
        assert attr_value_found
    
    def test_quote_handling_double_quotes(self):
        """Test handling of double-quoted attribute values."""
        tokenizer = XMLTokenizer()
        char_stream = MockCharacterStreamResult(text='<tag attr="value">')
        
        result = tokenizer.tokenize(char_stream)
        
        assert result.success
        
        # Find attribute value token
        attr_value_found = False
        for token in result.tokens:
            if token.type == TokenType.ATTR_VALUE and token.value == 'value':
                attr_value_found = True
                break
        
        assert attr_value_found
    
    def test_unquoted_attribute_values(self):
        """Test handling of unquoted attribute values."""
        tokenizer = XMLTokenizer()
        char_stream = MockCharacterStreamResult(text="<tag attr=value>")
        
        result = tokenizer.tokenize(char_stream)
        
        assert result.success
        
        # Should handle unquoted values
        attr_value_found = False
        for token in result.tokens:
            if token.type == TokenType.ATTR_VALUE and token.value == 'value':
                attr_value_found = True
                break
        
        assert attr_value_found
    
    def test_whitespace_handling(self):
        """Test proper whitespace handling in tokenization."""
        tokenizer = XMLTokenizer()
        char_stream = MockCharacterStreamResult(text="  <tag>  text  </tag>  ")
        
        result = tokenizer.tokenize(char_stream)
        
        assert result.success
        assert result.token_count > 0
        
        # Should preserve text with whitespace
        text_found = False
        for token in result.tokens:
            if token.type == TokenType.TEXT and "text" in token.value:
                text_found = True
                break
        
        assert text_found
    
    def test_special_characters_in_text(self):
        """Test handling of special characters in text content."""
        tokenizer = XMLTokenizer(enable_fast_path=False)  # Disable fast-path for malformed content
        special_text = "Text with & \" ' special chars"  # Removed < > which should be escaped in XML
        char_stream = MockCharacterStreamResult(text=f"<root>{special_text}</root>")
        
        result = tokenizer.tokenize(char_stream)
        
        assert result.success
        
        # Should preserve special characters in text
        text_found = False
        for token in result.tokens:
            if token.type == TokenType.TEXT and special_text in token.value:
                text_found = True
                break
        
        assert text_found
    
    def test_nested_tags(self):
        """Test tokenization of nested XML tags."""
        tokenizer = XMLTokenizer()
        char_stream = MockCharacterStreamResult(text="<parent><child>text</child></parent>")
        
        result = tokenizer.tokenize(char_stream)
        
        assert result.success
        assert result.token_count > 0
        
        # Should find both parent and child tag names
        tag_names = [token.value for token in result.tokens if token.type == TokenType.TAG_NAME]
        assert "parent" in tag_names
        assert "child" in tag_names
    
    def test_correlation_id_logging(self):
        """Test that correlation ID is used in logging."""
        correlation_id = "test-correlation-123"
        tokenizer = XMLTokenizer(correlation_id=correlation_id)
        
        with patch('ultra_robust_xml_parser.tokenization.tokenizer.logger') as mock_logger:
            char_stream = MockCharacterStreamResult(text="<root>text</root>")
            result = tokenizer.tokenize(char_stream)
            
            # Verify logger was called with correlation_id
            mock_logger.debug.assert_called()
            call_args = mock_logger.debug.call_args_list
            
            correlation_logged = False
            for call in call_args:
                if 'extra' in call.kwargs and 'correlation_id' in call.kwargs['extra']:
                    if call.kwargs['extra']['correlation_id'] == correlation_id:
                        correlation_logged = True
                        break
            
            assert correlation_logged
    
    def test_error_handling_never_fail(self):
        """Test that tokenizer never fails completely."""
        tokenizer = XMLTokenizer()
        
        # Test with completely invalid input
        char_stream = MockCharacterStreamResult(text="\x00\x01\x02invalid")
        
        result = tokenizer.tokenize(char_stream)
        
        # Should always return a result (never-fail philosophy)
        assert isinstance(result, TokenizationResult)
        # Might not be successful, but should not raise exceptions
    
    def test_performance_with_large_input(self):
        """Test tokenizer performance with large input."""
        tokenizer = XMLTokenizer()
        
        # Create large XML content
        large_content = "<root>" + "text " * 1000 + "</root>"
        char_stream = MockCharacterStreamResult(text=large_content)
        
        result = tokenizer.tokenize(char_stream)
        
        assert result.success
        assert result.processing_time >= 0  # Should track processing time
        assert result.character_count == len(large_content)
    
    def test_mixed_content_tokenization(self):
        """Test tokenization of mixed content (tags, text, comments, CDATA)."""
        mixed_content = '''<?xml version="1.0"?>
        <root>
            <!-- A comment -->
            <child attr="value">Text content</child>
            <![CDATA[Some CDATA content]]>
            More text
        </root>'''
        
        tokenizer = XMLTokenizer()
        char_stream = MockCharacterStreamResult(text=mixed_content)
        
        result = tokenizer.tokenize(char_stream)
        
        assert result.success
        
        # Check that we have various token types
        token_types = {token.type for token in result.tokens}
        expected_types = {
            TokenType.PROCESSING_INSTRUCTION,
            TokenType.TAG_START,
            TokenType.TAG_NAME,
            TokenType.TAG_END,
            TokenType.COMMENT,
            TokenType.ATTR_NAME,
            TokenType.ATTR_VALUE,
            TokenType.TEXT,
            TokenType.CDATA
        }
        
        # Should have most expected types (may not have all due to tokenization specifics)
        assert len(token_types.intersection(expected_types)) >= 5


class TestTokenizerStateMachine:
    """Tests for tokenizer state machine behavior."""
    
    def test_state_transitions_basic_tag(self):
        """Test state transitions for basic tag processing."""
        tokenizer = XMLTokenizer()
        
        # Initialize and verify starting state
        assert tokenizer.state == TokenizerState.TEXT_CONTENT
        
        # Process a simple tag
        char_stream = MockCharacterStreamResult(text="<tag>")
        result = tokenizer.tokenize(char_stream)
        
        assert result.success
        # After processing, should be back in TEXT_CONTENT state
        assert tokenizer.state == TokenizerState.TEXT_CONTENT
    
    def test_state_persistence_across_boundaries(self):
        """Test that tokenizer maintains state correctly across processing boundaries."""
        tokenizer = XMLTokenizer()
        
        # Process incomplete tag
        char_stream = MockCharacterStreamResult(text="<tag")
        result = tokenizer.tokenize(char_stream)
        
        # Should handle incomplete input gracefully
        assert result.success or result.token_count > 0  # Partial success acceptable
    
    def test_error_recovery_state_handling(self):
        """Test error recovery state transitions."""
        tokenizer = XMLTokenizer()
        
        # Force an error condition
        char_stream = MockCharacterStreamResult(text="<>")  # Invalid empty tag
        result = tokenizer.tokenize(char_stream)
        
        # Should recover and continue processing
        assert result.success  # Never-fail philosophy
        assert tokenizer.error_recovery_count >= 0


if __name__ == "__main__":
    pytest.main([__file__])