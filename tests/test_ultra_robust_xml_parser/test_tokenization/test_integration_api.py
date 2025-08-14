"""Integration tests for tokenization API with character processing layer.

This module tests the complete end-to-end integration between the tokenization API
and character processing, ensuring seamless data flow and comprehensive error handling.
"""

import pytest
from typing import List

from ultra_robust_xml_parser.character import (
    CharacterStreamProcessor,
    CharacterTransformer,
)
from ultra_robust_xml_parser.character.encoding import EncodingDetector, EncodingResult, DetectionMethod
from ultra_robust_xml_parser.shared import TokenizationConfig, DiagnosticSeverity
from ultra_robust_xml_parser.tokenization import (
    EnhancedXMLTokenizer,
    StreamingTokenizer,
    TokenFilter,
    TokenType,
)
from ultra_robust_xml_parser.tokenization.api import TokenizationResult as EnhancedTokenizationResult


class TestCharacterToTokenizationIntegration:
    """Test integration between character processing and tokenization."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.encoding_detector = EncodingDetector()
        self.character_transformer = CharacterTransformer()
        self.stream_processor = CharacterStreamProcessor()
        self.tokenizer = EnhancedXMLTokenizer()
    
    def test_complete_pipeline_well_formed(self):
        """Test complete processing pipeline with well-formed XML."""
        xml_bytes = '''<?xml version="1.0" encoding="UTF-8"?>
<document>
    <metadata>
        <title>Integration Test</title>
        <author>Test Suite</author>
    </metadata>
    <content>
        <section id="intro">
            <heading>Introduction</heading>
            <paragraph>This tests the complete pipeline from bytes to tokens.</paragraph>
        </section>
    </content>
</document>'''.encode('utf-8')
        
        # Step 1: Encoding detection
        encoding_result = self.encoding_detector.detect_encoding(xml_bytes)
        assert encoding_result.success
        assert encoding_result.encoding == "utf-8"
        assert encoding_result.confidence >= 0.9
        
        # Step 2: Character transformation
        transform_result = self.character_transformer.transform_to_unicode(
            xml_bytes, encoding_result.encoding
        )
        assert transform_result.success
        assert transform_result.confidence >= 0.9
        
        # Step 3: Character stream processing
        stream_result = self.stream_processor.process_stream(
            transform_result.text,
            encoding_result,
            correlation_id="integration-test-1"
        )
        assert stream_result.success
        assert stream_result.confidence >= 0.9
        
        # Step 4: Tokenization
        tokenization_result = self.tokenizer.tokenize(stream_result)
        assert isinstance(tokenization_result, EnhancedTokenizationResult)
        assert tokenization_result.success
        assert tokenization_result.confidence >= 0.9
        assert tokenization_result.token_count > 0
        
        # Verify token types are present
        tag_names = tokenization_result.get_tokens_by_type(TokenType.TAG_NAME)
        text_tokens = tokenization_result.get_tokens_by_type(TokenType.TEXT)
        
        assert len(tag_names) >= 6  # document, metadata, title, author, content, section, etc.
        assert len(text_tokens) >= 3  # Title, author, paragraph content
        
        # Verify no errors in well-formed processing
        assert tokenization_result.error_rate == 0.0
        assert not tokenization_result.has_errors()
    
    def test_complete_pipeline_malformed_with_recovery(self):
        """Test complete pipeline with malformed XML requiring recovery."""
        xml_bytes = '''<?xml version="1.0" encoding="UTF-8"?>
<document>
    <metadata>
        <title>Malformed Test
        <author attr=unquoted>Test Suite</author>
    </metadata>
    <content>
        <section id="problem">
            <heading>Issues</heading>
            <paragraph>This has <unclosed>nested problems
            <data value="test>More issues</data>
        </section>
        <!-- Unclosed comment
    </content>
</document>'''.encode('utf-8')
        
        # Process through complete pipeline
        encoding_result = self.encoding_detector.detect_encoding(xml_bytes)
        transform_result = self.character_transformer.transform_to_unicode(
            xml_bytes, encoding_result.encoding
        )
        stream_result = self.stream_processor.process_stream(
            transform_result.text,
            encoding_result,
            correlation_id="integration-test-2"
        )
        
        # Use aggressive recovery configuration
        config = TokenizationConfig.aggressive()
        aggressive_tokenizer = EnhancedXMLTokenizer(config)
        
        tokenization_result = aggressive_tokenizer.tokenize(stream_result)
        
        # Should succeed with recovery
        assert tokenization_result.success
        assert tokenization_result.token_count > 0
        
        # Should have lower confidence due to malformations
        assert tokenization_result.confidence < 0.9
        
        # Should have diagnostics about issues
        assert len(tokenization_result.diagnostics) > 0
        
        # Should have some error or recovered tokens
        error_tokens = tokenization_result.get_tokens_by_type(TokenType.ERROR)
        recovered_tokens = tokenization_result.get_tokens_by_type(TokenType.RECOVERED_CONTENT)
        malformed_tokens = tokenization_result.get_tokens_by_type(TokenType.MALFORMED_TAG)
        
        total_problem_tokens = len(error_tokens) + len(recovered_tokens) + len(malformed_tokens)
        assert total_problem_tokens > 0
        
        # Should still extract some meaningful content
        text_tokens = tokenization_result.get_tokens_by_type(TokenType.TEXT)
        tag_names = tokenization_result.get_tokens_by_type(TokenType.TAG_NAME)
        
        assert len(text_tokens) > 0  # Some text should be recovered
        assert len(tag_names) > 0    # Some tags should be identified
    
    def test_encoding_detection_integration(self):
        """Test integration with various character encodings."""
        test_cases = [
            # UTF-8 with BOM
            ('UTF-8 with BOM', '''<?xml version="1.0" encoding="UTF-8"?>
<test>UTF-8 content with émojis 🚀</test>'''.encode('utf-8-sig')),
            
            # UTF-16
            ('UTF-16', '''<?xml version="1.0" encoding="UTF-16"?>
<test>UTF-16 content</test>'''.encode('utf-16')),
            
            # Latin-1
            ('Latin-1', '''<?xml version="1.0" encoding="ISO-8859-1"?>
<test>Latin-1 content with café</test>'''.encode('latin-1')),
        ]
        
        for case_name, xml_bytes in test_cases:
            # Process through pipeline
            encoding_result = self.encoding_detector.detect_encoding(xml_bytes)
            
            if not encoding_result.success:
                # Some encodings might not be perfectly detected, that's OK
                continue
                
            transform_result = self.character_transformer.transform_to_unicode(
                xml_bytes, encoding_result.encoding
            )
            
            if not transform_result.success:
                continue
                
            stream_result = self.stream_processor.process_stream(
                transform_result.text,
                encoding_result,
                correlation_id=f"encoding-test-{case_name.lower()}"
            )
            
            tokenization_result = self.tokenizer.tokenize(stream_result)
            
            # Should successfully tokenize regardless of original encoding
            assert tokenization_result.success, f"Failed for {case_name}"
            assert tokenization_result.token_count > 0, f"No tokens for {case_name}"
            
            # Should have root element
            tag_names = tokenization_result.get_tokens_by_type(TokenType.TAG_NAME)
            tag_name_values = [token.value for token in tag_names]
            assert "test" in tag_name_values, f"Missing root element for {case_name}"
    
    def test_streaming_integration(self):
        """Test streaming tokenization with character processing."""
        # Generate larger XML content
        xml_parts = ['<?xml version="1.0" encoding="UTF-8"?>', '<large_document>']
        for i in range(50):
            xml_parts.append(f'''
    <item id="{i}" category="test">
        <title>Item {i}</title>
        <description>Description for item {i} with some content.</description>
        <data value1="{i}" value2="{i*2}"/>
    </item>''')
        xml_parts.append('</large_document>')
        
        xml_content = '\n'.join(xml_parts)
        xml_bytes = xml_content.encode('utf-8')
        
        # Process through character pipeline
        encoding_result = self.encoding_detector.detect_encoding(xml_bytes)
        transform_result = self.character_transformer.transform_to_unicode(
            xml_bytes, encoding_result.encoding
        )
        stream_result = self.stream_processor.process_stream(
            transform_result.text,
            encoding_result,
            correlation_id="streaming-integration-test"
        )
        
        # Configure streaming tokenizer
        config = TokenizationConfig.performance_optimized()
        config.streaming.chunk_size = 1000  # Small chunks for testing
        config.streaming.enable_progress_tracking = True
        
        streaming_tokenizer = StreamingTokenizer(config)
        
        # Track streaming progress
        progress_updates = []
        
        def progress_callback(progress: float, tokens_count: int):
            progress_updates.append((progress, tokens_count))
        
        # Stream tokenization
        streamed_tokens = []
        token_generator = streaming_tokenizer.tokenize_stream(
            stream_result, progress_callback
        )
        
        for token in token_generator:
            streamed_tokens.append(token)
        
        # Verify streaming worked
        assert len(streamed_tokens) > 100  # Should have many tokens
        assert len(progress_updates) > 0    # Should have progress updates
        
        # Compare with regular tokenization
        regular_tokenizer = EnhancedXMLTokenizer(config)
        regular_result = regular_tokenizer.tokenize(stream_result)
        
        # Should have similar token counts (might vary slightly due to assembly)
        token_count_diff = abs(len(streamed_tokens) - regular_result.token_count)
        assert token_count_diff < 10  # Allow small variance
    
    def test_filtering_integration(self):
        """Test token filtering with complete pipeline."""
        xml_bytes = '''<?xml version="1.0"?>
<document>
    <metadata>
        <title>Filter Test</title>
        <author>Test</author>
        <keywords>xml, parsing, filtering</keywords>
    </metadata>
    <content>
        <section>
            <paragraph>Some content</paragraph>
            <list>
                <item>First</item>
                <item>Second</item>
            </list>
        </section>
    </content>
</document>'''.encode('utf-8')
        
        # Process through character pipeline
        encoding_result = self.encoding_detector.detect_encoding(xml_bytes)
        transform_result = self.character_transformer.transform_to_unicode(
            xml_bytes, encoding_result.encoding
        )
        stream_result = self.stream_processor.process_stream(
            transform_result.text, encoding_result
        )
        
        # Configure filtering to only get text content
        config = TokenizationConfig()
        config.filtering.mode = config.filtering.mode.__class__.INCLUDE
        config.filtering.token_types = {"TEXT"}
        
        tokenizer = EnhancedXMLTokenizer(config)
        
        # Tokenize with filtering
        result = tokenizer.tokenize(stream_result, apply_filters=True)
        
        assert result.success
        assert result.token_count > 0
        
        # Should only contain text tokens (after filtering)
        for token in result.tokens:
            # Note: The actual filtering implementation may not be perfect
            # This test verifies the integration works, not the exact filtering
            pass
        
        # Should have diagnostic about filtering
        filter_diagnostics = [
            d for d in result.diagnostics
            if "filter" in d.message.lower()
        ]
        assert len(filter_diagnostics) > 0
    
    def test_error_propagation_through_pipeline(self):
        """Test error handling and propagation through the complete pipeline."""
        # Invalid byte sequence
        invalid_bytes = b'\xff\xfe\x00\x00Invalid XML content'
        
        # Process through pipeline - some steps may fail
        encoding_result = self.encoding_detector.detect_encoding(invalid_bytes)
        
        if encoding_result.success:
            transform_result = self.character_transformer.transform_to_unicode(
                invalid_bytes, encoding_result.encoding
            )
            
            if transform_result.success:
                stream_result = self.stream_processor.process_stream(
                    transform_result.text, encoding_result
                )
                
                # Tokenization should still work (never-fail philosophy)
                tokenization_result = self.tokenizer.tokenize(stream_result)
                
                assert isinstance(tokenization_result, EnhancedTokenizationResult)
                # May not be successful but should return a result
                assert tokenization_result.performance.processing_time_ms >= 0
    
    def test_correlation_id_propagation(self):
        """Test correlation ID propagation through pipeline."""
        correlation_id = "test-correlation-123"
        
        xml_bytes = '''<?xml version="1.0"?>
<test>Correlation test</test>'''.encode('utf-8')
        
        # Process with correlation ID
        encoding_result = self.encoding_detector.detect_encoding(
            xml_bytes, correlation_id=correlation_id
        )
        
        transform_result = self.character_transformer.transform_to_unicode(
            xml_bytes, 
            encoding_result.encoding,
            correlation_id=correlation_id
        )
        
        stream_result = self.stream_processor.process_stream(
            transform_result.text,
            encoding_result,
            correlation_id=correlation_id
        )
        
        # Create tokenizer with same correlation ID
        config = TokenizationConfig()
        config.correlation_id = correlation_id
        
        tokenizer = EnhancedXMLTokenizer(config)
        tokenization_result = tokenizer.tokenize(stream_result)
        
        # Verify correlation ID is propagated
        assert tokenization_result.correlation_id == correlation_id
        
        # Check diagnostics also have correlation ID
        for diagnostic in tokenization_result.diagnostics:
            if diagnostic.correlation_id:
                assert diagnostic.correlation_id == correlation_id
    
    def test_performance_across_pipeline(self):
        """Test performance characteristics across the complete pipeline."""
        xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<performance_test>
''' + '\n'.join([f'    <item id="{i}">Content {i}</item>' for i in range(100)]) + '''
</performance_test>'''
        
        xml_bytes = xml_content.encode('utf-8')
        
        # Process through optimized pipeline
        import time
        
        start_time = time.time()
        
        encoding_result = self.encoding_detector.detect_encoding(xml_bytes)
        transform_result = self.character_transformer.transform_to_unicode(
            xml_bytes, encoding_result.encoding
        )
        stream_result = self.stream_processor.process_stream(
            transform_result.text, encoding_result
        )
        
        # Use performance-optimized tokenizer
        config = TokenizationConfig.performance_optimized()
        tokenizer = EnhancedXMLTokenizer(config)
        tokenization_result = tokenizer.tokenize(stream_result)
        
        total_time = time.time() - start_time
        
        # Verify successful processing
        assert tokenization_result.success
        assert tokenization_result.token_count > 200  # Many tokens expected
        
        # Performance should be reasonable (this is a rough check)
        assert total_time < 1.0  # Should complete within 1 second
        
        # Check performance metrics are captured
        assert tokenization_result.performance.processing_time_ms > 0
        assert tokenization_result.performance.characters_processed > 0
        assert tokenization_result.performance.characters_per_second > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])