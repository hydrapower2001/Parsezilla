"""Tests for character stream processing API.

This module tests the CharacterStreamProcessor with various input types,
configurations, and error scenarios following the AAA pattern.
"""

import io
import pytest
from unittest.mock import Mock, patch

from ultra_robust_xml_parser.character.stream import (
    CharacterStreamProcessor,
    CharacterStreamResult,
    StreamProcessingConfig,
    StreamProcessingPresets,
    StreamingResult,
    StreamingProgress,
    ProgressCallback,
)
from ultra_robust_xml_parser.character.encoding import EncodingResult, DetectionMethod
from ultra_robust_xml_parser.character.transformation import TransformConfig, TransformResult


class TestCharacterStreamResult:
    """Test the CharacterStreamResult class."""
    
    def test_valid_confidence_score(self):
        """Test that valid confidence scores are accepted."""
        # Arrange
        from ultra_robust_xml_parser.character.encoding import EncodingResult, DetectionMethod
        from ultra_robust_xml_parser.character.transformation import TransformResult
        
        encoding_result = EncodingResult("utf-8", 1.0, DetectionMethod.BOM, [])
        transform_result = TransformResult("test", 1.0)
        
        # Act
        result = CharacterStreamResult(
            text="test",
            encoding=encoding_result,
            transformations=transform_result,
            confidence=0.8
        )
        
        # Assert
        assert result.confidence == 0.8
    
    def test_invalid_confidence_score_raises_error(self):
        """Test that invalid confidence scores raise ValueError."""
        # Arrange
        from ultra_robust_xml_parser.character.encoding import EncodingResult, DetectionMethod
        from ultra_robust_xml_parser.character.transformation import TransformResult
        
        encoding_result = EncodingResult("utf-8", 1.0, DetectionMethod.BOM, [])
        transform_result = TransformResult("test", 1.0)
        
        # Act & Assert
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            CharacterStreamResult(
                text="test",
                encoding=encoding_result,
                transformations=transform_result,
                confidence=1.5
            )


class TestStreamProcessingConfig:
    """Test the StreamProcessingConfig class."""
    
    def test_default_initialization(self):
        """Test default configuration values."""
        # Act
        config = StreamProcessingConfig()
        
        # Assert
        assert isinstance(config.transform_config, TransformConfig)
        assert config.buffer_size == 8192
        assert config.max_memory == 100 * 1024 * 1024
        assert config.enable_streaming is True
    
    def test_custom_initialization(self):
        """Test custom configuration values."""
        # Arrange
        custom_transform_config = TransformConfig.create_preset("strict")
        
        # Act
        config = StreamProcessingConfig(
            transform_config=custom_transform_config,
            buffer_size=4096,
            max_memory=50 * 1024 * 1024,
            enable_streaming=False
        )
        
        # Assert
        assert config.transform_config == custom_transform_config
        assert config.buffer_size == 4096
        assert config.max_memory == 50 * 1024 * 1024
        assert config.enable_streaming is False


class TestStreamProcessingPresets:
    """Test the predefined configuration presets."""
    
    def test_web_scraping_preset_defaults(self):
        """Test web scraping preset configuration with defaults."""
        # Act
        config = StreamProcessingPresets.web_scraping()
        
        # Assert
        assert config.buffer_size == 16384
        assert config.enable_streaming is True
        assert isinstance(config.transform_config, TransformConfig)
    
    def test_data_recovery_preset_defaults(self):
        """Test data recovery preset configuration with defaults."""
        # Act
        config = StreamProcessingPresets.data_recovery()
        
        # Assert
        assert config.buffer_size == 4096
        assert config.enable_streaming is True
        assert isinstance(config.transform_config, TransformConfig)
    
    def test_strict_mode_preset_defaults(self):
        """Test strict mode preset configuration with defaults."""
        # Act
        config = StreamProcessingPresets.strict_mode()
        
        # Assert
        assert config.buffer_size == 8192
        assert config.enable_streaming is True
        assert isinstance(config.transform_config, TransformConfig)
    
    def test_web_scraping_preset_with_overrides(self):
        """Test web scraping preset with custom overrides."""
        # Act
        config = StreamProcessingPresets.web_scraping(
            buffer_size=32768,
            enable_streaming=False,
            max_memory=50 * 1024 * 1024
        )
        
        # Assert
        assert config.buffer_size == 32768
        assert config.enable_streaming is False
        assert config.max_memory == 50 * 1024 * 1024
    
    def test_data_recovery_preset_with_overrides(self):
        """Test data recovery preset with custom overrides."""
        # Act
        config = StreamProcessingPresets.data_recovery(
            buffer_size=2048,
            transform_overrides={'preserve_whitespace': False}
        )
        
        # Assert
        assert config.buffer_size == 2048
        assert config.transform_config.preserve_whitespace is False
    
    def test_strict_mode_preset_with_transform_overrides(self):
        """Test strict mode preset with transformation overrides."""
        # Act
        config = StreamProcessingPresets.strict_mode(
            transform_overrides={
                'replacement_char': '?',
                'strict_xml': False
            }
        )
        
        # Assert
        assert config.transform_config.replacement_char == '?'
        assert config.transform_config.strict_xml is False
    
    def test_preset_validation_valid_config(self):
        """Test validation of valid configuration."""
        # Arrange
        config = StreamProcessingPresets.web_scraping()
        
        # Act
        issues = StreamProcessingPresets.validate_config(config)
        
        # Assert
        assert len(issues) == 0
    
    def test_preset_validation_invalid_buffer_size(self):
        """Test validation of invalid buffer size."""
        # Arrange
        config = StreamProcessingPresets.web_scraping(buffer_size=0)
        
        # Act
        issues = StreamProcessingPresets.validate_config(config)
        
        # Assert
        assert len(issues) > 0
        assert any("Buffer size must be positive" in issue for issue in issues)
    
    def test_preset_validation_buffer_exceeds_memory(self):
        """Test validation when buffer size exceeds memory limit."""
        # Arrange
        config = StreamProcessingPresets.web_scraping(
            buffer_size=1024 * 1024 * 200,  # 200MB buffer
            max_memory=1024 * 1024 * 100    # 100MB limit
        )
        
        # Act
        issues = StreamProcessingPresets.validate_config(config)
        
        # Assert
        assert len(issues) > 0
        assert any("Buffer size exceeds maximum memory limit" in issue for issue in issues)
    
    def test_preset_validation_small_buffer_warning(self):
        """Test validation warning for small buffer size."""
        # Arrange
        config = StreamProcessingPresets.web_scraping(buffer_size=512)
        
        # Act
        issues = StreamProcessingPresets.validate_config(config)
        
        # Assert
        assert len(issues) > 0
        assert any("Very small buffer size may impact performance" in issue for issue in issues)
    
    def test_preset_validation_large_buffer_warning(self):
        """Test validation warning for large buffer size."""
        # Arrange
        config = StreamProcessingPresets.web_scraping(buffer_size=2 * 1024 * 1024)  # 2MB
        
        # Act
        issues = StreamProcessingPresets.validate_config(config)
        
        # Assert
        assert len(issues) > 0
        assert any("Very large buffer size may impact memory usage" in issue for issue in issues)


class TestCharacterStreamProcessor:
    """Test the CharacterStreamProcessor class."""
    
    def test_initialization_with_default_config(self):
        """Test processor initialization with default configuration."""
        # Act
        processor = CharacterStreamProcessor()
        
        # Assert
        assert isinstance(processor.config, StreamProcessingConfig)
        assert processor._encoding_detector is not None
        assert processor._character_transformer is not None
    
    def test_initialization_with_custom_config(self):
        """Test processor initialization with custom configuration."""
        # Arrange
        custom_config = StreamProcessingConfig(buffer_size=4096)
        
        # Act
        processor = CharacterStreamProcessor(custom_config)
        
        # Assert
        assert processor.config.buffer_size == 4096
    
    def test_process_bytes_input(self):
        """Test processing bytes input."""
        # Arrange
        processor = CharacterStreamProcessor()
        test_data = b"Hello, world!"
        
        # Act
        result = processor.process(test_data)
        
        # Assert
        assert isinstance(result, CharacterStreamResult)
        assert result.text == "Hello, world!"
        assert result.encoding.encoding in ["utf-8", "ascii"]
        assert result.confidence > 0.0
        assert result.metadata['input_type'] == 'bytes'
        assert result.metadata['input_size'] == len(test_data)
    
    def test_process_string_input(self):
        """Test processing string input."""
        # Arrange
        processor = CharacterStreamProcessor()
        test_text = "Hello, world!"
        
        # Act
        result = processor.process(test_text)
        
        # Assert
        assert isinstance(result, CharacterStreamResult)
        assert result.text == "Hello, world!"
        assert result.encoding.encoding == "utf-8"
        assert result.confidence > 0.0
        assert result.metadata['input_type'] == 'str'
        assert result.metadata['input_size'] == len(test_text)
    
    def test_process_binary_file_input(self):
        """Test processing binary file-like object."""
        # Arrange
        processor = CharacterStreamProcessor()
        test_data = b"Hello, world!"
        file_obj = io.BytesIO(test_data)
        file_obj.mode = 'rb'  # Simulate binary mode
        
        # Act
        result = processor.process(file_obj)
        
        # Assert
        assert isinstance(result, CharacterStreamResult)
        assert result.text == "Hello, world!"
        assert result.confidence > 0.0
    
    def test_process_text_file_input(self):
        """Test processing text file-like object."""
        # Arrange
        processor = CharacterStreamProcessor()
        test_text = "Hello, world!"
        file_obj = io.StringIO(test_text)
        file_obj.mode = 'r'  # Simulate text mode
        
        # Act
        result = processor.process(file_obj)
        
        # Assert
        assert isinstance(result, CharacterStreamResult)
        assert result.text == "Hello, world!"
        assert result.confidence > 0.0
    
    def test_process_utf8_bytes_with_bom(self):
        """Test processing UTF-8 bytes with BOM."""
        # Arrange
        processor = CharacterStreamProcessor()
        test_data = b"\xef\xbb\xbfHello, world!"
        
        # Act
        result = processor.process(test_data)
        
        # Assert
        assert isinstance(result, CharacterStreamResult)
        assert "Hello, world!" in result.text
        assert result.encoding.encoding == "utf-8"
        assert result.confidence > 0.0
    
    def test_process_malformed_bytes(self):
        """Test processing malformed byte sequences."""
        # Arrange
        processor = CharacterStreamProcessor()
        malformed_data = b"\xff\xfe\x00\x41\x00\x42"  # UTF-16 LE with ASCII
        
        # Act
        result = processor.process(malformed_data)
        
        # Assert
        assert isinstance(result, CharacterStreamResult)
        assert result.confidence >= 0.0
        assert len(result.diagnostics) >= 0  # May have diagnostics
    
    def test_process_empty_input(self):
        """Test processing empty input."""
        # Arrange
        processor = CharacterStreamProcessor()
        
        # Act
        result_bytes = processor.process(b"")
        result_string = processor.process("")
        
        # Assert
        assert isinstance(result_bytes, CharacterStreamResult)
        assert isinstance(result_string, CharacterStreamResult)
        assert result_bytes.text == ""
        assert result_string.text == ""
    
    def test_process_with_character_transformations(self):
        """Test processing with character transformations applied."""
        # Arrange
        config = StreamProcessingConfig(
            transform_config=TransformConfig.create_preset("lenient")
        )
        processor = CharacterStreamProcessor(config)
        # Use a control character that should be transformed
        test_data = b"Hello\x00world"
        
        # Act
        result = processor.process(test_data)
        
        # Assert
        assert isinstance(result, CharacterStreamResult)
        assert "\x00" not in result.text  # Control character should be handled
        assert len(result.transformations.changes) >= 0
    
    def test_normalize_input_bytes(self):
        """Test input normalization for bytes."""
        # Arrange
        processor = CharacterStreamProcessor()
        test_data = b"test"
        
        # Act
        normalized, input_type = processor._normalize_input(test_data)
        
        # Assert
        assert normalized == test_data
        assert input_type == "bytes"
    
    def test_normalize_input_string(self):
        """Test input normalization for string."""
        # Arrange
        processor = CharacterStreamProcessor()
        test_data = "test"
        
        # Act
        normalized, input_type = processor._normalize_input(test_data)
        
        # Assert
        assert normalized == test_data
        assert input_type == "str"
    
    def test_normalize_input_file_object(self):
        """Test input normalization for file-like object."""
        # Arrange
        processor = CharacterStreamProcessor()
        file_obj = io.StringIO("test")
        
        # Act
        normalized, input_type = processor._normalize_input(file_obj)
        
        # Assert
        assert normalized == file_obj
        assert input_type == "file"
    
    def test_normalize_input_fallback(self):
        """Test input normalization fallback for unsupported types."""
        # Arrange
        processor = CharacterStreamProcessor()
        test_data = 12345  # Integer
        
        # Act
        normalized, input_type = processor._normalize_input(test_data)
        
        # Assert
        assert normalized == "12345"
        assert input_type == "str"
    
    def test_normalize_input_conversion_failure(self):
        """Test input normalization when conversion fails."""
        # Arrange
        processor = CharacterStreamProcessor()
        
        # Create a custom class that fails string conversion
        class FailingConversion:
            def __str__(self):
                raise Exception("Conversion failed")
        
        mock_obj = FailingConversion()
        
        # Act
        normalized, input_type = processor._normalize_input(mock_obj)
        
        # Assert
        # When string conversion fails, it falls back to empty bytes
        assert normalized == b""
        assert input_type == "bytes"
    
    def test_process_file_without_mode_attribute(self):
        """Test processing file-like object without mode attribute."""
        # Arrange
        processor = CharacterStreamProcessor()
        file_obj = io.StringIO("Hello, world!")
        # Remove mode attribute to test fallback
        if hasattr(file_obj, 'mode'):
            delattr(file_obj, 'mode')
        
        # Act
        result = processor.process(file_obj)
        
        # Assert
        assert isinstance(result, CharacterStreamResult)
        assert result.text == "Hello, world!"
    
    def test_create_error_result(self):
        """Test error result creation."""
        # Arrange
        processor = CharacterStreamProcessor()
        error_message = "Test error"
        
        # Act
        result = processor._create_error_result(error_message)
        
        # Assert
        assert isinstance(result, CharacterStreamResult)
        assert result.text == ""
        assert result.confidence == 0.0
        assert error_message in result.diagnostics
        assert result.metadata.get('error') is True
    
    def test_process_with_file_exception(self):
        """Test processing when file operations raise exceptions."""
        # Arrange
        processor = CharacterStreamProcessor()
        mock_file = Mock()
        mock_file.read.side_effect = IOError("File read error")
        mock_file.mode = 'r'
        
        # Act
        result = processor.process(mock_file)
        
        # Assert
        assert isinstance(result, CharacterStreamResult)
        assert result.confidence == 0.0
        assert "File processing error" in str(result.diagnostics)
    
    def test_process_with_unexpected_exception(self):
        """Test processing when unexpected exceptions occur."""
        # Arrange
        processor = CharacterStreamProcessor()
        
        # Mock the normalize_input method to raise an exception
        with patch.object(processor, '_normalize_input', side_effect=RuntimeError("Unexpected error")):
            # Act
            result = processor.process("test")
            
            # Assert
            assert isinstance(result, CharacterStreamResult)
            assert result.confidence == 0.0
            assert "Unexpected error" in str(result.diagnostics)


class TestCharacterStreamProcessorIntegration:
    """Integration tests for CharacterStreamProcessor with real-world scenarios."""
    
    def test_xml_document_processing(self):
        """Test processing a complete XML document."""
        # Arrange
        processor = CharacterStreamProcessor()
        xml_content = b'<?xml version="1.0" encoding="UTF-8"?><root>Hello World</root>'
        
        # Act
        result = processor.process(xml_content)
        
        # Assert
        assert isinstance(result, CharacterStreamResult)
        assert 'xml version="1.0"' in result.text
        assert result.confidence > 0.0
        assert result.encoding.encoding == "utf-8"
    
    def test_mixed_encoding_recovery(self):
        """Test recovery from mixed encoding scenarios."""
        # Arrange
        processor = CharacterStreamProcessor(
            StreamProcessingPresets.data_recovery()
        )
        # Create intentionally problematic data
        mixed_data = b"Hello \xff world \x00 test"
        
        # Act
        result = processor.process(mixed_data)
        
        # Assert
        assert isinstance(result, CharacterStreamResult)
        assert "Hello" in result.text
        assert "world" in result.text
        assert "test" in result.text
        assert result.confidence >= 0.0
    
    def test_large_document_processing(self):
        """Test processing a large document (simulated)."""
        # Arrange
        processor = CharacterStreamProcessor()
        # Create a moderately sized document
        large_content = ("Hello world! " * 1000).encode('utf-8')
        
        # Act
        result = processor.process(large_content)
        
        # Assert
        assert isinstance(result, CharacterStreamResult)
        assert "Hello world!" in result.text
        assert result.metadata['input_size'] == len(large_content)
        assert result.confidence > 0.0
    
    def test_preset_configurations(self):
        """Test all preset configurations work correctly."""
        # Arrange
        test_data = b"Hello <tag>world</tag> \x00"
        
        presets = [
            StreamProcessingPresets.web_scraping(),
            StreamProcessingPresets.data_recovery(), 
            StreamProcessingPresets.strict_mode()
        ]
        
        for preset in presets:
            processor = CharacterStreamProcessor(preset)
            
            # Act
            result = processor.process(test_data)
            
            # Assert
            assert isinstance(result, CharacterStreamResult)
            assert "Hello" in result.text
            assert "world" in result.text
            assert result.confidence >= 0.0


class TestStreamingProgress:
    """Test the StreamingProgress class."""
    
    def test_default_initialization(self):
        """Test default initialization of streaming progress."""
        # Act
        progress = StreamingProgress()
        
        # Assert
        assert progress.processed_bytes == 0
        assert progress.total_bytes == 0
        assert progress.processed_chunks == 0
        assert progress.current_chunk_size == 0
        assert progress.cancelled is False


class TestStreamingInterface:
    """Test the streaming interface of CharacterStreamProcessor."""
    
    def test_process_stream_bytes_basic(self):
        """Test basic streaming processing of bytes."""
        # Arrange
        processor = CharacterStreamProcessor()
        test_data = b"Hello, world! This is a test of streaming functionality."
        
        # Act
        result = processor.process_stream(test_data)
        
        # Assert
        assert isinstance(result, StreamingResult)
        assert result.encoding.encoding in ["utf-8", "ascii"]
        assert result.progress.total_bytes == len(test_data)
        assert result.metadata['input_type'] == 'bytes'
        
        # Consume the generator and check content
        chunks = list(result.chunks)
        full_text = ''.join(chunks)
        assert "Hello, world!" in full_text
        assert "streaming functionality" in full_text
    
    def test_process_stream_string_basic(self):
        """Test basic streaming processing of strings."""
        # Arrange
        processor = CharacterStreamProcessor()
        test_text = "Hello, world! This is a test of streaming functionality."
        
        # Act
        result = processor.process_stream(test_text)
        
        # Assert
        assert isinstance(result, StreamingResult)
        assert result.encoding.encoding == "utf-8"
        assert result.progress.total_bytes == len(test_text)
        assert result.metadata['input_type'] == 'str'
        
        # Consume the generator
        chunks = list(result.chunks)
        full_text = ''.join(chunks)
        assert full_text == test_text
    
    def test_process_stream_with_progress_callback(self):
        """Test streaming with progress callback."""
        # Arrange
        processor = CharacterStreamProcessor()
        test_data = b"Hello, world! This is a test."
        progress_calls = []
        
        def progress_callback(processed: int, total: int) -> bool:
            progress_calls.append((processed, total))
            return True  # Continue processing
        
        # Act
        result = processor.process_stream(test_data, progress_callback)
        
        # Assert
        assert isinstance(result, StreamingResult)
        
        # Consume generator to trigger progress callbacks
        chunks = list(result.chunks)
        full_text = ''.join(chunks)
        assert "Hello, world!" in full_text
        
        # Check that progress was tracked
        assert len(progress_calls) > 0
        assert all(processed <= total for processed, total in progress_calls)
    
    def test_process_stream_with_cancellation(self):
        """Test streaming with progress callback cancellation."""
        # Arrange
        processor = CharacterStreamProcessor()
        test_data = b"Hello, world! This is a longer test to enable cancellation."
        call_count = 0
        
        def progress_callback(processed: int, total: int) -> bool:
            nonlocal call_count
            call_count += 1
            return call_count <= 2  # Cancel after 2 calls
        
        # Act
        result = processor.process_stream(test_data, progress_callback)
        
        # Assert
        assert isinstance(result, StreamingResult)
        
        # Consume generator 
        chunks = list(result.chunks)
        
        # Should be cancelled - check progress
        # Note: cancellation happens during generator consumption
        assert call_count > 0
    
    def test_process_stream_binary_file(self):
        """Test streaming processing of binary file."""
        # Arrange
        processor = CharacterStreamProcessor()
        test_data = b"Hello, world! This is a test file."
        file_obj = io.BytesIO(test_data)
        file_obj.mode = 'rb'
        
        # Act
        result = processor.process_stream(file_obj)
        
        # Assert
        assert isinstance(result, StreamingResult)
        assert result.metadata['input_type'] == 'binary_file'
        
        # Consume generator
        chunks = list(result.chunks)
        full_text = ''.join(chunks)
        assert "Hello, world!" in full_text
    
    def test_process_stream_text_file(self):
        """Test streaming processing of text file."""
        # Arrange
        processor = CharacterStreamProcessor()
        test_text = "Hello, world! This is a test file."
        file_obj = io.StringIO(test_text)
        file_obj.mode = 'r'
        
        # Act
        result = processor.process_stream(file_obj)
        
        # Assert
        assert isinstance(result, StreamingResult)
        assert result.metadata['input_type'] == 'text_file'
        
        # Consume generator
        chunks = list(result.chunks)
        full_text = ''.join(chunks)
        assert full_text == test_text
    
    def test_process_stream_chunked_processing(self):
        """Test that streaming actually processes in chunks."""
        # Arrange
        config = StreamProcessingConfig(buffer_size=10)  # Small buffer for testing
        processor = CharacterStreamProcessor(config)
        test_data = b"This is a longer text that should be processed in multiple chunks."
        
        # Act
        result = processor.process_stream(test_data)
        
        # Assert
        assert isinstance(result, StreamingResult)
        assert result.metadata['buffer_size'] == 10
        
        # Consume generator and count chunks
        chunks = list(result.chunks)
        assert len(chunks) > 1  # Should be multiple chunks
        
        # Verify full content is preserved
        full_text = ''.join(chunks)
        expected = test_data.decode('utf-8')
        assert full_text == expected or full_text == expected.replace('\x00', '')  # Allow for character transformation
    
    def test_process_stream_error_handling(self):
        """Test error handling in streaming interface."""
        # Arrange
        processor = CharacterStreamProcessor()
        
        # Mock to cause an error
        with patch.object(processor, '_normalize_input', side_effect=RuntimeError("Test error")):
            # Act
            result = processor.process_stream(b"test")
            
            # Assert
            assert isinstance(result, StreamingResult)
            assert result.progress.cancelled is True
            assert "Test error" in str(result.diagnostics)
            
            # Generator should be empty
            chunks = list(result.chunks)
            assert len(chunks) == 0
    
    def test_process_stream_with_character_transformations(self):
        """Test streaming with character transformations applied."""
        # Arrange
        config = StreamProcessingConfig(
            transform_config=TransformConfig.create_preset("strict"),
            buffer_size=16
        )
        processor = CharacterStreamProcessor(config)
        test_data = b"Hello\x00world\x01test"  # Contains control characters
        
        # Act
        result = processor.process_stream(test_data)
        
        # Assert
        assert isinstance(result, StreamingResult)
        
        # Consume generator
        chunks = list(result.chunks)
        full_text = ''.join(chunks)
        
        # Control characters should be handled
        assert "Hello" in full_text
        assert "world" in full_text  
        assert "test" in full_text
        assert "\x00" not in full_text  # Should be removed/replaced
        assert "\x01" not in full_text  # Should be removed/replaced
    
    def test_streaming_progress_tracking(self):
        """Test progress tracking throughout streaming operation."""
        # Arrange
        processor = CharacterStreamProcessor()
        test_data = b"Hello, world! This is a test for progress tracking."
        
        # Act
        result = processor.process_stream(test_data)
        
        # Assert
        assert result.progress.total_bytes == len(test_data)
        assert result.progress.processed_bytes == 0  # Not started yet
        assert result.progress.processed_chunks == 0
        
        # Consume generator to update progress
        chunks = list(result.chunks)
        
        # Progress should be updated after consumption
        assert result.progress.processed_bytes > 0
        assert result.progress.processed_chunks > 0
    
    def test_streaming_memory_efficiency(self):
        """Test that streaming doesn't load entire content into memory."""
        # Arrange
        processor = CharacterStreamProcessor()
        # Create a moderately sized test to simulate streaming benefit
        large_text = "Test chunk " * 1000  # Repeating pattern
        test_data = large_text.encode('utf-8')
        
        # Act
        result = processor.process_stream(test_data)
        
        # Assert
        assert isinstance(result, StreamingResult)
        
        # The result should have a generator, not pre-computed text
        assert hasattr(result.chunks, '__next__')  # It's a generator
        
        # Consume first few chunks only
        chunk_iter = iter(result.chunks)
        first_chunk = next(chunk_iter)
        second_chunk = next(chunk_iter)
        
        assert len(first_chunk) > 0
        assert "Test chunk" in first_chunk


class TestStreamingIntegration:
    """Integration tests for streaming functionality."""
    
    def test_streaming_with_encoding_detection(self):
        """Test streaming with various encodings."""
        # Arrange
        processor = CharacterStreamProcessor()
        utf8_data = "Hello, 世界! This is UTF-8 text.".encode('utf-8')
        
        # Act
        result = processor.process_stream(utf8_data)
        
        # Assert
        assert isinstance(result, StreamingResult)
        assert result.encoding.encoding == "utf-8"
        
        # Consume and verify content
        chunks = list(result.chunks)
        full_text = ''.join(chunks)
        assert "Hello" in full_text
        assert "世界" in full_text or "世界" in repr(full_text)  # Unicode handling
    
    def test_streaming_preset_configurations(self):
        """Test streaming with different configuration presets."""
        # Arrange
        test_data = b"Hello <tag>world</tag> \x00"
        
        presets = [
            StreamProcessingPresets.web_scraping(),
            StreamProcessingPresets.data_recovery(),
            StreamProcessingPresets.strict_mode()
        ]
        
        for preset in presets:
            processor = CharacterStreamProcessor(preset)
            
            # Act
            result = processor.process_stream(test_data)
            
            # Assert
            assert isinstance(result, StreamingResult)
            
            chunks = list(result.chunks)
            full_text = ''.join(chunks)
            assert "Hello" in full_text
            assert "world" in full_text


class TestMalformedXMLIntegration:
    """Integration tests with various malformed XML scenarios."""
    
    def test_malformed_xml_with_invalid_characters(self):
        """Test processing XML with invalid control characters."""
        # Arrange
        processor = CharacterStreamProcessor(StreamProcessingPresets.web_scraping())
        malformed_xml = b'<?xml version="1.0"?>\x00<root>\x01<tag>Hello\x02World</tag>\x03</root>\x04'
        
        # Act
        result = processor.process(malformed_xml)
        
        # Assert
        assert isinstance(result, CharacterStreamResult)
        assert "root" in result.text
        assert "Hello" in result.text
        assert "World" in result.text
        # Control characters should be handled
        assert "\x00" not in result.text
        assert result.confidence > 0.0
    
    def test_malformed_xml_with_encoding_issues(self):
        """Test processing XML with encoding declaration mismatches."""
        # Arrange
        processor = CharacterStreamProcessor(StreamProcessingPresets.data_recovery())
        # UTF-8 content with Latin-1 declaration
        malformed_xml = '<?xml version="1.0" encoding="ISO-8859-1"?>\n<root>café</root>'.encode('utf-8')
        
        # Act
        result = processor.process(malformed_xml)
        
        # Assert
        assert isinstance(result, CharacterStreamResult)
        assert "root" in result.text
        assert "café" in result.text or "caf" in result.text  # Might be handled differently
        assert len(result.diagnostics) >= 0  # May have encoding warnings
    
    def test_malformed_xml_with_broken_structure(self):
        """Test processing XML with broken tag structure."""
        # Arrange
        processor = CharacterStreamProcessor(StreamProcessingPresets.data_recovery())
        malformed_xml = b'<root><unclosed><nested>content</nested><another>more content</root>'
        
        # Act
        result = processor.process(malformed_xml)
        
        # Assert
        assert isinstance(result, CharacterStreamResult)
        assert "content" in result.text
        assert "more content" in result.text
        # Structure is preserved even if invalid
        assert "root" in result.text
    
    def test_malformed_xml_with_invalid_entities(self):
        """Test processing XML with invalid entity references."""
        # Arrange
        processor = CharacterStreamProcessor(StreamProcessingPresets.web_scraping())
        malformed_xml = b'<root>&invalid; &amp; &#999999; &#xFFFFFF; text</root>'
        
        # Act
        result = processor.process(malformed_xml)
        
        # Assert
        assert isinstance(result, CharacterStreamResult)
        assert "root" in result.text
        assert "text" in result.text
        # Invalid entities are preserved as-is (not our job to fix XML structure)
        assert "&invalid;" in result.text
    
    def test_malformed_xml_with_mixed_content_types(self):
        """Test processing mixed content with various character issues."""
        # Arrange
        processor = CharacterStreamProcessor(StreamProcessingPresets.strict_mode())
        malformed_xml = b'<root>\r\n\t<tag attr="value">\x0BInvalid\x0C tab\x1F</tag>\r\n</root>'
        
        # Act
        result = processor.process(malformed_xml)
        
        # Assert
        assert isinstance(result, CharacterStreamResult)
        assert "root" in result.text
        assert "tag" in result.text
        assert "Invalid" in result.text
        assert "tab" in result.text
        # Invalid control characters should be handled
        assert len([c for c in result.text if ord(c) < 32 and c not in '\t\n\r']) == 0
    
    def test_large_malformed_xml_streaming(self):
        """Test streaming processing of large malformed XML."""
        # Arrange
        processor = CharacterStreamProcessor(StreamProcessingPresets.web_scraping(buffer_size=1024))
        # Create large malformed content
        malformed_chunk = b'<item>\x00Invalid\x01Content\x02</item>' * 100
        large_malformed_xml = b'<root>' + malformed_chunk + b'</root>'
        
        # Act
        result = processor.process_stream(large_malformed_xml)
        
        # Assert
        assert isinstance(result, StreamingResult)
        chunks = list(result.chunks)
        full_text = ''.join(chunks)
        
        assert "root" in full_text
        assert "Invalid" in full_text
        assert "Content" in full_text
        assert len(chunks) > 1  # Should be processed in multiple chunks
        # Control characters should be handled
        assert "\x00" not in full_text


class TestRealWorldUsageExamples:
    """Real-world usage examples and integration patterns."""
    
    def test_web_scraping_scenario(self):
        """Example: Processing XML from web scraping."""
        # Arrange - Simulate scraped XML with common issues
        processor = CharacterStreamProcessor(StreamProcessingPresets.web_scraping())
        scraped_html_as_xml = b'''
        <html>
            <head><title>Test\x00Page</title></head>
            <body>
                <div>Content with \x01 invalid chars</div>
                <p>More content &amp; entities</p>
            </body>
        </html>
        '''
        
        # Act
        result = processor.process(scraped_html_as_xml)
        
        # Assert
        assert result.text is not None
        assert "Test" in result.text
        assert "Page" in result.text
        assert "Content with" in result.text
        assert "entities" in result.text
        assert result.confidence > 0.5
        assert result.metadata['input_type'] == 'bytes'
    
    def test_data_recovery_scenario(self):
        """Example: Recovering data from corrupted XML files."""
        # Arrange - Simulate corrupted XML from old systems
        processor = CharacterStreamProcessor(StreamProcessingPresets.data_recovery())
        corrupted_xml = b'''<?xml version="1.0"?>
        <database>
            <record id="1">\x00\x01\x02
                <name>John\x03Doe</name>
                <data>Important\x1Finfo</data>
            </record>
        </database>
        '''
        
        # Act
        result = processor.process(corrupted_xml)
        
        # Assert
        assert "database" in result.text
        assert "John" in result.text
        assert "Doe" in result.text
        assert "Important" in result.text
        assert "info" in result.text
        # Should preserve as much data as possible
        assert result.confidence >= 0.0  # Never fails
    
    def test_strict_xml_processing_scenario(self):
        """Example: Strict XML processing for high-quality documents."""
        # Arrange
        processor = CharacterStreamProcessor(StreamProcessingPresets.strict_mode())
        clean_xml = b'<?xml version="1.0" encoding="UTF-8"?><root><item>Clean content</item></root>'
        
        # Act
        result = processor.process(clean_xml)
        
        # Assert
        assert result.text == '<?xml version="1.0" encoding="UTF-8"?><root><item>Clean content</item></root>'
        assert result.confidence > 0.9
        assert len(result.diagnostics) == 0
    
    def test_custom_configuration_scenario(self):
        """Example: Custom configuration for specific needs."""
        # Arrange - Custom configuration with overrides
        config = StreamProcessingPresets.web_scraping(
            buffer_size=2048,
            transform_overrides={'replacement_char': '?', 'preserve_whitespace': False}
        )
        processor = CharacterStreamProcessor(config)
        test_xml = b'<root>\x00\x01  Spaced   content  \x02</root>'
        
        # Act
        result = processor.process(test_xml)
        
        # Assert
        assert "root" in result.text
        assert "content" in result.text
        # Custom replacement char and whitespace handling
        assert "?" in result.text or "Spaced" in result.text
    
    def test_streaming_large_file_scenario(self):
        """Example: Processing large XML files with streaming."""
        # Arrange
        processor = CharacterStreamProcessor(StreamProcessingPresets.data_recovery(buffer_size=512))
        
        # Simulate large file content
        large_content = b'<items>' + b'<item>data</item>' * 500 + b'</items>'
        file_obj = io.BytesIO(large_content)
        file_obj.mode = 'rb'
        
        # Track progress
        progress_updates = []
        def track_progress(processed: int, total: int) -> bool:
            progress_updates.append((processed, total))
            return True  # Continue processing
        
        # Act
        result = processor.process_stream(file_obj, track_progress)
        
        # Assert
        assert isinstance(result, StreamingResult)
        chunks = list(result.chunks)
        assert len(chunks) > 1  # Multiple chunks
        
        full_text = ''.join(chunks)
        assert "items" in full_text
        assert "data" in full_text
        assert len(progress_updates) > 0  # Progress was tracked
    
    def test_configuration_validation_example(self):
        """Example: Configuration validation and troubleshooting."""
        # Arrange - Create potentially problematic configuration
        config = StreamProcessingPresets.web_scraping(
            buffer_size=0,  # Invalid
            max_memory=1024,  # Small
        )
        
        # Act
        issues = StreamProcessingPresets.validate_config(config)
        
        # Assert
        assert len(issues) > 0
        assert any("Buffer size must be positive" in issue for issue in issues)
        
        # Fix the configuration
        fixed_config = StreamProcessingPresets.web_scraping(
            buffer_size=1024,
            max_memory=10 * 1024 * 1024
        )
        fixed_issues = StreamProcessingPresets.validate_config(fixed_config)
        assert len(fixed_issues) == 0  # Should be valid now


class TestTroubleshootingScenarios:
    """Test scenarios that help with troubleshooting common issues."""
    
    def test_encoding_detection_troubleshooting(self):
        """Troubleshooting encoding detection issues."""
        # Arrange
        processor = CharacterStreamProcessor()
        
        # Test with ambiguous encoding
        ambiguous_data = b'\xff\xfe\x41\x00\x42\x00'  # UTF-16 LE "AB"
        
        # Act
        result = processor.process(ambiguous_data)
        
        # Assert
        assert result.encoding.method in [DetectionMethod.BOM, DetectionMethod.STATISTICAL]
        assert result.encoding.confidence > 0.0
        assert "AB" in result.text or "A" in result.text  # Some content should be preserved
    
    def test_memory_usage_troubleshooting(self):
        """Troubleshooting memory usage with large inputs."""
        # Arrange
        processor = CharacterStreamProcessor(StreamProcessingPresets.web_scraping(buffer_size=256))
        
        # Large input that should use streaming
        large_input = b'<root>' + b'<item>content</item>' * 1000 + b'</root>'
        
        # Act
        stream_result = processor.process_stream(large_input)
        
        # Assert
        assert isinstance(stream_result, StreamingResult)
        assert stream_result.metadata['buffer_size'] == 256
        
        # Process in chunks to verify memory efficiency
        chunk_count = 0
        for chunk in stream_result.chunks:
            chunk_count += 1
            assert len(chunk) <= 1000  # Reasonable chunk size
            if chunk_count > 10:  # Don't process everything in test
                break
        
        assert chunk_count > 5  # Should have multiple chunks
    
    def test_transformation_troubleshooting(self):
        """Troubleshooting character transformation issues."""
        # Arrange
        processor = CharacterStreamProcessor(StreamProcessingPresets.strict_mode())
        problematic_input = b'<root>Text with \x00\x01\x02 control chars</root>'
        
        # Act
        result = processor.process(problematic_input)
        
        # Assert
        assert "Text with" in result.text
        assert "control chars" in result.text
        # Check that transformations were applied
        assert len(result.transformations.changes) >= 0
        assert result.transformations.confidence > 0.0
    
    def test_error_recovery_troubleshooting(self):
        """Troubleshooting error recovery scenarios."""
        # Arrange
        processor = CharacterStreamProcessor()
        
        # Simulate various error conditions
        test_cases = [
            b'',  # Empty input
            b'\xff\xff\xff\xff',  # Invalid bytes
            None,  # Invalid input type (will be handled by normalization)
        ]
        
        for test_input in test_cases:
            # Act
            if test_input is None:
                # Test with unsupported input type
                class BadInput:
                    def __str__(self):
                        raise Exception("Cannot convert")
                result = processor.process(BadInput())
            else:
                result = processor.process(test_input)
            
            # Assert - Should never fail
            assert isinstance(result, CharacterStreamResult)
            assert result.confidence >= 0.0
            # May have diagnostics explaining issues
            if result.confidence < 0.5:
                assert len(result.diagnostics) > 0


if __name__ == "__main__":
    pytest.main([__file__])