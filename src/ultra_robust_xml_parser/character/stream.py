"""Character stream processing API with never-fail guarantee.

This module provides a high-level API for character stream processing that integrates
encoding detection and character transformation with comprehensive result objects.
"""

from dataclasses import dataclass, field
from typing import (
    Any,
    BinaryIO,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    TextIO,
    Union,
)

from .encoding import DetectionMethod, EncodingDetector, EncodingResult
from .transformation import (
    CharacterTransformer,
    TransformConfig,
    TransformResult,
    TransformationStrategy,
)

# Type definitions for input data
InputType = Union[bytes, str, BinaryIO, TextIO]
ProgressCallback = Callable[[int, int], bool]  # (processed, total) -> continue

# Configuration presets
DEFAULT_BUFFER_SIZE = 8192
MAX_MEMORY_SIZE = 100 * 1024 * 1024  # 100MB

# Buffer size thresholds for validation
MIN_BUFFER_SIZE = 1024  # 1KB
MAX_BUFFER_SIZE = 1024 * 1024  # 1MB


@dataclass
class CharacterStreamResult:
    """Result of character stream processing with comprehensive metadata.

    Attributes:
        text: Processed character stream as string
        encoding: Encoding detection result with confidence and method
        transformations: Character transformation result with changes and statistics
        confidence: Overall processing confidence (0.0-1.0)
        diagnostics: List of diagnostic messages from processing
        metadata: Additional processing metadata
    """
    text: str
    encoding: EncodingResult
    transformations: TransformResult
    confidence: float
    diagnostics: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate confidence score range."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Confidence must be between 0.0 and 1.0, got {self.confidence}"
            )


@dataclass
class StreamingProgress:
    """Progress information for streaming operations.

    Attributes:
        processed_bytes: Number of bytes processed so far
        total_bytes: Total bytes to process (0 if unknown)
        processed_chunks: Number of chunks processed
        current_chunk_size: Size of current chunk being processed
        cancelled: Whether the operation was cancelled
    """
    processed_bytes: int = 0
    total_bytes: int = 0
    processed_chunks: int = 0
    current_chunk_size: int = 0
    cancelled: bool = False


@dataclass
class StreamingResult:
    """Result of streaming character processing.

    Attributes:
        chunks: Generator yielding processed text chunks
        encoding: Final encoding detection result
        progress: Progress tracking information
        diagnostics: List of diagnostic messages
        metadata: Additional processing metadata
    """
    chunks: Generator[str, None, None]
    encoding: EncodingResult
    progress: StreamingProgress
    diagnostics: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class StreamProcessingConfig:
    """Configuration for character stream processing."""

    def __init__(
        self,
        transform_config: Optional[TransformConfig] = None,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        max_memory: int = MAX_MEMORY_SIZE,
        enable_streaming: bool = True,
    ) -> None:
        """Initialize stream processing configuration.

        Args:
            transform_config: Character transformation configuration
            buffer_size: Buffer size for streaming operations
            max_memory: Maximum memory usage before switching to streaming
            enable_streaming: Whether to enable streaming for large inputs
        """
        self.transform_config = transform_config or TransformConfig()
        self.buffer_size = buffer_size
        self.max_memory = max_memory
        self.enable_streaming = enable_streaming


class CharacterStreamProcessor:
    """Main character stream processor with never-fail guarantee.

    Provides a simple API for processing character streams with automatic encoding
    detection, character transformation, and comprehensive diagnostics.
    """

    def __init__(self, config: Optional[StreamProcessingConfig] = None) -> None:
        """Initialize the character stream processor.

        Args:
            config: Stream processing configuration
        """
        self.config = config or StreamProcessingConfig()
        self._encoding_detector = EncodingDetector()
        self._character_transformer = CharacterTransformer(self.config.transform_config)

    def process(self, input_data: InputType) -> CharacterStreamResult:
        """Process character stream with encoding detection and transformation.

        Args:
            input_data: Input data as bytes, string, or file-like object

        Returns:
            CharacterStreamResult with processed text and comprehensive metadata

        Raises:
            Never raises - follows never-fail philosophy with diagnostic information
        """
        try:
            # Step 1: Input validation and normalization
            normalized_data, input_type = self._normalize_input(input_data)

            # Step 2: Handle different input types
            if input_type == "bytes":
                return self._process_bytes(normalized_data)
            if input_type == "str":
                return self._process_string(normalized_data)
            if input_type == "file":
                return self._process_file(normalized_data)
            # Should never reach here due to normalization
            return self._create_error_result(
                f"Unsupported input type: {type(input_data)}"
            )

        except Exception as e:
            return self._create_error_result(f"Unexpected error: {e}")

    def process_stream(
        self,
        input_data: InputType,
        progress_callback: Optional[ProgressCallback] = None
    ) -> StreamingResult:
        """Process character stream with chunked streaming and progress tracking.

        Args:
            input_data: Input data as bytes, string, or file-like object
            progress_callback: Optional callback for progress updates

        Returns:
            StreamingResult with generator for processed chunks and progress info
        """
        try:
            # Initialize progress tracking
            progress = StreamingProgress()
            diagnostics: List[str] = []

            # Normalize and determine input size
            normalized_data, input_type = self._normalize_input(input_data)

            if input_type == "bytes":
                return self._process_bytes_stream(
                    normalized_data, progress, progress_callback, diagnostics
                )
            if input_type == "str":
                return self._process_string_stream(
                    normalized_data, progress, progress_callback, diagnostics
                )
            if input_type == "file":
                return self._process_file_stream(
                    normalized_data, progress, progress_callback, diagnostics
                )
            return self._create_error_streaming_result(
                f"Unsupported input type: {type(input_data)}", progress
            )

        except Exception as e:
            progress = StreamingProgress(cancelled=True)
            return self._create_error_streaming_result(
                f"Unexpected error: {e}", progress
            )

    def _normalize_input(self, input_data: InputType) -> tuple[Any, str]:
        """Normalize input data and determine its type.

        Args:
            input_data: Raw input data

        Returns:
            Tuple of (normalized_data, input_type)
        """
        if isinstance(input_data, bytes):
            return input_data, "bytes"
        if isinstance(input_data, str):
            return input_data, "str"
        if hasattr(input_data, "read"):
            # File-like object
            return input_data, "file"
        # Try to convert to string as fallback
        try:
            return str(input_data), "str"
        except Exception:
            return b"", "bytes"  # Ultimate fallback

    def _process_bytes(self, data: bytes) -> CharacterStreamResult:
        """Process bytes input with encoding detection.

        Args:
            data: Byte data to process

        Returns:
            CharacterStreamResult with processing results
        """
        diagnostics = []

        # Step 1: Encoding detection
        encoding_result = self._encoding_detector.detect(data)
        diagnostics.extend(
            [f"Encoding detection: {issue}" for issue in encoding_result.issues]
        )

        # Step 2: Decode to string
        try:
            if encoding_result.encoding:
                decoded_text = data.decode(encoding_result.encoding, errors="replace")
                decoded_check = data.decode(encoding_result.encoding, errors="ignore")
                if "replace" in str(decoded_check):
                    diagnostics.append(
                        "Some bytes could not be decoded and were replaced"
                    )
            else:
                decoded_text = data.decode("utf-8", errors="replace")
                diagnostics.append("Fallback to UTF-8 decoding with error replacement")
        except Exception as e:
            decoded_text = data.decode("utf-8", errors="replace")
            diagnostics.append(f"Decoding error: {e}, using UTF-8 fallback")

        # Step 3: Character transformation
        transform_result = self._character_transformer.transform(decoded_text)

        # Step 4: Calculate overall confidence
        overall_confidence = (
            encoding_result.confidence + transform_result.confidence
        ) / 2

        # Step 5: Combine diagnostics
        all_diagnostics = diagnostics + transform_result.issues

        return CharacterStreamResult(
            text=transform_result.text,
            encoding=encoding_result,
            transformations=transform_result,
            confidence=overall_confidence,
            diagnostics=all_diagnostics,
            metadata={
                "input_type": "bytes",
                "input_size": len(data),
                "output_size": len(transform_result.text),
            }
        )

    def _process_string(self, text: str) -> CharacterStreamResult:
        """Process string input (already decoded).

        Args:
            text: String data to process

        Returns:
            CharacterStreamResult with processing results
        """
        # Create a placeholder encoding result since we already have text
        encoding_result = EncodingResult(
            encoding="utf-8",  # Assumed since it's already a string
            confidence=1.0,
            method=DetectionMethod.FALLBACK,
            issues=[]
        )

        # Character transformation
        transform_result = self._character_transformer.transform(text)

        return CharacterStreamResult(
            text=transform_result.text,
            encoding=encoding_result,
            transformations=transform_result,
            confidence=transform_result.confidence,
            diagnostics=transform_result.issues,
            metadata={
                "input_type": "str",
                "input_size": len(text),
                "output_size": len(transform_result.text),
            }
        )

    def _process_file(self, file_obj: Union[BinaryIO, TextIO]) -> CharacterStreamResult:
        """Process file-like object input.

        Args:
            file_obj: File-like object to process

        Returns:
            CharacterStreamResult with processing results
        """
        try:
            # Determine if it's binary or text mode
            if hasattr(file_obj, "mode") and "b" in file_obj.mode:
                # Binary mode - read as bytes
                data = file_obj.read()
                if isinstance(data, bytes):
                    return self._process_bytes(data)
                return self._create_error_result("Binary file returned non-bytes data")
            # Text mode - read as string
            text = file_obj.read()
            if isinstance(text, str):
                return self._process_string(text)
            return self._create_error_result("Text file returned non-string data")

        except Exception as e:
            return self._create_error_result(f"File processing error: {e}")

    def _create_error_result(self, error_message: str) -> CharacterStreamResult:
        """Create a result object for error conditions.

        Args:
            error_message: Error description

        Returns:
            CharacterStreamResult with error information
        """

        return CharacterStreamResult(
            text="",
            encoding=EncodingResult(
                encoding="utf-8",
                confidence=0.0,
                method=DetectionMethod.FALLBACK,
                issues=[error_message]
            ),
            transformations=TransformResult(
                text="",
                confidence=0.0,
                issues=[error_message]
            ),
            confidence=0.0,
            diagnostics=[error_message],
            metadata={"error": True}
        )

    def _process_bytes_stream(
        self,
        data: bytes,
        progress: StreamingProgress,
        progress_callback: Optional[ProgressCallback],
        diagnostics: List[str]
    ) -> StreamingResult:
        """Process bytes input with streaming chunks."""
        # Set up progress tracking
        progress.total_bytes = len(data)

        # Initial encoding detection on first chunk
        detection_sample = data[:min(8192, len(data))]
        encoding_result = self._encoding_detector.detect(detection_sample)
        diagnostics.extend(
            [f"Encoding detection: {issue}" for issue in encoding_result.issues]
        )

        def chunk_generator() -> Generator[str, None, None]:
            """Generator that yields processed text chunks."""
            processed = 0
            buffer_size = self.config.buffer_size

            while processed < len(data):
                # Check for cancellation
                if progress_callback and not progress_callback(processed, len(data)):
                    progress.cancelled = True
                    break

                # Get next chunk
                chunk_end = min(processed + buffer_size, len(data))
                chunk_data = data[processed:chunk_end]
                progress.current_chunk_size = len(chunk_data)

                try:
                    # Decode chunk
                    if encoding_result.encoding:
                        decoded_chunk = chunk_data.decode(encoding_result.encoding, errors="replace")
                    else:
                        decoded_chunk = chunk_data.decode("utf-8", errors="replace")

                    # Transform chunk
                    transform_result = self._character_transformer.transform(decoded_chunk)

                    # Update progress
                    processed += len(chunk_data)
                    progress.processed_bytes = processed
                    progress.processed_chunks += 1

                    yield transform_result.text

                except Exception as e:
                    diagnostics.append(f"Chunk processing error: {e}")
                    # Yield empty string for failed chunks to maintain flow
                    yield ""
                    processed += len(chunk_data)
                    progress.processed_bytes = processed
                    progress.processed_chunks += 1

        return StreamingResult(
            chunks=chunk_generator(),
            encoding=encoding_result,
            progress=progress,
            diagnostics=diagnostics,
            metadata={
                "input_type": "bytes",
                "total_size": len(data),
                "buffer_size": self.config.buffer_size,
            }
        )

    def _process_string_stream(
        self,
        text: str,
        progress: StreamingProgress,
        progress_callback: Optional[ProgressCallback],
        diagnostics: List[str]
    ) -> StreamingResult:
        """Process string input with streaming chunks."""
        # Create placeholder encoding result for string input
        encoding_result = EncodingResult(
            encoding="utf-8",
            confidence=1.0,
            method=DetectionMethod.FALLBACK,
            issues=[]
        )

        # Set up progress tracking (using character count as "bytes")
        progress.total_bytes = len(text)

        def chunk_generator() -> Generator[str, None, None]:
            """Generator that yields processed text chunks."""
            processed = 0
            buffer_size = self.config.buffer_size

            while processed < len(text):
                # Check for cancellation
                if progress_callback and not progress_callback(processed, len(text)):
                    progress.cancelled = True
                    break

                # Get next chunk
                chunk_end = min(processed + buffer_size, len(text))
                chunk_text = text[processed:chunk_end]
                progress.current_chunk_size = len(chunk_text)

                try:
                    # Transform chunk
                    transform_result = self._character_transformer.transform(chunk_text)

                    # Update progress
                    processed += len(chunk_text)
                    progress.processed_bytes = processed
                    progress.processed_chunks += 1

                    yield transform_result.text

                except Exception as e:
                    diagnostics.append(f"Chunk processing error: {e}")
                    yield ""
                    processed += len(chunk_text)
                    progress.processed_bytes = processed
                    progress.processed_chunks += 1

        return StreamingResult(
            chunks=chunk_generator(),
            encoding=encoding_result,
            progress=progress,
            diagnostics=diagnostics,
            metadata={
                "input_type": "str",
                "total_size": len(text),
                "buffer_size": self.config.buffer_size,
            }
        )

    def _process_file_stream(
        self,
        file_obj: Union[BinaryIO, TextIO],
        progress: StreamingProgress,
        progress_callback: Optional[ProgressCallback],
        diagnostics: List[str]
    ) -> StreamingResult:
        """Process file-like object with streaming chunks."""
        try:
            # Try to get file size for progress tracking
            current_pos = file_obj.tell() if hasattr(file_obj, "tell") else 0
            if hasattr(file_obj, "seek") and hasattr(file_obj, "tell"):
                file_obj.seek(0, 2)  # Seek to end
                file_size = file_obj.tell()
                file_obj.seek(current_pos)  # Return to original position
                progress.total_bytes = file_size

            # Determine if binary or text mode
            if hasattr(file_obj, "mode") and "b" in file_obj.mode:
                # Binary mode - ensure it's actually a BinaryIO
                if hasattr(file_obj, "read") and hasattr(file_obj, "mode"):
                    # Type narrowing - we know it's binary mode at this point
                    return self._process_binary_file_stream(
                        file_obj, progress, progress_callback, diagnostics  # type: ignore
                    )
            # Text mode - ensure it's actually a TextIO
            if hasattr(file_obj, "read"):
                # Type narrowing - we know it's text mode at this point
                return self._process_text_file_stream(
                    file_obj, progress, progress_callback, diagnostics  # type: ignore
                )

            return self._create_error_streaming_result("Invalid file object", progress)

        except Exception as e:
            diagnostics.append(f"File streaming error: {e}")
            return self._create_error_streaming_result(f"File processing error: {e}", progress)

    def _process_binary_file_stream(
        self,
        file_obj: BinaryIO,
        progress: StreamingProgress,
        progress_callback: Optional[ProgressCallback],
        diagnostics: List[str]
    ) -> StreamingResult:
        """Process binary file with streaming."""
        # Read first chunk for encoding detection
        initial_chunk = file_obj.read(8192)
        if initial_chunk:
            # Reset file position
            file_obj.seek(0)

        encoding_result = self._encoding_detector.detect(initial_chunk)
        diagnostics.extend([f"Encoding detection: {issue}" for issue in encoding_result.issues])

        def chunk_generator() -> Generator[str, None, None]:
            """Generator that yields processed text chunks from binary file."""
            buffer_size = self.config.buffer_size

            while True:
                # Check for cancellation
                if progress_callback and not progress_callback(progress.processed_bytes, progress.total_bytes):
                    progress.cancelled = True
                    break

                # Read next chunk
                chunk_data = file_obj.read(buffer_size)
                if not chunk_data:
                    break

                progress.current_chunk_size = len(chunk_data)

                try:
                    # Decode chunk
                    if encoding_result.encoding:
                        decoded_chunk = chunk_data.decode(encoding_result.encoding, errors="replace")
                    else:
                        decoded_chunk = chunk_data.decode("utf-8", errors="replace")

                    # Transform chunk
                    transform_result = self._character_transformer.transform(decoded_chunk)

                    # Update progress
                    progress.processed_bytes += len(chunk_data)
                    progress.processed_chunks += 1

                    yield transform_result.text

                except Exception as e:
                    diagnostics.append(f"Chunk processing error: {e}")
                    yield ""
                    progress.processed_bytes += len(chunk_data)
                    progress.processed_chunks += 1

        return StreamingResult(
            chunks=chunk_generator(),
            encoding=encoding_result,
            progress=progress,
            diagnostics=diagnostics,
            metadata={
                "input_type": "binary_file",
                "buffer_size": self.config.buffer_size,
            }
        )

    def _process_text_file_stream(
        self,
        file_obj: TextIO,
        progress: StreamingProgress,
        progress_callback: Optional[ProgressCallback],
        diagnostics: List[str]
    ) -> StreamingResult:
        """Process text file with streaming."""
        encoding_result = EncodingResult(
            encoding="utf-8",
            confidence=1.0,
            method=DetectionMethod.FALLBACK,
            issues=[]
        )

        def chunk_generator() -> Generator[str, None, None]:
            """Generator that yields processed text chunks from text file."""
            buffer_size = self.config.buffer_size

            while True:
                # Check for cancellation
                if progress_callback and not progress_callback(progress.processed_bytes, progress.total_bytes):
                    progress.cancelled = True
                    break

                # Read next chunk
                chunk_text = file_obj.read(buffer_size)
                if not chunk_text:
                    break

                progress.current_chunk_size = len(chunk_text)

                try:
                    # Transform chunk
                    transform_result = self._character_transformer.transform(chunk_text)

                    # Update progress (using character count)
                    progress.processed_bytes += len(chunk_text)
                    progress.processed_chunks += 1

                    yield transform_result.text

                except Exception as e:
                    diagnostics.append(f"Chunk processing error: {e}")
                    yield ""
                    progress.processed_bytes += len(chunk_text)
                    progress.processed_chunks += 1

        return StreamingResult(
            chunks=chunk_generator(),
            encoding=encoding_result,
            progress=progress,
            diagnostics=diagnostics,
            metadata={
                "input_type": "text_file",
                "buffer_size": self.config.buffer_size,
            }
        )

    def _create_error_streaming_result(
        self, error_message: str, progress: StreamingProgress
    ) -> StreamingResult:
        """Create a streaming result object for error conditions."""

        def empty_generator() -> Generator[str, None, None]:
            """Empty generator for error cases."""
            # Empty generator - no items to yield
            return
            yield  # pragma: no cover

        return StreamingResult(
            chunks=empty_generator(),
            encoding=EncodingResult(
                encoding="utf-8",
                confidence=0.0,
                method=DetectionMethod.FALLBACK,
                issues=[error_message]
            ),
            progress=progress,
            diagnostics=[error_message],
            metadata={"error": True}
        )


# Configuration presets for common use cases
class StreamProcessingPresets:
    """Predefined configurations for common use cases."""

    @staticmethod
    def web_scraping(
        buffer_size: Optional[int] = None,
        max_memory: Optional[int] = None,
        enable_streaming: Optional[bool] = None,
        transform_overrides: Optional[Dict[str, Any]] = None
    ) -> StreamProcessingConfig:
        """Configuration optimized for web scraping scenarios.

        Args:
            buffer_size: Override default buffer size (16384)
            max_memory: Override default max memory limit
            enable_streaming: Override streaming setting (True)
            transform_overrides: Override transformation configuration settings

        Returns:
            StreamProcessingConfig with lenient settings and optional overrides
        """
        base_transform_config = TransformConfig.create_preset("lenient")

        # Apply transform overrides if provided
        if transform_overrides:
            base_transform_config = StreamProcessingPresets._apply_transform_overrides(
                base_transform_config, transform_overrides
            )

        return StreamProcessingConfig(
            transform_config=base_transform_config,
            buffer_size=buffer_size if buffer_size is not None else 16384,
            max_memory=max_memory if max_memory is not None else MAX_MEMORY_SIZE,
            enable_streaming=enable_streaming if enable_streaming is not None else True
        )

    @staticmethod
    def data_recovery(
        buffer_size: Optional[int] = None,
        max_memory: Optional[int] = None,
        enable_streaming: Optional[bool] = None,
        transform_overrides: Optional[Dict[str, Any]] = None
    ) -> StreamProcessingConfig:
        """Configuration optimized for data recovery scenarios.

        Args:
            buffer_size: Override default buffer size (4096)
            max_memory: Override default max memory limit
            enable_streaming: Override streaming setting (True)
            transform_overrides: Override transformation configuration settings

        Returns:
            StreamProcessingConfig with maximum recovery settings and optional overrides
        """
        base_transform_config = TransformConfig.create_preset("data_recovery")

        # Apply transform overrides if provided
        if transform_overrides:
            base_transform_config = StreamProcessingPresets._apply_transform_overrides(
                base_transform_config, transform_overrides
            )

        return StreamProcessingConfig(
            transform_config=base_transform_config,
            buffer_size=buffer_size if buffer_size is not None else 4096,
            max_memory=max_memory if max_memory is not None else MAX_MEMORY_SIZE,
            enable_streaming=enable_streaming if enable_streaming is not None else True
        )

    @staticmethod
    def strict_mode(
        buffer_size: Optional[int] = None,
        max_memory: Optional[int] = None,
        enable_streaming: Optional[bool] = None,
        transform_overrides: Optional[Dict[str, Any]] = None
    ) -> StreamProcessingConfig:
        """Configuration for strict XML compliance.

        Args:
            buffer_size: Override default buffer size (8192)
            max_memory: Override default max memory limit
            enable_streaming: Override streaming setting (True)
            transform_overrides: Override transformation configuration settings

        Returns:
            StreamProcessingConfig with strict settings and optional overrides
        """
        base_transform_config = TransformConfig.create_preset("strict")

        # Apply transform overrides if provided
        if transform_overrides:
            base_transform_config = StreamProcessingPresets._apply_transform_overrides(
                base_transform_config, transform_overrides
            )

        return StreamProcessingConfig(
            transform_config=base_transform_config,
            buffer_size=buffer_size if buffer_size is not None else 8192,
            max_memory=max_memory if max_memory is not None else MAX_MEMORY_SIZE,
            enable_streaming=enable_streaming if enable_streaming is not None else True
        )

    @staticmethod
    def _apply_transform_overrides(
        base_config: TransformConfig,
        overrides: Dict[str, Any]
    ) -> TransformConfig:
        """Apply overrides to a transform configuration.

        Args:
            base_config: Base configuration to modify
            overrides: Dictionary of attribute overrides

        Returns:
            New TransformConfig with applied overrides
        """
        # Create a new config by reconstructing with overrides
        # Get current values
        default_strategy = getattr(
            base_config, "default_strategy", TransformationStrategy.REPLACEMENT
        )
        context_strategies = getattr(base_config, "context_strategies", {})
        custom_mappings = getattr(base_config, "custom_mappings", {})
        replacement_char = getattr(base_config, "replacement_char", "\uFFFD")
        preserve_whitespace = getattr(base_config, "preserve_whitespace", True)
        strict_xml = getattr(base_config, "strict_xml", True)

        # Apply overrides
        if "default_strategy" in overrides:
            default_strategy = overrides["default_strategy"]
        if "context_strategies" in overrides:
            context_strategies = overrides["context_strategies"]
        if "custom_mappings" in overrides:
            custom_mappings = overrides["custom_mappings"]
        if "replacement_char" in overrides:
            replacement_char = overrides["replacement_char"]
        if "preserve_whitespace" in overrides:
            preserve_whitespace = overrides["preserve_whitespace"]
        if "strict_xml" in overrides:
            strict_xml = overrides["strict_xml"]

        return TransformConfig(
            default_strategy=default_strategy,
            context_strategies=context_strategies,
            custom_mappings=custom_mappings,
            replacement_char=replacement_char,
            preserve_whitespace=preserve_whitespace,
            strict_xml=strict_xml,
        )

    @staticmethod
    def validate_config(config: StreamProcessingConfig) -> List[str]:
        """Validate configuration combinations for potential issues.

        Args:
            config: Configuration to validate

        Returns:
            List of validation warnings or errors
        """
        issues = []

        # Check buffer size constraints
        if config.buffer_size <= 0:
            issues.append("Buffer size must be positive")
        elif config.buffer_size > config.max_memory:
            issues.append("Buffer size exceeds maximum memory limit")

        # Check memory constraints
        if config.max_memory <= 0:
            issues.append("Maximum memory must be positive")

        # Check transform config compatibility
        if (hasattr(config.transform_config, "strict_xml") and
            config.transform_config.strict_xml and not config.enable_streaming):
            issues.append(
                "Strict XML mode with streaming disabled may cause "
                "memory issues with large files"
            )

        # Warn about very small buffer sizes
        if config.buffer_size < MIN_BUFFER_SIZE:
            issues.append("Very small buffer size may impact performance")

        # Warn about very large buffer sizes
        if config.buffer_size > MAX_BUFFER_SIZE:
            issues.append("Very large buffer size may impact memory usage")

        return issues
