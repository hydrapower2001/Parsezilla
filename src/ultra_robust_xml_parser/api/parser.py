"""Core parser API with progressive disclosure for ultra-robust XML parsing.

This module provides the main parsing API with progressive disclosure from simple
module-level functions to advanced configuration classes, following never-fail
philosophy and Python conventions.
"""

import time
from pathlib import Path
from typing import Any, BinaryIO, Dict, Optional, TextIO, Union

from ultra_robust_xml_parser.character import (
    CharacterStreamProcessor,
    StreamProcessingConfig,
)
from ultra_robust_xml_parser.shared import (
    DiagnosticSeverity,
    TokenizationConfig,
    get_logger,
)
from ultra_robust_xml_parser.tokenization import EnhancedXMLTokenizer
from ultra_robust_xml_parser.tree import XMLTreeBuilder
from ultra_robust_xml_parser.tree.builder import ParseResult

# Type definitions for input data
InputType = Union[str, bytes, BinaryIO, TextIO, Path]

# Constants for API operations
PREVIEW_LENGTH = 100  # Max length for content preview in logs
MS_PER_SECOND = 1000  # Milliseconds per second conversion


def parse(input_data: InputType, correlation_id: Optional[str] = None) -> ParseResult:
    """Parse XML from various input sources with automatic type detection.

    This is the primary entry point for XML parsing with progressive disclosure.
    It automatically detects input type and routes to appropriate processing.

    Args:
        input_data: XML content as string, bytes, file-like object, or Path
        correlation_id: Optional correlation ID for request tracking

    Returns:
        ParseResult containing document tree, confidence, and comprehensive metadata

    Examples:
        Basic string parsing:
        >>> result = parse('<root><item>value</item></root>')
        >>> result.success
        True
        >>> result.tree.root.tag
        'root'

        File parsing:
        >>> result = parse(Path('document.xml'))
        >>> result.confidence
        0.95

        Bytes parsing:
        >>> result = parse(b'<?xml version="1.0"?><root/>')
        >>> result.tree.root.tag
        'root'
    """
    start_time = time.time()
    logger = get_logger(__name__, correlation_id, "parse")

    logger.info(
        "Starting universal parse operation",
        extra={
            "input_type": type(input_data).__name__,
            "has_correlation_id": correlation_id is not None
        }
    )

    try:
        # Detect input type and route accordingly
        if isinstance(input_data, (str, bytes)):
            return _parse_direct_content(input_data, correlation_id)
        if isinstance(input_data, Path):
            return _parse_path_object(input_data, correlation_id)
        if hasattr(input_data, "read"):
            return _parse_file_like_object(input_data, correlation_id)
        # Try to convert to string as fallback
        try:
            string_input = str(input_data)
            logger.warning(
                "Unknown input type converted to string",
                extra={"original_type": type(input_data).__name__}
            )
            return _parse_direct_content(string_input, correlation_id)
        except Exception as conversion_error:
            # Never-fail: return error result
            processing_time = (time.time() - start_time) * MS_PER_SECOND
            return _create_error_result(
                f"Unable to process input type {type(input_data)}: {conversion_error}",
                correlation_id,
                processing_time
            )

    except Exception as e:
        # Never-fail guarantee: return error result with diagnostics
        processing_time = (time.time() - start_time) * 1000
        logger.exception(
            "Parse operation failed",
            extra={
                "processing_time_ms": processing_time
            }
        )
        return _create_error_result(
            f"Parse operation failed: {e}",
            correlation_id,
            processing_time
        )


def parse_string(xml_string: str, correlation_id: Optional[str] = None) -> ParseResult:
    """Parse XML from a string with optimized processing.

    Dedicated function for string input processing with performance optimizations
    for text-based XML content.

    Args:
        xml_string: XML content as string
        correlation_id: Optional correlation ID for request tracking

    Returns:
        ParseResult containing document tree and comprehensive metadata

    Examples:
        Simple string parsing:
        >>> result = parse_string('<root><item id="1">Hello</item></root>')
        >>> result.success
        True
        >>> result.tree.root.find('item').get_attribute('id')
        '1'

        Malformed XML handling:
        >>> result = parse_string('<root><unclosed>content')
        >>> result.success
        True
        >>> result.repair_count > 0
        True
    """
    start_time = time.time()
    logger = get_logger(__name__, correlation_id, "parse_string")

    logger.info(
        "Starting string parse operation",
        extra={
            "content_length": len(xml_string),
            "preview": (
                xml_string[:PREVIEW_LENGTH] + "..."
                if len(xml_string) > PREVIEW_LENGTH else xml_string
            )
        }
    )

    try:
        return _parse_direct_content(xml_string, correlation_id)

    except Exception as e:
        # Never-fail guarantee
        processing_time = (time.time() - start_time) * 1000
        logger.exception(
            "String parse operation failed",
            extra={
                "processing_time_ms": processing_time
            }
        )
        return _create_error_result(
            f"String parse failed: {e}",
            correlation_id,
            processing_time
        )


def parse_file(
    file_path: Union[str, Path],
    encoding: Optional[str] = None,
    correlation_id: Optional[str] = None
) -> ParseResult:
    """Parse XML from a file with path handling and encoding detection.

    Dedicated function for file-based parsing with automatic encoding detection,
    proper resource management, and comprehensive error handling.

    Args:
        file_path: Path to XML file (string or Path object)
        encoding: Optional encoding override (auto-detected if not provided)
        correlation_id: Optional correlation ID for request tracking

    Returns:
        ParseResult containing document tree and comprehensive metadata

    Examples:
        File parsing with auto-detection:
        >>> result = parse_file('document.xml')
        >>> result.success
        True

        File parsing with encoding override:
        >>> result = parse_file('legacy.xml', encoding='latin1')
        >>> result.tree.encoding
        'latin1'

        Non-existent file handling:
        >>> result = parse_file('missing.xml')
        >>> result.success
        False
        >>> 'not found' in result.diagnostics[0].message.lower()
        True
    """
    start_time = time.time()
    logger = get_logger(__name__, correlation_id, "parse_file")

    # Convert to Path object for consistent handling
    path_obj = Path(file_path) if isinstance(file_path, str) else file_path

    logger.info(
        "Starting file parse operation",
        extra={
            "file_path": str(path_obj),
            "file_exists": path_obj.exists(),
            "encoding_override": encoding
        }
    )

    try:
        # Validate file before processing
        error_message = None
        if not path_obj.exists():
            error_message = f"File not found: {path_obj}"
        elif not path_obj.is_file():
            error_message = f"Path is not a file: {path_obj}"

        if error_message:
            processing_time = (time.time() - start_time) * MS_PER_SECOND
            return _create_error_result(error_message, correlation_id, processing_time)

        # Open and parse file with proper resource management
        try:
            if encoding:
                # Use specified encoding
                with path_obj.open(encoding=encoding, errors="replace") as file:
                    content = file.read()
                result = _parse_direct_content(content, correlation_id)
                # Add file metadata
                result.add_diagnostic(
                    DiagnosticSeverity.INFO,
                    f"File parsed with encoding: {encoding}",
                    "file_parser",
                    details={"file_path": str(path_obj), "encoding": encoding}
                )
            else:
                # Auto-detect encoding using binary mode first
                with path_obj.open("rb") as file:
                    raw_data = file.read()
                result = _parse_direct_content(raw_data, correlation_id)
                # Add file metadata
                result.add_diagnostic(
                    DiagnosticSeverity.INFO,
                    "File parsed with auto-detected encoding",
                    "file_parser",
                    details={"file_path": str(path_obj)}
                )

            # Add file-specific metadata to result
            if result.document:
                result.document.correlation_id = correlation_id

        except PermissionError:
            processing_time = (time.time() - start_time) * MS_PER_SECOND
            return _create_error_result(
                f"Permission denied accessing file: {path_obj}",
                correlation_id,
                processing_time
            )

        except UnicodeDecodeError as e:
            # Try with binary mode if text mode fails
            logger.warning(
                "Text mode failed, falling back to binary",
                extra={"error": str(e)}
            )
            try:
                with path_obj.open("rb") as file:
                    raw_data = file.read()
                return _parse_direct_content(raw_data, correlation_id)
            except Exception as fallback_error:
                processing_time = (time.time() - start_time) * MS_PER_SECOND
                return _create_error_result(
                    f"File encoding error and binary fallback failed: {fallback_error}",
                    correlation_id,
                    processing_time
                )
        else:
            return result

    except Exception as e:
        # Never-fail guarantee
        processing_time = (time.time() - start_time) * 1000
        logger.exception(
            "File parse operation failed",
            extra={
                "file_path": str(path_obj),
                "processing_time_ms": processing_time
            }
        )
        return _create_error_result(
            f"File parse failed: {e}",
            correlation_id,
            processing_time
        )


def _parse_direct_content(
    content: Union[str, bytes], correlation_id: Optional[str]
) -> ParseResult:
    """Internal function to parse direct content (string or bytes).

    Args:
        content: XML content as string or bytes
        correlation_id: Optional correlation ID for request tracking

    Returns:
        ParseResult with comprehensive parsing information
    """
    start_time = time.time()
    logger = get_logger(__name__, correlation_id, "parse_direct")

    try:
        # Step 1: Character processing (encoding detection and transformation)
        char_processor = CharacterStreamProcessor(
            config=StreamProcessingConfig()
        )
        char_result = char_processor.process(content)

        logger.debug(
            "Character processing completed",
            extra={
                "encoding": char_result.encoding.encoding,
                "confidence": char_result.confidence,
                "diagnostics_count": len(char_result.diagnostics)
            }
        )

        # Step 2: Tokenization with enhanced tokenizer
        tokenizer = EnhancedXMLTokenizer(
            config=TokenizationConfig.balanced(),
            correlation_id=correlation_id
        )
        tokenization_result = tokenizer.tokenize(char_result)

        logger.debug(
            "Tokenization completed",
            extra={
                "token_count": tokenization_result.token_count,
                "confidence": tokenization_result.confidence,
                "error_rate": tokenization_result.error_rate
            }
        )

        # Step 3: Tree building
        tree_builder = XMLTreeBuilder(correlation_id=correlation_id)
        parse_result = tree_builder.build(tokenization_result)

        processing_time = (time.time() - start_time) * 1000
        parse_result.performance.processing_time_ms = processing_time

        logger.info(
            "Direct content parsing completed",
            extra={
                "element_count": parse_result.element_count,
                "confidence": parse_result.confidence,
                "processing_time_ms": processing_time,
                "repair_count": parse_result.repair_count
            }
        )

        return parse_result

    except Exception as e:
        # Never-fail: return error result
        processing_time = (time.time() - start_time) * 1000
        logger.exception(
            "Direct content parsing failed",
            extra={
                "error": str(e),
                "processing_time_ms": processing_time
            }
        )
        return _create_error_result(
            f"Content parsing failed: {e}",
            correlation_id,
            processing_time
        )


def _parse_path_object(path_obj: Path, correlation_id: Optional[str]) -> ParseResult:
    """Internal function to parse Path objects.

    Args:
        path_obj: Path object to parse
        correlation_id: Optional correlation ID for request tracking

    Returns:
        ParseResult with parsing information
    """
    return parse_file(path_obj, correlation_id=correlation_id)


def _parse_file_like_object(
    file_obj: Union[BinaryIO, TextIO],
    correlation_id: Optional[str]
) -> ParseResult:
    """Internal function to parse file-like objects.

    Args:
        file_obj: File-like object to parse
        correlation_id: Optional correlation ID for request tracking

    Returns:
        ParseResult with parsing information
    """
    start_time = time.time()
    logger = get_logger(__name__, correlation_id, "parse_filelike")

    try:
        # Read content from file-like object
        content = file_obj.read()

        logger.info(
            "File-like object read",
            extra={
                "content_length": len(content) if content else 0,
                "content_type": type(content).__name__
            }
        )

        return _parse_direct_content(content, correlation_id)

    except Exception as e:
        # Never-fail guarantee
        processing_time = (time.time() - start_time) * 1000
        logger.exception(
            "File-like object parsing failed",
            extra={
                "error": str(e),
                "processing_time_ms": processing_time
            }
        )
        return _create_error_result(
            f"File-like object parse failed: {e}",
            correlation_id,
            processing_time
        )


def _create_error_result(
    error_message: str,
    correlation_id: Optional[str],
    processing_time: float
) -> ParseResult:
    """Create error result following never-fail philosophy.

    Args:
        error_message: Error description
        correlation_id: Optional correlation ID
        processing_time: Processing time in milliseconds

    Returns:
        ParseResult with error information
    """
    result = ParseResult(correlation_id=correlation_id)
    result.success = False
    result.confidence = 0.0
    result.performance.processing_time_ms = processing_time

    result.add_diagnostic(
        DiagnosticSeverity.CRITICAL,
        error_message,
        "api_parser"
    )

    return result


class UltraRobustXMLParser:
    """Advanced XML parser class with configurable initialization and reuse.

    Provides advanced configuration capabilities, parser instance reuse for improved
    performance, and comprehensive state management for multi-parse scenarios.

    Attributes:
        config: Current tokenization configuration
        char_config: Character processing configuration
        correlation_id: Correlation ID for request tracking

    Examples:
        Basic usage with default configuration:
        >>> parser = UltraRobustXMLParser()
        >>> result = parser.parse('<root><item>value</item></root>')
        >>> result.success
        True

        Advanced configuration:
        >>> config = TokenizationConfig.aggressive()
        >>> parser = UltraRobustXMLParser(config=config)
        >>> result = parser.parse(malformed_xml)
        >>> result.repair_count > 0
        True

        Parser reuse for performance:
        >>> parser = UltraRobustXMLParser()
        >>> results = [parser.parse(xml) for xml in xml_list]
        >>> all(r.success for r in results)
        True
    """

    def __init__(
        self,
        config: Optional[TokenizationConfig] = None,
        char_config: Optional[StreamProcessingConfig] = None,
        correlation_id: Optional[str] = None
    ) -> None:
        """Initialize advanced XML parser.

        Args:
            config: Tokenization configuration (defaults to balanced)
            char_config: Character processing configuration
            correlation_id: Optional correlation ID for request tracking
        """
        self.config = config or TokenizationConfig.balanced()
        self.char_config = char_config or StreamProcessingConfig()
        self.correlation_id = correlation_id or self.config.correlation_id

        self.logger = get_logger(__name__, self.correlation_id, "ultra_robust_parser")

        # Initialize components for reuse
        self._char_processor = CharacterStreamProcessor(config=self.char_config)
        self._tokenizer = EnhancedXMLTokenizer(
            config=self.config,
            correlation_id=self.correlation_id
        )
        self._tree_builder = XMLTreeBuilder(correlation_id=self.correlation_id)

        # Parser state for multi-parse scenarios
        self._parse_count = 0
        self._total_processing_time = 0.0
        self._successful_parses = 0

        self.logger.info(
            "UltraRobustXMLParser initialized",
            extra={
                "config_type": type(self.config).__name__,
                "correlation_id": self.correlation_id
            }
        )

    def parse(
        self,
        input_data: InputType,
        config_override: Optional[TokenizationConfig] = None,
        correlation_id_override: Optional[str] = None
    ) -> ParseResult:
        """Parse XML with configuration-aware processing.

        Args:
            input_data: XML content as various input types
            config_override: Optional configuration override for this parse
            correlation_id_override: Optional correlation ID override

        Returns:
            ParseResult with comprehensive parsing information

        Examples:
            Basic parsing:
            >>> parser = UltraRobustXMLParser()
            >>> result = parser.parse('<root/>')
            >>> result.success
            True

            Configuration override:
            >>> strict_config = TokenizationConfig.conservative()
            >>> result = parser.parse(xml, config_override=strict_config)
            >>> result.confidence >= 0.8
            True
        """
        start_time = time.time()
        effective_correlation_id = correlation_id_override or self.correlation_id

        self.logger.info(
            "Starting configured parse operation",
            extra={
                "input_type": type(input_data).__name__,
                "has_config_override": config_override is not None,
                "parse_count": self._parse_count + 1
            }
        )

        try:
            # Use override configuration if provided
            if config_override:
                # Create temporary components with override config
                tokenizer = EnhancedXMLTokenizer(
                    config=config_override,
                    correlation_id=effective_correlation_id
                )
                tree_builder = XMLTreeBuilder(correlation_id=effective_correlation_id)
            else:
                # Use existing components for performance
                tokenizer = self._tokenizer
                tree_builder = self._tree_builder

            # Process based on input type
            if isinstance(input_data, (str, bytes)):
                content = input_data
            elif isinstance(input_data, Path):
                return parse_file(
                    input_data,
                    correlation_id=effective_correlation_id
                )
            elif hasattr(input_data, "read"):
                content = input_data.read()
            else:
                content = str(input_data)

            # Step 1: Character processing
            char_result = self._char_processor.process(content)

            # Step 2: Tokenization
            tokenization_result = tokenizer.tokenize(char_result)

            # Step 3: Tree building
            parse_result = tree_builder.build(tokenization_result)

            # Update parser state
            processing_time = (time.time() - start_time) * MS_PER_SECOND
            parse_result.performance.processing_time_ms = processing_time

            self._parse_count += 1
            self._total_processing_time += processing_time
            if parse_result.success:
                self._successful_parses += 1

            self.logger.info(
                "Configured parse completed",
                extra={
                    "success": parse_result.success,
                    "confidence": parse_result.confidence,
                    "processing_time_ms": processing_time,
                    "total_parses": self._parse_count,
                    "success_rate": self._successful_parses / self._parse_count
                }
            )

            return parse_result

        except Exception as e:
            # Never-fail guarantee
            processing_time = (time.time() - start_time) * MS_PER_SECOND
            self._parse_count += 1
            self._total_processing_time += processing_time

            self.logger.exception(
                "Configured parse failed",
                extra={
                    "processing_time_ms": processing_time
                }
            )

            return _create_error_result(
                f"Configured parse failed: {e}",
                effective_correlation_id,
                processing_time
            )

    def reconfigure(
        self,
        config: Optional[TokenizationConfig] = None,
        char_config: Optional[StreamProcessingConfig] = None
    ) -> None:
        """Reconfigure parser with new settings.


        Args:
            config: New tokenization configuration
            char_config: New character processing configuration
        """
        if config:
            self.config = config
            self._tokenizer = EnhancedXMLTokenizer(
                config=self.config,
                correlation_id=self.correlation_id
            )
            self._tree_builder = XMLTreeBuilder(correlation_id=self.correlation_id)

        if char_config:
            self.char_config = char_config
            self._char_processor = CharacterStreamProcessor(config=self.char_config)

        self.logger.info(
            "Parser reconfigured",
            extra={
                "config_updated": config is not None,
                "char_config_updated": char_config is not None
            }
        )

    @property
    def statistics(self) -> Dict[str, Any]:
        """Get parser usage statistics.

        Returns:
            Dictionary with comprehensive parser statistics
        """
        return {
            "total_parses": self._parse_count,
            "successful_parses": self._successful_parses,
            "success_rate": (
                self._successful_parses / self._parse_count
                if self._parse_count > 0 else 0.0
            ),
            "total_processing_time_ms": self._total_processing_time,
            "average_processing_time_ms": (
                self._total_processing_time / self._parse_count
                if self._parse_count > 0 else 0.0
            ),
            "correlation_id": self.correlation_id,
        }

    def reset_statistics(self) -> None:
        """Reset parser usage statistics."""
        self._parse_count = 0
        self._total_processing_time = 0.0
        self._successful_parses = 0

        self.logger.info("Parser statistics reset")
