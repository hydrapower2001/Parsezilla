"""Comprehensive tokenization API with rich result objects and diagnostics.

This module provides a high-level API for XML tokenization with streaming support,
token filtering, configuration options, and comprehensive error reporting.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple

from ultra_robust_xml_parser.character import CharacterStreamResult
from ultra_robust_xml_parser.shared import (
    DiagnosticEntry,
    DiagnosticSeverity,
    PerformanceMetrics,
    TokenizationConfig,
    TokenizationMetadata,
    get_logger,
)
from ultra_robust_xml_parser.shared.config import FilterMode

from .tokenizer import Token, TokenType, XMLTokenizer


@dataclass
class TokenizationResult:
    """Comprehensive result object for tokenization operations."""

    # Core results
    tokens: List[Token] = field(default_factory=list)
    success: bool = True
    confidence: float = 1.0

    # Metadata and diagnostics
    metadata: TokenizationMetadata = field(default_factory=TokenizationMetadata)
    diagnostics: List[DiagnosticEntry] = field(default_factory=list)
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)

    # Source information
    character_result: Optional[CharacterStreamResult] = None
    correlation_id: Optional[str] = None

    def __post_init__(self) -> None:
        """Calculate derived statistics and validate result."""
        if self.tokens:
            self._update_metadata()
            self._calculate_confidence()

        # Validate confidence score
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")

    def _update_metadata(self) -> None:
        """Update metadata based on current tokens."""
        self.metadata.total_tokens = len(self.tokens)
        self.metadata.error_tokens = sum(
            1 for token in self.tokens if token.type == TokenType.ERROR
        )
        self.metadata.repaired_tokens = sum(
            1 for token in self.tokens if token.has_repairs
        )
        self.metadata.recovered_tokens = sum(
            1 for token in self.tokens
            if token.type in (TokenType.RECOVERED_CONTENT, TokenType.SYNTHETIC_CLOSE)
        )
        self.metadata.synthetic_tokens = sum(
            1 for token in self.tokens
            if token.type in (TokenType.SYNTHETIC_CLOSE, TokenType.BALANCED_STRUCTURE)
        )

        # Update token type distribution
        self.metadata.token_type_distribution.clear()
        for token in self.tokens:
            self.metadata.add_token_type(token.type.name)

    def _calculate_confidence(self) -> None:
        """Calculate overall confidence based on token confidence scores."""
        if not self.tokens:
            self.confidence = 1.0
            return

        # Calculate weighted confidence based on token quality
        total_weight = 0.0
        weighted_confidence = 0.0

        for token in self.tokens:
            # Weight tokens based on importance (content tokens weighted higher)
            weight = 1.0
            if token.type in (TokenType.TEXT, TokenType.TAG_NAME, TokenType.ATTR_VALUE):
                weight = 2.0
            elif token.type in (TokenType.ERROR, TokenType.MALFORMED_TAG):
                weight = 0.5

            weighted_confidence += token.confidence * weight
            total_weight += weight

        self.confidence = (
            weighted_confidence / total_weight if total_weight > 0 else 1.0
        )
        self.metadata.add_confidence_score("overall", self.confidence)

    @property
    def token_count(self) -> int:
        """Get total number of tokens."""
        return len(self.tokens)

    @property
    def error_rate(self) -> float:
        """Get error token rate."""
        return self.metadata.error_rate

    @property
    def repair_rate(self) -> float:
        """Get token repair rate."""
        return self.metadata.repair_rate

    @property
    def well_formed_tokens(self) -> List[Token]:
        """Get list of well-formed tokens (no repairs, high confidence)."""
        return [token for token in self.tokens if token.is_well_formed]

    @property
    def malformed_tokens(self) -> List[Token]:
        """Get list of malformed tokens (with repairs or low confidence)."""
        return [token for token in self.tokens if not token.is_well_formed]

    def get_tokens_by_type(self, token_type: TokenType) -> List[Token]:
        """Get all tokens of a specific type."""
        return [token for token in self.tokens if token.type == token_type]

    def get_tokens_in_range(self, start_offset: int, end_offset: int) -> List[Token]:
        """Get tokens within a specific character offset range."""
        return [
            token for token in self.tokens
            if start_offset <= token.position.offset < end_offset
        ]

    def get_high_confidence_tokens(self, threshold: float = 0.9) -> List[Token]:
        """Get tokens with confidence above threshold."""
        return [token for token in self.tokens if token.confidence >= threshold]

    def add_diagnostic(
        self,
        severity: DiagnosticSeverity,
        message: str,
        component: str,
        position: Optional[Dict[str, int]] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a diagnostic entry."""
        entry = DiagnosticEntry(
            severity=severity,
            message=message,
            component=component,
            position=position,
            details=details,
            correlation_id=self.correlation_id
        )
        self.diagnostics.append(entry)

    def get_diagnostics_by_severity(
        self,
        severity: DiagnosticSeverity
    ) -> List[DiagnosticEntry]:
        """Get diagnostics of a specific severity level."""
        return [diag for diag in self.diagnostics if diag.severity == severity]

    def has_errors(self) -> bool:
        """Check if result contains any error diagnostics."""
        return any(
            diag.severity in (DiagnosticSeverity.ERROR, DiagnosticSeverity.CRITICAL)
            for diag in self.diagnostics
        )

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics for the tokenization result."""
        return {
            "success": self.success,
            "confidence": self.confidence,
            "token_count": self.token_count,
            "error_rate": self.error_rate,
            "repair_rate": self.repair_rate,
            "recovery_rate": self.metadata.recovery_rate,
            "fast_path_used": self.metadata.fast_path_used,
            "processing_time_ms": self.performance.processing_time_ms,
            "characters_per_second": self.performance.characters_per_second,
            "diagnostics_count": len(self.diagnostics),
            "has_errors": self.has_errors(),
        }


class TokenFilter:
    """Token filtering and selection utilities."""

    def __init__(self, config: Optional[TokenizationConfig] = None) -> None:
        """Initialize token filter.

        Args:
            config: Configuration for filtering behavior
        """
        self.config = config or TokenizationConfig()
        self.filter_config = self.config.filtering

    def filter_by_type(
        self,
        tokens: List[Token],
        token_types: Set[TokenType]
    ) -> List[Token]:
        """Filter tokens by type."""
        if self.filter_config.mode == FilterMode.INCLUDE:
            return [
                token for token in tokens
                if token.type in token_types
            ]
        # EXCLUDE mode
        return [
            token for token in tokens
            if token.type not in token_types
        ]

    def filter_by_confidence(
        self,
        tokens: List[Token],
        threshold: Optional[float] = None
    ) -> List[Token]:
        """Filter tokens by confidence threshold."""
        actual_threshold = threshold or self.filter_config.confidence_threshold
        return [token for token in tokens if token.confidence >= actual_threshold]

    def filter_by_position_range(
        self,
        tokens: List[Token],
        start_offset: int,
        end_offset: int
    ) -> List[Token]:
        """Filter tokens by position range."""
        return [
            token for token in tokens
            if start_offset <= token.position.offset < end_offset
        ]

    def filter_by_content_pattern(
        self,
        tokens: List[Token],
        pattern: str
    ) -> List[Token]:
        """Filter tokens by content pattern matching."""
        import re  # noqa: PLC0415

        flags = 0 if self.filter_config.case_sensitive else re.IGNORECASE
        regex = re.compile(pattern, flags)

        return [
            token for token in tokens
            if regex.search(token.value)
        ]

    def apply_filters(
        self,
        tokens: List[Token],
        type_filter: Optional[Set[TokenType]] = None,
        confidence_threshold: Optional[float] = None,
        position_range: Optional[Tuple[int, int]] = None,
        content_pattern: Optional[str] = None
    ) -> List[Token]:
        """Apply multiple filters in sequence."""
        filtered_tokens = tokens

        if type_filter:
            filtered_tokens = self.filter_by_type(filtered_tokens, type_filter)

        if confidence_threshold is not None:
            filtered_tokens = self.filter_by_confidence(
                filtered_tokens, confidence_threshold
            )

        if position_range:
            start_offset, end_offset = position_range
            filtered_tokens = self.filter_by_position_range(
                filtered_tokens, start_offset, end_offset
            )

        if content_pattern:
            filtered_tokens = self.filter_by_content_pattern(
                filtered_tokens, content_pattern
            )

        # Apply max results limit
        if (
            self.filter_config.max_results is not None
            and len(filtered_tokens) > self.filter_config.max_results
        ):
            filtered_tokens = filtered_tokens[:self.filter_config.max_results]

        return filtered_tokens


class StreamingTokenizer:
    """Streaming tokenization interface for large inputs."""

    def __init__(
        self,
        config: Optional[TokenizationConfig] = None,
        correlation_id: Optional[str] = None
    ) -> None:
        """Initialize streaming tokenizer.

        Args:
            config: Configuration for tokenization behavior
            correlation_id: Optional correlation ID for request tracking
        """
        self.config = config or TokenizationConfig()
        self.correlation_id = correlation_id
        self.logger = get_logger(__name__, correlation_id, "streaming_tokenizer")

        # Initialize underlying tokenizer
        self.tokenizer = XMLTokenizer(
            correlation_id=correlation_id,
            enable_fast_path=self.config.performance.enable_fast_path,
            enable_recovery=True,
            enable_assembly=True
        )

        # Streaming state
        self._current_position = 0
        self._buffer = ""
        self._tokens_yielded = 0
        self._cancelled = False

    def tokenize_stream(
        self,
        char_stream: CharacterStreamResult,
        progress_callback: Optional[Callable[[float, int], None]] = None
    ) -> Generator[Token, None, TokenizationResult]:
        """Stream tokens from character stream.

        Args:
            char_stream: Character stream result to tokenize
            progress_callback: Optional callback for progress updates

        Yields:
            Individual Token objects as they are processed

        Returns:
            Final TokenizationResult when streaming is complete
        """
        start_time = time.time()
        all_tokens = []

        self.logger.info(
            "Starting streaming tokenization",
            extra={
                "content_length": len(char_stream.text) if char_stream.text else 0,
                "buffer_size": self.config.streaming.buffer_size
            }
        )

        try:
            # Process in chunks for streaming
            chunk_size = self.config.streaming.chunk_size
            content = char_stream.text or ""
            total_length = len(content)

            for i in range(0, total_length, chunk_size):
                if self._cancelled:
                    break

                chunk = content[i:i + chunk_size]
                chunk_result = CharacterStreamResult(
                    text=chunk,
                    encoding=char_stream.encoding,
                    transformations=char_stream.transformations,
                    confidence=char_stream.confidence,
                    diagnostics=[]
                )

                # Tokenize chunk
                chunk_tokenization = self.tokenizer.tokenize(chunk_result)

                # Yield tokens from chunk
                for token in chunk_tokenization.tokens:
                    all_tokens.append(token)
                    self._tokens_yielded += 1
                    yield token

                    # Check for cancellation and break from both loops
                    if self._cancelled:
                        break

                # Exit outer loop if cancelled during token processing
                if self._cancelled:
                    break

                # Progress callback
                if (
                    progress_callback
                    and self.config.streaming.enable_progress_tracking
                    and (
                        self._tokens_yielded
                        % self.config.streaming.progress_callback_interval
                    )
                    == 0
                ):
                    progress = (i + len(chunk)) / total_length
                    progress_callback(progress, self._tokens_yielded)

            # Create final result
            processing_time = (time.time() - start_time) * 1000  # Convert to ms

            result = TokenizationResult(
                tokens=all_tokens,
                success=not self._cancelled,
                character_result=char_stream,
                correlation_id=self.correlation_id
            )

            # Update performance metrics
            result.performance.processing_time_ms = processing_time
            result.performance.characters_processed = total_length
            result.performance.tokens_generated = len(all_tokens)

            self.logger.info(
                "Streaming tokenization completed",
                extra={
                    "tokens_yielded": self._tokens_yielded,
                    "processing_time_ms": processing_time,
                    "cancelled": self._cancelled
                }
            )

        except Exception as e:
            self.logger.error(
                "Streaming tokenization failed",
                extra={"error": str(e)}
            )

            # Return partial result on error
            processing_time = (time.time() - start_time) * 1000
            result = TokenizationResult(
                tokens=all_tokens,
                success=False,
                character_result=char_stream,
                correlation_id=self.correlation_id
            )
            result.performance.processing_time_ms = processing_time
            result.add_diagnostic(
                DiagnosticSeverity.ERROR,
                f"Streaming tokenization error: {e}",
                "streaming_tokenizer"
            )

            return result

        return result

    def cancel(self) -> None:
        """Cancel the streaming operation."""
        self._cancelled = True
        self.logger.info("Streaming tokenization cancelled")

    @property
    def is_cancelled(self) -> bool:
        """Check if streaming operation was cancelled."""
        return self._cancelled


class EnhancedXMLTokenizer:
    """Enhanced XML tokenizer with comprehensive API and configuration."""

    def __init__(
        self,
        config: Optional[TokenizationConfig] = None,
        correlation_id: Optional[str] = None
    ) -> None:
        """Initialize enhanced XML tokenizer.

        Args:
            config: Configuration for tokenization behavior
            correlation_id: Optional correlation ID for request tracking
        """
        self.config = config or TokenizationConfig()
        self.correlation_id = correlation_id or self.config.correlation_id
        self.logger = get_logger(__name__, self.correlation_id, "enhanced_tokenizer")

        # Initialize components
        self.tokenizer = XMLTokenizer(
            correlation_id=self.correlation_id,
            enable_fast_path=self.config.performance.enable_fast_path,
            enable_recovery=True,
            enable_assembly=True
        )
        self.token_filter = TokenFilter(self.config)
        self.streaming_tokenizer = StreamingTokenizer(self.config, self.correlation_id)

    def tokenize(
        self,
        char_stream: CharacterStreamResult,
        apply_filters: bool = False
    ) -> TokenizationResult:
        """Tokenize character stream with comprehensive result.

        Args:
            char_stream: Character stream result from character processing
            apply_filters: Whether to apply configured filters

        Returns:
            Comprehensive TokenizationResult with metadata and diagnostics
        """
        start_time = time.time()

        self.logger.info(
            "Starting enhanced tokenization",
            extra={
                "content_length": len(char_stream.text) if char_stream.text else 0,
                "apply_filters": apply_filters
            }
        )

        try:
            # Perform basic tokenization
            basic_result = self.tokenizer.tokenize(char_stream)

            # Create enhanced result
            result = TokenizationResult(
                tokens=basic_result.tokens,
                success=basic_result.success,
                character_result=char_stream,
                correlation_id=self.correlation_id
            )

            # Add diagnostics from basic tokenization
            for diagnostic in basic_result.diagnostics:
                result.add_diagnostic(
                    DiagnosticSeverity.INFO,
                    diagnostic,
                    "xml_tokenizer"
                )

            # Apply filters if requested
            if apply_filters and result.tokens:
                original_count = len(result.tokens)
                result.tokens = self.token_filter.apply_filters(result.tokens)

                if len(result.tokens) != original_count:
                    result.add_diagnostic(
                        DiagnosticSeverity.INFO,
                        f"Filtered {original_count - len(result.tokens)} tokens",
                        "token_filter"
                    )

            # Update performance metrics
            processing_time = (time.time() - start_time) * 1000
            result.performance.processing_time_ms = processing_time
            result.performance.characters_processed = (
                len(char_stream.text) if char_stream.text else 0
            )
            result.performance.tokens_generated = len(result.tokens)

            # Get additional statistics from tokenizer components
            if self.tokenizer.recovery_engine:
                recovery_stats = self.tokenizer.get_recovery_statistics()
                if recovery_stats:
                    result.performance.recovery_operations = recovery_stats.get(
                        "total_operations", 0
                    )

            if self.tokenizer.assembly_engine:
                assembly_stats = self.tokenizer.get_assembly_statistics()
                if assembly_stats:
                    result.performance.assembly_operations = assembly_stats.get(
                        "total_operations", 0
                    )

            # Update metadata
            result.metadata.fast_path_used = getattr(
                self.tokenizer, "fast_path_enabled", False
            )

            self.logger.info(
                "Enhanced tokenization completed",
                extra={
                    "token_count": result.token_count,
                    "confidence": result.confidence,
                    "processing_time_ms": processing_time
                }
            )

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self.logger.error(
                "Enhanced tokenization failed",
                extra={
                    "error": str(e),
                    "processing_time_ms": processing_time
                }
            )

            # Return error result (never-fail philosophy)
            result = TokenizationResult(
                tokens=[],
                success=False,
                character_result=char_stream,
                correlation_id=self.correlation_id
            )
            result.performance.processing_time_ms = processing_time
            result.add_diagnostic(
                DiagnosticSeverity.CRITICAL,
                f"Tokenization failed: {e}",
                "enhanced_tokenizer",
                details={"exception_type": type(e).__name__}
            )

            return result

        return result

    def tokenize_streaming(
        self,
        char_stream: CharacterStreamResult,
        progress_callback: Optional[Callable[[float, int], None]] = None
    ) -> Generator[Token, None, TokenizationResult]:
        """Stream tokenize character stream.

        Args:
            char_stream: Character stream result to tokenize
            progress_callback: Optional callback for progress updates

        Yields:
            Individual Token objects as they are processed

        Returns:
            Final TokenizationResult when streaming is complete
        """
        return self.streaming_tokenizer.tokenize_stream(char_stream, progress_callback)

    def configure(self, config: TokenizationConfig) -> None:
        """Update tokenizer configuration.

        Args:
            config: New configuration to apply
        """
        self.config = config
        self.correlation_id = config.correlation_id or self.correlation_id

        # Recreate components with new configuration
        self.tokenizer = XMLTokenizer(
            correlation_id=self.correlation_id,
            enable_fast_path=config.performance.enable_fast_path,
            enable_recovery=True,
            enable_assembly=True
        )
        self.token_filter = TokenFilter(config)
        self.streaming_tokenizer = StreamingTokenizer(config, self.correlation_id)

        self.logger.info("Tokenizer configuration updated")
