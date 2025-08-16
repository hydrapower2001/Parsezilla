"""Error recovery engine for malformed XML tokenization.

This module implements intelligent error recovery strategies that preserve content
and structure when possible, allowing extraction of useful information from severely
malformed XML documents.
"""

import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

from .tokenizer import Token, TokenizerState, TokenPosition, TokenType

# Constants for recovery optimization
MAX_RECOVERY_ATTEMPTS = 50
RECOVERY_CONFIDENCE_THRESHOLD = 0.3
PATTERN_CACHE_SIZE = 1000
ESCALATION_FAILURE_THRESHOLD = 3
SEVERE_MALFORMATION_THRESHOLD = 0.7
MAX_PATTERN_CORRELATIONS = 100
EFFECTIVENESS_UPDATE_FACTOR = 0.1
# Confidence distribution thresholds
CONFIDENCE_LOW = 0.2
CONFIDENCE_MID_LOW = 0.4
CONFIDENCE_MID = 0.6
CONFIDENCE_HIGH = 0.8
PATTERN_KEY_TRUNCATE_LENGTH = 50

logger = logging.getLogger(__name__)


class RecoveryStrategy(Enum):
    """Error recovery strategies for malformed XML content."""

    SKIP_UNTIL_VALID = auto()      # Skip corrupted content until valid structure found
    INSERT_MISSING_STRUCTURE = auto()  # Add synthetic elements to repair structure
    TREAT_AS_TEXT = auto()         # Convert malformed markup to text content
    BALANCED_REPAIR = auto()       # Attempt to balance unmatched tags
    CHARACTER_ESCAPE = auto()      # Escape invalid characters as text
    CONTEXT_INFERENCE = auto()     # Infer structure from context patterns


@dataclass
class RecoveryAction:
    """Result of a recovery operation with detailed metadata."""

    strategy: RecoveryStrategy
    success: bool
    tokens: List[Token]
    confidence: float
    description: str
    original_content: str
    repaired_content: str
    position: TokenPosition
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate recovery action values."""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")


@dataclass
class RecoveryContext:
    """Context information for error recovery decisions."""

    error_type: str
    error_position: TokenPosition
    surrounding_content: str
    tokenizer_state: TokenizerState
    recent_tokens: List[Token]
    malformation_severity: float
    pattern_history: List[str] = field(default_factory=list)

    @property
    def is_severe_malformation(self) -> bool:
        """Check if this is a severe malformation requiring aggressive recovery."""
        return self.malformation_severity > SEVERE_MALFORMATION_THRESHOLD


@dataclass
class RecoveryHistoryEntry:
    """Single entry in the recovery history log."""

    timestamp: float
    error_type: str
    error_position: TokenPosition
    strategy_used: RecoveryStrategy
    success: bool
    confidence: float
    tokens_generated: int
    processing_time_ms: float
    rationale: str
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def position_str(self) -> str:
        """Get position as string for logging."""
        return f"{self.error_position.line}:{self.error_position.column}"


@dataclass
class RecoveryStatistics:
    """Comprehensive recovery statistics and metrics."""

    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0
    strategy_counts: Dict[RecoveryStrategy, int] = field(default_factory=dict)
    strategy_success_rates: Dict[RecoveryStrategy, float] = field(default_factory=dict)
    error_type_counts: Dict[str, int] = field(default_factory=dict)
    total_processing_time_ms: float = 0.0
    average_confidence: float = 0.0
    pattern_correlations: Dict[str, RecoveryStrategy] = field(default_factory=dict)

    @property
    def overall_success_rate(self) -> float:
        """Calculate overall success rate."""
        if self.total_attempts == 0:
            return 1.0
        return self.successful_attempts / self.total_attempts

    @property
    def average_processing_time_ms(self) -> float:
        """Calculate average processing time per attempt."""
        if self.total_attempts == 0:
            return 0.0
        return self.total_processing_time_ms / self.total_attempts


class RecoveryHistory:
    """Comprehensive history tracking for error recovery operations.

    Tracks all recovery actions with detailed logging, rationale, and metadata
    for debugging, analysis, and confidence scoring.
    """

    def __init__(self, max_entries: int = 10000) -> None:
        """Initialize recovery history tracking.

        Args:
            max_entries: Maximum number of history entries to keep
        """
        self.max_entries = max_entries
        self.entries: List[RecoveryHistoryEntry] = []
        self.statistics = RecoveryStatistics()
        self._pattern_correlation_cache: Dict[str, List[RecoveryStrategy]] = {}

    def record_recovery(
        self,
        context: RecoveryContext,
        action: RecoveryAction,
        processing_time_ms: float,
        rationale: str,
        correlation_id: Optional[str] = None
    ) -> None:
        """Record a recovery operation in the history.

        Args:
            context: Error context that triggered recovery
            action: Recovery action taken
            processing_time_ms: Time taken to process recovery
            rationale: Detailed rationale for the recovery decision
            correlation_id: Optional correlation ID for tracking
        """
        entry = RecoveryHistoryEntry(
            timestamp=time.time(),
            error_type=context.error_type,
            error_position=context.error_position,
            strategy_used=action.strategy,
            success=action.success,
            confidence=action.confidence,
            tokens_generated=len(action.tokens),
            processing_time_ms=processing_time_ms,
            rationale=rationale,
            correlation_id=correlation_id,
            metadata={
                "surrounding_content": context.surrounding_content[:50],
                "tokenizer_state": context.tokenizer_state.name,
                "malformation_severity": context.malformation_severity,
                **action.metadata
            }
        )

        # Add entry and maintain size limit
        self.entries.append(entry)
        if len(self.entries) > self.max_entries:
            # Remove oldest entries
            self.entries = self.entries[-self.max_entries:]

        # Update statistics
        self._update_statistics(entry)

        # Update pattern correlations
        self._update_pattern_correlations(context, action.strategy)

        logger.debug(
            "Recovery operation recorded",
            extra={
                "component": "recovery_history",
                "correlation_id": correlation_id,
                "strategy": action.strategy.name,
                "success": action.success,
                "confidence": action.confidence,
                "position": entry.position_str
            }
        )

    def _update_statistics(self, entry: RecoveryHistoryEntry) -> None:
        """Update comprehensive statistics based on new entry.

        Args:
            entry: Recovery history entry to process
        """
        # Update totals
        self.statistics.total_attempts += 1
        if entry.success:
            self.statistics.successful_attempts += 1
        else:
            self.statistics.failed_attempts += 1

        # Update strategy counts and success rates
        strategy = entry.strategy_used
        if strategy not in self.statistics.strategy_counts:
            self.statistics.strategy_counts[strategy] = 0
            self.statistics.strategy_success_rates[strategy] = 0.0

        self.statistics.strategy_counts[strategy] += 1

        # Calculate strategy success rate
        strategy_entries = [e for e in self.entries if e.strategy_used == strategy]
        strategy_successes = sum(1 for e in strategy_entries if e.success)
        self.statistics.strategy_success_rates[strategy] = (
            strategy_successes / len(strategy_entries) if strategy_entries else 0.0
        )

        # Update error type counts
        if entry.error_type not in self.statistics.error_type_counts:
            self.statistics.error_type_counts[entry.error_type] = 0
        self.statistics.error_type_counts[entry.error_type] += 1

        # Update timing statistics
        self.statistics.total_processing_time_ms += entry.processing_time_ms

        # Update average confidence
        total_confidence = sum(e.confidence for e in self.entries)
        self.statistics.average_confidence = total_confidence / len(self.entries)

    def _update_pattern_correlations(
        self,
        context: RecoveryContext,
        successful_strategy: RecoveryStrategy
    ) -> None:
        """Update pattern correlations for strategy selection optimization.

        Args:
            context: Error context
            successful_strategy: Strategy that was successful
        """
        pattern_key = f"{context.error_type}:{context.tokenizer_state.name}"

        if pattern_key not in self._pattern_correlation_cache:
            self._pattern_correlation_cache[pattern_key] = []

        self._pattern_correlation_cache[pattern_key].append(successful_strategy)

        # Keep only recent correlations (last 100 per pattern)
        if len(self._pattern_correlation_cache[pattern_key]) > MAX_PATTERN_CORRELATIONS:
            self._pattern_correlation_cache[pattern_key] = (
                self._pattern_correlation_cache[pattern_key][-MAX_PATTERN_CORRELATIONS:]
            )

        # Update statistics pattern correlations with most common strategy
        strategy_counts = Counter(self._pattern_correlation_cache[pattern_key])
        most_common_strategy = strategy_counts.most_common(1)[0][0]
        self.statistics.pattern_correlations[pattern_key] = most_common_strategy

    def get_pattern_recommendation(
        self,
        context: RecoveryContext
    ) -> Optional[RecoveryStrategy]:
        """Get strategy recommendation based on historical patterns.

        Args:
            context: Current error context

        Returns:
            Recommended strategy or None if no pattern found
        """
        pattern_key = f"{context.error_type}:{context.tokenizer_state.name}"
        return self.statistics.pattern_correlations.get(pattern_key)

    def get_recent_entries(self, count: int = 10) -> List[RecoveryHistoryEntry]:
        """Get the most recent recovery entries.

        Args:
            count: Number of recent entries to return

        Returns:
            List of recent recovery history entries
        """
        return self.entries[-count:] if self.entries else []

    def get_entries_by_error_type(self, error_type: str) -> List[RecoveryHistoryEntry]:
        """Get all entries for a specific error type.

        Args:
            error_type: Error type to filter by

        Returns:
            List of entries matching the error type
        """
        return [entry for entry in self.entries if entry.error_type == error_type]

    def get_entries_by_strategy(
        self, strategy: RecoveryStrategy
    ) -> List[RecoveryHistoryEntry]:
        """Get all entries for a specific recovery strategy.

        Args:
            strategy: Recovery strategy to filter by

        Returns:
            List of entries using the specified strategy
        """
        return [entry for entry in self.entries if entry.strategy_used == strategy]

    def get_confidence_distribution(self) -> Dict[str, int]:
        """Get distribution of confidence levels across all entries.

        Returns:
            Dictionary mapping confidence ranges to counts
        """
        distribution = {
            "0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0,
            "0.6-0.8": 0, "0.8-1.0": 0
        }

        for entry in self.entries:
            confidence = entry.confidence
            if confidence < CONFIDENCE_LOW:
                distribution["0.0-0.2"] += 1
            elif confidence < CONFIDENCE_MID_LOW:
                distribution["0.2-0.4"] += 1
            elif confidence < CONFIDENCE_MID:
                distribution["0.4-0.6"] += 1
            elif confidence < CONFIDENCE_HIGH:
                distribution["0.6-0.8"] += 1
            else:
                distribution["0.8-1.0"] += 1

        return distribution

    def generate_detailed_report(self) -> Dict[str, Any]:
        """Generate comprehensive recovery history report.

        Returns:
            Detailed report dictionary with all statistics and analysis
        """
        recent_entries = self.get_recent_entries(20)
        confidence_dist = self.get_confidence_distribution()

        return {
            "summary": {
                "total_entries": len(self.entries),
                "overall_success_rate": self.statistics.overall_success_rate,
                "average_confidence": self.statistics.average_confidence,
                "average_processing_time_ms": self.statistics.average_processing_time_ms
            },
            "strategy_analysis": {
                "strategy_counts": {
                    strategy.name: count
                    for strategy, count in self.statistics.strategy_counts.items()
                },
                "strategy_success_rates": {
                    strategy.name: rate
                    for strategy, rate in self.statistics.strategy_success_rates.items()
                }
            },
            "error_analysis": dict(self.statistics.error_type_counts),
            "confidence_distribution": confidence_dist,
            "pattern_correlations": {
                pattern: strategy.name
                for pattern, strategy in self.statistics.pattern_correlations.items()
            },
            "recent_entries": [
                {
                    "timestamp": entry.timestamp,
                    "error_type": entry.error_type,
                    "position": entry.position_str,
                    "strategy": entry.strategy_used.name,
                    "success": entry.success,
                    "confidence": entry.confidence,
                    "rationale": entry.rationale
                }
                for entry in recent_entries
            ]
        }

    def clear_history(self) -> None:
        """Clear all recovery history and reset statistics."""
        self.entries.clear()
        self.statistics = RecoveryStatistics()
        self._pattern_correlation_cache.clear()


class ErrorRecoveryEngine:
    """Main error recovery engine for handling malformed XML content.

    Implements intelligent error recovery that preserves content and structure
    when possible, using multiple fallback strategies to ensure tokens are
    always generated.
    """

    def __init__(
        self,
        correlation_id: Optional[str] = None,
        enable_history: bool = True
    ) -> None:
        """Initialize the error recovery engine.

        Args:
            correlation_id: Optional correlation ID for tracking requests
            enable_history: Whether to enable comprehensive history tracking
        """
        self.correlation_id = correlation_id
        self.recovery_attempts = 0
        self.pattern_cache: Dict[str, RecoveryStrategy] = {}
        self.strategy_effectiveness: Dict[RecoveryStrategy, float] = dict.fromkeys(
            RecoveryStrategy, 0.5
        )
        self.escalation_failures = 0

        # Recovery history tracking
        self.history = RecoveryHistory() if enable_history else None

    def recover_from_error(
        self,
        context: RecoveryContext,
        original_char: str
    ) -> RecoveryAction:
        """Recover from a tokenization error using intelligent strategy selection.

        Args:
            context: Error context information for recovery decisions
            original_char: The character that caused the error

        Returns:
            RecoveryAction with tokens and metadata
        """
        start_time = time.time()

        if self.recovery_attempts >= MAX_RECOVERY_ATTEMPTS:
            logger.warning(
                "Maximum recovery attempts reached, using final fallback",
                extra={
                    "component": "error_recovery_engine",
                    "correlation_id": self.correlation_id,
                    "attempts": self.recovery_attempts
                }
            )
            action = self._final_fallback_recovery(context, original_char)

            # Record in history if enabled
            if self.history:
                processing_time_ms = (time.time() - start_time) * 1000
                self.history.record_recovery(
                    context, action, processing_time_ms,
                    "Maximum recovery attempts reached - final fallback used",
                    self.correlation_id
                )

            return action

        self.recovery_attempts += 1

        # Check history for pattern recommendations if available
        strategy = None
        rationale = "Standard strategy selection"

        if self.history:
            recommended_strategy = self.history.get_pattern_recommendation(context)
            if recommended_strategy:
                strategy = recommended_strategy
                rationale = (
                    f"Pattern-based recommendation from history: {strategy.name}"
                )
                logger.debug(
                    "Using pattern-based strategy recommendation",
                    extra={
                        "component": "error_recovery_engine",
                        "correlation_id": self.correlation_id,
                        "recommended_strategy": strategy.name
                    }
                )

        # Fallback to standard strategy selection if no recommendation
        if not strategy:
            strategy = self._select_recovery_strategy(context)
            rationale = f"Selected {strategy.name} based on error context analysis"

        logger.debug(
            "Attempting error recovery",
            extra={
                "component": "error_recovery_engine",
                "correlation_id": self.correlation_id,
                "strategy": strategy.name,
                "error_type": context.error_type,
                "position": (
                    f"{context.error_position.line}:{context.error_position.column}"
                )
            }
        )

        # Execute recovery strategy
        recovery_action = self._execute_recovery_strategy(
            strategy, context, original_char
        )

        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000

        # Update effectiveness tracking
        self._update_strategy_effectiveness(strategy, recovery_action.success)

        # Cache successful patterns
        if recovery_action.success:
            self._cache_successful_pattern(context, strategy)
        else:
            self.escalation_failures += 1

        # Record in history if enabled
        if self.history:
            detailed_rationale = (
                f"{rationale}. Strategy: {strategy.name}. "
                f"Success: {recovery_action.success}. "
                f"Tokens generated: {len(recovery_action.tokens)}. "
                f"Confidence: {recovery_action.confidence:.2f}"
            )
            self.history.record_recovery(
                context, recovery_action, processing_time_ms,
                detailed_rationale, self.correlation_id
            )

        return recovery_action

    def _select_recovery_strategy(self, context: RecoveryContext) -> RecoveryStrategy:
        """Select the optimal recovery strategy based on context and patterns.

        Args:
            context: Error context for strategy selection

        Returns:
            Selected RecoveryStrategy
        """
        # Check cached patterns first
        pattern_key = self._generate_pattern_key(context)
        if pattern_key in self.pattern_cache:
            cached_strategy = self.pattern_cache[pattern_key]
            logger.debug(
                "Using cached recovery strategy",
                extra={
                    "component": "error_recovery_engine",
                    "correlation_id": self.correlation_id,
                    "strategy": cached_strategy.name,
                    "pattern": (
                    pattern_key[:PATTERN_KEY_TRUNCATE_LENGTH] + "..."
                    if len(pattern_key) > PATTERN_KEY_TRUNCATE_LENGTH
                    else pattern_key
                )
                }
            )
            return cached_strategy

        # Select based on error context and severity
        if context.error_type == "invalid_character_sequence":
            if context.is_severe_malformation:
                return RecoveryStrategy.TREAT_AS_TEXT
            return RecoveryStrategy.CHARACTER_ESCAPE

        if context.error_type == "unmatched_tag":
            return RecoveryStrategy.BALANCED_REPAIR

        if context.error_type == "invalid_tag_structure":
            if context.tokenizer_state in (
                TokenizerState.TAG_OPENING,
                TokenizerState.TAG_NAME,
            ):
                return RecoveryStrategy.INSERT_MISSING_STRUCTURE
            return RecoveryStrategy.SKIP_UNTIL_VALID

        if context.error_type == "malformed_attribute":
            return RecoveryStrategy.SKIP_UNTIL_VALID

        # Default strategy based on effectiveness
        return max(
            self.strategy_effectiveness.items(),
            key=lambda x: x[1]
        )[0]

    def _execute_recovery_strategy(
        self,
        strategy: RecoveryStrategy,
        context: RecoveryContext,
        original_char: str
    ) -> RecoveryAction:
        """Execute the selected recovery strategy.

        Args:
            strategy: Recovery strategy to execute
            context: Error context
            original_char: Original character that caused error

        Returns:
            RecoveryAction result
        """
        try:
            # Execute appropriate recovery strategy
            strategy_methods = {
                RecoveryStrategy.SKIP_UNTIL_VALID: self._skip_until_valid_recovery,
                RecoveryStrategy.INSERT_MISSING_STRUCTURE: (
                    self._insert_missing_structure_recovery
                ),
                RecoveryStrategy.TREAT_AS_TEXT: self._treat_as_text_recovery,
                RecoveryStrategy.BALANCED_REPAIR: self._balanced_repair_recovery,
                RecoveryStrategy.CHARACTER_ESCAPE: self._character_escape_recovery,
                RecoveryStrategy.CONTEXT_INFERENCE: self._context_inference_recovery,
            }
            return strategy_methods[strategy](context, original_char)

        except Exception as e:
            logger.error(
                "Recovery strategy execution failed",
                extra={
                    "component": "error_recovery_engine",
                    "correlation_id": self.correlation_id,
                    "strategy": strategy.name,
                    "error": str(e)
                },
                exc_info=True
            )

            # Fallback to treat as text on any strategy failure
            return self._treat_as_text_recovery(context, original_char)

    def _skip_until_valid_recovery(
        self,
        context: RecoveryContext,
        original_char: str
    ) -> RecoveryAction:
        """Skip corrupted content until valid XML structure is found.

        Args:
            context: Error context
            original_char: Character that caused error

        Returns:
            RecoveryAction with skipped content as error token or no tokens if skipped entirely
        """
        # Create error token for skipped content
        error_token = Token(
            type=TokenType.INVALID_CHARS,
            value=original_char,
            position=context.error_position,
            confidence=0.1,
            raw_content=original_char
        )

        return RecoveryAction(
            strategy=RecoveryStrategy.SKIP_UNTIL_VALID,
            success=True,
            tokens=[error_token],
            confidence=0.8,
            description=(
                f"Skipped invalid character '{original_char}' until valid structure"
            ),
            original_content=original_char,
            repaired_content="",
            position=context.error_position,
            metadata={"skipped_chars": 1}
        )

    def _insert_missing_structure_recovery(
        self,
        context: RecoveryContext,
        original_char: str
    ) -> RecoveryAction:
        """Add synthetic elements to repair structural issues.

        Args:
            context: Error context
            original_char: Character that caused error

        Returns:
            RecoveryAction with synthetic structure tokens
        """
        synthetic_tokens = []

        # Analyze what structure might be missing
        if (
            context.tokenizer_state == TokenizerState.TAG_OPENING
            and original_char == ">"
        ):
            # Missing tag name - insert synthetic name
            synthetic_name_token = Token(
                type=TokenType.TAG_NAME,
                value="unknown",
                position=context.error_position,
                confidence=0.3,
                raw_content="unknown"
            )
            synthetic_tokens.append(synthetic_name_token)

            # Add the closing bracket
            closing_token = Token(
                type=TokenType.SYNTHETIC_CLOSE,
                value=">",
                position=context.error_position,
                confidence=0.8,
                raw_content=">"
            )
            synthetic_tokens.append(closing_token)

        elif context.tokenizer_state == TokenizerState.ATTR_VALUE_START:
            # Missing attribute value - insert empty value
            empty_value_token = Token(
                type=TokenType.ATTR_VALUE,
                value="",
                position=context.error_position,
                confidence=0.3,
                raw_content='""'
            )
            synthetic_tokens.append(empty_value_token)

        else:
            # Generic structure repair - treat problematic char as recovered content
            recovered_token = Token(
                type=TokenType.RECOVERED_CONTENT,
                value=original_char,
                position=context.error_position,
                confidence=0.5,
                raw_content=original_char
            )
            synthetic_tokens.append(recovered_token)

        return RecoveryAction(
            strategy=RecoveryStrategy.INSERT_MISSING_STRUCTURE,
            success=True,
            tokens=synthetic_tokens,
            confidence=0.6,
            description="Inserted synthetic structure for missing elements",
            original_content=original_char,
            repaired_content=" ".join(t.value for t in synthetic_tokens),
            position=context.error_position,
            metadata={"synthetic_tokens": len(synthetic_tokens)}
        )

    def _treat_as_text_recovery(
        self,
        context: RecoveryContext,
        original_char: str
    ) -> RecoveryAction:
        """Convert malformed markup to text content for preservation.

        Args:
            context: Error context
            original_char: Character that caused error

        Returns:
            RecoveryAction with character converted to text token
        """
        text_token = Token(
            type=TokenType.TEXT,
            value=original_char,
            position=context.error_position,
            confidence=0.9,  # High confidence for text preservation
            raw_content=original_char
        )

        return RecoveryAction(
            strategy=RecoveryStrategy.TREAT_AS_TEXT,
            success=True,
            tokens=[text_token],
            confidence=0.9,
            description="Converted malformed markup to text content",
            original_content=original_char,
            repaired_content=original_char,
            position=context.error_position,
            metadata={"preservation_strategy": "text_conversion"}
        )

    def _balanced_repair_recovery(
        self,
        context: RecoveryContext,
        original_char: str
    ) -> RecoveryAction:
        """Attempt to balance unmatched tags.

        Args:
            context: Error context
            original_char: Character that caused error

        Returns:
            RecoveryAction with balanced structure
        """
        synthetic_tokens = []

        # Analyze recent tokens to determine what balancing is needed
        if context.recent_tokens:
            # Look for unmatched opening tags
            tag_stack = []
            for token in context.recent_tokens[-10:]:  # Last 10 tokens
                if token.type == TokenType.TAG_NAME and token.value:
                    # Simple heuristic: if we see tag names, track them
                    tag_stack.append(token.value)
                elif token.type == TokenType.TAG_END and token.value == ">":
                    # Tag closed - but we don't know if it's opening or closing
                    pass

            # If we have unmatched tags and hit an error, try to close them
            if tag_stack and context.error_type == "unmatched_tag":
                # Generate synthetic closing tag
                last_tag = tag_stack[-1]
                synthetic_close = Token(
                    type=TokenType.BALANCED_STRUCTURE,
                    value=f"</{last_tag}>",
                    position=context.error_position,
                    confidence=0.4,
                    raw_content=f"</{last_tag}>"
                )
                synthetic_tokens.append(synthetic_close)

                return RecoveryAction(
                    strategy=RecoveryStrategy.BALANCED_REPAIR,
                    success=True,
                    tokens=synthetic_tokens,
                    confidence=0.4,
                    description=(
                        f"Inserted synthetic closing tag </{last_tag}> for balance"
                    ),
                    original_content=original_char,
                    repaired_content=f"</{last_tag}>",
                    position=context.error_position,
                    metadata={"balanced_tag": last_tag}
                )

        # Fallback to text treatment if no balancing strategy found
        return self._treat_as_text_recovery(context, original_char)

    def _character_escape_recovery(
        self,
        context: RecoveryContext,
        original_char: str
    ) -> RecoveryAction:
        """Escape invalid characters as text.

        Args:
            context: Error context
            original_char: Character that caused error

        Returns:
            RecoveryAction with escaped character
        """
        # Convert problematic characters to their escaped form if needed
        escaped_char = original_char
        if original_char == "<":
            escaped_char = "&lt;"
        elif original_char == ">":
            escaped_char = "&gt;"
        elif original_char == "&":
            escaped_char = "&amp;"

        text_token = Token(
            type=TokenType.TEXT,
            value=escaped_char,
            position=context.error_position,
            confidence=0.8,
            raw_content=original_char
        )

        return RecoveryAction(
            strategy=RecoveryStrategy.CHARACTER_ESCAPE,
            success=True,
            tokens=[text_token],
            confidence=0.8,
            description=(
                f"Escaped invalid character '{original_char}' to '{escaped_char}'"
            ),
            original_content=original_char,
            repaired_content=escaped_char,
            position=context.error_position,
            metadata={"escaped": escaped_char != original_char}
        )

    def _context_inference_recovery(
        self,
        context: RecoveryContext,
        original_char: str
    ) -> RecoveryAction:
        """Infer structure from context patterns.

        Args:
            context: Error context
            original_char: Character that caused error

        Returns:
            RecoveryAction with inferred structure
        """
        # Analyze surrounding content and recent tokens for patterns
        inference_confidence = 0.3
        inferred_tokens = []

        # Pattern 1: If we're in tag context and see a quote, infer attribute value
        if (context.tokenizer_state == TokenizerState.ATTR_VALUE_START and
            original_char in ('"', "'") and
            "=" in context.surrounding_content):

            # Infer this is an attribute value delimiter
            attr_value_token = Token(
                type=TokenType.ATTR_VALUE,
                value="",  # Empty value for now
                position=context.error_position,
                confidence=0.6,
                raw_content='""'
            )
            inferred_tokens.append(attr_value_token)
            inference_confidence = 0.6

        # Pattern 2: If we see < followed by non-letter, infer text content
        elif (original_char == "<" and
              len(context.surrounding_content) > 0 and
              not context.surrounding_content[-1].isalpha()):

            # This < is likely not a tag start, treat as text
            text_token = Token(
                type=TokenType.TEXT,
                value="&lt;",
                position=context.error_position,
                confidence=0.7,
                raw_content="<"
            )
            inferred_tokens.append(text_token)
            inference_confidence = 0.7

        # Pattern 3: If in text and see >, check if it should be escaped
        elif (context.tokenizer_state == TokenizerState.TEXT_CONTENT and
              original_char == ">" and
              not any(
                  token.type == TokenType.TAG_START
                  for token in context.recent_tokens[-3:]
              )):

            # Likely a standalone > that should be escaped
            text_token = Token(
                type=TokenType.TEXT,
                value="&gt;",
                position=context.error_position,
                confidence=0.7,
                raw_content=">"
            )
            inferred_tokens.append(text_token)
            inference_confidence = 0.7

        # Pattern 4: Analyze for common XML patterns like CDATA markers
        elif ("]]>" in context.surrounding_content and
              context.tokenizer_state == TokenizerState.TEXT_CONTENT):

            # Might be end of CDATA, but malformed
            recovered_token = Token(
                type=TokenType.RECOVERED_CONTENT,
                value=original_char,
                position=context.error_position,
                confidence=0.5,
                raw_content=original_char
            )
            inferred_tokens.append(recovered_token)
            inference_confidence = 0.5

        # Fallback to text treatment if no clear inference
        if not inferred_tokens:
            return self._treat_as_text_recovery(context, original_char)

        return RecoveryAction(
            strategy=RecoveryStrategy.CONTEXT_INFERENCE,
            success=True,
            tokens=inferred_tokens,
            confidence=inference_confidence,
            description="Inferred structure from context patterns",
            original_content=original_char,
            repaired_content=" ".join(t.value for t in inferred_tokens),
            position=context.error_position,
            metadata={"inference_patterns": len(inferred_tokens)}
        )

    def _final_fallback_recovery(
        self,
        context: RecoveryContext,
        original_char: str
    ) -> RecoveryAction:
        """Final fallback recovery when all else fails.

        Args:
            context: Error context
            original_char: Character that caused error

        Returns:
            RecoveryAction guaranteed to succeed
        """
        logger.warning(
            "Using final fallback recovery - treat as text",
            extra={
                "component": "error_recovery_engine",
                "correlation_id": self.correlation_id,
                "position": (
                    f"{context.error_position.line}:{context.error_position.column}"
                )
            }
        )

        text_token = Token(
            type=TokenType.TEXT,
            value=original_char,
            position=context.error_position,
            confidence=0.1,  # Low confidence for final fallback
            raw_content=original_char
        )

        return RecoveryAction(
            strategy=RecoveryStrategy.TREAT_AS_TEXT,
            success=True,
            tokens=[text_token],
            confidence=0.1,
            description="Final fallback - converted to text to prevent failure",
            original_content=original_char,
            repaired_content=original_char,
            position=context.error_position,
            metadata={"fallback": True, "attempts": self.recovery_attempts}
        )

    def _generate_pattern_key(self, context: RecoveryContext) -> str:
        """Generate a cache key for error patterns.

        Args:
            context: Error context

        Returns:
            String key for pattern caching
        """
        # Create a pattern key from error context
        return (
            f"{context.error_type}:"
            f"{context.tokenizer_state.name}:"
            f"{context.surrounding_content[:20]}"
        )

    def _cache_successful_pattern(
        self,
        context: RecoveryContext,
        strategy: RecoveryStrategy
    ) -> None:
        """Cache successful recovery patterns for reuse.

        Args:
            context: Error context
            strategy: Successful recovery strategy
        """
        pattern_key = self._generate_pattern_key(context)

        # Implement LRU cache by removing oldest entries if at capacity
        if len(self.pattern_cache) >= PATTERN_CACHE_SIZE:
            # Remove first (oldest) entry
            oldest_key = next(iter(self.pattern_cache))
            del self.pattern_cache[oldest_key]

        self.pattern_cache[pattern_key] = strategy

    def _update_strategy_effectiveness(
        self,
        strategy: RecoveryStrategy,
        success: bool
    ) -> None:
        """Update strategy effectiveness tracking.

        Args:
            strategy: Recovery strategy used
            success: Whether the strategy succeeded
        """
        current_effectiveness = self.strategy_effectiveness[strategy]

        # Simple moving average update
        if success:
            self.strategy_effectiveness[strategy] = min(
                1.0, current_effectiveness + EFFECTIVENESS_UPDATE_FACTOR
            )
        else:
            self.strategy_effectiveness[strategy] = max(
                EFFECTIVENESS_UPDATE_FACTOR,
                current_effectiveness - EFFECTIVENESS_UPDATE_FACTOR,
            )

    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get comprehensive recovery statistics.

        Returns:
            Dictionary with recovery performance metrics
        """
        return {
            "total_attempts": self.recovery_attempts,
            "strategy_effectiveness": dict(self.strategy_effectiveness),
            "pattern_cache_size": len(self.pattern_cache),
            "escalation_failures": self.escalation_failures,
            "success_rate": (
                1.0 - (self.escalation_failures / max(1, self.recovery_attempts))
            )
        }

    def reset_statistics(self) -> None:
        """Reset recovery statistics for new processing session."""
        self.recovery_attempts = 0
        self.escalation_failures = 0
        self.pattern_cache.clear()
        self.strategy_effectiveness = dict.fromkeys(RecoveryStrategy, 0.5)
        if self.history:
            self.history.clear_history()

    def get_recovery_history_report(self) -> Optional[Dict[str, Any]]:
        """Get comprehensive recovery history report.

        Returns:
            Detailed recovery history report or None if history is disabled
        """
        if not self.history:
            return None
        return self.history.generate_detailed_report()

    def get_recent_recovery_entries(
        self, count: int = 10
    ) -> List[RecoveryHistoryEntry]:
        """Get recent recovery history entries.

        Args:
            count: Number of recent entries to return

        Returns:
            List of recent recovery entries, empty if history disabled
        """
        if not self.history:
            return []
        return self.history.get_recent_entries(count)
