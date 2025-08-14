"""Character transformation system with never-fail guarantee.

This module implements comprehensive character transformation for handling invalid
XML characters through configurable strategies: removal, replacement, escape,
mapping, and preservation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Dict, List, Optional, Tuple

# XML 1.0 valid character ranges
XML_VALID_RANGES: List[Tuple[int, int]] = [
    (0x0009, 0x0009),  # Tab
    (0x000A, 0x000A),  # Line Feed
    (0x000D, 0x000D),  # Carriage Return
    (0x0020, 0xD7FF),  # Basic Multilingual Plane excluding surrogates
    (0xE000, 0xFFFD),  # Private Use and extended characters
    (0x10000, 0x10FFFF),  # Supplementary planes
]

# Surrogate range (invalid in XML)
SURROGATE_RANGE_START = 0xD800
SURROGATE_RANGE_END = 0xDFFF

# Non-characters (invalid in XML)
NONCHARACTER_RANGES: List[Tuple[int, int]] = [
    (0xFFFE, 0xFFFF),  # BMP non-characters
]

# Additional non-characters in supplementary planes
SUPPLEMENTARY_NONCHARS = [
    0x1FFFE, 0x1FFFF, 0x2FFFE, 0x2FFFF, 0x3FFFE, 0x3FFFF,
    0x4FFFE, 0x4FFFF, 0x5FFFE, 0x5FFFF, 0x6FFFE, 0x6FFFF,
    0x7FFFE, 0x7FFFF, 0x8FFFE, 0x8FFFF, 0x9FFFE, 0x9FFFF,
    0xAFFFE, 0xAFFFF, 0xBFFFE, 0xBFFFF, 0xCFFFE, 0xCFFFF,
    0xDFFFE, 0xDFFFF, 0xEFFFE, 0xEFFFF, 0xFFFFE, 0xFFFFF,
    0x10FFFE  # Note: 0x10FFFF is valid, only 0x10FFFE is a non-character
]

# Control characters (except allowed ones)
CONTROL_CHARS_START = 0x0000
CONTROL_CHARS_END = 0x001F
ALLOWED_CONTROL_CHARS = {0x0009, 0x000A, 0x000D}

# Performance optimization constants
FAST_PATH_ASCII_MAX = 0x7F
CACHE_SIZE_LIMIT = 1000


class TransformationStrategy(Enum):
    """Enumeration of character transformation strategies."""
    REMOVAL = "removal"
    REPLACEMENT = "replacement"
    ESCAPE = "escape"
    MAPPING = "mapping"
    PRESERVATION = "preservation"


class TransformationContext(Enum):
    """Enumeration of XML transformation contexts."""
    TAG_NAME = "tag_name"
    ATTRIBUTE_NAME = "attribute_name"
    ATTRIBUTE_VALUE = "attribute_value"
    TEXT_CONTENT = "text_content"


@dataclass
class TransformationChange:
    """Record of a single character transformation.

    Attributes:
        position: Character position in original text
        original_char: Original character
        transformed_char: Character after transformation (empty string if removed)
        strategy: Strategy used for transformation
        reason: Reason for transformation
    """
    position: int
    original_char: str
    transformed_char: str
    strategy: TransformationStrategy
    reason: str


@dataclass
class TransformResult:
    """Result of character transformation with metadata and diagnostics.

    Attributes:
        text: Transformed text
        confidence: Confidence score (0.0-1.0)
        changes: List of transformations made
        statistics: Transformation statistics
        issues: List of issues encountered
    """
    text: str
    confidence: float
    changes: List[TransformationChange] = field(default_factory=list)
    statistics: Dict[str, int] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate confidence score range."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Confidence must be between 0.0 and 1.0, got {self.confidence}"
            )


@dataclass
class TransformConfig:
    """Configuration for character transformation.

    Attributes:
        default_strategy: Default transformation strategy
        context_strategies: Strategy per context
        custom_mappings: Custom character mappings
        replacement_char: Character to use for replacement strategy
        preserve_whitespace: Whether to preserve whitespace characters
        strict_xml: Whether to enforce strict XML 1.0 compliance
    """
    default_strategy: TransformationStrategy = TransformationStrategy.REPLACEMENT
    context_strategies: Dict[TransformationContext, TransformationStrategy] = field(
        default_factory=dict
    )
    custom_mappings: Dict[str, str] = field(default_factory=dict)
    replacement_char: str = "\uFFFD"  # Unicode replacement character
    preserve_whitespace: bool = True
    strict_xml: bool = True

    @classmethod
    def create_preset(cls, preset: str) -> "TransformConfig":
        """Create configuration preset.

        Args:
            preset: Preset name ('strict', 'lenient', 'data_recovery')

        Returns:
            Configured TransformConfig instance
        """
        if preset == "strict":
            return cls(
                default_strategy=TransformationStrategy.REMOVAL,
                strict_xml=True,
                preserve_whitespace=False
            )
        if preset == "lenient":
            return cls(
                default_strategy=TransformationStrategy.REPLACEMENT,
                context_strategies={
                    TransformationContext.TAG_NAME: TransformationStrategy.REMOVAL,
                    TransformationContext.ATTRIBUTE_NAME: (
                        TransformationStrategy.REMOVAL
                    ),
                },
                strict_xml=True,
                preserve_whitespace=True
            )
        if preset == "data_recovery":
            return cls(
                default_strategy=TransformationStrategy.PRESERVATION,
                context_strategies={
                    TransformationContext.TAG_NAME: TransformationStrategy.REPLACEMENT,
                    TransformationContext.ATTRIBUTE_NAME: (
                        TransformationStrategy.REPLACEMENT
                    ),
                },
                strict_xml=False,
                preserve_whitespace=True
            )
        raise ValueError(f"Unknown preset: {preset}")


class XML10Validator:
    """XML 1.0 character validity checker with comprehensive validation."""

    # Cache for validation results to improve performance
    _validation_cache: ClassVar[Dict[int, bool]] = {}

    @classmethod
    def is_valid_xml_char(cls, char_code: int) -> bool:
        """Check if character code is valid in XML 1.0.

        Args:
            char_code: Unicode code point

        Returns:
            True if character is valid in XML 1.0
        """
        # Check cache first
        if char_code in cls._validation_cache:
            return cls._validation_cache[char_code]

        # Validate against XML 1.0 ranges
        is_valid = cls._validate_character_code(char_code)

        # Cache result if cache isn't too large
        if len(cls._validation_cache) < CACHE_SIZE_LIMIT:
            cls._validation_cache[char_code] = is_valid

        return is_valid

    @classmethod
    def _validate_character_code(cls, char_code: int) -> bool:
        """Internal validation logic for character codes."""
        # Check if in valid ranges
        for start, end in XML_VALID_RANGES:
            if start <= char_code <= end:
                # Further check for non-characters within valid ranges
                if char_code in SUPPLEMENTARY_NONCHARS:
                    return False
                # Check BMP non-characters
                for nstart, nend in NONCHARACTER_RANGES:
                    if nstart <= char_code <= nend:
                        return False
                return True

        return False

    @classmethod
    def get_validation_issues(cls, char_code: int) -> List[str]:
        """Get detailed validation issues for a character.

        Args:
            char_code: Unicode code point

        Returns:
            List of validation issues
        """
        issues = []

        if not cls.is_valid_xml_char(char_code):
            if SURROGATE_RANGE_START <= char_code <= SURROGATE_RANGE_END:
                issues.append(f"Surrogate character: U+{char_code:04X}")
            elif (CONTROL_CHARS_START <= char_code <= CONTROL_CHARS_END
                  and char_code not in ALLOWED_CONTROL_CHARS):
                issues.append(f"Control character: U+{char_code:04X}")
            elif char_code in SUPPLEMENTARY_NONCHARS or any(
                start <= char_code <= end for start, end in NONCHARACTER_RANGES
            ):
                issues.append(f"Non-character: U+{char_code:04X}")
            else:
                issues.append(f"Invalid XML character: U+{char_code:04X}")

        return issues

    @classmethod
    def clear_cache(cls) -> None:
        """Clear validation cache."""
        cls._validation_cache.clear()


class CharacterTransformer:
    """Main character transformation engine with context-aware processing."""

    def __init__(self, config: Optional[TransformConfig] = None) -> None:
        """Initialize transformer with configuration.

        Args:
            config: Transformation configuration (uses default if None)
        """
        self.config = config or TransformConfig()
        self.validator = XML10Validator()

        # HTML entity mappings for escape strategy
        self._html_entities = {
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&apos;",
        }

    def transform(
        self,
        text: str,
        context: TransformationContext = TransformationContext.TEXT_CONTENT
    ) -> TransformResult:
        """Transform text according to configuration and context.

        Args:
            text: Input text to transform
            context: Transformation context

        Returns:
            TransformResult with transformed text and metadata
        """
        if not text:
            return TransformResult(
                text="",
                confidence=1.0,
                statistics={"total_chars": 0, "transformed_chars": 0}
            )

        # Fast path for ASCII-only text
        if self._is_ascii_only(text):
            return self._fast_path_transform(text, context)

        # Full transformation with validation
        return self._full_transform(text, context)

    def _is_ascii_only(self, text: str) -> bool:
        """Check if text contains only ASCII characters."""
        try:
            return all(ord(char) <= FAST_PATH_ASCII_MAX for char in text)
        except Exception:
            return False

    def _fast_path_transform(
        self, text: str, context: TransformationContext
    ) -> TransformResult:
        """Fast path transformation for ASCII-only text."""
        # ASCII is always valid in XML except for control characters
        issues = []
        changes = []
        transformed_chars = []

        for i, char in enumerate(text):
            char_code = ord(char)

            # Check custom mappings first (even for valid characters)
            if char in self.config.custom_mappings:
                transformed_char, change = self._apply_strategy(
                    char, i, TransformationStrategy.MAPPING, "Custom mapping"
                )
                transformed_chars.append(transformed_char)
                if change:
                    changes.append(change)
            elif not self.validator.is_valid_xml_char(char_code):
                # Handle invalid control characters
                strategy = self._get_strategy_for_context(context)
                reason = f"Invalid control character: U+{char_code:04X}"
                transformed_char, change = self._apply_strategy(
                    char, i, strategy, reason
                )
                transformed_chars.append(transformed_char)
                if change:
                    changes.append(change)
                issues.append(reason)
            else:
                transformed_chars.append(char)

        result_text = "".join(transformed_chars)
        confidence = 1.0 if not changes else max(0.8, 1.0 - (len(changes) / len(text)))

        return TransformResult(
            text=result_text,
            confidence=confidence,
            changes=changes,
            statistics={
                "total_chars": len(text),
                "transformed_chars": len(changes),
                "invalid_chars": len([
                    c for c in changes
                    if c.strategy != TransformationStrategy.PRESERVATION
                ]),
                "ascii_fast_path": True
            },
            issues=list(set(issues))  # Remove duplicates
        )

    def _full_transform(
        self, text: str, context: TransformationContext
    ) -> TransformResult:
        """Full transformation with comprehensive validation."""
        changes = []
        issues = []
        transformed_chars = []

        for i, char in enumerate(text):
            char_code = ord(char)

            # Check custom mappings first (even for valid characters)
            if char in self.config.custom_mappings:
                transformed_char, change = self._apply_strategy(
                    char, i, TransformationStrategy.MAPPING, "Custom mapping"
                )
                transformed_chars.append(transformed_char)
                if change:
                    changes.append(change)
            else:
                validation_issues = self.validator.get_validation_issues(char_code)

                if validation_issues:
                    # Character needs transformation
                    strategy = self._get_strategy_for_context(context)
                    reason = "; ".join(validation_issues)
                    transformed_char, change = self._apply_strategy(
                        char, i, strategy, reason
                    )
                    transformed_chars.append(transformed_char)
                    if change:
                        changes.append(change)
                    issues.extend(validation_issues)
                else:
                    # Character is valid
                    transformed_chars.append(char)

        result_text = "".join(transformed_chars)
        confidence = self._calculate_confidence(len(text), len(changes), len(issues))

        return TransformResult(
            text=result_text,
            confidence=confidence,
            changes=changes,
            statistics={
                "total_chars": len(text),
                "transformed_chars": len(changes),
                "invalid_chars": len([
                    c for c in changes
                    if c.strategy != TransformationStrategy.PRESERVATION
                ]),
                "ascii_fast_path": False
            },
            issues=list(set(issues))  # Remove duplicates
        )

    def _get_strategy_for_context(
        self, context: TransformationContext
    ) -> TransformationStrategy:
        """Get transformation strategy for given context."""
        return self.config.context_strategies.get(
            context, self.config.default_strategy
        )

    def _apply_strategy(
        self,
        char: str,
        position: int,
        strategy: TransformationStrategy,
        reason: str
    ) -> Tuple[str, Optional[TransformationChange]]:
        """Apply transformation strategy to character.

        Returns:
            Tuple of (transformed_char, change_record)
        """
        # Check custom mappings first
        if char in self.config.custom_mappings:
            transformed = self.config.custom_mappings[char]
            change = TransformationChange(
                position=position,
                original_char=char,
                transformed_char=transformed,
                strategy=TransformationStrategy.MAPPING,
                reason="Custom mapping"
            )
            return transformed, change

        # Apply strategy
        if strategy == TransformationStrategy.REMOVAL:
            change = TransformationChange(
                position=position,
                original_char=char,
                transformed_char="",
                strategy=strategy,
                reason=reason
            )
            return "", change

        if strategy == TransformationStrategy.REPLACEMENT:
            change = TransformationChange(
                position=position,
                original_char=char,
                transformed_char=self.config.replacement_char,
                strategy=strategy,
                reason=reason
            )
            return self.config.replacement_char, change

        if strategy == TransformationStrategy.ESCAPE:
            if char in self._html_entities:
                escaped = self._html_entities[char]
            else:
                escaped = f"&#{ord(char)};"
            change = TransformationChange(
                position=position,
                original_char=char,
                transformed_char=escaped,
                strategy=strategy,
                reason=reason
            )
            return escaped, change

        if strategy == TransformationStrategy.PRESERVATION:
            # Keep character but record the issue
            change = TransformationChange(
                position=position,
                original_char=char,
                transformed_char=char,
                strategy=strategy,
                reason=f"Preserved: {reason}"
            )
            return char, change

        # Fallback to replacement
        change = TransformationChange(
            position=position,
            original_char=char,
            transformed_char=self.config.replacement_char,
            strategy=TransformationStrategy.REPLACEMENT,
            reason=f"Unknown strategy fallback: {reason}"
        )
        return self.config.replacement_char, change

    def _calculate_confidence(
        self, total_chars: int, changes_count: int, issues_count: int
    ) -> float:
        """Calculate confidence score based on transformation results."""
        if total_chars == 0:
            return 1.0

        # Base confidence starts high
        confidence = 1.0

        # Reduce confidence based on proportion of changes
        change_ratio = changes_count / total_chars
        confidence -= change_ratio * 0.3

        # Further reduce for issues
        issue_penalty = min(0.5, issues_count * 0.1)
        confidence -= issue_penalty

        # Ensure confidence stays in valid range
        return max(0.1, min(1.0, confidence))

