"""Token assembly and repair engine for smart XML token processing.

This module implements intelligent token assembly and repair that fixes common XML
issues to provide the highest quality tokens possible for downstream processing.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

from .tokenizer import Token, TokenPosition, TokenRepair, TokenType

# Constants for assembly and repair
MAX_TAG_NAME_LENGTH = 1000
MAX_ATTRIBUTE_VALUE_LENGTH = 10000
MAX_TEXT_MERGE_TOKENS = 50
XML_NAME_PATTERN = re.compile(r"^[a-zA-Z_:][-a-zA-Z0-9_:.]*$")
INVALID_TAG_CHARS = re.compile(r'[<>&"\']+')
ATTRIBUTE_QUOTE_PATTERN = re.compile(r'^["\'].*["\']$')
CONFIDENCE_HIGH_THRESHOLD = 0.8
CONFIDENCE_MID_THRESHOLD = 0.5
CONFIDENCE_LOW_THRESHOLD = 0.2

logger = logging.getLogger(__name__)


class RepairType(Enum):
    """Types of repairs that can be applied to tokens."""

    TAG_NAME_SANITIZATION = auto()      # Invalid character removal/replacement
    ATTRIBUTE_QUOTE_REPAIR = auto()     # Quote normalization and repair
    ATTRIBUTE_DUPLICATE_RESOLUTION = auto()  # Duplicate attribute handling
    TEXT_TOKEN_MERGING = auto()         # Adjacent text consolidation
    MALFORMED_TAG_REPAIR = auto()       # Missing brackets, structure fixes
    WHITESPACE_NORMALIZATION = auto()   # Whitespace handling
    CONFIDENCE_ADJUSTMENT = auto()      # Confidence score updates


class RepairSeverity(Enum):
    """Severity levels for repair operations."""

    MINOR = auto()      # Low impact on confidence (cosmetic fixes)
    MODERATE = auto()   # Medium impact on confidence (structural fixes)
    MAJOR = auto()      # High impact on confidence (data changes)
    CRITICAL = auto()   # Significant impact on confidence (major repairs)


@dataclass
class AssemblyResult:
    """Result of token assembly and repair operations."""

    tokens: List[Token]
    repairs_applied: int = 0
    confidence_adjustment: float = 0.0
    processing_time_ms: float = 0.0
    repair_summary: Dict[RepairType, int] = field(default_factory=dict)
    diagnostics: List[str] = field(default_factory=list)

    @property
    def repair_rate(self) -> float:
        """Calculate repair rate as percentage of tokens repaired."""
        if not self.tokens:
            return 0.0
        repaired_count = sum(1 for token in self.tokens if token.has_repairs)
        return repaired_count / len(self.tokens)


@dataclass
class AssemblyRepairAction:
    """Individual repair action with metadata for token assembly."""

    repair_type: RepairType
    severity: RepairSeverity
    description: str
    original_value: str
    repaired_value: str
    confidence_impact: float
    position: TokenPosition
    context: Dict[str, Any] = field(default_factory=dict)


class TokenAssemblyEngine:
    """Smart token assembly and repair engine.

    Provides intelligent assembly and repair of XML tokens to fix common issues
    and improve token quality for downstream processing.
    """
    def __init__(
        self,
        correlation_id: Optional[str] = None,
        enable_caching: bool = True,
        strict_mode: bool = False
    ) -> None:
        """Initialize the token assembly engine.

        Args:
            correlation_id: Optional correlation ID for tracking
            enable_caching: Enable repair result caching for performance
            strict_mode: Use strict XML compliance rules vs lenient repairs
        """
        self.correlation_id = correlation_id
        self.enable_caching = enable_caching
        self.strict_mode = strict_mode

        # Repair result caching for performance optimization
        self._repair_cache: Dict[str, AssemblyResult] = {}
        self._repair_statistics: Dict[RepairType, int] = {}

        # Context-aware repair rules
        self._tag_name_rules = self._init_tag_name_rules()
        self._attribute_rules = self._init_attribute_rules()
        self._text_merging_rules = self._init_text_merging_rules()

    def _init_tag_name_rules(self) -> Dict[str, Any]:
        """Initialize tag name sanitization rules."""
        return {
            "max_length": MAX_TAG_NAME_LENGTH,
            "invalid_char_replacement": "_",
            "preserve_namespaces": True,
            "allow_numeric_start": not self.strict_mode,
            "sanitize_patterns": [
                (r'[<>&"\']+', ""),          # Remove XML special chars
                (r"\s+", ""),                # Remove whitespace
                (r"^[0-9]", "_" if self.strict_mode else ""),  # Handle numeric start
                (r"[^\w\-.:_]", "_"),        # Replace other invalid chars
            ]
        }

    def _init_attribute_rules(self) -> Dict[str, Any]:
        """Initialize attribute handling rules."""
        return {
            "max_value_length": MAX_ATTRIBUTE_VALUE_LENGTH,
            "quote_preference": '"',      # Prefer double quotes
            "duplicate_resolution": "last_wins",  # Keep last duplicate
            "value_normalization": not self.strict_mode,
            "quote_repair_patterns": [
                (r'^(["\']).*(?!\1)$', r"\1VALUE\1"),  # Fix unmatched quotes
                (r'^([^"\']).*([^"\'])$', r'"\1VALUE\2"'),  # Add missing quotes
                (r'""', '"'),  # Fix double quotes
                (r"''", "'"),  # Fix double single quotes
            ]
        }

    def _init_text_merging_rules(self) -> Dict[str, Any]:
        """Initialize text token merging rules."""
        return {
            "max_merge_tokens": MAX_TEXT_MERGE_TOKENS,
            "preserve_whitespace": True,
            "normalize_whitespace": not self.strict_mode,
            "merge_adjacent_only": True,
            "confidence_threshold": CONFIDENCE_MID_THRESHOLD,
        }

    def assemble_and_repair_tokens(self, tokens: List[Token]) -> AssemblyResult:
        """Assemble and repair a list of tokens for optimal quality.

        Args:
            tokens: List of tokens to process

        Returns:
            AssemblyResult with processed tokens and repair metadata
        """
        import time
        start_time = time.time()

        logger.debug(
            "Starting token assembly and repair",
            extra={
                "component": "token_assembly_engine",
                "correlation_id": self.correlation_id,
                "input_token_count": len(tokens)
            }
        )

        # Check cache for identical token sequences
        if self.enable_caching:
            cache_key = self._generate_cache_key(tokens)
            if cache_key in self._repair_cache:
                cached_result = self._repair_cache[cache_key]
                logger.debug(
                    "Using cached assembly result",
                    extra={
                        "component": "token_assembly_engine",
                        "correlation_id": self.correlation_id,
                        "cache_key_hash": hash(cache_key) % 10000
                    }
                )
                return cached_result

        # Initialize result tracking
        repair_actions: List[AssemblyRepairAction] = []
        processed_tokens = tokens.copy()

        # Phase 1: Tag name sanitization and malformed tag repair
        processed_tokens, tag_repairs = self._repair_tag_names(processed_tokens)
        repair_actions.extend(tag_repairs)

        # Phase 2: Attribute handling and repair
        processed_tokens, attr_repairs = self._repair_attributes(processed_tokens)
        repair_actions.extend(attr_repairs)

        # Phase 3: Text token merging
        processed_tokens, text_repairs = self._merge_text_tokens(processed_tokens)
        repair_actions.extend(text_repairs)

        # Phase 4: Apply confidence scoring adjustments
        processed_tokens = self._apply_confidence_adjustments(
            processed_tokens, repair_actions
        )

        # Calculate processing metrics
        processing_time_ms = (time.time() - start_time) * 1000
        total_confidence_impact = sum(action.confidence_impact for action in repair_actions)

        # Build repair summary
        repair_summary = {}
        for action in repair_actions:
            if action.repair_type not in repair_summary:
                repair_summary[action.repair_type] = 0
            repair_summary[action.repair_type] += 1

        # Update statistics
        for repair_type, count in repair_summary.items():
            if repair_type not in self._repair_statistics:
                self._repair_statistics[repair_type] = 0
            self._repair_statistics[repair_type] += count

        # Create result
        result = AssemblyResult(
            tokens=processed_tokens,
            repairs_applied=len(repair_actions),
            confidence_adjustment=total_confidence_impact,
            processing_time_ms=processing_time_ms,
            repair_summary=repair_summary,
            diagnostics=[action.description for action in repair_actions[:10]]  # Top 10
        )

        # Cache result if enabled
        if self.enable_caching and len(self._repair_cache) < 1000:  # Prevent unlimited growth
            self._repair_cache[cache_key] = result

        logger.debug(
            "Token assembly and repair completed",
            extra={
                "component": "token_assembly_engine",
                "correlation_id": self.correlation_id,
                "output_token_count": len(processed_tokens),
                "repairs_applied": len(repair_actions),
                "processing_time_ms": processing_time_ms
            }
        )

        return result

    def _repair_tag_names(self, tokens: List[Token]) -> Tuple[List[Token], List[AssemblyRepairAction]]:
        """Repair and sanitize tag names for XML validity.

        Args:
            tokens: List of tokens to process

        Returns:
            Tuple of (repaired_tokens, repair_actions)
        """
        repaired_tokens = []
        repair_actions = []

        for token in tokens:
            if token.type == TokenType.TAG_NAME:
                original_value = token.value
                repaired_value, repairs = self._sanitize_tag_name(original_value)

                if repaired_value != original_value:
                    # Create repair action
                    severity = RepairSeverity.MODERATE if len(repairs) > 2 else RepairSeverity.MINOR
                    action = AssemblyRepairAction(
                        repair_type=RepairType.TAG_NAME_SANITIZATION,
                        severity=severity,
                        description=f"Sanitized tag name '{original_value}' to '{repaired_value}'",
                        original_value=original_value,
                        repaired_value=repaired_value,
                        confidence_impact=-0.1 * len(repairs),
                        position=token.position,
                        context={"sanitization_rules_applied": len(repairs)}
                    )
                    repair_actions.append(action)

                    # Create repaired token
                    token_repair = TokenRepair(
                        repair_type="tag_name_sanitization",
                        description=action.description,
                        original_content=original_value,
                        repaired_content=repaired_value,
                        confidence_impact=action.confidence_impact
                    )

                    repaired_token = Token(
                        type=token.type,
                        value=repaired_value,
                        position=token.position,
                        confidence=max(0.1, token.confidence + action.confidence_impact),
                        repairs=[*token.repairs, token_repair],
                        raw_content=token.raw_content
                    )
                    repaired_tokens.append(repaired_token)
                else:
                    repaired_tokens.append(token)
            else:
                repaired_tokens.append(token)

        return repaired_tokens, repair_actions

    def _sanitize_tag_name(self, tag_name: str) -> Tuple[str, List[str]]:
        """Sanitize a tag name according to XML validity rules.

        Args:
            tag_name: Original tag name to sanitize

        Returns:
            Tuple of (sanitized_name, list_of_applied_rules)
        """
        if not tag_name:
            return "unknown", ["empty_name_replacement"]

        sanitized = tag_name
        applied_rules = []

        # Apply sanitization patterns
        for pattern, replacement in self._tag_name_rules["sanitize_patterns"]:
            if re.search(pattern, sanitized):
                sanitized = re.sub(pattern, replacement, sanitized)
                applied_rules.append(f"pattern_{pattern}")

        # Handle length constraints
        if len(sanitized) > self._tag_name_rules["max_length"]:
            sanitized = sanitized[:self._tag_name_rules["max_length"]]
            applied_rules.append("length_truncation")

        # Ensure valid XML name start
        if sanitized and not XML_NAME_PATTERN.match(sanitized[0]):
            if sanitized[0].isdigit() and not self._tag_name_rules["allow_numeric_start"]:
                sanitized = self._tag_name_rules["invalid_char_replacement"] + sanitized
                applied_rules.append("numeric_start_fix")

        # Final validation - if completely invalid, use fallback
        if not sanitized or not XML_NAME_PATTERN.match(sanitized):
            sanitized = "repaired_tag"
            applied_rules.append("fallback_name")

        return sanitized, applied_rules

    def _repair_attributes(self, tokens: List[Token]) -> Tuple[List[Token], List[AssemblyRepairAction]]:
        """Repair attribute names and values.

        Args:
            tokens: List of tokens to process

        Returns:
            Tuple of (repaired_tokens, repair_actions)
        """
        repaired_tokens = []
        repair_actions = []

        # Track attribute names to detect duplicates
        current_tag_attributes: set[str] = set()
        in_tag = False

        for token in tokens:
            if token.type == TokenType.TAG_START:
                in_tag = True
                current_tag_attributes.clear()
                repaired_tokens.append(token)

            elif token.type == TokenType.TAG_END:
                in_tag = False
                current_tag_attributes.clear()
                repaired_tokens.append(token)

            elif token.type == TokenType.ATTR_NAME and in_tag:
                original_name = token.value

                # Check for duplicate attributes
                if original_name in current_tag_attributes:
                    # Handle duplicate based on resolution strategy
                    if self._attribute_rules["duplicate_resolution"] == "last_wins":
                        # Create repair action for duplicate handling
                        action = AssemblyRepairAction(
                            repair_type=RepairType.ATTRIBUTE_DUPLICATE_RESOLUTION,
                            severity=RepairSeverity.MINOR,
                            description=f"Duplicate attribute '{original_name}' resolved (keeping last)",
                            original_value=original_name,
                            repaired_value=original_name,
                            confidence_impact=-0.05,
                            position=token.position,
                            context={"duplicate_count": 2}  # Simplified for now
                        )
                        repair_actions.append(action)

                        # Add repair metadata to token
                        token_repair = TokenRepair(
                            repair_type="duplicate_attribute_resolution",
                            description=action.description,
                            original_content=original_name,
                            repaired_content=original_name,
                            confidence_impact=action.confidence_impact
                        )

                        repaired_token = Token(
                            type=token.type,
                            value=token.value,
                            position=token.position,
                            confidence=max(0.1, token.confidence + action.confidence_impact),
                            repairs=[*token.repairs, token_repair],
                            raw_content=token.raw_content
                        )
                        repaired_tokens.append(repaired_token)
                else:
                    current_tag_attributes.add(original_name)
                    repaired_tokens.append(token)

            elif token.type == TokenType.ATTR_VALUE and in_tag:
                original_value = token.value
                repaired_value, value_repairs = self._repair_attribute_value(original_value)

                if repaired_value != original_value or value_repairs:
                    # Create repair action for attribute value
                    severity = RepairSeverity.MINOR if len(value_repairs) == 1 else RepairSeverity.MODERATE
                    action = AssemblyRepairAction(
                        repair_type=RepairType.ATTRIBUTE_QUOTE_REPAIR,
                        severity=severity,
                        description=f"Repaired attribute value '{original_value}' to '{repaired_value}'",
                        original_value=original_value,
                        repaired_value=repaired_value,
                        confidence_impact=-0.05 * len(value_repairs),
                        position=token.position,
                        context={"repairs_applied": value_repairs}
                    )
                    repair_actions.append(action)

                    # Create repaired token
                    token_repair = TokenRepair(
                        repair_type="attribute_value_repair",
                        description=action.description,
                        original_content=original_value,
                        repaired_content=repaired_value,
                        confidence_impact=action.confidence_impact
                    )

                    repaired_token = Token(
                        type=token.type,
                        value=repaired_value,
                        position=token.position,
                        confidence=max(0.1, token.confidence + action.confidence_impact),
                        repairs=[*token.repairs, token_repair],
                        raw_content=token.raw_content
                    )
                    repaired_tokens.append(repaired_token)
                else:
                    repaired_tokens.append(token)
            else:
                repaired_tokens.append(token)

        return repaired_tokens, repair_actions

    def _repair_attribute_value(self, attr_value: str) -> Tuple[str, List[str]]:
        """Repair an attribute value with quote normalization.

        Args:
            attr_value: Original attribute value

        Returns:
            Tuple of (repaired_value, list_of_applied_repairs)
        """
        if not attr_value:
            return '""', ["empty_value_quoted"]

        repaired = attr_value
        applied_repairs = []

        # Handle length constraints
        if len(repaired) > self._attribute_rules["max_value_length"]:
            repaired = repaired[:self._attribute_rules["max_value_length"]]
            applied_repairs.append("length_truncation")

        # Skip quote repair patterns - the tokenizer already handles quotes correctly
        # and sends us the unquoted content. Quote repair patterns would corrupt
        # correctly tokenized attribute values.
        
        # For attribute values, we trust the tokenizer's output and only apply
        # length constraints and basic validation.

        return repaired, applied_repairs

    def _merge_text_tokens(self, tokens: List[Token]) -> Tuple[List[Token], List[AssemblyRepairAction]]:
        """Merge adjacent text tokens for cleaner content.

        Args:
            tokens: List of tokens to process

        Returns:
            Tuple of (merged_tokens, repair_actions)
        """
        if len(tokens) <= 1:
            return tokens, []

        merged_tokens = []
        repair_actions = []
        i = 0

        while i < len(tokens):
            current_token = tokens[i]

            if current_token.type == TokenType.TEXT:
                # Look for adjacent text tokens to merge
                merge_candidates = [current_token]
                j = i + 1

                while (j < len(tokens) and
                       j < i + self._text_merging_rules["max_merge_tokens"] and
                       tokens[j].type == TokenType.TEXT):
                    merge_candidates.append(tokens[j])
                    j += 1

                if len(merge_candidates) > 1:
                    # Merge the text tokens
                    merged_value = ""
                    lowest_confidence = 1.0
                    all_repairs = []

                    for candidate in merge_candidates:
                        if self._text_merging_rules["preserve_whitespace"]:
                            merged_value += candidate.value
                        else:
                            merged_value += candidate.value.strip() + " "
                        lowest_confidence = min(lowest_confidence, candidate.confidence)
                        all_repairs.extend(candidate.repairs)

                    if self._text_merging_rules["normalize_whitespace"]:
                        merged_value = re.sub(r"\s+", " ", merged_value.strip())

                    # Create repair action
                    action = AssemblyRepairAction(
                        repair_type=RepairType.TEXT_TOKEN_MERGING,
                        severity=RepairSeverity.MINOR,
                        description=f"Merged {len(merge_candidates)} adjacent text tokens",
                        original_value=f"[{len(merge_candidates)} tokens]",
                        repaired_value=merged_value[:50] + "..." if len(merged_value) > 50 else merged_value,
                        confidence_impact=0.05,  # Slight confidence boost for merging
                        position=current_token.position,
                        context={"merged_token_count": len(merge_candidates)}
                    )
                    repair_actions.append(action)

                    # Create merged token
                    token_repair = TokenRepair(
                        repair_type="text_token_merging",
                        description=action.description,
                        original_content=f"[{len(merge_candidates)} separate tokens]",
                        repaired_content="[merged token]",
                        confidence_impact=action.confidence_impact
                    )

                    merged_token = Token(
                        type=TokenType.TEXT,
                        value=merged_value,
                        position=current_token.position,
                        confidence=min(1.0, lowest_confidence + action.confidence_impact),
                        repairs=[*all_repairs, token_repair],
                        raw_content=merged_value
                    )
                    merged_tokens.append(merged_token)

                    # Skip the merged tokens
                    i = j
                else:
                    merged_tokens.append(current_token)
                    i += 1
            else:
                merged_tokens.append(current_token)
                i += 1

        return merged_tokens, repair_actions

    def _apply_confidence_adjustments(
        self,
        tokens: List[Token],
        repair_actions: List[AssemblyRepairAction]
    ) -> List[Token]:
        """Apply confidence score adjustments based on repair actions.

        Args:
            tokens: List of tokens to adjust
            repair_actions: List of repair actions that were applied

        Returns:
            List of tokens with adjusted confidence scores
        """
        # Create position-based mapping of confidence impacts
        position_impacts = {}
        for action in repair_actions:
            pos_key = f"{action.position.line}:{action.position.column}:{action.position.offset}"
            if pos_key not in position_impacts:
                position_impacts[pos_key] = 0.0
            position_impacts[pos_key] += action.confidence_impact

        adjusted_tokens = []
        for token in tokens:
            pos_key = f"{token.position.line}:{token.position.column}:{token.position.offset}"

            if pos_key in position_impacts:
                # Apply additional confidence adjustment if not already applied
                if not any(repair.repair_type == "confidence_adjustment" for repair in token.repairs):
                    additional_impact = position_impacts[pos_key] * 0.1  # Damping factor

                    if abs(additional_impact) > 0.01:  # Only if significant
                        token_repair = TokenRepair(
                            repair_type="confidence_adjustment",
                            description=f"Confidence adjusted by {additional_impact:.3f} due to repairs",
                            original_content=f"{token.confidence:.3f}",
                            repaired_content=f"{max(0.0, min(1.0, token.confidence + additional_impact)):.3f}",
                            confidence_impact=additional_impact
                        )

                        adjusted_token = Token(
                            type=token.type,
                            value=token.value,
                            position=token.position,
                            confidence=max(0.0, min(1.0, token.confidence + additional_impact)),
                            repairs=[*token.repairs, token_repair],
                            raw_content=token.raw_content
                        )
                        adjusted_tokens.append(adjusted_token)
                    else:
                        adjusted_tokens.append(token)
                else:
                    adjusted_tokens.append(token)
            else:
                adjusted_tokens.append(token)

        return adjusted_tokens

    def _generate_cache_key(self, tokens: List[Token]) -> str:
        """Generate a cache key for a token sequence.

        Args:
            tokens: List of tokens

        Returns:
            Cache key string
        """
        # Generate key based on token types and values
        key_parts = []
        for token in tokens[:20]:  # Limit to first 20 tokens for performance
            key_parts.append(f"{token.type.name}:{hash(token.value) % 10000}")
        return "|".join(key_parts)

    def get_repair_statistics(self) -> Dict[str, Any]:
        """Get comprehensive repair statistics.

        Returns:
            Dictionary with repair performance metrics
        """
        return {
            "repair_counts": dict(self._repair_statistics),
            "cache_size": len(self._repair_cache),
            "cache_enabled": self.enable_caching,
            "strict_mode": self.strict_mode,
            "total_repairs": sum(self._repair_statistics.values())
        }

    def clear_cache(self) -> None:
        """Clear the repair result cache."""
        self._repair_cache.clear()

    def reset_statistics(self) -> None:
        """Reset repair statistics."""
        self._repair_statistics.clear()
