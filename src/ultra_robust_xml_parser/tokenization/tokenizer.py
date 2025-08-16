"""Core XML tokenization implementation with robust state machine.

This module implements a fault-tolerant XML tokenizer that converts character streams
into meaningful XML tokens using a never-fail state machine approach.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum, auto

# Import recovery engine after declaration to avoid circular imports
# We'll use TYPE_CHECKING to handle the circular import
from typing import TYPE_CHECKING, List, Optional, Dict, Any

from ultra_robust_xml_parser.character import CharacterStreamResult

if TYPE_CHECKING:
    from .recovery import ErrorRecoveryEngine

# Constants for tokenization
HIGH_CONFIDENCE_THRESHOLD = 0.9
XML_CDATA_MAX_LENGTH = 9  # Length of "<![CDATA["
PI_MIN_LENGTH = 2  # Minimum length for processing instruction
UNICODE_START_OFFSET = 0x80  # Start of Unicode characters

logger = logging.getLogger(__name__)


class TokenType(Enum):
    """XML token types supported by the tokenizer."""

    TAG_START = auto()          # Opening tag marker: <
    TAG_END = auto()            # Closing tag marker: >
    TAG_NAME = auto()           # Element name within tags
    ATTR_NAME = auto()          # Attribute name within tags
    ATTR_VALUE = auto()         # Attribute value (quoted or unquoted)
    TEXT = auto()               # Character content between tags
    COMMENT = auto()            # XML comments: <!-- ... -->
    CDATA = auto()              # CDATA sections: <![CDATA[ ... ]]>
    PROCESSING_INSTRUCTION = auto()  # Processing instructions: <?xml ... ?>
    DOCTYPE = auto()            # DOCTYPE declarations
    WHITESPACE = auto()         # Whitespace between tokens
    ERROR = auto()              # Error recovery token

    # Enhanced token types for malformed content
    MALFORMED_TAG = auto()      # Malformed tag structure
    ORPHANED_END = auto()       # Orphaned closing tag without opening
    INVALID_CHARS = auto()      # Invalid character sequences

    # Recovery token types for synthetic content
    RECOVERED_CONTENT = auto()  # Content recovered from malformation
    SYNTHETIC_CLOSE = auto()    # Synthetic closing tag inserted by recovery
    BALANCED_STRUCTURE = auto() # Synthetic structure for tag balancing


class TokenizerState(Enum):
    """State machine states for XML tokenization."""

    TEXT_CONTENT = auto()       # Processing text content
    TAG_OPENING = auto()        # Inside opening tag <
    TAG_CLOSING = auto()        # Inside closing tag </
    TAG_NAME = auto()           # Reading tag name
    ATTR_NAME = auto()          # Reading attribute name
    ATTR_VALUE_START = auto()   # Before attribute value
    ATTR_VALUE_QUOTED = auto()  # Inside quoted attribute value
    ATTR_VALUE_UNQUOTED = auto()  # Inside unquoted attribute value
    COMMENT_START = auto()      # Starting comment <!--
    COMMENT_CONTENT = auto()    # Inside comment content
    COMMENT_END = auto()        # Ending comment -->
    CDATA_START = auto()        # Starting CDATA <![CDATA[
    CDATA_CONTENT = auto()      # Inside CDATA content
    CDATA_END = auto()          # Ending CDATA ]]>
    PI_START = auto()           # Starting processing instruction <?
    PI_CONTENT = auto()         # Inside processing instruction
    PI_END = auto()             # Ending processing instruction ?>
    DOT_PREFIX = auto()         # Processing dot-prefixed tag (e.g., <.doc>)
    DOT_PREFIX_CLOSING = auto() # Processing dot-prefixed closing tag (e.g., </.doc>)
    ERROR_RECOVERY = auto()     # Error recovery state


@dataclass
class TokenPosition:
    """Position information for XML tokens."""

    line: int
    column: int
    offset: int

    def __post_init__(self) -> None:
        """Validate position values."""
        if self.line < 1:
            raise ValueError("Line number must be >= 1")
        if self.column < 1:
            raise ValueError("Column number must be >= 1")
        if self.offset < 0:
            raise ValueError("Offset must be >= 0")


@dataclass
class TokenRepair:
    """Information about repairs made to recover from malformed XML."""

    repair_type: str
    description: str
    original_content: str
    repaired_content: str
    confidence_impact: float


@dataclass
class Token:
    """Represents a single XML token with position and confidence information."""

    type: TokenType
    value: str
    position: TokenPosition
    confidence: float = 1.0
    repairs: List[TokenRepair] = field(default_factory=list)
    raw_content: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate token values."""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")

    @property
    def has_repairs(self) -> bool:
        """Check if this token has any repairs."""
        return len(self.repairs) > 0

    @property
    def is_well_formed(self) -> bool:
        """Check if this token is well-formed (no repairs and high confidence)."""
        return not self.has_repairs and self.confidence >= HIGH_CONFIDENCE_THRESHOLD


@dataclass
class TokenizationResult:
    """Result of tokenization process with comprehensive metadata."""

    tokens: List[Token]
    success: bool = True
    confidence: float = 1.0
    total_repairs: int = 0
    error_count: int = 0
    processing_time: float = 0.0
    character_count: int = 0
    diagnostics: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Calculate derived statistics."""
        if self.tokens:
            self.total_repairs = sum(len(token.repairs) for token in self.tokens)
            self.error_count = sum(
                1 for token in self.tokens if token.type == TokenType.ERROR
            )
            self.confidence = (
                min(token.confidence for token in self.tokens) if self.tokens else 1.0
            )

    @property
    def token_count(self) -> int:
        """Get the total number of tokens."""
        return len(self.tokens)

    @property
    def well_formed_percentage(self) -> float:
        """Calculate percentage of well-formed tokens."""
        if not self.tokens:
            return 1.0
        well_formed_count = sum(1 for token in self.tokens if token.is_well_formed)
        return well_formed_count / len(self.tokens)


class XMLTokenizer:
    """Main XML tokenizer with robust state machine processing.
    
    Converts character streams into XML tokens using a fault-tolerant state machine
    that never fails and provides comprehensive error recovery.
    """

    def __init__(
        self,
        correlation_id: Optional[str] = None,
        enable_fast_path: bool = True,
        enable_recovery: bool = True,
        enable_assembly: bool = True
    ) -> None:
        """Initialize the XML tokenizer.
        
        Args:
            correlation_id: Optional correlation ID for tracking requests
            enable_fast_path: Enable fast-path optimization for well-formed XML
            enable_recovery: Enable error recovery engine for malformed XML
            enable_assembly: Enable token assembly and repair for higher quality tokens
        """
        self.correlation_id = correlation_id
        self.enable_fast_path = enable_fast_path
        self.enable_recovery = enable_recovery
        self.enable_assembly = enable_assembly

        # Initialize recovery engine if enabled (avoid circular import)
        self.recovery_engine: Optional["ErrorRecoveryEngine"] = None
        if enable_recovery:
            # Import here to avoid circular dependency
            from .recovery import ErrorRecoveryEngine
            self.recovery_engine = ErrorRecoveryEngine(
                correlation_id=correlation_id,
                enable_history=True
            )
            
        # Initialize assembly engine if enabled
        self.assembly_engine: Optional['TokenAssemblyEngine'] = None
        if enable_assembly:
            from .assembly import TokenAssemblyEngine
            self.assembly_engine = TokenAssemblyEngine(
                correlation_id=correlation_id,
                enable_caching=True,
                strict_mode=False
            )

        self._reset_state()

    def _reset_state(self) -> None:
        """Reset tokenizer state for new processing."""
        self.state = TokenizerState.TEXT_CONTENT
        self.current_token_start = TokenPosition(1, 1, 0)
        self.current_position = TokenPosition(1, 1, 0)
        self.token_buffer = ""
        self.tokens: List[Token] = []
        self.quote_char: Optional[str] = None
        self.fast_path_enabled = self.enable_fast_path
        self.error_recovery_count = 0
        # Number of chars to sample for fast-path detection
        self.fast_path_sample_size = 1000
        self.fast_path_threshold = 0.7     # Confidence threshold for fast-path

    def _detect_well_formed_xml(self, content: str) -> bool:
        """Detect if the XML is truly well-formed for fast-path optimization.
        
        Fast-path should ONLY be used for perfectly well-formed XML.
        Any malformation indicators should immediately disqualify fast-path.
        
        Args:
            content: XML content to analyze
            
        Returns:
            True ONLY if content is well-formed and safe for fast-path processing
        """
        if not content or not self.enable_fast_path:
            return False

        # Analyze the entire content for strict well-formedness
        content_stripped = content.strip()
        
        # Immediate disqualifiers - any of these means use robust processing
        malformation_indicators = [
            "<<", ">>", "<>", "><",  # Malformed tag sequences
            "< /",  # Spaces in closing tags
        ]
        
        for indicator in malformation_indicators:
            if indicator in content:
                logger.debug(
                    "Fast-path disabled due to malformation indicator",
                    extra={
                        "component": "xml_tokenizer",
                        "correlation_id": self.correlation_id,
                        "indicator": indicator
                    }
                )
                return False
        
        # Check for unquoted attributes - immediate disqualifier
        import re
        unquoted_attr_pattern = r'=\s*[a-zA-Z_][a-zA-Z0-9_]*(?=\s|>)'
        if re.search(unquoted_attr_pattern, content):
            logger.debug(
                "Fast-path disabled due to unquoted attributes",
                extra={
                    "component": "xml_tokenizer", 
                    "correlation_id": self.correlation_id
                }
            )
            return False
        
        # Check for proper tag balancing in entire content
        open_tags = content.count("<")
        close_tags = content.count(">")
        if open_tags != close_tags:
            logger.debug(
                "Fast-path disabled due to unbalanced tags",
                extra={
                    "component": "xml_tokenizer",
                    "correlation_id": self.correlation_id,
                    "open_tags": open_tags,
                    "close_tags": close_tags
                }
            )
            return False

        # Check for proper quote pairing in attributes
        single_quotes = content.count("'")
        double_quotes = content.count('"')
        if single_quotes % 2 != 0 or double_quotes % 2 != 0:
            logger.debug(
                "Fast-path disabled due to unbalanced quotes",
                extra={
                    "component": "xml_tokenizer",
                    "correlation_id": self.correlation_id,
                    "single_quotes": single_quotes,
                    "double_quotes": double_quotes
                }
            )
            return False

        # Basic structure requirements for well-formed XML
        if not (content_stripped.startswith("<") and content_stripped.endswith(">")):
            return False
            
        # If we get here, the XML appears truly well-formed
        logger.debug(
            "Fast-path enabled for well-formed XML",
            extra={
                "component": "xml_tokenizer",
                "correlation_id": self.correlation_id,
                "content_length": len(content)
            }
        )
        
        return True

    def _fast_path_tokenize(self, content: str) -> List[Token]:
        """Optimized tokenization for well-formed XML content.
        
        Args:
            content: Well-formed XML content
            
        Returns:
            List of tokens from optimized processing
        """
        tokens = []
        i = 0
        line = 1
        column = 1

        while i < len(content):
            start_pos = TokenPosition(line, column, i)

            if content[i] == "<":
                # Find end of tag
                tag_end = content.find(">", i)
                if tag_end == -1:
                    # Incomplete tag - fall back to robust processing
                    return []

                tag_content = content[i:tag_end + 1]

                # Parse tag efficiently
                if tag_content.startswith("<!--"):
                    # Comment
                    comment_end = content.find("-->", i)
                    if comment_end != -1:
                        comment_content = content[i + 4:comment_end]
                        tokens.append(Token(
                            type=TokenType.COMMENT,
                            value=comment_content,
                            position=start_pos,
                            confidence=1.0
                        ))
                        i = comment_end + 3
                        column += comment_end + 3 - start_pos.offset
                        continue

                elif tag_content.startswith("<![CDATA["):
                    # CDATA
                    cdata_end = content.find("]]>", i)
                    if cdata_end != -1:
                        cdata_content = content[i + 9:cdata_end]
                        tokens.append(Token(
                            type=TokenType.CDATA,
                            value=cdata_content,
                            position=start_pos,
                            confidence=1.0
                        ))
                        i = cdata_end + 3
                        column += cdata_end + 3 - start_pos.offset
                        continue

                elif tag_content.startswith("<?"):
                    # Processing instruction
                    pi_content = tag_content[2:-2]
                    tokens.append(Token(
                        type=TokenType.PROCESSING_INSTRUCTION,
                        value=pi_content,
                        position=start_pos,
                        confidence=1.0
                    ))

                else:
                    # Regular tag - simple tokenization for well-formed XML only
                    # Emit TAG_START
                    tokens.append(Token(
                        type=TokenType.TAG_START,
                        value="<",
                        position=start_pos,
                        confidence=1.0
                    ))

                    # Parse tag name and attributes simply (well-formed XML assumed)
                    tag_inner = tag_content[1:-1].strip()
                    if tag_inner.startswith("/"):
                        # Closing tag
                        tag_name = tag_inner[1:].strip()
                        tokens.append(Token(
                            type=TokenType.TAG_NAME,
                            value=tag_name,
                            position=TokenPosition(line, column + 2, i + 2),
                            confidence=1.0
                        ))
                    else:
                        # Opening tag - parse name and attributes
                        parts = tag_inner.split()
                        if parts:
                            tag_name = parts[0]
                            tokens.append(Token(
                                type=TokenType.TAG_NAME,
                                value=tag_name,
                                position=TokenPosition(line, column + 1, i + 1),
                                confidence=1.0
                            ))

                            # Simple attribute parsing for well-formed XML only
                            attr_text = " ".join(parts[1:])
                            if attr_text:
                                import re
                                # Only handle properly quoted attributes (well-formed XML)
                                attr_pattern = r'(\w+)=(["\'])([^"\']*)\2'
                                for match in re.finditer(attr_pattern, attr_text):
                                    attr_name, quote, attr_value = match.groups()
                                    tokens.append(Token(
                                        type=TokenType.ATTR_NAME,
                                        value=attr_name,
                                        position=TokenPosition(line, column, i),
                                        confidence=1.0
                                    ))
                                    tokens.append(Token(
                                        type=TokenType.ATTR_VALUE,
                                        value=attr_value,
                                        position=TokenPosition(line, column, i),
                                        confidence=1.0
                                    ))

                    # Emit TAG_END
                    tokens.append(Token(
                        type=TokenType.TAG_END,
                        value=">",
                        position=TokenPosition(
                            line, column + len(tag_content) - 1, i + len(tag_content) - 1
                        ),
                        confidence=1.0
                    ))

                i = tag_end + 1
                column += len(tag_content)

            else:
                # Text content
                text_start = i
                text_pos = start_pos

                # Find next tag or end of content
                next_tag = content.find("<", i)
                if next_tag == -1:
                    text_content = content[i:]
                    i = len(content)
                else:
                    text_content = content[i:next_tag]
                    i = next_tag

                # Update position for newlines
                newlines = text_content.count("\n")
                if newlines > 0:
                    line += newlines
                    column = len(text_content.split("\n")[-1]) + 1
                else:
                    column += len(text_content)

                # Only emit non-empty text tokens
                if text_content.strip():
                    tokens.append(Token(
                        type=TokenType.TEXT,
                        value=text_content,
                        position=text_pos,
                        confidence=1.0
                    ))

        return tokens

    def tokenize(self, char_stream: CharacterStreamResult) -> TokenizationResult:
        """Tokenize a character stream into XML tokens.
        
        Args:
            char_stream: Result from character processing layer
            
        Returns:
            TokenizationResult with tokens and comprehensive metadata
        """
        import time
        start_time = time.time()

        logger.debug(
            "Starting tokenization",
            extra={
                "component": "xml_tokenizer",
                "correlation_id": self.correlation_id,
                "char_count": len(char_stream.text) if char_stream.text else 0,
                "encoding": char_stream.encoding.encoding
            }
        )

        self._reset_state()

        try:
            # Process character stream content
            if char_stream.text:
                # Try fast-path optimization first
                if (
                    self.fast_path_enabled
                    and self._detect_well_formed_xml(char_stream.text)
                ):
                    logger.debug(
                        "Using fast-path tokenization",
                        extra={
                            "component": "xml_tokenizer",
                            "correlation_id": self.correlation_id,
                            "content_length": len(char_stream.text)
                        }
                    )

                    fast_tokens = self._fast_path_tokenize(char_stream.text)
                    if fast_tokens:  # Fast-path succeeded
                        self.tokens = fast_tokens
                    else:  # Fall back to robust processing
                        logger.debug(
                            "Fast-path failed, falling back to robust processing",
                            extra={
                                "component": "xml_tokenizer",
                                "correlation_id": self.correlation_id
                            }
                        )
                        self.fast_path_enabled = False
                        for char in char_stream.text:
                            self._process_character(char)
                        self._finalize_current_token()
                else:
                    # Use robust character-by-character processing
                    logger.debug(
                        "Using robust tokenization",
                        extra={
                            "component": "xml_tokenizer",
                            "correlation_id": self.correlation_id,
                            "content_length": len(char_stream.text)
                        }
                    )
                    for char in char_stream.text:
                        self._process_character(char)
                    self._finalize_current_token()

            # Apply token assembly and repair if enabled
            final_tokens = self.tokens
            assembly_diagnostics = []
            
            if self.assembly_engine and self.enable_assembly:
                try:
                    assembly_result = self.assembly_engine.assemble_and_repair_tokens(self.tokens)
                    final_tokens = assembly_result.tokens
                    
                    # Add assembly diagnostics
                    assembly_diagnostics.extend(assembly_result.diagnostics)
                    if assembly_result.repairs_applied > 0:
                        assembly_diagnostics.append(
                            f"Assembly applied {assembly_result.repairs_applied} repairs "
                            f"({assembly_result.repair_rate:.1%} token repair rate)"
                        )
                        
                    logger.debug(
                        "Token assembly completed",
                        extra={
                            "component": "xml_tokenizer",
                            "correlation_id": self.correlation_id,
                            "repairs_applied": assembly_result.repairs_applied,
                            "repair_rate": assembly_result.repair_rate,
                            "assembly_time_ms": assembly_result.processing_time_ms
                        }
                    )
                except Exception as e:
                    logger.warning(
                        "Token assembly failed, using unassembled tokens",
                        extra={
                            "component": "xml_tokenizer",
                            "correlation_id": self.correlation_id,
                            "error": str(e)
                        },
                        exc_info=True
                    )
                    # Continue with original tokens on assembly failure
                    final_tokens = self.tokens
                    assembly_diagnostics.append(f"Assembly failed: {e}")

            # Calculate processing time
            processing_time = time.time() - start_time

            # Create result
            result = TokenizationResult(
                tokens=final_tokens,
                success=True,
                processing_time=processing_time,
                character_count=len(char_stream.text) if char_stream.text else 0,
                diagnostics=assembly_diagnostics
            )

            logger.debug(
                "Tokenization completed",
                extra={
                    "component": "xml_tokenizer",
                    "correlation_id": self.correlation_id,
                    "token_count": result.token_count,
                    "confidence": result.confidence,
                    "processing_time": processing_time
                }
            )

            return result

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(
                "Tokenization failed",
                extra={
                    "component": "xml_tokenizer",
                    "correlation_id": self.correlation_id,
                    "error": str(e)
                },
                exc_info=True
            )

            # Return partial result even on failure (never-fail philosophy)
            # Don't attempt assembly on failure cases
            return TokenizationResult(
                tokens=self.tokens,
                success=False,
                processing_time=processing_time,
                character_count=len(char_stream.text) if char_stream.text else 0,
                diagnostics=[f"Tokenization error: {e!s}"]
            )

    def _process_character(self, char: str) -> None:
        """Process a single character through the state machine.
        
        Args:
            char: Single character to process
        """
        # State machine processing (before updating position)
        if self.state == TokenizerState.TEXT_CONTENT:
            self._process_text_content(char)
        elif self.state == TokenizerState.TAG_OPENING:
            self._process_tag_opening(char)
        elif self.state == TokenizerState.TAG_CLOSING:
            self._process_tag_closing(char)
        elif self.state == TokenizerState.TAG_NAME:
            self._process_tag_name(char)
        elif self.state == TokenizerState.ATTR_NAME:
            self._process_attr_name(char)
        elif self.state == TokenizerState.ATTR_VALUE_START:
            self._process_attr_value_start(char)
        elif self.state == TokenizerState.ATTR_VALUE_QUOTED:
            self._process_attr_value_quoted(char)
        elif self.state == TokenizerState.ATTR_VALUE_UNQUOTED:
            self._process_attr_value_unquoted(char)
        elif self.state == TokenizerState.COMMENT_START:
            self._process_comment_start(char)
        elif self.state == TokenizerState.COMMENT_CONTENT:
            self._process_comment_content(char)
        elif self.state == TokenizerState.CDATA_START:
            self._process_cdata_start(char)
        elif self.state == TokenizerState.CDATA_CONTENT:
            self._process_cdata_content(char)
        elif self.state == TokenizerState.PI_START:
            self._process_pi_start(char)
        elif self.state == TokenizerState.PI_CONTENT:
            self._process_pi_content(char)
        elif self.state == TokenizerState.DOT_PREFIX:
            self._process_dot_prefix(char)
        elif self.state == TokenizerState.DOT_PREFIX_CLOSING:
            self._process_dot_prefix_closing(char)
        else:
            # Error recovery - should never reach here with proper state machine
            self._enter_error_recovery(char, f"Unknown state: {self.state}")

        # Update position tracking after processing
        self._update_position(char)

    def _update_position(self, char: str) -> None:
        """Update position tracking based on character.
        
        Args:
            char: Character being processed
        """
        if char == "\n":
            self.current_position = TokenPosition(
                self.current_position.line + 1,
                1,
                self.current_position.offset + 1
            )
        elif char == "\t":
            # Tab advances to next tab stop (typically 8 characters)
            tab_stop = 8
            new_column = (
                ((self.current_position.column - 1) // tab_stop + 1) * tab_stop + 1
            )
            self.current_position = TokenPosition(
                self.current_position.line,
                new_column,
                self.current_position.offset + 1
            )
        else:
            self.current_position = TokenPosition(
                self.current_position.line,
                self.current_position.column + 1,
                self.current_position.offset + 1
            )

    def _process_text_content(self, char: str) -> None:
        """Process character in text content state."""
        if char == "<":
            # Look ahead to determine if this is a real tag or text that needs escaping
            # For now, we'll use a simple heuristic: assume it's a tag start
            # The recovery engine will handle cases where it's not
            
            # Finalize any accumulated text
            if self.token_buffer:
                self._emit_token(TokenType.TEXT, self.token_buffer)

            # Start new tag
            self._start_new_token()
            self.token_buffer = char
            self.state = TokenizerState.TAG_OPENING
        elif char == ">":
            # Convert standalone > to &gt; in text content
            if not self.token_buffer:
                self._start_new_token()
            self.token_buffer += "&gt;"
        else:
            # Accumulate text content
            if not self.token_buffer:
                self._start_new_token()
            self.token_buffer += char

    def _process_tag_opening(self, char: str) -> None:
        """Process character in tag opening state."""
        if char == "/":
            # Closing tag
            self.token_buffer += char
            self.state = TokenizerState.TAG_CLOSING
        elif char == "!":
            # Comment or CDATA
            self.token_buffer += char
            self.state = TokenizerState.COMMENT_START
        elif char == "?":
            # Processing instruction
            self.token_buffer += char
            self.state = TokenizerState.PI_START
        elif char == "<":
            # Double < - treat the first < as text content that needs entity encoding
            # Go back to text content state and emit &lt; then start new tag
            self._start_new_token()
            self.token_buffer = "&lt;"
            self._emit_token(TokenType.TEXT, self.token_buffer)
            # Start processing the second < as a new tag
            self._start_new_token()
            self.token_buffer = char
            # Stay in TAG_OPENING state to process this new <
        elif char.isspace():
            # Invalid - tag name cannot start with whitespace
            self._enter_error_recovery(char, "Tag name cannot start with whitespace")
        elif self._is_name_start_char(char):
            # Tag name start
            self._emit_token(TokenType.TAG_START, "<")
            self._start_new_token()
            self.token_buffer = char
            self.state = TokenizerState.TAG_NAME
        elif char == ".":
            # Potential dot-prefixed tag - switch to a special state to handle it
            self.state = TokenizerState.DOT_PREFIX
        else:
            # Invalid tag start
            self._enter_error_recovery(char, f"Invalid tag start character: {char}")

    def _process_tag_closing(self, char: str) -> None:
        """Process character in tag closing state (after </)."""
        if self._is_name_start_char(char):
            # Start of closing tag name - emit TAG_START with "/" to indicate closing tag
            self._emit_token(TokenType.TAG_START, "</")
            self._start_new_token()
            self.token_buffer = char
            self.state = TokenizerState.TAG_NAME
        elif char == ".":
            # Dot-prefixed closing tag (e.g., </.doc>) - switch to special state
            self.state = TokenizerState.DOT_PREFIX_CLOSING
        elif char.isspace():
            # Whitespace after </ - invalid but we can recover
            self._enter_error_recovery(char, "Whitespace after </")
        else:
            # Invalid character after </
            self._enter_error_recovery(char, f"Invalid character after </: {char}")

    def _process_dot_prefix(self, char: str) -> None:
        """Process character after seeing a dot in tag opening position (e.g., <.doc>)."""
        if self._is_name_start_char(char):
            # Valid tag name character after dot - emit TAG_START and start tag name
            # Skip the dot and treat this as the start of a valid tag name
            self._emit_token(TokenType.TAG_START, "<")
            self._start_new_token()
            self.token_buffer = char
            self.state = TokenizerState.TAG_NAME
        else:
            # Invalid character after dot - enter error recovery
            self._enter_error_recovery(char, f"Invalid character after dot in tag: {char}")

    def _process_dot_prefix_closing(self, char: str) -> None:
        """Process character after seeing a dot in closing tag position (e.g., </.doc>)."""
        if self._is_name_start_char(char):
            # Valid tag name character after dot - emit TAG_START and start tag name
            # Skip the dot and treat this as the start of a valid closing tag name
            self._emit_token(TokenType.TAG_START, "</")
            self._start_new_token()
            self.token_buffer = char
            self.state = TokenizerState.TAG_NAME
        else:
            # Invalid character after dot - enter error recovery
            self._enter_error_recovery(char, f"Invalid character after dot in closing tag: {char}")

    def _process_tag_name(self, char: str) -> None:
        """Process character in tag name state."""
        if self._is_name_char(char):
            self.token_buffer += char
        elif char.isspace():
            # End of tag name, expect attributes or tag end
            if self.token_buffer:  # Only emit if we have content
                self._emit_token(TokenType.TAG_NAME, self.token_buffer)
            # Skip whitespace and look for attributes
            self._start_new_token()
            self.state = TokenizerState.ATTR_NAME
        elif char == ">":
            # End of tag
            if self.token_buffer:
                self._emit_token(TokenType.TAG_NAME, self.token_buffer)
            self._emit_token(TokenType.TAG_END, ">")
            self.state = TokenizerState.TEXT_CONTENT
        elif char == "/":
            # Self-closing tag
            if self.token_buffer:
                self._emit_token(TokenType.TAG_NAME, self.token_buffer)
            # Start collecting the /
            self._start_new_token()
            self.token_buffer = char
        elif char == "=":
            # This tag name is actually an attribute name
            if self.token_buffer:
                self._emit_token(TokenType.ATTR_NAME, self.token_buffer)
            self.state = TokenizerState.ATTR_VALUE_START
        else:
            # Invalid character in tag name - try to complete the tag if we have a valid name
            if self.token_buffer and self._is_valid_tag_name(self.token_buffer):
                # Complete the incomplete tag by auto-closing it
                self._emit_token(TokenType.TAG_NAME, self.token_buffer)
                self._emit_token(TokenType.TAG_END, ">")
                self.state = TokenizerState.TEXT_CONTENT
                # Process the current character in text content state
                self._process_text_content(char)
            else:
                self._enter_error_recovery(char, f"Invalid character in tag name: {char}")

    def _process_attr_name(self, char: str) -> None:
        """Process character in attribute name state."""
        if self._is_name_char(char):
            self.token_buffer += char
        elif char == "=":
            self._handle_attr_name_equals()
        elif char.isspace():
            self._handle_attr_name_whitespace()
        elif char == ">":
            self._handle_attr_name_tag_end()
        elif char == "/":
            self._handle_attr_name_self_closing()
        elif not self.token_buffer and self._is_name_start_char(char):
            # Start of new attribute name if we don't have a buffer yet
            self.token_buffer = char
        else:
            self._enter_error_recovery(
                char, f"Invalid character in attribute name: {char}"
            )

    def _handle_attr_name_equals(self) -> None:
        """Handle equals sign in attribute name state."""
        # End of attribute name
        if self.token_buffer:
            self._emit_token(TokenType.ATTR_NAME, self.token_buffer)
        self.state = TokenizerState.ATTR_VALUE_START

    def _handle_attr_name_whitespace(self) -> None:
        """Handle whitespace in attribute name state."""
        # Whitespace - might be end of attribute name or just skippable
        if self.token_buffer:
            self._emit_token(TokenType.ATTR_NAME, self.token_buffer)
            self._start_new_token()
        # Stay in ATTR_NAME state to handle what comes next

    def _handle_attr_name_tag_end(self) -> None:
        """Handle tag end in attribute name state."""
        # End of tag without attribute value
        if self.token_buffer:
            self._emit_token(TokenType.ATTR_NAME, self.token_buffer)
        self._emit_token(TokenType.TAG_END, ">")
        self.state = TokenizerState.TEXT_CONTENT

    def _handle_attr_name_self_closing(self) -> None:
        """Handle self-closing tag marker in attribute name state."""
        # Self-closing tag
        if self.token_buffer:
            self._emit_token(TokenType.ATTR_NAME, self.token_buffer)
        self._start_new_token()
        self.token_buffer = "/"

    def _process_attr_value_start(self, char: str) -> None:
        """Process character at start of attribute value."""
        if char in ('"', "'"):
            # Quoted attribute value
            self.quote_char = char
            self._start_new_token()
            self.state = TokenizerState.ATTR_VALUE_QUOTED
        elif char.isspace():
            # Skip whitespace before attribute value
            pass
        elif char in (">", "/"):
            # Attribute without value - emit empty value
            self._emit_token(TokenType.ATTR_VALUE, "")
            # Handle the current character in appropriate state
            if char == ">":
                self._emit_token(TokenType.TAG_END, ">")
                self.state = TokenizerState.TEXT_CONTENT
        else:
            # Unquoted attribute value
            self._start_new_token()
            self.token_buffer = char
            self.state = TokenizerState.ATTR_VALUE_UNQUOTED

    def _process_attr_value_quoted(self, char: str) -> None:
        """Process character in quoted attribute value."""
        if char == self.quote_char:
            # End of quoted value
            self._emit_token(TokenType.ATTR_VALUE, self.token_buffer)
            self.quote_char = None
            # Look for more attributes or tag end
            self._start_new_token()
            self.state = TokenizerState.ATTR_NAME
        else:
            self.token_buffer += char

    def _process_attr_value_unquoted(self, char: str) -> None:
        """Process character in unquoted attribute value."""
        if char.isspace() or char in (">", "/"):
            # End of unquoted value
            self._emit_token(TokenType.ATTR_VALUE, self.token_buffer)
            if char == ">":
                self._emit_token(TokenType.TAG_END, ">")
                self.state = TokenizerState.TEXT_CONTENT
            elif char == "/":
                # Self-closing tag
                self._start_new_token()
                self.token_buffer = char
            else:
                # More attributes might follow
                self._start_new_token()
                self.state = TokenizerState.ATTR_NAME
        else:
            self.token_buffer += char

    def _process_comment_start(self, char: str) -> None:
        """Process character in comment start state."""
        self.token_buffer += char
        if self.token_buffer.endswith("<!--"):
            # Valid comment start
            self._start_new_token()
            self.state = TokenizerState.COMMENT_CONTENT
        elif self.token_buffer.endswith("<![CDATA["):
            # CDATA section start
            self._start_new_token()
            self.state = TokenizerState.CDATA_CONTENT
        elif len(self.token_buffer) > XML_CDATA_MAX_LENGTH:  # Maximum for <![CDATA[
            # Not a valid comment or CDATA
            self._enter_error_recovery(char, "Invalid comment or CDATA start")

    def _process_comment_content(self, char: str) -> None:
        """Process character in comment content."""
        self.token_buffer += char
        if self.token_buffer.endswith("-->"):
            # End of comment - check for and repair invalid -- sequences
            comment_content = self.token_buffer[:-3]  # Remove -->
            
            # Check if comment contains invalid -- sequences (not at the end)
            repaired_content = comment_content
            repairs = []
            
            # Find and replace -- sequences that are not part of the closing -->
            # Look for -- that's not followed by > (which would be the valid closing)
            if "--" in comment_content:
                # Replace -- with - (remove one dash)
                repaired_content = comment_content.replace("--", "-")
                
                if repaired_content != comment_content:
                    repair = TokenRepair(
                        repair_type="malformed_comment",
                        description="Comment contained invalid --, replaced with -",
                        original_content=comment_content,
                        repaired_content=repaired_content,
                        confidence_impact=-0.1
                    )
                    repairs.append(repair)
            
            # Emit the comment token with any repairs
            token = Token(
                type=TokenType.COMMENT,
                value=repaired_content,
                position=self.current_token_start,
                confidence=0.9 if repairs else 1.0,
                repairs=repairs
            )
            self.tokens.append(token)
            self._start_new_token()  # Reset token buffer
            self.state = TokenizerState.TEXT_CONTENT

    def _process_cdata_start(self, char: str) -> None:
        """Process character in CDATA start state."""
        self.token_buffer += char
        if self.token_buffer.endswith("<![CDATA["):
            # Valid CDATA start
            self._start_new_token()
            self.state = TokenizerState.CDATA_CONTENT
        elif len(self.token_buffer) > XML_CDATA_MAX_LENGTH:  # Maximum for <![CDATA[
            # Not a valid CDATA start
            self._enter_error_recovery(char, "Invalid CDATA start")

    def _process_cdata_content(self, char: str) -> None:
        """Process character in CDATA content."""
        self.token_buffer += char
        if self.token_buffer.endswith("]]>"):
            # End of CDATA
            cdata_content = self.token_buffer[:-3]  # Remove ]]>
            self._emit_token(TokenType.CDATA, cdata_content)
            self.state = TokenizerState.TEXT_CONTENT

    def _process_pi_start(self, char: str) -> None:
        """Process character in processing instruction start."""
        self.token_buffer += char
        # Transition to PI_CONTENT state after collecting initial characters
        if len(self.token_buffer) >= PI_MIN_LENGTH:  # Have <? at minimum
            self.state = TokenizerState.PI_CONTENT

    def _process_pi_content(self, char: str) -> None:
        """Process character in processing instruction content."""
        if char == '>' and not self.token_buffer.endswith('?'):
            # Malformed PI ending with > instead of ?> - emit as text immediately
            self.token_buffer += char
            
            # Create repair info
            repair = TokenRepair(
                repair_type="malformed_processing_instruction",
                description="Processing instruction malformed, treated as text",
                original_content=self.token_buffer,
                repaired_content=self.token_buffer,
                confidence_impact=-0.3
            )
            
            # Emit as text token
            token = Token(
                type=TokenType.TEXT,
                value=self.token_buffer,
                position=self.current_token_start,
                confidence=0.7,
                repairs=[repair]
            )
            self.tokens.append(token)
            
            # Reset and continue normal processing
            self._start_new_token()
            self.state = TokenizerState.TEXT_CONTENT
            return
        elif char == '<' and self.token_buffer.endswith('?'):
            # Malformed PI with ? but missing > - repair by adding missing >
            # Create repair info
            repair = TokenRepair(
                repair_type="malformed_processing_instruction",
                description="Processing instruction missing closing >, added missing >",
                original_content=self.token_buffer,
                repaired_content=self.token_buffer + ">",
                confidence_impact=-0.1
            )
            
            # Extract PI content (remove initial <? and trailing ?)
            pi_content = self.token_buffer[2:-1]  # Remove <? and ?
            
            # Emit the repaired PI
            token = Token(
                type=TokenType.PROCESSING_INSTRUCTION,
                value=pi_content,
                position=self.current_token_start,
                confidence=0.9,
                repairs=[repair]
            )
            self.tokens.append(token)
            
            # Now process the < as start of a new token (likely closing tag)
            self._start_new_token()
            self.token_buffer = char  # Start new token with the <
            self.state = TokenizerState.TAG_OPENING
            return
        
        self.token_buffer += char
        if self.token_buffer.endswith("?>"):
            # End of properly formed processing instruction
            pi_content = self.token_buffer[2:-2]  # Remove <? and ?>
            self._emit_token(TokenType.PROCESSING_INSTRUCTION, pi_content)
            self.state = TokenizerState.TEXT_CONTENT

    def _repair_text_entities(self, text: str) -> str:
        """Repair malformed entity references in text content."""
        import re
        
        # Pattern to match incomplete entities (& followed by entity name but no semicolon)
        incomplete_entity_pattern = r'&(amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+)(?![;a-zA-Z0-9])'
        
        def repair_entity(match):
            entity_name = match.group(1)
            return f'&{entity_name};'
        
        # Replace incomplete entities with complete ones
        return re.sub(incomplete_entity_pattern, repair_entity, text)

    def _skip_whitespace_and_transition(self) -> None:
        """Skip whitespace and transition to appropriate state."""
        # Reset to safe state - we'll determine the right state in the next character
        # This handles transitions after tag names or attribute names
        self.state = TokenizerState.TAG_NAME  # Will handle next char appropriately
        self._start_new_token()

    def _enter_error_recovery(self, char: str, reason: str) -> None:
        """Enter error recovery state.
        
        Args:
            char: Character that caused the error
            reason: Description of the error
        """
        self.error_recovery_count += 1

        logger.warning(
            "Entering error recovery",
            extra={
                "component": "xml_tokenizer",
                "correlation_id": self.correlation_id,
                "character": repr(char),
                "reason": reason,
                "position": (
                    f"{self.current_position.line}:{self.current_position.column}"
                )
            }
        )

        # Use recovery engine if available
        if self.recovery_engine and self.enable_recovery:
            try:
                # Import here to avoid circular dependency
                from .recovery import RecoveryContext

                # Create recovery context
                context = RecoveryContext(
                    error_type=reason.lower().replace(" ", "_"),
                    error_position=self.current_position,
                    surrounding_content=self._get_surrounding_content(),
                    tokenizer_state=self.state,
                    recent_tokens=self.tokens[-5:] if len(self.tokens) >= 5 else self.tokens,
                    malformation_severity=self._assess_malformation_severity(reason, char)
                )

                # Perform recovery
                recovery_action = self.recovery_engine.recover_from_error(context, char)

                # Add recovered tokens
                self.tokens.extend(recovery_action.tokens)

                # Update diagnostic info
                logger.info(
                    "Error recovery completed",
                    extra={
                        "component": "xml_tokenizer",
                        "correlation_id": self.correlation_id,
                        "strategy": recovery_action.strategy.name,
                        "success": recovery_action.success,
                        "confidence": recovery_action.confidence,
                        "tokens_generated": len(recovery_action.tokens)
                    }
                )

            except Exception as e:
                logger.error(
                    "Recovery engine failed",
                    extra={
                        "component": "xml_tokenizer",
                        "correlation_id": self.correlation_id,
                        "error": str(e)
                    },
                    exc_info=True
                )
                # Fallback to original error recovery
                self._fallback_error_recovery(char, reason)
        else:
            # Use fallback error recovery
            self._fallback_error_recovery(char, reason)

        # Reset to safe state
        self.state = TokenizerState.TEXT_CONTENT
        self._start_new_token()

    def _fallback_error_recovery(self, char: str, reason: str) -> None:
        """Fallback error recovery when recovery engine is not available.
        
        Args:
            char: Character that caused the error
            reason: Description of the error
        """
        # Create error token (original logic)
        repair = TokenRepair(
            repair_type="error_recovery",
            description=reason,
            original_content=char,
            repaired_content="",
            confidence_impact=-0.1
        )

        error_token = Token(
            type=TokenType.ERROR,
            value=char,
            position=self.current_position,
            confidence=0.1,
            repairs=[repair]
        )

        self.tokens.append(error_token)

    def _get_surrounding_content(self, window_size: int = 20) -> str:
        """Get surrounding content for error recovery context.
        
        Args:
            window_size: Number of characters to include around error
            
        Returns:
            String of surrounding content
        """
        # This is a simplified implementation
        # In practice, we'd need to track the original content
        recent_content = ""
        for token in self.tokens[-3:]:  # Last 3 tokens
            recent_content += token.value
        return recent_content[-window_size:] if len(recent_content) > window_size else recent_content

    def _assess_malformation_severity(self, reason: str, char: str) -> float:
        """Assess the severity of the malformation.
        
        Args:
            reason: Error reason
            char: Problematic character
            
        Returns:
            Severity score from 0.0 to 1.0
        """
        severity = 0.3  # Base severity

        # Increase severity based on error type
        if "invalid_character" in reason.lower():
            severity += 0.2
        if "tag" in reason.lower():
            severity += 0.3
        if "attribute" in reason.lower():
            severity += 0.2

        # Increase severity based on character
        if char in "<>":
            severity += 0.2
        elif not char.isalnum() and char not in " \t\n\r":
            severity += 0.1

        # Increase severity based on error recovery count
        if self.error_recovery_count > 5:
            severity += 0.2
        elif self.error_recovery_count > 10:
            severity += 0.3

        return min(1.0, severity)

    def _start_new_token(self) -> None:
        """Start tracking a new token."""
        # Position should be where the token actually starts,
        # which is the current position
        # We need to track the position before processing the character
        self.current_token_start = TokenPosition(
            self.current_position.line,
            self.current_position.column,
            self.current_position.offset
        )
        self.token_buffer = ""

    def _emit_token(self, token_type: TokenType, value: str) -> None:
        """Emit a token to the token list.
        
        Args:
            token_type: Type of token to emit
            value: Token value/content
        """
        token = Token(
            type=token_type,
            value=value,
            position=self.current_token_start,
            confidence=1.0,  # High confidence for successfully parsed tokens
            raw_content=value
        )

        self.tokens.append(token)
        self._start_new_token()

    def _finalize_current_token(self) -> None:
        """Finalize any remaining token in the buffer."""
        if self.token_buffer:
            if self.state == TokenizerState.TEXT_CONTENT:
                self._emit_token(TokenType.TEXT, self.token_buffer)
            elif self.state == TokenizerState.COMMENT_CONTENT:
                # Unclosed comment
                repair = TokenRepair(
                    repair_type="unclosed_comment",
                    description="Comment not properly closed",
                    original_content=self.token_buffer,
                    repaired_content=self.token_buffer,
                    confidence_impact=-0.2
                )

                token = Token(
                    type=TokenType.COMMENT,
                    value=self.token_buffer,
                    position=self.current_token_start,
                    confidence=0.8,
                    repairs=[repair]
                )
                self.tokens.append(token)
            else:
                # Other incomplete tokens - emit as text with repair
                repair = TokenRepair(
                    repair_type="incomplete_token",
                    description=f"Incomplete token in state {self.state.name}",
                    original_content=self.token_buffer,
                    repaired_content=self.token_buffer,
                    confidence_impact=-0.3
                )

                token = Token(
                    type=TokenType.TEXT,
                    value=self.token_buffer,
                    position=self.current_token_start,
                    confidence=0.7,
                    repairs=[repair]
                )
                self.tokens.append(token)

    def _is_name_start_char(self, char: str) -> bool:
        """Check if character can start an XML name."""
        return (char.isalpha() or
                char == "_" or
                char == ":" or
                ord(char) >= UNICODE_START_OFFSET)  # Unicode characters

    def _is_name_char(self, char: str) -> bool:
        """Check if character can be part of an XML name."""
        return (self._is_name_start_char(char) or
                char.isdigit() or
                char in ".-")

    def _is_valid_tag_name(self, name: str) -> bool:
        """Check if string is a valid XML tag name."""
        if not name:
            return False
        # Must start with valid name start character
        if not self._is_name_start_char(name[0]):
            return False
        # All characters must be valid name characters
        return all(self._is_name_char(char) for char in name)

    def get_recovery_statistics(self) -> Optional[Dict[str, Any]]:
        """Get comprehensive recovery statistics from the recovery engine.
        
        Returns:
            Recovery statistics dictionary or None if recovery is disabled
        """
        if not self.recovery_engine:
            return None
        return self.recovery_engine.get_recovery_statistics()

    def get_recovery_history_report(self) -> Optional[Dict[str, Any]]:
        """Get detailed recovery history report from the recovery engine.
        
        Returns:
            Recovery history report or None if recovery is disabled
        """
        if not self.recovery_engine:
            return None
        return self.recovery_engine.get_recovery_history_report()
    
    def get_assembly_statistics(self) -> Optional[Dict[str, Any]]:
        """Get comprehensive assembly statistics from the assembly engine.
        
        Returns:
            Assembly statistics dictionary or None if assembly is disabled
        """
        if not self.assembly_engine:
            return None
        return self.assembly_engine.get_repair_statistics()
