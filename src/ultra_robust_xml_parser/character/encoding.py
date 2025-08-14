"""Multi-stage encoding detection system with never-fail guarantee.

This module implements a comprehensive encoding detection system that uses multiple
detection strategies in sequence: BOM detection, XML declaration parsing, statistical
analysis, UTF-8 validation, and fallback mechanisms.
"""

import codecs
import re
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Dict, List, Optional

# UTF-8 byte constants
ASCII_MAX = 0x80
UTF8_CONTINUATION_MIN = 0x80
UTF8_CONTINUATION_MAX = 0xC0
UTF8_2BYTE_MIN = 0xC0
UTF8_2BYTE_MAX = 0xE0
UTF8_3BYTE_MIN = 0xE0
UTF8_3BYTE_MAX = 0xF0
UTF8_4BYTE_MIN = 0xF0
UTF8_4BYTE_MAX = 0xF8

# Codepoint thresholds for overlong sequence detection
OVERLONG_2BYTE_THRESHOLD = 0x80
OVERLONG_3BYTE_THRESHOLD = 0x800
OVERLONG_4BYTE_THRESHOLD = 0x10000

# Confidence thresholds
CONFIDENCE_HIGH_UTF8_STATISTICAL = 0.8
CONFIDENCE_MEDIUM_LATIN1 = 0.6
CONFIDENCE_BOM_THRESHOLD = 0.9
CONFIDENCE_XML_THRESHOLD = 0.8
CONFIDENCE_STATISTICAL_THRESHOLD = 0.7
CONFIDENCE_UTF8_VALIDATION_THRESHOLD = 0.6

# ASCII ratio thresholds for Latin-1 detection
LATIN1_HIGH_ASCII_RATIO = 0.9
LATIN1_MEDIUM_ASCII_RATIO = 0.7
LATIN1_HIGH_CONFIDENCE = 0.7
LATIN1_MEDIUM_CONFIDENCE = 0.6

# Fast path optimization
FAST_PATH_SAMPLE_SIZE = 1024

# ASCII and surrogate ranges
ASCII_EXTENDED_MAX = 128
SURROGATE_RANGE_START = 0xD800
SURROGATE_RANGE_END = 0xDFFF


class DetectionMethod(Enum):
    """Enumeration of encoding detection methods."""
    BOM = "bom"
    XML_DECLARATION = "xml_declaration"
    STATISTICAL = "statistical"
    UTF8_VALIDATION = "utf8_validation"
    FALLBACK = "fallback"


@dataclass
class EncodingResult:
    """Result of encoding detection with confidence scoring and metadata.

    Attributes:
        encoding: Detected encoding name (canonical form)
        confidence: Confidence score from 0.0 to 1.0
        method: Detection method used
        issues: List of issues found during detection
    """
    encoding: str
    confidence: float
    method: DetectionMethod
    issues: List[str]

    def __post_init__(self) -> None:
        """Validate confidence score range."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Confidence must be between 0.0 and 1.0, got {self.confidence}"
            )


class BOMDetector:
    """Byte Order Mark (BOM) detection for all major encodings."""

    # BOM patterns for different encodings
    BOM_PATTERNS: ClassVar[Dict[bytes, str]] = {
        b"\xef\xbb\xbf": "utf-8",
        b"\xff\xfe": "utf-16-le",
        b"\xfe\xff": "utf-16-be",
        b"\xff\xfe\x00\x00": "utf-32-le",
        b"\x00\x00\xfe\xff": "utf-32-be",
    }

    def detect(self, data: bytes) -> Optional[EncodingResult]:
        """Detect encoding based on BOM.

        Args:
            data: Byte data to analyze

        Returns:
            EncodingResult if BOM detected, None otherwise
        """
        if not data:
            return None

        # Check for UTF-32 BOMs first (longer patterns)
        for bom_bytes, encoding in sorted(
            self.BOM_PATTERNS.items(), key=lambda x: len(x[0]), reverse=True
        ):
            if data.startswith(bom_bytes):
                return EncodingResult(
                    encoding=encoding,
                    confidence=1.0,
                    method=DetectionMethod.BOM,
                    issues=[]
                )

        return None


class XMLDeclarationParser:
    """Parser for XML encoding declarations."""

    # Regex pattern to match XML declaration encoding
    XML_DECLARATION_PATTERN = re.compile(
        rb'<\?xml\s+.*?encoding\s*=\s*["\']([^"\']+)["\'].*?\?>',
        re.IGNORECASE
    )

    def parse_declaration(self, data: bytes) -> Optional[EncodingResult]:
        """Parse encoding from XML declaration.

        Args:
            data: Byte data to analyze

        Returns:
            EncodingResult if declaration found and valid, None otherwise
        """
        if not data:
            return None

        # Look for XML declaration in first 1024 bytes
        header = data[:1024]
        match = self.XML_DECLARATION_PATTERN.search(header)

        if not match:
            return None

        declared_encoding = match.group(1).decode("ascii", errors="ignore").lower()

        # Validate encoding name
        normalized_encoding = self._normalize_encoding(declared_encoding)
        if not self._is_valid_encoding(normalized_encoding):
            return EncodingResult(
                encoding="utf-8",  # Fallback
                confidence=0.3,
                method=DetectionMethod.XML_DECLARATION,
                issues=[f"Invalid declared encoding: {declared_encoding}"]
            )

        return EncodingResult(
            encoding=normalized_encoding,
            confidence=0.9,
            method=DetectionMethod.XML_DECLARATION,
            issues=[]
        )

    def _normalize_encoding(self, encoding: str) -> str:
        """Normalize encoding name to canonical form."""
        # Common encoding aliases
        aliases = {
            "utf8": "utf-8",
            "utf16": "utf-16",
            "utf32": "utf-32",
            "iso-8859-1": "latin-1",
            "windows-1252": "cp1252",
        }
        return aliases.get(encoding, encoding)

    def _is_valid_encoding(self, encoding: str) -> bool:
        """Check if encoding is supported by Python codecs."""
        try:
            codecs.lookup(encoding)
        except LookupError:
            return False
        else:
            return True


class StatisticalAnalyzer:
    """Statistical analysis engine for encoding detection."""

    def analyze(self, data: bytes) -> Optional[EncodingResult]:
        """Analyze character frequency patterns for encoding detection.

        Args:
            data: Byte data to analyze

        Returns:
            EncodingResult with statistical analysis results
        """
        if not data:
            return None

        # Use first 8KB for analysis
        sample = data[:8192]

        # Try UTF-8 first (most common)
        utf8_score = self._analyze_utf8_patterns(sample)
        if utf8_score > CONFIDENCE_HIGH_UTF8_STATISTICAL:
            return EncodingResult(
                encoding="utf-8",
                confidence=utf8_score,
                method=DetectionMethod.STATISTICAL,
                issues=[]
            )

        # Try Latin-1 patterns
        latin1_score = self._analyze_latin1_patterns(sample)
        if latin1_score > CONFIDENCE_MEDIUM_LATIN1:
            return EncodingResult(
                encoding="latin-1",
                confidence=latin1_score,
                method=DetectionMethod.STATISTICAL,
                issues=[]
            )

        # Default to UTF-8 with lower confidence
        return EncodingResult(
            encoding="utf-8",
            confidence=0.5,
            method=DetectionMethod.STATISTICAL,
            issues=["Inconclusive statistical analysis"]
        )

    def _analyze_utf8_patterns(self, data: bytes) -> float:
        """Analyze byte patterns for UTF-8 likelihood."""
        if not data:
            return 0.0

        total_bytes = len(data)
        valid_utf8_sequences = 0

        i = 0
        while i < total_bytes:
            byte = data[i]

            if byte < ASCII_MAX:
                # ASCII (single byte)
                valid_utf8_sequences += 1
                i += 1
            elif byte < UTF8_CONTINUATION_MAX:
                # Invalid start byte (continuation byte in wrong position)
                return 0.0
            else:
                # Multi-byte UTF-8 sequence
                sequence_result = self._validate_utf8_sequence(data, i)
                if sequence_result is None:
                    return 0.0

                valid_utf8_sequences += 1
                i += sequence_result

        return self._calculate_utf8_confidence(valid_utf8_sequences, total_bytes)

    def _validate_utf8_sequence(self, data: bytes, start_pos: int) -> Optional[int]:
        """Validate a UTF-8 multi-byte sequence starting at position.

        Returns:
            Number of bytes in the sequence if valid, None if invalid.
        """
        byte = data[start_pos]

        if byte < UTF8_2BYTE_MAX:
            return self._validate_2byte_sequence(data, start_pos)
        if byte < UTF8_3BYTE_MAX:
            return self._validate_3byte_sequence(data, start_pos)
        if byte < UTF8_4BYTE_MAX:
            return self._validate_4byte_sequence(data, start_pos)
        # Invalid UTF-8 start byte
        return None

    def _validate_2byte_sequence(self, data: bytes, pos: int) -> Optional[int]:
        """Validate a 2-byte UTF-8 sequence."""
        if pos + 1 >= len(data):
            return None

        if (UTF8_CONTINUATION_MIN <= data[pos + 1] < UTF8_CONTINUATION_MAX):
            return 2
        return None

    def _validate_3byte_sequence(self, data: bytes, pos: int) -> Optional[int]:
        """Validate a 3-byte UTF-8 sequence."""
        if pos + 2 >= len(data):
            return None

        if (UTF8_CONTINUATION_MIN <= data[pos + 1] < UTF8_CONTINUATION_MAX and
            UTF8_CONTINUATION_MIN <= data[pos + 2] < UTF8_CONTINUATION_MAX):
            return 3
        return None

    def _validate_4byte_sequence(self, data: bytes, pos: int) -> Optional[int]:
        """Validate a 4-byte UTF-8 sequence."""
        if pos + 3 >= len(data):
            return None

        if (UTF8_CONTINUATION_MIN <= data[pos + 1] < UTF8_CONTINUATION_MAX and
            UTF8_CONTINUATION_MIN <= data[pos + 2] < UTF8_CONTINUATION_MAX and
            UTF8_CONTINUATION_MIN <= data[pos + 3] < UTF8_CONTINUATION_MAX):
            return 4
        return None

    def _calculate_utf8_confidence(
        self, valid_sequences: int, total_bytes: int
    ) -> float:
        """Calculate confidence score based on valid UTF-8 sequences."""
        if total_bytes == 0:
            return 0.0

        # High confidence if all bytes form valid UTF-8
        return min(1.0, valid_sequences / (total_bytes * 0.5))

    def _analyze_latin1_patterns(self, data: bytes) -> float:
        """Analyze byte patterns for Latin-1 likelihood."""
        if not data:
            return 0.0

        # Latin-1 allows all byte values, so check for typical patterns
        ascii_count = sum(1 for b in data if b < ASCII_EXTENDED_MAX)
        # Note: Removed unused calculation of extended_count

        total = len(data)
        if total == 0:
            return 0.0

        ascii_ratio = ascii_count / total

        # High ASCII content suggests Latin-1 compatibility
        if ascii_ratio > LATIN1_HIGH_ASCII_RATIO:
            return LATIN1_HIGH_CONFIDENCE
        if ascii_ratio > LATIN1_MEDIUM_ASCII_RATIO:
            return LATIN1_MEDIUM_CONFIDENCE
        return 0.3


class UTF8Validator:
    """UTF-8 validation with overlong sequence and surrogate detection."""

    def validate(self, data: bytes) -> EncodingResult:
        """Validate UTF-8 encoding and detect issues.

        Args:
            data: Byte data to validate

        Returns:
            EncodingResult with validation results
        """
        if not data:
            return EncodingResult(
                encoding="utf-8",
                confidence=1.0,
                method=DetectionMethod.UTF8_VALIDATION,
                issues=[]
            )

        issues = []
        confidence = 1.0

        try:
            # Try basic UTF-8 decoding
            decoded = data.decode("utf-8", errors="strict")

            # Check for surrogate characters
            issues.extend(self._check_surrogates(decoded))

            # Check for overlong sequences
            issues.extend(self._check_overlong_sequences(data))

            # Reduce confidence based on issues
            if issues:
                confidence = max(0.1, 1.0 - (len(issues) * 0.2))

        except UnicodeDecodeError as e:
            issues.append(f"UTF-8 decode error: {e}")
            confidence = 0.0

        return EncodingResult(
            encoding="utf-8",
            confidence=confidence,
            method=DetectionMethod.UTF8_VALIDATION,
            issues=issues
        )

    def _check_surrogates(self, text: str) -> List[str]:
        """Check for surrogate characters in decoded text."""
        issues = []
        for i, char in enumerate(text):
            if SURROGATE_RANGE_START <= ord(char) <= SURROGATE_RANGE_END:
                char_code = ord(char)
                issues.append(
                    f"Surrogate character found at position {i}: U+{char_code:04X}"
                )
        return issues

    def _check_overlong_sequences(self, data: bytes) -> List[str]:
        """Check for overlong UTF-8 sequences."""
        issues = []
        i = 0

        while i < len(data):
            byte = data[i]

            if byte < ASCII_MAX:
                # ASCII, skip
                i += 1
            elif byte < UTF8_CONTINUATION_MAX:
                # Continuation byte in wrong position
                issues.append(f"Invalid UTF-8 continuation byte at position {i}")
                i += 1
            elif byte < UTF8_2BYTE_MAX:
                # 2-byte sequence
                if i + 1 < len(data):
                    codepoint = ((byte & 0x1F) << 6) | (data[i + 1] & 0x3F)
                    if codepoint < OVERLONG_2BYTE_THRESHOLD:
                        issues.append(f"Overlong 2-byte sequence at position {i}")
                i += 2
            elif byte < UTF8_3BYTE_MAX:
                # 3-byte sequence
                if i + 2 < len(data):
                    codepoint = (
                        ((byte & 0x0F) << 12) |
                        ((data[i + 1] & 0x3F) << 6) |
                        (data[i + 2] & 0x3F)
                    )
                    if codepoint < OVERLONG_3BYTE_THRESHOLD:
                        issues.append(f"Overlong 3-byte sequence at position {i}")
                i += 3
            elif byte < UTF8_4BYTE_MAX:
                # 4-byte sequence
                if i + 3 < len(data):
                    codepoint = (
                        ((byte & 0x07) << 18) |
                        ((data[i + 1] & 0x3F) << 12) |
                        ((data[i + 2] & 0x3F) << 6) |
                        (data[i + 3] & 0x3F)
                    )
                    if codepoint < OVERLONG_4BYTE_THRESHOLD:
                        issues.append(f"Overlong 4-byte sequence at position {i}")
                i += 4
            else:
                # Invalid start byte
                issues.append(f"Invalid UTF-8 start byte at position {i}")
                i += 1

        return issues


class EncodingDetector:
    """Main encoding detection class with never-fail guarantee.

    Implements a cascading detection strategy:
    1. BOM detection
    2. XML declaration parsing
    3. Statistical analysis
    4. UTF-8 validation
    5. Fallback to UTF-8
    """

    def __init__(self) -> None:
        """Initialize detection components."""
        self.bom_detector = BOMDetector()
        self.xml_parser = XMLDeclarationParser()
        self.statistical_analyzer = StatisticalAnalyzer()
        self.utf8_validator = UTF8Validator()

    def detect(self, data: bytes) -> EncodingResult:
        """Detect encoding using multi-stage detection system.

        Args:
            data: Byte data to analyze

        Returns:
            EncodingResult with detected encoding and metadata
        """
        if not data:
            return EncodingResult(
                encoding="utf-8",
                confidence=1.0,
                method=DetectionMethod.FALLBACK,
                issues=[]
            )

        # Stage 1: BOM detection (highest confidence)
        bom_result = self.bom_detector.detect(data)
        if bom_result and bom_result.confidence >= CONFIDENCE_BOM_THRESHOLD:
            return bom_result

        # Stage 2: XML declaration parsing
        xml_result = self.xml_parser.parse_declaration(data)
        if xml_result and xml_result.confidence >= CONFIDENCE_XML_THRESHOLD:
            return xml_result

        # Stage 3: Statistical analysis
        stat_result = self.statistical_analyzer.analyze(data)
        if stat_result and stat_result.confidence >= CONFIDENCE_STATISTICAL_THRESHOLD:
            return stat_result

        # Stage 4: UTF-8 validation
        utf8_result = self.utf8_validator.validate(data)
        if utf8_result.confidence >= CONFIDENCE_UTF8_VALIDATION_THRESHOLD:
            return utf8_result

        # Stage 5: Fallback to UTF-8 with error handling
        return EncodingResult(
            encoding="utf-8",
            confidence=0.5,
            method=DetectionMethod.FALLBACK,
            issues=["All detection methods failed, using fallback"]
        )

    def detect_with_fast_path(self, data: bytes) -> EncodingResult:
        """Optimized detection with fast-path for clean UTF-8.

        Args:
            data: Byte data to analyze

        Returns:
            EncodingResult with detected encoding
        """
        if not data:
            return self.detect(data)

        # Fast path: check if data is clean UTF-8 (ASCII subset)
        try:
            sample_size = min(FAST_PATH_SAMPLE_SIZE, len(data))
            if all(b < ASCII_MAX for b in data[:sample_size]):
                # All ASCII, definitely UTF-8 compatible
                return EncodingResult(
                    encoding="utf-8",
                    confidence=1.0,
                    method=DetectionMethod.UTF8_VALIDATION,
                    issues=[]
                )
        except Exception:
            # If fast path fails, fall back to full detection
            pass

        # Fall back to full detection
        return self.detect(data)
