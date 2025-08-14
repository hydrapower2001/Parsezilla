"""Security features for Ultra Robust XML Parser.

Provides comprehensive security protection including XML-based attack detection
and prevention (XXE, billion laughs, zip bombs), input validation, security
policy enforcement, and audit logging for production environments.
"""

import hashlib
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Pattern, Set, Union
# XML parsing imports removed - not needed for pattern-based detection

from ultra_robust_xml_parser.shared.logging import get_logger


class SecurityLevel(Enum):
    """Security enforcement levels."""
    
    PERMISSIVE = "permissive"
    STRICT = "strict"
    PARANOID = "paranoid"


class ThreatLevel(Enum):
    """Threat severity levels."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AttackType(Enum):
    """Types of XML-based attacks."""
    
    XXE = "xxe"  # XML External Entity
    BILLION_LAUGHS = "billion_laughs"  # Entity expansion
    ZIP_BOMB = "zip_bomb"  # Compressed content expansion
    QUADRATIC_BLOWUP = "quadratic_blowup"  # Exponential content expansion
    RESOURCE_EXHAUSTION = "resource_exhaustion"  # CPU/Memory exhaustion
    INJECTION = "injection"  # XML injection attacks
    NAMESPACE_CONFUSION = "namespace_confusion"  # Namespace manipulation
    SCHEMA_POISONING = "schema_poisoning"  # Malicious schema definitions


@dataclass
class SecurityThreat:
    """Detected security threat."""
    
    threat_id: str
    attack_type: AttackType
    threat_level: ThreatLevel
    description: str
    detected_patterns: List[str] = field(default_factory=list)
    mitigation_action: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert threat to dictionary representation."""
        return {
            "threat_id": self.threat_id,
            "attack_type": self.attack_type.value,
            "threat_level": self.threat_level.value,
            "description": self.description,
            "detected_patterns": self.detected_patterns,
            "mitigation_action": self.mitigation_action,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


@dataclass
class SecurityPolicy:
    """Security policy configuration."""
    
    # Input size limits
    max_input_size: int = 10 * 1024 * 1024  # 10MB
    max_attribute_count: int = 1000
    max_attribute_size: int = 10 * 1024  # 10KB
    max_element_depth: int = 100
    max_entity_expansion: int = 1000
    
    # Entity processing
    allow_external_entities: bool = False
    allow_parameter_entities: bool = False
    max_entity_references: int = 50
    
    # Feature controls
    allow_doctypes: bool = True
    allow_processing_instructions: bool = True
    allow_cdata_sections: bool = True
    
    # Content restrictions
    forbidden_elements: Set[str] = field(default_factory=set)
    forbidden_attributes: Set[str] = field(default_factory=set)
    allowed_protocols: Set[str] = field(default_factory=lambda: {"http", "https"})
    
    # Security level
    security_level: SecurityLevel = SecurityLevel.STRICT
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert policy to dictionary representation."""
        return {
            "max_input_size": self.max_input_size,
            "max_attribute_count": self.max_attribute_count,
            "max_attribute_size": self.max_attribute_size,
            "max_element_depth": self.max_element_depth,
            "max_entity_expansion": self.max_entity_expansion,
            "allow_external_entities": self.allow_external_entities,
            "allow_parameter_entities": self.allow_parameter_entities,
            "max_entity_references": self.max_entity_references,
            "allow_doctypes": self.allow_doctypes,
            "allow_processing_instructions": self.allow_processing_instructions,
            "allow_cdata_sections": self.allow_cdata_sections,
            "forbidden_elements": list(self.forbidden_elements),
            "forbidden_attributes": list(self.forbidden_attributes),
            "allowed_protocols": list(self.allowed_protocols),
            "security_level": self.security_level.value
        }


class AttackDetector:
    """Specialized detector for XML-based attacks.
    
    Provides detection capabilities for various XML-based attacks including
    XXE, billion laughs, zip bombs, and other security threats.
    
    Examples:
        Basic attack detection:
        >>> detector = AttackDetector()
        >>> threats = detector.detect_threats("<xml>content</xml>")
        >>> for threat in threats:
        ...     print(f"Threat: {threat.attack_type.value}")
        
        Custom detection patterns:
        >>> detector = AttackDetector()
        >>> detector.add_custom_pattern(AttackType.XXE, r"<!ENTITY.*SYSTEM")
        >>> threats = detector.detect_threats(suspicious_xml)
    """
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.STRICT):
        """Initialize attack detector.
        
        Args:
            security_level: Security enforcement level
        """
        self.security_level = security_level
        self.logger = get_logger(__name__, None, "attack_detector")
        
        # Attack detection patterns
        self._attack_patterns: Dict[AttackType, List[Pattern[str]]] = {}
        self._threat_counter = 0
        
        # Initialize detection patterns
        self._initialize_patterns()
    
    def _initialize_patterns(self) -> None:
        """Initialize attack detection patterns."""
        # XXE (XML External Entity) patterns
        self._attack_patterns[AttackType.XXE] = [
            re.compile(r'<!ENTITY\s+\w+\s+SYSTEM\s+["\'][^"\']+["\']', re.IGNORECASE),
            re.compile(r'<!ENTITY\s+\w+\s+PUBLIC\s+["\'][^"\']*["\']\s+["\'][^"\']+["\']', re.IGNORECASE),
            re.compile(r'&\w+;.*SYSTEM', re.IGNORECASE),
            re.compile(r'file://', re.IGNORECASE),
            re.compile(r'gopher://', re.IGNORECASE),
            re.compile(r'ftp://', re.IGNORECASE),
            re.compile(r'<!ENTITY.*%', re.IGNORECASE)  # Parameter entities
        ]
        
        # Billion Laughs (Entity Expansion) patterns
        self._attack_patterns[AttackType.BILLION_LAUGHS] = [
            re.compile(r'<!ENTITY\s+\w+\s+"[^"]*&\w+[^"]*"', re.IGNORECASE),
            re.compile(r'&\w+;.*&\w+;.*&\w+;', re.IGNORECASE),  # Multiple entity references
            re.compile(r'<!ENTITY\s+\w+\s+"(?:[^"]*&\w+;[^"]*){3,}"', re.IGNORECASE),  # Nested references
        ]
        
        # ZIP Bomb patterns (compressed content indicators)
        self._attack_patterns[AttackType.ZIP_BOMB] = [
            re.compile(r'<\w+[^>]*>\s*[A-Za-z0-9+/]{1000,}={0,2}\s*</\w+>', re.IGNORECASE),  # Base64 content
            re.compile(r'encoding\s*=\s*["\'](?:gzip|deflate|compress)["\']', re.IGNORECASE),
            re.compile(r'content-encoding:\s*(?:gzip|deflate)', re.IGNORECASE),
        ]
        
        # Quadratic Blowup patterns
        self._attack_patterns[AttackType.QUADRATIC_BLOWUP] = [
            re.compile(r'<(\w+)[^>]*>(?:[^<]*<\1[^>]*>[^<]*</\1>){10,}[^<]*</\1>', re.IGNORECASE),
            re.compile(r'<\w+[^>]*(?:\s+\w+\s*=\s*["\'][^"\']*["\']){20,}', re.IGNORECASE),  # Many attributes
        ]
        
        # Resource Exhaustion patterns (more specific to avoid false positives)
        self._attack_patterns[AttackType.RESOURCE_EXHAUSTION] = [
            re.compile(r'<\w+>[^<]{10000,}</\w+>', re.IGNORECASE),  # Very long element content (10K+)
            re.compile(r'(?:<\w+[^>]*>\s*){200,}', re.IGNORECASE),  # Very deep nesting (200+ levels)
            re.compile(r'<\w+(?:\s+\w+="[^"]*"){200,}', re.IGNORECASE),  # Many attributes (200+)
        ]
        
        # XML Injection patterns
        self._attack_patterns[AttackType.INJECTION] = [
            re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),
            re.compile(r'javascript:', re.IGNORECASE),
            re.compile(r'vbscript:', re.IGNORECASE),
            re.compile(r'on\w+\s*=', re.IGNORECASE),  # Event handlers
            re.compile(r'<\?php', re.IGNORECASE),
            re.compile(r'<%.*%>', re.IGNORECASE),
        ]
        
        # Namespace Confusion patterns
        self._attack_patterns[AttackType.NAMESPACE_CONFUSION] = [
            re.compile(r'xmlns:\w*=\s*["\'](?:javascript:|data:|vbscript:)', re.IGNORECASE),
            re.compile(r'<\w+:\w+.*xmlns:\w+\s*=\s*["\'][^"\']{200,}["\']', re.IGNORECASE),
            re.compile(r'xmlns\s*=\s*["\']["\']', re.IGNORECASE),  # Empty namespace
        ]
        
        # Schema Poisoning patterns
        self._attack_patterns[AttackType.SCHEMA_POISONING] = [
            re.compile(r'<!DOCTYPE[^>]+SYSTEM\s+["\']https?://[^"\']+\.dtd["\']', re.IGNORECASE),
            re.compile(r'xsi:schemaLocation\s*=\s*["\'].*https?://[^"\']+\.xsd', re.IGNORECASE),
            re.compile(r'<!ATTLIST.*CDATA\s+"[^"]*javascript:', re.IGNORECASE),
        ]
    
    def add_custom_pattern(self, attack_type: AttackType, pattern: str) -> None:
        """Add custom detection pattern.
        
        Args:
            attack_type: Type of attack the pattern detects
            pattern: Regular expression pattern
        """
        compiled_pattern = re.compile(pattern, re.IGNORECASE)
        
        if attack_type not in self._attack_patterns:
            self._attack_patterns[attack_type] = []
        
        self._attack_patterns[attack_type].append(compiled_pattern)
        
        self.logger.info(
            "Custom attack pattern added",
            extra={
                "attack_type": attack_type.value,
                "pattern": pattern
            }
        )
    
    def detect_threats(self, xml_content: str) -> List[SecurityThreat]:
        """Detect security threats in XML content.
        
        Args:
            xml_content: XML content to analyze
            
        Returns:
            List of detected security threats
        """
        threats = []
        
        for attack_type, patterns in self._attack_patterns.items():
            for pattern in patterns:
                matches = pattern.findall(xml_content)
                
                if matches:
                    self._threat_counter += 1
                    
                    threat = SecurityThreat(
                        threat_id=f"threat_{self._threat_counter:06d}",
                        attack_type=attack_type,
                        threat_level=self._get_threat_level(attack_type),
                        description=self._get_threat_description(attack_type),
                        detected_patterns=matches if isinstance(matches[0], str) else [str(m) for m in matches],
                        mitigation_action=self._get_mitigation_action(attack_type),
                        metadata={
                            "pattern_count": len(matches),
                            "content_length": len(xml_content),
                            "detection_method": "pattern_matching"
                        }
                    )
                    
                    threats.append(threat)
                    
                    self.logger.warning(
                        "Security threat detected",
                        extra={
                            "threat_id": threat.threat_id,
                            "attack_type": attack_type.value,
                            "threat_level": threat.threat_level.value,
                            "pattern_matches": len(matches)
                        }
                    )
        
        return threats
    
    def _get_threat_level(self, attack_type: AttackType) -> ThreatLevel:
        """Get threat level for attack type.
        
        Args:
            attack_type: Type of attack
            
        Returns:
            Appropriate threat level
        """
        critical_attacks = {AttackType.XXE, AttackType.BILLION_LAUGHS, AttackType.ZIP_BOMB}
        high_attacks = {AttackType.RESOURCE_EXHAUSTION, AttackType.INJECTION}
        medium_attacks = {AttackType.QUADRATIC_BLOWUP, AttackType.SCHEMA_POISONING}
        
        if attack_type in critical_attacks:
            return ThreatLevel.CRITICAL
        elif attack_type in high_attacks:
            return ThreatLevel.HIGH
        elif attack_type in medium_attacks:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW
    
    def _get_threat_description(self, attack_type: AttackType) -> str:
        """Get description for attack type."""
        descriptions = {
            AttackType.XXE: "XML External Entity (XXE) attack detected - external entity processing",
            AttackType.BILLION_LAUGHS: "Billion Laughs attack detected - exponential entity expansion",
            AttackType.ZIP_BOMB: "ZIP Bomb attack detected - compressed content expansion",
            AttackType.QUADRATIC_BLOWUP: "Quadratic Blowup attack detected - polynomial content expansion",
            AttackType.RESOURCE_EXHAUSTION: "Resource Exhaustion attack detected - excessive resource consumption",
            AttackType.INJECTION: "XML Injection attack detected - malicious script injection",
            AttackType.NAMESPACE_CONFUSION: "Namespace Confusion attack detected - namespace manipulation",
            AttackType.SCHEMA_POISONING: "Schema Poisoning attack detected - malicious schema definition"
        }
        return descriptions.get(attack_type, f"Unknown attack type: {attack_type.value}")
    
    def _get_mitigation_action(self, attack_type: AttackType) -> str:
        """Get mitigation action for attack type."""
        mitigations = {
            AttackType.XXE: "Disable external entity processing and use secure XML parser configuration",
            AttackType.BILLION_LAUGHS: "Limit entity expansion depth and disable entity processing",
            AttackType.ZIP_BOMB: "Implement content size limits and validate compressed content",
            AttackType.QUADRATIC_BLOWUP: "Limit element depth and attribute count",
            AttackType.RESOURCE_EXHAUSTION: "Implement input size limits and processing timeouts",
            AttackType.INJECTION: "Validate and sanitize all XML content before processing",
            AttackType.NAMESPACE_CONFUSION: "Validate namespace declarations and URIs",
            AttackType.SCHEMA_POISONING: "Use trusted schemas and disable external schema loading"
        }
        return mitigations.get(attack_type, "Apply appropriate security measures")


class InputValidator:
    """Comprehensive input validation for XML content.
    
    Provides multi-layer validation including size limits, content filtering,
    encoding validation, and structure verification to ensure safe XML processing.
    
    Examples:
        Basic input validation:
        >>> validator = InputValidator()
        >>> is_valid = validator.validate_input(xml_data)
        >>> if not is_valid:
        ...     print("Input validation failed")
        
        Custom validation rules:
        >>> policy = SecurityPolicy(max_input_size=5*1024*1024)
        >>> validator = InputValidator(policy)
        >>> result = validator.validate_xml_structure(xml_content)
    """
    
    def __init__(self, security_policy: SecurityPolicy = None):
        """Initialize input validator.
        
        Args:
            security_policy: Security policy for validation rules
        """
        self.policy = security_policy or SecurityPolicy()
        self.logger = get_logger(__name__, None, "input_validator")
        
        # Validation statistics
        self.validation_count = 0
        self.rejection_count = 0
        self.rejection_reasons: Dict[str, int] = {}
    
    def validate_input(self, data: Union[str, bytes]) -> bool:
        """Validate input data comprehensively.
        
        Args:
            data: Input data to validate
            
        Returns:
            True if input is valid and safe
        """
        self.validation_count += 1
        
        try:
            # Convert to string if needed
            if isinstance(data, bytes):
                try:
                    data = data.decode('utf-8', errors='strict')
                except UnicodeDecodeError:
                    self._record_rejection("invalid_encoding")
                    return False
            
            # Basic safety checks
            if not self._validate_basic_safety(data):
                return False
            
            # Size limits
            if not self._validate_size_limits(data):
                return False
            
            # Content validation
            if not self._validate_content(data):
                return False
            
            # Structure validation
            if not self._validate_xml_structure(data):
                return False
            
            self.logger.debug(
                "Input validation successful",
                extra={
                    "data_size": len(data),
                    "validation_count": self.validation_count
                }
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Input validation error: {e}")
            self._record_rejection("validation_error")
            return False
    
    def _validate_basic_safety(self, data: str) -> bool:
        """Validate basic input safety."""
        if not isinstance(data, str):
            self._record_rejection("invalid_type")
            return False
        
        # Check for null bytes
        if '\x00' in data:
            self._record_rejection("null_bytes")
            return False
        
        # Check for control characters (except whitespace)
        control_chars = set(chr(i) for i in range(0, 32)) - {'\t', '\n', '\r'}
        if any(c in data for c in control_chars):
            self._record_rejection("control_characters")
            return False
        
        return True
    
    def _validate_size_limits(self, data: str) -> bool:
        """Validate size limits."""
        if len(data) > self.policy.max_input_size:
            self._record_rejection("size_limit_exceeded")
            return False
        
        # Check for excessively long lines (potential attack)
        lines = data.split('\n')
        max_line_length = min(self.policy.max_input_size // 10, 100000)
        
        for line in lines:
            if len(line) > max_line_length:
                self._record_rejection("line_too_long")
                return False
        
        return True
    
    def _validate_content(self, data: str) -> bool:
        """Validate XML content for security issues."""
        # Check for forbidden elements
        for element in self.policy.forbidden_elements:
            if f"<{element}" in data.lower():
                self._record_rejection(f"forbidden_element_{element}")
                return False
        
        # Check for forbidden attributes
        for attr in self.policy.forbidden_attributes:
            if f" {attr}=" in data.lower():
                self._record_rejection(f"forbidden_attribute_{attr}")
                return False
        
        # Check for dangerous protocols
        for protocol in ["javascript:", "vbscript:", "data:", "file:"]:
            if protocol not in [p + ":" for p in self.policy.allowed_protocols]:
                if protocol in data.lower():
                    self._record_rejection(f"dangerous_protocol_{protocol[:-1]}")
                    return False
        
        return True
    
    def _validate_xml_structure(self, data: str) -> bool:
        """Validate XML structure and limits."""
        try:
            # Count element depth
            depth = 0
            max_depth = 0
            
            # Simple depth counting (not perfect but sufficient for validation)
            for char in data:
                if char == '<':
                    # Look ahead to determine if opening or closing tag
                    next_chars = data[data.index(char):data.index(char) + 10] if data.index(char) + 10 < len(data) else data[data.index(char):]
                    if not next_chars.startswith('</'):
                        depth += 1
                        max_depth = max(max_depth, depth)
                elif char == '>' and data[max(0, data.index(char) - 1)] != '/':
                    # This is a simplified approach
                    pass
            
            if max_depth > self.policy.max_element_depth:
                self._record_rejection("depth_limit_exceeded")
                return False
            
            # Count attributes in elements (simplified)
            element_pattern = re.compile(r'<\w+([^>]*)>')
            for match in element_pattern.finditer(data):
                attr_content = match.group(1)
                attr_count = len(re.findall(r'\w+\s*=\s*["\'][^"\']*["\']', attr_content))
                
                if attr_count > self.policy.max_attribute_count:
                    self._record_rejection("attribute_count_exceeded")
                    return False
            
            # Count entity references
            entity_count = len(re.findall(r'&\w+;', data))
            if entity_count > self.policy.max_entity_references:
                self._record_rejection("entity_reference_limit")
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Structure validation error: {e}")
            self._record_rejection("structure_validation_error")
            return False
    
    def _record_rejection(self, reason: str) -> None:
        """Record validation rejection."""
        self.rejection_count += 1
        self.rejection_reasons[reason] = self.rejection_reasons.get(reason, 0) + 1
        
        self.logger.warning(
            "Input validation failed",
            extra={
                "reason": reason,
                "rejection_count": self.rejection_count,
                "total_validations": self.validation_count
            }
        )
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics.
        
        Returns:
            Dictionary containing validation statistics
        """
        return {
            "validation_count": self.validation_count,
            "rejection_count": self.rejection_count,
            "rejection_rate": self.rejection_count / max(1, self.validation_count),
            "rejection_reasons": self.rejection_reasons.copy(),
            "policy": self.policy.to_dict()
        }


class SecurityManager:
    """Comprehensive security management system for Ultra Robust XML Parser.
    
    Provides complete security protection including attack detection, input
    validation, security policy enforcement, audit logging, and threat response
    for production environments with high-security requirements.
    
    Examples:
        Basic security management:
        >>> security_manager = SecurityManager()
        >>> is_safe = security_manager.validate_and_scan(xml_content)
        >>> if not is_safe:
        ...     print("Security threat detected")
        
        Custom security policy:
        >>> policy = SecurityPolicy(security_level=SecurityLevel.PARANOID)
        >>> security_manager = SecurityManager(policy)
        >>> threats = security_manager.scan_for_threats(xml_content)
        
        Security event handling:
        >>> def security_handler(event):
        ...     print(f"Security event: {event['type']}")
        >>> security_manager.add_security_handler(security_handler)
    """
    
    def __init__(
        self,
        security_policy: SecurityPolicy = None,
        enable_audit_logging: bool = True
    ):
        """Initialize security manager.
        
        Args:
            security_policy: Security policy configuration
            enable_audit_logging: Whether to enable audit logging
        """
        self.policy = security_policy or SecurityPolicy()
        self.enable_audit_logging = enable_audit_logging
        
        # Initialize components
        self.attack_detector = AttackDetector(self.policy.security_level)
        self.input_validator = InputValidator(self.policy)
        
        # Security event handlers
        self._security_handlers: List[Callable[[Dict[str, Any]], None]] = []
        
        # Statistics and state
        self.threat_history: List[SecurityThreat] = []
        self.max_threat_history = 1000
        self.blocked_requests = 0
        self.total_requests = 0
        
        self.logger = get_logger(__name__, None, "security_manager")
        
        self.logger.info(
            "Security manager initialized",
            extra={
                "security_level": self.policy.security_level.value,
                "audit_logging": enable_audit_logging
            }
        )
    
    def validate_and_scan(self, xml_content: Union[str, bytes]) -> bool:
        """Validate input and scan for security threats.
        
        Args:
            xml_content: XML content to validate and scan
            
        Returns:
            True if content is safe to process
        """
        self.total_requests += 1
        
        try:
            # Convert to string if needed
            if isinstance(xml_content, bytes):
                try:
                    xml_content = xml_content.decode('utf-8', errors='strict')
                except UnicodeDecodeError:
                    self.blocked_requests += 1
                    self._handle_security_event("encoding_error", {
                        "error": "Invalid UTF-8 encoding",
                        "action": "blocked"
                    })
                    return False
            
            # Input validation
            if not self.input_validator.validate_input(xml_content):
                self.blocked_requests += 1
                self._handle_security_event("validation_failed", {
                    "reason": "Input validation failed",
                    "action": "blocked",
                    "stats": self.input_validator.get_validation_stats()
                })
                return False
            
            # Threat detection
            threats = self.attack_detector.detect_threats(xml_content)
            
            if threats:
                # Store threats in history
                self.threat_history.extend(threats)
                if len(self.threat_history) > self.max_threat_history:
                    self.threat_history = self.threat_history[-self.max_threat_history:]
                
                # Determine if request should be blocked
                should_block = self._should_block_threats(threats)
                
                if should_block:
                    self.blocked_requests += 1
                    self._handle_security_event("threats_detected", {
                        "threats": [t.to_dict() for t in threats],
                        "threat_count": len(threats),
                        "action": "blocked"
                    })
                    return False
                else:
                    self._handle_security_event("threats_detected", {
                        "threats": [t.to_dict() for t in threats],
                        "threat_count": len(threats),
                        "action": "allowed_with_monitoring"
                    })
            
            # Content appears safe
            self._handle_security_event("validation_success", {
                "content_size": len(xml_content),
                "action": "allowed"
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Security validation error: {e}")
            self._handle_security_event("validation_error", {
                "error": str(e),
                "action": "blocked"
            })
            return False
    
    def scan_for_threats(self, xml_content: str) -> List[SecurityThreat]:
        """Scan content for security threats without blocking.
        
        Args:
            xml_content: XML content to scan
            
        Returns:
            List of detected threats
        """
        return self.attack_detector.detect_threats(xml_content)
    
    def _should_block_threats(self, threats: List[SecurityThreat]) -> bool:
        """Determine if threats should cause blocking.
        
        Args:
            threats: List of detected threats
            
        Returns:
            True if request should be blocked
        """
        if self.policy.security_level == SecurityLevel.PARANOID:
            return len(threats) > 0  # Block any threat
        
        elif self.policy.security_level == SecurityLevel.STRICT:
            # Block critical and high threats
            critical_or_high = any(
                t.threat_level in [ThreatLevel.CRITICAL, ThreatLevel.HIGH] 
                for t in threats
            )
            return critical_or_high
        
        else:  # PERMISSIVE
            # Block only critical threats
            critical = any(t.threat_level == ThreatLevel.CRITICAL for t in threats)
            return critical
    
    def add_security_handler(self, handler: Callable[[Dict[str, Any]], None]) -> None:
        """Add security event handler.
        
        Args:
            handler: Function to call on security events
        """
        self._security_handlers.append(handler)
        
        self.logger.info(
            "Security handler added",
            extra={"handlers_count": len(self._security_handlers)}
        )
    
    def remove_security_handler(self, handler: Callable[[Dict[str, Any]], None]) -> None:
        """Remove security event handler.
        
        Args:
            handler: Handler function to remove
        """
        if handler in self._security_handlers:
            self._security_handlers.remove(handler)
    
    def _handle_security_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Handle security event.
        
        Args:
            event_type: Type of security event
            event_data: Event data
        """
        event = {
            "timestamp": time.time(),
            "type": event_type,
            "data": event_data,
            "security_level": self.policy.security_level.value
        }
        
        # Audit logging
        if self.enable_audit_logging:
            self.logger.info(
                f"Security event: {event_type}",
                extra=event
            )
        
        # Call event handlers
        for handler in self._security_handlers:
            try:
                handler(event)
            except Exception as e:
                self.logger.error(f"Security handler error: {e}")
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Get security statistics.
        
        Returns:
            Dictionary containing security statistics
        """
        threat_counts = {}
        for threat in self.threat_history:
            attack_type = threat.attack_type.value
            threat_counts[attack_type] = threat_counts.get(attack_type, 0) + 1
        
        return {
            "total_requests": self.total_requests,
            "blocked_requests": self.blocked_requests,
            "block_rate": self.blocked_requests / max(1, self.total_requests),
            "threats_detected": len(self.threat_history),
            "threat_types": threat_counts,
            "validation_stats": self.input_validator.get_validation_stats(),
            "security_policy": self.policy.to_dict()
        }
    
    def update_security_policy(self, new_policy: SecurityPolicy) -> None:
        """Update security policy.
        
        Args:
            new_policy: New security policy to apply
        """
        old_level = self.policy.security_level
        self.policy = new_policy
        
        # Update components
        self.attack_detector.security_level = new_policy.security_level
        self.input_validator.policy = new_policy
        
        self.logger.info(
            "Security policy updated",
            extra={
                "old_level": old_level.value,
                "new_level": new_policy.security_level.value
            }
        )
        
        self._handle_security_event("policy_updated", {
            "old_policy": {"security_level": old_level.value},
            "new_policy": new_policy.to_dict()
        })
    
    def export_audit_log(self, output_path: Path, format_type: str = "json") -> None:
        """Export security audit log.
        
        Args:
            output_path: Path to write audit log
            format_type: Export format ('json', 'csv')
        """
        if format_type == "json":
            import json
            
            audit_data = {
                "export_timestamp": time.time(),
                "security_stats": self.get_security_stats(),
                "threat_history": [threat.to_dict() for threat in self.threat_history],
                "security_policy": self.policy.to_dict()
            }
            
            output_path.write_text(json.dumps(audit_data, indent=2))
            
        elif format_type == "csv":
            import csv
            
            with output_path.open('w', newline='') as csvfile:
                fieldnames = [
                    'timestamp', 'threat_id', 'attack_type', 'threat_level',
                    'description', 'mitigation_action', 'pattern_count'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for threat in self.threat_history:
                    writer.writerow({
                        'timestamp': threat.timestamp,
                        'threat_id': threat.threat_id,
                        'attack_type': threat.attack_type.value,
                        'threat_level': threat.threat_level.value,
                        'description': threat.description,
                        'mitigation_action': threat.mitigation_action,
                        'pattern_count': len(threat.detected_patterns)
                    })
        
        self.logger.info(
            "Security audit log exported",
            extra={
                "output_path": str(output_path),
                "format": format_type,
                "threat_count": len(self.threat_history)
            }
        )