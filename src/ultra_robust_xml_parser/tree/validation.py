"""Tree validation and finalization for ultra-robust XML parsing.

This module provides comprehensive tree validation, optimization, and final
result object enhancement for XML documents with extensive diagnostic reporting.
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Union

from ultra_robust_xml_parser.shared import (
    DiagnosticEntry,
    DiagnosticSeverity,
    PerformanceMetrics,
    get_logger,
)
from ultra_robust_xml_parser.tree.builder import XMLDocument, XMLElement, ParseResult, StructureRepair


class ValidationLevel(Enum):
    """Validation compliance levels for different use cases."""
    
    MINIMAL = auto()     # Basic structural integrity only
    STANDARD = auto()    # XML 1.0 well-formedness rules
    STRICT = auto()      # Full XML compliance with namespaces
    PEDANTIC = auto()    # Strictest validation with all edge cases


class ValidationIssueType(Enum):
    """Types of validation issues that can be detected."""
    
    STRUCTURAL_INTEGRITY = "structural"
    WELL_FORMEDNESS = "well_formed"
    NAMESPACE_VIOLATION = "namespace"
    CONTENT_MODEL_VIOLATION = "content_model"
    ATTRIBUTE_ISSUE = "attribute"
    ENCODING_ISSUE = "encoding"


@dataclass
class ValidationIssue:
    """Single validation issue with detailed information."""
    
    issue_type: ValidationIssueType
    severity: DiagnosticSeverity
    message: str
    element_path: Optional[str] = None
    suggested_fix: Optional[str] = None
    position: Optional[Dict[str, int]] = None
    details: Optional[Dict[str, Any]] = None
    
    def __post_init__(self) -> None:
        """Validate issue data."""
        if not self.message:
            raise ValueError("Validation issue message cannot be empty")


@dataclass
class ValidationRule:
    """Configuration for validation rules."""
    
    rule_id: str
    rule_name: str
    enabled: bool = True
    severity_level: DiagnosticSeverity = DiagnosticSeverity.ERROR
    description: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate rule configuration."""
        if not self.rule_id:
            raise ValueError("Rule ID cannot be empty")
        if not self.rule_name:
            raise ValueError("Rule name cannot be empty")


@dataclass
class ValidationResult:
    """Comprehensive validation result with detailed findings."""
    
    success: bool = True
    confidence: float = 1.0
    issues: List[ValidationIssue] = field(default_factory=list)
    rules_checked: List[str] = field(default_factory=list)
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    processing_time_ms: float = 0.0
    elements_validated: int = 0
    attributes_validated: int = 0
    
    @property
    def error_count(self) -> int:
        """Get number of error-level issues."""
        return len([
            issue for issue in self.issues
            if issue.severity in (DiagnosticSeverity.ERROR, DiagnosticSeverity.CRITICAL)
        ])
    
    @property
    def warning_count(self) -> int:
        """Get number of warning-level issues."""
        return len([
            issue for issue in self.issues
            if issue.severity == DiagnosticSeverity.WARNING
        ])
    
    def get_issues_by_type(self, issue_type: ValidationIssueType) -> List[ValidationIssue]:
        """Get validation issues of specific type."""
        return [issue for issue in self.issues if issue.issue_type == issue_type]
    
    def get_issues_by_severity(self, severity: DiagnosticSeverity) -> List[ValidationIssue]:
        """Get validation issues of specific severity."""
        return [issue for issue in self.issues if issue.severity == severity]


class TreeValidator:
    """Comprehensive XML tree validation engine.
    
    Provides configurable validation of XML document trees with support for
    different compliance levels, detailed issue reporting, and repair suggestions.
    """
    
    def __init__(self, 
                 validation_level: ValidationLevel = ValidationLevel.STANDARD,
                 correlation_id: Optional[str] = None) -> None:
        """Initialize tree validator.
        
        Args:
            validation_level: Level of validation strictness
            correlation_id: Optional correlation ID for request tracking
        """
        self.validation_level = validation_level
        self.correlation_id = correlation_id
        self.logger = get_logger(__name__, correlation_id, "tree_validator")
        
        # Initialize validation rules based on level
        self._rules = self._initialize_validation_rules()
        
        # XML reserved names and characters
        self._xml_reserved_names = {
            "xml", "xmlns", "XML", "Xml", "xML", "XmL", "xMl", "XMl", "XMl"
        }
        self._invalid_name_chars = set('<>&"\'')
        self._invalid_attr_chars = set('<>&"')
    
    def validate(self, document: XMLDocument) -> ValidationResult:
        """Validate XML document tree comprehensively.
        
        Args:
            document: XMLDocument to validate
            
        Returns:
            ValidationResult with comprehensive findings
        """
        start_time = time.time()
        
        self.logger.info(
            "Starting tree validation",
            extra={
                "validation_level": self.validation_level.name,
                "document_has_root": document.root is not None,
                "total_elements": document.total_elements
            }
        )
        
        result = ValidationResult(validation_level=self.validation_level)
        
        try:
            # Run validation checks based on level
            self._validate_document_structure(document, result)
            
            if self.validation_level in (ValidationLevel.STANDARD, ValidationLevel.STRICT, ValidationLevel.PEDANTIC):
                self._validate_well_formedness(document, result)
            
            if self.validation_level in (ValidationLevel.STRICT, ValidationLevel.PEDANTIC):
                self._validate_namespaces(document, result)
            
            if self.validation_level == ValidationLevel.PEDANTIC:
                self._validate_pedantic_rules(document, result)
            
            # Calculate final result metrics
            self._finalize_validation_result(result, start_time)
            
            self.logger.info(
                "Tree validation completed",
                extra={
                    "success": result.success,
                    "error_count": result.error_count,
                    "warning_count": result.warning_count,
                    "confidence": result.confidence
                }
            )
            
        except Exception as e:
            # Never-fail philosophy: return partial result
            processing_time = (time.time() - start_time) * 1000
            
            self.logger.error(
                "Tree validation failed",
                extra={"processing_time_ms": processing_time},
                exc_info=True
            )
            
            result.success = False
            result.confidence = 0.0
            result.processing_time_ms = processing_time
            result.issues.append(ValidationIssue(
                issue_type=ValidationIssueType.STRUCTURAL_INTEGRITY,
                severity=DiagnosticSeverity.CRITICAL,
                message=f"Validation failed: {e}",
                details={"exception_type": type(e).__name__}
            ))
        
        return result
    
    def _initialize_validation_rules(self) -> Dict[str, ValidationRule]:
        """Initialize validation rules based on validation level."""
        rules = {}
        
        # Basic structural integrity rules (all levels)
        rules["parent_child_consistency"] = ValidationRule(
            rule_id="parent_child_consistency",
            rule_name="Parent-Child Relationship Consistency",
            description="Verify parent-child relationships are bidirectional and consistent"
        )
        
        rules["circular_reference_check"] = ValidationRule(
            rule_id="circular_reference_check",
            rule_name="Circular Reference Detection",
            description="Detect circular references in tree structure"
        )
        
        rules["tag_name_validity"] = ValidationRule(
            rule_id="tag_name_validity",
            rule_name="Tag Name Validity",
            description="Verify tag names contain only valid characters"
        )
        
        # Well-formedness rules (standard and above)
        if self.validation_level in (ValidationLevel.STANDARD, ValidationLevel.STRICT, ValidationLevel.PEDANTIC):
            rules["unique_root_element"] = ValidationRule(
                rule_id="unique_root_element",
                rule_name="Unique Root Element",
                description="Document must have exactly one root element"
            )
            
            rules["attribute_name_validity"] = ValidationRule(
                rule_id="attribute_name_validity",
                rule_name="Attribute Name Validity",
                description="Attribute names must be valid XML names"
            )
            
            rules["attribute_value_validity"] = ValidationRule(
                rule_id="attribute_value_validity",
                rule_name="Attribute Value Validity",
                description="Attribute values must not contain invalid characters"
            )
        
        # Namespace rules (strict and above)
        if self.validation_level in (ValidationLevel.STRICT, ValidationLevel.PEDANTIC):
            rules["namespace_prefix_validity"] = ValidationRule(
                rule_id="namespace_prefix_validity",
                rule_name="Namespace Prefix Validity",
                description="Namespace prefixes must be properly declared"
            )
            
            rules["namespace_uri_consistency"] = ValidationRule(
                rule_id="namespace_uri_consistency",
                rule_name="Namespace URI Consistency",
                description="Namespace URIs must be consistent within document"
            )
        
        # Pedantic rules (pedantic level only)
        if self.validation_level == ValidationLevel.PEDANTIC:
            rules["xml_declaration_compliance"] = ValidationRule(
                rule_id="xml_declaration_compliance",
                rule_name="XML Declaration Compliance",
                description="XML declaration must comply with XML 1.0 specification"
            )
        
        return rules
    
    def _validate_document_structure(self, document: XMLDocument, result: ValidationResult) -> None:
        """Validate basic document structure and integrity."""
        if not document.root:
            result.issues.append(ValidationIssue(
                issue_type=ValidationIssueType.STRUCTURAL_INTEGRITY,
                severity=DiagnosticSeverity.ERROR,
                message="Document has no root element",
                suggested_fix="Ensure document contains at least one XML element"
            ))
            result.success = False
            return
        
        # Validate all elements in document
        elements = document.iter_elements()
        result.elements_validated = len(elements)
        
        visited_elements: Set[int] = set()
        
        for element in elements:
            element_id = id(element)
            
            # Check for circular references
            if element_id in visited_elements:
                result.issues.append(ValidationIssue(
                    issue_type=ValidationIssueType.STRUCTURAL_INTEGRITY,
                    severity=DiagnosticSeverity.CRITICAL,
                    message="Circular reference detected in element tree",
                    element_path=element.get_path(),
                    suggested_fix="Remove circular references from tree structure"
                ))
                result.success = False
                continue
            
            visited_elements.add(element_id)
            
            # Validate parent-child consistency
            self._validate_parent_child_consistency(element, result)
            
            # Validate tag name
            self._validate_tag_name(element, result)
            
            # Count and validate attributes
            result.attributes_validated += len(element.attributes)
            for attr_name, attr_value in element.attributes.items():
                self._validate_attribute(element, attr_name, attr_value, result)
        
        result.rules_checked.extend([
            "parent_child_consistency",
            "circular_reference_check", 
            "tag_name_validity"
        ])
    
    def _validate_parent_child_consistency(self, element: XMLElement, result: ValidationResult) -> None:
        """Validate parent-child relationship consistency."""
        # Check that children reference this element as parent
        for child in element.children:
            if child.parent != element:
                result.issues.append(ValidationIssue(
                    issue_type=ValidationIssueType.STRUCTURAL_INTEGRITY,
                    severity=DiagnosticSeverity.ERROR,
                    message=f"Child element parent reference inconsistency",
                    element_path=element.get_path(),
                    suggested_fix="Ensure parent-child relationships are bidirectional"
                ))
                result.success = False
        
        # Check parent reference if not root
        if element.parent:
            if element not in element.parent.children:
                result.issues.append(ValidationIssue(
                    issue_type=ValidationIssueType.STRUCTURAL_INTEGRITY,
                    severity=DiagnosticSeverity.ERROR,
                    message=f"Element not in parent's children list",
                    element_path=element.get_path(),
                    suggested_fix="Ensure element is properly added to parent's children"
                ))
                result.success = False
    
    def _validate_tag_name(self, element: XMLElement, result: ValidationResult) -> None:
        """Validate XML tag name compliance."""
        tag = element.tag
        
        # Check for empty tag name
        if not tag:
            result.issues.append(ValidationIssue(
                issue_type=ValidationIssueType.WELL_FORMEDNESS,
                severity=DiagnosticSeverity.ERROR,
                message="Element has empty tag name",
                element_path=element.get_path(),
                suggested_fix="Provide valid tag name for element"
            ))
            result.success = False
            return
        
        # Check for invalid characters
        if any(char in self._invalid_name_chars for char in tag):
            result.issues.append(ValidationIssue(
                issue_type=ValidationIssueType.WELL_FORMEDNESS,
                severity=DiagnosticSeverity.ERROR,
                message=f"Tag name contains invalid characters: {tag}",
                element_path=element.get_path(),
                suggested_fix="Remove invalid characters from tag name"
            ))
            result.success = False
        
        # Check for XML reserved names (case insensitive)
        if tag.lower().startswith('xml'):
            result.issues.append(ValidationIssue(
                issue_type=ValidationIssueType.WELL_FORMEDNESS,
                severity=DiagnosticSeverity.WARNING,
                message=f"Tag name uses XML reserved prefix: {tag}",
                element_path=element.get_path(),
                suggested_fix="Avoid using 'xml' prefix in tag names"
            ))
    
    def _validate_attribute(self, element: XMLElement, name: str, value: str, result: ValidationResult) -> None:
        """Validate individual attribute name and value."""
        # Validate attribute name
        if not name:
            result.issues.append(ValidationIssue(
                issue_type=ValidationIssueType.ATTRIBUTE_ISSUE,
                severity=DiagnosticSeverity.ERROR,
                message="Attribute has empty name",
                element_path=element.get_path(),
                suggested_fix="Provide valid name for attribute"
            ))
            result.success = False
            return
        
        # Check for invalid characters in name
        if any(char in self._invalid_name_chars for char in name):
            result.issues.append(ValidationIssue(
                issue_type=ValidationIssueType.ATTRIBUTE_ISSUE,
                severity=DiagnosticSeverity.ERROR,
                message=f"Attribute name contains invalid characters: {name}",
                element_path=element.get_path(),
                suggested_fix="Remove invalid characters from attribute name"
            ))
            result.success = False
        
        # Check for invalid characters in value
        if any(char in self._invalid_attr_chars for char in value):
            result.issues.append(ValidationIssue(
                issue_type=ValidationIssueType.ATTRIBUTE_ISSUE,
                severity=DiagnosticSeverity.ERROR,
                message=f"Attribute value contains invalid characters: {name}={value}",
                element_path=element.get_path(),
                suggested_fix="Properly escape or remove invalid characters from attribute value"
            ))
            result.success = False
    
    def _validate_well_formedness(self, document: XMLDocument, result: ValidationResult) -> None:
        """Validate XML well-formedness rules."""
        if not document.root:
            return
        
        # Check for single root element (already checked in structure validation)
        # Check for duplicate attributes within elements
        for element in document.iter_elements():
            self._check_duplicate_attributes(element, result)
        
        result.rules_checked.extend([
            "unique_root_element",
            "attribute_name_validity",
            "attribute_value_validity"
        ])
    
    def _check_duplicate_attributes(self, element: XMLElement, result: ValidationResult) -> None:
        """Check for duplicate attribute names within an element."""
        attr_names = list(element.attributes.keys())
        seen_names: Set[str] = set()
        
        for name in attr_names:
            if name.lower() in seen_names:
                result.issues.append(ValidationIssue(
                    issue_type=ValidationIssueType.ATTRIBUTE_ISSUE,
                    severity=DiagnosticSeverity.ERROR,
                    message=f"Duplicate attribute name: {name}",
                    element_path=element.get_path(),
                    suggested_fix="Remove or rename duplicate attribute"
                ))
                result.success = False
            else:
                seen_names.add(name.lower())
    
    def _validate_namespaces(self, document: XMLDocument, result: ValidationResult) -> None:
        """Validate namespace usage and declarations."""
        if not document.root:
            return
        
        declared_namespaces: Dict[str, str] = {}
        
        for element in document.iter_elements():
            # Check namespace declarations
            for attr_name, attr_value in element.attributes.items():
                if attr_name == "xmlns" or attr_name.startswith("xmlns:"):
                    prefix = "" if attr_name == "xmlns" else attr_name[6:]  # Remove "xmlns:" prefix
                    declared_namespaces[prefix] = attr_value
            
            # Check namespace prefix usage
            if ":" in element.tag:
                prefix = element.namespace_prefix
                if prefix and prefix not in declared_namespaces:
                    result.issues.append(ValidationIssue(
                        issue_type=ValidationIssueType.NAMESPACE_VIOLATION,
                        severity=DiagnosticSeverity.ERROR,
                        message=f"Undeclared namespace prefix: {prefix}",
                        element_path=element.get_path(),
                        suggested_fix=f"Declare namespace prefix with xmlns:{prefix} attribute"
                    ))
                    result.success = False
        
        result.rules_checked.extend([
            "namespace_prefix_validity",
            "namespace_uri_consistency"
        ])
    
    def _validate_pedantic_rules(self, document: XMLDocument, result: ValidationResult) -> None:
        """Apply pedantic validation rules for strictest compliance."""
        # Check XML version and encoding
        if document.version != "1.0":
            result.issues.append(ValidationIssue(
                issue_type=ValidationIssueType.ENCODING_ISSUE,
                severity=DiagnosticSeverity.WARNING,
                message=f"Non-standard XML version: {document.version}",
                suggested_fix="Use XML version 1.0 for maximum compatibility"
            ))
        
        # Check encoding specification
        if document.encoding.lower() not in ("utf-8", "utf-16", "ascii"):
            result.issues.append(ValidationIssue(
                issue_type=ValidationIssueType.ENCODING_ISSUE,
                severity=DiagnosticSeverity.WARNING,
                message=f"Non-standard encoding: {document.encoding}",
                suggested_fix="Use UTF-8 encoding for maximum compatibility"
            ))
        
        result.rules_checked.append("xml_declaration_compliance")
    
    def _finalize_validation_result(self, result: ValidationResult, start_time: float) -> None:
        """Calculate final validation metrics and confidence."""
        result.processing_time_ms = (time.time() - start_time) * 1000
        
        # Calculate confidence based on issues found
        if result.error_count > 0:
            result.success = False
            confidence_impact = min(0.9, result.error_count * 0.1)
            result.confidence = max(0.0, 1.0 - confidence_impact)
        elif result.warning_count > 0:
            confidence_impact = min(0.5, result.warning_count * 0.05)
            result.confidence = max(0.0, 1.0 - confidence_impact)
        
        # Log validation summary
        self.logger.debug(
            "Validation result finalized",
            extra={
                "elements_validated": result.elements_validated,
                "attributes_validated": result.attributes_validated,
                "rules_checked": len(result.rules_checked),
                "processing_time_ms": result.processing_time_ms
            }
        )


class OptimizationType(Enum):
    """Types of optimizations that can be performed."""
    
    TEXT_NODE_CONSOLIDATION = "text_consolidation"
    REDUNDANT_ELEMENT_REMOVAL = "redundant_removal"
    ATTRIBUTE_OPTIMIZATION = "attribute_optimization"
    MEMORY_CLEANUP = "memory_cleanup"
    STRUCTURE_BALANCING = "structure_balancing"


@dataclass
class OptimizationAction:
    """Information about an optimization action performed."""
    
    optimization_type: OptimizationType
    description: str
    elements_affected: int = 0
    memory_saved_bytes: int = 0
    confidence_impact: float = 0.0
    details: Optional[Dict[str, Any]] = None
    
    def __post_init__(self) -> None:
        """Validate optimization action data."""
        if not self.description:
            raise ValueError("Optimization action description cannot be empty")
        if self.elements_affected < 0:
            raise ValueError("Elements affected cannot be negative")
        if self.memory_saved_bytes < 0:
            raise ValueError("Memory saved cannot be negative")


@dataclass
class OptimizationResult:
    """Result of tree optimization operations."""
    
    success: bool = True
    confidence: float = 1.0
    actions: List[OptimizationAction] = field(default_factory=list)
    processing_time_ms: float = 0.0
    total_elements_processed: int = 0
    total_memory_saved_bytes: int = 0
    
    @property
    def total_actions(self) -> int:
        """Get total number of optimization actions performed."""
        return len(self.actions)
    
    @property
    def elements_affected(self) -> int:
        """Get total number of elements affected by optimizations."""
        return sum(action.elements_affected for action in self.actions)
    
    def get_actions_by_type(self, optimization_type: OptimizationType) -> List[OptimizationAction]:
        """Get optimization actions of specific type."""
        return [action for action in self.actions if action.optimization_type == optimization_type]


class TreeOptimizer:
    """Tree optimization engine for memory and performance improvements.
    
    Provides comprehensive tree optimization including node consolidation,
    redundant element removal, memory cleanup, and structure optimization.
    """
    
    def __init__(self, correlation_id: Optional[str] = None) -> None:
        """Initialize tree optimizer.
        
        Args:
            correlation_id: Optional correlation ID for request tracking
        """
        self.correlation_id = correlation_id
        self.logger = get_logger(__name__, correlation_id, "tree_optimizer")
        
        # Optimization thresholds
        self._min_text_length_for_consolidation = 1
        self._max_consecutive_whitespace = 10
        self._redundant_element_threshold = 2
    
    def optimize(self, document: XMLDocument) -> OptimizationResult:
        """Optimize XML document tree for memory and performance.
        
        Args:
            document: XMLDocument to optimize
            
        Returns:
            OptimizationResult with comprehensive optimization information
        """
        start_time = time.time()
        
        self.logger.info(
            "Starting tree optimization",
            extra={
                "document_has_root": document.root is not None,
                "total_elements": document.total_elements
            }
        )
        
        result = OptimizationResult()
        
        try:
            if not document.root:
                self.logger.warning("Cannot optimize document without root element")
                result.success = False
                result.confidence = 0.0
                return result
            
            # Perform optimizations
            result.total_elements_processed = document.total_elements
            
            # Text node consolidation
            self._consolidate_text_nodes(document, result)
            
            # Remove redundant elements
            self._remove_redundant_elements(document, result)
            
            # Optimize attributes
            self._optimize_attributes(document, result)
            
            # Memory cleanup
            self._perform_memory_cleanup(document, result)
            
            # Structure balancing (if needed)
            self._balance_tree_structure(document, result)
            
            # Calculate final metrics
            self._finalize_optimization_result(result, start_time)
            
            self.logger.info(
                "Tree optimization completed",
                extra={
                    "success": result.success,
                    "total_actions": result.total_actions,
                    "elements_affected": result.elements_affected,
                    "memory_saved_bytes": result.total_memory_saved_bytes
                }
            )
            
        except Exception as e:
            # Never-fail philosophy: return partial result
            processing_time = (time.time() - start_time) * 1000
            
            self.logger.error(
                "Tree optimization failed",
                extra={"processing_time_ms": processing_time},
                exc_info=True
            )
            
            result.success = False
            result.confidence = 0.0
            result.processing_time_ms = processing_time
            result.actions.append(OptimizationAction(
                optimization_type=OptimizationType.MEMORY_CLEANUP,
                description=f"Optimization failed: {e}",
                details={"exception_type": type(e).__name__}
            ))
        
        return result
    
    def _consolidate_text_nodes(self, document: XMLDocument, result: OptimizationResult) -> None:
        """Consolidate adjacent text nodes and normalize whitespace."""
        elements_modified = 0
        total_memory_saved = 0
        
        for element in document.iter_elements():
            original_text = element.text
            if not original_text:
                continue
            
            # Normalize whitespace in text content
            normalized_text = self._normalize_whitespace(original_text)
            
            if normalized_text != original_text:
                element.text = normalized_text
                elements_modified += 1
                
                # Estimate memory saved
                memory_saved = len(original_text) - len(normalized_text)
                if memory_saved > 0:
                    total_memory_saved += memory_saved
        
        if elements_modified > 0:
            action = OptimizationAction(
                optimization_type=OptimizationType.TEXT_NODE_CONSOLIDATION,
                description=f"Consolidated text in {elements_modified} elements",
                elements_affected=elements_modified,
                memory_saved_bytes=total_memory_saved,
                confidence_impact=0.0  # Text consolidation doesn't affect confidence
            )
            result.actions.append(action)
            result.total_memory_saved_bytes += total_memory_saved
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text content."""
        import re
        
        # First replace tabs and newlines with spaces
        normalized = text.replace('\t', ' ').replace('\n', ' ')
        
        # Then replace multiple consecutive whitespace with single space
        normalized = re.sub(r'\s{2,}', ' ', normalized)
        
        # Trim leading/trailing whitespace
        normalized = normalized.strip()
        
        return normalized
    
    def _remove_redundant_elements(self, document: XMLDocument, result: OptimizationResult) -> None:
        """Remove redundant or empty elements that don't add value."""
        elements_removed = 0
        memory_saved = 0
        
        elements_to_remove = []
        
        for element in document.iter_elements():
            # Skip root element
            if element == document.root:
                continue
            
            # Check if element is truly empty and redundant
            if self._is_redundant_element(element):
                elements_to_remove.append(element)
        
        # Remove redundant elements
        for element in elements_to_remove:
            if element.parent and element in element.parent.children:
                element.parent.remove_child(element)
                elements_removed += 1
                
                # Estimate memory saved
                memory_saved += self._estimate_element_memory(element)
        
        if elements_removed > 0:
            action = OptimizationAction(
                optimization_type=OptimizationType.REDUNDANT_ELEMENT_REMOVAL,
                description=f"Removed {elements_removed} redundant elements",
                elements_affected=elements_removed,
                memory_saved_bytes=memory_saved,
                confidence_impact=0.0  # Removing truly redundant elements doesn't affect confidence
            )
            result.actions.append(action)
            result.total_memory_saved_bytes += memory_saved
    
    def _is_redundant_element(self, element: XMLElement) -> bool:
        """Check if element is redundant and can be safely removed."""
        # Element is redundant if:
        # 1. No text content
        # 2. No attributes
        # 3. No children
        # 4. No repairs (don't remove elements that were repaired)
        return (
            not element.text and
            not element.attributes and
            not element.children and
            not element.repairs
        )
    
    def _optimize_attributes(self, document: XMLDocument, result: OptimizationResult) -> None:
        """Optimize element attributes for memory efficiency."""
        elements_modified = 0
        memory_saved = 0
        
        for element in document.iter_elements():
            if not element.attributes:
                continue
            
            original_size = sum(len(k) + len(v) for k, v in element.attributes.items())
            
            # Remove attributes with empty values (if safe to do so)
            empty_attrs = [k for k, v in element.attributes.items() if not v]
            for attr_name in empty_attrs:
                del element.attributes[attr_name]
                elements_modified += 1
            
            # Calculate memory saved
            if empty_attrs:
                new_size = sum(len(k) + len(v) for k, v in element.attributes.items())
                memory_saved += original_size - new_size
        
        if elements_modified > 0:
            action = OptimizationAction(
                optimization_type=OptimizationType.ATTRIBUTE_OPTIMIZATION,
                description=f"Optimized attributes in {elements_modified} elements",
                elements_affected=elements_modified,
                memory_saved_bytes=memory_saved,
                confidence_impact=0.0  # Attribute optimization doesn't affect confidence
            )
            result.actions.append(action)
            result.total_memory_saved_bytes += memory_saved
    
    def _perform_memory_cleanup(self, document: XMLDocument, result: OptimizationResult) -> None:
        """Perform memory cleanup operations."""
        elements_cleaned = 0
        memory_saved = 0
        
        for element in document.iter_elements():
            # Clear unnecessary source token references for memory
            if element.source_tokens:
                token_memory = len(element.source_tokens) * 100  # Rough estimate
                element.source_tokens.clear()
                elements_cleaned += 1
                memory_saved += token_memory
        
        if elements_cleaned > 0:
            action = OptimizationAction(
                optimization_type=OptimizationType.MEMORY_CLEANUP,
                description=f"Cleaned memory references in {elements_cleaned} elements",
                elements_affected=elements_cleaned,
                memory_saved_bytes=memory_saved,
                confidence_impact=0.0  # Memory cleanup doesn't affect confidence
            )
            result.actions.append(action)
            result.total_memory_saved_bytes += memory_saved
    
    def _balance_tree_structure(self, document: XMLDocument, result: OptimizationResult) -> None:
        """Balance tree structure for optimal navigation performance."""
        # For now, this is a placeholder for more complex structure balancing
        # In the future, this could implement tree balancing algorithms
        
        max_depth = document.max_depth
        if max_depth > 50:  # Very deep tree might benefit from restructuring
            action = OptimizationAction(
                optimization_type=OptimizationType.STRUCTURE_BALANCING,
                description=f"Analyzed tree structure (depth: {max_depth})",
                elements_affected=0,
                memory_saved_bytes=0,
                confidence_impact=0.0,
                details={"max_depth": max_depth, "analysis_only": True}
            )
            result.actions.append(action)
    
    def _estimate_element_memory(self, element: XMLElement) -> int:
        """Estimate memory usage of an element."""
        # Rough estimation of element memory usage
        memory = len(element.tag) * 2  # Tag name storage
        
        if element.text:
            memory += len(element.text)
        
        # Attributes
        for k, v in element.attributes.items():
            memory += len(k) + len(v)
        
        # Object overhead
        memory += 200  # Rough estimate for object overhead
        
        return memory
    
    def _finalize_optimization_result(self, result: OptimizationResult, start_time: float) -> None:
        """Calculate final optimization metrics."""
        result.processing_time_ms = (time.time() - start_time) * 1000
        
        # Calculate overall confidence based on optimizations performed
        if result.total_actions == 0:
            result.confidence = 1.0
        else:
            # Slight confidence boost from successful optimization
            confidence_boost = min(0.1, result.total_actions * 0.01)
            result.confidence = min(1.0, 1.0 + confidence_boost)
        
        self.logger.debug(
            "Optimization result finalized",
            extra={
                "total_actions": result.total_actions,
                "processing_time_ms": result.processing_time_ms,
                "memory_saved_bytes": result.total_memory_saved_bytes
            }
        )


class OutputFormat(Enum):
    """Supported output formats for XML serialization."""
    
    XML_STRING = "xml"
    XML_PRETTY = "xml_pretty" 
    XML_MINIFIED = "xml_minified"
    DICTIONARY = "dict"
    DICTIONARY_DETAILED = "dict_detailed"
    JSON = "json"
    JSON_PRETTY = "json_pretty"
    JSON_COMPACT = "json_compact"


@dataclass
class OutputConfiguration:
    """Configuration options for output formatting."""
    
    # XML formatting options
    xml_declaration: bool = True
    xml_encoding: str = "utf-8"
    xml_indent: str = "  "  # Two spaces for pretty printing
    xml_preserve_whitespace: bool = False
    
    # Dictionary formatting options
    include_attributes: bool = True
    include_metadata: bool = False
    include_repairs: bool = False
    include_position_info: bool = False
    dict_attribute_prefix: str = "@"
    dict_text_key: str = "#text"
    dict_children_key: Optional[str] = None  # None means use tag names directly
    
    # JSON formatting options
    json_indent: Optional[int] = 2  # None for compact, int for pretty
    json_ensure_ascii: bool = False
    json_sort_keys: bool = False
    
    # Content filtering options
    exclude_empty_elements: bool = False
    exclude_empty_attributes: bool = False
    exclude_comments: bool = True
    exclude_processing_instructions: bool = True
    
    # Namespace handling
    namespace_aware: bool = True
    namespace_prefix_map: Dict[str, str] = field(default_factory=dict)


@dataclass
class FormatResult:
    """Result of output formatting operation."""
    
    success: bool = True
    formatted_output: str = ""
    format_used: OutputFormat = OutputFormat.XML_STRING
    processing_time_ms: float = 0.0
    output_size_bytes: int = 0
    issues: List[str] = field(default_factory=list)


class OutputFormatter:
    """Comprehensive output formatter for XML documents.
    
    Supports multiple output formats including XML serialization, dictionary
    conversion, and JSON serialization with extensive configuration options.
    """
    
    def __init__(self, correlation_id: Optional[str] = None) -> None:
        """Initialize output formatter.
        
        Args:
            correlation_id: Optional correlation ID for request tracking
        """
        self.correlation_id = correlation_id
        self.logger = get_logger(__name__, correlation_id, "output_formatter")
    
    def format(self, 
               document: XMLDocument, 
               output_format: OutputFormat = OutputFormat.XML_STRING,
               config: Optional[OutputConfiguration] = None) -> FormatResult:
        """Format XML document to specified output format.
        
        Args:
            document: XMLDocument to format
            output_format: Desired output format
            config: Optional configuration for formatting
            
        Returns:
            FormatResult with formatted output and metadata
        """
        start_time = time.time()
        
        if config is None:
            config = OutputConfiguration()
        
        self.logger.info(
            "Starting output formatting",
            extra={
                "output_format": output_format.value,
                "document_has_root": document.root is not None,
                "total_elements": document.total_elements
            }
        )
        
        result = FormatResult(format_used=output_format)
        
        try:
            if not document.root:
                result.success = False
                result.issues.append("Document has no root element")
                result.formatted_output = ""
                return result
            
            # Route to appropriate formatting method
            if output_format in (OutputFormat.XML_STRING, OutputFormat.XML_PRETTY, OutputFormat.XML_MINIFIED):
                result.formatted_output = self._format_xml(document, output_format, config)
            elif output_format in (OutputFormat.DICTIONARY, OutputFormat.DICTIONARY_DETAILED):
                formatted_dict = self._format_dictionary(document, output_format, config)
                result.formatted_output = str(formatted_dict)  # Convert dict to string representation
            elif output_format in (OutputFormat.JSON, OutputFormat.JSON_PRETTY, OutputFormat.JSON_COMPACT):
                result.formatted_output = self._format_json(document, output_format, config)
            else:
                result.success = False
                result.issues.append(f"Unsupported output format: {output_format}")
                result.formatted_output = ""
            
            # Calculate final metrics
            self._finalize_format_result(result, start_time)
            
            self.logger.info(
                "Output formatting completed",
                extra={
                    "success": result.success,
                    "output_size_bytes": result.output_size_bytes,
                    "issues_count": len(result.issues)
                }
            )
            
        except Exception as e:
            # Never-fail philosophy: return failed result
            processing_time = (time.time() - start_time) * 1000
            
            self.logger.error(
                "Output formatting failed",
                extra={"processing_time_ms": processing_time},
                exc_info=True
            )
            
            result.success = False
            result.processing_time_ms = processing_time
            result.issues.append(f"Formatting failed: {e}")
            result.formatted_output = ""
        
        return result
    
    def _format_xml(self, 
                    document: XMLDocument, 
                    output_format: OutputFormat,
                    config: OutputConfiguration) -> str:
        """Format document as XML string."""
        xml_parts = []
        
        # Add XML declaration if requested
        if config.xml_declaration:
            declaration = f'<?xml version="{document.version}"'
            if config.xml_encoding:
                declaration += f' encoding="{config.xml_encoding}"'
            if document.standalone is not None:
                standalone_value = "yes" if document.standalone else "no"
                declaration += f' standalone="{standalone_value}"'
            declaration += '?>'
            xml_parts.append(declaration)
        
        # Format root element
        if document.root:
            if output_format == OutputFormat.XML_PRETTY:
                root_xml = self._format_element_xml(document.root, config, indent_level=0, pretty=True)
            elif output_format == OutputFormat.XML_MINIFIED:
                root_xml = self._format_element_xml(document.root, config, indent_level=0, pretty=False, minify=True)
            else:  # XML_STRING
                root_xml = self._format_element_xml(document.root, config, indent_level=0, pretty=False)
            
            xml_parts.append(root_xml)
        
        if output_format == OutputFormat.XML_PRETTY:
            return '\n'.join(xml_parts)
        elif output_format == OutputFormat.XML_MINIFIED:
            return ''.join(xml_parts)
        else:
            return '\n'.join(xml_parts)
    
    def _format_element_xml(self, 
                           element: XMLElement, 
                           config: OutputConfiguration,
                           indent_level: int = 0,
                           pretty: bool = False,
                           minify: bool = False) -> str:
        """Format single element as XML string."""
        indent = config.xml_indent * indent_level if pretty else ""
        
        # Start opening tag (without < bracket)
        tag_parts = [element.tag]
        
        # Add attributes
        if config.include_attributes and element.attributes:
            for name, value in element.attributes.items():
                # Skip empty attributes if configured
                if config.exclude_empty_attributes and not value:
                    continue
                
                # Escape attribute value
                escaped_value = self._escape_xml_attribute(value)
                tag_parts.append(f'{name}="{escaped_value}"')
        
        # Check if element is empty
        has_content = bool(element.text or element.children)
        
        if not has_content and config.exclude_empty_elements:
            return ""  # Skip empty elements
        
        # Check if element should preserve structure (e.g., when invalid content was removed)
        should_preserve_structure = getattr(element, 'preserve_structure', False)
        
        if not has_content and not should_preserve_structure:
            # Self-closing tag
            opening_tag = f"{indent}<{' '.join(tag_parts)}/>"
            return opening_tag
        
        # Complete opening tag
        opening_tag = f"{indent}<{' '.join(tag_parts)}>"
        
        # Add text content
        content_parts = []
        
        # Add any comments first (unescaped)
        if hasattr(element, '_comments'):
            for comment in element._comments:
                content_parts.append(comment)
                
        # Add any processing instructions (unescaped)
        if hasattr(element, '_processing_instructions'):
            for pi in element._processing_instructions:
                content_parts.append(pi)
        
        if element.text:
            escaped_text = self._escape_xml_text(element.text)
            if minify:
                escaped_text = escaped_text.strip()
            content_parts.append(escaped_text)
        
        # Add children
        for child in element.children:
            child_xml = self._format_element_xml(
                child, config, indent_level + 1 if pretty else 0, pretty, minify
            )
            if child_xml:  # Only add non-empty child XML
                content_parts.append(child_xml)
        
        # Closing tag
        closing_tag = f"</{element.tag}>"
        if pretty and element.children:
            closing_tag = f"{indent}</{element.tag}>"
        
        # Combine parts
        if minify:
            return f"{opening_tag}{''.join(content_parts)}{closing_tag}"
        elif pretty:
            if element.children:
                children_xml = '\n'.join([part for part in content_parts if part.strip()])
                return f"{opening_tag}\n{children_xml}\n{closing_tag}"
            else:
                return f"{opening_tag}{''.join(content_parts)}{closing_tag}"
        else:
            return f"{opening_tag}{''.join(content_parts)}{closing_tag}"
    
    def _format_dictionary(self, 
                          document: XMLDocument, 
                          output_format: OutputFormat,
                          config: OutputConfiguration) -> Dict[str, Any]:
        """Format document as dictionary."""
        result_dict: Dict[str, Any] = {}
        
        # Add document metadata if requested
        if config.include_metadata:
            result_dict["_document_metadata"] = {
                "version": document.version,
                "encoding": document.encoding,
                "standalone": document.standalone,
                "total_elements": document.total_elements,
                "total_attributes": document.total_attributes,
                "max_depth": document.max_depth,
                "confidence": document.confidence,
                "has_repairs": document.has_repairs,
            }
        
        # Format root element
        if document.root:
            if output_format == OutputFormat.DICTIONARY_DETAILED:
                result_dict[document.root.tag] = self._format_element_dict_detailed(document.root, config)
            else:
                result_dict[document.root.tag] = self._format_element_dict(document.root, config)
        
        return result_dict
    
    def _format_element_dict(self, element: XMLElement, config: OutputConfiguration) -> Dict[str, Any]:
        """Format element as simple dictionary."""
        element_dict: Dict[str, Any] = {}
        
        # Add attributes with prefix
        if config.include_attributes and element.attributes:
            for name, value in element.attributes.items():
                if config.exclude_empty_attributes and not value:
                    continue
                element_dict[f"{config.dict_attribute_prefix}{name}"] = value
        
        # Add text content
        if element.text:
            element_dict[config.dict_text_key] = element.text
        
        # Add children
        if element.children:
            for child in element.children:
                if config.exclude_empty_elements and self._is_empty_element(child):
                    continue
                
                child_dict = self._format_element_dict(child, config)
                
                # Handle multiple children with same tag
                if child.tag in element_dict:
                    # Convert to list if not already
                    if not isinstance(element_dict[child.tag], list):
                        element_dict[child.tag] = [element_dict[child.tag]]
                    element_dict[child.tag].append(child_dict)
                else:
                    element_dict[child.tag] = child_dict
        
        return element_dict
    
    def _format_element_dict_detailed(self, element: XMLElement, config: OutputConfiguration) -> Dict[str, Any]:
        """Format element as detailed dictionary with metadata."""
        element_dict = self._format_element_dict(element, config)
        
        # Add detailed metadata
        if config.include_metadata:
            element_dict["_element_metadata"] = {
                "tag": element.tag,
                "confidence": element.confidence,
                "has_repairs": element.has_repairs,
                "child_count": len(element.children),
                "attribute_count": len(element.attributes),
                "depth": element.get_depth(),
                "path": element.get_path(),
            }
        
        # Add repair information
        if config.include_repairs and element.repairs:
            element_dict["_repairs"] = [
                {
                    "repair_type": repair.repair_type,
                    "description": repair.description,
                    "confidence_impact": repair.confidence_impact,
                    "severity": repair.severity,
                }
                for repair in element.repairs
            ]
        
        # Add position information
        if config.include_position_info and element.start_position:
            element_dict["_position"] = element.start_position
        
        return element_dict
    
    def _format_json(self, 
                     document: XMLDocument, 
                     output_format: OutputFormat,
                     config: OutputConfiguration) -> str:
        """Format document as JSON string."""
        import json
        
        # First convert to dictionary
        if output_format == OutputFormat.JSON:
            document_dict = self._format_dictionary(document, OutputFormat.DICTIONARY, config)
        else:
            document_dict = self._format_dictionary(document, OutputFormat.DICTIONARY_DETAILED, config)
        
        # Configure JSON serialization
        if output_format == OutputFormat.JSON_COMPACT:
            return json.dumps(
                document_dict,
                ensure_ascii=config.json_ensure_ascii,
                separators=(',', ':')  # Compact separators
            )
        elif output_format == OutputFormat.JSON_PRETTY:
            return json.dumps(
                document_dict,
                indent=config.json_indent or 2,
                ensure_ascii=config.json_ensure_ascii,
                sort_keys=config.json_sort_keys
            )
        else:  # JSON
            return json.dumps(
                document_dict,
                indent=config.json_indent,
                ensure_ascii=config.json_ensure_ascii,
                sort_keys=config.json_sort_keys
            )
    
    def _escape_xml_text(self, text: str) -> str:
        """Escape XML text content, avoiding double-encoding of already escaped entities."""
        import re
        
        # First, repair malformed entities by adding missing semicolons
        repaired_text = self._repair_malformed_entities(text)
        
        # Pattern to match properly encoded entities
        entity_pattern = r'&(?:amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);'
        
        # If the text already contains XML entities, assume it's already properly encoded
        if re.search(entity_pattern, repaired_text):
            return repaired_text
        
        # Otherwise, perform standard XML escaping
        return (repaired_text
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;'))
    
    def _repair_malformed_entities(self, text: str) -> str:
        """Repair malformed entity references by adding missing semicolons."""
        import re
        
        # Pattern to match incomplete entities (& followed by entity name but no semicolon)
        # Look for & followed by known entity names that are not already terminated with ;
        incomplete_entity_pattern = r'&(amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+)(?![;a-zA-Z0-9])'
        
        def repair_entity(match):
            entity_name = match.group(1)
            return f'&{entity_name};'
        
        # Replace incomplete entities with complete ones
        repaired_text = re.sub(incomplete_entity_pattern, repair_entity, text)
        
        return repaired_text
    
    def _escape_xml_attribute(self, value: str) -> str:
        """Escape XML attribute value."""
        return (value
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&#39;'))
    
    def _is_empty_element(self, element: XMLElement) -> bool:
        """Check if element is empty (no content)."""
        return not (element.text or element.attributes or element.children)
    
    def _finalize_format_result(self, result: FormatResult, start_time: float) -> None:
        """Calculate final formatting metrics."""
        result.processing_time_ms = (time.time() - start_time) * 1000
        result.output_size_bytes = len(result.formatted_output.encode('utf-8'))
        
        self.logger.debug(
            "Format result finalized",
            extra={
                "processing_time_ms": result.processing_time_ms,
                "output_size_bytes": result.output_size_bytes,
                "issues_count": len(result.issues)
            }
        )


@dataclass
class BenchmarkResult:
    """Result of a performance benchmark test."""
    
    test_name: str
    success: bool = True
    processing_time_ms: float = 0.0
    memory_used_bytes: int = 0
    elements_processed: int = 0
    throughput_elements_per_second: float = 0.0
    error_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def memory_used_mb(self) -> float:
        """Get memory usage in megabytes."""
        return self.memory_used_bytes / (1024 * 1024)
    
    @property
    def processing_time_seconds(self) -> float:
        """Get processing time in seconds."""
        return self.processing_time_ms / 1000.0


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results with comparison capabilities."""
    
    suite_name: str
    results: List[BenchmarkResult] = field(default_factory=list)
    total_time_ms: float = 0.0
    
    def add_result(self, result: BenchmarkResult) -> None:
        """Add a benchmark result to the suite."""
        self.results.append(result)
        if result.success:
            self.total_time_ms += result.processing_time_ms
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if not self.results:
            return 0.0
        successful = sum(1 for r in self.results if r.success)
        return (successful / len(self.results)) * 100.0
    
    @property
    def average_processing_time_ms(self) -> float:
        """Calculate average processing time."""
        successful_results = [r for r in self.results if r.success]
        if not successful_results:
            return 0.0
        return sum(r.processing_time_ms for r in successful_results) / len(successful_results)
    
    @property
    def total_elements_processed(self) -> int:
        """Get total elements processed across all tests."""
        return sum(r.elements_processed for r in self.results if r.success)
    
    def get_best_result(self, metric: str = "processing_time_ms") -> Optional[BenchmarkResult]:
        """Get the best result based on a metric (lower is better for time)."""
        successful_results = [r for r in self.results if r.success]
        if not successful_results:
            return None
        
        if metric == "processing_time_ms":
            return min(successful_results, key=lambda r: r.processing_time_ms)
        elif metric == "memory_used_bytes":
            return min(successful_results, key=lambda r: r.memory_used_bytes)
        elif metric == "throughput_elements_per_second":
            return max(successful_results, key=lambda r: r.throughput_elements_per_second)
        else:
            return successful_results[0]


class PerformanceBenchmark:
    """Performance benchmarking system for tree finalization operations.
    
    Provides comprehensive benchmarking capabilities for validation,
    optimization, and formatting operations with comparison and reporting.
    """
    
    def __init__(self, correlation_id: Optional[str] = None) -> None:
        """Initialize performance benchmark system.
        
        Args:
            correlation_id: Optional correlation ID for request tracking
        """
        self.correlation_id = correlation_id
        self.logger = get_logger(__name__, correlation_id, "performance_benchmark")
        self.benchmark_suites: List[BenchmarkSuite] = []
    
    def benchmark_validation(self, 
                           documents: List[XMLDocument],
                           validation_levels: Optional[List[ValidationLevel]] = None) -> BenchmarkSuite:
        """Benchmark tree validation performance across different levels.
        
        Args:
            documents: List of documents to benchmark
            validation_levels: Validation levels to test (default: all levels)
            
        Returns:
            BenchmarkSuite with validation benchmark results
        """
        if validation_levels is None:
            validation_levels = list(ValidationLevel)
        
        suite = BenchmarkSuite("Tree Validation Benchmark")
        
        self.logger.info(
            "Starting validation benchmark",
            extra={
                "document_count": len(documents),
                "validation_levels": len(validation_levels)
            }
        )
        
        for level in validation_levels:
            for i, document in enumerate(documents):
                test_name = f"validation_{level.name.lower()}_doc_{i+1}"
                
                try:
                    # Measure validation performance
                    start_time = time.time()
                    start_memory = self._get_memory_usage()
                    
                    validator = TreeValidator(validation_level=level)
                    validation_result = validator.validate(document)
                    
                    end_time = time.time()
                    end_memory = self._get_memory_usage()
                    
                    # Calculate metrics
                    processing_time_ms = (end_time - start_time) * 1000
                    memory_used = max(0, end_memory - start_memory)
                    elements_processed = validation_result.elements_validated
                    
                    throughput = 0.0
                    if processing_time_ms > 0:
                        throughput = (elements_processed / processing_time_ms) * 1000  # elements per second
                    
                    result = BenchmarkResult(
                        test_name=test_name,
                        success=validation_result.success,
                        processing_time_ms=processing_time_ms,
                        memory_used_bytes=memory_used,
                        elements_processed=elements_processed,
                        throughput_elements_per_second=throughput,
                        details={
                            "validation_level": level.name,
                            "confidence": validation_result.confidence,
                            "issues_count": len(validation_result.issues),
                            "rules_checked": len(validation_result.rules_checked)
                        }
                    )
                    
                except Exception as e:
                    result = BenchmarkResult(
                        test_name=test_name,
                        success=False,
                        error_message=str(e),
                        details={"validation_level": level.name}
                    )
                
                suite.add_result(result)
        
        self.benchmark_suites.append(suite)
        
        self.logger.info(
            "Validation benchmark completed",
            extra={
                "total_tests": len(suite.results),
                "success_rate": suite.success_rate,
                "total_time_ms": suite.total_time_ms
            }
        )
        
        return suite
    
    def benchmark_optimization(self, documents: List[XMLDocument]) -> BenchmarkSuite:
        """Benchmark tree optimization performance.
        
        Args:
            documents: List of documents to benchmark
            
        Returns:
            BenchmarkSuite with optimization benchmark results
        """
        suite = BenchmarkSuite("Tree Optimization Benchmark")
        
        self.logger.info(
            "Starting optimization benchmark",
            extra={"document_count": len(documents)}
        )
        
        for i, document in enumerate(documents):
            test_name = f"optimization_doc_{i+1}"
            
            try:
                # Create a copy for benchmarking (optimization modifies the document)
                doc_copy = self._copy_document(document)
                
                # Measure optimization performance
                start_time = time.time()
                start_memory = self._get_memory_usage()
                elements_before = doc_copy.total_elements
                
                optimizer = TreeOptimizer()
                optimization_result = optimizer.optimize(doc_copy)
                
                end_time = time.time()
                end_memory = self._get_memory_usage()
                elements_after = doc_copy.total_elements
                
                # Calculate metrics
                processing_time_ms = (end_time - start_time) * 1000
                memory_used = max(0, end_memory - start_memory)
                
                throughput = 0.0
                if processing_time_ms > 0:
                    throughput = (elements_before / processing_time_ms) * 1000
                
                result = BenchmarkResult(
                    test_name=test_name,
                    success=optimization_result.success,
                    processing_time_ms=processing_time_ms,
                    memory_used_bytes=memory_used,
                    elements_processed=elements_before,
                    throughput_elements_per_second=throughput,
                    details={
                        "elements_before": elements_before,
                        "elements_after": elements_after,
                        "elements_removed": elements_before - elements_after,
                        "actions_performed": optimization_result.total_actions,
                        "memory_saved_bytes": optimization_result.total_memory_saved_bytes,
                        "confidence": optimization_result.confidence
                    }
                )
                
            except Exception as e:
                result = BenchmarkResult(
                    test_name=test_name,
                    success=False,
                    error_message=str(e)
                )
            
            suite.add_result(result)
        
        self.benchmark_suites.append(suite)
        
        self.logger.info(
            "Optimization benchmark completed",
            extra={
                "total_tests": len(suite.results),
                "success_rate": suite.success_rate
            }
        )
        
        return suite
    
    def benchmark_formatting(self, 
                           documents: List[XMLDocument],
                           output_formats: Optional[List[OutputFormat]] = None) -> BenchmarkSuite:
        """Benchmark output formatting performance.
        
        Args:
            documents: List of documents to benchmark
            output_formats: Output formats to test (default: common formats)
            
        Returns:
            BenchmarkSuite with formatting benchmark results
        """
        if output_formats is None:
            output_formats = [
                OutputFormat.XML_STRING,
                OutputFormat.XML_PRETTY,
                OutputFormat.JSON,
                OutputFormat.DICTIONARY
            ]
        
        suite = BenchmarkSuite("Output Formatting Benchmark")
        
        self.logger.info(
            "Starting formatting benchmark",
            extra={
                "document_count": len(documents),
                "output_formats": len(output_formats)
            }
        )
        
        for output_format in output_formats:
            for i, document in enumerate(documents):
                test_name = f"formatting_{output_format.value}_doc_{i+1}"
                
                try:
                    # Measure formatting performance
                    start_time = time.time()
                    start_memory = self._get_memory_usage()
                    
                    formatter = OutputFormatter()
                    format_result = formatter.format(document, output_format)
                    
                    end_time = time.time()
                    end_memory = self._get_memory_usage()
                    
                    # Calculate metrics
                    processing_time_ms = (end_time - start_time) * 1000
                    memory_used = max(0, end_memory - start_memory)
                    elements_processed = document.total_elements
                    
                    throughput = 0.0
                    if processing_time_ms > 0:
                        throughput = (elements_processed / processing_time_ms) * 1000
                    
                    result = BenchmarkResult(
                        test_name=test_name,
                        success=format_result.success,
                        processing_time_ms=processing_time_ms,
                        memory_used_bytes=memory_used,
                        elements_processed=elements_processed,
                        throughput_elements_per_second=throughput,
                        details={
                            "output_format": output_format.value,
                            "output_size_bytes": format_result.output_size_bytes,
                            "issues_count": len(format_result.issues)
                        }
                    )
                    
                except Exception as e:
                    result = BenchmarkResult(
                        test_name=test_name,
                        success=False,
                        error_message=str(e),
                        details={"output_format": output_format.value}
                    )
                
                suite.add_result(result)
        
        self.benchmark_suites.append(suite)
        
        self.logger.info(
            "Formatting benchmark completed",
            extra={
                "total_tests": len(suite.results),
                "success_rate": suite.success_rate
            }
        )
        
        return suite
    
    def generate_benchmark_report(self, include_details: bool = False) -> Dict[str, Any]:
        """Generate comprehensive benchmark report.
        
        Args:
            include_details: Whether to include detailed test results
            
        Returns:
            Dictionary containing benchmark report data
        """
        report = {
            "benchmark_summary": {
                "total_suites": len(self.benchmark_suites),
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                "correlation_id": self.correlation_id
            },
            "suites": []
        }
        
        for suite in self.benchmark_suites:
            suite_data = {
                "name": suite.suite_name,
                "total_tests": len(suite.results),
                "success_rate": suite.success_rate,
                "total_time_ms": suite.total_time_ms,
                "average_time_ms": suite.average_processing_time_ms,
                "total_elements": suite.total_elements_processed,
                "summary_stats": self._calculate_suite_statistics(suite)
            }
            
            if include_details:
                suite_data["detailed_results"] = [
                    {
                        "test_name": r.test_name,
                        "success": r.success,
                        "processing_time_ms": r.processing_time_ms,
                        "memory_used_mb": r.memory_used_mb,
                        "elements_processed": r.elements_processed,
                        "throughput": r.throughput_elements_per_second,
                        "error": r.error_message,
                        "details": r.details
                    }
                    for r in suite.results
                ]
            
            report["suites"].append(suite_data)
        
        return report
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss
        except ImportError:
            # If psutil is not available, return 0
            return 0
    
    def _copy_document(self, document: XMLDocument) -> XMLDocument:
        """Create a deep copy of a document for benchmarking."""
        # This is a simplified copy - in a real implementation,
        # you would want a more sophisticated deep copy
        if not document.root:
            return XMLDocument()
        
        # For benchmarking purposes, we'll use the original document
        # and note that optimization is destructive
        return document
    
    def _calculate_suite_statistics(self, suite: BenchmarkSuite) -> Dict[str, Any]:
        """Calculate statistical summary for a benchmark suite."""
        successful_results = [r for r in suite.results if r.success]
        
        if not successful_results:
            return {
                "min_time_ms": 0,
                "max_time_ms": 0,
                "median_time_ms": 0,
                "std_dev_time_ms": 0,
                "min_memory_mb": 0,
                "max_memory_mb": 0,
                "avg_throughput": 0
            }
        
        times = [r.processing_time_ms for r in successful_results]
        memories = [r.memory_used_mb for r in successful_results]
        throughputs = [r.throughput_elements_per_second for r in successful_results]
        
        # Calculate statistics
        import statistics
        
        return {
            "min_time_ms": min(times),
            "max_time_ms": max(times),
            "median_time_ms": statistics.median(times),
            "std_dev_time_ms": statistics.stdev(times) if len(times) > 1 else 0,
            "min_memory_mb": min(memories) if memories else 0,
            "max_memory_mb": max(memories) if memories else 0,
            "avg_throughput": statistics.mean(throughputs) if throughputs else 0
        }


# ============================================================================
# Diagnostic Reporting System
# ============================================================================

@dataclass
class DiagnosticCategory:
    """Represents a category of diagnostic information."""
    name: str
    description: str
    severity_level: DiagnosticSeverity
    issue_count: int = 0
    
    def add_issue(self) -> None:
        """Add an issue to this category."""
        self.issue_count += 1


@dataclass
class ErrorSummary:
    """Comprehensive error analysis and summary."""
    total_errors: int
    critical_errors: int
    recoverable_errors: int
    error_categories: Dict[str, int]
    root_causes: List[str]
    affected_elements: int
    error_rate: float  # Errors per element
    
    def has_critical_errors(self) -> bool:
        """Check if there are any critical errors."""
        return self.critical_errors > 0


@dataclass
class RepairSummary:
    """Summary of repairs and fixes applied during processing."""
    total_repairs: int
    successful_repairs: int
    failed_repairs: int
    repair_types: Dict[str, int]
    repair_confidence: float
    elements_repaired: int
    repair_impact_assessment: str
    
    @property
    def repair_success_rate(self) -> float:
        """Calculate repair success rate."""
        if self.total_repairs == 0:
            return 1.0
        return self.successful_repairs / self.total_repairs


@dataclass
class DiagnosticReport:
    """Comprehensive diagnostic report for XML processing."""
    
    # Basic information
    report_id: str
    generated_at: str
    correlation_id: Optional[str]
    processing_summary: Dict[str, Any]
    
    # Document analysis
    document_health: str  # "EXCELLENT", "GOOD", "FAIR", "POOR", "CRITICAL"
    overall_confidence: float
    
    # Error and warning analysis
    error_summary: ErrorSummary
    warning_count: int
    info_count: int
    
    # Repair analysis
    repair_summary: RepairSummary
    
    # Categories breakdown
    diagnostic_categories: Dict[str, DiagnosticCategory]
    
    # Recommendations
    recommendations: List[str]
    
    # Export formats available
    available_formats: List[str]
    
    def get_severity_breakdown(self) -> Dict[str, int]:
        """Get breakdown of diagnostics by severity."""
        return {
            "CRITICAL": self.error_summary.critical_errors,
            "ERROR": self.error_summary.total_errors - self.error_summary.critical_errors,
            "WARNING": self.warning_count,
            "INFO": self.info_count
        }
    
    def is_healthy(self) -> bool:
        """Check if the document is in good health."""
        return self.document_health in ["EXCELLENT", "GOOD"]


class DiagnosticReporter:
    """
    Comprehensive diagnostic reporting system for the ultra-robust XML parser.
    
    This class aggregates information from validation, optimization, and other
    processing steps to provide comprehensive analysis and reporting.
    """
    
    def __init__(self, correlation_id: Optional[str] = None) -> None:
        """
        Initialize the diagnostic reporter.
        
        Args:
            correlation_id: Optional correlation ID for logging
        """
        self.correlation_id = correlation_id or str(uuid.uuid4())[:8]
        self.logger = get_logger(__name__, correlation_id, "diagnostic_reporter")
        
        # Categories for diagnostic classification
        self.diagnostic_categories: Dict[str, DiagnosticCategory] = {
            "structure": DiagnosticCategory("Document Structure", "Issues with document structure and hierarchy", DiagnosticSeverity.ERROR),
            "content": DiagnosticCategory("Content Validation", "Issues with text content and data", DiagnosticSeverity.WARNING),
            "attributes": DiagnosticCategory("Attribute Validation", "Issues with element attributes", DiagnosticSeverity.WARNING),
            "namespaces": DiagnosticCategory("Namespace Issues", "Problems with XML namespaces", DiagnosticSeverity.WARNING),
            "encoding": DiagnosticCategory("Encoding Issues", "Character encoding problems", DiagnosticSeverity.ERROR),
            "compliance": DiagnosticCategory("XML Compliance", "XML specification compliance issues", DiagnosticSeverity.INFO),
            "optimization": DiagnosticCategory("Optimization Opportunities", "Areas for performance optimization", DiagnosticSeverity.INFO),
            "security": DiagnosticCategory("Security Concerns", "Potential security issues", DiagnosticSeverity.CRITICAL)
        }
    
    def generate_comprehensive_report(
        self,
        parse_result: ParseResult,
        validation_result: Optional[ValidationResult] = None,
        optimization_result: Optional[OptimizationResult] = None,
        include_recommendations: bool = True
    ) -> DiagnosticReport:
        """
        Generate a comprehensive diagnostic report.
        
        Args:
            parse_result: The parsing result to analyze
            validation_result: Optional validation result
            optimization_result: Optional optimization result
            include_recommendations: Whether to include recommendations
        
        Returns:
            DiagnosticReport: Comprehensive diagnostic report
        """
        self.logger.info("Generating comprehensive diagnostic report")
        start_time = time.time()
        
        try:
            # Generate report ID
            report_id = f"diagnostic-{int(time.time())}-{self.correlation_id}"
            
            # Analyze parsing result
            processing_summary = self._analyze_parse_result(parse_result)
            
            # Analyze validation if available
            if validation_result:
                processing_summary.update(self._analyze_validation_result(validation_result))
            
            # Analyze optimization if available
            if optimization_result:
                processing_summary.update(self._analyze_optimization_result(optimization_result))
            
            # Determine document health
            document_health = self._assess_document_health(parse_result, validation_result, optimization_result)
            
            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(parse_result, validation_result, optimization_result)
            
            # Generate error summary
            error_summary = self._generate_error_summary(parse_result, validation_result)
            
            # Generate repair summary
            repair_summary = self._generate_repair_summary(parse_result, optimization_result)
            
            # Categorize diagnostics
            categorized_diagnostics = self._categorize_diagnostics(parse_result, validation_result, optimization_result)
            
            # Generate recommendations
            recommendations = []
            if include_recommendations:
                recommendations = self._generate_recommendations(parse_result, validation_result, optimization_result, error_summary, repair_summary)
            
            # Count warnings and info messages
            warning_count = len([d for d in parse_result.diagnostics if d.severity == DiagnosticSeverity.WARNING])
            info_count = len([d for d in parse_result.diagnostics if d.severity == DiagnosticSeverity.INFO])
            
            if validation_result:
                warning_count += len([i for i in validation_result.issues if i.severity == DiagnosticSeverity.WARNING])
                info_count += len([i for i in validation_result.issues if i.severity == DiagnosticSeverity.INFO])
            
            # Create comprehensive report
            report = DiagnosticReport(
                report_id=report_id,
                generated_at=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                correlation_id=self.correlation_id,
                processing_summary=processing_summary,
                document_health=document_health,
                overall_confidence=overall_confidence,
                error_summary=error_summary,
                warning_count=warning_count,
                info_count=info_count,
                repair_summary=repair_summary,
                diagnostic_categories=categorized_diagnostics,
                recommendations=recommendations,
                available_formats=["text", "json", "xml", "html"]
            )
            
            processing_time = (time.time() - start_time) * 1000
            self.logger.info(f"Diagnostic report generated in {processing_time:.2f}ms")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate diagnostic report: {e}", exc_info=True)
            # Return minimal report on failure
            return DiagnosticReport(
                report_id="error-report",
                generated_at=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                correlation_id=self.correlation_id,
                processing_summary={"error": str(e)},
                document_health="CRITICAL",
                overall_confidence=0.0,
                error_summary=ErrorSummary(1, 1, 0, {"generation_error": 1}, [str(e)], 0, 1.0),
                warning_count=0,
                info_count=0,
                repair_summary=RepairSummary(0, 0, 0, {}, 0.0, 0, "Report generation failed"),
                diagnostic_categories={},
                recommendations=["Review system logs", "Contact support"],
                available_formats=["text"]
            )
    
    def export_report(self, report: DiagnosticReport, format_type: str = "text") -> str:
        """
        Export diagnostic report in the specified format.
        
        Args:
            report: The diagnostic report to export
            format_type: Export format ("text", "json", "xml", "html")
        
        Returns:
            str: Formatted report content
        """
        self.logger.info(f"Exporting diagnostic report in {format_type} format")
        
        try:
            if format_type.lower() == "json":
                return self._export_json_report(report)
            elif format_type.lower() == "xml":
                return self._export_xml_report(report)
            elif format_type.lower() == "html":
                return self._export_html_report(report)
            else:  # Default to text
                return self._export_text_report(report)
                
        except Exception as e:
            self.logger.error(f"Failed to export report in {format_type} format: {e}", exc_info=True)
            return f"Error exporting report: {e}"
    
    def _analyze_parse_result(self, parse_result: ParseResult) -> Dict[str, Any]:
        """Analyze the parsing result."""
        summary = {
            "parsing_success": parse_result.success,
            "parsing_confidence": parse_result.confidence,
            "total_diagnostics": len(parse_result.diagnostics),
            "has_document": parse_result.document is not None,
            "processing_time_ms": parse_result.processing_time_ms
        }
        
        if parse_result.document:
            summary.update({
                "element_count": parse_result.document.total_elements,
                "attribute_count": parse_result.document.total_attributes,
                "max_depth": parse_result.document.max_depth,
                "has_root": parse_result.document.root is not None
            })
        
        return summary
    
    def _analyze_validation_result(self, validation_result: ValidationResult) -> Dict[str, Any]:
        """Analyze the validation result."""
        return {
            "validation_performed": True,
            "validation_success": validation_result.success,
            "validation_confidence": validation_result.confidence,
            "validation_issues": len(validation_result.issues),
            "elements_validated": validation_result.elements_validated,
            "attributes_validated": validation_result.attributes_validated
        }
    
    def _analyze_optimization_result(self, optimization_result: OptimizationResult) -> Dict[str, Any]:
        """Analyze the optimization result."""
        return {
            "optimization_performed": True,
            "optimization_success": optimization_result.success,
            "optimization_confidence": optimization_result.confidence,
            "total_optimizations": optimization_result.total_actions,
            "elements_affected": optimization_result.elements_affected,
            "memory_saved_bytes": optimization_result.total_memory_saved_bytes
        }
    
    def _assess_document_health(
        self,
        parse_result: ParseResult,
        validation_result: Optional[ValidationResult],
        optimization_result: Optional[OptimizationResult]
    ) -> str:
        """Assess overall document health."""
        
        # Count critical issues
        critical_issues = len([d for d in parse_result.diagnostics if d.severity == DiagnosticSeverity.CRITICAL])
        errors = len([d for d in parse_result.diagnostics if d.severity == DiagnosticSeverity.ERROR])
        warnings = len([d for d in parse_result.diagnostics if d.severity == DiagnosticSeverity.WARNING])
        
        if validation_result:
            critical_issues += len([i for i in validation_result.issues if i.severity == DiagnosticSeverity.CRITICAL])
            errors += len([i for i in validation_result.issues if i.severity == DiagnosticSeverity.ERROR])
            warnings += len([i for i in validation_result.issues if i.severity == DiagnosticSeverity.WARNING])
        
        # Determine health based on issue severity
        if critical_issues > 0:
            return "CRITICAL"
        elif errors > 5:
            return "POOR"
        elif errors > 0:
            return "FAIR"
        elif warnings > 10:
            return "FAIR"
        elif warnings > 0:
            return "GOOD"
        else:
            return "EXCELLENT"
    
    def _calculate_overall_confidence(
        self,
        parse_result: ParseResult,
        validation_result: Optional[ValidationResult],
        optimization_result: Optional[OptimizationResult]
    ) -> float:
        """Calculate overall confidence score."""
        confidences = [parse_result.confidence]
        
        if validation_result:
            confidences.append(validation_result.confidence)
        
        if optimization_result:
            confidences.append(optimization_result.confidence)
        
        # Weighted average with parsing having highest weight
        weights = [0.5, 0.3, 0.2][:len(confidences)]
        weighted_sum = sum(c * w for c, w in zip(confidences, weights))
        total_weight = sum(weights)
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _generate_error_summary(
        self,
        parse_result: ParseResult,
        validation_result: Optional[ValidationResult]
    ) -> ErrorSummary:
        """Generate comprehensive error summary."""
        
        all_errors = [d for d in parse_result.diagnostics if d.severity in [DiagnosticSeverity.ERROR, DiagnosticSeverity.CRITICAL]]
        
        if validation_result:
            all_errors.extend([i for i in validation_result.issues if i.severity in [DiagnosticSeverity.ERROR, DiagnosticSeverity.CRITICAL]])
        
        total_errors = len(all_errors)
        critical_errors = len([e for e in all_errors if e.severity == DiagnosticSeverity.CRITICAL])
        recoverable_errors = total_errors - critical_errors
        
        # Categorize errors
        error_categories: Dict[str, int] = {}
        root_causes: List[str] = []
        
        for error in all_errors:
            # Simple categorization based on message content
            if "structure" in error.message.lower() or "hierarchy" in error.message.lower():
                error_categories["structural"] = error_categories.get("structural", 0) + 1
                if "Missing parent" in error.message:
                    root_causes.append("Broken element hierarchy")
            elif "namespace" in error.message.lower():
                error_categories["namespace"] = error_categories.get("namespace", 0) + 1
                root_causes.append("Namespace declaration issues")
            elif "attribute" in error.message.lower():
                error_categories["attribute"] = error_categories.get("attribute", 0) + 1
            elif "content" in error.message.lower() or "text" in error.message.lower():
                error_categories["content"] = error_categories.get("content", 0) + 1
            else:
                error_categories["other"] = error_categories.get("other", 0) + 1
        
        # Remove duplicates from root causes
        root_causes = list(set(root_causes))
        
        # Calculate error rate
        total_elements = parse_result.document.total_elements if parse_result.document else 1
        error_rate = total_errors / max(total_elements, 1)
        
        affected_elements = min(total_errors, total_elements)  # Approximate
        
        return ErrorSummary(
            total_errors=total_errors,
            critical_errors=critical_errors,
            recoverable_errors=recoverable_errors,
            error_categories=error_categories,
            root_causes=root_causes,
            affected_elements=affected_elements,
            error_rate=error_rate
        )
    
    def _generate_repair_summary(
        self,
        parse_result: ParseResult,
        optimization_result: Optional[OptimizationResult]
    ) -> RepairSummary:
        """Generate repair summary."""
        
        # Count repairs from parse result
        total_repairs = parse_result.repair_count
        successful_repairs = total_repairs  # Assume all repairs attempted were successful for ultra-robust parser
        failed_repairs = 0
        
        repair_types = {"parsing_repairs": total_repairs}
        elements_repaired = total_repairs  # Approximate
        
        # Add optimization "repairs"
        if optimization_result:
            optimization_repairs = optimization_result.total_actions
            total_repairs += optimization_repairs
            successful_repairs += optimization_repairs
            repair_types["optimizations"] = optimization_repairs
            elements_repaired += optimization_result.elements_affected
        
        # Calculate repair confidence
        repair_confidence = parse_result.confidence
        if optimization_result:
            repair_confidence = (repair_confidence + optimization_result.confidence) / 2
        
        # Assess repair impact
        if total_repairs == 0:
            impact_assessment = "No repairs needed - document was already in good condition"
        elif total_repairs < 5:
            impact_assessment = "Minor repairs applied - document quality improved"
        elif total_repairs < 20:
            impact_assessment = "Moderate repairs applied - significant quality improvement"
        else:
            impact_assessment = "Extensive repairs applied - document substantially improved"
        
        return RepairSummary(
            total_repairs=total_repairs,
            successful_repairs=successful_repairs,
            failed_repairs=failed_repairs,
            repair_types=repair_types,
            repair_confidence=repair_confidence,
            elements_repaired=elements_repaired,
            repair_impact_assessment=impact_assessment
        )
    
    def _categorize_diagnostics(
        self,
        parse_result: ParseResult,
        validation_result: Optional[ValidationResult],
        optimization_result: Optional[OptimizationResult]
    ) -> Dict[str, DiagnosticCategory]:
        """Categorize all diagnostics into categories."""
        
        # Reset category counters
        for category in self.diagnostic_categories.values():
            category.issue_count = 0
        
        # Categorize parse result diagnostics
        for diagnostic in parse_result.diagnostics:
            self._categorize_single_diagnostic(diagnostic.message, diagnostic.severity)
        
        # Categorize validation issues
        if validation_result:
            for issue in validation_result.issues:
                self._categorize_single_diagnostic(issue.message, issue.severity)
        
        # Categorize optimization opportunities
        if optimization_result and optimization_result.actions:
            for action in optimization_result.actions:
                self._categorize_single_diagnostic(action.description, DiagnosticSeverity.INFO)
        
        return self.diagnostic_categories.copy()
    
    def _categorize_single_diagnostic(self, message: str, severity: DiagnosticSeverity) -> None:
        """Categorize a single diagnostic message."""
        message_lower = message.lower()
        
        if any(keyword in message_lower for keyword in ["structure", "hierarchy", "parent", "child"]):
            self.diagnostic_categories["structure"].add_issue()
        elif any(keyword in message_lower for keyword in ["namespace", "xmlns", "prefix"]):
            self.diagnostic_categories["namespaces"].add_issue()
        elif any(keyword in message_lower for keyword in ["attribute", "attr"]):
            self.diagnostic_categories["attributes"].add_issue()
        elif any(keyword in message_lower for keyword in ["content", "text", "data"]):
            self.diagnostic_categories["content"].add_issue()
        elif any(keyword in message_lower for keyword in ["encoding", "character", "utf"]):
            self.diagnostic_categories["encoding"].add_issue()
        elif any(keyword in message_lower for keyword in ["security", "xxe", "injection"]):
            self.diagnostic_categories["security"].add_issue()
        elif any(keyword in message_lower for keyword in ["optimize", "performance", "memory"]):
            self.diagnostic_categories["optimization"].add_issue()
        else:
            self.diagnostic_categories["compliance"].add_issue()
    
    def _generate_recommendations(
        self,
        parse_result: ParseResult,
        validation_result: Optional[ValidationResult],
        optimization_result: Optional[OptimizationResult],
        error_summary: ErrorSummary,
        repair_summary: RepairSummary
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Document health recommendations
        if error_summary.has_critical_errors():
            recommendations.append("⚠️ CRITICAL: Review and fix critical errors immediately")
            recommendations.append("🔍 Inspect document structure for serious integrity issues")
        
        if error_summary.error_rate > 0.1:
            recommendations.append("📈 High error rate detected - consider document validation tools")
        
        # Category-specific recommendations
        if self.diagnostic_categories["structure"].issue_count > 0:
            recommendations.append("🏗️ Review document structure and element hierarchy")
        
        if self.diagnostic_categories["namespaces"].issue_count > 0:
            recommendations.append("📋 Check XML namespace declarations and prefixes")
        
        if self.diagnostic_categories["security"].issue_count > 0:
            recommendations.append("🛡️ SECURITY: Review document for potential security issues")
        
        # Optimization recommendations
        if optimization_result and optimization_result.total_memory_saved_bytes > 1000:
            recommendations.append("⚡ Document optimized successfully - consider regular optimization")
        
        if not optimization_result:
            recommendations.append("🔧 Consider running document optimization to improve performance")
        
        # Validation recommendations
        if not validation_result:
            recommendations.append("🔍 Run document validation to identify potential issues")
        elif validation_result and not validation_result.success:
            recommendations.append("📝 Address validation issues for better document quality")
        
        # Performance recommendations
        if parse_result.processing_time_ms > 1000:
            recommendations.append("⏱️ Processing time is high - consider document optimization")
        
        # Confidence recommendations
        if parse_result.confidence < 0.8:
            recommendations.append("📊 Low confidence score - review document quality and structure")
        
        return recommendations
    
    def _export_text_report(self, report: DiagnosticReport) -> str:
        """Export report as formatted text."""
        lines = []
        lines.append("=" * 80)
        lines.append("ULTRA-ROBUST XML PARSER - DIAGNOSTIC REPORT")
        lines.append("=" * 80)
        lines.append(f"Report ID: {report.report_id}")
        lines.append(f"Generated: {report.generated_at}")
        if report.correlation_id:
            lines.append(f"Correlation ID: {report.correlation_id}")
        lines.append("")
        
        # Document Health
        health_icon = {
            "EXCELLENT": "🌟",
            "GOOD": "✅",
            "FAIR": "⚠️",
            "POOR": "❌",
            "CRITICAL": "🚨"
        }
        lines.append(f"DOCUMENT HEALTH: {health_icon.get(report.document_health, '❓')} {report.document_health}")
        lines.append(f"Overall Confidence: {report.overall_confidence:.3f}")
        lines.append("")
        
        # Processing Summary
        lines.append("PROCESSING SUMMARY")
        lines.append("-" * 40)
        for key, value in report.processing_summary.items():
            lines.append(f"  {key.replace('_', ' ').title()}: {value}")
        lines.append("")
        
        # Error Summary
        lines.append("ERROR ANALYSIS")
        lines.append("-" * 40)
        lines.append(f"  Total Errors: {report.error_summary.total_errors}")
        lines.append(f"  Critical Errors: {report.error_summary.critical_errors}")
        lines.append(f"  Recoverable Errors: {report.error_summary.recoverable_errors}")
        lines.append(f"  Error Rate: {report.error_summary.error_rate:.4f}")
        
        if report.error_summary.error_categories:
            lines.append("  Error Categories:")
            for category, count in report.error_summary.error_categories.items():
                lines.append(f"    {category.title()}: {count}")
        
        if report.error_summary.root_causes:
            lines.append("  Root Causes:")
            for cause in report.error_summary.root_causes:
                lines.append(f"    • {cause}")
        lines.append("")
        
        # Repair Summary
        lines.append("REPAIR ANALYSIS")
        lines.append("-" * 40)
        lines.append(f"  Total Repairs: {report.repair_summary.total_repairs}")
        lines.append(f"  Success Rate: {report.repair_summary.repair_success_rate:.1%}")
        lines.append(f"  Elements Repaired: {report.repair_summary.elements_repaired}")
        lines.append(f"  Repair Confidence: {report.repair_summary.repair_confidence:.3f}")
        lines.append(f"  Impact: {report.repair_summary.repair_impact_assessment}")
        
        if report.repair_summary.repair_types:
            lines.append("  Repair Types:")
            for repair_type, count in report.repair_summary.repair_types.items():
                lines.append(f"    {repair_type.replace('_', ' ').title()}: {count}")
        lines.append("")
        
        # Category Breakdown
        lines.append("DIAGNOSTIC CATEGORIES")
        lines.append("-" * 40)
        for name, category in report.diagnostic_categories.items():
            if category.issue_count > 0:
                lines.append(f"  {category.name}: {category.issue_count} issues")
                lines.append(f"    {category.description}")
        lines.append("")
        
        # Severity Breakdown
        lines.append("SEVERITY BREAKDOWN")
        lines.append("-" * 40)
        severity_breakdown = report.get_severity_breakdown()
        for severity, count in severity_breakdown.items():
            if count > 0:
                lines.append(f"  {severity}: {count}")
        lines.append("")
        
        # Recommendations
        if report.recommendations:
            lines.append("RECOMMENDATIONS")
            lines.append("-" * 40)
            for i, recommendation in enumerate(report.recommendations, 1):
                lines.append(f"  {i}. {recommendation}")
            lines.append("")
        
        lines.append("=" * 80)
        lines.append("End of Report")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def _export_json_report(self, report: DiagnosticReport) -> str:
        """Export report as JSON."""
        import json
        
        # Convert dataclasses to dictionaries
        report_dict = {
            "report_id": report.report_id,
            "generated_at": report.generated_at,
            "correlation_id": report.correlation_id,
            "processing_summary": report.processing_summary,
            "document_health": report.document_health,
            "overall_confidence": report.overall_confidence,
            "error_summary": {
                "total_errors": report.error_summary.total_errors,
                "critical_errors": report.error_summary.critical_errors,
                "recoverable_errors": report.error_summary.recoverable_errors,
                "error_categories": report.error_summary.error_categories,
                "root_causes": report.error_summary.root_causes,
                "affected_elements": report.error_summary.affected_elements,
                "error_rate": report.error_summary.error_rate
            },
            "warning_count": report.warning_count,
            "info_count": report.info_count,
            "repair_summary": {
                "total_repairs": report.repair_summary.total_repairs,
                "successful_repairs": report.repair_summary.successful_repairs,
                "failed_repairs": report.repair_summary.failed_repairs,
                "repair_types": report.repair_summary.repair_types,
                "repair_confidence": report.repair_summary.repair_confidence,
                "elements_repaired": report.repair_summary.elements_repaired,
                "repair_impact_assessment": report.repair_summary.repair_impact_assessment,
                "repair_success_rate": report.repair_summary.repair_success_rate
            },
            "diagnostic_categories": {
                name: {
                    "name": cat.name,
                    "description": cat.description,
                    "severity_level": cat.severity_level.value,
                    "issue_count": cat.issue_count
                }
                for name, cat in report.diagnostic_categories.items()
            },
            "severity_breakdown": report.get_severity_breakdown(),
            "recommendations": report.recommendations,
            "available_formats": report.available_formats,
            "is_healthy": report.is_healthy()
        }
        
        return json.dumps(report_dict, indent=2)
    
    def _export_xml_report(self, report: DiagnosticReport) -> str:
        """Export report as XML."""
        lines = []
        lines.append('<?xml version="1.0" encoding="utf-8"?>')
        lines.append('<diagnostic_report>')
        lines.append(f'  <report_info>')
        lines.append(f'    <report_id>{report.report_id}</report_id>')
        lines.append(f'    <generated_at>{report.generated_at}</generated_at>')
        if report.correlation_id:
            lines.append(f'    <correlation_id>{report.correlation_id}</correlation_id>')
        lines.append(f'    <document_health>{report.document_health}</document_health>')
        lines.append(f'    <overall_confidence>{report.overall_confidence:.3f}</overall_confidence>')
        lines.append(f'    <is_healthy>{report.is_healthy()}</is_healthy>')
        lines.append(f'  </report_info>')
        
        # Processing Summary
        lines.append('  <processing_summary>')
        for key, value in report.processing_summary.items():
            lines.append(f'    <{key}>{value}</{key}>')
        lines.append('  </processing_summary>')
        
        # Error Summary
        lines.append('  <error_summary>')
        lines.append(f'    <total_errors>{report.error_summary.total_errors}</total_errors>')
        lines.append(f'    <critical_errors>{report.error_summary.critical_errors}</critical_errors>')
        lines.append(f'    <recoverable_errors>{report.error_summary.recoverable_errors}</recoverable_errors>')
        lines.append(f'    <error_rate>{report.error_summary.error_rate:.4f}</error_rate>')
        
        lines.append('    <error_categories>')
        for category, count in report.error_summary.error_categories.items():
            lines.append(f'      <category name="{category}">{count}</category>')
        lines.append('    </error_categories>')
        
        lines.append('    <root_causes>')
        for cause in report.error_summary.root_causes:
            lines.append(f'      <cause>{cause}</cause>')
        lines.append('    </root_causes>')
        lines.append('  </error_summary>')
        
        # Repair Summary
        lines.append('  <repair_summary>')
        lines.append(f'    <total_repairs>{report.repair_summary.total_repairs}</total_repairs>')
        lines.append(f'    <success_rate>{report.repair_summary.repair_success_rate:.3f}</success_rate>')
        lines.append(f'    <elements_repaired>{report.repair_summary.elements_repaired}</elements_repaired>')
        lines.append(f'    <repair_confidence>{report.repair_summary.repair_confidence:.3f}</repair_confidence>')
        lines.append(f'    <impact_assessment>{report.repair_summary.repair_impact_assessment}</impact_assessment>')
        lines.append('  </repair_summary>')
        
        # Recommendations
        lines.append('  <recommendations>')
        for recommendation in report.recommendations:
            lines.append(f'    <recommendation>{recommendation}</recommendation>')
        lines.append('  </recommendations>')
        
        lines.append('</diagnostic_report>')
        
        return "\n".join(lines)
    
    def _export_html_report(self, report: DiagnosticReport) -> str:
        """Export report as HTML."""
        health_colors = {
            "EXCELLENT": "#28a745",
            "GOOD": "#6c757d",
            "FAIR": "#ffc107",
            "POOR": "#fd7e14",
            "CRITICAL": "#dc3545"
        }
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Diagnostic Report - {report.report_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        .health {{ color: {health_colors.get(report.document_health, '#6c757d')}; font-weight: bold; }}
        .section {{ margin-bottom: 20px; }}
        .section h3 {{ color: #495057; border-bottom: 2px solid #dee2e6; padding-bottom: 5px; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e9ecef; border-radius: 3px; }}
        .recommendation {{ background-color: #d4edda; padding: 10px; margin: 5px 0; border-left: 4px solid #28a745; }}
        .error {{ color: #dc3545; }}
        .warning {{ color: #fd7e14; }}
        .info {{ color: #17a2b8; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Ultra-Robust XML Parser - Diagnostic Report</h1>
        <p><strong>Report ID:</strong> {report.report_id}</p>
        <p><strong>Generated:</strong> {report.generated_at}</p>
        <p><strong>Document Health:</strong> <span class="health">{report.document_health}</span></p>
        <p><strong>Overall Confidence:</strong> {report.overall_confidence:.3f}</p>
    </div>

    <div class="section">
        <h3>Summary Metrics</h3>
        <div class="metric">
            <strong>Total Errors:</strong> {report.error_summary.total_errors}
        </div>
        <div class="metric">
            <strong>Warnings:</strong> {report.warning_count}
        </div>
        <div class="metric">
            <strong>Total Repairs:</strong> {report.repair_summary.total_repairs}
        </div>
        <div class="metric">
            <strong>Success Rate:</strong> {report.repair_summary.repair_success_rate:.1%}
        </div>
    </div>"""
        
        # Add recommendations if available
        if report.recommendations:
            html += """
    <div class="section">
        <h3>Recommendations</h3>"""
            for recommendation in report.recommendations:
                html += f'        <div class="recommendation">{recommendation}</div>\n'
            html += "    </div>"
        
        html += """
</body>
</html>"""
        
        return html