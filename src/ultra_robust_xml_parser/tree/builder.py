"""Core tree building implementation for ultra-robust XML parsing.

This module implements the tree building engine that converts token streams into
hierarchical XML document structures with comprehensive error recovery and
never-fail guarantees.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from ultra_robust_xml_parser.shared import (
    DiagnosticEntry,
    DiagnosticSeverity,
    PerformanceMetrics,
    get_logger,
)
from ultra_robust_xml_parser.tokenization import (
    Token,
    TokenizationResult,
    TokenType,
)

# Constants for magic values used in well-formed checks
_WELL_FORMED_CONFIDENCE_THRESHOLD = 0.9
_REPAIR_CONFIDENCE_IMPACT_PER_REPAIR = 0.05
_REPAIR_CONFIDENCE_IMPACT_PER_ELEMENT = 0.02


@dataclass
class StructureRepair:
    """Information about structural repairs made during tree building."""

    repair_type: str
    description: str
    original_tokens: List[Token]
    confidence_impact: float
    severity: str = "minor"

    def __post_init__(self) -> None:
        """Validate repair information."""
        if not self.repair_type:
            raise ValueError("Repair type cannot be empty")
        if not self.description:
            raise ValueError("Repair description cannot be empty")
        if not (0.0 <= self.confidence_impact <= 1.0):
            raise ValueError("Confidence impact must be between 0.0 and 1.0")


@dataclass(eq=False)
class XMLElement:
    """Represents a single XML element in the document tree.

    Provides comprehensive element functionality including attributes, child
    management, text content, and tree navigation capabilities.
    """

    tag: str
    attributes: Dict[str, str] = field(default_factory=dict)
    text: Optional[str] = None
    children: List["XMLElement"] = field(default_factory=list)
    confidence: float = 1.0
    repairs: List[StructureRepair] = field(default_factory=list)
    parent: Optional["XMLElement"] = None
    preserve_structure: bool = False  # Prevents self-closing when content was removed

    # Source information for debugging and diagnostics
    source_tokens: List[Token] = field(default_factory=list)
    start_position: Optional[Dict[str, int]] = None
    end_position: Optional[Dict[str, int]] = None

    def __post_init__(self) -> None:
        """Validate element values and establish parent-child relationships."""
        if not self.tag:
            raise ValueError("Element tag cannot be empty")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")

        # Establish parent relationships for existing children
        for child in self.children:
            child.parent = self

    @property
    def has_repairs(self) -> bool:
        """Check if this element has any structural repairs."""
        return len(self.repairs) > 0

    @property
    def is_well_formed(self) -> bool:
        """Check if element is well-formed (no repairs, high confidence)."""
        return (
            not self.has_repairs
            and self.confidence >= _WELL_FORMED_CONFIDENCE_THRESHOLD
        )

    @property
    def full_text(self) -> str:
        """Get all text content including from child elements."""
        text_parts = []
        if self.text:
            text_parts.append(self.text)

        for child in self.children:
            child_text = child.full_text
            if child_text:
                text_parts.append(child_text)

        return " ".join(text_parts).strip()

    @property
    def tag_with_namespace(self) -> str:
        """Get tag name including namespace prefix if present."""
        return self.tag

    @property
    def local_name(self) -> str:
        """Get local tag name without namespace prefix."""
        if ":" in self.tag:
            return self.tag.split(":", 1)[1]
        return self.tag

    @property
    def namespace_prefix(self) -> Optional[str]:
        """Get namespace prefix if present."""
        if ":" in self.tag:
            return self.tag.split(":", 1)[0]
        return None

    def add_child(self, child: "XMLElement") -> None:
        """Add a child element and establish parent relationship."""
        if not isinstance(child, XMLElement):
            raise TypeError("Child must be an XMLElement instance")

        child.parent = self
        self.children.append(child)

    def remove_child(self, child: "XMLElement") -> bool:
        """Remove a child element and clear parent relationship."""
        if child in self.children:
            child.parent = None
            self.children.remove(child)
            return True
        return False

    def insert_child(self, index: int, child: "XMLElement") -> None:
        """Insert child element at specific index."""
        if not isinstance(child, XMLElement):
            raise TypeError("Child must be an XMLElement instance")
        if not (0 <= index <= len(self.children)):
            raise IndexError("Child index out of range")

        child.parent = self
        self.children.insert(index, child)

    def find_child(self, tag: str) -> Optional["XMLElement"]:
        """Find first direct child with matching tag name."""
        for child in self.children:
            if child.tag == tag:
                return child
        return None

    def find_children(self, tag: str) -> List["XMLElement"]:
        """Find all direct children with matching tag name."""
        return [child for child in self.children if child.tag == tag]

    def find(self, tag: str) -> Optional["XMLElement"]:
        """Find first descendant element with matching tag name."""
        # Check direct children first
        for child in self.children:
            if child.tag == tag:
                return child

        # Recursively search in child elements
        for child in self.children:
            found = child.find(tag)
            if found:
                return found

        return None

    def find_all(self, tag: str) -> List["XMLElement"]:
        """Find all descendant elements with matching tag name."""
        results = []

        # Check direct children
        for child in self.children:
            if child.tag == tag:
                results.append(child)
            # Recursively search child elements
            results.extend(child.find_all(tag))

        return results

    def find_by_attribute(
        self, name: str, value: Optional[str] = None
    ) -> List["XMLElement"]:
        """Find elements by attribute name and optionally value."""
        results = []

        # Check this element
        if name in self.attributes and (
            value is None or self.attributes[name] == value
        ):
            results.append(self)

        # Recursively search children
        for child in self.children:
            results.extend(child.find_by_attribute(name, value))

        return results

    def get_attribute(self, name: str, default: Optional[str] = None) -> Optional[str]:
        """Get attribute value with optional default."""
        return self.attributes.get(name, default)

    def set_attribute(self, name: str, value: str) -> None:
        """Set attribute value."""
        if not isinstance(name, str) or not isinstance(value, str):
            raise TypeError("Attribute name and value must be strings")
        self.attributes[name] = value

    def has_attribute(self, name: str) -> bool:
        """Check if element has specific attribute."""
        return name in self.attributes

    def get_path(self) -> str:
        """Get XPath-like path to this element."""
        if self.parent is None:
            return f"/{self.tag}"

        parent_path = self.parent.get_path()
        # Find position among siblings with same tag
        siblings = [child for child in self.parent.children if child.tag == self.tag]
        if len(siblings) > 1:
            try:
                position = siblings.index(self) + 1
            except ValueError:
                # Fallback if not found in siblings list
                position = 1
            return f"{parent_path}/{self.tag}[{position}]"

        return f"{parent_path}/{self.tag}"

    def get_depth(self) -> int:
        """Get depth of this element in the tree (root = 0)."""
        if self.parent is None:
            return 0
        return self.parent.get_depth() + 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert element to dictionary representation."""
        result: Dict[str, Any] = {
            "tag": self.tag,
            "attributes": dict(self.attributes),
            "confidence": self.confidence,
            "has_repairs": self.has_repairs,
        }

        if self.text:
            result["text"] = self.text

        if self.children:
            result["children"] = [child.to_dict() for child in self.children]

        return result


@dataclass
class XMLDocument:
    """Root XML document container with metadata and navigation.

    Represents the complete XML document tree with root element, metadata,
    and comprehensive document-level operations.
    """

    root: Optional[XMLElement] = None
    encoding: str = "utf-8"
    version: str = "1.0"
    standalone: Optional[bool] = None
    confidence: float = 1.0
    repairs: List[StructureRepair] = field(default_factory=list)

    # Document metadata
    processing_time_ms: float = 0.0
    total_elements: int = 0
    total_attributes: int = 0
    max_depth: int = 0

    # Source information
    source_tokens: List[Token] = field(default_factory=list)
    correlation_id: Optional[str] = None

    def __post_init__(self) -> None:
        """Calculate document statistics and validate structure."""
        if self.root:
            self._calculate_statistics()

        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")

    def _calculate_statistics(self) -> None:
        """Calculate document-wide statistics."""
        if not self.root:
            return

        elements = list(self.iter_elements())
        self.total_elements = len(elements)
        self.total_attributes = sum(len(elem.attributes) for elem in elements)
        self.max_depth = max((elem.get_depth() for elem in elements), default=0)

    @property
    def has_repairs(self) -> bool:
        """Check if document has any structural repairs."""
        return len(self.repairs) > 0

    @property
    def is_well_formed(self) -> bool:
        """Check if document is well-formed (no repairs, high confidence)."""
        return (
            not self.has_repairs
            and self.confidence >= _WELL_FORMED_CONFIDENCE_THRESHOLD
        )

    def iter_elements(self) -> List[XMLElement]:
        """Iterate over all elements in document order."""
        if not self.root:
            return []

        elements = []

        def collect_elements(element: XMLElement) -> None:
            elements.append(element)
            for child in element.children:
                collect_elements(child)

        collect_elements(self.root)
        return elements

    def find(self, tag: str) -> Optional[XMLElement]:
        """Find first element with matching tag name."""
        if not self.root:
            return None

        if self.root.tag == tag:
            return self.root

        return self.root.find(tag)

    def find_all(self, tag: str) -> List[XMLElement]:
        """Find all elements with matching tag name."""
        if not self.root:
            return []

        results = []
        if self.root.tag == tag:
            results.append(self.root)

        results.extend(self.root.find_all(tag))
        return results

    def find_by_attribute(
        self, name: str, value: Optional[str] = None
    ) -> List[XMLElement]:
        """Find elements by attribute name and optionally value."""
        if not self.root:
            return []

        return self.root.find_by_attribute(name, value)

    def get_element_by_id(self, id_value: str) -> Optional[XMLElement]:
        """Find element by ID attribute value."""
        return next(
            (elem for elem in self.find_by_attribute("id", id_value)),
            None
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary representation."""
        result: Dict[str, Any] = {
            "encoding": self.encoding,
            "version": self.version,
            "confidence": self.confidence,
            "has_repairs": self.has_repairs,
            "total_elements": self.total_elements,
            "total_attributes": self.total_attributes,
            "max_depth": self.max_depth,
        }

        if self.standalone is not None:
            result["standalone"] = self.standalone

        if self.root:
            result["root"] = self.root.to_dict()

        return result


@dataclass
class ParseResult:
    """Comprehensive result object for tree building operations.

    Contains the document tree, comprehensive metadata, diagnostics, and
    performance information following the never-fail philosophy.
    """

    # Core results
    document: XMLDocument = field(default_factory=XMLDocument)
    success: bool = True
    confidence: float = 1.0

    # Metadata and diagnostics
    diagnostics: List[DiagnosticEntry] = field(default_factory=list)
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)

    # Source information
    tokenization_result: Optional[TokenizationResult] = None
    correlation_id: Optional[str] = None
    
    # Validation and optimization results (enhanced for Story 3.4)
    validation_result: Optional[Any] = None  # ValidationResult - using Any to avoid circular import
    optimization_result: Optional[Any] = None  # OptimizationResult - using Any to avoid circular import

    def __post_init__(self) -> None:
        """Calculate derived statistics and validate result."""
        # Only auto-calculate confidence if it's still at the default value
        if self.confidence == 1.0 and self.document and self.document.root:
            self._update_confidence()

        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")

    def _update_confidence(self) -> None:
        """Calculate overall confidence based on document quality."""
        if not self.document or not self.document.root:
            self.confidence = 0.0
            return

        # Weight confidence based on document characteristics
        doc_confidence = self.document.confidence
        repair_impact = (
            len(self.document.repairs) * _REPAIR_CONFIDENCE_IMPACT_PER_REPAIR
        )

        # Consider element-level repairs
        elements = self.document.iter_elements()
        element_repairs = sum(len(elem.repairs) for elem in elements)
        element_impact = element_repairs * _REPAIR_CONFIDENCE_IMPACT_PER_ELEMENT

        # Calculate final confidence
        self.confidence = max(0.0, doc_confidence - repair_impact - element_impact)

    @property
    def element_count(self) -> int:
        """Get total number of elements in document."""
        return self.document.total_elements if self.document else 0

    @property
    def has_repairs(self) -> bool:
        """Check if result contains any structural repairs."""
        if not self.document:
            return False

        doc_repairs = self.document.has_repairs
        element_repairs = any(
            elem.has_repairs for elem in self.document.iter_elements()
        )

        return doc_repairs or element_repairs

    @property
    def repair_count(self) -> int:
        """Get total number of repairs applied."""
        if not self.document:
            return 0

        doc_repairs = len(self.document.repairs)
        element_repairs = sum(
            len(elem.repairs) for elem in self.document.iter_elements()
        )

        return doc_repairs + element_repairs
    
    @property
    def processing_time_ms(self) -> float:
        """Get processing time in milliseconds."""
        return self.performance.processing_time_ms

    def add_diagnostic(
        self,
        severity: DiagnosticSeverity,
        message: str,
        component: str,
        position: Optional[Dict[str, int]] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add diagnostic entry to result."""
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
        """Get diagnostics of specific severity level."""
        return [diag for diag in self.diagnostics if diag.severity == severity]

    def has_errors(self) -> bool:
        """Check if result contains any error diagnostics."""
        return any(
            diag.severity in (DiagnosticSeverity.ERROR, DiagnosticSeverity.CRITICAL)
            for diag in self.diagnostics
        )

    @property
    def parsing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive parsing statistics."""
        stats = {
            "element_count": self.element_count,
            "attribute_count": self.document.total_attributes if self.document else 0,
            "max_depth": self.document.max_depth if self.document else 0,
            "repair_count": self.repair_count,
            "has_repairs": self.has_repairs,
            "processing_time_ms": self.performance.processing_time_ms,
            "memory_used_bytes": self.performance.memory_used_bytes,
            "characters_processed": self.performance.characters_processed,
            "tokens_generated": self.performance.tokens_generated,
        }
        
        # Add tokenization statistics if available
        if self.tokenization_result and hasattr(self.tokenization_result, 'metadata'):
            tokenization_metadata = self.tokenization_result.metadata
            stats.update({
                "tokenization_error_rate": tokenization_metadata.error_rate,
                "tokenization_repair_rate": tokenization_metadata.repair_rate,
                "tokenization_recovery_rate": tokenization_metadata.recovery_rate,
                "fast_path_used": tokenization_metadata.fast_path_used,
            })
        
        return stats
    
    @property
    def validation_summary(self) -> Dict[str, Any]:
        """Get validation summary if validation was performed."""
        if not self.validation_result:
            return {"validation_performed": False}
        
        return {
            "validation_performed": True,
            "validation_success": self.validation_result.success,
            "validation_confidence": self.validation_result.confidence,
            "validation_error_count": self.validation_result.error_count,
            "validation_warning_count": self.validation_result.warning_count,
            "elements_validated": self.validation_result.elements_validated,
            "attributes_validated": self.validation_result.attributes_validated,
            "validation_level": self.validation_result.validation_level.name if hasattr(self.validation_result.validation_level, 'name') else str(self.validation_result.validation_level),
            "rules_checked": len(self.validation_result.rules_checked),
        }
    
    @property
    def optimization_summary(self) -> Dict[str, Any]:
        """Get optimization summary if optimization was performed."""
        if not self.optimization_result:
            return {"optimization_performed": False}
        
        return {
            "optimization_performed": True,
            "optimization_success": self.optimization_result.success,
            "optimization_confidence": self.optimization_result.confidence,
            "total_optimizations": self.optimization_result.total_actions,
            "elements_affected": self.optimization_result.elements_affected,
            "memory_saved_bytes": self.optimization_result.total_memory_saved_bytes,
            "elements_processed": self.optimization_result.total_elements_processed,
        }
    
    @property
    def overall_confidence_breakdown(self) -> Dict[str, float]:
        """Get detailed confidence breakdown from all processing stages."""
        breakdown = {
            "base_confidence": self.confidence,
            "document_confidence": self.document.confidence if self.document else 0.0,
        }
        
        # Add tokenization confidence if available
        if self.tokenization_result and hasattr(self.tokenization_result, 'confidence'):
            breakdown["tokenization_confidence"] = self.tokenization_result.confidence
        elif self.tokenization_result and hasattr(self.tokenization_result, 'metadata'):
            breakdown["tokenization_confidence"] = self.tokenization_result.metadata.overall_confidence
        
        # Add validation confidence if available
        if self.validation_result:
            breakdown["validation_confidence"] = self.validation_result.confidence
        
        # Add optimization confidence if available
        if self.optimization_result:
            breakdown["optimization_confidence"] = self.optimization_result.confidence
        
        # Calculate overall confidence as minimum of all stages
        all_confidences = [v for v in breakdown.values() if v > 0]
        if all_confidences:
            breakdown["overall_confidence"] = min(all_confidences)
        else:
            breakdown["overall_confidence"] = 0.0
        
        return breakdown
    
    def get_diagnostics_summary(self) -> Dict[str, Any]:
        """Get comprehensive diagnostics summary."""
        summary = {
            "total_diagnostics": len(self.diagnostics),
            "has_errors": self.has_errors(),
            "diagnostics_by_severity": {},
            "diagnostics_by_component": {},
        }
        
        # Count by severity
        from ultra_robust_xml_parser.shared import DiagnosticSeverity
        for severity in DiagnosticSeverity:
            count = len(self.get_diagnostics_by_severity(severity))
            if count > 0:
                summary["diagnostics_by_severity"][severity.name] = count
        
        # Count by component
        for diagnostic in self.diagnostics:
            component = diagnostic.component
            summary["diagnostics_by_component"][component] = (
                summary["diagnostics_by_component"].get(component, 0) + 1
            )
        
        # Add validation diagnostics if available
        if self.validation_result and hasattr(self.validation_result, 'issues'):
            summary["validation_issues"] = len(self.validation_result.issues)
            summary["validation_issues_by_type"] = {}
            for issue in self.validation_result.issues:
                issue_type = issue.issue_type.name if hasattr(issue.issue_type, 'name') else str(issue.issue_type)
                summary["validation_issues_by_type"][issue_type] = (
                    summary["validation_issues_by_type"].get(issue_type, 0) + 1
                )
        
        return summary
    
    def get_repair_summary(self) -> Dict[str, Any]:
        """Get comprehensive repair summary."""
        summary = {
            "total_repairs": self.repair_count,
            "document_repairs": len(self.document.repairs) if self.document else 0,
            "element_repairs": 0,
            "repair_types": {},
            "repair_impact_on_confidence": 0.0,
        }
        
        if self.document:
            # Count document repairs and categorize repair types
            for repair in self.document.repairs:
                repair_type = repair.repair_type
                summary["repair_types"][repair_type] = (
                    summary["repair_types"].get(repair_type, 0) + 1
                )
                summary["repair_impact_on_confidence"] += repair.confidence_impact
            
            # Count element repairs and categorize repair types
            for element in self.document.iter_elements():
                summary["element_repairs"] += len(element.repairs)
                for repair in element.repairs:
                    repair_type = repair.repair_type
                    summary["repair_types"][repair_type] = (
                        summary["repair_types"].get(repair_type, 0) + 1
                    )
                    summary["repair_impact_on_confidence"] += repair.confidence_impact
        
        return summary
    
    def summary(self) -> Dict[str, Any]:
        """Get comprehensive summary statistics for the parse result."""
        summary = {
            # Core results
            "success": self.success,
            "confidence": self.confidence,
            "document_well_formed": (
                self.document.is_well_formed if self.document else False
            ),
            
            # Statistics
            "parsing_statistics": self.parsing_statistics,
            "diagnostics_summary": self.get_diagnostics_summary(),
            "repair_summary": self.get_repair_summary(),
            "confidence_breakdown": self.overall_confidence_breakdown,
            
            # Processing stage results
            "validation": self.validation_summary,
            "optimization": self.optimization_summary,
        }
        
        return summary
    
    # Intuitive API properties for progressive disclosure (Story 4.1)
    
    @property
    def tree(self) -> XMLDocument:
        """Direct access to the XMLDocument tree for intuitive navigation.
        
        Returns:
            XMLDocument containing the parsed tree structure
            
        Examples:
            Accessing root element:
            >>> result = parse('<root><item>value</item></root>')
            >>> result.tree.root.tag
            'root'
            
            Finding elements:
            >>> item = result.tree.find('item')
            >>> item.text
            'value'
        """
        return self.document
    
    @property 
    def metadata(self) -> Dict[str, Any]:
        """Comprehensive parsing metadata for detailed analysis.
        
        Returns:
            Dictionary containing all parsing metadata, statistics, and diagnostics
            
        Examples:
            Parsing statistics:
            >>> result = parse('<root><item/></root>')
            >>> result.metadata['element_count']
            2
            
            Performance information:
            >>> result.metadata['processing_time_ms'] > 0
            True
            
            Confidence breakdown:
            >>> 'confidence_breakdown' in result.metadata
            True
        """
        metadata = {
            # Basic information
            "element_count": self.element_count,
            "repair_count": self.repair_count,
            "has_repairs": self.has_repairs,
            "processing_time_ms": self.processing_time_ms,
            
            # Comprehensive statistics
            "parsing_statistics": self.parsing_statistics,
            "confidence_breakdown": self.overall_confidence_breakdown,
            "diagnostics_summary": self.get_diagnostics_summary(),
            "repair_summary": self.get_repair_summary(),
            
            # Processing stage results
            "validation": self.validation_summary,
            "optimization": self.optimization_summary,
            
            # Source information
            "correlation_id": self.correlation_id,
            "has_tokenization_result": self.tokenization_result is not None,
        }
        
        # Add tokenization metadata if available
        if self.tokenization_result and hasattr(self.tokenization_result, 'metadata'):
            metadata["tokenization"] = {
                "error_rate": self.tokenization_result.metadata.error_rate,
                "repair_rate": self.tokenization_result.metadata.repair_rate,
                "recovery_rate": self.tokenization_result.metadata.recovery_rate,
                "fast_path_used": self.tokenization_result.metadata.fast_path_used,
                "confidence": self.tokenization_result.confidence,
            }
        
        return metadata


class XMLTreeBuilder:
    """Main tree builder for constructing XML document trees from token streams.

    Provides robust tree construction with comprehensive error recovery,
    never-fail guarantees, and memory-efficient processing for large documents.
    """

    def __init__(self, correlation_id: Optional[str] = None) -> None:
        """Initialize tree builder.

        Args:
            correlation_id: Optional correlation ID for request tracking
        """
        self.correlation_id = correlation_id
        self.logger = get_logger(__name__, correlation_id, "xml_tree_builder")

        # Tree building state
        self._element_stack: List[XMLElement] = []
        self._current_element: Optional[XMLElement] = None
        self._text_buffer: List[str] = []
        self._in_closing_tag = False
        self._potential_closing_tag = False  # Track when we see < followed by /

        # Statistics and diagnostics
        self._elements_created = 0
        self._repairs_applied = 0
        self._processing_start_time = 0.0

    def build(self, tokens: Union[TokenizationResult, List[Token]]) -> ParseResult:
        """Build XML document tree from token stream.

        Args:
            tokens: Either TokenizationResult or list of tokens to process

        Returns:
            ParseResult containing document tree and comprehensive metadata
        """
        self._processing_start_time = time.time()

        # Extract token list and metadata
        if hasattr(tokens, 'tokens') and hasattr(tokens, 'success'):
            # This is a TokenizationResult object
            token_list = tokens.tokens
            tokenization_result = tokens
        else:
            # This is a list of tokens
            token_list = tokens
            tokenization_result = None

        self.logger.info(
            "Starting tree building",
            extra={
                "token_count": len(token_list),
                "has_tokenization_metadata": tokenization_result is not None
            }
        )

        # Initialize result
        result = ParseResult(correlation_id=self.correlation_id)
        result.tokenization_result = tokenization_result
        
        # Add diagnostic for empty input
        if not token_list:
            result.add_diagnostic(
                DiagnosticSeverity.INFO,
                "No tokens provided - empty document created",
                "tree_builder",
                details={"input_type": "empty"}
            )

        try:
            # Reset internal state
            self._reset_state()

            # Process tokens into tree structure
            document = self._build_tree(token_list, result)
            result.document = document

            # Calculate final metrics
            self._finalize_result(result)

            self.logger.info(
                "Tree building completed",
                extra={
                    "element_count": result.element_count,
                    "confidence": result.confidence,
                    "repair_count": result.repair_count
                }
            )

        except Exception as e:
            # Never-fail philosophy: return partial result on error
            processing_time = (time.time() - self._processing_start_time) * 1000

            self.logger.exception(
                "Tree building failed",
                extra={
                    "processing_time_ms": processing_time
                }
            )

            result.success = False
            result.confidence = 0.0
            result.performance.processing_time_ms = processing_time
            result.add_diagnostic(
                DiagnosticSeverity.CRITICAL,
                f"Tree building failed: {e}",
                "xml_tree_builder",
                details={"exception_type": type(e).__name__}
            )

        return result

    def _reset_state(self) -> None:
        """Reset internal state for new tree building operation."""
        self._element_stack.clear()
        self._current_element = None
        self._text_buffer.clear()
        self._in_closing_tag = False
        self._potential_closing_tag = False
        self._elements_created = 0
        self._repairs_applied = 0

    def _build_tree(self, tokens: List[Token], result: ParseResult) -> XMLDocument:
        """Build document tree from token list.

        Args:
            tokens: List of tokens to process
            result: ParseResult for adding diagnostics

        Returns:
            XMLDocument with constructed tree
        """
        document = XMLDocument(
            correlation_id=self.correlation_id,
            source_tokens=tokens
        )

        # Process each token
        for i, token in enumerate(tokens):
            try:
                self._process_token(token, document, result)
            except Exception as e:
                # Log error but continue processing (never-fail)
                result.add_diagnostic(
                    DiagnosticSeverity.ERROR,
                    f"Error processing token {i}: {e}",
                    "token_processor",
                    position={"offset": token.position.offset}
                )
                continue

        # Finalize any remaining text content
        self._finalize_text_content()

        # Close any unclosed elements (graceful recovery)
        if self._element_stack:
            self._close_unclosed_elements(document, result)

        return document

    def _process_token(
        self, token: Token, document: XMLDocument, result: ParseResult
    ) -> None:
        """Process individual token for tree building.

        Args:
            token: Token to process
            document: Document being built
            result: ParseResult for diagnostics
        """
        if token.type == TokenType.TAG_START:
            self._process_tag_start_token(token)
        elif token.type == TokenType.TAG_NAME:
            self._process_tag_name_token(token, document, result)
        elif token.type == TokenType.TAG_END:
            self._process_tag_end_token(token)
        elif token.type == TokenType.TEXT:
            self._process_text_token(token)
        elif token.type == TokenType.ATTR_NAME:
            self._handle_attribute_name(token)
        elif token.type == TokenType.ATTR_VALUE:
            self._handle_attribute_value(token)
        elif token.type == TokenType.WHITESPACE:
            self._process_whitespace_token(token)
        elif token.type == TokenType.COMMENT:
            # Handle comment tokens separately from processing instructions
            self._handle_comment(token)
        elif token.type == TokenType.PROCESSING_INSTRUCTION:
            # Handle processing instructions
            self._handle_processing_instruction(token)
        else:
            self._handle_unrecognized_token(token, result)

    def _process_tag_start_token(self, token: Token) -> None:
        """Process TAG_START token."""
        self._flush_text_buffer()
        # Check if this is a closing tag (old format)
        if token.value == "</":
            self._in_closing_tag = True
        elif token.value == "<":
            # New format - we might see "/" next as a TEXT token
            self._potential_closing_tag = True
            self._in_closing_tag = False
        else:
            self._potential_closing_tag = False
            self._in_closing_tag = False

    def _process_tag_name_token(
        self, token: Token, document: XMLDocument, result: ParseResult
    ) -> None:
        """Process TAG_NAME token."""
        if self._in_closing_tag:
            self._handle_closing_tag_name(token, result)
        else:
            self._handle_tag_name(token, document)

    def _process_tag_end_token(self, token: Token) -> None:
        """Process TAG_END token."""
        # Handle closing the current tag
        if token.value == "/>":
            self._close_self_closing_tag()
        # Reset closing tag state
        self._in_closing_tag = False
        self._potential_closing_tag = False

    def _process_text_token(self, token: Token) -> None:
        """Process TEXT token, handling potential closing tag markers."""
        if self._potential_closing_tag and token.value == "/":
            # This is the "/" in a closing tag sequence: < / tagname >
            self._in_closing_tag = True
            self._potential_closing_tag = False
        else:
            # Regular text content
            self._potential_closing_tag = False
            self._add_text_content(token.value)

    def _process_whitespace_token(self, token: Token) -> None:
        """Process WHITESPACE token."""
        # Preserve significant whitespace
        if self._current_element or self._element_stack:
            self._add_text_content(token.value)

    def _handle_unrecognized_token(self, token: Token, result: ParseResult) -> None:
        """Handle unrecognized token types."""
        # Special handling for INVALID_CHARS tokens - convert to text content
        # when they contain valid XML text characters
        if token.type == TokenType.INVALID_CHARS:
            # Convert to text content to preserve valid characters like ?
            self._add_text_content(token.value)
            return
            
        result.add_diagnostic(
            DiagnosticSeverity.WARNING,
            f"Unhandled token type: {token.type}",
            "token_processor",
            position={"offset": token.position.offset},
        )

    def _handle_processing_instruction(self, token: Token) -> None:
        """Handle processing instruction tokens.
        
        Processing instructions should be included in the output XML as raw PI content.
        We store them specially to avoid escaping during output formatting.
        """
        pi_content = token.value.strip() if token.value else ""
        
        if not pi_content:
            # Invalid PI (empty content) - ensure parent element is not collapsed to self-closing
            if self._current_element is not None:
                self._current_element.preserve_structure = True
        else:
            # Valid PI - store it as a special attribute to preserve it unescaped
            # Format as proper processing instruction: <?target data?>
            pi_formatted = f"<?{pi_content}?>"
            
            # Add the PI as a special attribute that won't be escaped
            if self._current_element is not None:
                # Store PI content in a special way that prevents escaping
                if not hasattr(self._current_element, '_processing_instructions'):
                    self._current_element._processing_instructions = []
                self._current_element._processing_instructions.append(pi_formatted)
                
                # Also ensure element structure is preserved
                self._current_element.preserve_structure = True
            else:
                # If no current element, add to text buffer as regular text
                self._add_text_content(pi_formatted)

    def _handle_comment(self, token: Token) -> None:
        """Handle comment tokens.
        
        Comments should be included in the output XML as properly formatted comments.
        We store them specially to avoid escaping during output formatting.
        """
        comment_content = token.value.strip() if token.value else ""
        
        if not comment_content:
            # Empty comment - ensure parent element structure is preserved
            if self._current_element is not None:
                self._current_element.preserve_structure = True
        else:
            # Valid comment - store it as a special attribute that won't be escaped
            # Format as proper comment: <!-- content -->
            comment_formatted = f"<!-- {comment_content} -->"
            
            # Add the comment as a special attribute that won't be escaped
            if self._current_element is not None:
                # Store comment content in a special way that prevents escaping
                if not hasattr(self._current_element, '_comments'):
                    self._current_element._comments = []
                self._current_element._comments.append(comment_formatted)
                
                # Also ensure element structure is preserved
                self._current_element.preserve_structure = True
            else:
                # If no current element, add to text buffer as regular text
                self._add_text_content(comment_formatted)

    def _handle_tag_name(self, token: Token, document: XMLDocument) -> None:
        """Handle tag name token for element creation."""
        tag_name = token.value

        # Auto-close unclosed elements if this tag should be a sibling, not child
        if self._element_stack and self._should_auto_close_for_sibling(tag_name):
            self._auto_close_for_sibling(tag_name)

        # Create new element
        element = XMLElement(
            tag=tag_name,
            source_tokens=[token],
            confidence=token.confidence,
            start_position={
                "line": token.position.line,
                "column": token.position.column,
                "offset": token.position.offset,
            },
        )

        self._elements_created += 1

        # Set as root if this is the first element
        if not document.root and not self._element_stack:
            document.root = element

        # Add to parent if we have one
        if self._element_stack:
            parent = self._element_stack[-1]
            parent.add_child(element)

        # Push to stack for potential children
        self._element_stack.append(element)
        self._current_element = element

    def _handle_closing_tag_name(self, token: Token, result: ParseResult) -> None:
        """Handle closing tag name token."""
        if not self._element_stack:
            result.add_diagnostic(
                DiagnosticSeverity.WARNING,
                "Closing tag without matching opening tag",
                "structure_repair",
                position={"offset": token.position.offset}
            )
            return

        tag_name = token.value

        # Find matching opening tag in stack
        element = self._element_stack[-1]

        if element.tag == tag_name:
            # Perfect match - close normally
            self._close_current_element()
        else:
            # Mismatched tags - apply repair
            self._handle_mismatched_closing_tag(tag_name, result)

    def _close_self_closing_tag(self) -> None:
        """Close self-closing tag (e.g., <br/>)."""
        if self._element_stack:
            self._close_current_element()

    def _close_current_element(self) -> None:
        """Close the current element and update state."""
        if not self._element_stack:
            return

        element = self._element_stack.pop()

        # Finalize any text content
        if self._text_buffer:
            text_content = "".join(self._text_buffer).strip()
            if text_content and not element.text:
                element.text = text_content
            self._text_buffer.clear()

        # Update current element
        self._current_element = self._element_stack[-1] if self._element_stack else None

    def _handle_mismatched_closing_tag(
        self, tag_name: str, result: ParseResult
    ) -> None:
        """Handle mismatched closing tag with repair strategy."""
        # Look for matching opening tag in the stack
        matching_index = -1
        for i in range(len(self._element_stack) - 1, -1, -1):
            if self._element_stack[i].tag == tag_name:
                matching_index = i
                break

        if matching_index >= 0:
            # Close elements back to the matching one
            elements_to_close = len(self._element_stack) - matching_index

            repair = StructureRepair(
                repair_type="mismatched_tags",
                description=f"Auto-closed {elements_to_close} unclosed elements",
                original_tokens=[],
                confidence_impact=0.1 * elements_to_close,
            )

            # Close elements
            for _ in range(elements_to_close):
                if self._element_stack:
                    element = self._element_stack.pop()
                    element.repairs.append(repair)

            self._repairs_applied += 1
            result.add_diagnostic(
                DiagnosticSeverity.INFO,
                f"Applied tag mismatch repair for </{tag_name}>",
                "structure_repair",
            )
        else:
            # No matching tag found - ignore closing tag
            result.add_diagnostic(
                DiagnosticSeverity.WARNING,
                f"Orphaned closing tag </{tag_name}> ignored",
                "structure_repair",
            )

    def _handle_attribute_name(self, token: Token) -> None:
        """Handle attribute name token."""
        if self._current_element:
            # Store attribute name for pairing with value
            self._current_attribute_name = token.value

    def _handle_attribute_value(self, token: Token) -> None:
        """Handle attribute value token."""
        if self._current_element and hasattr(self, "_current_attribute_name"):
            # Clean attribute value (remove quotes)
            value = token.value.strip('"\'')
            self._current_element.set_attribute(self._current_attribute_name, value)
            delattr(self, "_current_attribute_name")

    def _add_text_content(self, text: str) -> None:
        """Add text content to buffer."""
        self._text_buffer.append(text)

    def _flush_text_buffer(self) -> None:
        """Flush text buffer to current element."""
        if self._text_buffer and self._current_element:
            text_content = "".join(self._text_buffer).strip()
            if text_content:
                if self._current_element.text:
                    self._current_element.text += text_content
                else:
                    self._current_element.text = text_content
            self._text_buffer.clear()

    def _finalize_text_content(self) -> None:
        """Finalize any remaining text content."""
        if self._text_buffer and self._current_element:
            text_content = "".join(self._text_buffer).strip()
            if text_content:
                if self._current_element.text:
                    self._current_element.text += text_content
                else:
                    self._current_element.text = text_content

    def _close_unclosed_elements(
        self, document: XMLDocument, result: ParseResult
    ) -> None:
        """Close any unclosed elements with repair information."""
        unclosed_count = len(self._element_stack)

        repair = StructureRepair(
            repair_type="unclosed_elements",
            description=f"Auto-closed {unclosed_count} unclosed elements",
            original_tokens=[],
            confidence_impact=min(1.0, _REPAIR_CONFIDENCE_IMPACT_PER_REPAIR * unclosed_count),
        )

        # Close all remaining elements
        while self._element_stack:
            element = self._element_stack.pop()
            element.repairs.append(repair)

        self._repairs_applied += 1

        result.add_diagnostic(
            DiagnosticSeverity.INFO,
            f"Auto-closed {unclosed_count} unclosed elements",
            "structure_repair",
        )

        document.repairs.append(repair)

    def _finalize_result(self, result: ParseResult) -> None:
        """Finalize parse result with metrics and statistics."""
        processing_time = (time.time() - self._processing_start_time) * 1000

        # Update performance metrics
        result.performance.processing_time_ms = processing_time
        result.performance.tokens_generated = self._elements_created

        if result.tokenization_result:
            # Handle different TokenizationResult classes
            if hasattr(result.tokenization_result, "performance"):
                result.performance.characters_processed = (
                    result.tokenization_result.performance.characters_processed
                )
            elif hasattr(result.tokenization_result, "character_count"):
                result.performance.characters_processed = (
                    result.tokenization_result.character_count
                )

        # Update document statistics
        if result.document and result.document.root:
            result.document.processing_time_ms = processing_time
            result.document._calculate_statistics()  # noqa: SLF001

        # Calculate final confidence based on repairs
        repair_impact = self._repairs_applied * _REPAIR_CONFIDENCE_IMPACT_PER_REPAIR
        base_confidence = result.document.confidence if result.document else 1.0
        result.confidence = max(0.0, base_confidence - repair_impact)

    def _should_auto_close_for_sibling(self, new_tag_name: str) -> bool:
        """Determine if unclosed elements should be auto-closed for a new sibling tag.
        
        Args:
            new_tag_name: Name of the new tag being opened
            
        Returns:
            True if the current element should be auto-closed to make room for a sibling
        """
        if not self._element_stack:
            return False
            
        current_element = self._element_stack[-1]
        current_tag = current_element.tag
        
        has_text_buffer = bool(self._text_buffer)
        has_element_text = bool(current_element.text and current_element.text.strip())
        has_text = has_text_buffer or has_element_text
        is_different_tag = current_tag != new_tag_name
        is_typical_child = self._is_typical_child_tag(current_tag, new_tag_name)
        
        
        # Simple heuristic: if we have accumulated text content in the current element,
        # and we're opening a new tag that's not typically a child, auto-close
        if (has_text and is_different_tag and not is_typical_child):
            return True
            
        return False
    
    def _is_typical_child_tag(self, parent_tag: str, child_tag: str) -> bool:
        """Determine if child_tag is typically a child of parent_tag.
        
        Args:
            parent_tag: The parent tag name
            child_tag: The potential child tag name
            
        Returns:
            True if child_tag is typically a child of parent_tag
        """
        # Simple heuristic: certain tags commonly contain others
        container_tags = {
            'html': ['head', 'body'],
            'head': ['title', 'meta', 'link', 'script', 'style'],
            'body': ['div', 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'span', 'a'],
            'div': ['p', 'span', 'a', 'img', 'br'],
            'table': ['thead', 'tbody', 'tr'],
            'tr': ['td', 'th'],
            'ul': ['li'],
            'ol': ['li'],
        }
        
        parent_lower = parent_tag.lower()
        child_lower = child_tag.lower()
        
        # Check if child is in the typical children list
        if parent_lower in container_tags:
            return child_lower in container_tags[parent_lower]
            
        # Default: assume sibling relationship for unknown tags with text content
        return False
        
    def _auto_close_for_sibling(self, new_tag_name: str) -> None:
        """Auto-close the current element to make room for a sibling tag.
        
        Args:
            new_tag_name: Name of the new sibling tag being opened
        """
        if not self._element_stack:
            return
            
        # Close the current element
        element = self._element_stack[-1]
        
        # Create repair record
        repair = StructureRepair(
            repair_type="unclosed_elements",
            description=f"Auto-closed unclosed element <{element.tag}> for sibling <{new_tag_name}>",
            original_tokens=element.source_tokens,
            confidence_impact=0.1,
            severity="minor"
        )
        
        element.repairs.append(repair)
        self._repairs_applied += 1
        
        # Close the element
        self._close_current_element()
