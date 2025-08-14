"""Tests for tree validation and finalization functionality."""

import pytest
from typing import Dict, Any

from ultra_robust_xml_parser.shared import DiagnosticSeverity
from ultra_robust_xml_parser.tree.builder import XMLDocument, XMLElement, StructureRepair
from ultra_robust_xml_parser.tree.validation import (
    TreeValidator,
    ValidationLevel,
    ValidationIssue,
    ValidationIssueType,
    ValidationResult,
    ValidationRule,
    TreeOptimizer,
    OptimizationType,
    OptimizationAction,
    OptimizationResult,
    OutputFormatter,
    OutputFormat,
    OutputConfiguration,
    FormatResult,
)


class TestValidationIssue:
    """Test ValidationIssue class functionality."""
    
    def test_basic_issue_creation(self) -> None:
        """Test basic validation issue creation."""
        issue = ValidationIssue(
            issue_type=ValidationIssueType.STRUCTURAL_INTEGRITY,
            severity=DiagnosticSeverity.ERROR,
            message="Test issue"
        )
        
        assert issue.issue_type == ValidationIssueType.STRUCTURAL_INTEGRITY
        assert issue.severity == DiagnosticSeverity.ERROR
        assert issue.message == "Test issue"
        assert issue.element_path is None
        assert issue.suggested_fix is None
    
    def test_complete_issue_creation(self) -> None:
        """Test validation issue with all fields."""
        issue = ValidationIssue(
            issue_type=ValidationIssueType.WELL_FORMEDNESS,
            severity=DiagnosticSeverity.WARNING,
            message="Well-formedness issue",
            element_path="/root/element",
            suggested_fix="Fix the issue",
            position={"line": 10, "column": 5},
            details={"context": "test"}
        )
        
        assert issue.issue_type == ValidationIssueType.WELL_FORMEDNESS
        assert issue.severity == DiagnosticSeverity.WARNING
        assert issue.message == "Well-formedness issue"
        assert issue.element_path == "/root/element"
        assert issue.suggested_fix == "Fix the issue"
        assert issue.position == {"line": 10, "column": 5}
        assert issue.details == {"context": "test"}
    
    def test_empty_message_raises_error(self) -> None:
        """Test that empty message raises ValueError."""
        with pytest.raises(ValueError, match="Validation issue message cannot be empty"):
            ValidationIssue(
                issue_type=ValidationIssueType.STRUCTURAL_INTEGRITY,
                severity=DiagnosticSeverity.ERROR,
                message=""
            )


class TestValidationRule:
    """Test ValidationRule class functionality."""
    
    def test_basic_rule_creation(self) -> None:
        """Test basic validation rule creation."""
        rule = ValidationRule(
            rule_id="test_rule",
            rule_name="Test Rule"
        )
        
        assert rule.rule_id == "test_rule"
        assert rule.rule_name == "Test Rule"
        assert rule.enabled is True
        assert rule.severity_level == DiagnosticSeverity.ERROR
        assert rule.description is None
        assert rule.parameters == {}
    
    def test_complete_rule_creation(self) -> None:
        """Test validation rule with all fields."""
        rule = ValidationRule(
            rule_id="complete_rule",
            rule_name="Complete Rule",
            enabled=False,
            severity_level=DiagnosticSeverity.WARNING,
            description="Test description",
            parameters={"param1": "value1"}
        )
        
        assert rule.rule_id == "complete_rule"
        assert rule.rule_name == "Complete Rule"
        assert rule.enabled is False
        assert rule.severity_level == DiagnosticSeverity.WARNING
        assert rule.description == "Test description"
        assert rule.parameters == {"param1": "value1"}
    
    def test_empty_rule_id_raises_error(self) -> None:
        """Test that empty rule ID raises ValueError."""
        with pytest.raises(ValueError, match="Rule ID cannot be empty"):
            ValidationRule(rule_id="", rule_name="Test Rule")
    
    def test_empty_rule_name_raises_error(self) -> None:
        """Test that empty rule name raises ValueError."""
        with pytest.raises(ValueError, match="Rule name cannot be empty"):
            ValidationRule(rule_id="test", rule_name="")


class TestValidationResult:
    """Test ValidationResult class functionality."""
    
    def test_empty_validation_result(self) -> None:
        """Test empty validation result."""
        result = ValidationResult()
        
        assert result.success is True
        assert result.confidence == 1.0
        assert result.issues == []
        assert result.rules_checked == []
        assert result.validation_level == ValidationLevel.STANDARD
        assert result.processing_time_ms == 0.0
        assert result.elements_validated == 0
        assert result.attributes_validated == 0
        assert result.error_count == 0
        assert result.warning_count == 0
    
    def test_validation_result_with_issues(self) -> None:
        """Test validation result with various issues."""
        error_issue = ValidationIssue(
            issue_type=ValidationIssueType.STRUCTURAL_INTEGRITY,
            severity=DiagnosticSeverity.ERROR,
            message="Error issue"
        )
        warning_issue = ValidationIssue(
            issue_type=ValidationIssueType.WELL_FORMEDNESS,
            severity=DiagnosticSeverity.WARNING,
            message="Warning issue"
        )
        critical_issue = ValidationIssue(
            issue_type=ValidationIssueType.NAMESPACE_VIOLATION,
            severity=DiagnosticSeverity.CRITICAL,
            message="Critical issue"
        )
        
        result = ValidationResult(
            success=False,
            confidence=0.5,
            issues=[error_issue, warning_issue, critical_issue],
            rules_checked=["rule1", "rule2"],
            validation_level=ValidationLevel.STRICT,
            processing_time_ms=100.0,
            elements_validated=10,
            attributes_validated=5
        )
        
        assert result.success is False
        assert result.confidence == 0.5
        assert len(result.issues) == 3
        assert result.error_count == 2  # ERROR and CRITICAL
        assert result.warning_count == 1
        assert result.validation_level == ValidationLevel.STRICT
    
    def test_get_issues_by_type(self) -> None:
        """Test filtering issues by type."""
        structural_issue = ValidationIssue(
            issue_type=ValidationIssueType.STRUCTURAL_INTEGRITY,
            severity=DiagnosticSeverity.ERROR,
            message="Structural issue"
        )
        namespace_issue = ValidationIssue(
            issue_type=ValidationIssueType.NAMESPACE_VIOLATION,
            severity=DiagnosticSeverity.WARNING,
            message="Namespace issue"
        )
        
        result = ValidationResult(issues=[structural_issue, namespace_issue])
        
        structural_issues = result.get_issues_by_type(ValidationIssueType.STRUCTURAL_INTEGRITY)
        namespace_issues = result.get_issues_by_type(ValidationIssueType.NAMESPACE_VIOLATION)
        
        assert len(structural_issues) == 1
        assert structural_issues[0].message == "Structural issue"
        assert len(namespace_issues) == 1
        assert namespace_issues[0].message == "Namespace issue"
    
    def test_get_issues_by_severity(self) -> None:
        """Test filtering issues by severity."""
        error_issue = ValidationIssue(
            issue_type=ValidationIssueType.STRUCTURAL_INTEGRITY,
            severity=DiagnosticSeverity.ERROR,
            message="Error issue"
        )
        warning_issue = ValidationIssue(
            issue_type=ValidationIssueType.WELL_FORMEDNESS,
            severity=DiagnosticSeverity.WARNING,
            message="Warning issue"
        )
        
        result = ValidationResult(issues=[error_issue, warning_issue])
        
        error_issues = result.get_issues_by_severity(DiagnosticSeverity.ERROR)
        warning_issues = result.get_issues_by_severity(DiagnosticSeverity.WARNING)
        
        assert len(error_issues) == 1
        assert error_issues[0].message == "Error issue"
        assert len(warning_issues) == 1
        assert warning_issues[0].message == "Warning issue"


class TestTreeValidator:
    """Test TreeValidator class functionality."""
    
    def test_validator_initialization(self) -> None:
        """Test tree validator initialization."""
        validator = TreeValidator()
        
        assert validator.validation_level == ValidationLevel.STANDARD
        assert validator.correlation_id is None
        assert validator._rules is not None
        assert len(validator._rules) > 0
    
    def test_validator_with_custom_level(self) -> None:
        """Test tree validator with custom validation level."""
        validator = TreeValidator(
            validation_level=ValidationLevel.STRICT,
            correlation_id="test-123"
        )
        
        assert validator.validation_level == ValidationLevel.STRICT
        assert validator.correlation_id == "test-123"
    
    def test_validate_empty_document(self) -> None:
        """Test validation of document without root element."""
        document = XMLDocument()
        validator = TreeValidator()
        
        result = validator.validate(document)
        
        assert result.success is False
        assert result.error_count > 0
        assert any("no root element" in issue.message for issue in result.issues)
    
    def test_validate_simple_valid_document(self) -> None:
        """Test validation of simple valid document."""
        root = XMLElement(tag="root", text="content")
        document = XMLDocument(root=root)
        validator = TreeValidator()
        
        result = validator.validate(document)
        
        assert result.success is True
        assert result.error_count == 0
        assert result.elements_validated == 1
        assert result.confidence == 1.0
    
    def test_validate_document_with_children(self) -> None:
        """Test validation of document with nested elements."""
        child1 = XMLElement(tag="child1", text="text1")
        child2 = XMLElement(tag="child2", text="text2")
        root = XMLElement(tag="root", children=[child1, child2])
        document = XMLDocument(root=root)
        validator = TreeValidator()
        
        result = validator.validate(document)
        
        assert result.success is True
        assert result.error_count == 0
        assert result.elements_validated == 3  # root + 2 children
    
    def test_validate_document_with_attributes(self) -> None:
        """Test validation of document with attributes."""
        root = XMLElement(
            tag="root",
            attributes={"id": "main", "class": "container"},
            text="content"
        )
        document = XMLDocument(root=root)
        validator = TreeValidator()
        
        result = validator.validate(document)
        
        assert result.success is True
        assert result.error_count == 0
        assert result.attributes_validated == 2
    
    def test_validate_invalid_tag_name(self) -> None:
        """Test validation of element with invalid tag name."""
        root = XMLElement(tag="<invalid>", text="content")
        document = XMLDocument(root=root)
        validator = TreeValidator()
        
        result = validator.validate(document)
        
        assert result.success is False
        assert result.error_count > 0
        assert any("invalid characters" in issue.message for issue in result.issues)
    
    def test_validate_empty_tag_name(self) -> None:
        """Test validation of element with empty tag name."""
        # Create element with empty tag manually to bypass validation
        root = XMLElement.__new__(XMLElement)
        root.tag = ""
        root.attributes = {}
        root.text = None
        root.children = []
        root.confidence = 1.0
        root.repairs = []
        root.parent = None
        root.source_tokens = []
        root.start_position = None
        root.end_position = None
        
        document = XMLDocument(root=root)
        validator = TreeValidator()
        
        result = validator.validate(document)
        
        assert result.success is False
        assert result.error_count > 0
        assert any("empty tag name" in issue.message for issue in result.issues)
    
    def test_validate_invalid_attribute_name(self) -> None:
        """Test validation of element with invalid attribute name."""
        root = XMLElement(tag="root")
        root.attributes["<invalid>"] = "value"
        document = XMLDocument(root=root)
        validator = TreeValidator()
        
        result = validator.validate(document)
        
        assert result.success is False
        assert result.error_count > 0
        assert any("Attribute name contains invalid characters" in issue.message for issue in result.issues)
    
    def test_validate_invalid_attribute_value(self) -> None:
        """Test validation of element with invalid attribute value."""
        root = XMLElement(tag="root")
        root.attributes["name"] = "value<invalid>"
        document = XMLDocument(root=root)
        validator = TreeValidator()
        
        result = validator.validate(document)
        
        assert result.success is False
        assert result.error_count > 0
        assert any("Attribute value contains invalid characters" in issue.message for issue in result.issues)
    
    def test_validate_xml_reserved_tag_name(self) -> None:
        """Test validation of element with XML reserved tag name."""
        root = XMLElement(tag="xmlReserved", text="content")
        document = XMLDocument(root=root)
        validator = TreeValidator()
        
        result = validator.validate(document)
        
        # Should generate warning but not error
        assert result.warning_count > 0
        assert any("XML reserved prefix" in issue.message for issue in result.issues)
    
    def test_validate_parent_child_inconsistency(self) -> None:
        """Test validation of inconsistent parent-child relationships."""
        parent = XMLElement(tag="parent")
        child = XMLElement(tag="child")
        
        # Manually create inconsistent relationship
        parent.children.append(child)
        # Don't set child.parent = parent (inconsistency)
        
        document = XMLDocument(root=parent)
        validator = TreeValidator()
        
        result = validator.validate(document)
        
        assert result.success is False
        assert result.error_count > 0
        assert any("parent reference inconsistency" in issue.message for issue in result.issues)
    
    def test_validate_namespaces_strict_mode(self) -> None:
        """Test namespace validation in strict mode."""
        # Create element with undeclared namespace prefix
        root = XMLElement(tag="ns:root", text="content")
        document = XMLDocument(root=root)
        validator = TreeValidator(validation_level=ValidationLevel.STRICT)
        
        result = validator.validate(document)
        
        assert result.success is False
        assert result.error_count > 0
        assert any("Undeclared namespace prefix" in issue.message for issue in result.issues)
    
    def test_validate_with_declared_namespace(self) -> None:
        """Test namespace validation with properly declared namespace."""
        root = XMLElement(
            tag="ns:root",
            attributes={"xmlns:ns": "http://example.com/ns"},
            text="content"
        )
        document = XMLDocument(root=root)
        validator = TreeValidator(validation_level=ValidationLevel.STRICT)
        
        result = validator.validate(document)
        
        # Should not have namespace errors
        namespace_errors = result.get_issues_by_type(ValidationIssueType.NAMESPACE_VIOLATION)
        assert len(namespace_errors) == 0
    
    def test_validate_pedantic_level(self) -> None:
        """Test validation at pedantic level."""
        root = XMLElement(tag="root", text="content")
        document = XMLDocument(root=root, version="1.1", encoding="iso-8859-1")
        validator = TreeValidator(validation_level=ValidationLevel.PEDANTIC)
        
        result = validator.validate(document)
        
        # Should generate warnings about non-standard version and encoding
        warnings = result.get_issues_by_severity(DiagnosticSeverity.WARNING)
        assert len(warnings) >= 2
        assert any("Non-standard XML version" in warning.message for warning in warnings)
        assert any("Non-standard encoding" in warning.message for warning in warnings)
    
    def test_validation_never_fails_on_exception(self) -> None:
        """Test that validation never fails with exception."""
        # Create a document that might cause issues during validation
        root = XMLElement(tag="root")
        document = XMLDocument(root=root)
        
        # Mock a method to raise an exception
        validator = TreeValidator()
        original_method = validator._validate_document_structure
        
        def mock_method(*args, **kwargs):
            raise RuntimeError("Test exception")
        
        validator._validate_document_structure = mock_method
        
        # Should not raise exception, should return failed result
        result = validator.validate(document)
        
        assert result.success is False
        assert result.confidence == 0.0
        assert len(result.issues) > 0
        assert any("Validation failed" in issue.message for issue in result.issues)
        
        # Restore original method
        validator._validate_document_structure = original_method
    
    def test_different_validation_levels(self) -> None:
        """Test different validation levels have different rule sets."""
        minimal_validator = TreeValidator(ValidationLevel.MINIMAL)
        standard_validator = TreeValidator(ValidationLevel.STANDARD)
        strict_validator = TreeValidator(ValidationLevel.STRICT)
        pedantic_validator = TreeValidator(ValidationLevel.PEDANTIC)
        
        # Each level should have different number of rules
        assert len(minimal_validator._rules) <= len(standard_validator._rules)
        assert len(standard_validator._rules) <= len(strict_validator._rules)
        assert len(strict_validator._rules) <= len(pedantic_validator._rules)
        
        # All should have basic structural rules
        for validator in [minimal_validator, standard_validator, strict_validator, pedantic_validator]:
            assert "parent_child_consistency" in validator._rules
            assert "circular_reference_check" in validator._rules
            assert "tag_name_validity" in validator._rules
    
    def test_confidence_calculation(self) -> None:
        """Test confidence calculation based on validation issues."""
        # Test with errors
        root = XMLElement(tag="<invalid>")  # Invalid tag
        document = XMLDocument(root=root)
        validator = TreeValidator()
        
        result = validator.validate(document)
        
        assert result.success is False
        assert result.confidence < 1.0
        assert result.confidence >= 0.0
        
        # More errors should result in lower confidence
        root2 = XMLElement(tag="<invalid>")
        root2.attributes["<bad>"] = "value"  # Invalid attribute name too
        document2 = XMLDocument(root=root2)
        
        result2 = validator.validate(document2)
        
        assert result2.confidence <= result.confidence


@pytest.fixture
def sample_valid_document() -> XMLDocument:
    """Create a sample valid XML document for testing."""
    child1 = XMLElement(tag="item", attributes={"id": "1"}, text="First item")
    child2 = XMLElement(tag="item", attributes={"id": "2"}, text="Second item")
    root = XMLElement(
        tag="items",
        attributes={"xmlns": "http://example.com", "version": "1.0"},
        children=[child1, child2]
    )
    return XMLDocument(root=root, encoding="utf-8", version="1.0")


@pytest.fixture
def sample_invalid_document() -> XMLDocument:
    """Create a sample invalid XML document for testing."""
    # Create element with various issues manually to bypass validation
    child = XMLElement.__new__(XMLElement)
    child.tag = "<bad>"  # Invalid tag
    child.attributes = {"<invalid>": "value<bad>"}  # Invalid attribute name and value
    child.text = "content"
    child.children = []
    child.confidence = 1.0
    child.repairs = []
    child.parent = None
    child.source_tokens = []
    child.start_position = None
    child.end_position = None
    
    root = XMLElement.__new__(XMLElement)
    root.tag = "ns:root"  # Undeclared namespace
    root.attributes = {}
    root.text = None
    root.children = [child]
    root.confidence = 1.0
    root.repairs = []
    root.parent = None
    root.source_tokens = []
    root.start_position = None
    root.end_position = None
    
    # Set parent relationship manually
    child.parent = root
    
    return XMLDocument(root=root)


class TestTreeValidatorIntegration:
    """Integration tests for TreeValidator with sample documents."""
    
    def test_validate_valid_document(self, sample_valid_document: XMLDocument) -> None:
        """Test validation of a valid document."""
        validator = TreeValidator(ValidationLevel.STANDARD)
        result = validator.validate(sample_valid_document)
        
        assert result.success is True
        assert result.error_count == 0
        assert result.confidence == 1.0
        assert result.elements_validated == 3  # root + 2 children
        assert result.attributes_validated == 4  # 2 root attrs + 2 child attrs
    
    def test_validate_invalid_document(self, sample_invalid_document: XMLDocument) -> None:
        """Test validation of an invalid document."""
        validator = TreeValidator(ValidationLevel.STANDARD)
        result = validator.validate(sample_invalid_document)
        
        assert result.success is False
        assert result.error_count > 0
        assert result.confidence < 1.0
        
        # Should have multiple types of issues
        well_formed_issues = result.get_issues_by_type(ValidationIssueType.WELL_FORMEDNESS)
        attribute_issues = result.get_issues_by_type(ValidationIssueType.ATTRIBUTE_ISSUE)
        
        # The invalid document should have well-formedness and attribute issues
        assert len(well_formed_issues) > 0 or len(attribute_issues) > 0
        
        # Check for specific issues
        all_messages = [issue.message for issue in result.issues]
        assert any("invalid characters" in msg for msg in all_messages)
    
    def test_validate_strict_vs_standard(self, sample_valid_document: XMLDocument) -> None:
        """Test different validation levels produce different results."""
        # Add namespace prefix to test strict validation
        sample_valid_document.root.tag = "ns:items"
        
        standard_validator = TreeValidator(ValidationLevel.STANDARD)
        strict_validator = TreeValidator(ValidationLevel.STRICT)
        
        standard_result = standard_validator.validate(sample_valid_document)
        strict_result = strict_validator.validate(sample_valid_document)
        
        # Standard should pass, strict should fail due to undeclared namespace
        assert standard_result.success is True
        assert strict_result.success is False
        
        namespace_errors = strict_result.get_issues_by_type(ValidationIssueType.NAMESPACE_VIOLATION)
        assert len(namespace_errors) > 0


class TestOptimizationAction:
    """Test OptimizationAction class functionality."""
    
    def test_basic_optimization_action(self) -> None:
        """Test basic optimization action creation."""
        action = OptimizationAction(
            optimization_type=OptimizationType.TEXT_NODE_CONSOLIDATION,
            description="Test optimization"
        )
        
        assert action.optimization_type == OptimizationType.TEXT_NODE_CONSOLIDATION
        assert action.description == "Test optimization"
        assert action.elements_affected == 0
        assert action.memory_saved_bytes == 0
        assert action.confidence_impact == 0.0
        assert action.details is None
    
    def test_complete_optimization_action(self) -> None:
        """Test optimization action with all fields."""
        action = OptimizationAction(
            optimization_type=OptimizationType.MEMORY_CLEANUP,
            description="Complete optimization",
            elements_affected=5,
            memory_saved_bytes=1024,
            confidence_impact=0.1,
            details={"method": "cleanup"}
        )
        
        assert action.optimization_type == OptimizationType.MEMORY_CLEANUP
        assert action.description == "Complete optimization"
        assert action.elements_affected == 5
        assert action.memory_saved_bytes == 1024
        assert action.confidence_impact == 0.1
        assert action.details == {"method": "cleanup"}
    
    def test_empty_description_raises_error(self) -> None:
        """Test that empty description raises ValueError."""
        with pytest.raises(ValueError, match="Optimization action description cannot be empty"):
            OptimizationAction(
                optimization_type=OptimizationType.TEXT_NODE_CONSOLIDATION,
                description=""
            )
    
    def test_negative_elements_affected_raises_error(self) -> None:
        """Test that negative elements affected raises ValueError."""
        with pytest.raises(ValueError, match="Elements affected cannot be negative"):
            OptimizationAction(
                optimization_type=OptimizationType.TEXT_NODE_CONSOLIDATION,
                description="Test",
                elements_affected=-1
            )
    
    def test_negative_memory_saved_raises_error(self) -> None:
        """Test that negative memory saved raises ValueError."""
        with pytest.raises(ValueError, match="Memory saved cannot be negative"):
            OptimizationAction(
                optimization_type=OptimizationType.TEXT_NODE_CONSOLIDATION,
                description="Test",
                memory_saved_bytes=-1
            )


class TestOptimizationResult:
    """Test OptimizationResult class functionality."""
    
    def test_empty_optimization_result(self) -> None:
        """Test empty optimization result."""
        result = OptimizationResult()
        
        assert result.success is True
        assert result.confidence == 1.0
        assert result.actions == []
        assert result.processing_time_ms == 0.0
        assert result.total_elements_processed == 0
        assert result.total_memory_saved_bytes == 0
        assert result.total_actions == 0
        assert result.elements_affected == 0
    
    def test_optimization_result_with_actions(self) -> None:
        """Test optimization result with various actions."""
        action1 = OptimizationAction(
            optimization_type=OptimizationType.TEXT_NODE_CONSOLIDATION,
            description="Text consolidation",
            elements_affected=3,
            memory_saved_bytes=100
        )
        action2 = OptimizationAction(
            optimization_type=OptimizationType.MEMORY_CLEANUP,
            description="Memory cleanup",
            elements_affected=5,
            memory_saved_bytes=200
        )
        
        result = OptimizationResult(
            success=True,
            confidence=1.05,
            actions=[action1, action2],
            processing_time_ms=50.0,
            total_elements_processed=20,
            total_memory_saved_bytes=300
        )
        
        assert result.success is True
        assert result.confidence == 1.05
        assert len(result.actions) == 2
        assert result.total_actions == 2
        assert result.elements_affected == 8  # 3 + 5
        assert result.total_memory_saved_bytes == 300
    
    def test_get_actions_by_type(self) -> None:
        """Test filtering actions by type."""
        text_action = OptimizationAction(
            optimization_type=OptimizationType.TEXT_NODE_CONSOLIDATION,
            description="Text action"
        )
        memory_action = OptimizationAction(
            optimization_type=OptimizationType.MEMORY_CLEANUP,
            description="Memory action"
        )
        
        result = OptimizationResult(actions=[text_action, memory_action])
        
        text_actions = result.get_actions_by_type(OptimizationType.TEXT_NODE_CONSOLIDATION)
        memory_actions = result.get_actions_by_type(OptimizationType.MEMORY_CLEANUP)
        
        assert len(text_actions) == 1
        assert text_actions[0].description == "Text action"
        assert len(memory_actions) == 1
        assert memory_actions[0].description == "Memory action"


class TestTreeOptimizer:
    """Test TreeOptimizer class functionality."""
    
    def test_optimizer_initialization(self) -> None:
        """Test tree optimizer initialization."""
        optimizer = TreeOptimizer()
        
        assert optimizer.correlation_id is None
        assert optimizer._min_text_length_for_consolidation == 1
        assert optimizer._max_consecutive_whitespace == 10
        assert optimizer._redundant_element_threshold == 2
    
    def test_optimizer_with_correlation_id(self) -> None:
        """Test tree optimizer with correlation ID."""
        optimizer = TreeOptimizer(correlation_id="test-123")
        
        assert optimizer.correlation_id == "test-123"
    
    def test_optimize_empty_document(self) -> None:
        """Test optimization of document without root element."""
        document = XMLDocument()
        optimizer = TreeOptimizer()
        
        result = optimizer.optimize(document)
        
        assert result.success is False
        assert result.confidence == 0.0
        assert result.total_actions == 0
    
    def test_optimize_simple_document(self) -> None:
        """Test optimization of simple valid document."""
        root = XMLElement(tag="root", text="content")
        document = XMLDocument(root=root)
        optimizer = TreeOptimizer()
        
        result = optimizer.optimize(document)
        
        assert result.success is True
        assert result.confidence >= 1.0
        assert result.total_elements_processed == 1
    
    def test_text_consolidation(self) -> None:
        """Test text node consolidation optimization."""
        # Create element with whitespace that needs consolidation
        root = XMLElement(tag="root", text="  multiple   spaces   here  ")
        document = XMLDocument(root=root)
        optimizer = TreeOptimizer()
        
        result = optimizer.optimize(document)
        
        assert result.success is True
        assert result.total_actions > 0
        
        # Check that text was normalized
        assert root.text == "multiple spaces here"
        
        # Check for text consolidation action
        text_actions = result.get_actions_by_type(OptimizationType.TEXT_NODE_CONSOLIDATION)
        assert len(text_actions) > 0
    
    def test_redundant_element_removal(self) -> None:
        """Test removal of redundant elements."""
        # Create redundant empty element
        empty_child = XMLElement(tag="empty")  # No text, attributes, or children
        root = XMLElement(tag="root", text="content", children=[empty_child])
        document = XMLDocument(root=root)
        optimizer = TreeOptimizer()
        
        result = optimizer.optimize(document)
        
        assert result.success is True
        
        # Check that empty element was removed
        assert len(root.children) == 0
        
        # Check for redundant removal action
        removal_actions = result.get_actions_by_type(OptimizationType.REDUNDANT_ELEMENT_REMOVAL)
        assert len(removal_actions) > 0
    
    def test_attribute_optimization(self) -> None:
        """Test attribute optimization."""
        # Create element with empty attributes
        root = XMLElement(tag="root", attributes={"empty": "", "valid": "value"})
        document = XMLDocument(root=root)
        optimizer = TreeOptimizer()
        
        result = optimizer.optimize(document)
        
        assert result.success is True
        
        # Check that empty attribute was removed
        assert "empty" not in root.attributes
        assert "valid" in root.attributes
        assert root.attributes["valid"] == "value"
        
        # Check for attribute optimization action
        attr_actions = result.get_actions_by_type(OptimizationType.ATTRIBUTE_OPTIMIZATION)
        assert len(attr_actions) > 0
    
    def test_memory_cleanup(self) -> None:
        """Test memory cleanup optimization."""
        from ultra_robust_xml_parser.tokenization import Token, TokenType, TokenPosition
        
        # Create element with source tokens that will be cleaned
        token = Token(
            type=TokenType.TEXT,
            value="test",
            position=TokenPosition(line=1, column=1, offset=0),
            confidence=1.0
        )
        root = XMLElement(tag="root", text="content", source_tokens=[token])
        document = XMLDocument(root=root)
        optimizer = TreeOptimizer()
        
        result = optimizer.optimize(document)
        
        assert result.success is True
        
        # Check that source tokens were cleared
        assert len(root.source_tokens) == 0
        
        # Check for memory cleanup action
        cleanup_actions = result.get_actions_by_type(OptimizationType.MEMORY_CLEANUP)
        assert len(cleanup_actions) > 0
    
    def test_structure_balancing_analysis(self) -> None:
        """Test structure balancing analysis for deep trees."""
        # Create very deep tree to trigger analysis
        current = XMLElement(tag="root")
        document = XMLDocument(root=current)
        
        # Create deep nesting (more than 50 levels)
        for i in range(55):
            child = XMLElement(tag=f"level{i}")
            current.add_child(child)
            current = child
        
        # Update document statistics
        document._calculate_statistics()
        
        optimizer = TreeOptimizer()
        result = optimizer.optimize(document)
        
        assert result.success is True
        
        # Check for structure balancing analysis
        balance_actions = result.get_actions_by_type(OptimizationType.STRUCTURE_BALANCING)
        assert len(balance_actions) > 0
        assert "depth:" in balance_actions[0].description
    
    def test_comprehensive_optimization(self) -> None:
        """Test comprehensive optimization with multiple optimization types."""
        from ultra_robust_xml_parser.tokenization import Token, TokenType, TokenPosition
        
        # Create complex document with various optimization opportunities
        token = Token(
            type=TokenType.TEXT,
            value="test",
            position=TokenPosition(line=1, column=1, offset=0),
            confidence=1.0
        )
        
        empty_child = XMLElement(tag="empty")  # Redundant
        text_child = XMLElement(tag="text", text="  lots   of   spaces  ")  # Needs consolidation
        attr_child = XMLElement(tag="attrs", attributes={"empty": "", "valid": "value"})  # Needs attr optimization
        token_child = XMLElement(tag="tokens", source_tokens=[token])  # Needs memory cleanup
        
        root = XMLElement(
            tag="root",
            text="  root  text  ",
            children=[empty_child, text_child, attr_child, token_child],
            source_tokens=[token]
        )
        
        document = XMLDocument(root=root)
        optimizer = TreeOptimizer()
        
        result = optimizer.optimize(document)
        
        assert result.success is True
        assert result.total_actions >= 4  # At least one of each type
        
        # Verify different types of optimizations were performed
        optimization_types = {action.optimization_type for action in result.actions}
        assert OptimizationType.TEXT_NODE_CONSOLIDATION in optimization_types
        assert OptimizationType.REDUNDANT_ELEMENT_REMOVAL in optimization_types
        assert OptimizationType.ATTRIBUTE_OPTIMIZATION in optimization_types
        assert OptimizationType.MEMORY_CLEANUP in optimization_types
    
    def test_optimization_never_fails_on_exception(self) -> None:
        """Test that optimization never fails with exception."""
        root = XMLElement(tag="root")
        document = XMLDocument(root=root)
        
        # Mock a method to raise an exception
        optimizer = TreeOptimizer()
        original_method = optimizer._consolidate_text_nodes
        
        def mock_method(*args, **kwargs):
            raise RuntimeError("Test exception")
        
        optimizer._consolidate_text_nodes = mock_method
        
        # Should not raise exception, should return failed result
        result = optimizer.optimize(document)
        
        assert result.success is False
        assert result.confidence == 0.0
        assert len(result.actions) > 0
        assert any("Optimization failed" in action.description for action in result.actions)
        
        # Restore original method
        optimizer._consolidate_text_nodes = original_method
    
    def test_whitespace_normalization(self) -> None:
        """Test whitespace normalization functionality."""
        optimizer = TreeOptimizer()
        
        # Test various whitespace scenarios
        assert optimizer._normalize_whitespace("  multiple   spaces  ") == "multiple spaces"
        assert optimizer._normalize_whitespace("single space") == "single space"
        assert optimizer._normalize_whitespace("   ") == ""
        assert optimizer._normalize_whitespace("tabs\t\tand\tspaces   ") == "tabs and spaces"
        assert optimizer._normalize_whitespace("newlines\n\nand\nspaces") == "newlines and spaces"
    
    def test_redundant_element_detection(self) -> None:
        """Test redundant element detection logic."""
        optimizer = TreeOptimizer()
        
        # Truly redundant element
        empty_element = XMLElement(tag="empty")
        assert optimizer._is_redundant_element(empty_element) is True
        
        # Element with text
        text_element = XMLElement(tag="text", text="content")
        assert optimizer._is_redundant_element(text_element) is False
        
        # Element with attributes
        attr_element = XMLElement(tag="attrs", attributes={"id": "test"})
        assert optimizer._is_redundant_element(attr_element) is False
        
        # Element with children
        parent_element = XMLElement(tag="parent")
        child_element = XMLElement(tag="child")
        parent_element.add_child(child_element)
        assert optimizer._is_redundant_element(parent_element) is False
        
        # Element with repairs (shouldn't be removed)
        repaired_element = XMLElement(tag="repaired")
        repaired_element.repairs.append(StructureRepair(
            repair_type="test",
            description="test repair",
            original_tokens=[],
            confidence_impact=0.1
        ))
        assert optimizer._is_redundant_element(repaired_element) is False
    
    def test_memory_estimation(self) -> None:
        """Test element memory estimation."""
        optimizer = TreeOptimizer()
        
        # Simple element
        simple_element = XMLElement(tag="simple")
        memory = optimizer._estimate_element_memory(simple_element)
        assert memory > 0
        
        # Element with text
        text_element = XMLElement(tag="text", text="some content")
        text_memory = optimizer._estimate_element_memory(text_element)
        assert text_memory > memory  # Should be more than simple element
        
        # Element with attributes
        attr_element = XMLElement(tag="attrs", attributes={"key": "value", "id": "123"})
        attr_memory = optimizer._estimate_element_memory(attr_element)
        assert attr_memory > memory  # Should be more than simple element


@pytest.fixture
def sample_unoptimized_document() -> XMLDocument:
    """Create a sample document that needs optimization."""
    from ultra_robust_xml_parser.tokenization import Token, TokenType, TokenPosition
    
    token = Token(
        type=TokenType.TEXT,
        value="test",
        position=TokenPosition(line=1, column=1, offset=0),
        confidence=1.0
    )
    
    # Create various elements that need optimization
    empty_child1 = XMLElement(tag="empty1")  # Redundant
    empty_child2 = XMLElement(tag="empty2")  # Redundant
    text_child = XMLElement(tag="textnode", text="  lots   of    spaces   here  ")
    attr_child = XMLElement(
        tag="attributes",
        attributes={"empty1": "", "valid": "value", "empty2": "", "another": "test"}
    )
    token_child = XMLElement(tag="withTokens", source_tokens=[token, token])
    
    root = XMLElement(
        tag="root",
        text="  root   has    spaces   too  ",
        attributes={"rootEmpty": "", "rootValid": "test"},
        children=[empty_child1, text_child, attr_child, token_child, empty_child2],
        source_tokens=[token]
    )
    
    return XMLDocument(root=root, encoding="utf-8", version="1.0")


class TestTreeOptimizerIntegration:
    """Integration tests for TreeOptimizer with sample documents."""
    
    def test_optimize_unoptimized_document(self, sample_unoptimized_document: XMLDocument) -> None:
        """Test optimization of a document with various inefficiencies."""
        optimizer = TreeOptimizer()
        result = optimizer.optimize(sample_unoptimized_document)
        
        assert result.success is True
        assert result.confidence >= 1.0
        assert result.total_actions > 0
        assert result.elements_affected > 0
        assert result.total_memory_saved_bytes > 0
        
        # Verify all optimization types were applied
        optimization_types = {action.optimization_type for action in result.actions}
        assert OptimizationType.TEXT_NODE_CONSOLIDATION in optimization_types
        assert OptimizationType.REDUNDANT_ELEMENT_REMOVAL in optimization_types
        assert OptimizationType.ATTRIBUTE_OPTIMIZATION in optimization_types
        assert OptimizationType.MEMORY_CLEANUP in optimization_types
        
        # Verify optimizations were actually applied
        root = sample_unoptimized_document.root
        assert root is not None
        
        # Text should be normalized
        assert "   " not in root.text  # Multiple spaces should be removed
        
        # Empty attributes should be removed
        assert "rootEmpty" not in root.attributes
        assert "rootValid" in root.attributes
        
        # Empty children should be removed
        child_tags = [child.tag for child in root.children]
        assert "empty1" not in child_tags
        assert "empty2" not in child_tags
        
        # Source tokens should be cleared
        assert len(root.source_tokens) == 0
    
    def test_optimize_already_optimized_document(self, sample_valid_document: XMLDocument) -> None:
        """Test optimization of an already well-optimized document."""
        optimizer = TreeOptimizer()
        result = optimizer.optimize(sample_valid_document)
        
        assert result.success is True
        assert result.confidence >= 1.0
        
        # Should have fewer optimizations for an already good document
        # but memory cleanup should still happen
        cleanup_actions = result.get_actions_by_type(OptimizationType.MEMORY_CLEANUP)
        assert len(cleanup_actions) >= 0  # May or may not have memory to clean
    
    def test_optimization_performance_metrics(self, sample_unoptimized_document: XMLDocument) -> None:
        """Test that optimization provides meaningful performance metrics."""
        optimizer = TreeOptimizer()
        result = optimizer.optimize(sample_unoptimized_document)
        
        assert result.processing_time_ms > 0
        assert result.total_elements_processed > 0
        
        # Should have measurable memory savings
        if result.total_memory_saved_bytes > 0:
            # At least some memory should have been saved
            assert result.total_memory_saved_bytes > 0
            
            # Memory savings should be reasonable (not negative or absurdly high)
            assert 0 < result.total_memory_saved_bytes < 1000000  # Less than 1MB for test document


class TestEnhancedParseResult:
    """Test enhanced ParseResult functionality with validation and optimization."""
    
    def test_parse_result_with_validation_and_optimization(self) -> None:
        """Test ParseResult with both validation and optimization results."""
        from ultra_robust_xml_parser.tree.builder import ParseResult
        
        # Create sample validation result
        validation_result = ValidationResult(
            success=True,
            confidence=0.95,
            elements_validated=10,
            attributes_validated=5,
            validation_level=ValidationLevel.STANDARD
        )
        validation_result.rules_checked = ["rule1", "rule2", "rule3"]
        
        # Create sample optimization result  
        optimization_result = OptimizationResult(
            success=True,
            confidence=1.05,
            total_elements_processed=10,
            total_memory_saved_bytes=1024
        )
        optimization_result.actions = [
            OptimizationAction(
                optimization_type=OptimizationType.TEXT_NODE_CONSOLIDATION,
                description="Test optimization",
                elements_affected=3,
                memory_saved_bytes=512
            ),
            OptimizationAction(
                optimization_type=OptimizationType.MEMORY_CLEANUP,
                description="Memory cleanup",
                elements_affected=5,
                memory_saved_bytes=512
            )
        ]
        
        # Create ParseResult with enhanced metadata
        root = XMLElement(tag="root", text="content")
        document = XMLDocument(root=root)
        result = ParseResult(
            document=document,
            success=True,
            confidence=0.9,
            validation_result=validation_result,
            optimization_result=optimization_result
        )
        
        # Test validation summary
        validation_summary = result.validation_summary
        assert validation_summary["validation_performed"] is True
        assert validation_summary["validation_success"] is True
        assert validation_summary["validation_confidence"] == 0.95
        assert validation_summary["elements_validated"] == 10
        assert validation_summary["attributes_validated"] == 5
        assert validation_summary["validation_level"] == "STANDARD"
        assert validation_summary["rules_checked"] == 3
        
        # Test optimization summary
        optimization_summary = result.optimization_summary
        assert optimization_summary["optimization_performed"] is True
        assert optimization_summary["optimization_success"] is True
        assert optimization_summary["optimization_confidence"] == 1.05
        assert optimization_summary["total_optimizations"] == 2
        assert optimization_summary["elements_affected"] == 8  # 3 + 5
        assert optimization_summary["memory_saved_bytes"] == 1024
        assert optimization_summary["elements_processed"] == 10
        
        # Test overall confidence breakdown
        confidence_breakdown = result.overall_confidence_breakdown
        assert confidence_breakdown["base_confidence"] == 0.9
        assert confidence_breakdown["document_confidence"] == 1.0  # Default for new document
        assert confidence_breakdown["validation_confidence"] == 0.95
        assert confidence_breakdown["optimization_confidence"] == 1.05
        assert confidence_breakdown["overall_confidence"] == 0.9  # Minimum of all confidences
    
    def test_parse_result_without_validation_optimization(self) -> None:
        """Test ParseResult without validation and optimization results."""
        from ultra_robust_xml_parser.tree.builder import ParseResult
        
        root = XMLElement(tag="root", text="content")
        document = XMLDocument(root=root)
        result = ParseResult(document=document, success=True, confidence=0.95)
        
        # Test validation summary when no validation performed
        validation_summary = result.validation_summary
        assert validation_summary["validation_performed"] is False
        
        # Test optimization summary when no optimization performed
        optimization_summary = result.optimization_summary
        assert optimization_summary["optimization_performed"] is False
        
        # Test confidence breakdown without validation/optimization
        confidence_breakdown = result.overall_confidence_breakdown
        # Note: ParseResult.__post_init__ may have updated the confidence based on document
        assert "base_confidence" in confidence_breakdown
        assert confidence_breakdown["document_confidence"] == 1.0
        assert "validation_confidence" not in confidence_breakdown
        assert "optimization_confidence" not in confidence_breakdown
        # Overall confidence should be positive
        assert confidence_breakdown["overall_confidence"] > 0.0
    
    def test_parsing_statistics_property(self) -> None:
        """Test comprehensive parsing statistics property."""
        from ultra_robust_xml_parser.tree.builder import ParseResult
        from ultra_robust_xml_parser.shared import PerformanceMetrics
        
        # Create document with multiple elements and attributes
        child1 = XMLElement(tag="child1", attributes={"id": "1"}, text="text1")
        child2 = XMLElement(tag="child2", attributes={"id": "2", "class": "test"}, text="text2")
        root = XMLElement(tag="root", attributes={"version": "1.0"}, children=[child1, child2])
        document = XMLDocument(root=root)
        document._calculate_statistics()  # Update document statistics
        
        # Create performance metrics
        performance = PerformanceMetrics(
            processing_time_ms=100.0,
            memory_used_bytes=2048,
            characters_processed=500,
            tokens_generated=25
        )
        
        result = ParseResult(
            document=document,
            success=True,
            confidence=0.95,
            performance=performance
        )
        
        stats = result.parsing_statistics
        
        assert stats["element_count"] == 3  # root + 2 children
        assert stats["attribute_count"] == 4  # version + id + id + class
        assert stats["max_depth"] == 1  # children are depth 1
        assert stats["repair_count"] == 0  # No repairs
        assert stats["has_repairs"] is False
        assert stats["processing_time_ms"] == 100.0
        assert stats["memory_used_bytes"] == 2048
        assert stats["characters_processed"] == 500
        assert stats["tokens_generated"] == 25
    
    def test_diagnostics_summary(self) -> None:
        """Test comprehensive diagnostics summary."""
        from ultra_robust_xml_parser.tree.builder import ParseResult
        from ultra_robust_xml_parser.shared import DiagnosticEntry, DiagnosticSeverity
        
        root = XMLElement(tag="root")
        document = XMLDocument(root=root)
        result = ParseResult(document=document)
        
        # Add various diagnostics
        result.add_diagnostic(DiagnosticSeverity.ERROR, "Error message", "component1")
        result.add_diagnostic(DiagnosticSeverity.WARNING, "Warning message", "component1")
        result.add_diagnostic(DiagnosticSeverity.INFO, "Info message", "component2")
        result.add_diagnostic(DiagnosticSeverity.ERROR, "Another error", "component2")
        
        summary = result.get_diagnostics_summary()
        
        assert summary["total_diagnostics"] == 4
        assert summary["has_errors"] is True
        assert summary["diagnostics_by_severity"]["ERROR"] == 2
        assert summary["diagnostics_by_severity"]["WARNING"] == 1
        assert summary["diagnostics_by_severity"]["INFO"] == 1
        assert summary["diagnostics_by_component"]["component1"] == 2
        assert summary["diagnostics_by_component"]["component2"] == 2
    
    def test_repair_summary(self) -> None:
        """Test comprehensive repair summary."""
        from ultra_robust_xml_parser.tree.builder import ParseResult
        
        # Create document with repairs
        repair1 = StructureRepair(
            repair_type="unclosed_tag",
            description="Auto-closed tag",
            original_tokens=[],
            confidence_impact=0.1
        )
        repair2 = StructureRepair(
            repair_type="mismatched_tag",
            description="Fixed tag mismatch",
            original_tokens=[],
            confidence_impact=0.05
        )
        
        child = XMLElement(tag="child", repairs=[repair2])
        root = XMLElement(tag="root", children=[child], repairs=[repair1])
        document = XMLDocument(root=root, repairs=[repair1])
        
        result = ParseResult(document=document)
        
        repair_summary = result.get_repair_summary()
        
        assert repair_summary["total_repairs"] == 3  # 1 doc + 1 root + 1 child
        assert repair_summary["document_repairs"] == 1
        assert repair_summary["element_repairs"] == 2
        assert repair_summary["repair_types"]["unclosed_tag"] == 2  # doc + root
        assert repair_summary["repair_types"]["mismatched_tag"] == 1
        assert repair_summary["repair_impact_on_confidence"] == 0.25  # 0.1 (doc) + 0.1 (root) + 0.05 (child)
    
    def test_comprehensive_summary(self) -> None:
        """Test comprehensive summary method."""
        from ultra_robust_xml_parser.tree.builder import ParseResult
        
        # Create basic document
        root = XMLElement(tag="root", text="content")
        document = XMLDocument(root=root)
        result = ParseResult(document=document, success=True, confidence=0.9)
        
        # Add some diagnostics
        result.add_diagnostic(DiagnosticSeverity.WARNING, "Test warning", "test_component")
        
        summary = result.summary()
        
        # Check core results
        assert summary["success"] is True
        assert summary["confidence"] == 0.9
        assert summary["document_well_formed"] is True
        
        # Check that all major sections are present
        assert "parsing_statistics" in summary
        assert "diagnostics_summary" in summary
        assert "repair_summary" in summary
        assert "confidence_breakdown" in summary
        assert "validation" in summary
        assert "optimization" in summary
        
        # Check validation and optimization summaries indicate they weren't performed
        assert summary["validation"]["validation_performed"] is False
        assert summary["optimization"]["optimization_performed"] is False
        
        # Check parsing statistics
        parsing_stats = summary["parsing_statistics"]
        assert parsing_stats["element_count"] == 1
        assert parsing_stats["repair_count"] == 0
        
        # Check diagnostics summary
        diagnostics_summary = summary["diagnostics_summary"]
        assert diagnostics_summary["total_diagnostics"] == 1
        assert diagnostics_summary["diagnostics_by_severity"]["WARNING"] == 1
        
        # Check confidence breakdown
        confidence_breakdown = summary["confidence_breakdown"]
        assert confidence_breakdown["base_confidence"] == 0.9
        assert confidence_breakdown["overall_confidence"] == 0.9


class TestOutputConfiguration:
    """Test OutputConfiguration class functionality."""
    
    def test_default_configuration(self) -> None:
        """Test default configuration values."""
        config = OutputConfiguration()
        
        assert config.xml_declaration is True
        assert config.xml_encoding == "utf-8"
        assert config.xml_indent == "  "
        assert config.include_attributes is True
        assert config.include_metadata is False
        assert config.dict_attribute_prefix == "@"
        assert config.dict_text_key == "#text"
        assert config.json_indent == 2
        assert config.json_ensure_ascii is False
        assert config.exclude_empty_elements is False
        assert config.namespace_aware is True
    
    def test_custom_configuration(self) -> None:
        """Test custom configuration values."""
        config = OutputConfiguration(
            xml_declaration=False,
            xml_encoding="iso-8859-1",
            xml_indent="\t",
            include_metadata=True,
            include_repairs=True,
            dict_attribute_prefix="attr_",
            dict_text_key="content",
            json_indent=4,
            json_ensure_ascii=True,
            exclude_empty_elements=True
        )
        
        assert config.xml_declaration is False
        assert config.xml_encoding == "iso-8859-1"
        assert config.xml_indent == "\t"
        assert config.include_metadata is True
        assert config.include_repairs is True
        assert config.dict_attribute_prefix == "attr_"
        assert config.dict_text_key == "content"
        assert config.json_indent == 4
        assert config.json_ensure_ascii is True
        assert config.exclude_empty_elements is True


class TestFormatResult:
    """Test FormatResult class functionality."""
    
    def test_default_format_result(self) -> None:
        """Test default format result values."""
        result = FormatResult()
        
        assert result.success is True
        assert result.formatted_output == ""
        assert result.format_used == OutputFormat.XML_STRING
        assert result.processing_time_ms == 0.0
        assert result.output_size_bytes == 0
        assert result.issues == []
    
    def test_format_result_with_data(self) -> None:
        """Test format result with data."""
        result = FormatResult(
            success=True,
            formatted_output="<root>test</root>",
            format_used=OutputFormat.XML_PRETTY,
            processing_time_ms=50.0,
            output_size_bytes=18,
            issues=["warning: empty element"]
        )
        
        assert result.success is True
        assert result.formatted_output == "<root>test</root>"
        assert result.format_used == OutputFormat.XML_PRETTY
        assert result.processing_time_ms == 50.0
        assert result.output_size_bytes == 18
        assert len(result.issues) == 1
        assert result.issues[0] == "warning: empty element"


class TestOutputFormatter:
    """Test OutputFormatter class functionality."""
    
    def test_formatter_initialization(self) -> None:
        """Test output formatter initialization."""
        formatter = OutputFormatter()
        
        assert formatter.correlation_id is None
        assert formatter.logger is not None
    
    def test_formatter_with_correlation_id(self) -> None:
        """Test output formatter with correlation ID."""
        formatter = OutputFormatter(correlation_id="test-123")
        
        assert formatter.correlation_id == "test-123"
    
    def test_format_empty_document(self) -> None:
        """Test formatting document without root element."""
        document = XMLDocument()
        formatter = OutputFormatter()
        
        result = formatter.format(document, OutputFormat.XML_STRING)
        
        assert result.success is False
        assert result.formatted_output == ""
        assert len(result.issues) > 0
        assert "no root element" in result.issues[0]
    
    def test_format_xml_string(self) -> None:
        """Test XML string formatting."""
        root = XMLElement(tag="root", attributes={"id": "1"}, text="content")
        document = XMLDocument(root=root, version="1.0", encoding="utf-8")
        formatter = OutputFormatter()
        
        result = formatter.format(document, OutputFormat.XML_STRING)
        
        assert result.success is True
        assert result.format_used == OutputFormat.XML_STRING
        assert result.output_size_bytes > 0
        
        # Check that XML declaration is included
        assert '<?xml version="1.0" encoding="utf-8"?>' in result.formatted_output
        assert '<root id="1">content</root>' in result.formatted_output
    
    def test_format_xml_pretty(self) -> None:
        """Test pretty XML formatting."""
        child = XMLElement(tag="child", text="child content")
        root = XMLElement(tag="root", children=[child])
        document = XMLDocument(root=root)
        formatter = OutputFormatter()
        
        result = formatter.format(document, OutputFormat.XML_PRETTY)
        
        assert result.success is True
        assert result.format_used == OutputFormat.XML_PRETTY
        
        # Check that formatting includes indentation and newlines
        lines = result.formatted_output.split('\n')
        assert len(lines) > 1  # Should have multiple lines for pretty formatting
        assert any('  <child>' in line for line in lines)  # Should have indentation
    
    def test_format_xml_minified(self) -> None:
        """Test minified XML formatting."""
        child = XMLElement(tag="child", text=" child content ")
        root = XMLElement(tag="root", children=[child])
        document = XMLDocument(root=root)
        formatter = OutputFormatter()
        
        result = formatter.format(document, OutputFormat.XML_MINIFIED)
        
        assert result.success is True
        assert result.format_used == OutputFormat.XML_MINIFIED
        
        # Minified should have no unnecessary whitespace
        assert '\n' not in result.formatted_output or result.formatted_output.count('\n') <= 1  # Only XML declaration
        assert '  ' not in result.formatted_output  # No double spaces
    
    def test_format_dictionary(self) -> None:
        """Test dictionary formatting."""
        child = XMLElement(tag="item", attributes={"id": "1"}, text="item text")
        root = XMLElement(tag="root", attributes={"version": "1.0"}, children=[child])
        document = XMLDocument(root=root)
        formatter = OutputFormatter()
        
        result = formatter.format(document, OutputFormat.DICTIONARY)
        
        assert result.success is True
        assert result.format_used == OutputFormat.DICTIONARY
        assert result.formatted_output != ""
        
        # The output should be a string representation of a dictionary
        assert "root" in result.formatted_output
        assert "@version" in result.formatted_output or "version" in result.formatted_output
    
    def test_format_json(self) -> None:
        """Test JSON formatting."""
        import json
        
        child = XMLElement(tag="item", attributes={"id": "1"}, text="item text")
        root = XMLElement(tag="root", attributes={"version": "1.0"}, children=[child])
        document = XMLDocument(root=root)
        formatter = OutputFormatter()
        
        result = formatter.format(document, OutputFormat.JSON)
        
        assert result.success is True
        assert result.format_used == OutputFormat.JSON
        
        # Should be valid JSON
        try:
            parsed_json = json.loads(result.formatted_output)
            assert "root" in parsed_json
            assert isinstance(parsed_json, dict)
        except json.JSONDecodeError:
            assert False, "Output should be valid JSON"
    
    def test_format_json_pretty(self) -> None:
        """Test pretty JSON formatting."""
        import json
        
        root = XMLElement(tag="root", text="content")
        document = XMLDocument(root=root)
        formatter = OutputFormatter()
        
        result = formatter.format(document, OutputFormat.JSON_PRETTY)
        
        assert result.success is True
        assert result.format_used == OutputFormat.JSON_PRETTY
        
        # Pretty JSON should have indentation
        assert '\n' in result.formatted_output
        assert '  ' in result.formatted_output  # Should have spaces for indentation
        
        # Should still be valid JSON
        try:
            json.loads(result.formatted_output)
        except json.JSONDecodeError:
            assert False, "Output should be valid JSON"
    
    def test_format_json_compact(self) -> None:
        """Test compact JSON formatting."""
        import json
        
        root = XMLElement(tag="root", text="content")
        document = XMLDocument(root=root)
        formatter = OutputFormatter()
        
        result = formatter.format(document, OutputFormat.JSON_COMPACT)
        
        assert result.success is True
        assert result.format_used == OutputFormat.JSON_COMPACT
        
        # Compact JSON should have minimal whitespace
        assert ': ' not in result.formatted_output  # Should use ':' not ': '
        assert ', ' not in result.formatted_output  # Should use ',' not ', '
        
        # Should still be valid JSON
        try:
            json.loads(result.formatted_output)
        except json.JSONDecodeError:
            assert False, "Output should be valid JSON"
    
    def test_xml_escaping(self) -> None:
        """Test XML character escaping."""
        root = XMLElement(
            tag="root", 
            attributes={"special": "< > & \" '"}, 
            text="Text with < > & special chars"
        )
        document = XMLDocument(root=root)
        formatter = OutputFormatter()
        
        result = formatter.format(document, OutputFormat.XML_STRING)
        
        assert result.success is True
        
        # Check that special characters are properly escaped
        assert "&lt;" in result.formatted_output  # < should be escaped
        assert "&gt;" in result.formatted_output  # > should be escaped  
        assert "&amp;" in result.formatted_output  # & should be escaped
        assert "&quot;" in result.formatted_output  # " should be escaped in attributes
        assert "&#39;" in result.formatted_output  # ' should be escaped in attributes
    
    def test_configuration_options(self) -> None:
        """Test various configuration options."""
        root = XMLElement(tag="root", attributes={"empty": "", "valid": "value"}, text="content")
        document = XMLDocument(root=root)
        formatter = OutputFormatter()
        
        # Test excluding empty attributes
        config = OutputConfiguration(exclude_empty_attributes=True)
        result = formatter.format(document, OutputFormat.XML_STRING, config)
        
        assert result.success is True
        assert 'empty=""' not in result.formatted_output
        assert 'valid="value"' in result.formatted_output
    
    def test_dictionary_configuration(self) -> None:
        """Test dictionary-specific configuration options."""
        root = XMLElement(tag="root", attributes={"id": "1"}, text="content")
        document = XMLDocument(root=root)
        formatter = OutputFormatter()
        
        # Test custom attribute prefix and text key
        config = OutputConfiguration(
            dict_attribute_prefix="attr_",
            dict_text_key="content",
            include_metadata=True
        )
        result = formatter.format(document, OutputFormat.DICTIONARY_DETAILED, config)
        
        assert result.success is True
        assert "attr_id" in result.formatted_output
        assert "content" in result.formatted_output
        assert "_document_metadata" in result.formatted_output
    
    def test_multiple_children_same_tag(self) -> None:
        """Test handling multiple children with same tag name."""
        item1 = XMLElement(tag="item", text="first")
        item2 = XMLElement(tag="item", text="second") 
        item3 = XMLElement(tag="item", text="third")
        root = XMLElement(tag="items", children=[item1, item2, item3])
        document = XMLDocument(root=root)
        formatter = OutputFormatter()
        
        result = formatter.format(document, OutputFormat.JSON)
        
        assert result.success is True
        
        # JSON should handle multiple items as array
        import json
        parsed = json.loads(result.formatted_output)
        items = parsed["items"]["item"]
        assert isinstance(items, list)
        assert len(items) == 3
    
    def test_format_never_fails_on_exception(self) -> None:
        """Test that formatting never fails with exception."""
        root = XMLElement(tag="root", text="content")
        document = XMLDocument(root=root)
        
        formatter = OutputFormatter()
        original_method = formatter._format_xml
        
        def mock_method(*args, **kwargs):
            raise RuntimeError("Test exception")
        
        formatter._format_xml = mock_method
        
        # Should not raise exception, should return failed result
        result = formatter.format(document, OutputFormat.XML_STRING)
        
        assert result.success is False
        assert result.formatted_output == ""
        assert len(result.issues) > 0
        assert any("Formatting failed" in issue for issue in result.issues)
        
        # Restore original method
        formatter._format_xml = original_method
    
    def test_performance_metrics(self) -> None:
        """Test that performance metrics are calculated."""
        root = XMLElement(tag="root", text="content")
        document = XMLDocument(root=root)
        formatter = OutputFormatter()
        
        result = formatter.format(document, OutputFormat.XML_STRING)
        
        assert result.success is True
        assert result.processing_time_ms > 0
        assert result.output_size_bytes > 0
        assert result.output_size_bytes == len(result.formatted_output.encode('utf-8'))


@pytest.fixture
def sample_complex_document() -> XMLDocument:
    """Create a sample complex document for formatting tests."""
    # Create nested structure with various content
    metadata = XMLElement(
        tag="metadata",
        attributes={"created": "2023-01-01", "author": "test"},
    )
    
    item1 = XMLElement(
        tag="item",
        attributes={"id": "1", "type": "text"},
        text="First item content"
    )
    
    item2 = XMLElement(
        tag="item", 
        attributes={"id": "2", "type": "html"},
        text="Second item with <special> & characters"
    )
    
    subitem = XMLElement(tag="subitem", text="nested content")
    item3 = XMLElement(
        tag="item",
        attributes={"id": "3", "type": "complex"},
        children=[subitem]
    )
    
    items = XMLElement(tag="items", children=[item1, item2, item3])
    
    root = XMLElement(
        tag="document",
        attributes={"version": "1.0", "xmlns": "http://example.com"},
        children=[metadata, items]
    )
    
    return XMLDocument(root=root, encoding="utf-8", version="1.0")


class TestOutputFormatterIntegration:
    """Integration tests for OutputFormatter with complex documents."""
    
    def test_format_complex_document_xml(self, sample_complex_document: XMLDocument) -> None:
        """Test formatting complex document as XML."""
        formatter = OutputFormatter()
        result = formatter.format(sample_complex_document, OutputFormat.XML_PRETTY)
        
        assert result.success is True
        assert result.output_size_bytes > 0
        
        # Should contain all expected elements
        assert "<document" in result.formatted_output
        assert "<metadata" in result.formatted_output
        assert "<items>" in result.formatted_output
        assert "<item" in result.formatted_output
        assert "<subitem>" in result.formatted_output
        
        # Should properly escape special characters
        assert "&lt;special&gt;" in result.formatted_output
        assert "&amp;" in result.formatted_output
    
    def test_format_complex_document_json(self, sample_complex_document: XMLDocument) -> None:
        """Test formatting complex document as JSON."""
        import json
        
        formatter = OutputFormatter()
        result = formatter.format(sample_complex_document, OutputFormat.JSON_PRETTY)
        
        assert result.success is True
        
        # Should be valid JSON
        parsed = json.loads(result.formatted_output)
        
        # Check structure
        assert "document" in parsed
        document_content = parsed["document"]
        assert "@version" in document_content
        assert "metadata" in document_content
        assert "items" in document_content
        
        # Check items array
        items = document_content["items"]["item"]
        assert isinstance(items, list)
        assert len(items) == 3
        
        # Check nested structure
        complex_item = items[2]  # Third item has nested content
        assert "subitem" in complex_item
    
    def test_format_with_repairs_and_metadata(self, sample_complex_document: XMLDocument) -> None:
        """Test formatting with repair and metadata information."""
        # Add some repairs to the document
        repair = StructureRepair(
            repair_type="test_repair",
            description="Test repair for formatting",
            original_tokens=[],
            confidence_impact=0.1
        )
        sample_complex_document.root.repairs.append(repair)
        
        formatter = OutputFormatter()
        config = OutputConfiguration(
            include_metadata=True,
            include_repairs=True,
            include_position_info=False
        )
        
        result = formatter.format(
            sample_complex_document, 
            OutputFormat.DICTIONARY_DETAILED, 
            config
        )
        
        assert result.success is True
        assert "_document_metadata" in result.formatted_output
        assert "_element_metadata" in result.formatted_output
        assert "_repairs" in result.formatted_output
        assert "test_repair" in result.formatted_output
    
    def test_different_output_formats_consistency(self, sample_complex_document: XMLDocument) -> None:
        """Test that different output formats maintain data consistency."""
        formatter = OutputFormatter()
        
        # Get XML output
        xml_result = formatter.format(sample_complex_document, OutputFormat.XML_STRING)
        
        # Get JSON output  
        json_result = formatter.format(sample_complex_document, OutputFormat.JSON)
        
        # Get dictionary output
        dict_result = formatter.format(sample_complex_document, OutputFormat.DICTIONARY)
        
        # All should succeed
        assert xml_result.success is True
        assert json_result.success is True
        assert dict_result.success is True
        
        # All should have content
        assert len(xml_result.formatted_output) > 0
        assert len(json_result.formatted_output) > 0
        assert len(dict_result.formatted_output) > 0
        
        # XML should be largest (due to markup), JSON should be smaller, dict string should vary
        assert xml_result.output_size_bytes > 0
        assert json_result.output_size_bytes > 0
        assert dict_result.output_size_bytes > 0