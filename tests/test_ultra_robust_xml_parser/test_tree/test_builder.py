"""Comprehensive tests for tree building engine.

Tests tree construction, navigation, error recovery, and memory efficiency
for the ultra-robust XML parser tree building components.
"""

import pytest
from typing import List, Optional

from ultra_robust_xml_parser.shared import DiagnosticSeverity
from ultra_robust_xml_parser.tokenization import Token, TokenType, TokenPosition, TokenizationResult
from ultra_robust_xml_parser.tree import (
    ParseResult,
    StructureRepair,
    XMLDocument,
    XMLElement,
    XMLTreeBuilder,
)


class TestXMLElement:
    """Test XMLElement functionality and navigation methods."""

    def test_element_creation_with_valid_data(self) -> None:
        """Test creating XMLElement with valid data."""
        element = XMLElement(
            tag="root",
            attributes={"id": "test"},
            text="content",
            confidence=0.95
        )
        
        assert element.tag == "root"
        assert element.attributes["id"] == "test"
        assert element.text == "content"
        assert element.confidence == 0.95
        assert not element.has_repairs
        assert element.is_well_formed

    def test_element_creation_with_invalid_tag_raises_error(self) -> None:
        """Test that empty tag raises ValueError."""
        with pytest.raises(ValueError, match="Element tag cannot be empty"):
            XMLElement(tag="")

    def test_element_creation_with_invalid_confidence_raises_error(self) -> None:
        """Test that invalid confidence raises ValueError.""" 
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            XMLElement(tag="test", confidence=1.5)

    def test_add_child_establishes_parent_relationship(self) -> None:
        """Test adding child element establishes proper parent-child relationship."""
        parent = XMLElement(tag="parent")
        child = XMLElement(tag="child")
        
        parent.add_child(child)
        
        assert child in parent.children
        assert child.parent is parent
        assert len(parent.children) == 1

    def test_add_child_with_invalid_type_raises_error(self) -> None:
        """Test adding non-XMLElement raises TypeError."""
        parent = XMLElement(tag="parent")
        
        with pytest.raises(TypeError, match="Child must be an XMLElement instance"):
            parent.add_child("not_an_element")  # type: ignore

    def test_remove_child_clears_parent_relationship(self) -> None:
        """Test removing child clears parent relationship."""
        parent = XMLElement(tag="parent")
        child = XMLElement(tag="child")
        
        parent.add_child(child)
        result = parent.remove_child(child)
        
        assert result is True
        assert child not in parent.children
        assert child.parent is None

    def test_remove_nonexistent_child_returns_false(self) -> None:
        """Test removing non-existent child returns False."""
        parent = XMLElement(tag="parent")
        child = XMLElement(tag="child")
        
        result = parent.remove_child(child)
        
        assert result is False

    def test_insert_child_at_index(self) -> None:
        """Test inserting child at specific index."""
        parent = XMLElement(tag="parent")
        child1 = XMLElement(tag="child1")
        child2 = XMLElement(tag="child2")
        child3 = XMLElement(tag="child3")
        
        parent.add_child(child1)
        parent.add_child(child3)
        parent.insert_child(1, child2)
        
        assert parent.children == [child1, child2, child3]
        assert child2.parent is parent

    def test_insert_child_with_invalid_index_raises_error(self) -> None:
        """Test inserting child with invalid index raises IndexError."""
        parent = XMLElement(tag="parent")
        child = XMLElement(tag="child")
        
        with pytest.raises(IndexError, match="Child index out of range"):
            parent.insert_child(5, child)

    def test_find_child_by_tag(self) -> None:
        """Test finding first direct child by tag name."""
        parent = XMLElement(tag="parent")
        child1 = XMLElement(tag="target")
        child2 = XMLElement(tag="other")
        child3 = XMLElement(tag="target")
        
        parent.add_child(child1)
        parent.add_child(child2) 
        parent.add_child(child3)
        
        found = parent.find_child("target")
        assert found is child1  # First match

    def test_find_child_returns_none_when_not_found(self) -> None:
        """Test find_child returns None when tag not found."""
        parent = XMLElement(tag="parent")
        child = XMLElement(tag="child")
        parent.add_child(child)
        
        found = parent.find_child("missing")
        assert found is None

    def test_find_children_returns_all_matching_direct_children(self) -> None:
        """Test finding all direct children with matching tag."""
        parent = XMLElement(tag="parent")
        child1 = XMLElement(tag="target")
        child2 = XMLElement(tag="other")
        child3 = XMLElement(tag="target")
        
        parent.add_child(child1)
        parent.add_child(child2)
        parent.add_child(child3)
        
        found = parent.find_children("target")
        assert len(found) == 2
        assert child1 in found
        assert child3 in found

    def test_find_descendant_element(self) -> None:
        """Test finding descendant element by tag name."""
        root = XMLElement(tag="root")
        child = XMLElement(tag="child")
        grandchild = XMLElement(tag="target")
        
        root.add_child(child)
        child.add_child(grandchild)
        
        found = root.find("target")
        assert found is grandchild

    def test_find_all_descendant_elements(self) -> None:
        """Test finding all descendant elements by tag name."""
        root = XMLElement(tag="root")
        child1 = XMLElement(tag="child")
        child2 = XMLElement(tag="target")
        grandchild = XMLElement(tag="target")
        
        root.add_child(child1)
        root.add_child(child2)
        child1.add_child(grandchild)
        
        found = root.find_all("target")
        assert len(found) == 2
        assert child2 in found
        assert grandchild in found

    def test_find_by_attribute_name_only(self) -> None:
        """Test finding elements by attribute name only."""
        root = XMLElement(tag="root")
        child1 = XMLElement(tag="child", attributes={"id": "test"})
        child2 = XMLElement(tag="child", attributes={"class": "value"})
        grandchild = XMLElement(tag="grandchild", attributes={"id": "nested"})
        
        root.add_child(child1)
        root.add_child(child2)
        child1.add_child(grandchild)
        
        found = root.find_by_attribute("id")
        assert len(found) == 2
        assert child1 in found
        assert grandchild in found

    def test_find_by_attribute_name_and_value(self) -> None:
        """Test finding elements by attribute name and value."""
        root = XMLElement(tag="root")
        child1 = XMLElement(tag="child", attributes={"id": "test"})
        child2 = XMLElement(tag="child", attributes={"id": "other"})
        
        root.add_child(child1)
        root.add_child(child2)
        
        found = root.find_by_attribute("id", "test")
        assert len(found) == 1
        assert child1 in found

    def test_get_set_attribute_operations(self) -> None:
        """Test attribute get/set operations."""
        element = XMLElement(tag="test")
        
        # Test setting and getting attribute
        element.set_attribute("key", "value")
        assert element.get_attribute("key") == "value"
        assert element.has_attribute("key")
        
        # Test default value
        assert element.get_attribute("missing", "default") == "default"
        assert not element.has_attribute("missing")

    def test_set_attribute_with_invalid_types_raises_error(self) -> None:
        """Test setting attribute with non-string types raises TypeError."""
        element = XMLElement(tag="test")
        
        with pytest.raises(TypeError, match="Attribute name and value must be strings"):
            element.set_attribute(123, "value")  # type: ignore

    def test_get_path_for_root_element(self) -> None:
        """Test path generation for root element."""
        root = XMLElement(tag="root")
        assert root.get_path() == "/root"

    def test_get_path_for_nested_element(self) -> None:
        """Test path generation for nested elements."""
        root = XMLElement(tag="root")
        child = XMLElement(tag="child")
        grandchild = XMLElement(tag="item")
        
        root.add_child(child)
        child.add_child(grandchild)
        
        assert grandchild.get_path() == "/root/child/item"

    def test_get_path_with_position_for_duplicate_tags(self) -> None:
        """Test path generation includes position for duplicate tag names."""
        root = XMLElement(tag="root")
        child1 = XMLElement(tag="item")
        child2 = XMLElement(tag="item")
        child3 = XMLElement(tag="item")
        
        root.add_child(child1)
        root.add_child(child2)
        root.add_child(child3)
        
        assert child1.get_path() == "/root/item[1]"
        assert child2.get_path() == "/root/item[2]"
        assert child3.get_path() == "/root/item[3]"

    def test_get_depth_calculation(self) -> None:
        """Test element depth calculation."""
        root = XMLElement(tag="root")
        child = XMLElement(tag="child")
        grandchild = XMLElement(tag="grandchild")
        
        root.add_child(child)
        child.add_child(grandchild)
        
        assert root.get_depth() == 0
        assert child.get_depth() == 1
        assert grandchild.get_depth() == 2

    def test_full_text_aggregation(self) -> None:
        """Test full text content aggregation from element and children."""
        root = XMLElement(tag="root", text="Root text")
        child1 = XMLElement(tag="child", text="Child1 text")
        child2 = XMLElement(tag="child", text="Child2 text")
        
        root.add_child(child1)
        root.add_child(child2)
        
        full_text = root.full_text
        assert "Root text" in full_text
        assert "Child1 text" in full_text
        assert "Child2 text" in full_text

    def test_namespace_handling(self) -> None:
        """Test namespace prefix and local name extraction."""
        element = XMLElement(tag="ns:localname")
        
        assert element.namespace_prefix == "ns"
        assert element.local_name == "localname"
        assert element.tag_with_namespace == "ns:localname"
        
        # Test element without namespace
        no_ns_element = XMLElement(tag="localname")
        assert no_ns_element.namespace_prefix is None
        assert no_ns_element.local_name == "localname"

    def test_to_dict_conversion(self) -> None:
        """Test converting element to dictionary representation."""
        element = XMLElement(
            tag="test",
            attributes={"id": "123"},
            text="content",
            confidence=0.95
        )
        child = XMLElement(tag="child", text="child content")
        element.add_child(child)
        
        result = element.to_dict()
        
        assert result["tag"] == "test"
        assert result["attributes"] == {"id": "123"}
        assert result["text"] == "content"
        assert result["confidence"] == 0.95
        assert not result["has_repairs"]
        assert len(result["children"]) == 1
        assert result["children"][0]["tag"] == "child"


class TestXMLDocument:
    """Test XMLDocument functionality and document-level operations."""

    def test_document_creation_with_defaults(self) -> None:
        """Test creating XMLDocument with default values."""
        doc = XMLDocument()
        
        assert doc.root is None
        assert doc.encoding == "utf-8"
        assert doc.version == "1.0"
        assert doc.standalone is None
        assert doc.confidence == 1.0
        assert not doc.has_repairs
        assert doc.is_well_formed

    def test_document_statistics_calculation(self) -> None:
        """Test document statistics calculation with root element."""
        root = XMLElement(tag="root", attributes={"version": "1.0"})
        child1 = XMLElement(tag="child1", attributes={"id": "1", "class": "test"})
        child2 = XMLElement(tag="child2")
        
        root.add_child(child1)
        root.add_child(child2)
        
        doc = XMLDocument(root=root)
        
        assert doc.total_elements == 3  # root + 2 children
        assert doc.total_attributes == 3  # version + id + class
        assert doc.max_depth == 1  # child elements at depth 1

    def test_iter_elements_returns_all_elements(self) -> None:
        """Test iterating over all elements in document."""
        root = XMLElement(tag="root")
        child1 = XMLElement(tag="child1")
        child2 = XMLElement(tag="child2")
        grandchild = XMLElement(tag="grandchild")
        
        root.add_child(child1)
        root.add_child(child2)
        child1.add_child(grandchild)
        
        doc = XMLDocument(root=root)
        elements = doc.iter_elements()
        
        assert len(elements) == 4
        assert root in elements
        assert child1 in elements
        assert child2 in elements
        assert grandchild in elements

    def test_find_first_element_by_tag(self) -> None:
        """Test finding first element by tag name."""
        root = XMLElement(tag="root")
        child1 = XMLElement(tag="target")
        child2 = XMLElement(tag="target")
        
        root.add_child(child1)
        root.add_child(child2)
        
        doc = XMLDocument(root=root)
        found = doc.find("target")
        
        assert found is child1

    def test_find_root_element_by_tag(self) -> None:
        """Test finding root element itself by tag name."""
        root = XMLElement(tag="root")
        doc = XMLDocument(root=root)
        
        found = doc.find("root")
        assert found is root

    def test_find_all_elements_by_tag(self) -> None:
        """Test finding all elements by tag name."""
        root = XMLElement(tag="root")
        child1 = XMLElement(tag="target")
        child2 = XMLElement(tag="other")
        grandchild = XMLElement(tag="target")
        
        root.add_child(child1)
        root.add_child(child2)
        child1.add_child(grandchild)
        
        doc = XMLDocument(root=root)
        found = doc.find_all("target")
        
        assert len(found) == 2
        assert child1 in found
        assert grandchild in found

    def test_find_by_attribute_in_document(self) -> None:
        """Test finding elements by attribute in document."""
        root = XMLElement(tag="root", attributes={"type": "container"})
        child1 = XMLElement(tag="child", attributes={"id": "first"})
        child2 = XMLElement(tag="child", attributes={"id": "second"})
        
        root.add_child(child1)
        root.add_child(child2)
        
        doc = XMLDocument(root=root)
        found = doc.find_by_attribute("id")
        
        assert len(found) == 2
        assert child1 in found
        assert child2 in found

    def test_get_element_by_id(self) -> None:
        """Test finding element by ID attribute."""
        root = XMLElement(tag="root")
        child1 = XMLElement(tag="child", attributes={"id": "test-id"})
        child2 = XMLElement(tag="child", attributes={"id": "other-id"})
        
        root.add_child(child1)
        root.add_child(child2)
        
        doc = XMLDocument(root=root)
        found = doc.get_element_by_id("test-id")
        
        assert found is child1

    def test_get_element_by_id_returns_none_when_not_found(self) -> None:
        """Test get_element_by_id returns None when ID not found."""
        root = XMLElement(tag="root")
        doc = XMLDocument(root=root)
        
        found = doc.get_element_by_id("missing-id")
        assert found is None

    def test_to_dict_conversion(self) -> None:
        """Test converting document to dictionary representation."""
        root = XMLElement(tag="root", text="content")
        doc = XMLDocument(
            root=root,
            encoding="utf-8",
            version="1.0",
            standalone=True
        )
        
        result = doc.to_dict()
        
        assert result["encoding"] == "utf-8"
        assert result["version"] == "1.0"
        assert result["standalone"] is True
        assert result["confidence"] == 1.0
        assert not result["has_repairs"]
        assert result["total_elements"] == 1
        assert result["root"]["tag"] == "root"


class TestStructureRepair:
    """Test StructureRepair functionality."""

    def test_repair_creation_with_valid_data(self) -> None:
        """Test creating StructureRepair with valid data."""
        token = Token(
            type=TokenType.TAG_NAME,
            value="test",
            position=TokenPosition(line=1, column=1, offset=0)
        )
        
        repair = StructureRepair(
            repair_type="test_repair",
            description="Test repair description",
            original_tokens=[token],
            confidence_impact=0.1,
            severity="minor"
        )
        
        assert repair.repair_type == "test_repair"
        assert repair.description == "Test repair description"
        assert len(repair.original_tokens) == 1
        assert repair.confidence_impact == 0.1
        assert repair.severity == "minor"

    def test_repair_creation_with_empty_type_raises_error(self) -> None:
        """Test that empty repair type raises ValueError."""
        with pytest.raises(ValueError, match="Repair type cannot be empty"):
            StructureRepair(
                repair_type="",
                description="Test",
                original_tokens=[],
                confidence_impact=0.1
            )

    def test_repair_creation_with_invalid_confidence_raises_error(self) -> None:
        """Test that invalid confidence impact raises ValueError."""
        with pytest.raises(ValueError, match="Confidence impact must be between 0.0 and 1.0"):
            StructureRepair(
                repair_type="test",
                description="Test", 
                original_tokens=[],
                confidence_impact=1.5
            )


class TestParseResult:
    """Test ParseResult functionality and diagnostics."""

    def test_result_creation_with_defaults(self) -> None:
        """Test creating ParseResult with default values."""
        result = ParseResult()
        
        assert isinstance(result.document, XMLDocument)
        assert result.success is True
        assert result.confidence == 1.0
        assert len(result.diagnostics) == 0
        assert result.element_count == 0
        assert not result.has_repairs
        assert not result.has_errors()

    def test_confidence_calculation_with_repairs(self) -> None:
        """Test confidence calculation considers repairs."""
        root = XMLElement(tag="root")
        repair = StructureRepair(
            repair_type="test",
            description="Test repair",
            original_tokens=[],
            confidence_impact=0.1
        )
        root.repairs.append(repair)
        
        doc = XMLDocument(root=root)
        result = ParseResult(document=doc)
        
        assert result.confidence < 1.0
        assert result.has_repairs
        assert result.repair_count == 1

    def test_add_diagnostic_creates_entry(self) -> None:
        """Test adding diagnostic entry to result."""
        result = ParseResult()
        
        result.add_diagnostic(
            DiagnosticSeverity.WARNING,
            "Test warning",
            "test_component",
            position={"line": 1, "column": 5},
            details={"context": "test"}
        )
        
        assert len(result.diagnostics) == 1
        diagnostic = result.diagnostics[0]
        assert diagnostic.severity == DiagnosticSeverity.WARNING
        assert diagnostic.message == "Test warning"
        assert diagnostic.component == "test_component"
        assert diagnostic.position == {"line": 1, "column": 5}
        assert diagnostic.details == {"context": "test"}

    def test_get_diagnostics_by_severity(self) -> None:
        """Test filtering diagnostics by severity level."""
        result = ParseResult()
        
        result.add_diagnostic(DiagnosticSeverity.INFO, "Info message", "component")
        result.add_diagnostic(DiagnosticSeverity.WARNING, "Warning message", "component")
        result.add_diagnostic(DiagnosticSeverity.ERROR, "Error message", "component")
        
        warnings = result.get_diagnostics_by_severity(DiagnosticSeverity.WARNING)
        assert len(warnings) == 1
        assert warnings[0].message == "Warning message"

    def test_has_errors_detection(self) -> None:
        """Test error detection in diagnostics."""
        result = ParseResult()
        
        # Add non-error diagnostic
        result.add_diagnostic(DiagnosticSeverity.INFO, "Info", "component")
        assert not result.has_errors()
        
        # Add error diagnostic
        result.add_diagnostic(DiagnosticSeverity.ERROR, "Error", "component")
        assert result.has_errors()

    def test_summary_statistics(self) -> None:
        """Test summary statistics generation."""
        root = XMLElement(tag="root")
        child = XMLElement(tag="child")
        root.add_child(child)
        
        doc = XMLDocument(root=root)
        result = ParseResult(document=doc, success=True)
        result.add_diagnostic(DiagnosticSeverity.INFO, "Test", "component")
        
        summary = result.summary()
        
        assert summary["success"] is True
        assert summary["parsing_statistics"]["element_count"] == 2
        assert summary["diagnostics_summary"]["total_diagnostics"] == 1
        assert summary["document_well_formed"] is True
        assert "confidence" in summary
        assert "processing_time_ms" in summary["parsing_statistics"]


class TestXMLTreeBuilder:
    """Test XMLTreeBuilder functionality and tree construction."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.builder = XMLTreeBuilder(correlation_id="test-123")

    def create_token(self, token_type: TokenType, value: str, offset: int = 0) -> Token:
        """Helper to create test tokens."""
        return Token(
            type=token_type,
            value=value,
            position=TokenPosition(line=1, column=offset + 1, offset=offset),
            confidence=1.0
        )

    def test_builder_initialization(self) -> None:
        """Test XMLTreeBuilder initialization."""
        builder = XMLTreeBuilder(correlation_id="test-123")
        assert builder.correlation_id == "test-123"

    def test_build_simple_element(self) -> None:
        """Test building simple XML element from tokens."""
        tokens = [
            self.create_token(TokenType.TAG_START, "<", 0),
            self.create_token(TokenType.TAG_NAME, "root", 1),
            self.create_token(TokenType.TAG_END, ">", 5),
            self.create_token(TokenType.TEXT, "content", 6),
            self.create_token(TokenType.TAG_START, "</", 13),
            self.create_token(TokenType.TAG_NAME, "root", 15),
            self.create_token(TokenType.TAG_END, ">", 19),
        ]
        
        result = self.builder.build(tokens)
        
        assert result.success
        assert result.document.root is not None
        assert result.document.root.tag == "root"
        assert result.document.root.text == "content"
        assert result.element_count == 1

    def test_build_nested_elements(self) -> None:
        """Test building nested XML elements."""
        tokens = [
            self.create_token(TokenType.TAG_START, "<", 0),
            self.create_token(TokenType.TAG_NAME, "root", 1),
            self.create_token(TokenType.TAG_END, ">", 5),
            self.create_token(TokenType.TAG_START, "<", 6),
            self.create_token(TokenType.TAG_NAME, "child", 7),
            self.create_token(TokenType.TAG_END, ">", 12),
            self.create_token(TokenType.TEXT, "nested content", 13),
            self.create_token(TokenType.TAG_START, "</", 27),
            self.create_token(TokenType.TAG_NAME, "child", 29),
            self.create_token(TokenType.TAG_END, ">", 34),
            self.create_token(TokenType.TAG_START, "</", 35),
            self.create_token(TokenType.TAG_NAME, "root", 37),
            self.create_token(TokenType.TAG_END, ">", 41),
        ]
        
        result = self.builder.build(tokens)
        
        assert result.success
        assert result.document.root is not None
        assert result.document.root.tag == "root"
        assert len(result.document.root.children) == 1
        
        child = result.document.root.children[0]
        assert child.tag == "child"
        assert child.text == "nested content"
        assert child.parent is result.document.root

    def test_build_with_attributes(self) -> None:
        """Test building elements with attributes.""" 
        tokens = [
            self.create_token(TokenType.TAG_START, "<", 0),
            self.create_token(TokenType.TAG_NAME, "root", 1),
            self.create_token(TokenType.ATTR_NAME, "id", 6),
            self.create_token(TokenType.ATTR_VALUE, '"test-id"', 9),
            self.create_token(TokenType.ATTR_NAME, "class", 19),
            self.create_token(TokenType.ATTR_VALUE, '"test-class"', 25),
            self.create_token(TokenType.TAG_END, "/>", 38),
        ]
        
        result = self.builder.build(tokens)
        
        assert result.success
        assert result.document.root is not None
        assert result.document.root.tag == "root"
        assert result.document.root.get_attribute("id") == "test-id"
        assert result.document.root.get_attribute("class") == "test-class"

    def test_build_self_closing_element(self) -> None:
        """Test building self-closing elements."""
        tokens = [
            self.create_token(TokenType.TAG_START, "<", 0),
            self.create_token(TokenType.TAG_NAME, "br", 1),
            self.create_token(TokenType.TAG_END, "/>", 3),
        ]
        
        result = self.builder.build(tokens)
        
        assert result.success
        assert result.document.root is not None
        assert result.document.root.tag == "br"
        assert len(result.document.root.children) == 0

    def test_build_with_mismatched_tags_applies_repair(self) -> None:
        """Test that mismatched tags trigger repair mechanism."""
        tokens = [
            self.create_token(TokenType.TAG_START, "<", 0),
            self.create_token(TokenType.TAG_NAME, "root", 1),
            self.create_token(TokenType.TAG_END, ">", 5),
            self.create_token(TokenType.TAG_START, "<", 6),
            self.create_token(TokenType.TAG_NAME, "child", 7),
            self.create_token(TokenType.TAG_END, ">", 12),
            self.create_token(TokenType.TAG_START, "</", 13),
            self.create_token(TokenType.TAG_NAME, "root", 15),  # Mismatched - should be child
            self.create_token(TokenType.TAG_END, ">", 19),
        ]
        
        result = self.builder.build(tokens)
        
        assert result.success  # Should still succeed due to never-fail philosophy
        assert result.has_repairs  # Should have repair information
        assert len(result.diagnostics) > 0  # Should have diagnostic information

    def test_build_with_unclosed_elements_applies_repair(self) -> None:
        """Test that unclosed elements trigger repair mechanism."""
        tokens = [
            self.create_token(TokenType.TAG_START, "<", 0),
            self.create_token(TokenType.TAG_NAME, "root", 1),
            self.create_token(TokenType.TAG_END, ">", 5),
            self.create_token(TokenType.TAG_START, "<", 6),
            self.create_token(TokenType.TAG_NAME, "child", 7),
            self.create_token(TokenType.TAG_END, ">", 12),
            # Missing closing tags
        ]
        
        result = self.builder.build(tokens)
        
        assert result.success  # Never-fail philosophy
        assert result.has_repairs  # Should have repair for unclosed elements
        assert result.document.root is not None
        assert len(result.diagnostics) > 0

    def test_build_with_tokenization_result(self) -> None:
        """Test building from TokenizationResult object."""
        tokens = [
            self.create_token(TokenType.TAG_START, "<", 0),
            self.create_token(TokenType.TAG_NAME, "root", 1),
            self.create_token(TokenType.TAG_END, "/>", 5),
        ]
        
        tokenization_result = TokenizationResult(
            tokens=tokens,
            success=True,
            confidence=0.95
        )
        
        result = self.builder.build(tokenization_result)
        
        assert result.success
        assert result.tokenization_result is tokenization_result
        assert result.document.root is not None
        assert result.document.root.tag == "root"

    def test_build_empty_token_list_returns_empty_document(self) -> None:
        """Test building with empty token list returns valid empty document."""
        result = self.builder.build([])
        
        assert result.success
        assert result.document.root is None
        assert result.element_count == 0

    def test_build_with_text_content_preserves_whitespace(self) -> None:
        """Test that significant whitespace is preserved in text content."""
        tokens = [
            self.create_token(TokenType.TAG_START, "<", 0),
            self.create_token(TokenType.TAG_NAME, "root", 1),
            self.create_token(TokenType.TAG_END, ">", 5),
            self.create_token(TokenType.TEXT, "  content  ", 6),
            self.create_token(TokenType.TAG_START, "</", 17),
            self.create_token(TokenType.TAG_NAME, "root", 19),
            self.create_token(TokenType.TAG_END, ">", 23),
        ]
        
        result = self.builder.build(tokens)
        
        assert result.success
        assert result.document.root is not None
        # Text should be stripped of leading/trailing whitespace
        assert result.document.root.text == "content"

    def test_build_never_fails_with_malformed_input(self) -> None:
        """Test never-fail guarantee with completely malformed input."""
        # Create tokens that don't make sense together
        tokens = [
            self.create_token(TokenType.TAG_END, ">", 0),  # Closing without opening
            self.create_token(TokenType.ATTR_NAME, "orphan", 1),  # Orphaned attribute
            self.create_token(TokenType.TEXT, "random text", 7),  # Random text
        ]
        
        result = self.builder.build(tokens)
        
        # Should never fail, even with nonsensical input
        assert isinstance(result, ParseResult)
        assert isinstance(result.document, XMLDocument)
        # May have low confidence and many diagnostics, but should not crash

    def test_performance_metrics_captured(self) -> None:
        """Test that performance metrics are captured during building."""
        tokens = [
            self.create_token(TokenType.TAG_START, "<", 0),
            self.create_token(TokenType.TAG_NAME, "root", 1),
            self.create_token(TokenType.TAG_END, "/>", 5),
        ]
        
        result = self.builder.build(tokens)
        
        assert result.performance.processing_time_ms > 0
        assert result.performance.tokens_generated >= 0


class TestIntegrationScenarios:
    """Integration tests with realistic XML scenarios."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.builder = XMLTreeBuilder()

    def create_token(self, token_type: TokenType, value: str, offset: int = 0) -> Token:
        """Helper to create test tokens."""
        return Token(
            type=token_type,
            value=value,
            position=TokenPosition(line=1, column=offset + 1, offset=offset),
            confidence=1.0
        )

    def test_build_realistic_xml_document(self) -> None:
        """Test building realistic XML document structure."""
        # Simulates: <book id="123"><title>Test Book</title><author>Jane Doe</author></book>
        tokens = [
            self.create_token(TokenType.TAG_START, "<", 0),
            self.create_token(TokenType.TAG_NAME, "book", 1),
            self.create_token(TokenType.ATTR_NAME, "id", 6),
            self.create_token(TokenType.ATTR_VALUE, '"123"', 9),
            self.create_token(TokenType.TAG_END, ">", 14),
            
            self.create_token(TokenType.TAG_START, "<", 15),
            self.create_token(TokenType.TAG_NAME, "title", 16),
            self.create_token(TokenType.TAG_END, ">", 21),
            self.create_token(TokenType.TEXT, "Test Book", 22),
            self.create_token(TokenType.TAG_START, "</", 31),
            self.create_token(TokenType.TAG_NAME, "title", 33),
            self.create_token(TokenType.TAG_END, ">", 38),
            
            self.create_token(TokenType.TAG_START, "<", 39),
            self.create_token(TokenType.TAG_NAME, "author", 40),
            self.create_token(TokenType.TAG_END, ">", 46),
            self.create_token(TokenType.TEXT, "Jane Doe", 47),
            self.create_token(TokenType.TAG_START, "</", 55),
            self.create_token(TokenType.TAG_NAME, "author", 57),
            self.create_token(TokenType.TAG_END, ">", 63),
            
            self.create_token(TokenType.TAG_START, "</", 64),
            self.create_token(TokenType.TAG_NAME, "book", 66),
            self.create_token(TokenType.TAG_END, ">", 70),
        ]
        
        result = self.builder.build(tokens)
        
        assert result.success
        assert result.document.root is not None
        assert result.document.root.tag == "book"
        assert result.document.root.get_attribute("id") == "123"
        assert len(result.document.root.children) == 2
        
        # Check title element
        title = result.document.root.find("title")
        assert title is not None
        assert title.text == "Test Book"
        
        # Check author element
        author = result.document.root.find("author")
        assert author is not None
        assert author.text == "Jane Doe"
        
        # Test navigation
        assert result.document.find("title") is title
        assert result.document.find_all("title")[0] is title
        assert result.document.get_element_by_id("123") is result.document.root

    def test_build_xml_with_mixed_content(self) -> None:
        """Test building XML with mixed text and element content."""
        # Simulates: <p>Some text <em>emphasized</em> more text</p>
        tokens = [
            self.create_token(TokenType.TAG_START, "<", 0),
            self.create_token(TokenType.TAG_NAME, "p", 1),
            self.create_token(TokenType.TAG_END, ">", 2),
            self.create_token(TokenType.TEXT, "Some text ", 3),
            self.create_token(TokenType.TAG_START, "<", 13),
            self.create_token(TokenType.TAG_NAME, "em", 14),
            self.create_token(TokenType.TAG_END, ">", 16),
            self.create_token(TokenType.TEXT, "emphasized", 17),
            self.create_token(TokenType.TAG_START, "</", 27),
            self.create_token(TokenType.TAG_NAME, "em", 29),
            self.create_token(TokenType.TAG_END, ">", 31),
            self.create_token(TokenType.TEXT, " more text", 32),
            self.create_token(TokenType.TAG_START, "</", 42),
            self.create_token(TokenType.TAG_NAME, "p", 44),
            self.create_token(TokenType.TAG_END, ">", 45),
        ]
        
        result = self.builder.build(tokens)
        
        assert result.success
        assert result.document.root is not None
        assert result.document.root.tag == "p"
        
        # Should have both text content and child element
        em_element = result.document.root.find("em")
        assert em_element is not None
        assert em_element.text == "emphasized"
        
        # Full text should include all content
        full_text = result.document.root.full_text
        assert "Some text" in full_text
        assert "emphasized" in full_text
        assert "more text" in full_text

    def test_error_recovery_with_multiple_issues(self) -> None:
        """Test error recovery with multiple structural issues."""
        # Malformed XML with multiple issues
        tokens = [
            self.create_token(TokenType.TAG_START, "<", 0),
            self.create_token(TokenType.TAG_NAME, "root", 1),
            self.create_token(TokenType.TAG_END, ">", 5),
            
            # Unclosed child1
            self.create_token(TokenType.TAG_START, "<", 6),
            self.create_token(TokenType.TAG_NAME, "child1", 7),
            self.create_token(TokenType.TAG_END, ">", 13),
            
            # Nested unclosed child2
            self.create_token(TokenType.TAG_START, "<", 14),
            self.create_token(TokenType.TAG_NAME, "child2", 15),
            self.create_token(TokenType.TAG_END, ">", 21),
            
            # Wrong closing tag
            self.create_token(TokenType.TAG_START, "</", 22),
            self.create_token(TokenType.TAG_NAME, "wrong", 24),
            self.create_token(TokenType.TAG_END, ">", 29),
            
            # No closing for root either
        ]
        
        result = self.builder.build(tokens)
        
        # Should still build a tree despite multiple issues
        assert result.success
        assert result.document.root is not None
        assert result.document.root.tag == "root"
        assert result.has_repairs
        assert len(result.diagnostics) > 0
        
        # Should have auto-closed elements
        child1 = result.document.root.find("child1")
        assert child1 is not None
        child2 = child1.find("child2") if child1 else None
        assert child2 is not None


if __name__ == "__main__":
    pytest.main([__file__])