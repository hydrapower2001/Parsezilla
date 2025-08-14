"""Tests for the automated test case generation module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ultra_robust_xml_parser.tools.testing import (
    ComplexityLevel,
    CorpusBasedGenerator,
    EdgeCaseGenerator,
    MalformationType,
    TestCase,
    TestCaseGenerator,
    TestSuite,
    XMLMalformationGenerator,
)


class TestTestCase:
    """Test TestCase data class."""
    
    def test_test_case_creation(self):
        """Test test case creation."""
        test_case = TestCase(
            id="test_001",
            content="<root>test</root>",
            malformation_type=MalformationType.UNCLOSED_TAG,
            complexity_level=ComplexityLevel.SIMPLE,
            expected_success=False,
            metadata={"info": "test"},
            source="generated"
        )
        
        assert test_case.id == "test_001"
        assert test_case.content == "<root>test</root>"
        assert test_case.malformation_type == MalformationType.UNCLOSED_TAG
        assert test_case.complexity_level == ComplexityLevel.SIMPLE
        assert test_case.expected_success is False
        assert test_case.metadata["info"] == "test"
        assert test_case.source == "generated"
    
    def test_test_case_to_dict(self):
        """Test test case conversion to dictionary."""
        test_case = TestCase(
            id="test_002",
            content="<root/>",
            malformation_type=None,
            complexity_level=ComplexityLevel.MODERATE,
            expected_success=True
        )
        
        case_dict = test_case.to_dict()
        
        assert case_dict["id"] == "test_002"
        assert case_dict["content"] == "<root/>"
        assert case_dict["malformation_type"] is None
        assert case_dict["complexity_level"] == "moderate"
        assert case_dict["expected_success"] is True


class TestTestSuite:
    """Test TestSuite data class."""
    
    def test_test_suite_creation(self):
        """Test test suite creation."""
        suite = TestSuite(
            name="test_suite",
            generation_config={"setting": "value"}
        )
        
        assert suite.name == "test_suite"
        assert suite.size == 0
        assert suite.generation_config["setting"] == "value"
    
    def test_test_suite_properties(self):
        """Test test suite computed properties."""
        suite = TestSuite("test_suite")
        
        # Add test cases
        suite.test_cases.append(TestCase(
            "case1", "<root/>", MalformationType.UNCLOSED_TAG, ComplexityLevel.SIMPLE, False
        ))
        suite.test_cases.append(TestCase(
            "case2", "<root/>", MalformationType.UNCLOSED_TAG, ComplexityLevel.MODERATE, False
        ))
        suite.test_cases.append(TestCase(
            "case3", "<root/>", MalformationType.MISMATCHED_TAG, ComplexityLevel.SIMPLE, False
        ))
        suite.test_cases.append(TestCase(
            "case4", "<root/>", None, ComplexityLevel.COMPLEX, True
        ))
        
        assert suite.size == 4
        
        # Check malformation coverage
        coverage = suite.malformation_coverage
        assert coverage["unclosed_tag"] == 2
        assert coverage["mismatched_tag"] == 1
        
        # Check complexity distribution
        distribution = suite.complexity_distribution
        assert distribution["simple"] == 2
        assert distribution["moderate"] == 1
        assert distribution["complex"] == 1


class TestXMLMalformationGenerator:
    """Test XMLMalformationGenerator class."""
    
    def test_generator_creation(self):
        """Test malformation generator creation."""
        generator = XMLMalformationGenerator(seed=42)
        assert generator is not None
    
    def test_generate_unclosed_tag(self):
        """Test unclosed tag generation."""
        generator = XMLMalformationGenerator(seed=42)
        
        # Test with default XML
        result = generator.generate_unclosed_tag()
        assert result is not None
        assert len(result) > 0
        
        # Test with custom XML
        base_xml = "<root><item>content</item></root>"
        result = generator.generate_unclosed_tag(base_xml)
        assert result != base_xml  # Should be modified
        assert "<item>" in result or "<root>" in result  # Some tags should remain
    
    def test_generate_mismatched_tag(self):
        """Test mismatched tag generation."""
        generator = XMLMalformationGenerator(seed=42)
        
        base_xml = "<root><item>content</item></root>"
        result = generator.generate_mismatched_tag(base_xml)
        
        # Should contain mismatched tags
        assert result != base_xml
        assert "<" in result and ">" in result
    
    def test_generate_invalid_character(self):
        """Test invalid character generation."""
        generator = XMLMalformationGenerator(seed=42)
        
        base_xml = "<root><item>content</item></root>"
        result = generator.generate_invalid_character(base_xml)
        
        # Should contain the base structure but with invalid chars
        assert "root" in result or "item" in result
    
    def test_generate_broken_attribute(self):
        """Test broken attribute generation."""
        generator = XMLMalformationGenerator(seed=42)
        
        base_xml = '<root><item id="value">content</item></root>'
        result = generator.generate_broken_attribute(base_xml)
        
        # Should be different from original
        assert result != base_xml
    
    def test_generate_nested_overflow(self):
        """Test nested overflow generation."""
        generator = XMLMalformationGenerator()
        
        result = generator.generate_nested_overflow(depth=10)
        
        # Should contain nested elements
        assert "<level0>" in result
        assert "</level9>" in result
        assert "content" in result
    
    def test_generate_entity_overflow(self):
        """Test entity overflow generation."""
        generator = XMLMalformationGenerator()
        
        result = generator.generate_entity_overflow(multiplier=5)
        
        # Should contain entity definitions
        assert "<!ENTITY" in result
        assert "<!DOCTYPE" in result
        assert "&lol" in result
    
    def test_generate_cdata_corruption(self):
        """Test CDATA corruption generation."""
        generator = XMLMalformationGenerator(seed=42)
        
        base_xml = "<root><![CDATA[Some content]]></root>"
        result = generator.generate_cdata_corruption(base_xml)
        
        # Should be modified
        assert result != base_xml
        assert "CDATA" in result or "root" in result
    
    def test_generate_namespace_error(self):
        """Test namespace error generation."""
        generator = XMLMalformationGenerator(seed=42)
        
        base_xml = '<root xmlns:ns="http://example.com"><ns:item>content</ns:item></root>'
        result = generator.generate_namespace_error(base_xml)
        
        # Should be different from original
        assert result != base_xml


class TestEdgeCaseGenerator:
    """Test EdgeCaseGenerator class."""
    
    def test_generator_creation(self):
        """Test edge case generator creation."""
        generator = EdgeCaseGenerator(seed=42)
        assert generator is not None
    
    def test_generate_empty_document(self):
        """Test empty document generation."""
        generator = EdgeCaseGenerator()
        result = generator.generate_empty_document()
        assert result == ""
    
    def test_generate_whitespace_only(self):
        """Test whitespace-only generation."""
        generator = EdgeCaseGenerator(seed=42)
        result = generator.generate_whitespace_only()
        
        assert len(result) > 0
        assert result.strip() == ""  # Only whitespace
    
    def test_generate_single_character(self):
        """Test single character generation."""
        generator = EdgeCaseGenerator(seed=42)
        result = generator.generate_single_character()
        
        assert len(result) == 1
        assert result in ['<', '>', '&', '"', "'", '=']
    
    def test_generate_huge_attribute_value(self):
        """Test huge attribute value generation."""
        generator = EdgeCaseGenerator()
        result = generator.generate_huge_attribute_value(size=100)
        
        assert len(result) > 100  # Should be longer due to XML structure
        assert 'attr="' in result
        assert 'x' * 100 in result  # The huge value
    
    def test_generate_many_attributes(self):
        """Test many attributes generation."""
        generator = EdgeCaseGenerator()
        result = generator.generate_many_attributes(count=10)
        
        # Should contain many attributes
        assert result.count('attr') == 10
        assert result.count('=') == 10
    
    def test_generate_unicode_extremes(self):
        """Test unicode extremes generation."""
        generator = EdgeCaseGenerator(seed=42)
        result = generator.generate_unicode_extremes()
        
        assert "<root>" in result
        assert "</root>" in result
        assert len(result) > len("<root></root>")  # Should contain unicode chars
    
    def test_generate_processing_instruction_edge_cases(self):
        """Test processing instruction edge cases."""
        generator = EdgeCaseGenerator(seed=42)
        result = generator.generate_processing_instruction_edge_cases()
        
        assert "<?" in result  # Should contain PI
        assert "<root/>" in result  # Should have root element


class TestCorpusBasedGenerator:
    """Test CorpusBasedGenerator class."""
    
    def test_generator_creation(self):
        """Test corpus-based generator creation."""
        generator = CorpusBasedGenerator()
        assert generator is not None
    
    def test_analyze_corpus_empty(self):
        """Test corpus analysis with empty corpus."""
        generator = CorpusBasedGenerator([])
        generator.analyze_corpus()
        
        assert len(generator.patterns.get('tags', [])) == 0
        assert len(generator.patterns.get('attributes', [])) == 0
    
    def test_analyze_corpus_with_files(self):
        """Test corpus analysis with actual files."""
        # Create temporary XML files
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus_path = Path(temp_dir) / "sample.xml"
            corpus_path.write_text(
                '<root xmlns="http://example.com" id="test">'
                '<item class="sample">content</item>'
                '</root>'
            )
            
            generator = CorpusBasedGenerator([corpus_path])
            generator.analyze_corpus()
            
            # Should have extracted some patterns
            assert 'tags' in generator.patterns
            assert 'attributes' in generator.patterns
    
    def test_generate_from_patterns(self):
        """Test pattern-based generation."""
        generator = CorpusBasedGenerator()
        
        # Test without patterns
        result = generator.generate_from_patterns()
        assert "No corpus patterns available" in result
        
        # Test with patterns
        generator.patterns = {
            'tags': ['<item id="1">'],
            'attributes': [' class="test"']
        }
        result = generator.generate_from_patterns()
        assert len(result) > 0
    
    def test_generate_structure_variation(self):
        """Test structure variation generation."""
        generator = CorpusBasedGenerator()
        
        # Test without structures
        result = generator.generate_structure_variation()
        assert "No corpus structures available" in result
        
        # Test with structures
        generator.structures = ['<item>content</item>']
        result = generator.generate_structure_variation()
        assert len(result) > 0


class TestTestCaseGenerator:
    """Test TestCaseGenerator class."""
    
    def test_generator_creation(self):
        """Test test case generator creation."""
        generator = TestCaseGenerator(seed=42)
        assert generator is not None
        assert generator.seed == 42
    
    def test_generate_malformed_cases(self):
        """Test malformed case generation."""
        generator = TestCaseGenerator(seed=42)
        
        malformation_types = [MalformationType.UNCLOSED_TAG, MalformationType.MISMATCHED_TAG]
        cases = generator.generate_malformed_cases(malformation_types, count=5)
        
        assert len(cases) == 5
        for case in cases:
            assert isinstance(case, TestCase)
            assert case.malformation_type in malformation_types
            assert case.expected_success is False
            assert "malformed_" in case.id
    
    def test_generate_edge_cases(self):
        """Test edge case generation."""
        generator = TestCaseGenerator(seed=42)
        
        cases = generator.generate_edge_cases(count=5)
        
        assert len(cases) <= 5  # May be less due to generation failures
        for case in cases:
            assert isinstance(case, TestCase)
            assert case.malformation_type is None
            assert case.expected_success is True
            assert "edge_" in case.id
    
    def test_generate_corpus_based_cases(self):
        """Test corpus-based case generation."""
        generator = TestCaseGenerator(seed=42)
        
        cases = generator.generate_corpus_based_cases(count=3)
        
        assert len(cases) == 3
        for case in cases:
            assert isinstance(case, TestCase)
            assert case.malformation_type is None
            assert case.expected_success is True
            assert "corpus_" in case.id
    
    def test_generate_test_suite(self):
        """Test comprehensive test suite generation."""
        generator = TestCaseGenerator(seed=42)
        
        suite = generator.generate_test_suite(
            "test_suite",
            total_cases=10,
            malformation_ratio=0.6,
            edge_case_ratio=0.3,
            corpus_ratio=0.1
        )
        
        assert isinstance(suite, TestSuite)
        assert suite.name == "test_suite"
        assert suite.size == 10
        
        # Check generation config
        config = suite.generation_config
        assert config["total_cases"] == 10
        assert config["malformation_ratio"] == 0.6
        
        # Should have test cases
        assert len(suite.test_cases) == 10
    
    @patch('ultra_robust_xml_parser.tools.testing.UltraRobustXMLParser')
    def test_validate_test_suite(self, mock_parser_class):
        """Test test suite validation."""
        # Mock parser and result
        mock_parser = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_parser.parse.return_value = mock_result
        mock_parser_class.return_value = mock_parser
        
        generator = TestCaseGenerator()
        
        # Create simple test suite
        suite = TestSuite("validation_test")
        suite.test_cases.append(TestCase(
            "case1", "<root/>", None, ComplexityLevel.SIMPLE, True
        ))
        suite.test_cases.append(TestCase(
            "case2", "<root", MalformationType.UNCLOSED_TAG, ComplexityLevel.SIMPLE, False
        ))
        
        # Validate suite
        results = generator.validate_test_suite(suite)
        
        assert results["total_cases"] == 2
        assert results["successful_parses"] == 2  # Mock always returns success
        assert "performance_stats" in results
        assert "coverage_analysis" in results
    
    def test_export_test_suite_json(self):
        """Test JSON export of test suite."""
        generator = TestCaseGenerator()
        
        suite = TestSuite("export_test")
        suite.test_cases.append(TestCase(
            "case1", "<root/>", None, ComplexityLevel.SIMPLE, True
        ))
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            export_path = Path(f.name)
        
        try:
            generator.export_test_suite(suite, export_path, "json")
            
            assert export_path.exists()
            
            # Verify exported content
            exported_data = json.loads(export_path.read_text())
            assert exported_data["name"] == "export_test"
            assert len(exported_data["test_cases"]) == 1
            assert exported_data["metadata"]["size"] == 1
            
        finally:
            export_path.unlink()
    
    def test_export_test_suite_csv(self):
        """Test CSV export of test suite."""
        generator = TestCaseGenerator()
        
        suite = TestSuite("csv_export_test")
        suite.test_cases.append(TestCase(
            "case1", "<root/>", MalformationType.UNCLOSED_TAG, ComplexityLevel.SIMPLE, False
        ))
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            export_path = Path(f.name)
        
        try:
            generator.export_test_suite(suite, export_path, "csv")
            
            assert export_path.exists()
            
            # Verify CSV content
            content = export_path.read_text()
            assert "id,malformation_type,complexity_level,expected_success,source" in content
            assert "case1,unclosed_tag,simple,False,generated" in content
            
        finally:
            export_path.unlink()


class TestMalformationTypes:
    """Test malformation type enum."""
    
    def test_malformation_types_exist(self):
        """Test that all expected malformation types exist."""
        expected_types = [
            "unclosed_tag",
            "mismatched_tag", 
            "invalid_character",
            "broken_attribute",
            "missing_declaration",
            "invalid_encoding",
            "nested_overflow",
            "entity_overflow",
            "cdata_corruption",
            "namespace_error"
        ]
        
        for expected_type in expected_types:
            assert hasattr(MalformationType, expected_type.upper())
            assert MalformationType[expected_type.upper()].value == expected_type


class TestComplexityLevels:
    """Test complexity level enum."""
    
    def test_complexity_levels_exist(self):
        """Test that all complexity levels exist."""
        expected_levels = ["simple", "moderate", "complex", "extreme"]
        
        for expected_level in expected_levels:
            assert hasattr(ComplexityLevel, expected_level.upper())
            assert ComplexityLevel[expected_level.upper()].value == expected_level


@pytest.mark.integration
class TestTestCaseGeneratorIntegration:
    """Integration tests for test case generator."""
    
    def test_full_generation_workflow(self):
        """Test complete test generation workflow."""
        generator = TestCaseGenerator(seed=123)
        
        # Generate comprehensive test suite
        suite = generator.generate_test_suite(
            "integration_test",
            total_cases=20,
            malformation_ratio=0.5,
            edge_case_ratio=0.3,
            corpus_ratio=0.2
        )
        
        # Verify suite properties
        assert suite.size == 20
        assert len(suite.malformation_coverage) > 0
        assert len(suite.complexity_distribution) > 0
        
        # Test export functionality
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            export_path = Path(f.name)
        
        try:
            generator.export_test_suite(suite, export_path, "json")
            assert export_path.exists()
            
            # Verify exported structure
            exported_data = json.loads(export_path.read_text())
            assert exported_data["name"] == "integration_test"
            assert len(exported_data["test_cases"]) == 20
            
        finally:
            export_path.unlink()
    
    def test_reproducible_generation(self):
        """Test that generation is reproducible with same seed."""
        seed = 456
        
        # Generate first suite
        generator1 = TestCaseGenerator(seed=seed)
        suite1 = generator1.generate_test_suite("repro_test1", 10)
        
        # Generate second suite with same seed
        generator2 = TestCaseGenerator(seed=seed)
        suite2 = generator2.generate_test_suite("repro_test2", 10)
        
        # Should have identical content (though different names)
        assert len(suite1.test_cases) == len(suite2.test_cases)
        
        for case1, case2 in zip(suite1.test_cases, suite2.test_cases):
            # Content should be identical (IDs will differ due to suite name)
            assert case1.content == case2.content
            assert case1.malformation_type == case2.malformation_type
            assert case1.complexity_level == case2.complexity_level
    
    def test_corpus_based_generation_with_files(self):
        """Test corpus-based generation with actual files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sample corpus files
            corpus1 = Path(temp_dir) / "sample1.xml"
            corpus1.write_text(
                '<?xml version="1.0"?>'
                '<document xmlns="http://example.com">'
                '<section id="intro" class="primary">'
                '<title>Introduction</title>'
                '<content>Sample content here</content>'
                '</section>'
                '</document>'
            )
            
            corpus2 = Path(temp_dir) / "sample2.xml"
            corpus2.write_text(
                '<root>'
                '<items>'
                '<item id="1" type="data">First item</item>'
                '<item id="2" type="info">Second item</item>'
                '</items>'
                '</root>'
            )
            
            # Generate test cases based on corpus
            generator = TestCaseGenerator(corpus_paths=[corpus1, corpus2])
            
            # Generate corpus-based cases
            corpus_cases = generator.generate_corpus_based_cases(5)
            
            assert len(corpus_cases) == 5
            for case in corpus_cases:
                assert case.source == "generated"
                assert case.expected_success is True
                assert "corpus_" in case.id