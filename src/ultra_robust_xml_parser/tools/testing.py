"""Automated test case generation tools for Ultra Robust XML Parser.

Provides comprehensive test case generation including malformation generation,
edge case testing, corpus-based generation, and coverage analysis.
"""

import random
import re
import string
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union

from ultra_robust_xml_parser import UltraRobustXMLParser, parse_string
from ultra_robust_xml_parser.shared.logging import get_logger


class MalformationType(Enum):
    """Types of XML malformations to generate."""
    
    UNCLOSED_TAG = "unclosed_tag"
    MISMATCHED_TAG = "mismatched_tag"
    INVALID_CHARACTER = "invalid_character"
    BROKEN_ATTRIBUTE = "broken_attribute"
    MISSING_DECLARATION = "missing_declaration"
    INVALID_ENCODING = "invalid_encoding"
    NESTED_OVERFLOW = "nested_overflow"
    ENTITY_OVERFLOW = "entity_overflow"
    CDATA_CORRUPTION = "cdata_corruption"
    NAMESPACE_ERROR = "namespace_error"


class ComplexityLevel(Enum):
    """Test case complexity levels."""
    
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXTREME = "extreme"


@dataclass
class TestCase:
    """Generated test case with metadata."""
    
    id: str
    content: str
    malformation_type: Optional[MalformationType]
    complexity_level: ComplexityLevel
    expected_success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = "generated"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert test case to dictionary representation."""
        return {
            "id": self.id,
            "content": self.content,
            "malformation_type": self.malformation_type.value if self.malformation_type else None,
            "complexity_level": self.complexity_level.value,
            "expected_success": self.expected_success,
            "metadata": self.metadata,
            "source": self.source
        }


@dataclass
class TestSuite:
    """Collection of test cases with analysis."""
    
    name: str
    test_cases: List[TestCase] = field(default_factory=list)
    generation_config: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def size(self) -> int:
        """Number of test cases in the suite."""
        return len(self.test_cases)
    
    @property
    def malformation_coverage(self) -> Dict[str, int]:
        """Count of test cases by malformation type."""
        coverage = {}
        for test_case in self.test_cases:
            if test_case.malformation_type:
                maltype = test_case.malformation_type.value
                coverage[maltype] = coverage.get(maltype, 0) + 1
        return coverage
    
    @property
    def complexity_distribution(self) -> Dict[str, int]:
        """Distribution of test cases by complexity level."""
        distribution = {}
        for test_case in self.test_cases:
            level = test_case.complexity_level.value
            distribution[level] = distribution.get(level, 0) + 1
        return distribution


class XMLMalformationGenerator:
    """Generator for malformed XML test cases."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize malformation generator.
        
        Args:
            seed: Random seed for reproducible generation
        """
        if seed is not None:
            random.seed(seed)
        self.logger = get_logger(__name__, None, "malformation_generator")
    
    def generate_unclosed_tag(self, base_xml: str = None) -> str:
        """Generate XML with unclosed tags."""
        if not base_xml:
            base_xml = f"<root><item>content</item><another>more content"
        
        # Remove random closing tags
        tags = re.findall(r'</[^>]+>', base_xml)
        if tags:
            tag_to_remove = random.choice(tags)
            return base_xml.replace(tag_to_remove, "", 1)
        
        return base_xml
    
    def generate_mismatched_tag(self, base_xml: str = None) -> str:
        """Generate XML with mismatched opening/closing tags."""
        if not base_xml:
            base_xml = "<root><item>content</item></root>"
        
        # Find tag pairs and mismatch them
        open_tags = re.findall(r'<([^/>][^>]*)>', base_xml)
        if open_tags:
            tag_to_change = random.choice(open_tags)
            new_tag = f"changed_{tag_to_change}"
            # Change only the closing tag
            pattern = f"</{re.escape(tag_to_change)}>"
            return re.sub(pattern, f"</{new_tag}>", base_xml, count=1)
        
        return base_xml
    
    def generate_invalid_character(self, base_xml: str = None) -> str:
        """Generate XML with invalid characters."""
        if not base_xml:
            base_xml = "<root><item>content</item></root>"
        
        # Insert invalid XML characters
        invalid_chars = ['\x00', '\x01', '\x08', '\x0B', '\x0C', '\x0E', '\x1F']
        invalid_char = random.choice(invalid_chars)
        
        # Insert at random position in content
        content_match = re.search(r'>([^<]+)<', base_xml)
        if content_match:
            content = content_match.group(1)
            insert_pos = random.randint(0, len(content))
            new_content = content[:insert_pos] + invalid_char + content[insert_pos:]
            return base_xml.replace(content, new_content, 1)
        
        return base_xml
    
    def generate_broken_attribute(self, base_xml: str = None) -> str:
        """Generate XML with broken attributes."""
        if not base_xml:
            base_xml = '<root><item id="value">content</item></root>'
        
        # Break attributes in various ways
        break_types = [
            lambda x: x.replace('="', '='),  # Remove quote
            lambda x: x.replace('"', ''),    # Remove all quotes
            lambda x: x.replace('=', ''),    # Remove equals sign
            lambda x: x + ' broken="',       # Unclosed quote
        ]
        
        break_func = random.choice(break_types)
        return break_func(base_xml)
    
    def generate_nested_overflow(self, depth: int = 1000) -> str:
        """Generate deeply nested XML that might cause stack overflow."""
        opening_tags = "".join(f"<level{i}>" for i in range(depth))
        closing_tags = "".join(f"</level{depth-1-i}>" for i in range(depth))
        return f"{opening_tags}content{closing_tags}"
    
    def generate_entity_overflow(self, multiplier: int = 1000) -> str:
        """Generate XML with entity expansion attacks (billion laughs style)."""
        entities = []
        for i in range(10):
            if i == 0:
                entities.append(f'<!ENTITY lol{i} "lol">')
            else:
                refs = "&lol" + str(i-1) + ";" * multiplier
                entities.append(f'<!ENTITY lol{i} "{refs}">')
        
        entity_declarations = "".join(entities)
        return f'<!DOCTYPE root [{entity_declarations}]><root>&lol9;</root>'
    
    def generate_cdata_corruption(self, base_xml: str = None) -> str:
        """Generate XML with corrupted CDATA sections."""
        if not base_xml:
            base_xml = "<root><![CDATA[Some content here]]></root>"
        
        # Corrupt CDATA in various ways
        corruptions = [
            lambda x: x.replace("]]>", "]>"),      # Remove one bracket
            lambda x: x.replace("<![CDATA[", "![CDATA["),  # Remove opening <
            lambda x: x.replace("]]>", "]]>>"),    # Add extra >
            lambda x: x.replace("<![CDATA[", "<![CDATA[]]><![CDATA["),  # Split CDATA
        ]
        
        corruption = random.choice(corruptions)
        return corruption(base_xml)
    
    def generate_namespace_error(self, base_xml: str = None) -> str:
        """Generate XML with namespace errors."""
        if not base_xml:
            base_xml = '<root xmlns:ns="http://example.com"><ns:item>content</ns:item></root>'
        
        # Create namespace errors
        errors = [
            lambda x: x.replace('xmlns:ns="http://example.com"', ''),  # Remove namespace declaration
            lambda x: x.replace('ns:', 'undefined:'),  # Use undefined prefix
            lambda x: x.replace('"http://example.com"', ''),  # Empty namespace URI
        ]
        
        error_func = random.choice(errors)
        return error_func(base_xml)
    
    def generate_missing_declaration(self, base_xml: str = None) -> str:
        """Generate XML with missing or malformed XML declaration."""
        if not base_xml:
            base_xml = '<?xml version="1.0" encoding="UTF-8"?><root>content</root>'
        
        # Create declaration issues
        issues = [
            lambda x: x.replace('<?xml version="1.0" encoding="UTF-8"?>', ''),  # Remove declaration
            lambda x: x.replace('<?xml', '<?XML'),  # Wrong case
            lambda x: x.replace('version="1.0"', 'version="2.0"'),  # Invalid version
            lambda x: x.replace('encoding="UTF-8"', 'encoding=UTF-8'),  # Missing quotes
        ]
        
        issue_func = random.choice(issues)
        return issue_func(base_xml)
    
    def generate_invalid_encoding(self, base_xml: str = None) -> str:
        """Generate XML with invalid encoding declarations."""
        if not base_xml:
            base_xml = '<?xml version="1.0" encoding="UTF-8"?><root>content</root>'
        
        # Create encoding issues
        issues = [
            lambda x: x.replace('encoding="UTF-8"', 'encoding="INVALID-ENCODING"'),
            lambda x: x.replace('encoding="UTF-8"', 'encoding="UTF-99"'),
            lambda x: x.replace('UTF-8', 'iso-8859-999'),  # Invalid ISO encoding
        ]
        
        issue_func = random.choice(issues)
        return issue_func(base_xml)


class EdgeCaseGenerator:
    """Generator for XML edge cases and boundary conditions."""
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
        self.logger = get_logger(__name__, None, "edge_case_generator")
    
    def generate_empty_document(self) -> str:
        """Generate empty XML document."""
        return ""
    
    def generate_whitespace_only(self) -> str:
        """Generate document with only whitespace."""
        whitespace_chars = [' ', '\t', '\n', '\r']
        return ''.join(random.choice(whitespace_chars) for _ in range(50))
    
    def generate_single_character(self) -> str:
        """Generate single character documents."""
        chars = ['<', '>', '&', '"', "'", '=']
        return random.choice(chars)
    
    def generate_huge_attribute_value(self, size: int = 10000) -> str:
        """Generate XML with extremely large attribute values."""
        huge_value = 'x' * size
        return f'<root attr="{huge_value}">content</root>'
    
    def generate_many_attributes(self, count: int = 1000) -> str:
        """Generate element with many attributes."""
        attrs = [f'attr{i}="value{i}"' for i in range(count)]
        attr_string = ' '.join(attrs)
        return f'<root {attr_string}>content</root>'
    
    def generate_unicode_extremes(self) -> str:
        """Generate XML with extreme Unicode characters."""
        # High Unicode code points and special characters (avoiding problematic surrogates)
        extreme_chars = [
            '\U0001F600',  # Emoji
            '\u2603',      # Snowman (safe high Unicode)
            '\u00A0',      # Non-breaking space
            '\u200B',      # Zero-width space
            '\u3042',      # Hiragana character
            '\u05D0',      # Hebrew character
            '\u0391',      # Greek character
        ]
        
        content = ''.join(random.choice(extreme_chars) for _ in range(10))
        return f'<root>{content}</root>'
    
    def generate_processing_instruction_edge_cases(self) -> str:
        """Generate edge cases for processing instructions."""
        cases = [
            '<?xml version="1.0"?><?target data?>',
            '<?xml-stylesheet type="text/xsl" href="style.xsl"?><?xml version="1.0"?>',
            '<?xml version="1.0"?><??>',  # Empty PI
            '<?xml version="1.0"?><? ?>',  # PI with space
        ]
        
        return random.choice(cases) + '<root/>'


class CorpusBasedGenerator:
    """Generate test cases based on real-world XML corpus analysis."""
    
    def __init__(self, corpus_paths: List[Path] = None):
        """Initialize corpus-based generator.
        
        Args:
            corpus_paths: Paths to XML files for corpus analysis
        """
        self.corpus_paths = corpus_paths or []
        self.patterns: Dict[str, List[str]] = {}
        self.structures: List[str] = []
        self.logger = get_logger(__name__, None, "corpus_generator")
    
    def analyze_corpus(self) -> None:
        """Analyze corpus to extract patterns and structures."""
        tag_patterns = set()
        attribute_patterns = set()
        
        for corpus_path in self.corpus_paths:
            try:
                if corpus_path.exists() and corpus_path.is_file():
                    content = corpus_path.read_text(encoding='utf-8', errors='ignore')
                    
                    # Extract tag patterns
                    tags = re.findall(r'<[^!?/][^>]*>', content)
                    tag_patterns.update(tags[:100])  # Limit for memory
                    
                    # Extract attribute patterns  
                    attrs = re.findall(r'\s+\w+\s*=\s*["\'][^"\']*["\']', content)
                    attribute_patterns.update(attrs[:100])
                    
                    # Store structure samples
                    if len(self.structures) < 50:
                        # Extract small XML fragments
                        fragments = re.findall(r'<[^>]+>.*?</[^>]+>', content)
                        self.structures.extend(fragments[:10])
                        
            except Exception as e:
                self.logger.warning(f"Failed to analyze corpus file {corpus_path}: {e}")
        
        self.patterns['tags'] = list(tag_patterns)
        self.patterns['attributes'] = list(attribute_patterns)
        
        self.logger.info(
            "Corpus analysis completed",
            extra={
                "tag_patterns": len(self.patterns['tags']),
                "attribute_patterns": len(self.patterns['attributes']),
                "structures": len(self.structures)
            }
        )
    
    def generate_from_patterns(self) -> str:
        """Generate XML based on analyzed patterns."""
        if not self.patterns:
            return "<root>No corpus patterns available</root>"
        
        # Combine random patterns
        if self.patterns['tags']:
            tag = random.choice(self.patterns['tags'])
            # Modify to create variations
            modified_tag = tag.replace('>', ' generated="true">')
            return f"{modified_tag}Generated content</{''.join(tag.split()[0])[1:]}>"
        
        return "<root>Fallback content</root>"
    
    def generate_structure_variation(self) -> str:
        """Generate variations of corpus structures."""
        if not self.structures:
            return "<root>No corpus structures available</root>"
        
        structure = random.choice(self.structures)
        # Create variations by modifying content or attributes
        variations = [
            lambda x: re.sub(r'>([^<]+)<', r'>modified_\1<', x),
            lambda x: x.replace('="', '_var="'),
            lambda x: f'<wrapper>{x}</wrapper>',
        ]
        
        variation_func = random.choice(variations)
        return variation_func(structure)


class TestCaseGenerator:
    """Comprehensive test case generator for XML parser robustness testing.
    
    Generates various types of test cases including malformed XML, edge cases,
    boundary conditions, and corpus-based variations to thoroughly test parser
    robustness and error handling capabilities.
    
    Examples:
        Basic test generation:
        >>> generator = TestCaseGenerator()
        >>> suite = generator.generate_test_suite("robustness_tests", 100)
        >>> print(f"Generated {suite.size} test cases")
        
        Targeted malformation testing:
        >>> generator = TestCaseGenerator()
        >>> malformed_cases = generator.generate_malformed_cases(
        ...     [MalformationType.UNCLOSED_TAG, MalformationType.MISMATCHED_TAG], 
        ...     count=50
        ... )
        
        Corpus-based generation:
        >>> generator = TestCaseGenerator(corpus_paths=[Path("samples.xml")])
        >>> corpus_suite = generator.generate_corpus_based_suite(25)
    """
    
    def __init__(
        self,
        seed: Optional[int] = None,
        corpus_paths: Optional[List[Path]] = None
    ):
        """Initialize test case generator.
        
        Args:
            seed: Random seed for reproducible generation
            corpus_paths: Paths to XML files for corpus-based generation
        """
        self.seed = seed
        if seed is not None:
            random.seed(seed)
        
        self.malformation_generator = XMLMalformationGenerator(seed)
        self.edge_case_generator = EdgeCaseGenerator(seed)
        self.corpus_generator = CorpusBasedGenerator(corpus_paths)
        
        self.logger = get_logger(__name__, None, "test_case_generator")
        
        # Initialize corpus analysis if paths provided
        if corpus_paths:
            self.corpus_generator.analyze_corpus()
    
    def generate_test_suite(
        self,
        suite_name: str,
        total_cases: int,
        malformation_ratio: float = 0.6,
        edge_case_ratio: float = 0.3,
        corpus_ratio: float = 0.1
    ) -> TestSuite:
        """Generate a comprehensive test suite.
        
        Args:
            suite_name: Name for the test suite
            total_cases: Total number of test cases to generate
            malformation_ratio: Proportion of malformed test cases
            edge_case_ratio: Proportion of edge case test cases
            corpus_ratio: Proportion of corpus-based test cases
            
        Returns:
            TestSuite containing generated test cases
        """
        suite = TestSuite(
            name=suite_name,
            generation_config={
                "total_cases": total_cases,
                "malformation_ratio": malformation_ratio,
                "edge_case_ratio": edge_case_ratio,
                "corpus_ratio": corpus_ratio,
                "seed": self.seed
            }
        )
        
        # Calculate case counts
        malformation_count = int(total_cases * malformation_ratio)
        edge_case_count = int(total_cases * edge_case_ratio)
        corpus_count = int(total_cases * corpus_ratio)
        
        # Adjust for rounding
        remaining = total_cases - (malformation_count + edge_case_count + corpus_count)
        malformation_count += remaining
        
        self.logger.info(
            "Generating test suite",
            extra={
                "suite_name": suite_name,
                "malformation_cases": malformation_count,
                "edge_cases": edge_case_count,
                "corpus_cases": corpus_count
            }
        )
        
        # Generate malformed cases
        malformed_cases = self.generate_malformed_cases(
            list(MalformationType),
            malformation_count
        )
        suite.test_cases.extend(malformed_cases)
        
        # Generate edge cases
        edge_cases = self.generate_edge_cases(edge_case_count)
        suite.test_cases.extend(edge_cases)
        
        # Generate corpus-based cases
        corpus_cases = self.generate_corpus_based_cases(corpus_count)
        suite.test_cases.extend(corpus_cases)
        
        self.logger.info(
            "Test suite generation completed",
            extra={
                "suite_name": suite_name,
                "actual_cases": len(suite.test_cases),
                "malformation_coverage": len(suite.malformation_coverage),
                "complexity_distribution": suite.complexity_distribution
            }
        )
        
        return suite
    
    def generate_malformed_cases(
        self,
        malformation_types: List[MalformationType],
        count: int
    ) -> List[TestCase]:
        """Generate malformed XML test cases.
        
        Args:
            malformation_types: Types of malformations to generate
            count: Number of test cases to generate
            
        Returns:
            List of malformed test cases
        """
        cases = []
        
        for i in range(count):
            malformation_type = random.choice(malformation_types)
            complexity = random.choice(list(ComplexityLevel))
            
            # Generate base XML based on complexity
            if complexity == ComplexityLevel.SIMPLE:
                base_xml = "<root><item>content</item></root>"
            elif complexity == ComplexityLevel.MODERATE:
                base_xml = '<root xmlns="http://example.com"><item id="1" class="test">content</item><item id="2">more</item></root>'
            elif complexity == ComplexityLevel.COMPLEX:
                base_xml = '''<?xml version="1.0" encoding="UTF-8"?>
                <root xmlns:ns="http://example.com" xmlns:other="http://other.com">
                    <ns:section>
                        <item id="1" class="primary">
                            <title>Test Title</title>
                            <content><![CDATA[Some CDATA content]]></content>
                        </item>
                        <other:metadata>
                            <created>2024-01-01</created>
                            <author>Test Author</author>
                        </other:metadata>
                    </ns:section>
                </root>'''
            else:  # EXTREME
                base_xml = self.malformation_generator.generate_nested_overflow(100)
            
            # Apply malformation
            if malformation_type == MalformationType.UNCLOSED_TAG:
                content = self.malformation_generator.generate_unclosed_tag(base_xml)
            elif malformation_type == MalformationType.MISMATCHED_TAG:
                content = self.malformation_generator.generate_mismatched_tag(base_xml)
            elif malformation_type == MalformationType.INVALID_CHARACTER:
                content = self.malformation_generator.generate_invalid_character(base_xml)
            elif malformation_type == MalformationType.BROKEN_ATTRIBUTE:
                content = self.malformation_generator.generate_broken_attribute(base_xml)
            elif malformation_type == MalformationType.NESTED_OVERFLOW:
                content = self.malformation_generator.generate_nested_overflow(1000)
            elif malformation_type == MalformationType.ENTITY_OVERFLOW:
                content = self.malformation_generator.generate_entity_overflow(100)
            elif malformation_type == MalformationType.CDATA_CORRUPTION:
                content = self.malformation_generator.generate_cdata_corruption(base_xml)
            elif malformation_type == MalformationType.NAMESPACE_ERROR:
                content = self.malformation_generator.generate_namespace_error(base_xml)
            elif malformation_type == MalformationType.MISSING_DECLARATION:
                content = self.malformation_generator.generate_missing_declaration(base_xml)
            elif malformation_type == MalformationType.INVALID_ENCODING:
                content = self.malformation_generator.generate_invalid_encoding(base_xml)
            else:
                content = base_xml  # Fallback
            
            test_case = TestCase(
                id=f"malformed_{malformation_type.value}_{i:04d}",
                content=content,
                malformation_type=malformation_type,
                complexity_level=complexity,
                expected_success=False,  # Malformed cases should be handled gracefully
                metadata={
                    "generation_method": "malformation",
                    "base_complexity": complexity.value
                }
            )
            
            cases.append(test_case)
        
        return cases
    
    def generate_edge_cases(self, count: int) -> List[TestCase]:
        """Generate edge case test cases.
        
        Args:
            count: Number of edge cases to generate
            
        Returns:
            List of edge case test cases
        """
        cases = []
        
        edge_generators = [
            ("empty_document", self.edge_case_generator.generate_empty_document),
            ("whitespace_only", self.edge_case_generator.generate_whitespace_only),
            ("single_character", self.edge_case_generator.generate_single_character),
            ("huge_attribute", lambda: self.edge_case_generator.generate_huge_attribute_value(5000)),
            ("many_attributes", lambda: self.edge_case_generator.generate_many_attributes(100)),
            ("unicode_extremes", self.edge_case_generator.generate_unicode_extremes),
            ("processing_instruction", self.edge_case_generator.generate_processing_instruction_edge_cases),
        ]
        
        for i in range(count):
            edge_type, generator_func = random.choice(edge_generators)
            
            try:
                content = generator_func()
                complexity = ComplexityLevel.COMPLEX if "huge" in edge_type or "many" in edge_type else ComplexityLevel.MODERATE
                
                test_case = TestCase(
                    id=f"edge_{edge_type}_{i:04d}",
                    content=content,
                    malformation_type=None,
                    complexity_level=complexity,
                    expected_success=True,  # Edge cases should still parse successfully
                    metadata={
                        "generation_method": "edge_case",
                        "edge_type": edge_type
                    }
                )
                
                cases.append(test_case)
                
            except Exception as e:
                self.logger.warning(f"Failed to generate edge case {edge_type}: {e}")
        
        return cases
    
    def generate_corpus_based_cases(self, count: int) -> List[TestCase]:
        """Generate corpus-based test cases.
        
        Args:
            count: Number of corpus-based cases to generate
            
        Returns:
            List of corpus-based test cases
        """
        cases = []
        
        for i in range(count):
            generation_method = random.choice(["patterns", "structures"])
            
            if generation_method == "patterns":
                content = self.corpus_generator.generate_from_patterns()
            else:
                content = self.corpus_generator.generate_structure_variation()
            
            test_case = TestCase(
                id=f"corpus_{generation_method}_{i:04d}",
                content=content,
                malformation_type=None,
                complexity_level=ComplexityLevel.MODERATE,
                expected_success=True,
                metadata={
                    "generation_method": "corpus_based",
                    "corpus_method": generation_method
                }
            )
            
            cases.append(test_case)
        
        return cases
    
    def validate_test_suite(self, test_suite: TestSuite, parser: UltraRobustXMLParser = None) -> Dict[str, Any]:
        """Validate test suite by running cases through the parser.
        
        Args:
            test_suite: Test suite to validate
            parser: Parser instance to use (creates new if None)
            
        Returns:
            Validation results and coverage analysis
        """
        if parser is None:
            parser = UltraRobustXMLParser()
        
        results = {
            "total_cases": len(test_suite.test_cases),
            "successful_parses": 0,
            "failed_parses": 0,
            "parser_errors": 0,
            "expectation_mismatches": 0,
            "coverage_analysis": {},
            "performance_stats": {
                "total_time_ms": 0,
                "average_time_ms": 0,
                "max_time_ms": 0,
                "min_time_ms": float('inf')
            }
        }
        
        for test_case in test_suite.test_cases:
            try:
                import time
                start_time = time.time()
                
                parse_result = parser.parse(test_case.content)
                
                processing_time = (time.time() - start_time) * 1000
                results["performance_stats"]["total_time_ms"] += processing_time
                results["performance_stats"]["max_time_ms"] = max(
                    results["performance_stats"]["max_time_ms"], processing_time
                )
                results["performance_stats"]["min_time_ms"] = min(
                    results["performance_stats"]["min_time_ms"], processing_time
                )
                
                if parse_result.success:
                    results["successful_parses"] += 1
                else:
                    results["failed_parses"] += 1
                
                # Check expectations
                if parse_result.success != test_case.expected_success:
                    results["expectation_mismatches"] += 1
                
            except Exception as e:
                results["parser_errors"] += 1
                self.logger.warning(f"Parser error for test case {test_case.id}: {e}")
        
        # Calculate averages
        if results["total_cases"] > 0:
            results["performance_stats"]["average_time_ms"] = (
                results["performance_stats"]["total_time_ms"] / results["total_cases"]
            )
        
        # Coverage analysis
        results["coverage_analysis"] = {
            "malformation_types_covered": len(test_suite.malformation_coverage),
            "complexity_levels_covered": len(test_suite.complexity_distribution),
            "malformation_distribution": test_suite.malformation_coverage,
            "complexity_distribution": test_suite.complexity_distribution
        }
        
        self.logger.info(
            "Test suite validation completed",
            extra={
                "suite_name": test_suite.name,
                "total_cases": results["total_cases"],
                "successful_parses": results["successful_parses"],
                "expectation_mismatches": results["expectation_mismatches"]
            }
        )
        
        return results
    
    def export_test_suite(self, test_suite: TestSuite, output_path: Path, format_type: str = "json") -> None:
        """Export test suite to file.
        
        Args:
            test_suite: Test suite to export
            output_path: Path to write the test suite
            format_type: Export format ('json', 'xml', 'csv')
        """
        if format_type == "json":
            import json
            
            suite_data = {
                "name": test_suite.name,
                "generation_config": test_suite.generation_config,
                "test_cases": [case.to_dict() for case in test_suite.test_cases],
                "metadata": {
                    "size": test_suite.size,
                    "malformation_coverage": test_suite.malformation_coverage,
                    "complexity_distribution": test_suite.complexity_distribution
                }
            }
            
            # Use ensure_ascii=True to avoid Unicode encoding issues with extreme characters
            output_path.write_text(json.dumps(suite_data, indent=2, ensure_ascii=True))
            
        elif format_type == "csv":
            import csv
            
            with output_path.open('w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['id', 'malformation_type', 'complexity_level', 'expected_success', 'source']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for case in test_suite.test_cases:
                    writer.writerow({
                        'id': case.id,
                        'malformation_type': case.malformation_type.value if case.malformation_type else '',
                        'complexity_level': case.complexity_level.value,
                        'expected_success': case.expected_success,
                        'source': case.source
                    })
        
        self.logger.info(
            "Test suite exported",
            extra={
                "suite_name": test_suite.name,
                "output_path": str(output_path),
                "format": format_type,
                "case_count": len(test_suite.test_cases)
            }
        )