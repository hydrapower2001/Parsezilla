"""Tests for integration adapters framework."""

import pytest
import time
from typing import Any
from unittest.mock import Mock, patch

from ultra_robust_xml_parser.api.adapters import (
    IntegrationAdapter,
    AdapterType,
    ConversionDirection,
    AdapterMetadata,
    ConversionResult,
    AdapterRegistry,
    AdapterPerformanceProfiler,
    register_adapter,
    get_adapter,
    list_available_adapters,
    get_adapters_by_type,
    validate_adapter_compatibility,
    LxmlAdapter,
    BeautifulSoupAdapter,
    ElementTreeAdapter,
    PandasAdapter,
)
from ultra_robust_xml_parser.shared import DiagnosticSeverity
from ultra_robust_xml_parser.tree.builder import ParseResult


class TestAdapter(IntegrationAdapter):
    """Test adapter implementation for testing."""
    
    def __init__(self, correlation_id: str = None, available: bool = True):
        """Initialize test adapter."""
        super().__init__(correlation_id)
        self._available = available
        self._conversion_data = None
    
    @property
    def metadata(self) -> AdapterMetadata:
        """Get adapter metadata."""
        return AdapterMetadata(
            name="test-adapter",
            version="1.0.0",
            adapter_type=AdapterType.XML_LIBRARY,
            target_library="test-lib",
            supported_versions=["1.0.0"],
            description="Test adapter for unit testing"
        )
    
    def is_available(self) -> bool:
        """Check availability."""
        return self._available
    
    def to_target(self, parse_result: ParseResult) -> ConversionResult:
        """Convert to target format."""
        start_time = time.time()
        
        try:
            # Simulate conversion work
            converted_data = f"converted_{parse_result.tree.root.tag if parse_result.tree else 'none'}"
            processing_time = (time.time() - start_time) * 1000
            
            self._record_performance(processing_time)
            
            return ConversionResult(
                success=True,
                converted_data=converted_data,
                original_data=parse_result,
                conversion_time_ms=processing_time,
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            return self._create_error_result(str(e), parse_result, processing_time)
    
    def from_target(self, target_data: Any) -> ConversionResult:
        """Convert from target format."""
        start_time = time.time()
        
        try:
            # Simulate conversion work - return mock ParseResult
            mock_result = Mock(spec=ParseResult)
            mock_result.success = True
            mock_result.confidence = 0.9
            
            processing_time = (time.time() - start_time) * 1000
            self._record_performance(processing_time)
            
            return ConversionResult(
                success=True,
                converted_data=mock_result,
                original_data=target_data,
                conversion_time_ms=processing_time,
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            return self._create_error_result(str(e), target_data, processing_time)


class TestAdapterPerformanceProfiler:
    """Tests for AdapterPerformanceProfiler."""
    
    def test_record_conversion(self):
        """Test recording conversion performance."""
        profiler = AdapterPerformanceProfiler()
        
        profiler.record_conversion("test-adapter", 100.0)
        profiler.record_conversion("test-adapter", 150.0)
        profiler.record_conversion("test-adapter", 200.0)
        
        stats = profiler.get_statistics("test-adapter")
        
        assert stats["count"] == 3
        assert stats["average_ms"] == 150.0
        assert stats["min_ms"] == 100.0
        assert stats["max_ms"] == 200.0
        assert stats["total_ms"] == 450.0
    
    def test_get_statistics_empty(self):
        """Test getting statistics for non-existent adapter."""
        profiler = AdapterPerformanceProfiler()
        stats = profiler.get_statistics("non-existent")
        
        assert stats == {}
    
    def test_metric_limit(self):
        """Test that metrics are limited to prevent memory growth."""
        profiler = AdapterPerformanceProfiler()
        
        # Add more than the limit (1000)
        for i in range(1100):
            profiler.record_conversion("test-adapter", float(i))
        
        stats = profiler.get_statistics("test-adapter")
        assert stats["count"] == 1000  # Should be limited
    
    def test_get_all_statistics(self):
        """Test getting statistics for all adapters."""
        profiler = AdapterPerformanceProfiler()
        
        profiler.record_conversion("adapter1", 100.0)
        profiler.record_conversion("adapter2", 200.0)
        
        all_stats = profiler.get_all_statistics()
        
        assert "adapter1" in all_stats
        assert "adapter2" in all_stats
        assert all_stats["adapter1"]["average_ms"] == 100.0
        assert all_stats["adapter2"]["average_ms"] == 200.0


class TestIntegrationAdapter:
    """Tests for IntegrationAdapter base class."""
    
    def test_initialization(self):
        """Test adapter initialization."""
        adapter = TestAdapter("test-correlation-123")
        
        assert adapter.correlation_id == "test-correlation-123"
        assert adapter.metadata.name == "test-adapter"
        assert adapter.metadata.adapter_type == AdapterType.XML_LIBRARY
        assert adapter.is_available() is True
    
    def test_to_target_conversion(self):
        """Test conversion to target format."""
        adapter = TestAdapter()
        
        # Create mock parse result
        mock_parse_result = Mock(spec=ParseResult)
        mock_parse_result.tree = Mock()
        mock_parse_result.tree.root = Mock()
        mock_parse_result.tree.root.tag = "root"
        
        result = adapter.to_target(mock_parse_result)
        
        assert result.success is True
        assert result.converted_data == "converted_root"
        assert result.conversion_time_ms >= 0  # Allow zero time for very fast operations
        assert result.original_data == mock_parse_result
    
    def test_from_target_conversion(self):
        """Test conversion from target format."""
        adapter = TestAdapter()
        
        target_data = "test-target-data"
        result = adapter.from_target(target_data)
        
        assert result.success is True
        assert result.converted_data.success is True
        assert result.converted_data.confidence == 0.9
        assert result.conversion_time_ms >= 0  # Allow zero time for very fast operations
        assert result.original_data == target_data
    
    def test_validation_basic(self):
        """Test basic conversion validation."""
        adapter = TestAdapter()
        
        # Test successful validation
        result = adapter.validate_conversion(
            "original", "converted", ConversionDirection.TO_TARGET
        )
        assert result is True
        
        # Test failed validation (None result)
        result = adapter.validate_conversion(
            "original", None, ConversionDirection.TO_TARGET
        )
        assert result is False
    
    def test_validation_caching(self):
        """Test that validation results are cached."""
        adapter = TestAdapter()
        
        # First call
        result1 = adapter.validate_conversion(
            "test", "converted", ConversionDirection.TO_TARGET
        )
        
        # Second call should use cache
        result2 = adapter.validate_conversion(
            "test", "converted", ConversionDirection.TO_TARGET
        )
        
        assert result1 == result2
        assert len(adapter._validation_cache) > 0
    
    def test_performance_tracking(self):
        """Test performance metrics tracking."""
        adapter = TestAdapter()
        
        # Perform some conversions
        mock_parse_result = Mock(spec=ParseResult)
        mock_parse_result.tree = Mock()
        mock_parse_result.tree.root = Mock()
        mock_parse_result.tree.root.tag = "test"
        
        adapter.to_target(mock_parse_result)
        adapter.to_target(mock_parse_result)
        
        stats = adapter.get_performance_stats()
        assert "test-adapter" in stats
        assert stats["test-adapter"]["count"] == 2
        assert stats["test-adapter"]["average_ms"] >= 0
    
    def test_error_result_creation(self):
        """Test error result creation."""
        adapter = TestAdapter()
        
        error_result = adapter._create_error_result(
            "Test error message", "original_data", 100.0
        )
        
        assert error_result.success is False
        assert error_result.converted_data is None
        assert error_result.original_data == "original_data"
        assert error_result.conversion_time_ms == 100.0
        assert len(error_result.errors) == 1
        assert error_result.errors[0] == "Test error message"
        assert len(error_result.diagnostics) == 1
        assert error_result.diagnostics[0].severity == DiagnosticSeverity.ERROR


class TestAdapterRegistry:
    """Tests for AdapterRegistry."""
    
    def test_register_adapter(self):
        """Test adapter registration."""
        registry = AdapterRegistry()
        
        registry.register(TestAdapter)
        
        adapter = registry.get_adapter("test-adapter")
        assert adapter is not None
        assert adapter.metadata.name == "test-adapter"
    
    def test_get_adapter_with_correlation_id(self):
        """Test getting adapter with correlation ID."""
        registry = AdapterRegistry()
        registry.register(TestAdapter)
        
        adapter = registry.get_adapter("test-adapter", "test-correlation")
        
        assert adapter is not None
        assert adapter.correlation_id == "test-correlation"
    
    def test_get_non_existent_adapter(self):
        """Test getting non-existent adapter."""
        registry = AdapterRegistry()
        
        adapter = registry.get_adapter("non-existent")
        assert adapter is None
    
    def test_list_available_adapters(self):
        """Test listing available adapters."""
        registry = AdapterRegistry()
        registry.register(TestAdapter)
        
        adapters = registry.list_available_adapters()
        
        assert len(adapters) == 1
        assert adapters[0].name == "test-adapter"
        assert adapters[0].adapter_type == AdapterType.XML_LIBRARY
    
    def test_get_adapters_by_type(self):
        """Test getting adapters by type."""
        registry = AdapterRegistry()
        registry.register(TestAdapter)
        
        xml_adapters = registry.get_adapters_by_type(AdapterType.XML_LIBRARY)
        data_adapters = registry.get_adapters_by_type(AdapterType.DATA_FRAME)
        
        assert "test-adapter" in xml_adapters
        assert len(data_adapters) == 0
    
    def test_adapter_unavailable(self):
        """Test handling of unavailable adapters."""
        class UnavailableAdapter(TestAdapter):
            def __init__(self, correlation_id=None):
                super().__init__(correlation_id, available=False)
        
        registry = AdapterRegistry()
        registry.register(UnavailableAdapter)
        
        adapter = registry.get_adapter("test-adapter")
        assert adapter is None
        
        available_adapters = registry.list_available_adapters()
        assert len(available_adapters) == 0
    
    def test_adapter_instance_reuse(self):
        """Test that adapter instances are reused appropriately."""
        registry = AdapterRegistry()
        registry.register(TestAdapter)
        
        adapter1 = registry.get_adapter("test-adapter", "correlation-1")
        adapter2 = registry.get_adapter("test-adapter", "correlation-1")
        adapter3 = registry.get_adapter("test-adapter", "correlation-2")
        
        assert adapter1 is adapter2  # Same correlation ID, should reuse
        assert adapter1 is not adapter3  # Different correlation ID, should be different


class TestGlobalFunctions:
    """Tests for global adapter functions."""
    
    def setup_method(self):
        """Set up test environment."""
        # Clear global registry for clean tests
        from ultra_robust_xml_parser.api.adapters import _adapter_registry
        _adapter_registry._adapters.clear()
        _adapter_registry._instances.clear()
    
    def test_register_adapter_global(self):
        """Test global adapter registration."""
        register_adapter(TestAdapter)
        
        adapter = get_adapter("test-adapter")
        assert adapter is not None
        assert adapter.metadata.name == "test-adapter"
    
    def test_list_available_adapters_global(self):
        """Test global listing of available adapters."""
        register_adapter(TestAdapter)
        
        adapters = list_available_adapters()
        assert len(adapters) == 1
        assert adapters[0].name == "test-adapter"
    
    def test_get_adapters_by_type_global(self):
        """Test global get adapters by type."""
        register_adapter(TestAdapter)
        
        xml_adapters = get_adapters_by_type(AdapterType.XML_LIBRARY)
        assert "test-adapter" in xml_adapters
    
    def test_validate_adapter_compatibility(self):
        """Test adapter compatibility validation."""
        register_adapter(TestAdapter)
        
        # Available adapter
        assert validate_adapter_compatibility("test-adapter") is True
        
        # Non-existent adapter
        assert validate_adapter_compatibility("non-existent") is False


class TestAdapterTypes:
    """Tests for adapter type enums and metadata."""
    
    def test_adapter_type_enum(self):
        """Test AdapterType enum values."""
        assert AdapterType.XML_LIBRARY is not None
        assert AdapterType.DATA_FRAME is not None
        assert AdapterType.WEB_FRAMEWORK is not None
        assert AdapterType.PLUGIN is not None
    
    def test_conversion_direction_enum(self):
        """Test ConversionDirection enum values."""
        assert ConversionDirection.TO_TARGET is not None
        assert ConversionDirection.FROM_TARGET is not None
    
    def test_adapter_metadata_creation(self):
        """Test AdapterMetadata dataclass."""
        metadata = AdapterMetadata(
            name="test-adapter",
            version="1.0.0",
            adapter_type=AdapterType.XML_LIBRARY,
            target_library="test-lib",
            supported_versions=["1.0.0", "1.1.0"],
            description="Test adapter"
        )
        
        assert metadata.name == "test-adapter"
        assert metadata.version == "1.0.0"
        assert metadata.adapter_type == AdapterType.XML_LIBRARY
        assert metadata.target_library == "test-lib"
        assert len(metadata.supported_versions) == 2
        assert metadata.author == "ultra-robust-xml-parser"  # Default value
    
    def test_conversion_result_creation(self):
        """Test ConversionResult dataclass."""
        result = ConversionResult(
            success=True,
            converted_data="converted",
            original_data="original",
            conversion_time_ms=100.0
        )
        
        assert result.success is True
        assert result.converted_data == "converted"
        assert result.original_data == "original"
        assert result.conversion_time_ms == 100.0
        assert len(result.warnings) == 0  # Default empty list
        assert len(result.errors) == 0    # Default empty list
        assert len(result.metadata) == 0  # Default empty dict
        assert len(result.diagnostics) == 0  # Default empty list


class TestXMLLibraryAdapters:
    """Tests for XML library specific adapters."""
    
    def test_element_tree_adapter_metadata(self):
        """Test ElementTree adapter metadata."""
        adapter = ElementTreeAdapter()
        metadata = adapter.metadata
        
        assert metadata.name == "elementtree"
        assert metadata.adapter_type == AdapterType.XML_LIBRARY
        assert metadata.target_library == "xml.etree.ElementTree"
        assert adapter.is_available() is True  # Always available
    
    def test_element_tree_conversion_basic(self):
        """Test ElementTree adapter basic conversion."""
        adapter = ElementTreeAdapter()
        
        # Create mock parse result
        mock_parse_result = Mock(spec=ParseResult)
        mock_parse_result.success = True
        mock_parse_result.tree = Mock()
        mock_parse_result.tree.root = Mock()
        mock_parse_result.tree.root.tag = "root"
        mock_parse_result.tree.root.text = "content"
        mock_parse_result.tree.root.attributes = {"id": "123"}
        mock_parse_result.tree.root.children = []
        
        # Test to_target conversion
        result = adapter.to_target(mock_parse_result)
        
        assert result.success is True
        assert hasattr(result.converted_data, 'tag')
        assert result.converted_data.tag == "root"
        assert result.conversion_time_ms >= 0
        
        # Test validation
        validation_result = adapter.validate_conversion(
            mock_parse_result, result.converted_data, ConversionDirection.TO_TARGET
        )
        assert validation_result is True
    
    def test_element_tree_from_target(self):
        """Test ElementTree from_target conversion."""
        adapter = ElementTreeAdapter()
        
        # Create mock ElementTree element
        mock_element = Mock()
        mock_element.tag = "root"
        
        with patch('xml.etree.ElementTree.tostring') as mock_tostring:
            mock_tostring.return_value = "<root>test</root>"
            
            with patch('ultra_robust_xml_parser.api.parse') as mock_parse:
                mock_result = Mock(spec=ParseResult)
                mock_result.success = True
                mock_parse.return_value = mock_result
                
                result = adapter.from_target(mock_element)
                
                assert result.success is True
                assert result.converted_data == mock_result
                mock_parse.assert_called_once()
    
    def test_lxml_adapter_metadata(self):
        """Test LxmlAdapter metadata."""
        adapter = LxmlAdapter()
        metadata = adapter.metadata
        
        assert metadata.name == "lxml"
        assert metadata.adapter_type == AdapterType.XML_LIBRARY
        assert metadata.target_library == "lxml"
    
    def test_lxml_adapter_availability(self):
        """Test LxmlAdapter availability checking."""
        adapter = LxmlAdapter()
        
        # Test availability (will depend on whether lxml is actually installed)
        availability = adapter.is_available()
        assert isinstance(availability, bool)
    
    def test_beautifulsoup_adapter_metadata(self):
        """Test BeautifulSoupAdapter metadata."""
        adapter = BeautifulSoupAdapter()
        metadata = adapter.metadata
        
        assert metadata.name == "beautifulsoup"
        assert metadata.adapter_type == AdapterType.XML_LIBRARY
        assert metadata.target_library == "beautifulsoup4"
    
    def test_beautifulsoup_adapter_availability(self):
        """Test BeautifulSoupAdapter availability checking."""
        adapter = BeautifulSoupAdapter()
        
        # This will depend on whether BeautifulSoup is actually installed
        # Since it's optional, we test both cases
        availability = adapter.is_available()
        assert isinstance(availability, bool)
    
    def test_adapter_error_handling(self):
        """Test adapter error handling for invalid input."""
        adapter = ElementTreeAdapter()
        
        # Test with invalid parse result
        invalid_parse_result = Mock(spec=ParseResult)
        invalid_parse_result.success = False
        invalid_parse_result.tree = None
        
        result = adapter.to_target(invalid_parse_result)
        
        assert result.success is False
        assert "not successful or has no tree" in result.errors[0]
        assert result.converted_data is None
        
        # Test from_target with invalid data
        invalid_target = "not an element"
        result = adapter.from_target(invalid_target)
        
        assert result.success is False
        assert "not a valid ElementTree element" in result.errors[0]
        assert result.converted_data is None
    
    def test_xml_library_adapters_registration(self):
        """Test that XML library adapters are properly registered."""
        # ElementTree should always be available
        elementtree_adapter = get_adapter("elementtree")
        assert elementtree_adapter is not None
        assert isinstance(elementtree_adapter, ElementTreeAdapter)
        
        # Get all XML library adapters
        xml_adapters = get_adapters_by_type(AdapterType.XML_LIBRARY)
        assert "elementtree" in xml_adapters
        
        # Check if lxml and BeautifulSoup adapters are registered (depends on availability)
        available_adapters = list_available_adapters()
        adapter_names = [adapter.name for adapter in available_adapters]
        
        # ElementTree should always be there
        assert "elementtree" in adapter_names
        
        # Check compatibility validation
        assert validate_adapter_compatibility("elementtree") is True
        assert validate_adapter_compatibility("non-existent-adapter") is False
    
    def test_adapter_performance_profiling(self):
        """Test that XML library adapters record performance metrics."""
        adapter = ElementTreeAdapter()
        
        # Create mock parse result
        mock_parse_result = Mock(spec=ParseResult)
        mock_parse_result.success = True
        mock_parse_result.tree = Mock()
        mock_parse_result.tree.root = Mock()
        mock_parse_result.tree.root.tag = "test"
        mock_parse_result.tree.root.text = None
        mock_parse_result.tree.root.attributes = {}
        mock_parse_result.tree.root.children = []
        
        # Perform conversion
        result = adapter.to_target(mock_parse_result)
        
        assert result.success is True
        
        # Check performance stats
        stats = adapter.get_performance_stats()
        assert "elementtree" in stats
        assert stats["elementtree"]["count"] >= 1


class TestPandasAdapter:
    """Tests for PandasAdapter."""
    
    def test_pandas_adapter_metadata(self):
        """Test PandasAdapter metadata."""
        adapter = PandasAdapter()
        metadata = adapter.metadata
        
        assert metadata.name == "pandas"
        assert metadata.adapter_type == AdapterType.DATA_FRAME
        assert metadata.target_library == "pandas"
    
    def test_pandas_adapter_availability(self):
        """Test PandasAdapter availability checking."""
        adapter = PandasAdapter()
        
        # This will depend on whether pandas is actually installed
        availability = adapter.is_available()
        assert isinstance(availability, bool)
    
    def test_pandas_conversion_basic(self):
        """Test PandasAdapter basic conversion."""
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not available")
        
        adapter = PandasAdapter()
        
        # Create mock parse result
        mock_parse_result = Mock(spec=ParseResult)
        mock_parse_result.success = True
        mock_parse_result.tree = Mock()
        mock_parse_result.tree.root = Mock()
        mock_parse_result.tree.root.tag = "root"
        mock_parse_result.tree.root.text = "content"
        mock_parse_result.tree.root.attributes = {}
        mock_parse_result.tree.root.children = []
        
        # Mock the _extract_data_from_tree method to return some data
        adapter._extract_data_from_tree = Mock(return_value=[
            {'tag': 'root', 'text': 'content', 'path': '/'}
        ])
        
        result = adapter.to_target(mock_parse_result)
        
        assert result.success is True
        assert hasattr(result.converted_data, 'shape')  # Should be a DataFrame
        assert result.conversion_time_ms >= 0
        assert 'dataframe_shape' in result.metadata
    
    def test_pandas_error_handling(self):
        """Test PandasAdapter error handling."""
        adapter = PandasAdapter()
        
        # Test with invalid parse result
        invalid_parse_result = Mock(spec=ParseResult)
        invalid_parse_result.success = False
        invalid_parse_result.tree = None
        
        result = adapter.to_target(invalid_parse_result)
        
        assert result.success is False
        assert "not successful or has no tree" in result.errors[0]
        
        # Test from_target with invalid data
        invalid_target = "not a dataframe"
        result = adapter.from_target(invalid_target)
        
        assert result.success is False
        assert "not a pandas DataFrame" in result.errors[0]
    
    def test_extract_flat_data(self):
        """Test the flat data extraction method."""
        adapter = PandasAdapter()
        
        # Create mock element
        mock_element = Mock()
        mock_element.tag = "root"
        mock_element.text = "content"
        mock_element.attributes = {"id": "123", "type": "test"}
        mock_element.children = []
        
        data = []
        adapter._extract_flat_data(mock_element, data, "/")
        
        assert len(data) == 1
        row = data[0]
        assert row['tag'] == "root"
        assert row['text'] == "content"
        assert row['path'] == "/"
        assert row['attr_id'] == "123"
        assert row['attr_type'] == "test"
    
    def test_convert_dataframe_to_xml(self):
        """Test DataFrame to XML conversion."""
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not available")
        
        adapter = PandasAdapter()
        
        # Test empty DataFrame
        empty_df = pd.DataFrame()
        xml = adapter._convert_dataframe_to_xml(empty_df)
        assert xml == "<root></root>"
        
        # Test DataFrame with tag and text columns
        df = pd.DataFrame({
            'tag': ['root'],
            'text': ['content'],
            'path': ['/']
        })
        
        xml = adapter._convert_dataframe_to_xml(df)
        assert "<root>content</root>" == xml
        
        # Test fallback structure
        df_fallback = pd.DataFrame({
            'col1': ['value1'],
            'col2': ['value2']
        })
        
        xml = adapter._convert_dataframe_to_xml(df_fallback)
        assert "<data>" in xml
        assert "<item>" in xml
        assert "<col1>value1</col1>" in xml
        assert "<col2>value2</col2>" in xml
    
    def test_pandas_from_target_conversion(self):
        """Test conversion from pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not available")
        
        adapter = PandasAdapter()
        
        # Create a simple DataFrame
        df = pd.DataFrame({'tag': ['root'], 'text': ['test'], 'path': ['/']})
        
        with patch('ultra_robust_xml_parser.api.parse') as mock_parse:
            mock_result = Mock(spec=ParseResult)
            mock_result.success = True
            mock_parse.return_value = mock_result
            
            result = adapter.from_target(df)
            
            assert result.success is True
            assert result.converted_data == mock_result
            mock_parse.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])