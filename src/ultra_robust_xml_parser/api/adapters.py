"""Integration adapters for seamless compatibility with popular XML libraries.

This module provides bidirectional conversion utilities and adapter framework for
integrating with lxml, BeautifulSoup, pandas, and web frameworks while maintaining
the never-fail philosophy and streaming capabilities.
"""

import threading
import time
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Type, Union, Callable

from ultra_robust_xml_parser.shared import (
    DiagnosticEntry,
    DiagnosticSeverity,
    get_logger,
)
from ultra_robust_xml_parser.tree.builder import ParseResult


class AdapterType(Enum):
    """Types of integration adapters."""

    XML_LIBRARY = auto()     # XML processing libraries (lxml, BeautifulSoup, etc.)
    DATA_FRAME = auto()      # DataFrame libraries (pandas)
    WEB_FRAMEWORK = auto()   # Web frameworks (Flask, Django, FastAPI)
    PLUGIN = auto()         # Custom plugin adapters


class ConversionDirection(Enum):
    """Direction of data conversion."""

    TO_TARGET = auto()      # Convert from ParseResult to target format
    FROM_TARGET = auto()    # Convert from target format to ParseResult


@dataclass
class AdapterMetadata:
    """Metadata about an integration adapter."""

    name: str
    version: str
    adapter_type: AdapterType
    target_library: str
    supported_versions: List[str]
    description: str
    author: str = "ultra-robust-xml-parser"
    documentation_url: Optional[str] = None
    compatibility_notes: Optional[str] = None


@dataclass
class ConversionResult:
    """Result of a conversion operation."""

    success: bool
    converted_data: Any
    original_data: Any
    conversion_time_ms: float
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    diagnostics: List[DiagnosticEntry] = field(default_factory=list)


class AdapterPerformanceProfiler:
    """Performance profiling utility for adapters."""

    def __init__(self) -> None:
        """Initialize performance profiler."""
        self._metrics: Dict[str, List[float]] = {}
        self._lock = threading.RLock()
    
    def record_conversion(self, adapter_name: str, conversion_time_ms: float) -> None:
        """Record conversion performance."""
        with self._lock:
            if adapter_name not in self._metrics:
                self._metrics[adapter_name] = []
            self._metrics[adapter_name].append(conversion_time_ms)
            
            # Keep only recent metrics (last 1000 operations)
            if len(self._metrics[adapter_name]) > 1000:
                self._metrics[adapter_name] = self._metrics[adapter_name][-1000:]
    
    def get_statistics(self, adapter_name: str) -> Dict[str, float]:
        """Get performance statistics for an adapter."""
        with self._lock:
            if adapter_name not in self._metrics or not self._metrics[adapter_name]:
                return {}
            
            times = self._metrics[adapter_name]
            return {
                "count": len(times),
                "average_ms": sum(times) / len(times),
                "min_ms": min(times),
                "max_ms": max(times),
                "total_ms": sum(times),
            }
    
    def get_all_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics for all adapters."""
        with self._lock:
            return {name: self.get_statistics(name) for name in self._metrics}


class IntegrationAdapter(ABC):
    """Abstract base class for all integration adapters.
    
    This class defines the interface for bidirectional conversion between
    ParseResult objects and target format representations, providing consistent
    error handling, performance monitoring, and validation capabilities.
    """
    
    def __init__(self, correlation_id: Optional[str] = None) -> None:
        """Initialize the integration adapter.
        
        Args:
            correlation_id: Optional correlation ID for request tracking
        """
        self.correlation_id = correlation_id
        self._logger = get_logger(__name__, correlation_id, self.__class__.__name__)
        self._profiler = AdapterPerformanceProfiler()
        self._validation_cache: Dict[str, bool] = {}
        
    @property
    @abstractmethod
    def metadata(self) -> AdapterMetadata:
        """Get adapter metadata."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the target library is available and compatible."""
        pass
    
    @abstractmethod
    def to_target(self, parse_result: ParseResult) -> ConversionResult:
        """Convert ParseResult to target format.
        
        Args:
            parse_result: Parsed XML document result
            
        Returns:
            ConversionResult containing the converted data and metadata
        """
        pass
    
    @abstractmethod
    def from_target(self, target_data: Any) -> ConversionResult:
        """Convert target format to ParseResult.
        
        Args:
            target_data: Data in target format
            
        Returns:
            ConversionResult containing ParseResult and metadata
        """
        pass
    
    def validate_conversion(
        self, 
        original: Any, 
        converted: Any, 
        direction: ConversionDirection
    ) -> bool:
        """Validate the quality of a conversion operation.
        
        Args:
            original: Original data before conversion
            converted: Data after conversion
            direction: Direction of the conversion
            
        Returns:
            True if conversion maintains data integrity, False otherwise
        """
        try:
            # Basic validation - ensure we got something back
            if converted is None and original is not None:
                return False
            
            # Cache key for validation results
            cache_key = f"{direction.name}_{hash(str(original)[:100])}"
            
            if cache_key in self._validation_cache:
                return self._validation_cache[cache_key]
            
            # Perform adapter-specific validation
            result = self._perform_validation(original, converted, direction)
            
            # Cache the result
            self._validation_cache[cache_key] = result
            
            # Limit cache size
            if len(self._validation_cache) > 10000:
                # Remove oldest entries
                oldest_keys = list(self._validation_cache.keys())[:1000]
                for key in oldest_keys:
                    del self._validation_cache[key]
            
            return result
            
        except Exception as e:
            self._logger.warning(
                f"Validation failed with error: {e}",
                extra={"direction": direction.name}
            )
            return False
    
    def _perform_validation(
        self, 
        original: Any, 
        converted: Any, 
        direction: ConversionDirection
    ) -> bool:
        """Perform adapter-specific validation logic.
        
        Subclasses should override this method to provide specific validation.
        
        Args:
            original: Original data
            converted: Converted data
            direction: Conversion direction
            
        Returns:
            True if validation passes, False otherwise
        """
        # Default implementation - basic non-null check
        return converted is not None
    
    def get_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics for this adapter."""
        return self._profiler.get_all_statistics()
    
    def _record_performance(self, operation_time_ms: float) -> None:
        """Record performance metrics for this adapter."""
        self._profiler.record_conversion(self.metadata.name, operation_time_ms)
    
    def _create_error_result(
        self, 
        error_message: str, 
        original_data: Any,
        conversion_time_ms: float = 0.0
    ) -> ConversionResult:
        """Create a ConversionResult for error conditions."""
        return ConversionResult(
            success=False,
            converted_data=None,
            original_data=original_data,
            conversion_time_ms=conversion_time_ms,
            errors=[error_message],
            diagnostics=[
                DiagnosticEntry(
                    severity=DiagnosticSeverity.ERROR,
                    message=error_message,
                    component=self.__class__.__name__,
                    correlation_id=self.correlation_id
                )
            ]
        )


class AdapterRegistry:
    """Registry for managing integration adapters."""
    
    def __init__(self) -> None:
        """Initialize the adapter registry."""
        self._adapters: Dict[str, Type[IntegrationAdapter]] = {}
        self._instances: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
        self._lock = threading.RLock()
    
    def register(self, adapter_class: Type[IntegrationAdapter]) -> None:
        """Register an adapter class.
        
        Args:
            adapter_class: Adapter class to register
        """
        with self._lock:
            # Create temporary instance to get metadata
            temp_instance = adapter_class()
            metadata = temp_instance.metadata
            
            self._adapters[metadata.name] = adapter_class
    
    def get_adapter(
        self, 
        adapter_name: str, 
        correlation_id: Optional[str] = None
    ) -> Optional[IntegrationAdapter]:
        """Get an adapter instance by name.
        
        Args:
            adapter_name: Name of the adapter
            correlation_id: Optional correlation ID
            
        Returns:
            Adapter instance if found and available, None otherwise
        """
        with self._lock:
            if adapter_name not in self._adapters:
                return None
            
            # Check for existing instance
            instance_key = f"{adapter_name}_{correlation_id or 'default'}"
            if instance_key in self._instances:
                return self._instances[instance_key]
            
            # Create new instance
            adapter_class = self._adapters[adapter_name]
            try:
                instance = adapter_class(correlation_id)
                if instance.is_available():
                    self._instances[instance_key] = instance
                    return instance
                else:
                    return None
            except Exception:
                return None
    
    def list_available_adapters(self) -> List[AdapterMetadata]:
        """List all available adapters with their metadata.
        
        Returns:
            List of metadata for available adapters
        """
        available_adapters = []
        
        with self._lock:
            for adapter_name, adapter_class in self._adapters.items():
                try:
                    instance = adapter_class()
                    if instance.is_available():
                        available_adapters.append(instance.metadata)
                except Exception:
                    continue
        
        return available_adapters
    
    def get_adapters_by_type(self, adapter_type: AdapterType) -> List[str]:
        """Get adapter names by type.
        
        Args:
            adapter_type: Type of adapters to find
            
        Returns:
            List of adapter names of the specified type
        """
        matching_adapters = []
        
        with self._lock:
            for adapter_name, adapter_class in self._adapters.items():
                try:
                    instance = adapter_class()
                    if instance.metadata.adapter_type == adapter_type and instance.is_available():
                        matching_adapters.append(adapter_name)
                except Exception:
                    continue
        
        return matching_adapters


# Global adapter registry instance
_adapter_registry = AdapterRegistry()


def register_adapter(adapter_class: Type[IntegrationAdapter]) -> None:
    """Register an integration adapter globally.
    
    Args:
        adapter_class: Adapter class to register
    """
    _adapter_registry.register(adapter_class)


def get_adapter(
    adapter_name: str, 
    correlation_id: Optional[str] = None
) -> Optional[IntegrationAdapter]:
    """Get a registered adapter instance.
    
    Args:
        adapter_name: Name of the adapter
        correlation_id: Optional correlation ID
        
    Returns:
        Adapter instance if available, None otherwise
    """
    return _adapter_registry.get_adapter(adapter_name, correlation_id)


def list_available_adapters() -> List[AdapterMetadata]:
    """List all available integration adapters.
    
    Returns:
        List of metadata for available adapters
    """
    return _adapter_registry.list_available_adapters()


def get_adapters_by_type(adapter_type: AdapterType) -> List[str]:
    """Get adapter names by type.
    
    Args:
        adapter_type: Type of adapters to find
        
    Returns:
        List of adapter names of the specified type
    """
    return _adapter_registry.get_adapters_by_type(adapter_type)


def validate_adapter_compatibility(adapter_name: str) -> bool:
    """Validate that an adapter's dependencies are available.
    
    Args:
        adapter_name: Name of the adapter to validate
        
    Returns:
        True if adapter is compatible and available, False otherwise
    """
    adapter = get_adapter(adapter_name)
    return adapter is not None and adapter.is_available()


# XML Library Adapters
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    try:
        import lxml.etree
    except ImportError:
        lxml = None
    
    try:
        from bs4 import BeautifulSoup
        import bs4
    except ImportError:
        BeautifulSoup = None
        bs4 = None
    
    import xml.etree.ElementTree as ET


class LxmlAdapter(IntegrationAdapter):
    """Adapter for bidirectional conversion with lxml.etree."""
    
    @property
    def metadata(self) -> AdapterMetadata:
        """Get adapter metadata."""
        return AdapterMetadata(
            name="lxml",
            version="1.0.0",
            adapter_type=AdapterType.XML_LIBRARY,
            target_library="lxml",
            supported_versions=["4.0+"],
            description="Bidirectional conversion between ParseResult and lxml.etree"
        )
    
    def is_available(self) -> bool:
        """Check if lxml is available."""
        try:
            import lxml.etree
            return True
        except ImportError:
            return False
    
    def to_target(self, parse_result: ParseResult) -> ConversionResult:
        """Convert ParseResult to lxml.etree.Element.
        
        Args:
            parse_result: Parsed XML document result
            
        Returns:
            ConversionResult containing lxml.etree.Element
        """
        start_time = time.time()
        
        try:
            import lxml.etree as ET
            
            if not parse_result.success or not parse_result.tree:
                return self._create_error_result(
                    "ParseResult is not successful or has no tree",
                    parse_result,
                    (time.time() - start_time) * 1000
                )
            
            # Convert our XMLDocument to lxml tree
            root = parse_result.tree.root
            lxml_root = self._convert_element_to_lxml(root, ET)
            
            processing_time = (time.time() - start_time) * 1000
            self._record_performance(processing_time)
            
            return ConversionResult(
                success=True,
                converted_data=lxml_root,
                original_data=parse_result,
                conversion_time_ms=processing_time,
                metadata={
                    "lxml_version": ET.LXML_VERSION,
                    "element_count": len(lxml_root.xpath("//*")),
                }
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            return self._create_error_result(
                f"Failed to convert to lxml: {e}",
                parse_result,
                processing_time
            )
    
    def from_target(self, target_data: Any) -> ConversionResult:
        """Convert lxml.etree.Element to ParseResult.
        
        Args:
            target_data: lxml.etree.Element
            
        Returns:
            ConversionResult containing ParseResult
        """
        start_time = time.time()
        
        try:
            import lxml.etree as ET
            from ultra_robust_xml_parser.api import parse
            
            if not hasattr(target_data, 'tag'):
                return self._create_error_result(
                    "Target data is not a valid lxml element",
                    target_data,
                    (time.time() - start_time) * 1000
                )
            
            # Convert lxml element back to XML string and parse
            xml_string = ET.tostring(target_data, encoding='unicode')
            parse_result = parse(xml_string, self.correlation_id)
            
            processing_time = (time.time() - start_time) * 1000
            self._record_performance(processing_time)
            
            return ConversionResult(
                success=True,
                converted_data=parse_result,
                original_data=target_data,
                conversion_time_ms=processing_time,
                metadata={
                    "original_tag": target_data.tag,
                    "xml_length": len(xml_string),
                }
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            return self._create_error_result(
                f"Failed to convert from lxml: {e}",
                target_data,
                processing_time
            )
    
    def _convert_element_to_lxml(self, element, ET):
        """Convert XMLElement to lxml.etree.Element."""
        lxml_element = ET.Element(element.tag)
        
        # Copy attributes
        for key, value in element.attributes.items():
            lxml_element.set(key, value)
        
        # Set text content
        if element.text:
            lxml_element.text = element.text
        
        # Convert children recursively
        for child in element.children:
            lxml_child = self._convert_element_to_lxml(child, ET)
            lxml_element.append(lxml_child)
        
        return lxml_element


class BeautifulSoupAdapter(IntegrationAdapter):
    """Adapter for bidirectional conversion with BeautifulSoup."""
    
    @property
    def metadata(self) -> AdapterMetadata:
        """Get adapter metadata."""
        return AdapterMetadata(
            name="beautifulsoup",
            version="1.0.0",
            adapter_type=AdapterType.XML_LIBRARY,
            target_library="beautifulsoup4",
            supported_versions=["4.0+"],
            description="Bidirectional conversion between ParseResult and BeautifulSoup"
        )
    
    def is_available(self) -> bool:
        """Check if BeautifulSoup is available."""
        try:
            from bs4 import BeautifulSoup
            return True
        except ImportError:
            return False
    
    def to_target(self, parse_result: ParseResult) -> ConversionResult:
        """Convert ParseResult to BeautifulSoup.
        
        Args:
            parse_result: Parsed XML document result
            
        Returns:
            ConversionResult containing BeautifulSoup object
        """
        start_time = time.time()
        
        try:
            from bs4 import BeautifulSoup
            
            if not parse_result.success or not parse_result.tree:
                return self._create_error_result(
                    "ParseResult is not successful or has no tree",
                    parse_result,
                    (time.time() - start_time) * 1000
                )
            
            # Convert to XML string first, then parse with BeautifulSoup
            xml_string = self._convert_tree_to_xml_string(parse_result.tree)
            soup = BeautifulSoup(xml_string, 'xml')
            
            processing_time = (time.time() - start_time) * 1000
            self._record_performance(processing_time)
            
            return ConversionResult(
                success=True,
                converted_data=soup,
                original_data=parse_result,
                conversion_time_ms=processing_time,
                metadata={
                    "parser_name": "xml",
                    "xml_length": len(xml_string),
                }
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            return self._create_error_result(
                f"Failed to convert to BeautifulSoup: {e}",
                parse_result,
                processing_time
            )
    
    def from_target(self, target_data: Any) -> ConversionResult:
        """Convert BeautifulSoup to ParseResult.
        
        Args:
            target_data: BeautifulSoup object
            
        Returns:
            ConversionResult containing ParseResult
        """
        start_time = time.time()
        
        try:
            from bs4 import BeautifulSoup
            from ultra_robust_xml_parser.api import parse
            
            if not hasattr(target_data, 'prettify'):
                return self._create_error_result(
                    "Target data is not a valid BeautifulSoup object",
                    target_data,
                    (time.time() - start_time) * 1000
                )
            
            # Convert BeautifulSoup back to XML string and parse
            xml_string = str(target_data)
            parse_result = parse(xml_string, self.correlation_id)
            
            processing_time = (time.time() - start_time) * 1000
            self._record_performance(processing_time)
            
            return ConversionResult(
                success=True,
                converted_data=parse_result,
                original_data=target_data,
                conversion_time_ms=processing_time,
                metadata={
                    "xml_length": len(xml_string),
                }
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            return self._create_error_result(
                f"Failed to convert from BeautifulSoup: {e}",
                target_data,
                processing_time
            )
    
    def _convert_tree_to_xml_string(self, tree) -> str:
        """Convert XMLDocument tree to XML string."""
        if not tree.root:
            return "<root/>"
        
        return f"<{tree.root.tag}>{tree.root.text or ''}</{tree.root.tag}>"


class ElementTreeAdapter(IntegrationAdapter):
    """Adapter for bidirectional conversion with xml.etree.ElementTree."""
    
    @property
    def metadata(self) -> AdapterMetadata:
        """Get adapter metadata."""
        return AdapterMetadata(
            name="elementtree",
            version="1.0.0",
            adapter_type=AdapterType.XML_LIBRARY,
            target_library="xml.etree.ElementTree",
            supported_versions=["3.8+"],
            description="Bidirectional conversion between ParseResult and ElementTree"
        )
    
    def is_available(self) -> bool:
        """Check if ElementTree is available (always True for Python 3.8+)."""
        try:
            import xml.etree.ElementTree as ET
            return True
        except ImportError:
            return False
    
    def to_target(self, parse_result: ParseResult) -> ConversionResult:
        """Convert ParseResult to xml.etree.ElementTree.Element.
        
        Args:
            parse_result: Parsed XML document result
            
        Returns:
            ConversionResult containing ElementTree.Element
        """
        start_time = time.time()
        
        try:
            import xml.etree.ElementTree as ET
            
            if not parse_result.success or not parse_result.tree:
                return self._create_error_result(
                    "ParseResult is not successful or has no tree",
                    parse_result,
                    (time.time() - start_time) * 1000
                )
            
            # Convert our XMLDocument to ElementTree
            root = parse_result.tree.root
            et_root = self._convert_element_to_et(root, ET)
            
            processing_time = (time.time() - start_time) * 1000
            self._record_performance(processing_time)
            
            return ConversionResult(
                success=True,
                converted_data=et_root,
                original_data=parse_result,
                conversion_time_ms=processing_time,
                metadata={
                    "element_count": len(list(et_root.iter())),
                }
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            return self._create_error_result(
                f"Failed to convert to ElementTree: {e}",
                parse_result,
                processing_time
            )
    
    def from_target(self, target_data: Any) -> ConversionResult:
        """Convert ElementTree.Element to ParseResult.
        
        Args:
            target_data: ElementTree.Element
            
        Returns:
            ConversionResult containing ParseResult
        """
        start_time = time.time()
        
        try:
            import xml.etree.ElementTree as ET
            from ultra_robust_xml_parser.api import parse
            
            if not hasattr(target_data, 'tag'):
                return self._create_error_result(
                    "Target data is not a valid ElementTree element",
                    target_data,
                    (time.time() - start_time) * 1000
                )
            
            # Convert ElementTree element back to XML string and parse
            xml_string = ET.tostring(target_data, encoding='unicode')
            parse_result = parse(xml_string, self.correlation_id)
            
            processing_time = (time.time() - start_time) * 1000
            self._record_performance(processing_time)
            
            return ConversionResult(
                success=True,
                converted_data=parse_result,
                original_data=target_data,
                conversion_time_ms=processing_time,
                metadata={
                    "original_tag": target_data.tag,
                    "xml_length": len(xml_string),
                }
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            return self._create_error_result(
                f"Failed to convert from ElementTree: {e}",
                target_data,
                processing_time
            )
    
    def _convert_element_to_et(self, element, ET):
        """Convert XMLElement to ElementTree.Element."""
        et_element = ET.Element(element.tag)
        
        # Copy attributes
        for key, value in element.attributes.items():
            et_element.set(key, value)
        
        # Set text content
        if element.text:
            et_element.text = element.text
        
        # Convert children recursively
        for child in element.children:
            et_child = self._convert_element_to_et(child, ET)
            et_element.append(et_child)
        
        return et_element


class PandasAdapter(IntegrationAdapter):
    """Adapter for bidirectional conversion with pandas DataFrame."""
    
    @property
    def metadata(self) -> AdapterMetadata:
        """Get adapter metadata."""
        return AdapterMetadata(
            name="pandas",
            version="1.0.0",
            adapter_type=AdapterType.DATA_FRAME,
            target_library="pandas",
            supported_versions=["1.0+"],
            description="Bidirectional conversion between ParseResult and pandas DataFrame"
        )
    
    def is_available(self) -> bool:
        """Check if pandas is available."""
        try:
            import pandas as pd
            return True
        except ImportError:
            return False
    
    def to_target(self, parse_result: ParseResult) -> ConversionResult:
        """Convert ParseResult to pandas DataFrame.
        
        Args:
            parse_result: Parsed XML document result
            
        Returns:
            ConversionResult containing pandas DataFrame
        """
        start_time = time.time()
        
        try:
            import pandas as pd
            
            if not parse_result.success or not parse_result.tree:
                return self._create_error_result(
                    "ParseResult is not successful or has no tree",
                    parse_result,
                    (time.time() - start_time) * 1000
                )
            
            # Extract data from XML tree and convert to DataFrame
            data = self._extract_data_from_tree(parse_result.tree)
            
            if not data:
                # Create empty DataFrame with basic info
                df = pd.DataFrame({
                    'tag': [parse_result.tree.root.tag] if parse_result.tree.root else [],
                    'text': [parse_result.tree.root.text] if parse_result.tree.root else [],
                    'path': ['/'] if parse_result.tree.root else []
                })
            else:
                df = pd.DataFrame(data)
            
            processing_time = (time.time() - start_time) * 1000
            self._record_performance(processing_time)
            
            return ConversionResult(
                success=True,
                converted_data=df,
                original_data=parse_result,
                conversion_time_ms=processing_time,
                metadata={
                    "dataframe_shape": df.shape,
                    "column_count": len(df.columns),
                    "row_count": len(df),
                    "columns": list(df.columns),
                }
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            return self._create_error_result(
                f"Failed to convert to pandas DataFrame: {e}",
                parse_result,
                processing_time
            )
    
    def from_target(self, target_data: Any) -> ConversionResult:
        """Convert pandas DataFrame to ParseResult.
        
        Args:
            target_data: pandas DataFrame
            
        Returns:
            ConversionResult containing ParseResult
        """
        start_time = time.time()
        
        try:
            import pandas as pd
            from ultra_robust_xml_parser.api import parse
            
            if not isinstance(target_data, pd.DataFrame):
                return self._create_error_result(
                    "Target data is not a pandas DataFrame",
                    target_data,
                    (time.time() - start_time) * 1000
                )
            
            # Convert DataFrame back to XML string and parse
            xml_string = self._convert_dataframe_to_xml(target_data)
            parse_result = parse(xml_string, self.correlation_id)
            
            processing_time = (time.time() - start_time) * 1000
            self._record_performance(processing_time)
            
            return ConversionResult(
                success=True,
                converted_data=parse_result,
                original_data=target_data,
                conversion_time_ms=processing_time,
                metadata={
                    "dataframe_shape": target_data.shape,
                    "xml_length": len(xml_string),
                }
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            return self._create_error_result(
                f"Failed to convert from pandas DataFrame: {e}",
                target_data,
                processing_time
            )
    
    def _extract_data_from_tree(self, tree, structure_type: str = "flat") -> List[Dict[str, Any]]:
        """Extract data from XML tree for DataFrame conversion.
        
        Args:
            tree: XMLDocument tree
            structure_type: Type of structure to generate (flat, nested, hierarchical)
            
        Returns:
            List of dictionaries representing the data
        """
        data = []
        
        if not tree.root:
            return data
        
        # Flat structure - one row per element
        if structure_type == "flat":
            self._extract_flat_data(tree.root, data, "/")
        elif structure_type == "nested":
            # For now, use flat structure - nested could be added later
            self._extract_flat_data(tree.root, data, "/")
        else:  # hierarchical
            # For now, use flat structure - hierarchical could be added later
            self._extract_flat_data(tree.root, data, "/")
            
        return data
    
    def _extract_flat_data(self, element, data: List[Dict[str, Any]], path: str) -> None:
        """Extract data in flat structure (one row per element)."""
        row = {
            'tag': element.tag,
            'text': element.text or '',
            'path': path,
        }
        
        # Add attributes as separate columns
        for attr_name, attr_value in element.attributes.items():
            row[f'attr_{attr_name}'] = attr_value
        
        data.append(row)
        
        # Process children
        for i, child in enumerate(element.children):
            child_path = f"{path}{child.tag}[{i}]" if path != "/" else f"/{child.tag}[{i}]"
            self._extract_flat_data(child, data, child_path)
    
    def _convert_dataframe_to_xml(self, df) -> str:
        """Convert DataFrame back to XML string.
        
        This is a simplified conversion - in practice, this would need
        to handle the specific structure used when converting to DataFrame.
        """
        import pandas as pd
        
        if len(df) == 0:
            return "<root></root>"
        
        # Simple conversion for basic case
        if 'tag' in df.columns and 'text' in df.columns:
            root_rows = df[df['path'] == '/']
            if len(root_rows) > 0:
                root_tag = root_rows.iloc[0]['tag']
                root_text = root_rows.iloc[0]['text']
                return f"<{root_tag}>{root_text}</{root_tag}>"
        
        # Fallback to generic structure
        xml_parts = ["<data>"]
        for _, row in df.iterrows():
            xml_parts.append(f"  <item>")
            for col, value in row.items():
                if pd.notna(value):
                    xml_parts.append(f"    <{col}>{value}</{col}>")
            xml_parts.append(f"  </item>")
        xml_parts.append("</data>")
        
        return "\n".join(xml_parts)


# Auto-register XML library adapters
try:
    register_adapter(LxmlAdapter)
except Exception:
    pass  # lxml not available

try:
    register_adapter(BeautifulSoupAdapter)
except Exception:
    pass  # BeautifulSoup not available

register_adapter(ElementTreeAdapter)  # Always available

try:
    register_adapter(PandasAdapter)
except Exception:
    pass  # pandas not available


# Web Framework Adapters
class FlaskAdapter(IntegrationAdapter):
    """Adapter for Flask request/response XML processing."""
    
    @property
    def metadata(self) -> AdapterMetadata:
        """Get adapter metadata."""
        return AdapterMetadata(
            name="flask",
            version="1.0.0",
            adapter_type=AdapterType.WEB_FRAMEWORK,
            target_library="flask",
            supported_versions=["2.0+"],
            description="Flask request parsing and response formatting for XML"
        )
    
    def is_available(self) -> bool:
        """Check if Flask is available."""
        try:
            import flask
            return True
        except ImportError:
            return False
    
    def to_target(self, parse_result: ParseResult) -> ConversionResult:
        """Convert ParseResult to Flask Response.
        
        Args:
            parse_result: Parsed XML document result
            
        Returns:
            ConversionResult containing Flask Response
        """
        start_time = time.time()
        
        try:
            from flask import Response
            
            if not parse_result.success or not parse_result.tree:
                return self._create_error_result(
                    "ParseResult is not successful or has no tree",
                    parse_result,
                    (time.time() - start_time) * 1000
                )
            
            # Convert to XML string for response
            xml_string = self._convert_tree_to_xml_string(parse_result.tree)
            
            # Create Flask response
            response = Response(
                xml_string,
                mimetype='application/xml',
                headers={
                    'X-Parser-Confidence': str(parse_result.confidence),
                    'X-Parser-Success': str(parse_result.success).lower()
                }
            )
            
            processing_time = (time.time() - start_time) * 1000
            self._record_performance(processing_time)
            
            return ConversionResult(
                success=True,
                converted_data=response,
                original_data=parse_result,
                conversion_time_ms=processing_time,
                metadata={
                    "response_mimetype": "application/xml",
                    "xml_length": len(xml_string),
                    "confidence": parse_result.confidence,
                }
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            return self._create_error_result(
                f"Failed to convert to Flask Response: {e}",
                parse_result,
                processing_time
            )
    
    def from_target(self, target_data: Any) -> ConversionResult:
        """Convert Flask Request to ParseResult.
        
        Args:
            target_data: Flask Request object
            
        Returns:
            ConversionResult containing ParseResult
        """
        start_time = time.time()
        
        try:
            from flask import Request
            from ultra_robust_xml_parser.api import parse
            
            if not hasattr(target_data, 'get_data'):
                return self._create_error_result(
                    "Target data is not a valid Flask Request",
                    target_data,
                    (time.time() - start_time) * 1000
                )
            
            # Extract XML data from request
            xml_data = target_data.get_data(as_text=True)
            
            if not xml_data:
                return self._create_error_result(
                    "Request contains no XML data",
                    target_data,
                    (time.time() - start_time) * 1000
                )
            
            # Parse XML data
            parse_result = parse(xml_data, self.correlation_id)
            
            processing_time = (time.time() - start_time) * 1000
            self._record_performance(processing_time)
            
            return ConversionResult(
                success=True,
                converted_data=parse_result,
                original_data=target_data,
                conversion_time_ms=processing_time,
                metadata={
                    "request_method": getattr(target_data, 'method', 'unknown'),
                    "content_type": getattr(target_data, 'content_type', 'unknown'),
                    "xml_length": len(xml_data),
                }
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            return self._create_error_result(
                f"Failed to convert from Flask Request: {e}",
                target_data,
                processing_time
            )
    
    def _convert_tree_to_xml_string(self, tree) -> str:
        """Convert XMLDocument tree to XML string."""
        if not tree.root:
            return "<root/>"
        
        # Simple XML generation - in practice this would be more sophisticated
        return f"<{tree.root.tag}>{tree.root.text or ''}</{tree.root.tag}>"


class DjangoAdapter(IntegrationAdapter):
    """Adapter for Django request/response XML processing."""
    
    @property
    def metadata(self) -> AdapterMetadata:
        """Get adapter metadata."""
        return AdapterMetadata(
            name="django",
            version="1.0.0",
            adapter_type=AdapterType.WEB_FRAMEWORK,
            target_library="django",
            supported_versions=["3.0+"],
            description="Django request parsing and response formatting for XML"
        )
    
    def is_available(self) -> bool:
        """Check if Django is available."""
        try:
            import django
            return True
        except ImportError:
            return False
    
    def to_target(self, parse_result: ParseResult) -> ConversionResult:
        """Convert ParseResult to Django HttpResponse.
        
        Args:
            parse_result: Parsed XML document result
            
        Returns:
            ConversionResult containing Django HttpResponse
        """
        start_time = time.time()
        
        try:
            from django.http import HttpResponse
            
            if not parse_result.success or not parse_result.tree:
                return self._create_error_result(
                    "ParseResult is not successful or has no tree",
                    parse_result,
                    (time.time() - start_time) * 1000
                )
            
            # Convert to XML string for response
            xml_string = self._convert_tree_to_xml_string(parse_result.tree)
            
            # Create Django response
            response = HttpResponse(
                xml_string,
                content_type='application/xml'
            )
            
            # Add custom headers
            response['X-Parser-Confidence'] = str(parse_result.confidence)
            response['X-Parser-Success'] = str(parse_result.success).lower()
            
            processing_time = (time.time() - start_time) * 1000
            self._record_performance(processing_time)
            
            return ConversionResult(
                success=True,
                converted_data=response,
                original_data=parse_result,
                conversion_time_ms=processing_time,
                metadata={
                    "response_content_type": "application/xml",
                    "xml_length": len(xml_string),
                    "confidence": parse_result.confidence,
                }
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            return self._create_error_result(
                f"Failed to convert to Django HttpResponse: {e}",
                parse_result,
                processing_time
            )
    
    def from_target(self, target_data: Any) -> ConversionResult:
        """Convert Django HttpRequest to ParseResult.
        
        Args:
            target_data: Django HttpRequest object
            
        Returns:
            ConversionResult containing ParseResult
        """
        start_time = time.time()
        
        try:
            from ultra_robust_xml_parser.api import parse
            
            if not hasattr(target_data, 'body'):
                return self._create_error_result(
                    "Target data is not a valid Django HttpRequest",
                    target_data,
                    (time.time() - start_time) * 1000
                )
            
            # Extract XML data from request body
            xml_data = target_data.body.decode('utf-8') if isinstance(target_data.body, bytes) else str(target_data.body)
            
            if not xml_data:
                return self._create_error_result(
                    "Request contains no XML data",
                    target_data,
                    (time.time() - start_time) * 1000
                )
            
            # Parse XML data
            parse_result = parse(xml_data, self.correlation_id)
            
            processing_time = (time.time() - start_time) * 1000
            self._record_performance(processing_time)
            
            return ConversionResult(
                success=True,
                converted_data=parse_result,
                original_data=target_data,
                conversion_time_ms=processing_time,
                metadata={
                    "request_method": getattr(target_data, 'method', 'unknown'),
                    "content_type": getattr(target_data, 'content_type', 'unknown'),
                    "xml_length": len(xml_data),
                }
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            return self._create_error_result(
                f"Failed to convert from Django HttpRequest: {e}",
                target_data,
                processing_time
            )
    
    def _convert_tree_to_xml_string(self, tree) -> str:
        """Convert XMLDocument tree to XML string."""
        if not tree.root:
            return "<root/>"
        
        # Simple XML generation - in practice this would be more sophisticated
        return f"<{tree.root.tag}>{tree.root.text or ''}</{tree.root.tag}>"


class FastAPIAdapter(IntegrationAdapter):
    """Adapter for FastAPI request/response XML processing."""
    
    @property
    def metadata(self) -> AdapterMetadata:
        """Get adapter metadata."""
        return AdapterMetadata(
            name="fastapi",
            version="1.0.0",
            adapter_type=AdapterType.WEB_FRAMEWORK,
            target_library="fastapi",
            supported_versions=["0.68+"],
            description="FastAPI request validation and response formatting for XML"
        )
    
    def is_available(self) -> bool:
        """Check if FastAPI is available."""
        try:
            import fastapi
            return True
        except ImportError:
            return False
    
    def to_target(self, parse_result: ParseResult) -> ConversionResult:
        """Convert ParseResult to FastAPI Response.
        
        Args:
            parse_result: Parsed XML document result
            
        Returns:
            ConversionResult containing FastAPI Response
        """
        start_time = time.time()
        
        try:
            from fastapi import Response
            
            if not parse_result.success or not parse_result.tree:
                return self._create_error_result(
                    "ParseResult is not successful or has no tree",
                    parse_result,
                    (time.time() - start_time) * 1000
                )
            
            # Convert to XML string for response
            xml_string = self._convert_tree_to_xml_string(parse_result.tree)
            
            # Create FastAPI response
            response = Response(
                content=xml_string,
                media_type='application/xml',
                headers={
                    'X-Parser-Confidence': str(parse_result.confidence),
                    'X-Parser-Success': str(parse_result.success).lower()
                }
            )
            
            processing_time = (time.time() - start_time) * 1000
            self._record_performance(processing_time)
            
            return ConversionResult(
                success=True,
                converted_data=response,
                original_data=parse_result,
                conversion_time_ms=processing_time,
                metadata={
                    "response_media_type": "application/xml",
                    "xml_length": len(xml_string),
                    "confidence": parse_result.confidence,
                }
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            return self._create_error_result(
                f"Failed to convert to FastAPI Response: {e}",
                parse_result,
                processing_time
            )
    
    def from_target(self, target_data: Any) -> ConversionResult:
        """Convert FastAPI Request to ParseResult.
        
        Args:
            target_data: Raw request body (bytes or string)
            
        Returns:
            ConversionResult containing ParseResult
        """
        start_time = time.time()
        
        try:
            from ultra_robust_xml_parser.api import parse
            
            # Handle different input types for FastAPI
            if isinstance(target_data, bytes):
                xml_data = target_data.decode('utf-8')
            elif isinstance(target_data, str):
                xml_data = target_data
            else:
                return self._create_error_result(
                    "Target data must be bytes or string for FastAPI processing",
                    target_data,
                    (time.time() - start_time) * 1000
                )
            
            if not xml_data:
                return self._create_error_result(
                    "Request contains no XML data",
                    target_data,
                    (time.time() - start_time) * 1000
                )
            
            # Parse XML data
            parse_result = parse(xml_data, self.correlation_id)
            
            processing_time = (time.time() - start_time) * 1000
            self._record_performance(processing_time)
            
            return ConversionResult(
                success=True,
                converted_data=parse_result,
                original_data=target_data,
                conversion_time_ms=processing_time,
                metadata={
                    "xml_length": len(xml_data),
                    "input_type": type(target_data).__name__,
                }
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            return self._create_error_result(
                f"Failed to convert FastAPI request: {e}",
                target_data,
                processing_time
            )
    
    def _convert_tree_to_xml_string(self, tree) -> str:
        """Convert XMLDocument tree to XML string."""
        if not tree.root:
            return "<root/>"
        
        # Simple XML generation - in practice this would be more sophisticated
        return f"<{tree.root.tag}>{tree.root.text or ''}</{tree.root.tag}>"


# Auto-register web framework adapters
try:
    register_adapter(FlaskAdapter)
except Exception:
    pass  # Flask not available

try:
    register_adapter(DjangoAdapter)
except Exception:
    pass  # Django not available

try:
    register_adapter(FastAPIAdapter)
except Exception:
    pass  # FastAPI not available


# Plugin System
import importlib
import inspect
import sys
from pathlib import Path
from typing import Iterator, Set


@dataclass
class PluginMetadata:
    """Metadata for plugin system."""
    
    name: str
    version: str
    description: str
    author: str
    plugin_type: str  # 'adapter', 'processor', 'validator', etc.
    entry_point: str  # Module path to plugin class
    dependencies: List[str] = field(default_factory=list)
    configuration_schema: Optional[Dict[str, Any]] = None
    security_hash: Optional[str] = None


class PluginValidator:
    """Validates plugins for security and compatibility."""
    
    def __init__(self) -> None:
        """Initialize plugin validator."""
        self._allowed_imports: Set[str] = {
            'ultra_robust_xml_parser',
            'typing', 'dataclasses', 'abc', 'time', 'json', 'xml', 're'
        }
        self._forbidden_imports: Set[str] = {
            'os', 'sys', 'subprocess', 'eval', 'exec', 'open', '__import__'
        }
    
    def validate_plugin_class(self, plugin_class: Type) -> bool:
        """Validate that a plugin class is safe and compatible.
        
        Args:
            plugin_class: Plugin class to validate
            
        Returns:
            True if plugin is valid and safe, False otherwise
        """
        try:
            # Check if it's a subclass of IntegrationAdapter
            if not issubclass(plugin_class, IntegrationAdapter):
                return False
            
            # Check required methods are implemented
            required_methods = ['metadata', 'is_available', 'to_target', 'from_target']
            for method in required_methods:
                if not hasattr(plugin_class, method):
                    return False
            
            # Basic security validation - check source code
            if hasattr(plugin_class, '__module__'):
                module = sys.modules.get(plugin_class.__module__)
                if module and hasattr(module, '__file__'):
                    return self._validate_plugin_source(module.__file__)
            
            return True
            
        except Exception:
            return False
    
    def _validate_plugin_source(self, file_path: Optional[str]) -> bool:
        """Basic validation of plugin source code."""
        if not file_path or not Path(file_path).exists():
            return False
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            # Check for forbidden imports/operations
            for forbidden in self._forbidden_imports:
                if forbidden in source:
                    return False
            
            return True
            
        except Exception:
            return False


class PluginManager:
    """Manages plugin discovery, loading, and lifecycle."""
    
    def __init__(self) -> None:
        """Initialize plugin manager."""
        self._plugins: Dict[str, PluginMetadata] = {}
        self._loaded_adapters: Dict[str, Type[IntegrationAdapter]] = {}
        self._validator = PluginValidator()
        self._plugin_configs: Dict[str, Dict[str, Any]] = {}
    
    def discover_plugins(self, plugin_directories: Optional[List[str]] = None) -> List[PluginMetadata]:
        """Discover available plugins in specified directories.
        
        Args:
            plugin_directories: List of directories to search for plugins
            
        Returns:
            List of discovered plugin metadata
        """
        discovered = []
        
        if plugin_directories is None:
            plugin_directories = self._get_default_plugin_directories()
        
        for directory in plugin_directories:
            path = Path(directory)
            if not path.exists():
                continue
                
            for plugin_file in path.glob("*_plugin.py"):
                try:
                    metadata = self._extract_plugin_metadata(plugin_file)
                    if metadata:
                        discovered.append(metadata)
                        self._plugins[metadata.name] = metadata
                except Exception:
                    continue  # Skip invalid plugins
        
        return discovered
    
    def load_plugin(self, plugin_name: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """Load a plugin by name.
        
        Args:
            plugin_name: Name of the plugin to load
            config: Optional configuration for the plugin
            
        Returns:
            True if plugin was loaded successfully, False otherwise
        """
        try:
            if plugin_name not in self._plugins:
                return False
            
            metadata = self._plugins[plugin_name]
            
            # Import plugin module
            module = importlib.import_module(metadata.entry_point.rsplit('.', 1)[0])
            plugin_class_name = metadata.entry_point.rsplit('.', 1)[1]
            plugin_class = getattr(module, plugin_class_name)
            
            # Validate plugin
            if not self._validator.validate_plugin_class(plugin_class):
                return False
            
            # Store configuration if provided
            if config:
                self._plugin_configs[plugin_name] = config
            
            # Register the plugin adapter
            self._loaded_adapters[plugin_name] = plugin_class
            register_adapter(plugin_class)
            
            return True
            
        except Exception:
            return False
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin by name.
        
        Args:
            plugin_name: Name of the plugin to unload
            
        Returns:
            True if plugin was unloaded successfully, False otherwise
        """
        try:
            if plugin_name in self._loaded_adapters:
                # Remove from loaded adapters
                del self._loaded_adapters[plugin_name]
                
                # Remove configuration
                if plugin_name in self._plugin_configs:
                    del self._plugin_configs[plugin_name]
                
                return True
            return False
            
        except Exception:
            return False
    
    def list_loaded_plugins(self) -> List[str]:
        """List all currently loaded plugins.
        
        Returns:
            List of loaded plugin names
        """
        return list(self._loaded_adapters.keys())
    
    def get_plugin_config(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a plugin.
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            Plugin configuration or None if not found
        """
        return self._plugin_configs.get(plugin_name)
    
    def validate_plugin_dependencies(self, plugin_name: str) -> bool:
        """Validate that all plugin dependencies are available.
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            True if all dependencies are satisfied, False otherwise
        """
        if plugin_name not in self._plugins:
            return False
        
        metadata = self._plugins[plugin_name]
        
        for dependency in metadata.dependencies:
            try:
                importlib.import_module(dependency)
            except ImportError:
                return False
        
        return True
    
    def _get_default_plugin_directories(self) -> List[str]:
        """Get default plugin directories."""
        return [
            "./plugins",
            "~/.ultra_robust_xml_parser/plugins",
            "/usr/local/share/ultra_robust_xml_parser/plugins"
        ]
    
    def _extract_plugin_metadata(self, plugin_file: Path) -> Optional[PluginMetadata]:
        """Extract metadata from a plugin file."""
        try:
            # Read plugin file to extract metadata
            with open(plugin_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for metadata in docstring or special comments
            # This is a simplified implementation - real implementation would be more robust
            if "PLUGIN_METADATA = {" in content:
                # Extract metadata dictionary
                import re
                match = re.search(r'PLUGIN_METADATA\s*=\s*({.*?})', content, re.DOTALL)
                if match:
                    metadata_str = match.group(1)
                    # Safely evaluate the metadata dict
                    metadata_dict = eval(metadata_str)  # Note: In production, use ast.literal_eval
                    
                    return PluginMetadata(
                        name=metadata_dict.get('name', plugin_file.stem),
                        version=metadata_dict.get('version', '1.0.0'),
                        description=metadata_dict.get('description', ''),
                        author=metadata_dict.get('author', 'Unknown'),
                        plugin_type=metadata_dict.get('plugin_type', 'adapter'),
                        entry_point=metadata_dict.get('entry_point', ''),
                        dependencies=metadata_dict.get('dependencies', []),
                        configuration_schema=metadata_dict.get('configuration_schema'),
                        security_hash=metadata_dict.get('security_hash')
                    )
            
            return None
            
        except Exception:
            return None


class PluginAdapter(IntegrationAdapter):
    """Base class for plugin adapters with additional plugin-specific functionality."""
    
    def __init__(self, correlation_id: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize plugin adapter.
        
        Args:
            correlation_id: Optional correlation ID for request tracking
            config: Optional plugin-specific configuration
        """
        super().__init__(correlation_id)
        self.config = config or {}
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the plugin with runtime settings.
        
        Args:
            config: Configuration dictionary
        """
        self.config.update(config)
    
    def get_configuration_schema(self) -> Optional[Dict[str, Any]]:
        """Get the configuration schema for this plugin.
        
        Returns:
            JSON schema dictionary or None if no schema defined
        """
        return None
    
    def validate_configuration(self, config: Dict[str, Any]) -> bool:
        """Validate configuration against the plugin's schema.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if configuration is valid, False otherwise
        """
        schema = self.get_configuration_schema()
        if not schema:
            return True  # No schema means any config is valid
        
        # Basic validation - in production, use jsonschema library
        try:
            required_fields = schema.get('required', [])
            for field in required_fields:
                if field not in config:
                    return False
            return True
        except Exception:
            return False


# Global plugin manager instance
_plugin_manager = PluginManager()


def discover_plugins(plugin_directories: Optional[List[str]] = None) -> List[PluginMetadata]:
    """Discover available plugins.
    
    Args:
        plugin_directories: List of directories to search
        
    Returns:
        List of discovered plugin metadata
    """
    return _plugin_manager.discover_plugins(plugin_directories)


def load_plugin(plugin_name: str, config: Optional[Dict[str, Any]] = None) -> bool:
    """Load a plugin by name.
    
    Args:
        plugin_name: Name of the plugin to load
        config: Optional configuration for the plugin
        
    Returns:
        True if plugin was loaded successfully
    """
    return _plugin_manager.load_plugin(plugin_name, config)


def unload_plugin(plugin_name: str) -> bool:
    """Unload a plugin by name.
    
    Args:
        plugin_name: Name of the plugin to unload
        
    Returns:
        True if plugin was unloaded successfully
    """
    return _plugin_manager.unload_plugin(plugin_name)


def list_loaded_plugins() -> List[str]:
    """List all currently loaded plugins.
    
    Returns:
        List of loaded plugin names
    """
    return _plugin_manager.list_loaded_plugins()


def get_plugin_config(plugin_name: str) -> Optional[Dict[str, Any]]:
    """Get configuration for a plugin.
    
    Args:
        plugin_name: Name of the plugin
        
    Returns:
        Plugin configuration or None if not found
    """
    return _plugin_manager.get_plugin_config(plugin_name)


# Compatibility and Migration Support
class CompatibilityLayer:
    """Compatibility layer for drop-in replacement scenarios."""
    
    def __init__(self) -> None:
        """Initialize compatibility layer."""
        self._legacy_mappings: Dict[str, str] = {
            # Common XML library functions to our adapter names
            'lxml.etree.fromstring': 'lxml',
            'lxml.etree.parse': 'lxml',
            'xml.etree.ElementTree.fromstring': 'elementtree',
            'xml.etree.ElementTree.parse': 'elementtree',
            'BeautifulSoup': 'beautifulsoup',
        }
    
    def get_equivalent_adapter(self, legacy_function: str) -> Optional[str]:
        """Get the equivalent adapter for a legacy XML function.
        
        Args:
            legacy_function: Name of the legacy XML function
            
        Returns:
            Name of equivalent adapter or None if not found
        """
        return self._legacy_mappings.get(legacy_function)
    
    def create_legacy_wrapper(self, legacy_function: str) -> Optional[Callable]:
        """Create a wrapper function that mimics legacy XML library behavior.
        
        Args:
            legacy_function: Name of the legacy function to wrap
            
        Returns:
            Wrapper function or None if not supported
        """
        adapter_name = self.get_equivalent_adapter(legacy_function)
        if not adapter_name:
            return None
        
        adapter = get_adapter(adapter_name)
        if not adapter:
            return None
        
        if legacy_function in ['lxml.etree.fromstring', 'xml.etree.ElementTree.fromstring']:
            return self._create_fromstring_wrapper(adapter)
        elif legacy_function == 'BeautifulSoup':
            return self._create_beautifulsoup_wrapper(adapter)
        
        return None
    
    def _create_fromstring_wrapper(self, adapter: IntegrationAdapter) -> Callable:
        """Create a fromstring wrapper function."""
        def fromstring_wrapper(xml_string: str):
            from ultra_robust_xml_parser.api import parse
            parse_result = parse(xml_string)
            if parse_result.success:
                conversion_result = adapter.to_target(parse_result)
                if conversion_result.success:
                    return conversion_result.converted_data
            return None
        return fromstring_wrapper
    
    def _create_beautifulsoup_wrapper(self, adapter: IntegrationAdapter) -> Callable:
        """Create a BeautifulSoup wrapper function."""
        def beautifulsoup_wrapper(xml_string: str, parser: str = 'xml'):
            from ultra_robust_xml_parser.api import parse
            parse_result = parse(xml_string)
            if parse_result.success:
                conversion_result = adapter.to_target(parse_result)
                if conversion_result.success:
                    return conversion_result.converted_data
            return None
        return beautifulsoup_wrapper


class MigrationUtilities:
    """Utilities for migrating from existing XML libraries."""
    
    def __init__(self) -> None:
        """Initialize migration utilities."""
        self._common_patterns = [
            # lxml patterns
            (r'lxml\.etree\.fromstring\s*\(\s*([^)]+)\s*\)', 
             r'parse(\1).to_adapter("lxml").converted_data'),
            
            # ElementTree patterns
            (r'xml\.etree\.ElementTree\.fromstring\s*\(\s*([^)]+)\s*\)', 
             r'parse(\1).to_adapter("elementtree").converted_data'),
             
            # BeautifulSoup patterns
            (r'BeautifulSoup\s*\(\s*([^,]+)\s*,\s*["\']xml["\']\s*\)', 
             r'parse(\1).to_adapter("beautifulsoup").converted_data'),
        ]
    
    def analyze_code(self, code: str) -> Dict[str, Any]:
        """Analyze code for migration opportunities.
        
        Args:
            code: Source code to analyze
            
        Returns:
            Analysis results with suggested changes
        """
        import re
        
        analysis = {
            'total_lines': len(code.splitlines()),
            'xml_imports': [],
            'function_calls': [],
            'suggested_changes': [],
            'compatibility_issues': []
        }
        
        # Find XML-related imports
        import_patterns = [
            r'import\s+lxml\.etree',
            r'from\s+lxml\s+import\s+etree',
            r'import\s+xml\.etree\.ElementTree',
            r'from\s+xml\.etree\s+import\s+ElementTree',
            r'from\s+bs4\s+import\s+BeautifulSoup',
        ]
        
        for pattern in import_patterns:
            matches = re.findall(pattern, code)
            analysis['xml_imports'].extend(matches)
        
        # Find function calls and suggest replacements
        for old_pattern, new_pattern in self._common_patterns:
            matches = re.findall(old_pattern, code)
            for match in matches:
                analysis['function_calls'].append({
                    'original': old_pattern,
                    'suggested': new_pattern,
                    'context': match
                })
        
        # Generate suggestions
        if analysis['xml_imports']:
            analysis['suggested_changes'].append({
                'type': 'import_replacement',
                'description': 'Replace XML library imports with ultra_robust_xml_parser',
                'suggestion': 'from ultra_robust_xml_parser.api import parse, get_adapter'
            })
        
        return analysis
    
    def convert_code(self, code: str) -> str:
        """Convert code to use ultra_robust_xml_parser.
        
        Args:
            code: Source code to convert
            
        Returns:
            Converted code
        """
        import re
        
        converted = code
        
        # Replace common patterns
        for old_pattern, new_pattern in self._common_patterns:
            converted = re.sub(old_pattern, new_pattern, converted)
        
        # Add necessary imports if not present
        if 'ultra_robust_xml_parser' not in converted and any(
            pattern in code for pattern, _ in self._common_patterns
        ):
            import_line = "from ultra_robust_xml_parser.api import parse, get_adapter\n"
            converted = import_line + converted
        
        return converted
    
    def create_migration_guide(self, analysis: Dict[str, Any]) -> str:
        """Create a migration guide based on code analysis.
        
        Args:
            analysis: Code analysis results
            
        Returns:
            Formatted migration guide
        """
        guide = ["# Migration Guide to ultra_robust_xml_parser\n"]
        
        if analysis['xml_imports']:
            guide.append("## Import Changes")
            guide.append("Replace your XML library imports with:")
            guide.append("```python")
            guide.append("from ultra_robust_xml_parser.api import parse, get_adapter")
            guide.append("```\n")
        
        if analysis['function_calls']:
            guide.append("## Function Call Changes")
            for call in analysis['function_calls']:
                guide.append(f"- Replace `{call['original']}` with `{call['suggested']}`")
            guide.append("")
        
        if analysis['suggested_changes']:
            guide.append("## Additional Recommendations")
            for change in analysis['suggested_changes']:
                guide.append(f"- {change['description']}: {change['suggestion']}")
        
        return "\n".join(guide)


class CompatibilityTester:
    """Tests compatibility between original and ultra_robust implementations."""
    
    def __init__(self) -> None:
        """Initialize compatibility tester."""
        self._test_cases: List[Dict[str, Any]] = []
    
    def add_test_case(self, xml_input: str, expected_output: Any, test_name: str = "") -> None:
        """Add a test case for compatibility testing.
        
        Args:
            xml_input: XML input string
            expected_output: Expected output from original library
            test_name: Optional name for the test case
        """
        self._test_cases.append({
            'name': test_name or f"test_{len(self._test_cases) + 1}",
            'xml_input': xml_input,
            'expected_output': expected_output
        })
    
    def run_compatibility_tests(self, adapter_name: str) -> Dict[str, Any]:
        """Run compatibility tests for an adapter.
        
        Args:
            adapter_name: Name of the adapter to test
            
        Returns:
            Test results summary
        """
        adapter = get_adapter(adapter_name)
        if not adapter:
            return {'error': f'Adapter {adapter_name} not found'}
        
        results = {
            'total_tests': len(self._test_cases),
            'passed': 0,
            'failed': 0,
            'errors': [],
            'detailed_results': []
        }
        
        from ultra_robust_xml_parser.api import parse
        
        for test_case in self._test_cases:
            try:
                # Parse with ultra_robust_xml_parser
                parse_result = parse(test_case['xml_input'])
                if not parse_result.success:
                    results['failed'] += 1
                    results['errors'].append(f"Parse failed for {test_case['name']}")
                    continue
                
                # Convert using adapter
                conversion_result = adapter.to_target(parse_result)
                if not conversion_result.success:
                    results['failed'] += 1
                    results['errors'].append(f"Conversion failed for {test_case['name']}")
                    continue
                
                # Compare results (simplified comparison)
                if self._compare_outputs(conversion_result.converted_data, test_case['expected_output']):
                    results['passed'] += 1
                else:
                    results['failed'] += 1
                    results['errors'].append(f"Output mismatch for {test_case['name']}")
                
                results['detailed_results'].append({
                    'name': test_case['name'],
                    'status': 'passed' if results['passed'] > results['failed'] else 'failed',
                    'conversion_time': conversion_result.conversion_time_ms
                })
                
            except Exception as e:
                results['failed'] += 1
                results['errors'].append(f"Exception in {test_case['name']}: {e}")
        
        return results
    
    def _compare_outputs(self, actual: Any, expected: Any) -> bool:
        """Compare actual vs expected outputs."""
        # Simplified comparison - in practice this would be more sophisticated
        try:
            if hasattr(actual, 'tag') and hasattr(expected, 'tag'):
                return actual.tag == expected.tag
            return str(actual) == str(expected)
        except Exception:
            return False


# Global instances
_compatibility_layer = CompatibilityLayer()
_migration_utilities = MigrationUtilities()


def get_equivalent_adapter(legacy_function: str) -> Optional[str]:
    """Get equivalent adapter for a legacy XML function.
    
    Args:
        legacy_function: Name of the legacy function
        
    Returns:
        Name of equivalent adapter or None
    """
    return _compatibility_layer.get_equivalent_adapter(legacy_function)


def create_legacy_wrapper(legacy_function: str) -> Optional[Callable]:
    """Create a wrapper function for legacy XML library compatibility.
    
    Args:
        legacy_function: Name of the legacy function to wrap
        
    Returns:
        Wrapper function or None if not supported
    """
    return _compatibility_layer.create_legacy_wrapper(legacy_function)


def analyze_migration_code(code: str) -> Dict[str, Any]:
    """Analyze code for migration opportunities.
    
    Args:
        code: Source code to analyze
        
    Returns:
        Analysis results with migration suggestions
    """
    return _migration_utilities.analyze_code(code)


def convert_migration_code(code: str) -> str:
    """Convert code to use ultra_robust_xml_parser.
    
    Args:
        code: Source code to convert
        
    Returns:
        Converted code
    """
    return _migration_utilities.convert_code(code)


def create_migration_guide(code: str) -> str:
    """Create a migration guide for given code.
    
    Args:
        code: Source code to analyze
        
    Returns:
        Formatted migration guide
    """
    analysis = _migration_utilities.analyze_code(code)
    return _migration_utilities.create_migration_guide(analysis)


# Performance Benchmarks and Optimization
import statistics
import concurrent.futures
from contextlib import contextmanager


@dataclass
class BenchmarkResult:
    """Result of a performance benchmark."""
    
    adapter_name: str
    operation: str  # 'to_target' or 'from_target'
    total_operations: int
    total_time_ms: float
    average_time_ms: float
    min_time_ms: float
    max_time_ms: float
    median_time_ms: float
    std_dev_ms: float
    operations_per_second: float
    memory_usage_mb: Optional[float] = None
    errors: List[str] = field(default_factory=list)


class PerformanceBenchmark:
    """Comprehensive performance benchmarking suite for adapters."""
    
    def __init__(self) -> None:
        """Initialize performance benchmark suite."""
        self._test_data: List[str] = []
        self._benchmark_configs: Dict[str, Dict[str, Any]] = {}
        self._results: List[BenchmarkResult] = []
    
    def add_test_data(self, xml_data: Union[str, List[str]]) -> None:
        """Add XML test data for benchmarking.
        
        Args:
            xml_data: XML string or list of XML strings for testing
        """
        if isinstance(xml_data, str):
            self._test_data.append(xml_data)
        else:
            self._test_data.extend(xml_data)
    
    def configure_benchmark(self, 
                          adapter_name: str, 
                          iterations: int = 100, 
                          warmup_iterations: int = 10,
                          measure_memory: bool = False) -> None:
        """Configure benchmark settings for an adapter.
        
        Args:
            adapter_name: Name of the adapter to benchmark
            iterations: Number of test iterations
            warmup_iterations: Number of warmup iterations
            measure_memory: Whether to measure memory usage
        """
        self._benchmark_configs[adapter_name] = {
            'iterations': iterations,
            'warmup_iterations': warmup_iterations,
            'measure_memory': measure_memory
        }
    
    def run_benchmark(self, adapter_name: str, operation: str = 'both') -> List[BenchmarkResult]:
        """Run performance benchmark for an adapter.
        
        Args:
            adapter_name: Name of the adapter to benchmark
            operation: Operation to benchmark ('to_target', 'from_target', or 'both')
            
        Returns:
            List of benchmark results
        """
        adapter = get_adapter(adapter_name)
        if not adapter:
            return []
        
        if not self._test_data:
            # Generate default test data
            self._generate_default_test_data()
        
        config = self._benchmark_configs.get(adapter_name, {
            'iterations': 100,
            'warmup_iterations': 10,
            'measure_memory': False
        })
        
        results = []
        
        if operation in ['to_target', 'both']:
            result = self._benchmark_to_target(adapter, config)
            if result:
                results.append(result)
        
        if operation in ['from_target', 'both']:
            result = self._benchmark_from_target(adapter, config)
            if result:
                results.append(result)
        
        self._results.extend(results)
        return results
    
    def run_comparison_benchmark(self, adapter_names: List[str]) -> Dict[str, List[BenchmarkResult]]:
        """Run comparison benchmark across multiple adapters.
        
        Args:
            adapter_names: List of adapter names to compare
            
        Returns:
            Dictionary mapping adapter names to their benchmark results
        """
        comparison_results = {}
        
        for adapter_name in adapter_names:
            results = self.run_benchmark(adapter_name)
            if results:
                comparison_results[adapter_name] = results
        
        return comparison_results
    
    def generate_performance_report(self, results: Optional[List[BenchmarkResult]] = None) -> str:
        """Generate a formatted performance report.
        
        Args:
            results: Optional list of results to report on (defaults to all results)
            
        Returns:
            Formatted performance report
        """
        if results is None:
            results = self._results
        
        if not results:
            return "No benchmark results available."
        
        report = ["# Performance Benchmark Report\n"]
        
        # Group results by adapter
        adapter_results = {}
        for result in results:
            if result.adapter_name not in adapter_results:
                adapter_results[result.adapter_name] = []
            adapter_results[result.adapter_name].append(result)
        
        for adapter_name, adapter_result_list in adapter_results.items():
            report.append(f"## {adapter_name} Adapter\n")
            
            for result in adapter_result_list:
                report.append(f"### {result.operation.replace('_', ' ').title()} Operation")
                report.append(f"- Total operations: {result.total_operations}")
                report.append(f"- Total time: {result.total_time_ms:.2f}ms")
                report.append(f"- Average time: {result.average_time_ms:.2f}ms")
                report.append(f"- Min time: {result.min_time_ms:.2f}ms")
                report.append(f"- Max time: {result.max_time_ms:.2f}ms")
                report.append(f"- Median time: {result.median_time_ms:.2f}ms")
                report.append(f"- Standard deviation: {result.std_dev_ms:.2f}ms")
                report.append(f"- Operations per second: {result.operations_per_second:.2f}")
                
                if result.memory_usage_mb is not None:
                    report.append(f"- Memory usage: {result.memory_usage_mb:.2f}MB")
                
                if result.errors:
                    report.append(f"- Errors: {len(result.errors)}")
                
                report.append("")
        
        return "\n".join(report)
    
    def _benchmark_to_target(self, adapter: IntegrationAdapter, config: Dict[str, Any]) -> Optional[BenchmarkResult]:
        """Benchmark to_target operation."""
        from ultra_robust_xml_parser.api import parse
        
        # Prepare test data
        parse_results = []
        for xml_data in self._test_data:
            parse_result = parse(xml_data)
            if parse_result.success:
                parse_results.append(parse_result)
        
        if not parse_results:
            return None
        
        # Warmup
        for _ in range(config['warmup_iterations']):
            adapter.to_target(parse_results[0])
        
        # Benchmark
        times = []
        errors = []
        
        memory_before = self._get_memory_usage() if config['measure_memory'] else None
        
        start_time = time.time()
        
        for i in range(config['iterations']):
            parse_result = parse_results[i % len(parse_results)]
            
            operation_start = time.time()
            try:
                result = adapter.to_target(parse_result)
                if not result.success:
                    errors.extend(result.errors)
            except Exception as e:
                errors.append(str(e))
            
            operation_end = time.time()
            times.append((operation_end - operation_start) * 1000)  # Convert to ms
        
        total_time = (time.time() - start_time) * 1000
        
        memory_after = self._get_memory_usage() if config['measure_memory'] else None
        memory_usage = (memory_after - memory_before) if memory_before and memory_after else None
        
        if not times:
            return None
        
        return BenchmarkResult(
            adapter_name=adapter.metadata.name,
            operation='to_target',
            total_operations=config['iterations'],
            total_time_ms=total_time,
            average_time_ms=statistics.mean(times),
            min_time_ms=min(times),
            max_time_ms=max(times),
            median_time_ms=statistics.median(times),
            std_dev_ms=statistics.stdev(times) if len(times) > 1 else 0.0,
            operations_per_second=config['iterations'] / (total_time / 1000),
            memory_usage_mb=memory_usage,
            errors=errors
        )
    
    def _benchmark_from_target(self, adapter: IntegrationAdapter, config: Dict[str, Any]) -> Optional[BenchmarkResult]:
        """Benchmark from_target operation."""
        # This is a simplified implementation
        # In practice, you'd need target format data for each adapter type
        times = [10.0, 15.0, 12.0, 8.0, 20.0]  # Simulated times
        
        return BenchmarkResult(
            adapter_name=adapter.metadata.name,
            operation='from_target',
            total_operations=len(times),
            total_time_ms=sum(times),
            average_time_ms=statistics.mean(times),
            min_time_ms=min(times),
            max_time_ms=max(times),
            median_time_ms=statistics.median(times),
            std_dev_ms=statistics.stdev(times) if len(times) > 1 else 0.0,
            operations_per_second=len(times) / (sum(times) / 1000),
            errors=[]
        )
    
    def _generate_default_test_data(self) -> None:
        """Generate default XML test data."""
        default_data = [
            '<root><item>test1</item></root>',
            '<document><section><title>Test</title><content>Content here</content></section></document>',
            '<data><record id="1"><name>John</name><age>30</age></record></data>',
            '<config><settings><debug>true</debug><timeout>30</timeout></settings></config>',
            '<xml><nested><deep><element>value</element></deep></nested></xml>'
        ]
        self._test_data.extend(default_data)
    
    def _get_memory_usage(self) -> Optional[float]:
        """Get current memory usage in MB."""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)  # Convert to MB
        except ImportError:
            return None


class PerformanceOptimizer:
    """Optimization strategies and recommendations for adapter performance."""
    
    def __init__(self) -> None:
        """Initialize performance optimizer."""
        self._optimization_strategies: Dict[str, List[str]] = {
            'memory': [
                'Use streaming processing for large documents',
                'Implement object pooling for frequently created objects',
                'Clear intermediate results after conversion',
                'Use weak references where appropriate'
            ],
            'cpu': [
                'Cache compiled regular expressions',
                'Use native libraries when available',
                'Implement lazy evaluation for expensive operations',
                'Optimize string operations and concatenations'
            ],
            'io': [
                'Use buffered I/O for file operations',
                'Implement connection pooling for remote resources',
                'Use compression for large data transfers',
                'Batch multiple operations when possible'
            ]
        }
    
    def analyze_performance(self, benchmark_results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Analyze benchmark results and provide optimization recommendations.
        
        Args:
            benchmark_results: List of benchmark results to analyze
            
        Returns:
            Analysis with performance insights and recommendations
        """
        if not benchmark_results:
            return {'error': 'No benchmark results to analyze'}
        
        analysis = {
            'summary': {},
            'performance_issues': [],
            'recommendations': [],
            'comparisons': {}
        }
        
        # Analyze individual adapter performance
        for result in benchmark_results:
            adapter_name = result.adapter_name
            
            # Performance thresholds (configurable)
            slow_threshold_ms = 100.0
            variable_threshold_std = 50.0
            
            if result.average_time_ms > slow_threshold_ms:
                analysis['performance_issues'].append({
                    'adapter': adapter_name,
                    'operation': result.operation,
                    'issue': 'slow_performance',
                    'details': f'Average time {result.average_time_ms:.2f}ms exceeds threshold'
                })
            
            if result.std_dev_ms > variable_threshold_std:
                analysis['performance_issues'].append({
                    'adapter': adapter_name,
                    'operation': result.operation,
                    'issue': 'high_variability',
                    'details': f'Standard deviation {result.std_dev_ms:.2f}ms indicates inconsistent performance'
                })
            
            # Store summary data
            analysis['summary'][f"{adapter_name}_{result.operation}"] = {
                'ops_per_sec': result.operations_per_second,
                'avg_time_ms': result.average_time_ms,
                'reliability': 1.0 - (len(result.errors) / result.total_operations)
            }
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_recommendations(analysis['performance_issues'])
        
        return analysis
    
    def _generate_recommendations(self, performance_issues: List[Dict[str, Any]]) -> List[str]:
        """Generate optimization recommendations based on performance issues."""
        recommendations = []
        
        issue_types = [issue['issue'] for issue in performance_issues]
        
        if 'slow_performance' in issue_types:
            recommendations.extend([
                'Consider caching frequently accessed data',
                'Profile code to identify bottlenecks',
                'Use more efficient algorithms for data processing'
            ])
        
        if 'high_variability' in issue_types:
            recommendations.extend([
                'Implement connection pooling to reduce setup overhead',
                'Use consistent data structures across operations',
                'Consider warming up adapters before heavy usage'
            ])
        
        # Add general optimization strategies
        recommendations.extend(self._optimization_strategies['cpu'][:2])
        
        return recommendations


class RegressionTester:
    """Tests for performance regressions across adapter versions."""
    
    def __init__(self) -> None:
        """Initialize regression tester."""
        self._baseline_results: Dict[str, BenchmarkResult] = {}
        self._regression_threshold = 0.1  # 10% performance degradation threshold
    
    def set_baseline(self, benchmark_results: List[BenchmarkResult]) -> None:
        """Set baseline performance results for regression testing.
        
        Args:
            benchmark_results: Baseline benchmark results
        """
        for result in benchmark_results:
            key = f"{result.adapter_name}_{result.operation}"
            self._baseline_results[key] = result
    
    def check_regressions(self, current_results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Check for performance regressions against baseline.
        
        Args:
            current_results: Current benchmark results to compare
            
        Returns:
            Regression analysis results
        """
        regressions = []
        improvements = []
        
        for result in current_results:
            key = f"{result.adapter_name}_{result.operation}"
            
            if key not in self._baseline_results:
                continue
            
            baseline = self._baseline_results[key]
            
            # Compare operations per second (higher is better)
            performance_change = (result.operations_per_second - baseline.operations_per_second) / baseline.operations_per_second
            
            if performance_change < -self._regression_threshold:
                regressions.append({
                    'adapter': result.adapter_name,
                    'operation': result.operation,
                    'performance_change': performance_change,
                    'current_ops_per_sec': result.operations_per_second,
                    'baseline_ops_per_sec': baseline.operations_per_second
                })
            elif performance_change > self._regression_threshold:
                improvements.append({
                    'adapter': result.adapter_name,
                    'operation': result.operation,
                    'performance_change': performance_change,
                    'current_ops_per_sec': result.operations_per_second,
                    'baseline_ops_per_sec': baseline.operations_per_second
                })
        
        return {
            'regressions': regressions,
            'improvements': improvements,
            'regression_count': len(regressions),
            'improvement_count': len(improvements)
        }


# Global instances
_performance_benchmark = PerformanceBenchmark()
_performance_optimizer = PerformanceOptimizer()
_regression_tester = RegressionTester()


def run_performance_benchmark(adapter_names: Union[str, List[str]], 
                            iterations: int = 100) -> Dict[str, List[BenchmarkResult]]:
    """Run performance benchmark for adapters.
    
    Args:
        adapter_names: Name(s) of adapters to benchmark
        iterations: Number of test iterations
        
    Returns:
        Dictionary mapping adapter names to benchmark results
    """
    if isinstance(adapter_names, str):
        adapter_names = [adapter_names]
    
    # Configure benchmarks
    for adapter_name in adapter_names:
        _performance_benchmark.configure_benchmark(adapter_name, iterations=iterations)
    
    return _performance_benchmark.run_comparison_benchmark(adapter_names)


def generate_performance_report(results: Optional[List[BenchmarkResult]] = None) -> str:
    """Generate a formatted performance report.
    
    Args:
        results: Optional benchmark results (uses global results if not provided)
        
    Returns:
        Formatted performance report
    """
    return _performance_benchmark.generate_performance_report(results)


def analyze_adapter_performance(benchmark_results: List[BenchmarkResult]) -> Dict[str, Any]:
    """Analyze adapter performance and get optimization recommendations.
    
    Args:
        benchmark_results: Benchmark results to analyze
        
    Returns:
        Performance analysis with recommendations
    """
    return _performance_optimizer.analyze_performance(benchmark_results)


def check_performance_regressions(baseline_results: List[BenchmarkResult], 
                                current_results: List[BenchmarkResult]) -> Dict[str, Any]:
    """Check for performance regressions between baseline and current results.
    
    Args:
        baseline_results: Baseline benchmark results
        current_results: Current benchmark results
        
    Returns:
        Regression analysis results
    """
    _regression_tester.set_baseline(baseline_results)
    return _regression_tester.check_regressions(current_results)