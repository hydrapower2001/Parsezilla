# Ultra Robust XML Parser - Examples and Integration Templates

This directory contains examples, templates, and integration patterns for the ultra_robust_xml_parser library. These resources help developers quickly integrate the library into their projects and learn best practices for various use cases.

## Directory Structure

```
examples/
├── README.md                          # This file
├── templates/                         # Templates for custom development
│   └── custom_adapter_template.py     # Template for creating custom adapters
└── integrations/                      # Integration examples
    ├── flask_xml_api_example.py       # Flask REST API integration
    ├── pandas_data_analysis_example.py # Pandas data analysis workflow
    └── migration_helper_example.py     # Migration from legacy XML libraries
```

## Templates

### Custom Adapter Template
- **File**: `templates/custom_adapter_template.py`
- **Purpose**: Provides a complete template for creating custom integration adapters
- **Features**:
  - Plugin metadata definition
  - Bidirectional conversion implementation
  - Error handling and performance tracking
  - Validation and testing examples
  - Comprehensive documentation

**Usage**:
```bash
# Copy the template
cp templates/custom_adapter_template.py my_custom_adapter.py

# Customize for your target format
# Edit the class name, metadata, and conversion logic
# Test your adapter implementation
python my_custom_adapter.py
```

## Integration Examples

### Flask REST API Integration
- **File**: `integrations/flask_xml_api_example.py`
- **Purpose**: Demonstrates building XML processing REST APIs with Flask
- **Features**:
  - XML parsing endpoint with error handling
  - Format conversion endpoints
  - XML validation with detailed diagnostics
  - Performance benchmarking endpoints
  - Proper HTTP status codes and XML responses

**Endpoints**:
- `POST /xml/parse` - Parse XML and return analysis
- `POST /xml/convert/<format>` - Convert XML to specified format
- `POST /xml/validate` - Validate XML structure
- `POST /xml/benchmark/<adapter>` - Benchmark adapter performance

**Run the example**:
```bash
cd examples/integrations/
python flask_xml_api_example.py

# Test with curl
curl -X POST http://localhost:5000/xml/parse \
  -H "Content-Type: application/xml" \
  -d "<books><book><title>Test</title></book></books>"
```

### Pandas Data Analysis Integration
- **File**: `integrations/pandas_data_analysis_example.py`
- **Purpose**: Shows how to convert XML data to pandas DataFrames for analysis
- **Features**:
  - XML to DataFrame conversion
  - Business data extraction and analysis
  - Statistical analysis with pandas
  - Data visualization preparation
  - Reverse conversion demonstration

**Run the example**:
```bash
cd examples/integrations/
pip install pandas  # if not already installed
python pandas_data_analysis_example.py
```

### Migration Helper
- **File**: `integrations/migration_helper_example.py`
- **Purpose**: Helps migrate from existing XML libraries (lxml, ElementTree, BeautifulSoup)
- **Features**:
  - Automated code analysis
  - Migration opportunity identification
  - Automatic code conversion
  - Legacy wrapper functions
  - Complete migration strategy and checklist

**Run the example**:
```bash
cd examples/integrations/
python migration_helper_example.py
```

## Quick Start Guide

### 1. Basic Usage
```python
from ultra_robust_xml_parser.api import parse, get_adapter

# Parse XML
xml_data = "<root><item>value</item></root>"
result = parse(xml_data)

if result.success:
    print(f"Parsed successfully (confidence: {result.confidence})")
    
    # Convert to different formats
    pandas_adapter = get_adapter('pandas')
    if pandas_adapter:
        df_result = pandas_adapter.to_target(result)
        print(f"DataFrame shape: {df_result.converted_data.shape}")
```

### 2. Web Framework Integration
```python
from flask import Flask, request
from ultra_robust_xml_parser.api import parse, get_adapter

app = Flask(__name__)

@app.route('/xml', methods=['POST'])
def process_xml():
    # Get Flask adapter for seamless request/response handling
    flask_adapter = get_adapter('flask')
    conversion_result = flask_adapter.from_target(request)
    
    if conversion_result.success:
        parse_result = conversion_result.converted_data
        # Process the parsed XML...
        return flask_adapter.to_target(parse_result).converted_data
```

### 3. Custom Adapter Development
```python
from ultra_robust_xml_parser.api.adapters import IntegrationAdapter, AdapterMetadata

class MyCustomAdapter(IntegrationAdapter):
    @property
    def metadata(self):
        return AdapterMetadata(
            name="my-adapter",
            adapter_type=AdapterType.PLUGIN,
            # ... other metadata
        )
    
    def to_target(self, parse_result):
        # Convert ParseResult to your format
        pass
    
    def from_target(self, target_data):
        # Convert your format to ParseResult
        pass
```

### 4. Performance Benchmarking
```python
from ultra_robust_xml_parser.api import run_performance_benchmark

# Benchmark multiple adapters
results = run_performance_benchmark(['elementtree', 'pandas'], iterations=100)

# Analyze performance
for adapter_name, benchmark_results in results.items():
    for result in benchmark_results:
        print(f"{adapter_name} {result.operation}: {result.operations_per_second:.2f} ops/sec")
```

## Best Practices

### Error Handling
```python
# Always check for success
result = parse(xml_data)
if not result.success:
    for diagnostic in result.diagnostics:
        print(f"{diagnostic.severity}: {diagnostic.message}")

# Adapter conversions
conversion_result = adapter.to_target(parse_result)
if not conversion_result.success:
    print(f"Conversion errors: {conversion_result.errors}")
```

### Performance Optimization
```python
# Use correlation IDs for tracking
result = parse(xml_data, correlation_id="request-123")

# Monitor adapter performance
adapter = get_adapter('pandas')
stats = adapter.get_performance_stats()
print(f"Average conversion time: {stats['pandas']['average_ms']}ms")

# Use appropriate adapters for your use case
if need_dataframe_analysis:
    adapter = get_adapter('pandas')
elif need_web_response:
    adapter = get_adapter('flask')  # or 'django', 'fastapi'
```

### Plugin Development
```python
# Always include plugin metadata
PLUGIN_METADATA = {
    'name': 'my-plugin',
    'version': '1.0.0',
    'description': 'My custom adapter',
    'dependencies': ['required_library'],
    'configuration_schema': {
        'type': 'object',
        'properties': {
            'setting1': {'type': 'string'}
        }
    }
}

# Implement proper validation
def is_available(self):
    try:
        import required_library
        return True
    except ImportError:
        return False
```

## Testing Your Integrations

### Unit Testing
```python
import unittest
from ultra_robust_xml_parser.api import parse

class TestMyIntegration(unittest.TestCase):
    def test_basic_parsing(self):
        result = parse("<test>data</test>")
        self.assertTrue(result.success)
        self.assertIsNotNone(result.tree)
    
    def test_adapter_conversion(self):
        result = parse("<test>data</test>")
        adapter = get_adapter('my-adapter')
        conversion = adapter.to_target(result)
        self.assertTrue(conversion.success)
```

### Performance Testing
```python
from ultra_robust_xml_parser.api import run_performance_benchmark

# Test your adapter performance
results = run_performance_benchmark('my-adapter', iterations=1000)
assert results['my-adapter'][0].operations_per_second > 100  # Performance threshold
```

## Contributing

To contribute new examples or templates:

1. Follow the existing code structure and documentation style
2. Include comprehensive error handling
3. Add usage examples and test cases
4. Update this README with your contribution
5. Ensure code follows the project's coding standards

## Support

For questions about these examples or integration help:
- Check the main documentation
- Review the test files for additional usage patterns
- Open an issue with specific questions or problems

## License

These examples are provided under the same license as the ultra_robust_xml_parser library.