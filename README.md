# Ultra-Robust XML Parser

[![CI](https://github.com/ultra-robust-xml-parser/ultra-robust-xml-parser/workflows/CI/badge.svg)](https://github.com/ultra-robust-xml-parser/ultra-robust-xml-parser/actions)
[![codecov](https://codecov.io/gh/ultra-robust-xml-parser/ultra-robust-xml-parser/branch/main/graph/badge.svg)](https://codecov.io/gh/ultra-robust-xml-parser/ultra-robust-xml-parser)
[![PyPI version](https://badge.fury.io/py/ultra-robust-xml-parser.svg)](https://badge.fury.io/py/ultra-robust-xml-parser)
[![Python Support](https://img.shields.io/pypi/pyversions/ultra-robust-xml-parser.svg)](https://pypi.org/project/ultra-robust-xml-parser/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A never-fail XML parser that handles any malformed XML input gracefully with comprehensive encoding detection, character transformation, and error recovery.

## Features

- **Never-Fail Philosophy**: Always returns usable results, even from severely malformed XML
- **Multi-Stage Encoding Detection**: Automatic encoding detection with confidence scoring
- **Character Transformation**: Configurable handling of invalid characters
- **Streaming Processing**: Constant memory usage regardless of document size
- **Comprehensive Error Recovery**: Repairs malformed XML structure while preserving content
- **Pure Python**: Universal compatibility with optional performance extensions

## Quick Start

### Installation

```bash
pip install ultra-robust-xml-parser
```

### Basic Usage

```python
import ultra_robust_xml_parser

# Parse any XML, no matter how malformed
result = ultra_robust_xml_parser.parse(malformed_xml_string)

# Access parsed content
print(result.elements)
print(f"Confidence: {result.confidence}")
print(f"Issues found: {result.diagnostics}")
```

## Project Structure

This project follows a layered architecture with four distinct processing layers:

1. **Character Processing Layer**: Encoding detection and character transformation
2. **Tokenization Engine**: XML tokenization with error recovery
3. **Tree Building Engine**: Structure repair and content organization  
4. **API & Integration Layer**: Progressive disclosure and integration adapters

## Development

### Prerequisites

- Python 3.8 or higher
- [uv](https://github.com/astral-sh/uv) for package management

### Setup Development Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/ultra-robust-xml-parser/ultra-robust-xml-parser.git
   cd ultra-robust-xml-parser
   ```

2. Install dependencies:
   ```bash
   uv sync --dev
   ```

3. Run tests:
   ```bash
   uv run pytest
   ```

4. Run code quality checks:
   ```bash
   uv run black src/ tests/
   uv run ruff check src/ tests/
   uv run mypy src/ tests/
   ```

### Project Commands

- **Format code**: `uv run black src/ tests/`
- **Lint code**: `uv run ruff check src/ tests/`
- **Type check**: `uv run mypy src/ tests/`
- **Run tests**: `uv run pytest`
- **Run tests with coverage**: `uv run pytest --cov=src/ultra_robust_xml_parser`
- **Build package**: `uv build`

## Architecture

The ultra-robust XML parser is designed as a layered monolithic library with these core principles:

- **Never-fail processing pipeline**: Each layer handles its own errors and passes cleaned results upward
- **Pure Python core with optional extensions**: Universal compatibility with performance optimization capability
- **Streaming-first design**: Constant memory usage regardless of document size
- **Progressive disclosure API**: Simple functions for basic use, rich objects for advanced scenarios

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Testing

The project maintains high testing standards:

- **95%+ test coverage** required
- **Test-Driven Development (TDD)** approach
- **Comprehensive malformation testing** using property-based testing
- **Cross-platform testing** on Linux, Windows, and macOS
- **Multi-version Python support** (3.8-3.12)

## Security

- Input validation and sanitization
- No logging of potentially sensitive XML content
- Regular security audits of dependencies
- Comprehensive fuzz testing for robustness

## Performance

- **Streaming processing** for large documents
- **Memory-efficient** design with constant memory usage
- **Optional Cython extensions** for performance-critical paths
- **Lazy evaluation** and intelligent caching

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with modern Python packaging standards (PEP 621)
- Uses industry-standard tools: black, ruff, mypy, pytest
- Follows semantic versioning
- Comprehensive CI/CD with GitHub Actions

## Developer Tools

The Ultra Robust XML Parser includes comprehensive developer tools for production environments:

### 🚀 Performance Profiling
- Function-level timing analysis with decorators and context managers
- Memory usage tracking and optimization recommendations
- Statistical analysis (averages, percentiles, call graphs)
- Export capabilities (JSON, CSV, HTML reports)

### 🐛 Debug Mode  
- Interactive debugging with configurable logging levels
- Conditional breakpoints and real-time inspection
- Component-specific debugging (encoding, tokenization, tree building)
- Comprehensive debug reports with correlation IDs

### 🧪 Automated Testing
- Comprehensive malformation test case generation
- Edge case and encoding-specific test scenarios
- Configurable complexity levels and coverage targets
- Export to pytest, unittest, or raw JSON formats

### 🧠 Memory Management
- Object pooling for high-throughput scenarios
- Real-time memory monitoring with configurable thresholds
- Streaming buffers for memory-efficient large file processing
- Garbage collection optimization for XML workloads

### 🔒 Security Features
- XML attack detection (XXE, Billion Laughs, ZIP Bombs, Injection)
- Multi-layer input validation and sanitization
- Configurable security policies (PERMISSIVE, STRICT, PARANOID)
- Comprehensive audit logging and security reporting

### 📊 Production Ready
- Health monitoring and alerting
- Structured logging with correlation IDs
- Horizontal scaling support
- Emergency procedures and troubleshooting guides

```python
# Quick example using all tools
from ultra_robust_xml_parser.tools import *

# Initialize integrated tooling
profiler = PerformanceProfiler(enable_memory_tracking=True)
memory_manager = MemoryManager()
security_manager = SecurityManager()

# Process XML with full monitoring
@profiler.profile("secure_xml_processing")
def process_xml_with_monitoring(xml_data):
    # Security validation
    if not security_manager.validate_and_scan(xml_data):
        raise SecurityError("XML failed security validation")
    
    # Memory-efficient processing
    with memory_manager.get_from_pool("strings") as temp_obj:
        return parse_xml(xml_data)

# Generate comprehensive reports
profiler.export_report("performance.json")
memory_manager.export_stats("memory.json") 
security_manager.export_audit_log("security.json")
```

## Roadmap

- [x] Character processing layer implementation
- [x] Tokenization engine with error recovery  
- [x] Tree building with structure repair
- [x] API layer with integration adapters
- [x] **Developer tools and production features** ✨
- [x] **Comprehensive documentation and guides** 📚
- [ ] Performance optimizations with Cython
- [ ] Benchmark suite and performance tracking

## Documentation

- **[API Reference](docs/api/tools.md)** - Complete API documentation for developer tools
- **[User Guide](docs/guides/developer-tools.md)** - Comprehensive usage guide with examples
- **[Tutorial](docs/tutorials/tools-tutorial.md)** - Step-by-step tutorial for all tools
- **[Production Guide](docs/guides/production-deployment.md)** - Production deployment and best practices
- **[Examples](examples/)** - Real-world examples and integration patterns

## Support

- **Documentation**: [docs/](docs/) - Comprehensive documentation and guides
- **Issues**: [GitHub Issues](https://github.com/ultra-robust-xml-parser/ultra-robust-xml-parser/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ultra-robust-xml-parser/ultra-robust-xml-parser/discussions)