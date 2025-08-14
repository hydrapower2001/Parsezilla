# Contributing to Ultra-Robust XML Parser

Thank you for your interest in contributing to the Ultra-Robust XML Parser! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)

## Code of Conduct

This project adheres to a code of conduct that promotes a welcoming and inclusive environment. Please be respectful and professional in all interactions.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- Git for version control
- Basic understanding of XML parsing concepts

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YOUR_USERNAME/ultra-robust-xml-parser.git
   cd ultra-robust-xml-parser
   ```

2. **Set up Development Environment**
   ```bash
   uv sync --dev
   ```

3. **Verify Setup**
   ```bash
   uv run pytest
   uv run black --check src/ tests/
   uv run ruff check src/ tests/
   uv run mypy src/ tests/
   ```

## Development Workflow

### Branch Strategy

- `main`: Stable release branch
- `develop`: Integration branch for new features
- `feature/feature-name`: Feature development branches
- `bugfix/issue-description`: Bug fix branches
- `hotfix/critical-fix`: Critical production fixes

### Creating a Feature Branch

```bash
git checkout develop
git pull origin develop
git checkout -b feature/your-feature-name
```

### Committing Changes

- Write clear, descriptive commit messages
- Follow conventional commit format: `type(scope): description`
- Examples:
  - `feat(parser): add encoding detection with BOM support`
  - `fix(tests): resolve flaky encoding detection test`
  - `docs(readme): update installation instructions`

## Coding Standards

### Code Formatting and Linting

This project uses strict code quality tools:

- **Black**: Code formatting (88 character line length)
- **Ruff**: Fast linting and import sorting
- **Mypy**: Static type checking

Run before committing:
```bash
uv run black src/ tests/
uv run ruff check --fix src/ tests/
uv run mypy src/ tests/
```

### Code Style Guidelines

- **Type Hints**: All functions must have complete type annotations
- **Docstrings**: Use Google-style docstrings for all public functions
- **Naming Conventions**:
  - Classes: PascalCase (`XMLTokenizer`)
  - Functions/Methods: snake_case (`detect_encoding`)
  - Constants: UPPER_SNAKE_CASE (`DEFAULT_ENCODING`)
  - Private members: Leading underscore (`_internal_state`)

### Architecture Principles

- **Never-Fail Philosophy**: All public APIs must return Result objects, never raise exceptions
- **Streaming First**: Design all components to support incremental processing
- **Type Safety**: Complete type annotations for mypy validation
- **Confidence Scoring**: Every parsing decision must contribute to confidence scoring (0.0-1.0)

## Testing Guidelines

### Test Requirements

- **95% minimum coverage** (100% for error recovery paths)
- **AAA Pattern**: Arrange, Act, Assert structure
- **Descriptive names**: Test functions should describe the scenario
- **Multiple test types**: Unit, integration, and property-based tests

### Test Structure

```
tests/
├── test_ultra_robust_xml_parser/
│   ├── test_init.py                    # Package initialization tests
│   ├── test_character/                 # Character processing tests
│   ├── test_tokenization/              # Tokenization tests
│   ├── test_tree/                      # Tree building tests
│   └── test_api/                       # API layer tests
└── integration/                        # Integration tests
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src/ultra_robust_xml_parser --cov-report=html

# Run specific test file
uv run pytest tests/test_ultra_robust_xml_parser/test_init.py

# Run tests matching pattern
uv run pytest -k "test_encoding"
```

### Writing Tests

```python
def test_encoding_detection_with_bom_returns_correct_encoding() -> None:
    """Test that BOM detection correctly identifies UTF-8 encoding."""
    # Arrange
    utf8_bom_data = b'\xef\xbb\xbf<?xml version="1.0"?><root>test</root>'
    
    # Act
    result = detect_encoding(utf8_bom_data)
    
    # Assert
    assert result.encoding == 'utf-8'
    assert result.confidence >= 0.9
    assert result.method == 'bom_detection'
```

## Documentation

### Docstring Requirements

All public functions must have Google-style docstrings:

```python
def detect_encoding(data: bytes, max_bytes: int = 8192) -> EncodingResult:
    """Detect the encoding of byte data using multi-stage analysis.
    
    Args:
        data: Byte data to analyze for encoding detection
        max_bytes: Maximum number of bytes to analyze for performance
        
    Returns:
        EncodingResult containing detected encoding and confidence score
        
    Raises:
        ValueError: If data is empty or max_bytes is negative
        
    Example:
        >>> data = b'<?xml version="1.0" encoding="UTF-8"?><root>test</root>'
        >>> result = detect_encoding(data)
        >>> result.encoding
        'utf-8'
        >>> result.confidence
        0.95
    """
```

### README Updates

When adding new features, update:
- Feature list in README.md
- Usage examples
- API documentation links

## Pull Request Process

### Before Submitting

1. **Code Quality Checks**
   ```bash
   uv run black src/ tests/
   uv run ruff check src/ tests/
   uv run mypy src/ tests/
   ```

2. **Test Suite**
   ```bash
   uv run pytest --cov=src/ultra_robust_xml_parser
   ```

3. **Build Verification**
   ```bash
   uv build
   ```

### Pull Request Template

- **Description**: Clear description of changes and motivation
- **Type of Change**: Bug fix, new feature, breaking change, documentation
- **Testing**: How the changes have been tested
- **Checklist**: 
  - [ ] Code follows style guidelines
  - [ ] Self-review completed
  - [ ] Tests added/updated
  - [ ] Documentation updated
  - [ ] No breaking changes (or clearly marked)

### Review Process

- All pull requests require review from project maintainers
- CI must pass (all tests, linting, type checking)
- Documentation must be updated for new features
- Breaking changes require special consideration

## Issue Reporting

### Bug Reports

Include:
- **Environment**: Python version, OS, package version
- **Reproduction**: Minimal code example
- **Expected vs Actual**: What you expected vs what happened
- **Error Messages**: Full stack traces if applicable

### Feature Requests

Include:
- **Use Case**: Why this feature would be valuable
- **Proposed Solution**: How you envision it working
- **Alternatives**: Other approaches you considered

### Performance Issues

Include:
- **Benchmark Data**: Performance measurements
- **Test Data**: Representative input that causes issues
- **System Info**: Hardware specs, Python version

## Development Tips

### Performance Considerations

- Use lazy evaluation where possible
- Implement streaming processing for large inputs
- Profile memory usage for optimization opportunities
- Consider Cython extensions for critical paths

### Error Handling

- Never raise unhandled exceptions in public APIs
- Return Result objects with diagnostics
- Implement graceful degradation strategies
- Preserve content over strict compliance

### Testing Malformed XML

- Test with various encoding issues
- Include boundary conditions and edge cases
- Use property-based testing for comprehensive coverage
- Test with real-world malformed XML samples

## Community

- **Discussions**: Use GitHub Discussions for questions and ideas
- **Issues**: Use GitHub Issues for bugs and feature requests
- **Security**: Report security issues privately to maintainers

Thank you for contributing to Ultra-Robust XML Parser!