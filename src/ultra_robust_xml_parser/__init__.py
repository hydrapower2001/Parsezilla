"""Ultra-Robust XML Parser.

A never-fail XML parser that handles any malformed XML input gracefully with
comprehensive encoding detection, character transformation, and error recovery.

Progressive API Disclosure:
- Level 1: Simple functions - parse(), parse_string(), parse_file()
- Level 2: Configured parser - UltraRobustXMLParser class
- Level 3: Streaming parser - (future enhancement)
- Level 4: Plugin system - (future enhancement)
"""

__version__ = "0.1.0"
__author__ = "Ultra Robust XML Parser Team"

# Progressive API disclosure - Level 1: Simple functions
# Progressive API disclosure - Level 2: Advanced configuration
from .api import UltraRobustXMLParser, parse, parse_file, parse_string
from .character.stream import StreamProcessingConfig

# Configuration classes for advanced usage
from .shared.config import TokenizationConfig

# Core result objects for all API levels
from .tree.builder import ParseResult, XMLDocument, XMLElement

__all__ = [
    # Version and metadata
    "__author__",
    "__version__",

    # Level 1: Simple parsing functions (progressive disclosure entry point)
    "parse",
    "parse_string",
    "parse_file",

    # Level 2: Advanced parser class
    "UltraRobustXMLParser",

    # Result objects and data structures
    "ParseResult",
    "XMLDocument",
    "XMLElement",

    # Configuration classes for advanced usage
    "TokenizationConfig",
    "StreamProcessingConfig",
]
