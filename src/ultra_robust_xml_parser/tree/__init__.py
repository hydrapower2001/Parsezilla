"""Tree building engine for ultra-robust XML parsing.

This module provides a tree building engine that constructs coherent XML document
trees from token streams through structure repair, tag balancing, and content
organization.

Key Components:
    XMLTreeBuilder: Main tree construction class for processing token streams
    XMLDocument: Root document container with metadata and navigation
    XMLElement: Individual XML element with attributes, text, and children
    ParseResult: Comprehensive result object with tree, metadata, and diagnostics
"""

from .builder import (
    ParseResult,
    StructureRepair,
    XMLDocument,
    XMLElement,
    XMLTreeBuilder,
)

__all__ = [
    "ParseResult",
    "StructureRepair",
    "XMLDocument",
    "XMLElement",
    "XMLTreeBuilder",
]
