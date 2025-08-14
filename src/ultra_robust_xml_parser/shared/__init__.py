"""Shared utilities for ultra-robust XML parsing.

This module provides shared data structures, configuration objects, result types,
and utility functions used across all processing layers.
"""

from .result import (
    DiagnosticEntry,
    DiagnosticSeverity,
    PerformanceMetrics,
    TokenizationMetadata,
)
from .config import (
    AssemblyConfig,
    FilterConfig,
    PerformanceConfig,
    RecoveryConfig,
    StreamingConfig,
    TokenizationConfig,
)
from .logging import (
    CorrelationLogger,
    get_logger,
)

__all__ = [
    "DiagnosticEntry",
    "DiagnosticSeverity",
    "PerformanceMetrics",
    "TokenizationMetadata",
    "AssemblyConfig",
    "FilterConfig",
    "PerformanceConfig",
    "RecoveryConfig",
    "StreamingConfig",
    "TokenizationConfig",
    "CorrelationLogger",
    "get_logger",
]