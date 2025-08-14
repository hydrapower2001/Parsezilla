"""Result objects and diagnostic types for ultra-robust XML parsing.

This module defines comprehensive result objects that provide rich metadata,
diagnostics, and performance information for all parsing operations.
"""

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional


class DiagnosticSeverity(Enum):
    """Severity levels for diagnostic entries."""
    
    DEBUG = auto()      # Debug-level information
    INFO = auto()       # Informational messages
    WARNING = auto()    # Warnings about potential issues
    ERROR = auto()      # Error conditions that were recovered
    CRITICAL = auto()   # Critical errors that impact quality


@dataclass
class DiagnosticEntry:
    """Single diagnostic entry with context information."""
    
    severity: DiagnosticSeverity
    message: str
    component: str
    position: Optional[Dict[str, int]] = None
    details: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)
    correlation_id: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Validate diagnostic entry."""
        if not self.message:
            raise ValueError("Diagnostic message cannot be empty")
        if not self.component:
            raise ValueError("Diagnostic component cannot be empty")


@dataclass
class PerformanceMetrics:
    """Performance metrics for parsing operations."""
    
    processing_time_ms: float = 0.0
    memory_used_bytes: int = 0
    characters_processed: int = 0
    tokens_generated: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    recovery_operations: int = 0
    assembly_operations: int = 0
    
    @property
    def characters_per_second(self) -> float:
        """Calculate characters processed per second."""
        if self.processing_time_ms <= 0:
            return 0.0
        return (self.characters_processed * 1000.0) / self.processing_time_ms
    
    @property
    def tokens_per_second(self) -> float:
        """Calculate tokens generated per second."""
        if self.processing_time_ms <= 0:
            return 0.0
        return (self.tokens_generated * 1000.0) / self.processing_time_ms
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_accesses = self.cache_hits + self.cache_misses
        if total_accesses == 0:
            return 0.0
        return self.cache_hits / total_accesses
    
    @property
    def memory_per_character(self) -> float:
        """Calculate memory usage per character."""
        if self.characters_processed == 0:
            return 0.0
        return self.memory_used_bytes / self.characters_processed


@dataclass
class TokenizationMetadata:
    """Comprehensive metadata for tokenization operations."""
    
    total_tokens: int = 0
    error_tokens: int = 0
    repaired_tokens: int = 0
    recovered_tokens: int = 0
    synthetic_tokens: int = 0
    fast_path_used: bool = False
    recovery_strategies_used: List[str] = field(default_factory=list)
    assembly_repairs_applied: int = 0
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    token_type_distribution: Dict[str, int] = field(default_factory=dict)
    
    @property
    def error_rate(self) -> float:
        """Calculate error token rate."""
        if self.total_tokens == 0:
            return 0.0
        return self.error_tokens / self.total_tokens
    
    @property
    def repair_rate(self) -> float:
        """Calculate token repair rate."""
        if self.total_tokens == 0:
            return 0.0
        return self.repaired_tokens / self.total_tokens
    
    @property
    def recovery_rate(self) -> float:
        """Calculate recovery operation rate."""
        if self.total_tokens == 0:
            return 0.0
        return self.recovered_tokens / self.total_tokens
    
    @property
    def synthetic_rate(self) -> float:
        """Calculate synthetic token rate."""
        if self.total_tokens == 0:
            return 0.0
        return self.synthetic_tokens / self.total_tokens
    
    @property
    def overall_confidence(self) -> float:
        """Calculate overall confidence score."""
        if not self.confidence_scores:
            return 1.0
        return min(self.confidence_scores.values())
    
    def add_token_type(self, token_type: str) -> None:
        """Add a token type to the distribution."""
        self.token_type_distribution[token_type] = (
            self.token_type_distribution.get(token_type, 0) + 1
        )
    
    def add_confidence_score(self, component: str, score: float) -> None:
        """Add a confidence score for a component."""
        if not (0.0 <= score <= 1.0):
            raise ValueError("Confidence score must be between 0.0 and 1.0")
        self.confidence_scores[component] = score