"""Tokenization engine for ultra-robust XML parsing.

This module provides a fault-tolerant tokenization engine that converts character
streams into meaningful XML tokens using a robust state machine with comprehensive
error recovery capabilities.

Key Components:
    XMLTokenizer: Main tokenization class for processing character streams
    Token: Represents individual XML tokens with position and confidence information
    TokenType: Enumeration of all supported XML token types
    TokenPosition: Position tracking for debugging and error reporting
    TokenizerState: State machine states for tokenization processing
"""

from .assembly import (
    AssemblyRepairAction,
    AssemblyResult,
    RepairSeverity,
    RepairType,
    TokenAssemblyEngine,
)
from .recovery import (
    ErrorRecoveryEngine,
    RecoveryAction,
    RecoveryContext,
    RecoveryHistory,
    RecoveryHistoryEntry,
    RecoveryStatistics,
    RecoveryStrategy,
)
from .api import (
    EnhancedXMLTokenizer,
    StreamingTokenizer,
    TokenFilter,
    TokenizationResult as EnhancedTokenizationResult,
)
from .tokenizer import (
    Token,
    TokenizationResult,
    TokenizerState,
    TokenPosition,
    TokenRepair,
    TokenType,
    XMLTokenizer,
)

__all__ = [
    "AssemblyRepairAction",
    "AssemblyResult",
    "EnhancedTokenizationResult",
    "EnhancedXMLTokenizer",
    "ErrorRecoveryEngine",
    "RecoveryAction",
    "RecoveryContext",
    "RecoveryHistory",
    "RecoveryHistoryEntry",
    "RecoveryStatistics",
    "RecoveryStrategy",
    "RepairSeverity",
    "RepairType",
    "StreamingTokenizer",
    "Token",
    "TokenAssemblyEngine",
    "TokenFilter",
    "TokenPosition",
    "TokenRepair",
    "TokenType",
    "TokenizationResult",
    "TokenizerState",
    "XMLTokenizer",
]
