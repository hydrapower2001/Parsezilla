"""Character processing layer for ultra robust XML parser.

This module provides encoding detection, character transformation, and stream processing
functionality following the never-fail philosophy.
"""

from .transformation import (
    CharacterTransformer,
    TransformConfig,
    TransformResult,
    TransformationStrategy,
    TransformationContext,
    XML10Validator,
)
from .stream import (
    CharacterStreamProcessor,
    CharacterStreamResult,
    StreamProcessingConfig,
    StreamProcessingPresets,
    StreamingResult,
    StreamingProgress,
    ProgressCallback,
)

__all__ = [
    # Modules
    "encoding", 
    "transformation",
    "stream",
    # Main classes for direct access
    "CharacterTransformer",
    "TransformConfig", 
    "TransformResult",
    "TransformationStrategy",
    "TransformationContext",
    "XML10Validator",
    # Stream processing classes
    "CharacterStreamProcessor",
    "CharacterStreamResult",
    "StreamProcessingConfig",
    "StreamProcessingPresets",
    "StreamingResult",
    "StreamingProgress",
    "ProgressCallback",
]
