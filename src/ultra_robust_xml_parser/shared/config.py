"""Configuration classes for ultra-robust XML parsing.

This module provides configuration objects for all parsing components,
enabling fine-tuned control over parsing behavior and performance.
"""

import json
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple


class RecoveryStrategy(Enum):
    """Error recovery strategy options."""

    CONSERVATIVE = auto()   # Minimal recovery, prefer original content
    BALANCED = auto()      # Balanced recovery with reasonable repairs
    AGGRESSIVE = auto()    # Maximum recovery with extensive repairs
    CUSTOM = auto()        # Custom recovery configuration


class FilterMode(Enum):
    """Token filtering mode options."""

    INCLUDE = auto()       # Include only specified tokens
    EXCLUDE = auto()       # Exclude specified tokens
    TRANSFORM = auto()     # Transform specified tokens


@dataclass
class RecoveryConfig:
    """Configuration for error recovery operations."""

    strategy: RecoveryStrategy = RecoveryStrategy.BALANCED
    max_recovery_attempts: int = 10
    recovery_timeout_ms: float = 1000.0
    enable_context_analysis: bool = True
    enable_structural_repair: bool = True
    enable_content_preservation: bool = True
    confidence_threshold: float = 0.3
    severity_escalation_threshold: int = 5
    custom_recovery_rules: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate recovery configuration."""
        if self.max_recovery_attempts < 0:
            raise ValueError("max_recovery_attempts must be >= 0")
        if self.recovery_timeout_ms < 0:
            raise ValueError("recovery_timeout_ms must be >= 0")
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")


@dataclass
class AssemblyConfig:
    """Configuration for token assembly and repair operations."""

    enable_smart_assembly: bool = True
    enable_context_repair: bool = True
    enable_structure_validation: bool = True
    max_assembly_depth: int = 100
    assembly_timeout_ms: float = 500.0
    repair_confidence_threshold: float = 0.5
    enable_caching: bool = True
    cache_size_limit: int = 1000
    strict_mode: bool = False

    def __post_init__(self) -> None:
        """Validate assembly configuration."""
        if self.max_assembly_depth < 0:
            raise ValueError("max_assembly_depth must be >= 0")
        if self.assembly_timeout_ms < 0:
            raise ValueError("assembly_timeout_ms must be >= 0")
        if not (0.0 <= self.repair_confidence_threshold <= 1.0):
            raise ValueError("repair_confidence_threshold must be between 0.0 and 1.0")
        if self.cache_size_limit < 0:
            raise ValueError("cache_size_limit must be >= 0")


@dataclass
class FilterConfig:
    """Configuration for token filtering and selection."""

    mode: FilterMode = FilterMode.INCLUDE
    token_types: Set[str] = field(default_factory=set)
    position_ranges: List[Tuple[int, int]] = field(default_factory=list)
    content_patterns: List[str] = field(default_factory=list)
    confidence_threshold: float = 0.0
    enable_pattern_matching: bool = False
    case_sensitive: bool = True
    max_results: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate filter configuration."""
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        if self.max_results is not None and self.max_results < 0:
            raise ValueError("max_results must be >= 0 or None")


@dataclass
class StreamingConfig:
    """Configuration for streaming tokenization operations."""

    buffer_size: int = 8192
    chunk_size: int = 1024
    enable_progress_tracking: bool = True
    progress_callback_interval: int = 1000
    enable_cancellation: bool = True
    memory_limit_bytes: Optional[int] = None
    enable_backpressure: bool = True
    backpressure_threshold: int = 10000

    def __post_init__(self) -> None:
        """Validate streaming configuration."""
        if self.buffer_size <= 0:
            raise ValueError("buffer_size must be > 0")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if self.progress_callback_interval <= 0:
            raise ValueError("progress_callback_interval must be > 0")
        if self.memory_limit_bytes is not None and self.memory_limit_bytes <= 0:
            raise ValueError("memory_limit_bytes must be > 0 or None")
        if self.backpressure_threshold <= 0:
            raise ValueError("backpressure_threshold must be > 0")


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""

    enable_fast_path: bool = True
    fast_path_threshold: float = 0.7
    fast_path_sample_size: int = 1000
    enable_caching: bool = True
    cache_size_limit: int = 10000
    enable_lazy_evaluation: bool = True
    enable_parallel_processing: bool = False
    max_worker_threads: int = 4

    def __post_init__(self) -> None:
        """Validate performance configuration."""
        if not (0.0 <= self.fast_path_threshold <= 1.0):
            raise ValueError("fast_path_threshold must be between 0.0 and 1.0")
        if self.fast_path_sample_size <= 0:
            raise ValueError("fast_path_sample_size must be > 0")
        if self.cache_size_limit < 0:
            raise ValueError("cache_size_limit must be >= 0")
        if self.max_worker_threads <= 0:
            raise ValueError("max_worker_threads must be > 0")


@dataclass
class TokenizationConfig:
    """Comprehensive configuration for tokenization operations."""

    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    assembly: AssemblyConfig = field(default_factory=AssemblyConfig)
    filtering: FilterConfig = field(default_factory=FilterConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)

    correlation_id: Optional[str] = None
    enable_diagnostics: bool = True
    diagnostic_level: str = "INFO"
    enable_metrics: bool = True

    @classmethod
    def conservative(cls) -> "TokenizationConfig":
        """Create configuration optimized for accuracy and minimal changes."""
        config = cls()
        config.recovery.strategy = RecoveryStrategy.CONSERVATIVE
        config.assembly.strict_mode = True
        config.assembly.repair_confidence_threshold = 0.8
        config.performance.enable_fast_path = False
        return config

    @classmethod
    def balanced(cls) -> "TokenizationConfig":
        """Create balanced configuration for good performance and accuracy."""
        return cls()  # Default configuration is balanced

    @classmethod
    def aggressive(cls) -> "TokenizationConfig":
        """Create configuration optimized for maximum recovery and repair."""
        config = cls()
        config.recovery.strategy = RecoveryStrategy.AGGRESSIVE
        config.recovery.max_recovery_attempts = 20
        config.assembly.enable_smart_assembly = True
        config.assembly.enable_context_repair = True
        config.performance.enable_fast_path = True
        return config

    @classmethod
    def performance_optimized(cls) -> "TokenizationConfig":
        """Create configuration optimized for maximum performance."""
        config = cls()
        config.performance.enable_fast_path = True
        config.performance.enable_caching = True
        config.performance.enable_parallel_processing = True
        config.assembly.enable_caching = True
        config.streaming.enable_backpressure = True
        return config

    def validate(self) -> None:
        """Validate the complete configuration."""
        # Configurations validate themselves in __post_init__
        # Additional cross-component validation can be added here


@dataclass
class CharacterConfig:
    """Configuration for character processing layer."""

    # Encoding detection settings
    enable_encoding_detection: bool = True
    encoding_detection_sample_size: int = 8192
    encoding_confidence_threshold: float = 0.7
    fallback_encoding: str = "utf-8"
    detect_bom: bool = True

    # Character transformation settings
    enable_transformation: bool = True
    normalize_whitespace: bool = True
    preserve_entity_references: bool = True
    handle_control_characters: bool = True
    transform_line_endings: bool = True

    # Stream processing settings
    buffer_size: int = 8192
    chunk_size: int = 1024
    enable_streaming: bool = True
    memory_limit_bytes: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate character configuration."""
        if self.encoding_detection_sample_size <= 0:
            raise ValueError("encoding_detection_sample_size must be > 0")
        if not (0.0 <= self.encoding_confidence_threshold <= 1.0):
            raise ValueError(
                "encoding_confidence_threshold must be between 0.0 and 1.0"
            )
        if self.buffer_size <= 0:
            raise ValueError("buffer_size must be > 0")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if self.memory_limit_bytes is not None and self.memory_limit_bytes <= 0:
            raise ValueError("memory_limit_bytes must be > 0 or None")


@dataclass
class TreeConfig:
    """Configuration for tree building and structure repair."""

    # Tree building settings
    enable_structure_repair: bool = True
    max_repair_depth: int = 50
    repair_confidence_threshold: float = 0.6
    enable_content_organization: bool = True
    preserve_original_structure: bool = True

    # Validation settings
    enable_validation: bool = True
    validation_strictness: str = "balanced"  # strict, balanced, lenient
    enable_namespace_processing: bool = True
    validate_wellformedness: bool = True
    validate_references: bool = True

    # Memory and performance settings
    max_tree_depth: int = 1000
    max_elements_per_level: int = 10000
    enable_lazy_loading: bool = True
    enable_tree_caching: bool = True
    cache_size_limit: int = 1000

    def __post_init__(self) -> None:
        """Validate tree configuration."""
        if self.max_repair_depth < 0:
            raise ValueError("max_repair_depth must be >= 0")
        if not (0.0 <= self.repair_confidence_threshold <= 1.0):
            raise ValueError(
                "repair_confidence_threshold must be between 0.0 and 1.0"
            )
        if self.validation_strictness not in ["strict", "balanced", "lenient"]:
            raise ValueError(
                "validation_strictness must be 'strict', 'balanced', or 'lenient'"
            )
        if self.max_tree_depth <= 0:
            raise ValueError("max_tree_depth must be > 0")
        if self.max_elements_per_level <= 0:
            raise ValueError("max_elements_per_level must be > 0")
        if self.cache_size_limit < 0:
            raise ValueError("cache_size_limit must be >= 0")


@dataclass
class ApiConfig:
    """Configuration for API layer behavior."""

    # Output format settings
    default_output_format: str = "dict"  # dict, xml, json, text
    enable_pretty_printing: bool = True
    include_diagnostic_info: bool = True
    include_confidence_scores: bool = True
    include_timing_info: bool = False

    # Error handling settings
    never_fail_mode: bool = True
    return_partial_results: bool = True
    include_error_details: bool = True
    escalate_critical_errors: bool = False

    # Performance settings
    enable_result_caching: bool = True
    cache_timeout_seconds: float = 300.0
    enable_async_processing: bool = False
    max_concurrent_operations: int = 4

    def __post_init__(self) -> None:
        """Validate API configuration."""
        valid_formats = ["dict", "xml", "json", "text"]
        if self.default_output_format not in valid_formats:
            raise ValueError(f"default_output_format must be one of {valid_formats}")
        if self.cache_timeout_seconds < 0:
            raise ValueError("cache_timeout_seconds must be >= 0")
        if self.max_concurrent_operations <= 0:
            raise ValueError("max_concurrent_operations must be > 0")


@dataclass
class GlobalConfig:
    """Global configuration settings that apply across all components."""

    # Logging and diagnostics
    logging_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    enable_correlation_tracking: bool = True
    enable_performance_profiling: bool = False
    diagnostic_detail_level: str = "standard"  # minimal, standard, detailed, comp.

    # Security and validation
    enable_input_sanitization: bool = True
    max_input_size_bytes: Optional[int] = None
    enable_security_scanning: bool = False
    allow_external_entities: bool = False

    # Global performance settings
    enable_global_caching: bool = True
    global_cache_size_limit: int = 15000  # Increased to accommodate default cache sizes
    enable_memory_optimization: bool = True
    garbage_collection_threshold: int = 1000

    def __post_init__(self) -> None:
        """Validate global configuration."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.logging_level not in valid_levels:
            raise ValueError(f"logging_level must be one of {valid_levels}")

        valid_detail_levels = ["minimal", "standard", "detailed", "comprehensive"]
        if self.diagnostic_detail_level not in valid_detail_levels:
            raise ValueError(
                f"diagnostic_detail_level must be one of {valid_detail_levels}"
            )

        if self.max_input_size_bytes is not None and self.max_input_size_bytes <= 0:
            raise ValueError("max_input_size_bytes must be > 0 or None")
        if self.global_cache_size_limit < 0:
            raise ValueError("global_cache_size_limit must be >= 0")
        if self.garbage_collection_threshold <= 0:
            raise ValueError("garbage_collection_threshold must be > 0")


class ConfigError(Exception):
    """Base exception for configuration errors."""


class ConfigValidationError(ConfigError):
    """Exception raised when configuration validation fails."""

    def __init__(self, message: str, field_name: Optional[str] = None,
                 suggestions: Optional[List[str]] = None):
        super().__init__(message)
        self.field_name = field_name
        self.suggestions = suggestions or []


@dataclass(frozen=True)
class ParserConfig:
    """Comprehensive configuration for all XML parser components.

    This class provides immutable configuration objects that control the behavior
    of all parser layers: character processing, tokenization, tree building, and API.
    Thread-safe due to frozen dataclass implementation.
    """

    # Component configurations
    character: CharacterConfig = field(default_factory=CharacterConfig)
    tokenization: TokenizationConfig = field(default_factory=TokenizationConfig)
    tree: TreeConfig = field(default_factory=TreeConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    global_: GlobalConfig = field(default_factory=GlobalConfig)

    # Metadata
    version: str = "1.0.0"
    name: Optional[str] = None
    description: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate the complete parser configuration."""
        try:
            # Validate all component configurations
            self.character.__post_init__()
            self.tokenization.validate()
            self.tree.__post_init__()
            self.api.__post_init__()
            self.global_.__post_init__()

            # Cross-component validation
            self._validate_cross_component_dependencies()

        except ValueError as e:
            raise ConfigValidationError(str(e)) from e

    def _validate_cross_component_dependencies(self) -> None:
        """Validate dependencies between different component configurations."""
        # Ensure memory limits are consistent
        if (
            self.character.memory_limit_bytes is not None
            and self.global_.max_input_size_bytes is not None
            and self.character.memory_limit_bytes > self.global_.max_input_size_bytes
        ):
                raise ConfigValidationError(
                    "Character processing memory limit exceeds global input size limit",
                    suggestions=["Reduce character.memory_limit_bytes",
                               "Increase global_.max_input_size_bytes"]
                )

        # Ensure cache sizes don't exceed global limits
        total_cache_size = (
            self.tokenization.performance.cache_size_limit +
            self.tokenization.assembly.cache_size_limit +
            self.tree.cache_size_limit
        )
        if total_cache_size > self.global_.global_cache_size_limit:
            raise ConfigValidationError(
                f"Combined component cache sizes ({total_cache_size}) "
                f"exceed global limit ({self.global_.global_cache_size_limit})",
                suggestions=[
                    "Reduce individual cache limits",
                    "Increase global cache limit",
                ],
            )

    def override(self, **kwargs: Any) -> "ParserConfig":
        """Create a new configuration with specific overrides.

        Args:
            **kwargs: Keyword arguments for configuration fields to override

        Returns:
            New ParserConfig instance with overrides applied

        Example:
            >>> config = ParserConfig()
            >>> new_config = config.override(
            ...     character__buffer_size=16384,
            ...     api__never_fail_mode=False
            ... )
        """
        # Convert nested field notation (e.g., "character__buffer_size") to nested dict
        nested_overrides: Dict[str, Any] = {}
        for key, value in kwargs.items():
            if "__" in key:
                component, field_name = key.split("__", 1)
                # Handle global_ field name mapping
                if component == "global_":
                    component = "global_"
                if component not in nested_overrides:
                    nested_overrides[component] = {}
                nested_overrides[component][field_name] = value
            else:
                nested_overrides[key] = value

        # Create new component configurations with overrides
        new_fields = {}

        for field_name in ["character", "tokenization", "tree", "api", "global_"]:
            current_config = getattr(self, field_name)
            if field_name in nested_overrides:
                # Create new instance with overrides
                new_config = replace(current_config, **nested_overrides[field_name])
                new_fields[field_name] = new_config
            else:
                new_fields[field_name] = current_config

        # Handle top-level overrides (non-component fields)
        for key, value in nested_overrides.items():
            if key not in ["character", "tokenization", "tree", "api", "global_"]:
                new_fields[key] = value

        return replace(self, **new_fields)

    def validate_compatibility(self, other: "ParserConfig") -> List[str]:
        """Check compatibility with another configuration.

        Args:
            other: Configuration to compare against

        Returns:
            List of compatibility warnings/issues
        """
        warnings = []

        # Version compatibility
        if self.version != other.version:
            warnings.append(f"Version mismatch: {self.version} vs {other.version}")

        # Performance implications
        if (
            self.tokenization.performance.enable_fast_path
            != other.tokenization.performance.enable_fast_path
        ):
            warnings.append(
                "Fast path settings differ - may impact performance consistency"
            )

        if (self.global_.enable_global_caching != other.global_.enable_global_caching):
            warnings.append("Global caching settings differ - may impact memory usage")

        return warnings

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format.

        Returns:
            Dictionary representation of the configuration
        """
        def _dataclass_to_dict(obj: Any) -> Any:
            """Recursively convert dataclass to dict."""
            if hasattr(obj, "__dataclass_fields__"):
                result: Dict[str, Any] = {}
                for field in obj.__dataclass_fields__:
                    result[field] = _dataclass_to_dict(getattr(obj, field))
                return result
            if isinstance(obj, Enum):
                return obj.name
            if isinstance(obj, (list, tuple, set)):
                return [_dataclass_to_dict(item) for item in obj]
            if isinstance(obj, dict):
                return {key: _dataclass_to_dict(value) for key, value in obj.items()}
            return obj

        result = _dataclass_to_dict(self)
        # Ensure we return a Dict[str, Any]
        if not isinstance(result, dict):
            raise ConfigValidationError("Configuration serialization failed")
        return result

    def to_json(self, indent: int = 2) -> str:
        """Convert configuration to JSON string.

        Args:
            indent: JSON indentation level

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParserConfig":
        """Create configuration from dictionary.

        Args:
            data: Dictionary containing configuration data

        Returns:
            ParserConfig instance created from dictionary
        """
        # This is a simplified implementation - full implementation would
        # handle nested dataclass reconstruction with proper type conversion

        def _dict_to_dataclass(data_dict: Dict[str, Any], target_class: type) -> Any:
            """Convert dict to dataclass instance."""
            if not hasattr(target_class, "__dataclass_fields__"):
                return data_dict

            field_values: Dict[str, Any] = {}
            for field_name, field_info in target_class.__dataclass_fields__.items():
                if field_name in data_dict:
                    value = data_dict[field_name]
                    field_type = field_info.type

                    # Handle nested dataclasses
                    if hasattr(field_type, "__dataclass_fields__"):
                        field_values[field_name] = _dict_to_dataclass(value, field_type)
                    # Handle enums
                    elif hasattr(field_type, "__members__"):
                        if isinstance(value, str):
                            field_values[field_name] = field_type[value]
                        else:
                            field_values[field_name] = value
                    else:
                        field_values[field_name] = value

            return target_class(**field_values)

        result = _dict_to_dataclass(data, cls)
        # Ensure we return the correct type
        if not isinstance(result, cls):
            raise ConfigValidationError(f"Failed to deserialize to {cls.__name__}")
        return result

    @classmethod
    def from_json(cls, json_str: str) -> "ParserConfig":
        """Create configuration from JSON string.

        Args:
            json_str: JSON string containing configuration data

        Returns:
            ParserConfig instance created from JSON
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    # Preset factory methods
    @classmethod
    def maximum_robustness(cls) -> "ParserConfig":
        """Create configuration preset for maximum fault tolerance."""
        return cls(
            character=CharacterConfig(
                enable_encoding_detection=True,
                encoding_confidence_threshold=0.5,
                enable_transformation=True,
                handle_control_characters=True
            ),
            tokenization=TokenizationConfig.aggressive(),
            tree=TreeConfig(
                enable_structure_repair=True,
                max_repair_depth=100,
                repair_confidence_threshold=0.3,
                validation_strictness="lenient"
            ),
            api=ApiConfig(
                never_fail_mode=True,
                return_partial_results=True,
                include_error_details=True
            ),
            global_=GlobalConfig(
                diagnostic_detail_level="comprehensive",
                enable_input_sanitization=True
            ),
            name="maximum_robustness",
            description=(
                "Configuration optimized for maximum fault tolerance and error recovery"
            )
        )

    @classmethod
    def web_scraping(cls) -> "ParserConfig":
        """Create configuration preset optimized for web scraping and HTML-like XML."""
        return cls(
            character=CharacterConfig(
                normalize_whitespace=True,
                handle_control_characters=True,
                transform_line_endings=True
            ),
            tokenization=TokenizationConfig.balanced(),
            tree=TreeConfig(
                enable_structure_repair=True,
                validation_strictness="lenient",
                enable_namespace_processing=False
            ),
            api=ApiConfig(
                never_fail_mode=True,
                return_partial_results=True,
                default_output_format="dict"
            ),
            global_=GlobalConfig(
                allow_external_entities=False,
                enable_input_sanitization=True
            ),
            name="web_scraping",
            description=(
                "Configuration optimized for parsing HTML-like XML from web sources"
            )
        )

    @classmethod
    def data_recovery(cls) -> "ParserConfig":
        """Create preset for extracting content from severely damaged XML."""
        return cls(
            character=CharacterConfig(
                enable_transformation=True,
                preserve_entity_references=False,
                handle_control_characters=True
            ),
            tokenization=TokenizationConfig.aggressive(),
            tree=TreeConfig(
                enable_structure_repair=True,
                max_repair_depth=200,
                repair_confidence_threshold=0.1,
                preserve_original_structure=False,
                validation_strictness="lenient"
            ),
            api=ApiConfig(
                never_fail_mode=True,
                return_partial_results=True,
                include_diagnostic_info=True
            ),
            global_=GlobalConfig(
                diagnostic_detail_level="comprehensive",
                enable_memory_optimization=True
            ),
            name="data_recovery",
            description=(
                "Configuration optimized for maximum content extraction from "
                "damaged XML"
            )
        )

    @classmethod
    def performance_optimized(cls) -> "ParserConfig":
        """Create configuration preset for high-speed processing of well-formed XML."""
        return cls(
            character=CharacterConfig(
                buffer_size=16384,
                chunk_size=4096,
                enable_streaming=True
            ),
            tokenization=TokenizationConfig.performance_optimized(),
            tree=TreeConfig(
                enable_structure_repair=False,
                validation_strictness="strict",
                enable_lazy_loading=True,
                enable_tree_caching=True
            ),
            api=ApiConfig(
                include_diagnostic_info=False,
                include_confidence_scores=False,
                enable_result_caching=True,
                enable_async_processing=True
            ),
            global_=GlobalConfig(
                diagnostic_detail_level="minimal",
                enable_global_caching=True,
                enable_memory_optimization=True
            ),
            name="performance_optimized",
            description=(
                "Configuration optimized for high-speed processing of well-formed XML"
            )
        )
