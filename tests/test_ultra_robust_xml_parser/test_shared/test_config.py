"""Comprehensive tests for configuration system."""

import json
import pytest
from dataclasses import replace
from typing import Dict, Any

from ultra_robust_xml_parser.shared.config import (
    # Configuration classes
    ParserConfig,
    CharacterConfig,
    TokenizationConfig,
    TreeConfig,
    ApiConfig,
    GlobalConfig,
    
    # Component configs
    RecoveryConfig,
    AssemblyConfig,
    FilterConfig,
    StreamingConfig,
    PerformanceConfig,
    
    # Enums
    RecoveryStrategy,
    FilterMode,
    
    # Exceptions
    ConfigError,
    ConfigValidationError
)


class TestCharacterConfig:
    """Test suite for CharacterConfig."""
    
    def test_default_configuration(self):
        """Test default character configuration values."""
        config = CharacterConfig()
        
        assert config.enable_encoding_detection is True
        assert config.encoding_detection_sample_size == 8192
        assert config.encoding_confidence_threshold == 0.7
        assert config.fallback_encoding == "utf-8"
        assert config.detect_bom is True
        
        assert config.enable_transformation is True
        assert config.normalize_whitespace is True
        assert config.preserve_entity_references is True
        assert config.handle_control_characters is True
        assert config.transform_line_endings is True
        
        assert config.buffer_size == 8192
        assert config.chunk_size == 1024
        assert config.enable_streaming is True
        assert config.memory_limit_bytes is None
    
    def test_character_config_validation_success(self):
        """Test successful character configuration validation."""
        config = CharacterConfig(
            encoding_detection_sample_size=4096,
            encoding_confidence_threshold=0.8,
            buffer_size=16384,
            chunk_size=2048,
            memory_limit_bytes=1024*1024
        )
        # Should not raise any exception
        config.__post_init__()
    
    def test_character_config_validation_failures(self):
        """Test character configuration validation failures."""
        # Test invalid encoding_detection_sample_size
        with pytest.raises(ValueError, match="encoding_detection_sample_size must be > 0"):
            CharacterConfig(encoding_detection_sample_size=0)
        
        with pytest.raises(ValueError, match="encoding_detection_sample_size must be > 0"):
            CharacterConfig(encoding_detection_sample_size=-1)
        
        # Test invalid encoding_confidence_threshold
        with pytest.raises(ValueError, match="encoding_confidence_threshold must be between 0.0 and 1.0"):
            CharacterConfig(encoding_confidence_threshold=-0.1)
        
        with pytest.raises(ValueError, match="encoding_confidence_threshold must be between 0.0 and 1.0"):
            CharacterConfig(encoding_confidence_threshold=1.1)
        
        # Test invalid buffer_size
        with pytest.raises(ValueError, match="buffer_size must be > 0"):
            CharacterConfig(buffer_size=0)
        
        # Test invalid chunk_size
        with pytest.raises(ValueError, match="chunk_size must be > 0"):
            CharacterConfig(chunk_size=0)
        
        # Test invalid memory_limit_bytes
        with pytest.raises(ValueError, match="memory_limit_bytes must be > 0 or None"):
            CharacterConfig(memory_limit_bytes=0)


class TestTreeConfig:
    """Test suite for TreeConfig."""
    
    def test_default_configuration(self):
        """Test default tree configuration values."""
        config = TreeConfig()
        
        assert config.enable_structure_repair is True
        assert config.max_repair_depth == 50
        assert config.repair_confidence_threshold == 0.6
        assert config.enable_content_organization is True
        assert config.preserve_original_structure is True
        
        assert config.enable_validation is True
        assert config.validation_strictness == "balanced"
        assert config.enable_namespace_processing is True
        assert config.validate_wellformedness is True
        assert config.validate_references is True
        
        assert config.max_tree_depth == 1000
        assert config.max_elements_per_level == 10000
        assert config.enable_lazy_loading is True
        assert config.enable_tree_caching is True
        assert config.cache_size_limit == 1000
    
    def test_tree_config_validation_success(self):
        """Test successful tree configuration validation."""
        config = TreeConfig(
            max_repair_depth=100,
            repair_confidence_threshold=0.8,
            validation_strictness="strict",
            max_tree_depth=500,
            max_elements_per_level=5000,
            cache_size_limit=2000
        )
        config.__post_init__()
    
    def test_tree_config_validation_failures(self):
        """Test tree configuration validation failures."""
        # Test invalid max_repair_depth
        with pytest.raises(ValueError, match="max_repair_depth must be >= 0"):
            TreeConfig(max_repair_depth=-1)
        
        # Test invalid repair_confidence_threshold
        with pytest.raises(ValueError, match="repair_confidence_threshold must be between 0.0 and 1.0"):
            TreeConfig(repair_confidence_threshold=1.5)
        
        # Test invalid validation_strictness
        with pytest.raises(ValueError, match="validation_strictness must be 'strict', 'balanced', or 'lenient'"):
            TreeConfig(validation_strictness="invalid")
        
        # Test invalid max_tree_depth
        with pytest.raises(ValueError, match="max_tree_depth must be > 0"):
            TreeConfig(max_tree_depth=0)
        
        # Test invalid max_elements_per_level
        with pytest.raises(ValueError, match="max_elements_per_level must be > 0"):
            TreeConfig(max_elements_per_level=0)
        
        # Test invalid cache_size_limit
        with pytest.raises(ValueError, match="cache_size_limit must be >= 0"):
            TreeConfig(cache_size_limit=-1)


class TestApiConfig:
    """Test suite for ApiConfig."""
    
    def test_default_configuration(self):
        """Test default API configuration values."""
        config = ApiConfig()
        
        assert config.default_output_format == "dict"
        assert config.enable_pretty_printing is True
        assert config.include_diagnostic_info is True
        assert config.include_confidence_scores is True
        assert config.include_timing_info is False
        
        assert config.never_fail_mode is True
        assert config.return_partial_results is True
        assert config.include_error_details is True
        assert config.escalate_critical_errors is False
        
        assert config.enable_result_caching is True
        assert config.cache_timeout_seconds == 300.0
        assert config.enable_async_processing is False
        assert config.max_concurrent_operations == 4
    
    def test_api_config_validation_success(self):
        """Test successful API configuration validation."""
        config = ApiConfig(
            default_output_format="json",
            cache_timeout_seconds=600.0,
            max_concurrent_operations=8
        )
        config.__post_init__()
    
    def test_api_config_validation_failures(self):
        """Test API configuration validation failures."""
        # Test invalid default_output_format
        with pytest.raises(ValueError, match="default_output_format must be one of"):
            ApiConfig(default_output_format="invalid")
        
        # Test invalid cache_timeout_seconds
        with pytest.raises(ValueError, match="cache_timeout_seconds must be >= 0"):
            ApiConfig(cache_timeout_seconds=-1.0)
        
        # Test invalid max_concurrent_operations
        with pytest.raises(ValueError, match="max_concurrent_operations must be > 0"):
            ApiConfig(max_concurrent_operations=0)


class TestGlobalConfig:
    """Test suite for GlobalConfig."""
    
    def test_default_configuration(self):
        """Test default global configuration values."""
        config = GlobalConfig()
        
        assert config.logging_level == "INFO"
        assert config.enable_correlation_tracking is True
        assert config.enable_performance_profiling is False
        assert config.diagnostic_detail_level == "standard"
        
        assert config.enable_input_sanitization is True
        assert config.max_input_size_bytes is None
        assert config.enable_security_scanning is False
        assert config.allow_external_entities is False
        
        assert config.enable_global_caching is True
        assert config.global_cache_size_limit == 15000
        assert config.enable_memory_optimization is True
        assert config.garbage_collection_threshold == 1000
    
    def test_global_config_validation_success(self):
        """Test successful global configuration validation."""
        config = GlobalConfig(
            logging_level="DEBUG",
            diagnostic_detail_level="comprehensive",
            max_input_size_bytes=1024*1024,
            global_cache_size_limit=20000,
            garbage_collection_threshold=2000
        )
        config.__post_init__()
    
    def test_global_config_validation_failures(self):
        """Test global configuration validation failures."""
        # Test invalid logging_level
        with pytest.raises(ValueError, match="logging_level must be one of"):
            GlobalConfig(logging_level="INVALID")
        
        # Test invalid diagnostic_detail_level
        with pytest.raises(ValueError, match="diagnostic_detail_level must be one of"):
            GlobalConfig(diagnostic_detail_level="invalid")
        
        # Test invalid max_input_size_bytes
        with pytest.raises(ValueError, match="max_input_size_bytes must be > 0 or None"):
            GlobalConfig(max_input_size_bytes=0)
        
        # Test invalid global_cache_size_limit
        with pytest.raises(ValueError, match="global_cache_size_limit must be >= 0"):
            GlobalConfig(global_cache_size_limit=-1)
        
        # Test invalid garbage_collection_threshold
        with pytest.raises(ValueError, match="garbage_collection_threshold must be > 0"):
            GlobalConfig(garbage_collection_threshold=0)


class TestParserConfig:
    """Test suite for ParserConfig."""
    
    def test_default_parser_config(self):
        """Test default parser configuration creation."""
        config = ParserConfig()
        
        assert isinstance(config.character, CharacterConfig)
        assert isinstance(config.tokenization, TokenizationConfig)
        assert isinstance(config.tree, TreeConfig)
        assert isinstance(config.api, ApiConfig)
        assert isinstance(config.global_, GlobalConfig)
        
        assert config.version == "1.0.0"
        assert config.name is None
        assert config.description is None
    
    def test_parser_config_immutability(self):
        """Test that ParserConfig is immutable (frozen dataclass)."""
        config = ParserConfig()
        
        # Should not be able to modify fields directly
        with pytest.raises(AttributeError):
            config.version = "2.0.0"
        
        with pytest.raises(AttributeError):
            config.name = "test"
    
    def test_parser_config_cross_component_validation(self):
        """Test cross-component validation in ParserConfig."""
        # Test memory limit consistency
        character_config = CharacterConfig(memory_limit_bytes=1000)
        global_config = GlobalConfig(max_input_size_bytes=500)
        
        with pytest.raises(ConfigValidationError, match="Character processing memory limit exceeds global input size limit"):
            ParserConfig(character=character_config, global_=global_config)
        
        # Test cache size validation
        tokenization_config = TokenizationConfig()
        tokenization_config.performance.cache_size_limit = 5000
        tokenization_config.assembly.cache_size_limit = 4000
        tree_config = TreeConfig(cache_size_limit=2000)
        global_config = GlobalConfig(global_cache_size_limit=10000)  # Total: 11000 > 10000
        
        with pytest.raises(ConfigValidationError, match="Combined component cache sizes .* exceed global limit"):
            ParserConfig(
                tokenization=tokenization_config,
                tree=tree_config,
                global_=global_config
            )
    
    def test_parser_config_override(self):
        """Test configuration override functionality."""
        config = ParserConfig()
        
        # Test nested field override
        new_config = config.override(
            character__buffer_size=16384,
            api__never_fail_mode=False,
            name="test_config"
        )
        
        assert new_config.character.buffer_size == 16384
        assert new_config.api.never_fail_mode is False
        assert new_config.name == "test_config"
        
        # Original config should be unchanged
        assert config.character.buffer_size == 8192
        assert config.api.never_fail_mode is True
        assert config.name is None
    
    def test_parser_config_compatibility_validation(self):
        """Test configuration compatibility checking."""
        config1 = ParserConfig(version="1.0.0")
        config2 = ParserConfig(version="2.0.0")
        
        warnings = config1.validate_compatibility(config2)
        assert len(warnings) > 0
        assert any("Version mismatch" in warning for warning in warnings)
        
        # Test performance setting differences  
        config3 = ParserConfig()
        global_config = replace(config3.global_, enable_global_caching=False)
        config4 = replace(config3, global_=global_config)
        
        warnings = config3.validate_compatibility(config4)
        assert any("Global caching settings differ" in warning for warning in warnings)
    
    def test_parser_config_serialization(self):
        """Test configuration serialization to dict and JSON."""
        config = ParserConfig(
            name="test_config",
            description="Test configuration"
        )
        
        # Test to_dict
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["name"] == "test_config"
        assert config_dict["description"] == "Test configuration"
        assert "character" in config_dict
        assert "tokenization" in config_dict
        
        # Test to_json
        config_json = config.to_json()
        assert isinstance(config_json, str)
        
        # Verify JSON is valid
        parsed_json = json.loads(config_json)
        assert parsed_json["name"] == "test_config"
    
    def test_parser_config_deserialization(self):
        """Test configuration deserialization from dict and JSON."""
        original_config = ParserConfig(
            name="test_config",
            description="Test configuration"
        )
        
        # Test dict round trip
        config_dict = original_config.to_dict()
        restored_config = ParserConfig.from_dict(config_dict)
        
        assert restored_config.name == original_config.name
        assert restored_config.description == original_config.description
        assert restored_config.version == original_config.version
        
        # Test JSON round trip
        config_json = original_config.to_json()
        restored_from_json = ParserConfig.from_json(config_json)
        
        assert restored_from_json.name == original_config.name
        assert restored_from_json.description == original_config.description


class TestPresetConfigurations:
    """Test suite for preset configuration factory methods."""
    
    def test_maximum_robustness_preset(self):
        """Test maximum robustness preset configuration."""
        config = ParserConfig.maximum_robustness()
        
        assert config.name == "maximum_robustness"
        assert "maximum fault tolerance" in config.description.lower()
        
        # Verify robustness settings
        assert config.character.enable_encoding_detection is True
        assert config.character.encoding_confidence_threshold == 0.5
        assert config.character.handle_control_characters is True
        
        assert config.tokenization.recovery.strategy == RecoveryStrategy.AGGRESSIVE
        
        assert config.tree.enable_structure_repair is True
        assert config.tree.max_repair_depth == 100
        assert config.tree.repair_confidence_threshold == 0.3
        assert config.tree.validation_strictness == "lenient"
        
        assert config.api.never_fail_mode is True
        assert config.api.return_partial_results is True
        assert config.api.include_error_details is True
        
        assert config.global_.diagnostic_detail_level == "comprehensive"
    
    def test_web_scraping_preset(self):
        """Test web scraping preset configuration."""
        config = ParserConfig.web_scraping()
        
        assert config.name == "web_scraping"
        assert "web" in config.description.lower() and "xml" in config.description.lower()
        
        # Verify web scraping optimizations
        assert config.character.normalize_whitespace is True
        assert config.character.handle_control_characters is True
        assert config.character.transform_line_endings is True
        
        assert config.tree.enable_structure_repair is True
        assert config.tree.validation_strictness == "lenient"
        assert config.tree.enable_namespace_processing is False
        
        assert config.api.never_fail_mode is True
        assert config.api.return_partial_results is True
        assert config.api.default_output_format == "dict"
        
        assert config.global_.allow_external_entities is False
        assert config.global_.enable_input_sanitization is True
    
    def test_data_recovery_preset(self):
        """Test data recovery preset configuration."""
        config = ParserConfig.data_recovery()
        
        assert config.name == "data_recovery"
        assert "content extraction" in config.description.lower() and "damaged" in config.description.lower()
        
        # Verify data recovery settings
        assert config.character.enable_transformation is True
        assert config.character.preserve_entity_references is False
        assert config.character.handle_control_characters is True
        
        assert config.tokenization.recovery.strategy == RecoveryStrategy.AGGRESSIVE
        
        assert config.tree.enable_structure_repair is True
        assert config.tree.max_repair_depth == 200
        assert config.tree.repair_confidence_threshold == 0.1
        assert config.tree.preserve_original_structure is False
        assert config.tree.validation_strictness == "lenient"
        
        assert config.api.never_fail_mode is True
        assert config.api.return_partial_results is True
        assert config.api.include_diagnostic_info is True
        
        assert config.global_.diagnostic_detail_level == "comprehensive"
        assert config.global_.enable_memory_optimization is True
    
    def test_performance_optimized_preset(self):
        """Test performance optimized preset configuration."""
        config = ParserConfig.performance_optimized()
        
        assert config.name == "performance_optimized"
        assert "performance" in config.description.lower() or "speed" in config.description.lower()
        
        # Verify performance optimizations
        assert config.character.buffer_size == 16384
        assert config.character.chunk_size == 4096
        assert config.character.enable_streaming is True
        
        assert config.tokenization.performance.enable_fast_path is True
        assert config.tokenization.performance.enable_caching is True
        assert config.tokenization.performance.enable_parallel_processing is True
        
        assert config.tree.enable_structure_repair is False
        assert config.tree.validation_strictness == "strict"
        assert config.tree.enable_lazy_loading is True
        assert config.tree.enable_tree_caching is True
        
        assert config.api.include_diagnostic_info is False
        assert config.api.include_confidence_scores is False
        assert config.api.enable_result_caching is True
        assert config.api.enable_async_processing is True
        
        assert config.global_.diagnostic_detail_level == "minimal"
        assert config.global_.enable_global_caching is True
        assert config.global_.enable_memory_optimization is True


class TestConfigValidationError:
    """Test suite for ConfigValidationError."""
    
    def test_config_validation_error_creation(self):
        """Test ConfigValidationError creation and attributes."""
        error = ConfigValidationError(
            "Test error message",
            field_name="test_field",
            suggestions=["suggestion1", "suggestion2"]
        )
        
        assert str(error) == "Test error message"
        assert error.field_name == "test_field"
        assert error.suggestions == ["suggestion1", "suggestion2"]
    
    def test_config_validation_error_without_optional_fields(self):
        """Test ConfigValidationError creation without optional fields."""
        error = ConfigValidationError("Test error message")
        
        assert str(error) == "Test error message"
        assert error.field_name is None
        assert error.suggestions == []


class TestConfigurationIntegration:
    """Integration tests for the complete configuration system."""
    
    def test_complete_configuration_workflow(self):
        """Test complete configuration creation, override, and serialization workflow."""
        # Create base configuration
        base_config = ParserConfig.web_scraping()
        
        # Apply runtime overrides
        runtime_config = base_config.override(
            character__buffer_size=32768,
            tree__max_repair_depth=75,
            api__enable_async_processing=True
        )
        
        # Serialize configuration
        config_json = runtime_config.to_json()
        
        # Deserialize configuration
        restored_config = ParserConfig.from_json(config_json)
        
        # Verify configuration integrity
        assert restored_config.name == "web_scraping"
        assert restored_config.character.buffer_size == 32768
        assert restored_config.tree.max_repair_depth == 75
        assert restored_config.api.enable_async_processing is True
        
        # Verify preset-specific settings are preserved
        assert restored_config.tree.enable_namespace_processing is False
        assert restored_config.global_.allow_external_entities is False
    
    def test_configuration_validation_error_handling(self):
        """Test comprehensive configuration validation and error handling."""
        # Test validation error with suggestions
        with pytest.raises(ConfigValidationError) as exc_info:
            ParserConfig(
                character=CharacterConfig(memory_limit_bytes=2000),
                global_=GlobalConfig(max_input_size_bytes=1000)
            )
        
        error = exc_info.value
        assert "Character processing memory limit exceeds global input size limit" in str(error)
        assert len(error.suggestions) > 0
        assert any("Reduce character.memory_limit_bytes" in suggestion for suggestion in error.suggestions)
    
    def test_thread_safety_verification(self):
        """Test that configurations are thread-safe (immutable)."""
        config = ParserConfig()
        
        # Verify frozen dataclass behavior
        assert hasattr(ParserConfig, '__dataclass_params__')
        assert ParserConfig.__dataclass_params__.frozen is True
        
        # Test that override creates new instance
        new_config = config.override(name="test")
        assert config is not new_config
        assert config.name is None
        assert new_config.name == "test"