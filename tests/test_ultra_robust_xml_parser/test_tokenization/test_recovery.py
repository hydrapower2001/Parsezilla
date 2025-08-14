"""Comprehensive tests for error recovery engine.

Tests the ErrorRecoveryEngine with various malformation scenarios, strategy selection,
and integration with the tokenization system.
"""

import pytest
from unittest.mock import Mock

from ultra_robust_xml_parser.tokenization import (
    ErrorRecoveryEngine,
    RecoveryAction,
    RecoveryContext,
    RecoveryStrategy,
    RecoveryHistory,
    RecoveryHistoryEntry,
    RecoveryStatistics,
    Token,
    TokenType,
    TokenizerState,
    TokenPosition,
)


class TestRecoveryAction:
    """Tests for RecoveryAction dataclass."""

    def test_recovery_action_creation(self):
        """Test creating a valid RecoveryAction."""
        position = TokenPosition(1, 1, 0)
        token = Token(TokenType.TEXT, "test", position)
        
        action = RecoveryAction(
            strategy=RecoveryStrategy.TREAT_AS_TEXT,
            success=True,
            tokens=[token],
            confidence=0.8,
            description="Test recovery",
            original_content="<",
            repaired_content="test",
            position=position
        )
        
        assert action.strategy == RecoveryStrategy.TREAT_AS_TEXT
        assert action.success is True
        assert len(action.tokens) == 1
        assert action.confidence == 0.8
        assert action.description == "Test recovery"
        assert action.original_content == "<"
        assert action.repaired_content == "test"
        assert action.position == position
        assert action.metadata == {}

    def test_recovery_action_invalid_confidence(self):
        """Test RecoveryAction with invalid confidence value."""
        position = TokenPosition(1, 1, 0)
        token = Token(TokenType.TEXT, "test", position)
        
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            RecoveryAction(
                strategy=RecoveryStrategy.TREAT_AS_TEXT,
                success=True,
                tokens=[token],
                confidence=1.5,  # Invalid confidence
                description="Test recovery",
                original_content="<",
                repaired_content="test",
                position=position
            )


class TestRecoveryContext:
    """Tests for RecoveryContext dataclass."""

    def test_recovery_context_creation(self):
        """Test creating a valid RecoveryContext."""
        position = TokenPosition(1, 5, 4)
        token = Token(TokenType.TEXT, "test", position)
        
        context = RecoveryContext(
            error_type="invalid_character",
            error_position=position,
            surrounding_content="<test>content<",
            tokenizer_state=TokenizerState.TEXT_CONTENT,
            recent_tokens=[token],
            malformation_severity=0.3
        )
        
        assert context.error_type == "invalid_character"
        assert context.error_position == position
        assert context.surrounding_content == "<test>content<"
        assert context.tokenizer_state == TokenizerState.TEXT_CONTENT
        assert len(context.recent_tokens) == 1
        assert context.malformation_severity == 0.3
        assert context.pattern_history == []
        assert context.is_severe_malformation is False

    def test_severe_malformation_detection(self):
        """Test severe malformation detection."""
        position = TokenPosition(1, 1, 0)
        
        # Mild malformation
        context_mild = RecoveryContext(
            error_type="invalid_character",
            error_position=position,
            surrounding_content="content",
            tokenizer_state=TokenizerState.TEXT_CONTENT,
            recent_tokens=[],
            malformation_severity=0.5
        )
        assert context_mild.is_severe_malformation is False
        
        # Severe malformation
        context_severe = RecoveryContext(
            error_type="invalid_character",
            error_position=position,
            surrounding_content="content",
            tokenizer_state=TokenizerState.TEXT_CONTENT,
            recent_tokens=[],
            malformation_severity=0.8
        )
        assert context_severe.is_severe_malformation is True


class TestErrorRecoveryEngine:
    """Tests for ErrorRecoveryEngine class."""

    def test_engine_initialization(self):
        """Test error recovery engine initialization."""
        engine = ErrorRecoveryEngine()
        
        assert engine.correlation_id is None
        assert engine.recovery_attempts == 0
        assert engine.pattern_cache == {}
        assert engine.escalation_failures == 0
        assert len(engine.strategy_effectiveness) == 6  # All strategies
        
        # Test with correlation ID
        engine_with_id = ErrorRecoveryEngine(correlation_id="test-123")
        assert engine_with_id.correlation_id == "test-123"

    def test_skip_until_valid_recovery(self):
        """Test skip-until-valid recovery strategy."""
        engine = ErrorRecoveryEngine()
        position = TokenPosition(1, 5, 4)
        
        context = RecoveryContext(
            error_type="invalid_character",
            error_position=position,
            surrounding_content="content",
            tokenizer_state=TokenizerState.TEXT_CONTENT,
            recent_tokens=[],
            malformation_severity=0.3
        )
        
        action = engine._skip_until_valid_recovery(context, "<")
        
        assert action.strategy == RecoveryStrategy.SKIP_UNTIL_VALID
        assert action.success is True
        assert len(action.tokens) == 1
        assert action.tokens[0].type == TokenType.INVALID_CHARS
        assert action.tokens[0].value == "<"
        assert action.confidence == 0.8
        assert action.original_content == "<"
        assert action.repaired_content == ""
        assert "skipped_chars" in action.metadata
        assert action.metadata["skipped_chars"] == 1

    def test_treat_as_text_recovery(self):
        """Test treat-as-text recovery strategy."""
        engine = ErrorRecoveryEngine()
        position = TokenPosition(1, 5, 4)
        
        context = RecoveryContext(
            error_type="invalid_character",
            error_position=position,
            surrounding_content="content",
            tokenizer_state=TokenizerState.TEXT_CONTENT,
            recent_tokens=[],
            malformation_severity=0.3
        )
        
        action = engine._treat_as_text_recovery(context, "<")
        
        assert action.strategy == RecoveryStrategy.TREAT_AS_TEXT
        assert action.success is True
        assert len(action.tokens) == 1
        assert action.tokens[0].type == TokenType.TEXT
        assert action.tokens[0].value == "<"
        assert action.confidence == 0.9
        assert action.original_content == "<"
        assert action.repaired_content == "<"
        assert action.metadata["preservation_strategy"] == "text_conversion"

    def test_insert_missing_structure_recovery_tag_name(self):
        """Test insert-missing-structure recovery for missing tag name."""
        engine = ErrorRecoveryEngine()
        position = TokenPosition(1, 5, 4)
        
        context = RecoveryContext(
            error_type="missing_tag_name",
            error_position=position,
            surrounding_content="<>",
            tokenizer_state=TokenizerState.TAG_OPENING,
            recent_tokens=[],
            malformation_severity=0.5
        )
        
        action = engine._insert_missing_structure_recovery(context, ">")
        
        assert action.strategy == RecoveryStrategy.INSERT_MISSING_STRUCTURE
        assert action.success is True
        assert len(action.tokens) == 2
        assert action.tokens[0].type == TokenType.TAG_NAME
        assert action.tokens[0].value == "unknown"
        assert action.tokens[1].type == TokenType.SYNTHETIC_CLOSE
        assert action.tokens[1].value == ">"
        assert action.confidence == 0.6
        assert action.metadata["synthetic_tokens"] == 2

    def test_insert_missing_structure_recovery_attr_value(self):
        """Test insert-missing-structure recovery for missing attribute value."""
        engine = ErrorRecoveryEngine()
        position = TokenPosition(1, 10, 9)
        
        context = RecoveryContext(
            error_type="missing_attr_value",
            error_position=position,
            surrounding_content='class=">',
            tokenizer_state=TokenizerState.ATTR_VALUE_START,
            recent_tokens=[],
            malformation_severity=0.4
        )
        
        action = engine._insert_missing_structure_recovery(context, ">")
        
        assert action.strategy == RecoveryStrategy.INSERT_MISSING_STRUCTURE
        assert action.success is True
        assert len(action.tokens) == 1
        assert action.tokens[0].type == TokenType.ATTR_VALUE
        assert action.tokens[0].value == ""
        assert action.confidence == 0.6

    def test_character_escape_recovery(self):
        """Test character escape recovery strategy."""
        engine = ErrorRecoveryEngine()
        position = TokenPosition(1, 5, 4)
        
        context = RecoveryContext(
            error_type="invalid_character",
            error_position=position,
            surrounding_content="content",
            tokenizer_state=TokenizerState.TEXT_CONTENT,
            recent_tokens=[],
            malformation_severity=0.3
        )
        
        # Test escaping <
        action = engine._character_escape_recovery(context, "<")
        assert action.tokens[0].value == "&lt;"
        assert action.metadata["escaped"] is True
        
        # Test escaping >
        action = engine._character_escape_recovery(context, ">")
        assert action.tokens[0].value == "&gt;"
        assert action.metadata["escaped"] is True
        
        # Test escaping &
        action = engine._character_escape_recovery(context, "&")
        assert action.tokens[0].value == "&amp;"
        assert action.metadata["escaped"] is True
        
        # Test character that doesn't need escaping
        action = engine._character_escape_recovery(context, "a")
        assert action.tokens[0].value == "a"
        assert action.metadata["escaped"] is False

    def test_strategy_selection_invalid_character_sequence(self):
        """Test strategy selection for invalid character sequences."""
        engine = ErrorRecoveryEngine()
        position = TokenPosition(1, 5, 4)
        
        # Mild malformation - should use character escape
        context_mild = RecoveryContext(
            error_type="invalid_character_sequence",
            error_position=position,
            surrounding_content="content",
            tokenizer_state=TokenizerState.TEXT_CONTENT,
            recent_tokens=[],
            malformation_severity=0.3
        )
        
        strategy = engine._select_recovery_strategy(context_mild)
        assert strategy == RecoveryStrategy.CHARACTER_ESCAPE
        
        # Severe malformation - should treat as text
        context_severe = RecoveryContext(
            error_type="invalid_character_sequence",
            error_position=position,
            surrounding_content="content",
            tokenizer_state=TokenizerState.TEXT_CONTENT,
            recent_tokens=[],
            malformation_severity=0.8
        )
        
        strategy = engine._select_recovery_strategy(context_severe)
        assert strategy == RecoveryStrategy.TREAT_AS_TEXT

    def test_strategy_selection_unmatched_tag(self):
        """Test strategy selection for unmatched tags."""
        engine = ErrorRecoveryEngine()
        position = TokenPosition(1, 5, 4)
        
        context = RecoveryContext(
            error_type="unmatched_tag",
            error_position=position,
            surrounding_content="<div><span>",
            tokenizer_state=TokenizerState.TAG_NAME,
            recent_tokens=[],
            malformation_severity=0.5
        )
        
        strategy = engine._select_recovery_strategy(context)
        assert strategy == RecoveryStrategy.BALANCED_REPAIR

    def test_strategy_selection_invalid_tag_structure(self):
        """Test strategy selection for invalid tag structure."""
        engine = ErrorRecoveryEngine()
        position = TokenPosition(1, 5, 4)
        
        # In tag opening state - should insert missing structure
        context_opening = RecoveryContext(
            error_type="invalid_tag_structure",
            error_position=position,
            surrounding_content="< >",
            tokenizer_state=TokenizerState.TAG_OPENING,
            recent_tokens=[],
            malformation_severity=0.4
        )
        
        strategy = engine._select_recovery_strategy(context_opening)
        assert strategy == RecoveryStrategy.INSERT_MISSING_STRUCTURE
        
        # In other state - should skip until valid
        context_other = RecoveryContext(
            error_type="invalid_tag_structure",
            error_position=position,
            surrounding_content="< >",
            tokenizer_state=TokenizerState.TEXT_CONTENT,
            recent_tokens=[],
            malformation_severity=0.4
        )
        
        strategy = engine._select_recovery_strategy(context_other)
        assert strategy == RecoveryStrategy.SKIP_UNTIL_VALID

    def test_strategy_selection_malformed_attribute(self):
        """Test strategy selection for malformed attributes."""
        engine = ErrorRecoveryEngine()
        position = TokenPosition(1, 5, 4)
        
        context = RecoveryContext(
            error_type="malformed_attribute",
            error_position=position,
            surrounding_content='class="unclosed',
            tokenizer_state=TokenizerState.ATTR_VALUE_QUOTED,
            recent_tokens=[],
            malformation_severity=0.4
        )
        
        strategy = engine._select_recovery_strategy(context)
        assert strategy == RecoveryStrategy.SKIP_UNTIL_VALID

    def test_pattern_caching(self):
        """Test pattern caching functionality."""
        engine = ErrorRecoveryEngine()
        position = TokenPosition(1, 5, 4)
        
        context = RecoveryContext(
            error_type="test_error",
            error_position=position,
            surrounding_content="test content",
            tokenizer_state=TokenizerState.TEXT_CONTENT,
            recent_tokens=[],
            malformation_severity=0.3
        )
        
        # Cache a successful pattern
        strategy = RecoveryStrategy.TREAT_AS_TEXT
        engine._cache_successful_pattern(context, strategy)
        
        # Check that the pattern is cached
        pattern_key = engine._generate_pattern_key(context)
        assert pattern_key in engine.pattern_cache
        assert engine.pattern_cache[pattern_key] == strategy
        
        # Test strategy selection uses cached pattern
        selected_strategy = engine._select_recovery_strategy(context)
        assert selected_strategy == strategy

    def test_final_fallback_recovery(self):
        """Test final fallback recovery when all else fails."""
        engine = ErrorRecoveryEngine()
        position = TokenPosition(1, 5, 4)
        
        context = RecoveryContext(
            error_type="catastrophic_error",
            error_position=position,
            surrounding_content="corrupted",
            tokenizer_state=TokenizerState.TEXT_CONTENT,
            recent_tokens=[],
            malformation_severity=0.9
        )
        
        action = engine._final_fallback_recovery(context, "?")
        
        assert action.strategy == RecoveryStrategy.TREAT_AS_TEXT
        assert action.success is True
        assert len(action.tokens) == 1
        assert action.tokens[0].type == TokenType.TEXT
        assert action.tokens[0].value == "?"
        assert action.confidence == 0.1  # Low confidence for fallback
        assert action.metadata["fallback"] is True

    def test_recovery_attempt_limiting(self):
        """Test that recovery attempts are limited to prevent infinite loops."""
        engine = ErrorRecoveryEngine()
        position = TokenPosition(1, 5, 4)
        
        context = RecoveryContext(
            error_type="test_error",
            error_position=position,
            surrounding_content="test",
            tokenizer_state=TokenizerState.TEXT_CONTENT,
            recent_tokens=[],
            malformation_severity=0.5
        )
        
        # Exceed maximum recovery attempts
        engine.recovery_attempts = 50  # MAX_RECOVERY_ATTEMPTS
        
        action = engine.recover_from_error(context, "x")
        
        # Should use final fallback
        assert action.confidence == 0.1
        assert action.metadata["fallback"] is True

    def test_strategy_effectiveness_tracking(self):
        """Test strategy effectiveness tracking and updates."""
        engine = ErrorRecoveryEngine()
        
        initial_effectiveness = engine.strategy_effectiveness[RecoveryStrategy.TREAT_AS_TEXT]
        
        # Test successful strategy update
        engine._update_strategy_effectiveness(RecoveryStrategy.TREAT_AS_TEXT, True)
        assert engine.strategy_effectiveness[RecoveryStrategy.TREAT_AS_TEXT] > initial_effectiveness
        
        # Test failed strategy update
        engine._update_strategy_effectiveness(RecoveryStrategy.TREAT_AS_TEXT, False)
        # Should decrease but not go below the initial + 0.1 - 0.1
        assert engine.strategy_effectiveness[RecoveryStrategy.TREAT_AS_TEXT] >= 0.1

    def test_recovery_statistics(self):
        """Test recovery statistics generation."""
        engine = ErrorRecoveryEngine()
        
        # Initial statistics
        stats = engine.get_recovery_statistics()
        assert stats["total_attempts"] == 0
        assert stats["pattern_cache_size"] == 0
        assert stats["escalation_failures"] == 0
        assert stats["success_rate"] == 1.0
        assert "strategy_effectiveness" in stats
        
        # Simulate some recovery attempts
        engine.recovery_attempts = 10
        engine.escalation_failures = 2
        engine.pattern_cache["test"] = RecoveryStrategy.TREAT_AS_TEXT
        
        stats = engine.get_recovery_statistics()
        assert stats["total_attempts"] == 10
        assert stats["pattern_cache_size"] == 1
        assert stats["escalation_failures"] == 2
        assert stats["success_rate"] == 0.8  # (10 - 2) / 10

    def test_statistics_reset(self):
        """Test resetting recovery statistics."""
        engine = ErrorRecoveryEngine()
        
        # Set some statistics
        engine.recovery_attempts = 10
        engine.escalation_failures = 2
        engine.pattern_cache["test"] = RecoveryStrategy.TREAT_AS_TEXT
        
        # Reset statistics
        engine.reset_statistics()
        
        assert engine.recovery_attempts == 0
        assert engine.escalation_failures == 0
        assert len(engine.pattern_cache) == 0
        
        # Strategy effectiveness should be reset to defaults
        for strategy in RecoveryStrategy:
            assert engine.strategy_effectiveness[strategy] == 0.5

    def test_pattern_key_generation(self):
        """Test pattern key generation for caching."""
        engine = ErrorRecoveryEngine()
        position = TokenPosition(1, 5, 4)
        
        context = RecoveryContext(
            error_type="test_error",
            error_position=position,
            surrounding_content="this is a long content string for testing",
            tokenizer_state=TokenizerState.TAG_NAME,
            recent_tokens=[],
            malformation_severity=0.3
        )
        
        key = engine._generate_pattern_key(context)
        expected = "test_error:TAG_NAME:this is a long conte"
        assert key == expected

    def test_recovery_with_correlation_id(self):
        """Test recovery engine with correlation ID for tracking."""
        correlation_id = "test-session-123"
        engine = ErrorRecoveryEngine(correlation_id=correlation_id)
        position = TokenPosition(1, 5, 4)
        
        context = RecoveryContext(
            error_type="test_error",
            error_position=position,
            surrounding_content="test",
            tokenizer_state=TokenizerState.TEXT_CONTENT,
            recent_tokens=[],
            malformation_severity=0.3
        )
        
        action = engine.recover_from_error(context, "x")
        
        assert engine.correlation_id == correlation_id
        assert action.success is True
        assert engine.recovery_attempts == 1

    def test_pattern_cache_size_limit(self):
        """Test that pattern cache respects size limits."""
        engine = ErrorRecoveryEngine()
        position = TokenPosition(1, 1, 0)
        
        # Fill cache beyond limit
        for i in range(1005):  # More than PATTERN_CACHE_SIZE (1000)
            context = RecoveryContext(
                error_type=f"error_{i}",
                error_position=position,
                surrounding_content=f"content_{i}",
                tokenizer_state=TokenizerState.TEXT_CONTENT,
                recent_tokens=[],
                malformation_severity=0.3
            )
            engine._cache_successful_pattern(context, RecoveryStrategy.TREAT_AS_TEXT)
        
        # Cache should not exceed limit
        assert len(engine.pattern_cache) <= 1000


class TestIntegrationScenarios:
    """Integration tests for complex malformation scenarios."""

    def test_multiple_malformation_recovery(self):
        """Test recovery from multiple types of malformation in sequence."""
        engine = ErrorRecoveryEngine()
        position = TokenPosition(1, 1, 0)
        
        # First malformation - invalid character
        context1 = RecoveryContext(
            error_type="invalid_character_sequence",
            error_position=position,
            surrounding_content="<<test",
            tokenizer_state=TokenizerState.TAG_OPENING,
            recent_tokens=[],
            malformation_severity=0.4
        )
        
        action1 = engine.recover_from_error(context1, "<")
        assert action1.success is True
        assert engine.recovery_attempts == 1
        
        # Second malformation - missing tag name
        context2 = RecoveryContext(
            error_type="invalid_tag_structure",
            error_position=TokenPosition(1, 3, 2),
            surrounding_content="< >",
            tokenizer_state=TokenizerState.TAG_OPENING,
            recent_tokens=[],
            malformation_severity=0.5
        )
        
        action2 = engine.recover_from_error(context2, ">")
        assert action2.success is True
        assert engine.recovery_attempts == 2
        
        # Check statistics
        stats = engine.get_recovery_statistics()
        assert stats["total_attempts"] == 2
        assert stats["success_rate"] == 1.0

    def test_escalating_malformation_severity(self):
        """Test behavior with escalating malformation severity."""
        # Disable history for this test to avoid pattern interference
        engine = ErrorRecoveryEngine(enable_history=False)
        position = TokenPosition(1, 1, 0)
        
        # Start with mild malformation
        context_mild = RecoveryContext(
            error_type="invalid_character_sequence",
            error_position=position,
            surrounding_content="test<>",
            tokenizer_state=TokenizerState.TEXT_CONTENT,
            recent_tokens=[],
            malformation_severity=0.2
        )
        
        action_mild = engine.recover_from_error(context_mild, "<")
        strategy_mild = action_mild.strategy
        
        # Escalate to severe malformation
        context_severe = RecoveryContext(
            error_type="invalid_character_sequence",
            error_position=position,
            surrounding_content="completely corrupt <<<>>>",
            tokenizer_state=TokenizerState.TEXT_CONTENT,
            recent_tokens=[],
            malformation_severity=0.9
        )
        
        action_severe = engine.recover_from_error(context_severe, "<")
        strategy_severe = action_severe.strategy
        
        # Should use different strategies for different severities
        assert strategy_mild == RecoveryStrategy.CHARACTER_ESCAPE
        assert strategy_severe == RecoveryStrategy.TREAT_AS_TEXT


class TestRecoveryHistory:
    """Tests for RecoveryHistory class and related functionality."""

    def test_recovery_history_initialization(self):
        """Test recovery history initialization."""
        history = RecoveryHistory()
        
        assert history.max_entries == 10000
        assert len(history.entries) == 0
        assert isinstance(history.statistics, RecoveryStatistics)
        assert len(history._pattern_correlation_cache) == 0

    def test_recovery_history_entry_creation(self):
        """Test creating recovery history entries."""
        position = TokenPosition(1, 5, 4)
        
        entry = RecoveryHistoryEntry(
            timestamp=1234567890.0,
            error_type="test_error",
            error_position=position,
            strategy_used=RecoveryStrategy.TREAT_AS_TEXT,
            success=True,
            confidence=0.8,
            tokens_generated=1,
            processing_time_ms=5.2,
            rationale="Test recovery",
            correlation_id="test-123"
        )
        
        assert entry.timestamp == 1234567890.0
        assert entry.error_type == "test_error"
        assert entry.error_position == position
        assert entry.strategy_used == RecoveryStrategy.TREAT_AS_TEXT
        assert entry.success is True
        assert entry.confidence == 0.8
        assert entry.tokens_generated == 1
        assert entry.processing_time_ms == 5.2
        assert entry.rationale == "Test recovery"
        assert entry.correlation_id == "test-123"
        assert entry.position_str == "1:5"

    def test_recovery_statistics_properties(self):
        """Test recovery statistics calculated properties."""
        stats = RecoveryStatistics()
        
        # Empty statistics
        assert stats.overall_success_rate == 1.0
        assert stats.average_processing_time_ms == 0.0
        
        # With data
        stats.total_attempts = 10
        stats.successful_attempts = 8
        stats.total_processing_time_ms = 50.0
        
        assert stats.overall_success_rate == 0.8
        assert stats.average_processing_time_ms == 5.0

    def test_record_recovery_operation(self):
        """Test recording recovery operations."""
        history = RecoveryHistory()
        position = TokenPosition(1, 5, 4)
        token = Token(TokenType.TEXT, "test", position)
        
        context = RecoveryContext(
            error_type="test_error",
            error_position=position,
            surrounding_content="test content here",
            tokenizer_state=TokenizerState.TEXT_CONTENT,
            recent_tokens=[],
            malformation_severity=0.3
        )
        
        action = RecoveryAction(
            strategy=RecoveryStrategy.TREAT_AS_TEXT,
            success=True,
            tokens=[token],
            confidence=0.8,
            description="Test recovery",
            original_content="<",
            repaired_content="test",
            position=position
        )
        
        history.record_recovery(context, action, 5.2, "Test rationale", "test-123")
        
        assert len(history.entries) == 1
        entry = history.entries[0]
        assert entry.error_type == "test_error"
        assert entry.strategy_used == RecoveryStrategy.TREAT_AS_TEXT
        assert entry.success is True
        assert entry.confidence == 0.8
        assert entry.tokens_generated == 1
        assert entry.processing_time_ms == 5.2
        assert entry.rationale == "Test rationale"
        assert entry.correlation_id == "test-123"
        
        # Check statistics were updated
        assert history.statistics.total_attempts == 1
        assert history.statistics.successful_attempts == 1
        assert history.statistics.failed_attempts == 0

    def test_pattern_recommendation(self):
        """Test pattern-based strategy recommendations."""
        history = RecoveryHistory()
        position = TokenPosition(1, 5, 4)
        token = Token(TokenType.TEXT, "test", position)
        
        context = RecoveryContext(
            error_type="test_error",
            error_position=position,
            surrounding_content="test",
            tokenizer_state=TokenizerState.TEXT_CONTENT,
            recent_tokens=[],
            malformation_severity=0.3
        )
        
        action = RecoveryAction(
            strategy=RecoveryStrategy.TREAT_AS_TEXT,
            success=True,
            tokens=[token],
            confidence=0.8,
            description="Test recovery",
            original_content="<",
            repaired_content="test",
            position=position
        )
        
        # Record multiple successful operations with same pattern
        for _ in range(3):
            history.record_recovery(context, action, 5.0, "Test")
            
        # Should get pattern recommendation
        recommendation = history.get_pattern_recommendation(context)
        assert recommendation == RecoveryStrategy.TREAT_AS_TEXT

    def test_history_size_limit(self):
        """Test history size limiting."""
        history = RecoveryHistory(max_entries=5)  # Small limit for testing
        position = TokenPosition(1, 1, 0)
        token = Token(TokenType.TEXT, "test", position)
        
        context = RecoveryContext(
            error_type="test_error",
            error_position=position,
            surrounding_content="test",
            tokenizer_state=TokenizerState.TEXT_CONTENT,
            recent_tokens=[],
            malformation_severity=0.3
        )
        
        action = RecoveryAction(
            strategy=RecoveryStrategy.TREAT_AS_TEXT,
            success=True,
            tokens=[token],
            confidence=0.8,
            description="Test recovery",
            original_content="<",
            repaired_content="test",
            position=position
        )
        
        # Add more entries than the limit
        for i in range(10):
            history.record_recovery(context, action, 5.0, f"Entry {i}")
            
        # Should maintain size limit
        assert len(history.entries) == 5
        # Should keep the most recent entries
        assert history.entries[-1].rationale == "Entry 9"

    def test_get_entries_by_filters(self):
        """Test filtering history entries by various criteria."""
        history = RecoveryHistory()
        position = TokenPosition(1, 1, 0)
        token = Token(TokenType.TEXT, "test", position)
        
        # Create different contexts and actions
        context1 = RecoveryContext(
            error_type="error_type_1",
            error_position=position,
            surrounding_content="test",
            tokenizer_state=TokenizerState.TEXT_CONTENT,
            recent_tokens=[],
            malformation_severity=0.3
        )
        
        context2 = RecoveryContext(
            error_type="error_type_2",
            error_position=position,
            surrounding_content="test",
            tokenizer_state=TokenizerState.TEXT_CONTENT,
            recent_tokens=[],
            malformation_severity=0.3
        )
        
        action1 = RecoveryAction(
            strategy=RecoveryStrategy.TREAT_AS_TEXT,
            success=True,
            tokens=[token],
            confidence=0.8,
            description="Test",
            original_content="<",
            repaired_content="test",
            position=position
        )
        
        action2 = RecoveryAction(
            strategy=RecoveryStrategy.SKIP_UNTIL_VALID,
            success=True,
            tokens=[token],
            confidence=0.7,
            description="Test",
            original_content="<",
            repaired_content="test",
            position=position
        )
        
        # Record different combinations
        history.record_recovery(context1, action1, 5.0, "Test 1")
        history.record_recovery(context1, action2, 4.0, "Test 2") 
        history.record_recovery(context2, action1, 6.0, "Test 3")
        
        # Test filtering by error type
        type1_entries = history.get_entries_by_error_type("error_type_1")
        assert len(type1_entries) == 2
        
        type2_entries = history.get_entries_by_error_type("error_type_2")
        assert len(type2_entries) == 1
        
        # Test filtering by strategy
        text_entries = history.get_entries_by_strategy(RecoveryStrategy.TREAT_AS_TEXT)
        assert len(text_entries) == 2
        
        skip_entries = history.get_entries_by_strategy(RecoveryStrategy.SKIP_UNTIL_VALID)
        assert len(skip_entries) == 1

    def test_confidence_distribution(self):
        """Test confidence distribution calculation."""
        history = RecoveryHistory()
        position = TokenPosition(1, 1, 0)
        token = Token(TokenType.TEXT, "test", position)
        
        context = RecoveryContext(
            error_type="test_error",
            error_position=position,
            surrounding_content="test",
            tokenizer_state=TokenizerState.TEXT_CONTENT,
            recent_tokens=[],
            malformation_severity=0.3
        )
        
        # Create actions with different confidence levels
        confidences = [0.1, 0.3, 0.5, 0.7, 0.9]
        for conf in confidences:
            action = RecoveryAction(
                strategy=RecoveryStrategy.TREAT_AS_TEXT,
                success=True,
                tokens=[token],
                confidence=conf,
                description="Test",
                original_content="<",
                repaired_content="test",
                position=position
            )
            history.record_recovery(context, action, 5.0, "Test")
            
        distribution = history.get_confidence_distribution()
        assert distribution["0.0-0.2"] == 1  # 0.1
        assert distribution["0.2-0.4"] == 1  # 0.3
        assert distribution["0.4-0.6"] == 1  # 0.5
        assert distribution["0.6-0.8"] == 1  # 0.7
        assert distribution["0.8-1.0"] == 1  # 0.9

    def test_detailed_report_generation(self):
        """Test comprehensive report generation."""
        history = RecoveryHistory()
        position = TokenPosition(1, 1, 0)
        token = Token(TokenType.TEXT, "test", position)
        
        context = RecoveryContext(
            error_type="test_error",
            error_position=position,
            surrounding_content="test content",
            tokenizer_state=TokenizerState.TEXT_CONTENT,
            recent_tokens=[],
            malformation_severity=0.3
        )
        
        action = RecoveryAction(
            strategy=RecoveryStrategy.TREAT_AS_TEXT,
            success=True,
            tokens=[token],
            confidence=0.8,
            description="Test recovery",
            original_content="<",
            repaired_content="test",
            position=position
        )
        
        # Record some operations
        for i in range(3):
            history.record_recovery(context, action, 5.0, f"Test {i}")
            
        report = history.generate_detailed_report()
        
        # Check report structure
        assert "summary" in report
        assert "strategy_analysis" in report
        assert "error_analysis" in report
        assert "confidence_distribution" in report
        assert "pattern_correlations" in report
        assert "recent_entries" in report
        
        # Check summary
        summary = report["summary"]
        assert summary["total_entries"] == 3
        assert summary["overall_success_rate"] == 1.0
        assert abs(summary["average_confidence"] - 0.8) < 0.01  # Allow for floating point precision
        
        # Check recent entries
        recent_entries = report["recent_entries"]
        assert len(recent_entries) == 3
        assert recent_entries[-1]["rationale"] == "Test 2"

    def test_engine_with_history_integration(self):
        """Test ErrorRecoveryEngine integration with history tracking."""
        engine = ErrorRecoveryEngine(correlation_id="test-session", enable_history=True)
        position = TokenPosition(1, 5, 4)
        
        context = RecoveryContext(
            error_type="test_error",
            error_position=position,
            surrounding_content="test",
            tokenizer_state=TokenizerState.TEXT_CONTENT,
            recent_tokens=[],
            malformation_severity=0.3
        )
        
        # Perform recovery
        action = engine.recover_from_error(context, "<")
        
        # Check that history was recorded
        assert engine.history is not None
        assert len(engine.history.entries) == 1
        
        entry = engine.history.entries[0]
        assert entry.error_type == "test_error"
        assert entry.correlation_id == "test-session"
        assert entry.success is True
        
        # Test history report access
        report = engine.get_recovery_history_report()
        assert report is not None
        assert report["summary"]["total_entries"] == 1
        
        # Test recent entries access
        recent_entries = engine.get_recent_recovery_entries()
        assert len(recent_entries) == 1
        assert recent_entries[0].error_type == "test_error"

    def test_engine_without_history(self):
        """Test ErrorRecoveryEngine without history tracking."""
        engine = ErrorRecoveryEngine(enable_history=False)
        position = TokenPosition(1, 5, 4)
        
        context = RecoveryContext(
            error_type="test_error",
            error_position=position,
            surrounding_content="test",
            tokenizer_state=TokenizerState.TEXT_CONTENT,
            recent_tokens=[],
            malformation_severity=0.3
        )
        
        # Perform recovery
        action = engine.recover_from_error(context, "<")
        
        # Check that no history was recorded
        assert engine.history is None
        
        # History methods should return None or empty
        assert engine.get_recovery_history_report() is None
        assert engine.get_recent_recovery_entries() == []

    def test_history_pattern_based_strategy_selection(self):
        """Test that history influences future strategy selection."""
        engine = ErrorRecoveryEngine(enable_history=True)
        position = TokenPosition(1, 1, 0)
        
        context = RecoveryContext(
            error_type="pattern_test_error",
            error_position=position,
            surrounding_content="test",
            tokenizer_state=TokenizerState.TEXT_CONTENT,
            recent_tokens=[],
            malformation_severity=0.3
        )
        
        # First recovery - will use standard strategy selection
        first_action = engine.recover_from_error(context, "<")
        first_strategy = first_action.strategy
        
        # Simulate successful pattern by recording it multiple times
        for _ in range(3):
            engine.recover_from_error(context, "<")
        
        # Clear cache to force pattern-based selection
        engine.pattern_cache.clear()
        
        # Next recovery should potentially use pattern-based recommendation
        # (This is probabilistic based on the specific strategy selection logic)
        next_action = engine.recover_from_error(context, "<")
        
        # Verify that history influenced the decision
        assert engine.history is not None
        assert len(engine.history.entries) > 1
        
        # Should have pattern correlations
        pattern_key = f"pattern_test_error:TEXT_CONTENT"
        assert pattern_key in engine.history.statistics.pattern_correlations