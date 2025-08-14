"""Tests for the debugging module."""

import json
import tempfile
import time
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

from ultra_robust_xml_parser.tools.debugging import (
    Breakpoint,
    BreakpointType,
    DebugConsole,
    DebugLevel,
    DebugMode,
    DebugSession,
    DebugState,
)


class TestDebugState:
    """Test DebugState data class."""
    
    def test_debug_state_creation(self):
        """Test debug state creation."""
        debug_state = DebugState(
            layer_name="test_layer",
            step_name="test_step",
            timestamp=1000.0,
            data={"key": "value"},
            metadata={"info": "test"}
        )
        
        assert debug_state.layer_name == "test_layer"
        assert debug_state.step_name == "test_step"
        assert debug_state.timestamp == 1000.0
        assert debug_state.data["key"] == "value"
        assert debug_state.metadata["info"] == "test"
    
    def test_debug_state_to_dict(self):
        """Test debug state conversion to dictionary."""
        debug_state = DebugState(
            layer_name="layer",
            step_name="step",
            timestamp=1000.0,
            data={"data": "test"},
            metadata={"meta": "info"}
        )
        
        state_dict = debug_state.to_dict()
        
        assert state_dict["layer_name"] == "layer"
        assert state_dict["step_name"] == "step"
        assert state_dict["timestamp"] == 1000.0
        assert state_dict["data"]["data"] == "test"
        assert state_dict["metadata"]["meta"] == "info"


class TestBreakpoint:
    """Test Breakpoint data class."""
    
    def test_breakpoint_creation(self):
        """Test breakpoint creation."""
        breakpoint = Breakpoint(
            id="test_bp",
            breakpoint_type=BreakpointType.STATE_CHANGE,
            layer_name="test_layer",
            step_name="test_step"
        )
        
        assert breakpoint.id == "test_bp"
        assert breakpoint.breakpoint_type == BreakpointType.STATE_CHANGE
        assert breakpoint.layer_name == "test_layer"
        assert breakpoint.step_name == "test_step"
        assert breakpoint.enabled is True
        assert breakpoint.hit_count == 0
    
    def test_breakpoint_matches_layer_filter(self):
        """Test breakpoint matching with layer filter."""
        breakpoint = Breakpoint(
            id="layer_bp",
            breakpoint_type=BreakpointType.LAYER_ENTRY,
            layer_name="target_layer"
        )
        
        # Matching state
        matching_state = DebugState("target_layer", "any_step", 1000.0)
        assert breakpoint.matches(matching_state) is True
        
        # Non-matching state
        non_matching_state = DebugState("other_layer", "any_step", 1000.0)
        assert breakpoint.matches(non_matching_state) is False
    
    def test_breakpoint_matches_step_filter(self):
        """Test breakpoint matching with step filter."""
        breakpoint = Breakpoint(
            id="step_bp",
            breakpoint_type=BreakpointType.STATE_CHANGE,
            step_name="target_step"
        )
        
        # Matching state
        matching_state = DebugState("any_layer", "target_step", 1000.0)
        assert breakpoint.matches(matching_state) is True
        
        # Non-matching state
        non_matching_state = DebugState("any_layer", "other_step", 1000.0)
        assert breakpoint.matches(non_matching_state) is False
    
    def test_breakpoint_custom_condition(self):
        """Test breakpoint with custom condition."""
        def error_condition(state: DebugState) -> bool:
            return state.data.get("error", False)
        
        breakpoint = Breakpoint(
            id="error_bp",
            breakpoint_type=BreakpointType.CUSTOM_CONDITION,
            condition=error_condition
        )
        
        # State with error
        error_state = DebugState("layer", "step", 1000.0, {"error": True})
        assert breakpoint.matches(error_state) is True
        
        # State without error
        normal_state = DebugState("layer", "step", 1000.0, {"error": False})
        assert breakpoint.matches(normal_state) is False
    
    def test_breakpoint_disabled(self):
        """Test disabled breakpoint."""
        breakpoint = Breakpoint(
            id="disabled_bp",
            breakpoint_type=BreakpointType.STATE_CHANGE,
            enabled=False
        )
        
        state = DebugState("layer", "step", 1000.0)
        assert breakpoint.matches(state) is False
    
    def test_breakpoint_hit_tracking(self):
        """Test breakpoint hit tracking."""
        breakpoint = Breakpoint(
            id="hit_bp",
            breakpoint_type=BreakpointType.STATE_CHANGE
        )
        
        assert breakpoint.hit_count == 0
        
        breakpoint.hit()
        assert breakpoint.hit_count == 1
        
        breakpoint.hit()
        assert breakpoint.hit_count == 2


class TestDebugSession:
    """Test DebugSession class."""
    
    def test_debug_session_creation(self):
        """Test debug session creation."""
        session = DebugSession("test_session", DebugLevel.VERBOSE)
        
        assert session.session_id == "test_session"
        assert session.debug_level == DebugLevel.VERBOSE
        assert len(session.states) == 0
        assert len(session.breakpoints) == 0
        assert session.active is True
        assert session.step_mode is False
        assert session.current_state_index == -1
    
    def test_add_state(self):
        """Test adding states to session."""
        session = DebugSession("test")
        
        state1 = DebugState("layer1", "step1", 1000.0)
        state2 = DebugState("layer2", "step2", 2000.0)
        
        session.add_state(state1)
        assert len(session.states) == 1
        assert session.current_state_index == 0
        assert session.get_current_state() == state1
        
        session.add_state(state2)
        assert len(session.states) == 2
        assert session.current_state_index == 1
        assert session.get_current_state() == state2
    
    def test_breakpoint_management(self):
        """Test breakpoint management in session."""
        session = DebugSession("test")
        
        breakpoint = Breakpoint("bp1", BreakpointType.STATE_CHANGE)
        
        # Add breakpoint
        session.add_breakpoint(breakpoint)
        assert "bp1" in session.breakpoints
        assert session.breakpoints["bp1"] == breakpoint
        
        # Remove breakpoint
        assert session.remove_breakpoint("bp1") is True
        assert "bp1" not in session.breakpoints
        
        # Remove non-existent breakpoint
        assert session.remove_breakpoint("nonexistent") is False
    
    def test_check_breakpoints(self):
        """Test breakpoint checking."""
        session = DebugSession("test")
        
        # Add breakpoints
        bp1 = Breakpoint("bp1", BreakpointType.STATE_CHANGE, layer_name="target_layer")
        bp2 = Breakpoint("bp2", BreakpointType.LAYER_ENTRY, step_name="target_step")
        bp3 = Breakpoint("bp3", BreakpointType.ERROR_CONDITION, enabled=False)
        
        session.add_breakpoint(bp1)
        session.add_breakpoint(bp2)
        session.add_breakpoint(bp3)
        
        # Test matching state
        state = DebugState("target_layer", "target_step", 1000.0)
        matched = session.check_breakpoints(state)
        
        assert len(matched) == 2  # bp1 and bp2 should match, bp3 is disabled
        matched_ids = [bp.id for bp in matched]
        assert "bp1" in matched_ids
        assert "bp2" in matched_ids
        assert "bp3" not in matched_ids
        
        # Check hit counts
        assert bp1.hit_count == 1
        assert bp2.hit_count == 1
        assert bp3.hit_count == 0


class TestDebugConsole:
    """Test DebugConsole interactive interface."""
    
    def test_console_creation(self):
        """Test console creation."""
        session = DebugSession("test")
        console = DebugConsole(session)
        
        assert console.debug_session == session
        assert console.prompt == "(debug) "
    
    def test_state_command(self):
        """Test state display command."""
        session = DebugSession("test")
        state = DebugState("layer", "step", 1000.0, {"key": "value"}, {"meta": "info"})
        session.add_state(state)
        
        console = DebugConsole(session)
        
        # Capture output
        with patch('builtins.print') as mock_print:
            console.do_state("")
            
            # Should print state information
            mock_print.assert_called()
            calls = mock_print.call_args_list
            call_args = [str(call) for call in calls]
            
            # Check that layer and step are mentioned
            assert any("layer" in arg for arg in call_args)
            assert any("step" in arg for arg in call_args)
    
    def test_history_command(self):
        """Test history display command."""
        session = DebugSession("test")
        
        # Add multiple states
        for i in range(5):
            state = DebugState(f"layer{i}", f"step{i}", 1000.0 + i)
            session.add_state(state)
        
        console = DebugConsole(session)
        
        with patch('builtins.print') as mock_print:
            console.do_history("3")  # Show last 3 states
            
            # Should print history
            mock_print.assert_called()
            # Should have called print for each of the 3 states
            assert len(mock_print.call_args_list) == 3
    
    def test_goto_command(self):
        """Test goto state command."""
        session = DebugSession("test")
        
        # Add states
        for i in range(3):
            state = DebugState(f"layer{i}", f"step{i}", 1000.0 + i)
            session.add_state(state)
        
        console = DebugConsole(session)
        
        # Go to state 1
        with patch('builtins.print') as mock_print:
            console.do_goto("1")
            
            assert session.current_state_index == 1
            mock_print.assert_called()
    
    def test_breakpoint_commands(self):
        """Test breakpoint management commands."""
        session = DebugSession("test")
        console = DebugConsole(session)
        
        # Add breakpoint
        with patch('builtins.print') as mock_print:
            console.do_breakpoint("add bp1 state_change layer1")
            
            assert "bp1" in session.breakpoints
            mock_print.assert_called_with("Added breakpoint bp1")
        
        # List breakpoints
        with patch('builtins.print') as mock_print:
            console.do_breakpoint("list")
            
            mock_print.assert_called()
        
        # Disable breakpoint
        with patch('builtins.print') as mock_print:
            console.do_breakpoint("disable bp1")
            
            assert session.breakpoints["bp1"].enabled is False
            mock_print.assert_called_with("Breakpoint bp1 disabled")
        
        # Remove breakpoint
        with patch('builtins.print') as mock_print:
            console.do_breakpoint("remove bp1")
            
            assert "bp1" not in session.breakpoints
            mock_print.assert_called_with("Removed breakpoint bp1")


class TestDebugMode:
    """Test DebugMode class."""
    
    def test_debug_mode_creation(self):
        """Test debug mode creation."""
        debug_mode = DebugMode(DebugLevel.VERBOSE)
        
        assert debug_mode.default_level == DebugLevel.VERBOSE
        assert len(debug_mode.sessions) == 0
        assert debug_mode.active_session is None
        assert debug_mode.interactive_mode is False
    
    def test_session_management(self):
        """Test debug session management."""
        debug_mode = DebugMode()
        
        # Start session
        session = debug_mode.start_session("test_session", DebugLevel.DETAILED)
        
        assert session.session_id == "test_session"
        assert session.debug_level == DebugLevel.DETAILED
        assert "test_session" in debug_mode.sessions
        assert debug_mode.active_session == session
        assert session.active is True
        
        # End session
        debug_mode.end_session(session)
        
        assert session.active is False
        assert debug_mode.active_session is None
    
    def test_capture_state(self):
        """Test state capture."""
        debug_mode = DebugMode()
        session = debug_mode.start_session("test")
        
        # Capture state
        data = {"key": "value", "count": 10}
        metadata = {"info": "test"}
        
        state = debug_mode.capture_state(
            session,
            data,
            "test_layer",
            "test_step",
            metadata
        )
        
        assert isinstance(state, DebugState)
        assert state.layer_name == "test_layer"
        assert state.step_name == "test_step"
        assert state.data == data
        assert state.metadata == metadata
        assert len(session.states) == 1
    
    def test_add_breakpoint(self):
        """Test adding breakpoints."""
        debug_mode = DebugMode()
        session = debug_mode.start_session("test")
        
        # Add breakpoint
        debug_mode.add_breakpoint(
            session,
            "test_bp",
            BreakpointType.STATE_CHANGE,
            layer_name="test_layer"
        )
        
        assert "test_bp" in session.breakpoints
        bp = session.breakpoints["test_bp"]
        assert bp.id == "test_bp"
        assert bp.breakpoint_type == BreakpointType.STATE_CHANGE
        assert bp.layer_name == "test_layer"
    
    def test_interactive_mode(self):
        """Test interactive mode enabling."""
        debug_mode = DebugMode()
        session = debug_mode.start_session("test")
        
        assert debug_mode.interactive_mode is False
        assert session.step_mode is False
        
        debug_mode.enable_interactive_mode(session)
        
        assert debug_mode.interactive_mode is True
        assert session.step_mode is True
    
    def test_get_session_summary(self):
        """Test session summary generation."""
        debug_mode = DebugMode()
        session = debug_mode.start_session("test", DebugLevel.NORMAL)
        
        # Add some states
        debug_mode.capture_state(session, {"data": 1}, "layer1", "step1")
        debug_mode.capture_state(session, {"data": 2}, "layer1", "step2")
        debug_mode.capture_state(session, {"data": 3}, "layer2", "step1")
        
        # Add breakpoint
        debug_mode.add_breakpoint(session, "bp1", BreakpointType.STATE_CHANGE)
        session.breakpoints["bp1"].hit_count = 2
        
        summary = debug_mode.get_session_summary(session)
        
        assert summary["session_id"] == "test"
        assert summary["debug_level"] == "normal"
        assert summary["states_captured"] == 3
        assert summary["breakpoints_configured"] == 1
        assert summary["breakpoints_hit"] == 2
        
        # Check layer statistics
        layer_stats = summary["layer_statistics"]
        assert "layer1" in layer_stats
        assert "layer2" in layer_stats
        assert layer_stats["layer1"]["count"] == 2
        assert layer_stats["layer2"]["count"] == 1
        assert set(layer_stats["layer1"]["steps"]) == {"step1", "step2"}


class TestDebugContext:
    """Test DebugContext context manager."""
    
    def test_debug_context_normal_execution(self):
        """Test debug context with normal execution."""
        debug_mode = DebugMode()
        session = debug_mode.start_session("test")
        
        with debug_mode.debug_context(session, "test_layer", "test_step") as context:
            assert isinstance(context, type(debug_mode.debug_context(session, "", "")))
            
            # Capture some state within context
            context.capture({"operation": "processing"}, "process")
        
        # Should have captured entry, process, and exit states
        assert len(session.states) == 3
        
        states = session.states
        assert states[0].step_name == "test_step_enter"
        assert states[1].step_name == "test_step_process"
        assert states[2].step_name == "test_step_exit"
        
        # Exit state should indicate success
        assert states[2].data["success"] is True
    
    def test_debug_context_with_exception(self):
        """Test debug context with exception."""
        debug_mode = DebugMode()
        session = debug_mode.start_session("test")
        
        try:
            with debug_mode.debug_context(session, "test_layer", "test_step"):
                raise ValueError("Test exception")
        except ValueError:
            pass  # Expected
        
        # Should have captured entry and exit states
        assert len(session.states) == 2
        
        exit_state = session.states[1]
        assert exit_state.step_name == "test_step_exit"
        assert exit_state.data["success"] is False
        assert "exception" in exit_state.metadata
        assert exit_state.metadata["exception"]["type"] == "ValueError"


class TestDebugModeIntegration:
    """Integration tests for debug mode functionality."""
    
    def test_breakpoint_hit_handling(self):
        """Test breakpoint hit handling in non-interactive mode."""
        debug_mode = DebugMode()
        session = debug_mode.start_session("test")
        
        # Add breakpoint
        debug_mode.add_breakpoint(
            session,
            "test_bp",
            BreakpointType.STATE_CHANGE,
            layer_name="target_layer"
        )
        
        # Capture state that matches breakpoint
        with patch('builtins.print') as mock_print:
            debug_mode.capture_state(
                session,
                {"data": "test"},
                "target_layer",
                "test_step"
            )
            
            # In non-interactive mode, should just log
            assert len(session.states) == 1
            assert session.breakpoints["test_bp"].hit_count == 1
    
    def test_step_mode_handling(self):
        """Test step mode handling."""
        debug_mode = DebugMode()
        session = debug_mode.start_session("test")
        session.step_mode = True
        
        with patch('builtins.print') as mock_print:
            debug_mode.capture_state(
                session,
                {"data": "test"},
                "layer",
                "step"
            )
            
            # Should print step information
            mock_print.assert_called()
    
    def test_debug_session_export(self):
        """Test debug session export functionality."""
        debug_mode = DebugMode()
        session = debug_mode.start_session("export_test", DebugLevel.DETAILED)
        
        # Add some data
        debug_mode.capture_state(session, {"data": "test"}, "layer", "step")
        debug_mode.add_breakpoint(session, "bp1", BreakpointType.STATE_CHANGE)
        session.breakpoints["bp1"].hit_count = 1
        
        # Create console and test export
        console = DebugConsole(session)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            export_path = f.name
        
        try:
            with patch('builtins.print') as mock_print:
                console.do_export(export_path)
                
                mock_print.assert_called_with(f"Debug session exported to {export_path}")
            
            # Verify exported content
            with open(export_path, 'r') as f:
                exported_data = json.load(f)
            
            assert exported_data["session_id"] == "export_test"
            assert exported_data["debug_level"] == "detailed"
            assert len(exported_data["states"]) == 1
            assert len(exported_data["breakpoints"]) == 1
            
        finally:
            Path(export_path).unlink()


@pytest.mark.integration
class TestDebugModeRealUsage:
    """Test debug mode in realistic usage scenarios."""
    
    def test_parsing_layer_debugging(self):
        """Test debugging a parsing layer."""
        debug_mode = DebugMode(DebugLevel.VERBOSE)
        session = debug_mode.start_session("parsing_debug")
        
        # Simulate character processing layer
        with debug_mode.debug_context(session, "character_processing", "encoding_detection") as context:
            context.capture({"input_bytes": b"<xml>", "encoding_hint": None}, "start")
            context.capture({"detected_encoding": "utf-8", "confidence": 0.9}, "detected")
            context.capture({"output_string": "<xml>", "final_encoding": "utf-8"}, "converted")
        
        # Simulate tokenization layer
        with debug_mode.debug_context(session, "tokenization", "token_extraction") as context:
            context.capture({"input": "<xml>", "position": 0}, "start")
            context.capture({"token_type": "START_TAG", "token_value": "<xml>"}, "extracted")
        
        # Verify captured states
        # Each context creates: enter + captures + exit
        # Layer 1: enter + 3 captures + exit = 5 states
        # Layer 2: enter + 2 captures + exit = 4 states
        # Total = 9 states
        expected_states = 9
        actual_states = len(session.states)
        assert actual_states == expected_states, f"Expected {expected_states} states, got {actual_states}"
        
        # Verify layer coverage
        layer_names = [state.layer_name for state in session.states]
        assert "character_processing" in layer_names
        assert "tokenization" in layer_names
        
        # Get summary
        summary = debug_mode.get_session_summary(session)
        assert summary["states_captured"] == expected_states
        assert len(summary["layer_statistics"]) == 2
    
    def test_error_condition_debugging(self):
        """Test debugging with error conditions."""
        debug_mode = DebugMode()
        session = debug_mode.start_session("error_debug")
        
        # Add error condition breakpoint
        def has_error(state: DebugState) -> bool:
            return "error" in state.data and state.data["error"] is not None
        
        debug_mode.add_breakpoint(
            session,
            "error_bp",
            BreakpointType.CUSTOM_CONDITION,
            condition=has_error
        )
        
        # Normal processing - no breakpoint hit
        debug_mode.capture_state(session, {"data": "normal", "error": None}, "layer", "step1")
        assert session.breakpoints["error_bp"].hit_count == 0
        
        # Error condition - breakpoint should hit
        with patch('builtins.print'):  # Suppress debug output
            debug_mode.capture_state(session, {"data": "failed", "error": "Parse error"}, "layer", "step2")
        
        assert session.breakpoints["error_bp"].hit_count == 1