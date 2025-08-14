"""Comprehensive debugging tools for Ultra Robust XML Parser.

Provides advanced debugging capabilities including state inspection, parsing breakpoints,
step-through debugging, interactive console, and debug visualization.
"""

import cmd
import json
import pprint
import sys
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union

from ultra_robust_xml_parser.shared.logging import get_logger


class DebugLevel(Enum):
    """Debug verbosity levels."""
    MINIMAL = "minimal"
    NORMAL = "normal"
    VERBOSE = "verbose"
    DETAILED = "detailed"


class BreakpointType(Enum):
    """Types of debugging breakpoints."""
    STATE_CHANGE = "state_change"
    ERROR_CONDITION = "error_condition"
    LAYER_ENTRY = "layer_entry"
    LAYER_EXIT = "layer_exit"
    CUSTOM_CONDITION = "custom_condition"


@dataclass
class DebugState:
    """Snapshot of parsing state for debugging."""
    
    layer_name: str
    step_name: str
    timestamp: float
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert debug state to dictionary."""
        return {
            "layer_name": self.layer_name,
            "step_name": self.step_name,
            "timestamp": self.timestamp,
            "data": self.data,
            "metadata": self.metadata
        }


@dataclass
class Breakpoint:
    """Debugging breakpoint configuration."""
    
    id: str
    breakpoint_type: BreakpointType
    condition: Optional[Callable[[DebugState], bool]] = None
    layer_name: Optional[str] = None
    step_name: Optional[str] = None
    enabled: bool = True
    hit_count: int = 0
    
    def matches(self, debug_state: DebugState) -> bool:
        """Check if this breakpoint matches the current debug state."""
        if not self.enabled:
            return False
        
        # Check layer name filter
        if self.layer_name and self.layer_name != debug_state.layer_name:
            return False
        
        # Check step name filter
        if self.step_name and self.step_name != debug_state.step_name:
            return False
        
        # Check custom condition
        if self.condition and not self.condition(debug_state):
            return False
        
        return True
    
    def hit(self) -> None:
        """Record a breakpoint hit."""
        self.hit_count += 1


class DebugSession:
    """Container for a debugging session."""
    
    def __init__(self, session_id: str, debug_level: DebugLevel = DebugLevel.NORMAL):
        self.session_id = session_id
        self.debug_level = debug_level
        self.states: List[DebugState] = []
        self.breakpoints: Dict[str, Breakpoint] = {}
        self.active = True
        self.step_mode = False
        self.current_state_index = -1
    
    def add_state(self, state: DebugState) -> None:
        """Add a debug state to the session."""
        self.states.append(state)
        self.current_state_index = len(self.states) - 1
    
    def get_current_state(self) -> Optional[DebugState]:
        """Get the current debug state."""
        if 0 <= self.current_state_index < len(self.states):
            return self.states[self.current_state_index]
        return None
    
    def add_breakpoint(self, breakpoint: Breakpoint) -> None:
        """Add a breakpoint to the session."""
        self.breakpoints[breakpoint.id] = breakpoint
    
    def remove_breakpoint(self, breakpoint_id: str) -> bool:
        """Remove a breakpoint from the session."""
        if breakpoint_id in self.breakpoints:
            del self.breakpoints[breakpoint_id]
            return True
        return False
    
    def check_breakpoints(self, debug_state: DebugState) -> List[Breakpoint]:
        """Check if any breakpoints match the current state."""
        matched = []
        for breakpoint in self.breakpoints.values():
            if breakpoint.matches(debug_state):
                breakpoint.hit()
                matched.append(breakpoint)
        return matched


class DebugConsole(cmd.Cmd):
    """Interactive debugging console."""
    
    intro = "Ultra Robust XML Parser Debug Console\nType 'help' for commands.\n"
    prompt = "(debug) "
    
    def __init__(self, debug_session: DebugSession):
        super().__init__()
        self.debug_session = debug_session
        self.completekey = 'tab'
    
    def do_state(self, arg):
        """Show current parsing state: state [field_name]"""
        current_state = self.debug_session.get_current_state()
        if not current_state:
            print("No current state available")
            return
        
        if arg:
            # Show specific field
            field_name = arg.strip()
            if field_name in current_state.data:
                pprint.pprint(current_state.data[field_name])
            elif field_name in current_state.metadata:
                pprint.pprint(current_state.metadata[field_name])
            else:
                print(f"Field '{field_name}' not found")
        else:
            # Show full state
            print(f"Layer: {current_state.layer_name}")
            print(f"Step: {current_state.step_name}")
            print(f"Timestamp: {current_state.timestamp}")
            print("Data:")
            pprint.pprint(current_state.data)
            print("Metadata:")
            pprint.pprint(current_state.metadata)
    
    def do_history(self, arg):
        """Show state history: history [count]"""
        count = 10  # Default count
        if arg:
            try:
                count = int(arg.strip())
            except ValueError:
                print("Invalid count value")
                return
        
        states = self.debug_session.states[-count:]
        for i, state in enumerate(states, start=len(self.debug_session.states) - len(states)):
            marker = " -> " if i == self.debug_session.current_state_index else "    "
            print(f"{marker}[{i:3d}] {state.layer_name}.{state.step_name}")
    
    def do_goto(self, arg):
        """Go to specific state: goto <index>"""
        if not arg:
            print("Usage: goto <index>")
            return
        
        try:
            index = int(arg.strip())
            if 0 <= index < len(self.debug_session.states):
                self.debug_session.current_state_index = index
                print(f"Moved to state {index}")
                self.do_state("")
            else:
                print(f"Invalid index. Range: 0-{len(self.debug_session.states) - 1}")
        except ValueError:
            print("Invalid index value")
    
    def do_breakpoint(self, arg):
        """Manage breakpoints: breakpoint list|add|remove|enable|disable <args>"""
        if not arg:
            self.do_help("breakpoint")
            return
        
        parts = arg.split()
        command = parts[0]
        
        if command == "list":
            if not self.debug_session.breakpoints:
                print("No breakpoints set")
                return
            
            for bp_id, bp in self.debug_session.breakpoints.items():
                status = "enabled" if bp.enabled else "disabled"
                print(f"{bp_id}: {bp.breakpoint_type.value} ({status}) hits: {bp.hit_count}")
                if bp.layer_name:
                    print(f"  Layer: {bp.layer_name}")
                if bp.step_name:
                    print(f"  Step: {bp.step_name}")
        
        elif command == "add":
            if len(parts) < 3:
                print("Usage: breakpoint add <id> <type> [layer] [step]")
                return
            
            bp_id = parts[1]
            bp_type_str = parts[2]
            layer_name = parts[3] if len(parts) > 3 else None
            step_name = parts[4] if len(parts) > 4 else None
            
            try:
                bp_type = BreakpointType(bp_type_str)
                breakpoint = Breakpoint(
                    id=bp_id,
                    breakpoint_type=bp_type,
                    layer_name=layer_name,
                    step_name=step_name
                )
                self.debug_session.add_breakpoint(breakpoint)
                print(f"Added breakpoint {bp_id}")
            except ValueError:
                print(f"Invalid breakpoint type: {bp_type_str}")
        
        elif command == "remove":
            if len(parts) < 2:
                print("Usage: breakpoint remove <id>")
                return
            
            bp_id = parts[1]
            if self.debug_session.remove_breakpoint(bp_id):
                print(f"Removed breakpoint {bp_id}")
            else:
                print(f"Breakpoint {bp_id} not found")
        
        elif command in ["enable", "disable"]:
            if len(parts) < 2:
                print(f"Usage: breakpoint {command} <id>")
                return
            
            bp_id = parts[1]
            if bp_id in self.debug_session.breakpoints:
                self.debug_session.breakpoints[bp_id].enabled = (command == "enable")
                print(f"Breakpoint {bp_id} {command}d")
            else:
                print(f"Breakpoint {bp_id} not found")
        
        else:
            print(f"Unknown breakpoint command: {command}")
    
    def do_step(self, arg):
        """Enable/disable step mode: step [on|off]"""
        if not arg:
            status = "on" if self.debug_session.step_mode else "off"
            print(f"Step mode is {status}")
            return
        
        arg = arg.strip().lower()
        if arg in ["on", "true", "1"]:
            self.debug_session.step_mode = True
            print("Step mode enabled")
        elif arg in ["off", "false", "0"]:
            self.debug_session.step_mode = False
            print("Step mode disabled")
        else:
            print("Usage: step [on|off]")
    
    def do_continue(self, arg):
        """Continue execution: continue"""
        print("Continuing execution...")
        return True  # Exit the console
    
    def do_quit(self, arg):
        """Quit debugging session: quit"""
        self.debug_session.active = False
        print("Debugging session terminated")
        return True
    
    def do_export(self, arg):
        """Export debug session: export <filename>"""
        if not arg:
            print("Usage: export <filename>")
            return
        
        filename = arg.strip()
        try:
            session_data = {
                "session_id": self.debug_session.session_id,
                "debug_level": self.debug_session.debug_level.value,
                "states": [state.to_dict() for state in self.debug_session.states],
                "breakpoints": {
                    bp_id: {
                        "id": bp.id,
                        "type": bp.breakpoint_type.value,
                        "layer_name": bp.layer_name,
                        "step_name": bp.step_name,
                        "enabled": bp.enabled,
                        "hit_count": bp.hit_count
                    }
                    for bp_id, bp in self.debug_session.breakpoints.items()
                }
            }
            
            with open(filename, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            print(f"Debug session exported to {filename}")
        
        except Exception as e:
            print(f"Error exporting session: {e}")


class DebugMode:
    """Comprehensive debugging system for XML parsing.
    
    Provides state inspection, breakpoints, step-through debugging, and interactive console
    capabilities for deep analysis of parsing operations.
    
    Examples:
        Basic debugging:
        >>> debug_mode = DebugMode()
        >>> session = debug_mode.start_session("parse_debug")
        >>> with debug_mode.debug_context(session, "tokenization", "process_token"):
        ...     # Parsing code here
        ...     debug_mode.capture_state(session, {"token": current_token})
        
        Interactive debugging:
        >>> debug_mode = DebugMode()
        >>> session = debug_mode.start_session("interactive")
        >>> debug_mode.add_breakpoint(session, "token_error", BreakpointType.ERROR_CONDITION)
        >>> debug_mode.enable_interactive_mode(session)
    """
    
    def __init__(self, default_level: DebugLevel = DebugLevel.NORMAL):
        """Initialize debug mode.
        
        Args:
            default_level: Default debug verbosity level
        """
        self.default_level = default_level
        self.sessions: Dict[str, DebugSession] = {}
        self.active_session: Optional[DebugSession] = None
        self.interactive_mode = False
        self.logger = get_logger(__name__, None, "debug_mode")
    
    def start_session(self, session_id: str, debug_level: Optional[DebugLevel] = None) -> DebugSession:
        """Start a new debugging session.
        
        Args:
            session_id: Unique identifier for the session
            debug_level: Debug verbosity level for this session
            
        Returns:
            DebugSession object
        """
        level = debug_level or self.default_level
        session = DebugSession(session_id, level)
        self.sessions[session_id] = session
        self.active_session = session
        
        self.logger.info(
            "Started debug session",
            extra={
                "session_id": session_id,
                "debug_level": level.value,
                "interactive_mode": self.interactive_mode
            }
        )
        
        return session
    
    def end_session(self, session: DebugSession) -> None:
        """End a debugging session.
        
        Args:
            session: Session to end
        """
        session.active = False
        
        if self.active_session == session:
            self.active_session = None
        
        self.logger.info(
            "Ended debug session",
            extra={
                "session_id": session.session_id,
                "states_captured": len(session.states),
                "breakpoints_hit": sum(bp.hit_count for bp in session.breakpoints.values())
            }
        )
    
    def debug_context(self, session: DebugSession, layer_name: str, step_name: str) -> "DebugContext":
        """Create a debug context for a parsing step.
        
        Args:
            session: Debug session
            layer_name: Name of the processing layer
            step_name: Name of the processing step
            
        Returns:
            Context manager for debugging
        """
        return DebugContext(self, session, layer_name, step_name)
    
    def capture_state(
        self,
        session: DebugSession,
        data: Dict[str, Any],
        layer_name: str = "",
        step_name: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> DebugState:
        """Capture current parsing state for debugging.
        
        Args:
            session: Debug session
            data: State data to capture
            layer_name: Name of the processing layer
            step_name: Name of the processing step
            metadata: Additional metadata
            
        Returns:
            Created DebugState object
        """
        import time
        
        debug_state = DebugState(
            layer_name=layer_name,
            step_name=step_name,
            timestamp=time.time(),
            data=data.copy(),
            metadata=metadata or {}
        )
        
        session.add_state(debug_state)
        
        # Check for breakpoints
        matched_breakpoints = session.check_breakpoints(debug_state)
        
        # Handle breakpoint hits
        if matched_breakpoints or session.step_mode:
            self._handle_debug_break(session, debug_state, matched_breakpoints)
        
        return debug_state
    
    def add_breakpoint(
        self,
        session: DebugSession,
        breakpoint_id: str,
        breakpoint_type: BreakpointType,
        condition: Optional[Callable[[DebugState], bool]] = None,
        layer_name: Optional[str] = None,
        step_name: Optional[str] = None
    ) -> None:
        """Add a debugging breakpoint.
        
        Args:
            session: Debug session
            breakpoint_id: Unique identifier for the breakpoint
            breakpoint_type: Type of breakpoint
            condition: Custom condition function
            layer_name: Layer name filter
            step_name: Step name filter
        """
        breakpoint = Breakpoint(
            id=breakpoint_id,
            breakpoint_type=breakpoint_type,
            condition=condition,
            layer_name=layer_name,
            step_name=step_name
        )
        
        session.add_breakpoint(breakpoint)
        
        self.logger.debug(
            "Added breakpoint",
            extra={
                "session_id": session.session_id,
                "breakpoint_id": breakpoint_id,
                "breakpoint_type": breakpoint_type.value
            }
        )
    
    def enable_interactive_mode(self, session: DebugSession) -> None:
        """Enable interactive debugging mode.
        
        Args:
            session: Debug session to make interactive
        """
        self.interactive_mode = True
        session.step_mode = True
        
        self.logger.info(
            "Enabled interactive debugging mode",
            extra={"session_id": session.session_id}
        )
    
    def _handle_debug_break(
        self,
        session: DebugSession,
        debug_state: DebugState,
        matched_breakpoints: List[Breakpoint]
    ) -> None:
        """Handle a debug breakpoint or step."""
        if matched_breakpoints:
            print(f"\nBreakpoint hit: {[bp.id for bp in matched_breakpoints]}")
        elif session.step_mode:
            print(f"\nStep: {debug_state.layer_name}.{debug_state.step_name}")
        
        if self.interactive_mode:
            console = DebugConsole(session)
            try:
                console.cmdloop()
            except KeyboardInterrupt:
                print("\nDebug session interrupted")
                session.active = False
        else:
            # Non-interactive mode: just log the breakpoint
            self.logger.warning(
                "Debug breakpoint hit",
                extra={
                    "session_id": session.session_id,
                    "layer_name": debug_state.layer_name,
                    "step_name": debug_state.step_name,
                    "breakpoints": [bp.id for bp in matched_breakpoints]
                }
            )
    
    def get_session_summary(self, session: DebugSession) -> Dict[str, Any]:
        """Get a summary of the debugging session.
        
        Args:
            session: Debug session to summarize
            
        Returns:
            Summary dictionary
        """
        layer_stats = {}
        for state in session.states:
            layer = state.layer_name
            if layer not in layer_stats:
                layer_stats[layer] = {"count": 0, "steps": set()}
            layer_stats[layer]["count"] += 1
            layer_stats[layer]["steps"].add(state.step_name)
        
        # Convert sets to lists for JSON serialization
        for layer in layer_stats:
            layer_stats[layer]["steps"] = list(layer_stats[layer]["steps"])
        
        return {
            "session_id": session.session_id,
            "debug_level": session.debug_level.value,
            "active": session.active,
            "step_mode": session.step_mode,
            "states_captured": len(session.states),
            "breakpoints_configured": len(session.breakpoints),
            "breakpoints_hit": sum(bp.hit_count for bp in session.breakpoints.values()),
            "layer_statistics": layer_stats
        }


class DebugContext:
    """Context manager for debugging a parsing step."""
    
    def __init__(self, debug_mode: DebugMode, session: DebugSession, layer_name: str, step_name: str):
        self.debug_mode = debug_mode
        self.session = session
        self.layer_name = layer_name
        self.step_name = step_name
        self.start_time: Optional[float] = None
    
    def __enter__(self) -> "DebugContext":
        """Enter debug context."""
        import time
        self.start_time = time.time()
        
        # Capture entry state
        self.debug_mode.capture_state(
            self.session,
            {"action": "enter"},
            self.layer_name,
            f"{self.step_name}_enter",
            {"start_time": self.start_time}
        )
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit debug context."""
        import time
        end_time = time.time()
        
        # Capture exit state
        metadata = {
            "start_time": self.start_time,
            "end_time": end_time,
            "duration_ms": (end_time - self.start_time) * 1000 if self.start_time else 0
        }
        
        if exc_type:
            metadata["exception"] = {
                "type": exc_type.__name__,
                "message": str(exc_val),
                "traceback": traceback.format_tb(exc_tb)
            }
        
        self.debug_mode.capture_state(
            self.session,
            {"action": "exit", "success": exc_type is None},
            self.layer_name,
            f"{self.step_name}_exit",
            metadata
        )
    
    def capture(self, data: Dict[str, Any], step_suffix: str = "") -> DebugState:
        """Capture state within this context.
        
        Args:
            data: State data to capture
            step_suffix: Optional suffix for step name
            
        Returns:
            Created DebugState
        """
        step_name = f"{self.step_name}_{step_suffix}" if step_suffix else self.step_name
        return self.debug_mode.capture_state(
            self.session,
            data,
            self.layer_name,
            step_name
        )