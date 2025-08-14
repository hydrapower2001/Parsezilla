"""Structured logging utilities for ultra-robust XML parsing.

This module provides correlation-aware logging with performance optimization
and structured output for debugging and monitoring.
"""

import logging
import time
from typing import Any, Dict, Optional


class CorrelationLogger:
    """Logger that automatically includes correlation ID and component information."""
    
    def __init__(
        self, 
        name: str, 
        correlation_id: Optional[str] = None,
        component: Optional[str] = None
    ) -> None:
        """Initialize correlation logger.
        
        Args:
            name: Logger name (typically __name__)
            correlation_id: Optional correlation ID for request tracking
            component: Component name for structured logging
        """
        self.logger = logging.getLogger(name)
        self.correlation_id = correlation_id
        self.component = component or name.split('.')[-1]
    
    def _get_extra(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get logging extra data with correlation info.
        
        Args:
            extra: Additional extra data
            
        Returns:
            Combined extra data with correlation info
        """
        combined_extra = {
            "component": self.component,
            "correlation_id": self.correlation_id,
        }
        
        if extra:
            combined_extra.update(extra)
            
        return combined_extra
    
    def debug(
        self, 
        message: str, 
        extra: Optional[Dict[str, Any]] = None,
        exc_info: bool = False
    ) -> None:
        """Log debug message with correlation info."""
        self.logger.debug(message, extra=self._get_extra(extra), exc_info=exc_info)
    
    def info(
        self, 
        message: str, 
        extra: Optional[Dict[str, Any]] = None,
        exc_info: bool = False
    ) -> None:
        """Log info message with correlation info."""
        self.logger.info(message, extra=self._get_extra(extra), exc_info=exc_info)
    
    def warning(
        self, 
        message: str, 
        extra: Optional[Dict[str, Any]] = None,
        exc_info: bool = False
    ) -> None:
        """Log warning message with correlation info."""
        self.logger.warning(message, extra=self._get_extra(extra), exc_info=exc_info)
    
    def error(
        self, 
        message: str, 
        extra: Optional[Dict[str, Any]] = None,
        exc_info: bool = True
    ) -> None:
        """Log error message with correlation info."""
        self.logger.error(message, extra=self._get_extra(extra), exc_info=exc_info)
    
    def critical(
        self, 
        message: str, 
        extra: Optional[Dict[str, Any]] = None,
        exc_info: bool = True
    ) -> None:
        """Log critical message with correlation info."""
        self.logger.critical(message, extra=self._get_extra(extra), exc_info=exc_info)
    
    def exception(
        self, 
        message: str, 
        extra: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log exception message with correlation info and traceback."""
        self.logger.exception(message, extra=self._get_extra(extra))


def get_logger(
    name: str, 
    correlation_id: Optional[str] = None,
    component: Optional[str] = None
) -> CorrelationLogger:
    """Get a correlation-aware logger instance.
    
    Args:
        name: Logger name (typically __name__)
        correlation_id: Optional correlation ID for request tracking  
        component: Component name for structured logging
        
    Returns:
        CorrelationLogger instance
    """
    return CorrelationLogger(name, correlation_id, component)