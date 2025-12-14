# ============================================================================
# ANALYTICS & LOGGING MODULE
# ============================================================================
# Purpose:
#   Comprehensive logging and monitoring system for pipeline operations
#   Handles operation logging, status updates, and event bus integration
# ============================================================================

import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

# Local imports
from Config.settings import paths, logging_config
from Utils.system_io_helpers import write_json_atomic, append_jsonl

logger = logging.getLogger(__name__)

# Event bus integration (lazy import to avoid circular dependency)
_event_bus = None
_live_monitoring_enabled = False


def _init_event_bus():
    """Initialize event bus for live monitoring (lazy loading)"""
    global _event_bus, _live_monitoring_enabled
    if _event_bus is None:
        try:
            from Program.performance_monitor import event_bus
            _event_bus = event_bus
            _live_monitoring_enabled = True
        except ImportError:
            _live_monitoring_enabled = False
            _event_bus = None


class AnalyticsLogger:
    """
    Analytics and logging handler for document pipeline operations.
    
    Provides comprehensive logging with:
    - Operation history tracking (JSONL format)
    - Real-time status updates
    - Event bus integration for live monitoring
    """
    
    def __init__(self, status_file: Path = None):
        """
        Initialize analytics logger.
        
        Args:
            status_file: Path to status file for real-time monitoring
        """
        self.status_file = status_file or paths.STATUS_FILE
        _init_event_bus()
    
    def log_operation(self, operation_type: str, operation: str, 
                     metadata: Optional[Dict] = None, status: str = "THINKING"):
        """
        Comprehensive logging system for all pipeline operations.
        
        Args:
            operation_type: Type of operation (question_input, embedding_query, faiss_search, etc.)
            operation: Human-readable operation description
            metadata: Additional structured data for the operation
            status: Current pipeline status (THINKING, IDLE, ERROR, PROCESSING)
        """
        try:
            timestamp = time.time()
            operation_id = f"{operation_type}_{timestamp}"
            
            # Standardized log entry with rich metadata
            log_entry = {
                "timestamp": timestamp,
                "datetime": datetime.fromtimestamp(timestamp).isoformat(),
                "operation_type": operation_type,
                "operation": operation,
                "operation_id": operation_id,
                "status": status,
                "metadata": metadata or {}
            }
            
            # Write to operation history (JSONL format for easy parsing)
            history_file = paths.HISTORY_FILE
            append_jsonl(history_file, log_entry)
            
            # Publish to event bus for real-time monitoring (push-based)
            if _live_monitoring_enabled and _event_bus:
                _event_bus.publish(log_entry)
            
            # Also update current status for real-time monitoring
            self._update_status_only(status, operation, metadata)
            
            logger.debug(f"[{operation_type.upper()}] {operation[:80]}")
            
        except Exception as e:
            logger.error(f"Failed to log operation: {e}")
    
    def _update_status_only(self, status: str, operation: str, metadata: Optional[Dict] = None):
        """
        Update status file only (used internally by log_operation).
        
        Args:
            status: Status type (IDLE, THINKING, PROCESSING, ERROR)
            operation: Human-readable operation description
            metadata: Additional structured data
        """
        try:
            timestamp = time.time()
            status_data = {
                "status": status,
                "operation": operation,
                "timestamp": timestamp,
                "operation_id": f"{status}_{timestamp}",
                "metadata": metadata or {}
            }
            
            # Write status atomically to prevent partial reads
            write_json_atomic(self.status_file, status_data)
            
        except Exception as e:
            logger.error(f"Failed to update status: {e}")
    
    def update_status(self, status: str, operation: str, metadata: Optional[Dict] = None):
        """
        Backward compatibility wrapper - use log_operation for new code.
        
        Args:
            status: Status type
            operation: Operation description
            metadata: Additional data
        """
        self.log_operation(
            operation_type="general",
            operation=operation,
            metadata=metadata,
            status=status
        )
