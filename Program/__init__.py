# ============================================================================
# PROGRAM PACKAGE
# ============================================================================
# Purpose:
#   Main application package for the Document Q&A Agent
#   Desktop application with document upload, Q&A, and monitoring
# ============================================================================

from .rag_desktop_app import DocumentQAApp, get_pipeline, get_documents
from .performance_monitor import LiveMonitorGUI, event_bus

__all__ = ['DocumentQAApp', 'get_pipeline', 'get_documents', 'LiveMonitorGUI', 'event_bus']
