"""
Live System Monitoring for Document Q&A Agent - GUI Version
Task-manager-style real-time display with scrollable window interface
"""

# ============================================================================
# CONSTANTS & IMPORTS
# ============================================================================

# Standard library imports
import os
import sys
import time
import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from collections import deque
from queue import Queue

# Third-party imports
import psutil
import tkinter as tk
from tkinter import ttk, scrolledtext

# Local imports - configuration and utilities
from Config.settings import paths, performance_config
from Utils.system_io_helpers import read_jsonl, format_timestamp, get_gpu_info
from Logic.rag_pipeline_orchestrator import DocumentPipeline

# Monitor imports
from Monitor.general_info_monitor import GeneralInfoMonitor
from Monitor.question_input_monitor import QuestionInputMonitor
from Monitor.embedding_query_monitor import EmbeddingQueryMonitor
from Monitor.faiss_search_monitor import FAISSSearchMonitor

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# File paths from configuration
STATUS_FILE = paths.STATUS_FILE
HISTORY_FILE = paths.HISTORY_FILE
INDEX_PATH = str(paths.FAISS_INDEX_FILE)
DOCS_PATH = str(paths.DOCS_DIR)

# UI Configuration from config
MAX_OPERATIONS_DISPLAY = performance_config.MAX_OPERATIONS_DISPLAY
QUEUE_PROCESS_INTERVAL_MS = performance_config.QUEUE_PROCESS_INTERVAL_MS
STATUS_UPDATE_INTERVAL_MS = performance_config.STATUS_UPDATE_INTERVAL_MS


# ============================================================================
# DATA MODELS & CLASSES
# ============================================================================

class OperationEventBus:
    """
    Thread-safe event bus for pushing operations to monitors in real-time.
    
    Provides publish-subscribe pattern for operation events across the application.
    """
    
    def __init__(self):
        """Initialize the event bus with empty subscriber list and thread lock."""
        self.subscribers = []
        self.lock = threading.Lock()
    
    def subscribe(self, callback):
        """
        Subscribe to operation events.
        
        Args:
            callback: Function to call when operations are published
        """
        with self.lock:
            self.subscribers.append(callback)
    
    def unsubscribe(self, callback):
        """
        Unsubscribe from operation events.
        
        Args:
            callback: Function to remove from subscribers
        """
        with self.lock:
            if callback in self.subscribers:
                self.subscribers.remove(callback)
    
    def publish(self, operation_data):
        """
        Publish operation to all subscribers (called from backend thread).
        
        Args:
            operation_data: Dictionary containing operation information
        """
        with self.lock:
            for callback in self.subscribers[:]:  # Copy to avoid modification during iteration
                try:
                    callback(operation_data)
                except Exception as e:
                    logger.error(f"Error in subscriber callback: {e}")


# Global event bus instance
event_bus = OperationEventBus()


class LiveMonitorGUI:
    """
    Main GUI application for live system monitoring.
    
    Provides task-manager-style interface with multiple monitor views
    for tracking document Q&A pipeline operations in real-time.
    """
    
    def __init__(self):
        """Initialize the live monitor GUI with default state and event handling."""
        # Pipeline state
        self.pipeline = None
        self.status = "IDLE"
        self.last_operation = "System started"
        self.operation_count = 0
        self.start_time = time.time()
        self.running = True
        
        # File paths
        self.status_file = STATUS_FILE
        self.history_file = HISTORY_FILE
        
        # Operation tracking
        self.operation_history = deque(maxlen=MAX_OPERATIONS_DISPLAY)
        
        # Current view and active monitor
        self.current_view = "menu"  # Start at main menu
        self.active_monitor = None  # Currently active monitor instance
        
        # Event-driven operation queue (thread-safe)
        self.operation_queue = Queue()
        
        # Load historical data on startup
        self.all_operations = self._load_historical_operations()
        self.general_info_displayed_count = len(self.all_operations)  # Track what's displayed in General Info
        
        # Create the GUI
        self.root = tk.Tk()
        self.root.title("Document Q&A Agent - Live Monitor")
        self.root.geometry("900x700")
        self.root.configure(bg="#1e1e1e")
        
        # Configure styles
        self.setup_styles()
        
        # Create main container that we'll switch between
        self.main_container = ttk.Frame(self.root)
        self.main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Show menu initially
        self.show_menu()
        
        # Subscribe to operation events
        event_bus.subscribe(self._on_operation_event)
        
        # Initialize pipeline in background
        init_thread = threading.Thread(target=self.init_pipeline_async, daemon=True)
        init_thread.start()
        
        # Start GUI update processor (processes queued operations on GUI thread)
        self._process_operation_queue()
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    # ========================================================================
    # UI SETUP & STYLING
    # ========================================================================
    
    def setup_styles(self):
        """
        Setup ttk styles for dark theme.
        
        Configures colors, fonts, and button styles for the entire application.
        """
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        bg_color = "#1e1e1e"
        fg_color = "#ffffff"
        accent_color = "#007acc"
        
        style.configure(".", background=bg_color, foreground=fg_color)
        style.configure("TFrame", background=bg_color)
        style.configure("TLabel", background=bg_color, foreground=fg_color, font=("Consolas", 10))
        style.configure("Title.TLabel", font=("Consolas", 14, "bold"), foreground=accent_color)
        style.configure("Header.TLabel", font=("Consolas", 11, "bold"), foreground="#4ec9b0")
        style.configure("Status.TLabel", font=("Consolas", 12, "bold"))
        
        # Button styles
        style.configure("Menu.TButton", font=("Consolas", 11, "bold"), padding=20)
        style.map("Menu.TButton",
                 background=[('active', '#007acc'), ('!active', '#2d2d30')],
                 foreground=[('active', '#ffffff'), ('!active', '#ffffff')])
    
    # ========================================================================
    # NAVIGATION & VIEW MANAGEMENT
    # ========================================================================
    
    def clear_container(self):
        """
        Clear the main container.
        
        Destroys all child widgets in the main container frame.
        Stops auto-refresh on active monitor if present.
        """
        # Stop auto-refresh on current monitor
        if self.active_monitor and hasattr(self.active_monitor, 'stop_auto_refresh'):
            self.active_monitor.stop_auto_refresh()
        
        for widget in self.main_container.winfo_children():
            widget.destroy()
    
    def show_menu(self):
        """
        Show the main menu with monitor options.
        
        Displays a grid of buttons for accessing different monitor views.
        """
        self.clear_container()
        self.current_view = "menu"
        
        # Create menu frame
        menu_frame = ttk.Frame(self.main_container, padding="20")
        menu_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.main_container.columnconfigure(0, weight=1)
        self.main_container.rowconfigure(0, weight=1)
        menu_frame.columnconfigure(0, weight=1)
        menu_frame.columnconfigure(1, weight=1)
        menu_frame.columnconfigure(2, weight=1)
        
        # Title
        title = ttk.Label(menu_frame, text="LIVE MONITOR - SELECT VIEW", style="Title.TLabel")
        title.grid(row=0, column=0, columnspan=3, pady=(0, 30))
        
        # Grid of monitor buttons (3 columns) - Data Flow Pipeline
        monitors = [
            # === SYSTEM OVERVIEW ===
            ("General Info", lambda: self.show_generic_monitor("General Info", GeneralInfoMonitor)),
            
            # === QUESTION ANSWERING FLOW (in order) ===
            ("Question Input", lambda: self.show_generic_monitor("Question Input", QuestionInputMonitor)),
            ("Embedding Query", lambda: self.show_generic_monitor("Embedding Query", EmbeddingQueryMonitor)),
            ("FAISS Search", lambda: self.show_generic_monitor("FAISS Search", FAISSSearchMonitor)),
            ("Relevance Filter", lambda: self.show_placeholder("Relevance Filter")),
            ("Context Builder", lambda: self.show_placeholder("Context Builder")),
            ("Prompt Assembly", lambda: self.show_placeholder("Prompt Assembly")),
            ("Ollama LLM", lambda: self.show_placeholder("Ollama LLM")),
            ("Response Stream", lambda: self.show_placeholder("Response Stream")),
            
            # === DOCUMENT PROCESSING FLOW (in order) ===
            ("File Upload", lambda: self.show_placeholder("File Upload")),
            ("Docling Parser", lambda: self.show_placeholder("Docling Parser")),
            ("Text Splitter", lambda: self.show_placeholder("Text Splitter")),
            ("Embedding Gen", lambda: self.show_placeholder("Embedding Gen")),
            ("FAISS Indexing", lambda: self.show_placeholder("FAISS Indexing")),
            ("Index Storage", lambda: self.show_placeholder("Index Storage")),
            
            # === SYSTEM MONITORING ===
            ("Status Manager", lambda: self.show_placeholder("Status Manager")),
            ("Operation History", lambda: self.show_placeholder("Operation History")),
            ("Error Tracking", lambda: self.show_placeholder("Error Tracking")),
            ("Performance", lambda: self.show_placeholder("Performance")),
            ("GPU Monitor", lambda: self.show_placeholder("GPU Monitor")),
            ("Reserved", lambda: self.show_placeholder("Reserved")),
        ]
        
        row = 1
        col = 0
        for name, command in monitors:
            btn = ttk.Button(menu_frame, text=name, command=command, style="Menu.TButton", width=20)
            btn.grid(row=row, column=col, padx=10, pady=10, sticky=(tk.W, tk.E))
            col += 1
            if col > 2:  # 3 columns
                col = 0
                row += 1
    
    def show_generic_monitor(self, name, monitor_class):
        """
        Generic method to show any monitor view.
        
        Args:
            name: Display name for the monitor
            monitor_class: Monitor class to instantiate and display
        """
        self.clear_container()
        self.current_view = name.lower().replace(" ", "_")
        
        monitor = monitor_class(self.main_container, self)
        monitor.show()
        
        # Store monitor for updates
        self.active_monitor = monitor
    
    def show_placeholder(self, name):
        """
        Show placeholder for monitors not yet implemented.
        
        Args:
            name: Name of the monitor view
        """
        self.clear_container()
        self.current_view = name
        self.active_monitor = None
        
        # Create placeholder frame
        placeholder_frame = ttk.Frame(self.main_container, padding="20")
        placeholder_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.main_container.columnconfigure(0, weight=1)
        self.main_container.rowconfigure(0, weight=1)
        
        # Back button
        back_btn = ttk.Button(placeholder_frame, text="← Back to Menu", command=self.show_menu)
        back_btn.grid(row=0, column=0, sticky=tk.W, pady=(0, 20))
        
        # Title
        title = ttk.Label(placeholder_frame, text=f"{name} Monitor", style="Title.TLabel")
        title.grid(row=1, column=0, pady=(0, 20))
        
        # Placeholder message
        msg = ttk.Label(placeholder_frame, text=f"Monitor view for '{name}' will be implemented here.", 
                       font=("Consolas", 12))
        msg.grid(row=2, column=0)
    
    # ========================================================================
    # EVENT-DRIVEN SYSTEM METHODS
    # ========================================================================
    
    def _load_historical_operations(self):
        """
        Load existing operations on startup.
        
        Reads historical operations from JSONL file for initial display.
        
        Returns:
            List of operation dictionaries
        """
        try:
            return read_jsonl(self.history_file)
        except Exception as e:
            logger.error(f"Error loading historical operations: {e}")
            return []
    
    def _on_operation_event(self, operation_data):
        """
        Callback when new operation is published (called from backend thread).
        
        Args:
            operation_data: Dictionary containing operation information
        """
        # Add to queue for GUI thread processing
        self.operation_queue.put(operation_data)
    
    def _process_operation_queue(self):
        """
        Process queued operations on GUI thread.
        
        Processes all pending operations from the queue and updates
        the active monitor and operation count. Scheduled every 50ms.
        """
        try:
            while not self.operation_queue.empty():
                operation_data = self.operation_queue.get_nowait()
                
                # Add to all operations list
                self.all_operations.append(operation_data)
                self.operation_count = len(self.all_operations)
                
                # Update active monitor if exists
                if self.active_monitor:
                    self.active_monitor.on_new_operation(operation_data)
        except Exception as e:
            logger.error(f"Error processing operation queue: {e}")
        
        # Schedule next check (every 50ms for responsive UI)
        self.root.after(QUEUE_PROCESS_INTERVAL_MS, self._process_operation_queue)
    
    # ========================================================================
    # DATA ACCESS & STATUS
    # ========================================================================
    
    def load_operation_history(self):
        """
        Return all operations (for monitors that need full history).
        Reloads from file to ensure fresh data is available.
        
        Returns:
            List of all operation dictionaries
        """
        try:
            # Reload from file to get latest operations
            fresh_operations = read_jsonl(self.history_file)
            # Update our cached list with any operations we might have missed
            self.all_operations = fresh_operations
            self.operation_count = len(self.all_operations)
            return self.all_operations
        except Exception as e:
            logger.error(f"Error reloading operation history: {e}")
            return self.all_operations
    
    def read_status(self):
        """
        Read current status from file.
        
        Returns:
            Tuple of (status, last_operation, timestamp, operation_id)
        """
        try:
            if self.status_file.exists():
                with open(self.status_file, 'r') as f:
                    data = json.load(f)
                    return (data.get("status", "IDLE"),
                           data.get("operation", "Unknown"),
                           data.get("timestamp", 0),
                           data.get("operation_id", ""))
        except Exception as e:
            logger.error(f"Error reading status: {e}")
        
        return ("IDLE", "Unknown", 0, "")
    
    def get_gpu_info(self):
        """
        Get GPU information if available.
        
        Returns:
            String with GPU name and memory info, or error message
        """
        gpu_info_dict = get_gpu_info()
        return gpu_info_dict.get('message', 'GPU info unavailable')
    
    def format_uptime(self, seconds):
        """
        Format uptime as HH:MM:SS.
        
        Args:
            seconds: Number of seconds to format
            
        Returns:
            Formatted time string (HH:MM:SS)
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    # ========================================================================
    # BACKGROUND INITIALIZATION
    # ========================================================================
    
    def init_pipeline_async(self):
        """
        Initialize pipeline in background.
        
        Runs in a separate thread to avoid blocking the GUI.
        """
        try:
            self.pipeline = DocumentPipeline()
            logger.info("Pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
    
    def on_closing(self):
        """
        Handle window close event.
        
        Cleans up resources and shuts down threads gracefully.
        """
        self.running = False
        event_bus.unsubscribe(self._on_operation_event)
        self.root.destroy()
    
    def run(self):
        """
        Start the GUI application.
        
        Enters the Tkinter main event loop.
        """
        self.root.mainloop()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    monitor = LiveMonitorGUI()
    monitor.run()
