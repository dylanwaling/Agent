"""
Live System Monitoring for Document Q&A Agent - Performance Monitor
Contains BaseMonitor, LiveMonitorGUI, and OperationEventBus for the monitoring system.
"""

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
from config.settings import paths, performance_config
from utils.system_io_helpers import read_jsonl, format_timestamp, get_gpu_info
from logic.rag_pipeline_orchestrator import DocumentPipeline

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
# EVENT BUS
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


# ============================================================================
# BASE MONITOR CLASS
# ============================================================================


class BaseMonitor:
    """
    Base class for all monitor views with common functionality.
    
    Provides reusable UI components and data management for monitor screens.
    Subclasses should override show() and on_new_operation() methods.
    """
    
    def __init__(self, parent, gui_instance):
        self.parent = parent
        self.gui = gui_instance
        self.main_frame = None
        self.widgets = {}
        self.items = []  # Store all items for this monitor
        self.displayed_count = 0  # Track what's displayed
        self.refresh_job = None  # Store periodic refresh job ID
        self.refresh_interval_ms = 2000  # Refresh every 2 seconds
        
    def create_frame(self):
        """Create the main frame for this monitor"""
        self.main_frame = ttk.Frame(self.parent, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.parent.columnconfigure(0, weight=1)
        self.parent.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(0, weight=1)
        return self.main_frame
    
    def add_back_button(self, row=0):
        """
        Add back to menu button.
        
        Args:
            row: Grid row position for the button
            
        Returns:
            Next available row number
        """
        btn = ttk.Button(self.main_frame, text="← Back to Menu", command=self.gui.show_menu)
        btn.grid(row=row, column=0, sticky=tk.W, pady=(0, 10))
        return row + 1
    
    def add_title(self, title_text, row=1):
        """
        Add title label to the monitor.
        
        Args:
            title_text: Title text to display
            row: Grid row position
            
        Returns:
            Next available row number
        """
        title = ttk.Label(self.main_frame, text=title_text, style="Title.TLabel")
        title.grid(row=row, column=0, pady=(0, 15), sticky=tk.W)
        return row + 1
    
    def add_stat_frame(self, title, stats_config, row):
        """
        Add a statistics frame with labels.
        
        Args:
            title: Frame title
            stats_config: List of tuples [(label_text, widget_key, default_value), ...]
            row: Grid row position
            
        Returns:
            Next available row number
        """
        frame = ttk.LabelFrame(self.main_frame, text=title, padding="10")
        frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        frame.columnconfigure(1, weight=1)
        
        for i, (label_text, widget_key, default_value) in enumerate(stats_config):
            ttk.Label(frame, text=f"{label_text}:").grid(row=i, column=0, sticky=tk.W, padx=(0, 10), pady=(5 if i > 0 else 0, 0))
            label = ttk.Label(frame, text=default_value)
            label.grid(row=i, column=1, sticky=tk.W, pady=(5 if i > 0 else 0, 0))
            self.widgets[widget_key] = label
        
        return row + 1
    
    def add_scrollable_text(self, title, height, row):
        """
        Add a scrollable text widget.
        
        Args:
            title: Frame title
            height: Text widget height in lines
            row: Grid row position
            
        Returns:
            Tuple of (text_widget, next_row)
        """
        frame = ttk.LabelFrame(self.main_frame, text=title, padding="10")
        frame.grid(row=row, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        self.main_frame.rowconfigure(row, weight=1)
        
        text_widget = scrolledtext.ScrolledText(frame, height=height, width=80,
                                               bg="#252526", fg="#d4d4d4",
                                               font=("Consolas", 9), wrap=tk.WORD,
                                               relief=tk.FLAT, borderwidth=0)
        text_widget.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        text_widget.config(state=tk.DISABLED)
        
        return text_widget, row + 1
    
    def update_text_widget(self, text_widget, new_lines, auto_scroll=True):
        """
        Smart update for scrollable text widgets with auto-scroll.
        
        Args:
            text_widget: The text widget to update
            new_lines: List of text lines to append
            auto_scroll: Whether to auto-scroll if at bottom
        """
        yview = text_widget.yview()
        at_bottom = yview[1] >= 0.99
        
        text_widget.config(state=tk.NORMAL)
        for line in new_lines:
            text_widget.insert(tk.END, line + "\n")
        text_widget.config(state=tk.DISABLED)
        
        if auto_scroll and at_bottom:
            text_widget.see(tk.END)
    
    def filter_operations(self, operations, keywords):
        """
        Filter operations by keywords.
        
        Args:
            operations: List of operation dictionaries
            keywords: List of keywords to filter by
            
        Returns:
            Filtered list of operations
        """
        filtered = []
        for op in operations:
            op_text = op.get('operation', '')
            if any(keyword in op_text for keyword in keywords):
                filtered.append(op)
        return filtered
    
    def _schedule_scroll_to_bottom(self):
        """
        Schedule multiple scroll attempts to ensure text widget starts at bottom.
        
        Uses multiple delayed attempts to handle timing issues with widget rendering.
        """
        def scroll():
            try:
                self.text_widget.update_idletasks()
                self.text_widget.yview_moveto(1.0)
                self.text_widget.see(tk.END)
            except:
                pass
        
        # Multiple attempts with increasing delays
        for delay in [10, 50, 150, 300]:
            self.gui.root.after(delay, scroll)
    
    def _add_item_to_display(self, item, format_func):
        """
        Add single item to display (event-driven).
        
        Args:
            item: Data item to display
            format_func: Function to format item into display lines
        """
        if not hasattr(self, 'text_widget'):
            return
        
        # Check if at bottom
        yview = self.text_widget.yview()
        at_bottom = yview[1] >= 0.99
        
        # Add to display
        self.text_widget.config(state=tk.NORMAL)
        lines = format_func(item)
        for line in lines:
            self.text_widget.insert(tk.END, line + "\n")
        self.text_widget.config(state=tk.DISABLED)
        
        # Auto-scroll if at bottom
        if at_bottom:
            self.text_widget.see(tk.END)
        
        self.displayed_count += 1
    
    def _load_initial_items(self, all_operations, extract_func, format_func, max_display=50):
        """
        Load historical items on monitor show.
        
        Args:
            all_operations: Complete list of historical operations
            extract_func: Function to extract relevant items from operations
            format_func: Function to format items for display
            max_display: Maximum number of items to display initially
        """
        self.items = extract_func(all_operations)
        
        # Display last N items
        display_items = self.items[-max_display:]
        if display_items:
            self.text_widget.config(state=tk.NORMAL)
            for item in display_items:
                lines = format_func(item)
                for line in lines:
                    self.text_widget.insert(tk.END, line + "\n")
            self.text_widget.config(state=tk.DISABLED)
            self.text_widget.see(tk.END)
        
        self.displayed_count = len(self.items)
    
    def on_new_operation(self, operation_data):
        """Called when new operation arrives (event-driven) - override in subclasses"""
        pass
    
    def show(self):
        """Override this method in subclasses"""
        raise NotImplementedError
    
    def update_stats(self):
        """Override this method in subclasses for periodic stat updates"""
        pass
    
    def start_auto_refresh(self, extract_func, format_func):
        """
        Start periodic refresh to reload data from file.
        
        Args:
            extract_func: Function to extract relevant items from operations
            format_func: Function to format items for display
        """
        self._extract_func = extract_func
        self._format_func = format_func
        self._schedule_refresh()
    
    def _schedule_refresh(self):
        """Schedule the next refresh cycle"""
        if self.refresh_job:
            self.gui.root.after_cancel(self.refresh_job)
        self.refresh_job = self.gui.root.after(self.refresh_interval_ms, self._do_refresh)
    
    def _do_refresh(self):
        """Perform periodic refresh of data from file"""
        try:
            # Reload operations from file
            all_operations = self.gui.load_operation_history()
            new_items = self._extract_func(all_operations)
            
            # Check if there are new items
            if len(new_items) > len(self.items):
                # Add new items to display
                new_count = len(new_items) - len(self.items)
                for item in new_items[-new_count:]:
                    self._add_item_to_display(item, self._format_func)
                
                # Update items list BEFORE updating stats
                self.items = new_items
                
                # Update stats if the method exists
                if hasattr(self, '_update_stats'):
                    self._update_stats()
            
            # Always update stats to keep them current even if no new items
            elif hasattr(self, '_update_stats'):
                self._update_stats()
        except Exception as e:
            logger.error(f"Error during monitor refresh: {e}")
        
        # Schedule next refresh
        self._schedule_refresh()
    
    def stop_auto_refresh(self):
        """Stop the periodic refresh"""
        if self.refresh_job:
            self.gui.root.after_cancel(self.refresh_job)
            self.refresh_job = None


# ============================================================================
# MAIN GUI APPLICATION
# ============================================================================

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
        from monitor.operations.general_info_monitor import GeneralInfoMonitor
        from monitor.operations.question_input_monitor import QuestionInputMonitor
        from monitor.operations.embedding_query_monitor import EmbeddingQueryMonitor
        from monitor.operations.faiss_search_monitor import FAISSSearchMonitor
        
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
