"""
Live System Monitoring for Document Q&A Agent - GUI Version
Task-manager-style real-time display with scrollable window interface
"""

import os
import sys
import time
import threading
import json
import logging
from datetime import datetime
import psutil
from pathlib import Path
from backend_logic import DocumentPipeline
import tkinter as tk
from tkinter import ttk, scrolledtext
from collections import deque
from queue import Queue

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# EVENT SYSTEM - Push-based data delivery
# ============================================================================

class OperationEventBus:
    """Thread-safe event bus for pushing operations to monitors in real-time"""
    
    def __init__(self):
        self.subscribers = []
        self.lock = threading.Lock()
    
    def subscribe(self, callback):
        """Subscribe to operation events"""
        with self.lock:
            self.subscribers.append(callback)
    
    def unsubscribe(self, callback):
        """Unsubscribe from operation events"""
        with self.lock:
            if callback in self.subscribers:
                self.subscribers.remove(callback)
    
    def publish(self, operation_data):
        """Publish operation to all subscribers (called from backend thread)"""
        with self.lock:
            for callback in self.subscribers[:]:  # Copy to avoid modification during iteration
                try:
                    callback(operation_data)
                except Exception as e:
                    logger.error(f"Error in subscriber callback: {e}")

# Global event bus instance
event_bus = OperationEventBus()

# ============================================================================
# BASE MONITOR CLASS - Reusable for all monitor views
# ============================================================================

class BaseMonitor:
    """Base class for all monitor views with common functionality"""
    
    def __init__(self, parent, gui_instance):
        self.parent = parent
        self.gui = gui_instance
        self.main_frame = None
        self.widgets = {}
        self.items = []  # Store all items for this monitor
        self.displayed_count = 0  # Track what's displayed
        
    def create_frame(self):
        """Create the main frame for this monitor"""
        self.main_frame = ttk.Frame(self.parent, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.parent.columnconfigure(0, weight=1)
        self.parent.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(0, weight=1)
        return self.main_frame
    
    def add_back_button(self, row=0):
        """Add back to menu button"""
        btn = ttk.Button(self.main_frame, text="← Back to Menu", command=self.gui.show_menu)
        btn.grid(row=row, column=0, sticky=tk.W, pady=(0, 10))
        return row + 1
    
    def add_title(self, title_text, row=1):
        """Add title label"""
        title = ttk.Label(self.main_frame, text=title_text, style="Title.TLabel")
        title.grid(row=row, column=0, pady=(0, 15), sticky=tk.W)
        return row + 1
    
    def add_stat_frame(self, title, stats_config, row):
        """
        Add a statistics frame with labels
        stats_config: list of tuples [(label_text, widget_key, default_value), ...]
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
        """Add a scrollable text widget"""
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
        """Smart update for scrollable text widgets"""
        yview = text_widget.yview()
        at_bottom = yview[1] >= 0.99
        
        text_widget.config(state=tk.NORMAL)
        for line in new_lines:
            text_widget.insert(tk.END, line + "\n")
        text_widget.config(state=tk.DISABLED)
        
        if auto_scroll and at_bottom:
            text_widget.see(tk.END)
    
    def filter_operations(self, operations, keywords):
        """Filter operations by keywords"""
        filtered = []
        for op in operations:
            op_text = op.get('operation', '')
            if any(keyword in op_text for keyword in keywords):
                filtered.append(op)
        return filtered
    
    def _schedule_scroll_to_bottom(self):
        """Schedule multiple scroll attempts to ensure text widget starts at bottom"""
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
        """Add single item to display (event-driven)"""
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
        """Load historical items on monitor show"""
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


# ============================================================================
# MAIN GUI CLASS
# ============================================================================

class LiveMonitorGUI:
    def __init__(self):
        self.pipeline = None
        self.status = "IDLE"
        self.last_operation = "System started"
        self.operation_count = 0
        self.start_time = time.time()
        self.running = True
        self.status_file = Path("data/pipeline_status.json")
        self.history_file = Path("data/operation_history.jsonl")
        self.operation_history = deque(maxlen=50)  # Keep last 50 operations
        
        # Current view and active monitor
        self.current_view = "menu"  # Start at main menu
        self.active_monitor = None  # Currently active monitor instance
        
        # Event-driven operation queue (thread-safe)
        self.operation_queue = Queue()
        
        # Load historical data on startup
        self.all_operations = self._load_historical_operations()
        
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
        
        # Start file watcher thread (watches for new operations from file)
        self.watcher_thread = threading.Thread(target=self._file_watcher_loop, daemon=True)
        self.watcher_thread.start()
        
        # Start GUI update processor (processes queued operations on GUI thread)
        self._process_operation_queue()
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def setup_styles(self):
        """Setup ttk styles for dark theme"""
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
    
    def clear_container(self):
        """Clear the main container"""
        for widget in self.main_container.winfo_children():
            widget.destroy()
    
    def show_menu(self):
        """Show the main menu with monitor options"""
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
            ("General Info", self.show_general_info),
            
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
        """Generic method to show any monitor view"""
        self.clear_container()
        self.current_view = name.lower().replace(" ", "_")
        
        monitor = monitor_class(self.main_container, self)
        monitor.show()
        
        # Store monitor for updates
        self.active_monitor = monitor
    
    def show_placeholder(self, name):
        """Show placeholder for monitors not yet implemented"""
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
        """Load existing operations on startup"""
        operations = []
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            operations.append(json.loads(line))
        except Exception as e:
            logger.error(f"Error loading historical operations: {e}")
        return operations
    
    def _on_operation_event(self, operation_data):
        """Callback when new operation is published (called from backend thread)"""
        # Add to queue for GUI thread processing
        self.operation_queue.put(operation_data)
    
    def _file_watcher_loop(self):
        """Watch file for new operations (for operations not using event bus)"""
        last_size = 0
        if self.history_file.exists():
            last_size = self.history_file.stat().st_size
        
        while self.running:
            try:
                if self.history_file.exists():
                    current_size = self.history_file.stat().st_size
                    if current_size > last_size:
                        # File grew, read new lines
                        with open(self.history_file, 'r', encoding='utf-8') as f:
                            f.seek(last_size)
                            for line in f:
                                if line.strip():
                                    operation_data = json.loads(line)
                                    self.operation_queue.put(operation_data)
                        last_size = current_size
                
                time.sleep(0.1)  # Check every 100ms
            except Exception as e:
                logger.error(f"File watcher error: {e}")
                time.sleep(1)
    
    def _process_operation_queue(self):
        """Process queued operations on GUI thread"""
        try:
            while not self.operation_queue.empty():
                operation_data = self.operation_queue.get_nowait()
                
                # Add to all operations list
                self.all_operations.append(operation_data)
                self.operation_count = len(self.all_operations)
                
                # Update general info if visible
                if self.current_view == "general_info":
                    self._update_general_info_with_operation(operation_data)
                
                # Update active monitor if exists
                if self.active_monitor:
                    self.active_monitor.on_new_operation(operation_data)
        except Exception as e:
            logger.error(f"Error processing operation queue: {e}")
        
        # Schedule next check (every 50ms for responsive UI)
        self.root.after(50, self._process_operation_queue)
    
    def load_operation_history(self):
        """Return all operations (for monitors that need full history)"""
        return self.all_operations
    
    def read_status(self):
        """Read current status from file"""
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
        """Get GPU information if available"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                gpu_memory_used = torch.cuda.memory_allocated(0) / (1024**3)
                return f"{gpu_name} ({gpu_memory_used:.1f}/{gpu_memory:.1f} GB)"
            return "No GPU detected"
        except:
            return "GPU info unavailable"
    
    def format_uptime(self, seconds):
        """Format uptime as HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def _update_general_info_with_operation(self, operation_data):
        """Update general info view with new operation (event-driven)"""
        if not hasattr(self, 'operations_text'):
            return
        
        # Format operation
        op_time = datetime.fromtimestamp(operation_data.get('timestamp', 0)).strftime('%H:%M:%S')
        formatted = f"[{op_time}] {operation_data.get('operation', 'Unknown')}"
        self.operation_history.append(formatted)
        
        # Update GUI
        self.count_label.config(text=str(self.operation_count))
        
        # Check if user is at bottom
        yview = self.operations_text.yview()
        at_bottom = yview[1] >= 0.99
        
        self.operations_text.config(state=tk.NORMAL)
        self.operations_text.insert(tk.END, formatted + "\n")
        self.operations_text.config(state=tk.DISABLED)
        
        if at_bottom:
            self.operations_text.see(tk.END)
    
    def update_general_info(self):
        """Update General Info view (periodic updates for status/time only)"""
        # Read current status
        status, last_op, timestamp, operation_id = self.read_status()
        
        # Update status
        self.status = status
        self.last_operation = last_op
        
        # Update status label with color
        status_colors = {
            "IDLE": "#4ec9b0",      # Green
            "THINKING": "#dcdcaa",   # Yellow
            "ERROR": "#f48771",      # Red
            "PROCESSING": "#569cd6"  # Blue
        }
        self.status_label.config(text=status, foreground=status_colors.get(status, "#ffffff"))
        self.operation_label.config(text=last_op)
        
        # Update pipeline status
        index_path = "data/index/faiss_index/index.faiss"
        if os.path.exists(index_path):
            index_size = os.path.getsize(index_path) / 1024  # KB
            self.index_label.config(text=f"Loaded ({index_size:.1f} KB)", foreground="#4ec9b0")
        else:
            self.index_label.config(text="Not found", foreground="#f48771")
        
        docs_path = "data/documents"
        if os.path.exists(docs_path):
            doc_files = [f for f in os.listdir(docs_path) if f.endswith(('.pdf', '.docx', '.txt', '.md'))]
            self.docs_label.config(text=f"{len(doc_files)} files in {docs_path}", foreground="#4ec9b0")
        else:
            self.docs_label.config(text="Folder not found", foreground="#f48771")
        
        # Update runtime info
        uptime = time.time() - self.start_time
        self.uptime_label.config(text=self.format_uptime(uptime))
        self.time_label.config(text=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        # Update status bar with last update time
        if timestamp > 0:
            age = time.time() - timestamp
            self.statusbar.config(text=f"Monitor running | Event-driven | Last status: {age:.1f}s ago")
        
        # Schedule next update (1 second for status/time updates)
        if self.current_view == "general_info":
            self.root.after(1000, self.update_general_info)
    
    # ========================================================================
    # BACKGROUND INITIALIZATION
    # ========================================================================
    
    def init_pipeline_async(self):
        """Initialize pipeline in background"""
        try:
            self.pipeline = DocumentPipeline()
            logger.info("Pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
    
    def on_closing(self):
        """Handle window close event"""
        self.running = False
        event_bus.unsubscribe(self._on_operation_event)
        self.root.destroy()
    
    def run(self):
        """Start the GUI application"""
        self.root.mainloop()
    
    # ========================================================================
    # MONITOR VIEW: GENERAL INFO
    # ========================================================================
    
    def show_general_info(self):
        """Show the general info monitor (original view)"""
        # Clear current container
        for widget in self.main_container.winfo_children():
            widget.destroy()
        
        self.current_view = "general_info"
        
        # Create the original widgets in a new frame
        self.create_general_info_widgets()
        
        # Populate with historical operations
        if self.all_operations:
            self.operations_text.config(state=tk.NORMAL)
            for op in self.all_operations[-50:]:  # Last 50 operations
                op_time = datetime.fromtimestamp(op.get('timestamp', 0)).strftime('%H:%M:%S')
                formatted = f"[{op_time}] {op.get('operation', 'Unknown')}"
                self.operation_history.append(formatted)
                self.operations_text.insert(tk.END, formatted + "\n")
            self.operations_text.config(state=tk.DISABLED)
            self.operations_text.see(tk.END)
        
        # Start periodic updates for status/time
        self.root.after(1000, self.update_general_info)
        
    def create_general_info_widgets(self):
        """Create all GUI widgets for General Info view"""
        # Main container with padding
        main_frame = ttk.Frame(self.main_container, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.main_container.columnconfigure(0, weight=1)
        self.main_container.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        
        # Back button
        back_btn = ttk.Button(main_frame, text="← Back to Menu", command=self.show_menu)
        back_btn.grid(row=0, column=0, sticky=tk.W, pady=(0, 10))
        
        # Title
        title_label = ttk.Label(main_frame, text="GENERAL INFO - LIVE SYSTEM MONITOR", 
                               style="Title.TLabel")
        title_label.grid(row=1, column=0, pady=(0, 15), sticky=tk.W)
        
        # Process Status Section
        status_frame = ttk.LabelFrame(main_frame, text="PROCESS STATUS", padding="10")
        status_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        status_frame.columnconfigure(1, weight=1)
        
        ttk.Label(status_frame, text="Status:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.status_label = ttk.Label(status_frame, text="IDLE", style="Status.TLabel", foreground="#4ec9b0")
        self.status_label.grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(status_frame, text="Last Operation:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(5, 0))
        self.operation_label = ttk.Label(status_frame, text="System started")
        self.operation_label.grid(row=1, column=1, sticky=tk.W, pady=(5, 0))
        
        ttk.Label(status_frame, text="Operations Count:").grid(row=2, column=0, sticky=tk.W, padx=(0, 10), pady=(5, 0))
        self.count_label = ttk.Label(status_frame, text="0")
        self.count_label.grid(row=2, column=1, sticky=tk.W, pady=(5, 0))
        
        # Recent Operations Section (Scrollable)
        ops_frame = ttk.LabelFrame(main_frame, text="RECENT OPERATIONS", padding="10")
        ops_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        ops_frame.columnconfigure(0, weight=1)
        ops_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)
        
        self.operations_text = scrolledtext.ScrolledText(ops_frame, height=8, width=80, 
                                                         bg="#252526", fg="#d4d4d4",
                                                         font=("Consolas", 9), wrap=tk.WORD,
                                                         relief=tk.FLAT, borderwidth=0)
        self.operations_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.operations_text.config(state=tk.DISABLED)
        
        # Pipeline Status Section
        pipeline_frame = ttk.LabelFrame(main_frame, text="PIPELINE STATUS", padding="10")
        pipeline_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        pipeline_frame.columnconfigure(1, weight=1)
        
        ttk.Label(pipeline_frame, text="Index Status:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.index_label = ttk.Label(pipeline_frame, text="Checking...")
        self.index_label.grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(pipeline_frame, text="Documents:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(5, 0))
        self.docs_label = ttk.Label(pipeline_frame, text="Checking...")
        self.docs_label.grid(row=1, column=1, sticky=tk.W, pady=(5, 0))
        
        # Runtime Info Section
        runtime_frame = ttk.LabelFrame(main_frame, text="RUNTIME INFORMATION", padding="10")
        runtime_frame.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        runtime_frame.columnconfigure(1, weight=1)
        
        ttk.Label(runtime_frame, text="Uptime:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.uptime_label = ttk.Label(runtime_frame, text="00:00:00")
        self.uptime_label.grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(runtime_frame, text="Current Time:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(5, 0))
        self.time_label = ttk.Label(runtime_frame, text=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        self.time_label.grid(row=1, column=1, sticky=tk.W, pady=(5, 0))
        
        # Status bar at bottom
        self.statusbar = ttk.Label(main_frame, text="Monitor running | Refresh: 0.5s", 
                                  relief=tk.SUNKEN, anchor=tk.W)
        self.statusbar.grid(row=6, column=0, sticky=(tk.W, tk.E), pady=(5, 0))


# ============================================================================
# MONITOR IMPLEMENTATIONS
# ============================================================================

class QuestionInputMonitor(BaseMonitor):
    """Monitor for incoming questions"""
    
    def show(self):
        self.create_frame()
        row = self.add_back_button()
        row = self.add_title("QUESTION INPUT MONITOR", row)
        
        row = self.add_stat_frame("QUESTION STATISTICS", [
            ("Total Questions", "total", "0"),
            ("Avg Question Length", "avg_length", "0 chars"),
            ("Last Question", "last", "N/A")
        ], row)
        
        self.text_widget, row = self.add_scrollable_text("RECENT QUESTIONS", 15, row)
        
        # Load historical data
        all_operations = self.gui.load_operation_history()
        self._load_initial_items(all_operations, self._extract_questions, lambda q: [
            f"[{q['time']}] ({q['length']} chars) {'[STREAMING]' if q.get('streaming') else ''}",
            f"  {q['question']}",
            ""
        ])
        self._update_stats()
    
    def on_new_operation(self, operation_data):
        """Handle new operation event"""
        op_type = operation_data.get('operation_type', '')
        if op_type == 'question_input':
            metadata = operation_data.get('metadata', {})
            question = metadata.get('question', operation_data.get('operation', 'N/A'))
            if question and len(question) > 3:
                q_data = {
                    'question': question,
                    'time': datetime.fromtimestamp(operation_data.get('timestamp', 0)).strftime('%H:%M:%S'),
                    'length': metadata.get('question_length', len(question)),
                    'streaming': metadata.get('streaming', False)
                }
                self.items.append(q_data)
                self._add_item_to_display(q_data, lambda q: [
                    f"[{q['time']}] ({q['length']} chars) {'[STREAMING]' if q.get('streaming') else ''}",
                    f"  {q['question']}",
                    ""
                ])
                self._update_stats()
    
    def _update_stats(self):
        """Update statistics labels"""
        self.widgets['total'].config(text=str(len(self.items)))
        if self.items:
            total_length = sum(q['length'] for q in self.items)
            avg_length = total_length / len(self.items)
            self.widgets['avg_length'].config(text=f"{avg_length:.0f} chars")
            last_q = self.items[-1]['question']
            self.widgets['last'].config(text=last_q[:60] + "..." if len(last_q) > 60 else last_q)
        else:
            self.widgets['avg_length'].config(text="0 chars")
            self.widgets['last'].config(text="N/A")
    
    def _extract_questions(self, operations):
        """Extract question data from operations"""
        questions = []
        for op in operations:
            op_type = op.get('operation_type', '')
            if op_type == 'question_input':
                metadata = op.get('metadata', {})
                question = metadata.get('question', op.get('operation', 'N/A'))
                if question and len(question) > 3:
                    questions.append({
                        'question': question,
                        'time': datetime.fromtimestamp(op.get('timestamp', 0)).strftime('%H:%M:%S'),
                        'length': metadata.get('question_length', len(question)),
                        'streaming': metadata.get('streaming', False)
                    })
        return questions



class EmbeddingQueryMonitor(BaseMonitor):
    """Monitor for query embedding process"""
    
    def show(self):
        self.create_frame()
        row = self.add_back_button()
        row = self.add_title("EMBEDDING QUERY MONITOR", row)
        
        row = self.add_stat_frame("EMBEDDING MODEL", [
            ("Model", "model", "sentence-transformers/all-MiniLM-L6-v2"),
            ("Vector Dimension", "dim", "384"),
            ("Device", "device", "Checking...")
        ], row)
        
        row = self.add_stat_frame("EMBEDDING STATISTICS", [
            ("Total Embeddings", "total", "0"),
            ("Avg Time", "avg_time", "0.000s")
        ], row)
        
        self.text_widget, row = self.add_scrollable_text("RECENT EMBEDDING OPERATIONS", 10, row)
        
        # Load historical data
        all_operations = self.gui.load_operation_history()
        self._load_initial_items(all_operations, self._extract_embeddings, lambda e: [
            f"[{e['time']}] {e['query'][:60]}...",
            f"  → Model: {e['model']}, Dim: {e['dim']}, Device: {e['device']}",
            ""
        ])
        self._update_stats()
    
    def on_new_operation(self, operation_data):
        """Handle new operation event"""
        op_type = operation_data.get('operation_type', '')
        if op_type == 'embedding_query':
            metadata = operation_data.get('metadata', {})
            e_data = {
                'query': metadata.get('query', 'N/A'),
                'model': metadata.get('model', 'unknown'),
                'dim': metadata.get('dimensions', 0),
                'device': metadata.get('device', 'unknown'),
                'time_ms': metadata.get('search_time_ms', 0),
                'time': datetime.fromtimestamp(operation_data.get('timestamp', 0)).strftime('%H:%M:%S')
            }
            self.items.append(e_data)
            self._add_item_to_display(e_data, lambda e: [
                f"[{e['time']}] {e['query'][:60]}...",
                f"  → Model: {e['model']}, Dim: {e['dim']}, Device: {e['device']}",
                ""
            ])
            self._update_stats()
    
    def _update_stats(self):
        """Update statistics labels"""
        self._update_device_status()
        self.widgets['total'].config(text=str(len(self.items)))
        if self.items:
            avg_time = sum(e.get('time_ms', 0) for e in self.items) / len(self.items)
            self.widgets['avg_time'].config(text=f"{avg_time/1000:.3f}s")
        else:
            self.widgets['avg_time'].config(text="0.000s")
    
    def _update_device_status(self):
        """Check and update GPU/CPU status"""
        try:
            import torch
            device = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
            color = "#4ec9b0" if torch.cuda.is_available() else "#dcdcaa"
            self.widgets['device'].config(text=device, foreground=color)
        except:
            self.widgets['device'].config(text="CPU", foreground="#dcdcaa")
    
    def _extract_embeddings(self, operations):
        """Extract embedding data from operations"""
        embeddings = []
        for op in operations:
            op_type = op.get('operation_type', '')
            if op_type == 'embedding_query':
                metadata = op.get('metadata', {})
                embeddings.append({
                    'query': metadata.get('query', 'N/A'),
                    'model': metadata.get('model', 'unknown'),
                    'dim': metadata.get('dimensions', 0),
                    'device': metadata.get('device', 'unknown'),
                    'time_ms': metadata.get('search_time_ms', 0),
                    'time': datetime.fromtimestamp(op.get('timestamp', 0)).strftime('%H:%M:%S')
                })
        return embeddings
    




class FAISSSearchMonitor(BaseMonitor):
    """Monitor for FAISS vector search operations"""
    
    def show(self):
        self.create_frame()
        row = self.add_back_button()
        row = self.add_title("FAISS SEARCH MONITOR", row)
        
        row = self.add_stat_frame("INDEX INFORMATION", [
            ("Index Status", "index_status", "Checking..."),
            ("Total Vectors", "vector_count", "0"),
            ("Index Size", "index_size", "0 KB")
        ], row)
        
        row = self.add_stat_frame("SEARCH STATISTICS", [
            ("Total Searches", "total_searches", "0"),
            ("K Value", "k_value", "100"),
            ("Avg Results", "avg_results", "0"),
            ("Avg Search Time", "avg_time", "0.000s")
        ], row)
        
        self.text_widget, row = self.add_scrollable_text("RECENT SEARCH OPERATIONS", 10, row)
        
        # Load historical data
        all_operations = self.gui.load_operation_history()
        self._load_initial_items(all_operations, self._extract_searches, lambda s: [
            f"[{s['time']}] Query: {s['query'][:50]}...",
            f"  → K={s['k']}, Results={s['num_results']}, Index={s['index_size']}, Time={s['time_ms']:.2f}ms",
            ""
        ])
        self._update_stats()
    
    def on_new_operation(self, operation_data):
        """Handle new operation event"""
        op_type = operation_data.get('operation_type', '')
        if op_type == 'faiss_search':
            metadata = operation_data.get('metadata', {})
            s_data = {
                'query': metadata.get('query', 'N/A'),
                'k': metadata.get('k', 100),
                'num_results': metadata.get('num_results', 0),
                'index_size': metadata.get('index_size', 0),
                'time_ms': metadata.get('search_time_ms', 0),
                'device': metadata.get('device', 'unknown'),
                'time': datetime.fromtimestamp(operation_data.get('timestamp', 0)).strftime('%H:%M:%S')
            }
            self.items.append(s_data)
            self._add_item_to_display(s_data, lambda s: [
                f"[{s['time']}] Query: {s['query'][:50]}...",
                f"  → K={s['k']}, Results={s['num_results']}, Index={s['index_size']}, Time={s['time_ms']:.2f}ms",
                ""
            ])
            self._update_stats()
    
    def _update_stats(self):
        """Update statistics labels"""
        self._update_index_status()
        self.widgets['total_searches'].config(text=str(len(self.items)))
        if self.items:
            avg_results = sum(s.get('num_results', 0) for s in self.items) / len(self.items)
            avg_time = sum(s.get('time_ms', 0) for s in self.items) / len(self.items)
            self.widgets['avg_results'].config(text=f"{avg_results:.0f}")
            self.widgets['avg_time'].config(text=f"{avg_time/1000:.3f}s")
        else:
            self.widgets['avg_results'].config(text="0")
            self.widgets['avg_time'].config(text="0.000s")
    
    def _update_index_status(self):
        """Check and update FAISS index status"""
        index_path = "data/index/faiss_index/index.faiss"
        if os.path.exists(index_path):
            index_size = os.path.getsize(index_path) / 1024
            self.widgets['index_status'].config(text="Loaded", foreground="#4ec9b0")
            self.widgets['index_size'].config(text=f"{index_size:.1f} KB")
            
            try:
                if self.gui.pipeline and self.gui.pipeline.vectorstore:
                    vector_count = self.gui.pipeline.vectorstore.index.ntotal
                    self.widgets['vector_count'].config(text=str(vector_count))
                else:
                    self.widgets['vector_count'].config(text="Unknown")
            except:
                self.widgets['vector_count'].config(text="Unknown")
        else:
            self.widgets['index_status'].config(text="Not Found", foreground="#f48771")
            self.widgets['index_size'].config(text="0 KB")
            self.widgets['vector_count'].config(text="0")
    
    def _extract_searches(self, operations):
        """Extract search data from operations"""
        searches = []
        for op in operations:
            op_type = op.get('operation_type', '')
            if op_type == 'faiss_search':
                metadata = op.get('metadata', {})
                searches.append({
                    'query': metadata.get('query', 'N/A'),
                    'k': metadata.get('k', 100),
                    'num_results': metadata.get('num_results', 0),
                    'index_size': metadata.get('index_size', 0),
                    'time_ms': metadata.get('search_time_ms', 0),
                    'device': metadata.get('device', 'unknown'),
                    'time': datetime.fromtimestamp(op.get('timestamp', 0)).strftime('%H:%M:%S')
                })
        return searches


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    monitor = LiveMonitorGUI()
    monitor.run()
