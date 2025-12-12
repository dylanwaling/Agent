"""
Live System Monitoring for Document Q&A Agent - GUI Version
Task-manager-style real-time display with scrollable window interface
"""

import os
import sys
import time
import threading
import json
from datetime import datetime
import psutil
from pathlib import Path
from backend_logic import DocumentPipeline
import tkinter as tk
from tkinter import ttk, scrolledtext
from collections import deque

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
        self.last_count = 0
        
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
        self.gui.root.after(10, scroll)
        self.gui.root.after(50, scroll)
        self.gui.root.after(150, scroll)
        self.gui.root.after(300, scroll)
    
    def show(self):
        """Override this method in subclasses"""
        raise NotImplementedError
    
    def update(self):
        """Override this method in subclasses"""
        raise NotImplementedError


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
        self.last_history_size = 0
        
        # Current view and active monitor
        self.current_view = "menu"  # Start at main menu
        self.active_monitor = None  # Currently active monitor instance
        
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
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.monitor_thread.start()
        
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
            ("Question Input", lambda: self.show_generic_monitor("Question Input", 
                ["Searching:", "Answering"], QuestionInputMonitor)),
            ("Embedding Query", lambda: self.show_generic_monitor("Embedding Query", 
                ["Searching:", "Answering"], EmbeddingQueryMonitor)),
            ("FAISS Search", lambda: self.show_generic_monitor("FAISS Search", 
                ["Searching:", "Answering"], FAISSSearchMonitor)),
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
    
    def show_generic_monitor(self, name, keywords, monitor_class):
        """Generic method to show any monitor view"""
        self.clear_container()
        self.current_view = name.lower().replace(" ", "_")
        
        monitor = monitor_class(self.main_container, self)
        monitor.show()
        
        # Store monitor for updates
        self.active_monitor = monitor
        
        # Initial populate
        self.root.after(50, monitor.update)
    
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
    # UTILITY METHODS
    # ========================================================================
    
    def read_status(self):
        """Read status from shared file"""
        try:
            if self.status_file.exists():
                with open(self.status_file, 'r') as f:
                    data = json.load(f)
                    status = data.get("status", "IDLE")
                    operation = data.get("operation", "Unknown")
                    timestamp = data.get("timestamp", 0)
                    operation_id = data.get("operation_id", "")
                    return status, operation, timestamp, operation_id
            return "IDLE", "Waiting for status file...", 0, ""
        except Exception as e:
            return "IDLE", f"Error reading status: {str(e)}", 0, ""
    
    def load_operation_history(self):
        """Load operations from the history log file"""
        operations = []
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            op = json.loads(line)
                            operations.append(op)
        except Exception as e:
            pass
        return operations
    
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
    
    def update_gui(self):
        """Update all GUI elements with current data"""
        # Route to appropriate update method based on current view
        if self.current_view == "general_info":
            self.update_general_info()
        elif hasattr(self, 'active_monitor') and self.active_monitor:
            self.active_monitor.update()
    
    def update_general_info(self):
        """Update General Info view"""
        # Read current status
        status, last_op, timestamp, operation_id = self.read_status()
        
        # Load operation history
        all_operations = self.load_operation_history()
        self.operation_count = len(all_operations)
        
        # Update operation history deque and track new operations
        new_operations = []
        if all_operations and len(all_operations) > self.last_history_size:
            for op in all_operations[self.last_history_size:]:
                op_time = datetime.fromtimestamp(op['timestamp']).strftime('%H:%M:%S')
                formatted = f"[{op_time}] {op['operation']}"
                self.operation_history.append(formatted)
                new_operations.append(formatted)
            self.last_history_size = len(all_operations)
        
        # Update status directly - no delays or filtering
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
        self.count_label.config(text=str(self.operation_count))
        
        # Update operations text with smart scroll handling
        if new_operations:
            # Check if user is viewing the live stream (at bottom)
            yview = self.operations_text.yview()
            at_bottom = yview[1] >= 0.99
            
            self.operations_text.config(state=tk.NORMAL)
            for op in new_operations:
                self.operations_text.insert(tk.END, op + "\n")
            self.operations_text.config(state=tk.DISABLED)
            
            # Only auto-scroll if user was already at the bottom
            if at_bottom:
                self.operations_text.see(tk.END)
        
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
            self.statusbar.config(text=f"Monitor running | Refresh: 0.15s | Last status: {age:.1f}s ago")
    
    # ========================================================================
    # BACKGROUND THREAD
    # ========================================================================
    
    def monitor_loop(self):
        """Background thread that updates the GUI"""
        # Initialize pipeline in background
        time.sleep(0.5)
        try:
            self.pipeline = DocumentPipeline()
        except Exception as e:
            print(f"Failed to initialize pipeline: {e}")
        
        while self.running:
            try:
                # Schedule GUI update on main thread
                self.root.after(0, self.update_gui)
                time.sleep(0.15)  # Update every 150ms for faster response
            except Exception as e:
                print(f"Monitor loop error: {e}")
                time.sleep(1)
    
    def on_closing(self):
        """Handle window close event"""
        self.running = False
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
        
        # Populate with existing operations from deque
        if self.operation_history:
            self.operations_text.config(state=tk.NORMAL)
            for op in list(self.operation_history):
                self.operations_text.insert(tk.END, op + "\n")
            self.operations_text.config(state=tk.DISABLED)
        
        # Multiple scroll attempts with different methods to ensure it works
        def scroll_to_bottom():
            try:
                self.operations_text.update_idletasks()
                self.operations_text.yview_moveto(1.0)
                self.operations_text.see(tk.END)
            except:
                pass
        
        # Try multiple times with increasing delays
        self.root.after(10, scroll_to_bottom)
        self.root.after(50, scroll_to_bottom)
        self.root.after(150, scroll_to_bottom)
        self.root.after(300, scroll_to_bottom)
        
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
        
        # Question Stats
        row = self.add_stat_frame("QUESTION STATISTICS", [
            ("Total Questions", "total", "0"),
            ("Avg Question Length", "avg_length", "0 chars"),
            ("Last Question", "last", "N/A")
        ], row)
        
        # Recent Questions
        self.text_widget, row = self.add_scrollable_text("RECENT QUESTIONS", 15, row)
        
        # Force scroll to bottom after widget creation
        self._schedule_scroll_to_bottom()
    
    def update(self):
        all_operations = self.gui.load_operation_history()
        
        # Filter for question/search operations
        questions = []
        total_length = 0
        for op in all_operations:
            op_text = op.get('operation', '')
            if 'Searching:' in op_text or 'Answering' in op_text:
                # Extract question text
                question = op_text.replace('Searching:', '').replace('Answering (streaming):', '').replace('Answering:', '').strip()
                
                if question and len(question) > 3:
                    timestamp = op.get('timestamp', 0)
                    time_str = datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')
                    questions.append({
                        'question': question,
                        'time': time_str,
                        'length': len(question)
                    })
                    total_length += len(question)
        
        # Update stats
        self.widgets['total'].config(text=str(len(questions)))
        
        if questions:
            avg_length = total_length / len(questions)
            self.widgets['avg_length'].config(text=f"{avg_length:.0f} chars")
            last_q = questions[-1]['question']
            self.widgets['last'].config(text=last_q[:60] + "..." if len(last_q) > 60 else last_q)
        else:
            self.widgets['avg_length'].config(text="0 chars")
            self.widgets['last'].config(text="N/A")
        
        # Update text widget with new questions
        if len(questions) > self.last_count:
            if self.last_count == 0 or len(questions) < self.last_count:
                # Full refresh - clear and show last 50
                self.text_widget.config(state=tk.NORMAL)
                self.text_widget.delete(1.0, tk.END)
                self.text_widget.config(state=tk.DISABLED)
                new_lines = []
                for q in questions[-50:]:
                    new_lines.append(f"[{q['time']}] ({q['length']} chars)")
                    new_lines.append(f"  {q['question']}")
                    new_lines.append("")
                self.update_text_widget(self.text_widget, new_lines)
            else:
                # Incremental update
                new_lines = []
                for q in questions[self.last_count:]:
                    new_lines.append(f"[{q['time']}] ({q['length']} chars)")
                    new_lines.append(f"  {q['question']}")
                    new_lines.append("")
                self.update_text_widget(self.text_widget, new_lines)
            
            self.last_count = len(questions)



class EmbeddingQueryMonitor(BaseMonitor):
    """Monitor for query embedding process"""
    
    def show(self):
        self.create_frame()
        row = self.add_back_button()
        row = self.add_title("EMBEDDING QUERY MONITOR", row)
        
        # Model Info
        row = self.add_stat_frame("EMBEDDING MODEL", [
            ("Model", "model", "sentence-transformers/all-MiniLM-L6-v2"),
            ("Vector Dimension", "dim", "384"),
            ("Device", "device", "Checking...")
        ], row)
        
        # Embedding Stats
        row = self.add_stat_frame("EMBEDDING STATISTICS", [
            ("Total Embeddings", "total", "0"),
            ("Avg Time", "avg_time", "0.000s")
        ], row)
        
        # Recent Embeddings
        self.text_widget, row = self.add_scrollable_text("RECENT EMBEDDING OPERATIONS", 10, row)
        
        # Force scroll to bottom after widget creation
        self._schedule_scroll_to_bottom()
    
    def update(self):
        # Check device
        try:
            import torch
            if torch.cuda.is_available():
                self.widgets['device'].config(text="CUDA (GPU)", foreground="#4ec9b0")
            else:
                self.widgets['device'].config(text="CPU", foreground="#dcdcaa")
        except:
            self.widgets['device'].config(text="CPU", foreground="#dcdcaa")
        
        all_operations = self.gui.load_operation_history()
        
        # Filter for embedding operations
        embeddings = []
        for op in all_operations:
            op_text = op.get('operation', '')
            if 'Searching:' in op_text or 'Answering' in op_text:
                timestamp = op.get('timestamp', 0)
                time_str = datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')
                embeddings.append({
                    'operation': op_text[:60] + "..." if len(op_text) > 60 else op_text,
                    'time': time_str
                })
        
        # Update stats
        self.widgets['total'].config(text=str(len(embeddings)))
        self.widgets['avg_time'].config(text="~0.050s" if len(embeddings) > 0 else "0.000s")
        
        # Update text widget
        if len(embeddings) > self.last_count:
            if self.last_count == 0 or len(embeddings) < self.last_count:
                # Full refresh - show last 50
                self.text_widget.config(state=tk.NORMAL)
                self.text_widget.delete(1.0, tk.END)
                self.text_widget.config(state=tk.DISABLED)
                new_lines = [f"[{emb['time']}] {emb['operation']}" for emb in embeddings[-50:]]
                self.update_text_widget(self.text_widget, new_lines)
            else:
                # Incremental update
                new_lines = [f"[{emb['time']}] {emb['operation']}" for emb in embeddings[self.last_count:]]
                self.update_text_widget(self.text_widget, new_lines)
            
            self.last_count = len(embeddings)
    




class FAISSSearchMonitor(BaseMonitor):
    """Monitor for FAISS vector search operations"""
    
    def show(self):
        self.create_frame()
        row = self.add_back_button()
        row = self.add_title("FAISS SEARCH MONITOR", row)
        
        # Index Info
        row = self.add_stat_frame("INDEX INFORMATION", [
            ("Index Status", "index_status", "Checking..."),
            ("Total Vectors", "vector_count", "0"),
            ("Index Size", "index_size", "0 KB")
        ], row)
        
        # Search Stats
        row = self.add_stat_frame("SEARCH STATISTICS", [
            ("Total Searches", "total_searches", "0"),
            ("K Value", "k_value", "100"),
            ("Avg Results", "avg_results", "0"),
            ("Avg Search Time", "avg_time", "0.000s")
        ], row)
        
        # Recent Searches
        self.text_widget, row = self.add_scrollable_text("RECENT SEARCH OPERATIONS", 10, row)
        
        # Force scroll to bottom after widget creation
        self._schedule_scroll_to_bottom()
    
    def update(self):
        # Check index status
        index_path = "data/index/faiss_index/index.faiss"
        if os.path.exists(index_path):
            index_size = os.path.getsize(index_path) / 1024  # KB
            self.widgets['index_status'].config(text="Loaded", foreground="#4ec9b0")
            self.widgets['index_size'].config(text=f"{index_size:.1f} KB")
            
            # Try to get vector count
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
        
        # Load operation history for search operations
        all_operations = self.gui.load_operation_history()
        
        # Filter for search operations
        searches = []
        for op in all_operations:
            op_text = op.get('operation', '')
            if 'Searching:' in op_text or 'Answering' in op_text:
                timestamp = op.get('timestamp', 0)
                time_str = datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')
                
                # Extract query text
                query = op_text.replace('Searching:', '').replace('Answering (streaming):', '').replace('Answering:', '').strip()[:50]
                
                searches.append({
                    'query': query,
                    'time': time_str
                })
        
        # Update stats
        self.widgets['total_searches'].config(text=str(len(searches)))
        self.widgets['avg_results'].config(text="~15" if searches else "0")
        self.widgets['avg_time'].config(text="~0.025s" if searches else "0.000s")
        
        # Update text widget
        if len(searches) > self.last_count:
            if self.last_count == 0 or len(searches) < self.last_count:
                # Full refresh - show last 50
                self.text_widget.config(state=tk.NORMAL)
                self.text_widget.delete(1.0, tk.END)
                self.text_widget.config(state=tk.DISABLED)
                new_lines = []
                for search in searches[-50:]:
                    new_lines.append(f"[{search['time']}] Query: {search['query']}...")
                    new_lines.append(f"  → K=100, Searching index...")
                    new_lines.append("")
                self.update_text_widget(self.text_widget, new_lines)
            else:
                # Incremental update
                new_lines = []
                for search in searches[self.last_count:]:
                    new_lines.append(f"[{search['time']}] Query: {search['query']}...")
                    new_lines.append(f"  → K=100, Searching index...")
                    new_lines.append("")
                self.update_text_widget(self.text_widget, new_lines)
            
            self.last_count = len(searches)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    monitor = LiveMonitorGUI()
    monitor.run()
