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
        
        # Current view
        self.current_view = "menu"  # Start at main menu
        
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
    
    def show_menu(self):
        """Show the main menu with monitor options"""
        # Clear current container
        for widget in self.main_container.winfo_children():
            widget.destroy()
        
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
            ("Question Input", self.show_question_input),
            ("Embedding Query", self.show_embedding_query),
            ("FAISS Search", self.show_faiss_search),
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
    
    def show_placeholder(self, name):
        """Show placeholder for monitors not yet implemented"""
        # Clear current container
        for widget in self.main_container.winfo_children():
            widget.destroy()
        
        self.current_view = name
        
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
        elif self.current_view == "question_input":
            self.update_question_input()
        elif self.current_view == "embedding_query":
            self.update_embedding_query()
        elif self.current_view == "faiss_search":
            self.update_faiss_search()
    
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
            # Start at bottom
            self.operations_text.see(tk.END)
        
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
# MONITOR VIEW: QUESTION INPUT
# ============================================================================
    
    def show_question_input(self):
        """Monitor for incoming questions"""
        for widget in self.main_container.winfo_children():
            widget.destroy()
        
        self.current_view = "question_input"
        # Reset counter when view is created
        self.qi_last_count = 0
        
        main_frame = ttk.Frame(self.main_container, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.main_container.columnconfigure(0, weight=1)
        self.main_container.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        
        ttk.Button(main_frame, text="← Back to Menu", command=self.show_menu).grid(row=0, column=0, sticky=tk.W, pady=(0, 10))
        ttk.Label(main_frame, text="QUESTION INPUT MONITOR", style="Title.TLabel").grid(row=1, column=0, pady=(0, 15), sticky=tk.W)
        
        # Question Stats
        stats_frame = ttk.LabelFrame(main_frame, text="QUESTION STATISTICS", padding="10")
        stats_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        stats_frame.columnconfigure(1, weight=1)
        
        ttk.Label(stats_frame, text="Total Questions:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.qi_total_label = ttk.Label(stats_frame, text="0")
        self.qi_total_label.grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(stats_frame, text="Avg Question Length:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(5, 0))
        self.qi_avg_length_label = ttk.Label(stats_frame, text="0 chars")
        self.qi_avg_length_label.grid(row=1, column=1, sticky=tk.W, pady=(5, 0))
        
        ttk.Label(stats_frame, text="Last Question:").grid(row=2, column=0, sticky=tk.W, padx=(0, 10), pady=(5, 0))
        self.qi_last_label = ttk.Label(stats_frame, text="N/A")
        self.qi_last_label.grid(row=2, column=1, sticky=tk.W, pady=(5, 0))
        
        # Recent Questions
        questions_frame = ttk.LabelFrame(main_frame, text="RECENT QUESTIONS", padding="10")
        questions_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        questions_frame.columnconfigure(0, weight=1)
        questions_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)
        
        self.qi_questions_text = scrolledtext.ScrolledText(questions_frame, height=15, width=80,
                                                          bg="#252526", fg="#d4d4d4",
                                                          font=("Consolas", 9), wrap=tk.WORD,
                                                          relief=tk.FLAT, borderwidth=0)
        self.qi_questions_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.qi_questions_text.config(state=tk.DISABLED)
        
        # Populate with existing data immediately
        self.root.after(50, self._populate_question_input)
    
    def _populate_question_input(self):
        """Initial population of Question Input view"""
        # Force an update to populate the view
        self.update_question_input()


# ============================================================================
# MONITOR VIEW: EMBEDDING QUERY
# ============================================================================
    
    def show_embedding_query(self):
        """Monitor for query embedding process"""
        for widget in self.main_container.winfo_children():
            widget.destroy()
        
        self.current_view = "embedding_query"
        # Reset counter when view is created
        self.eq_last_count = 0
        
        main_frame = ttk.Frame(self.main_container, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.main_container.columnconfigure(0, weight=1)
        self.main_container.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        
        ttk.Button(main_frame, text="← Back to Menu", command=self.show_menu).grid(row=0, column=0, sticky=tk.W, pady=(0, 10))
        ttk.Label(main_frame, text="EMBEDDING QUERY MONITOR", style="Title.TLabel").grid(row=1, column=0, pady=(0, 15), sticky=tk.W)
        
        # Model Info
        model_frame = ttk.LabelFrame(main_frame, text="EMBEDDING MODEL", padding="10")
        model_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        model_frame.columnconfigure(1, weight=1)
        
        ttk.Label(model_frame, text="Model:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.eq_model_label = ttk.Label(model_frame, text="sentence-transformers/all-MiniLM-L6-v2")
        self.eq_model_label.grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(model_frame, text="Vector Dimension:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(5, 0))
        self.eq_dim_label = ttk.Label(model_frame, text="384")
        self.eq_dim_label.grid(row=1, column=1, sticky=tk.W, pady=(5, 0))
        
        ttk.Label(model_frame, text="Device:").grid(row=2, column=0, sticky=tk.W, padx=(0, 10), pady=(5, 0))
        self.eq_device_label = ttk.Label(model_frame, text="Checking...")
        self.eq_device_label.grid(row=2, column=1, sticky=tk.W, pady=(5, 0))
        
        # Embedding Stats
        stats_frame = ttk.LabelFrame(main_frame, text="EMBEDDING STATISTICS", padding="10")
        stats_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        stats_frame.columnconfigure(1, weight=1)
        
        ttk.Label(stats_frame, text="Total Embeddings:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.eq_total_label = ttk.Label(stats_frame, text="0")
        self.eq_total_label.grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(stats_frame, text="Avg Time:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(5, 0))
        self.eq_avg_time_label = ttk.Label(stats_frame, text="0.000s")
        self.eq_avg_time_label.grid(row=1, column=1, sticky=tk.W, pady=(5, 0))
        
        # Recent Embeddings
        recent_frame = ttk.LabelFrame(main_frame, text="RECENT EMBEDDING OPERATIONS", padding="10")
        recent_frame.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        recent_frame.columnconfigure(0, weight=1)
        recent_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)
        
        self.eq_recent_text = scrolledtext.ScrolledText(recent_frame, height=10, width=80,
                                                       bg="#252526", fg="#d4d4d4",
                                                       font=("Consolas", 9), wrap=tk.WORD,
                                                       relief=tk.FLAT, borderwidth=0)
        self.eq_recent_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.eq_recent_text.config(state=tk.DISABLED)
        
        # Populate with existing data immediately
        self.root.after(50, self._populate_embedding_query)
    
    def _populate_embedding_query(self):
        """Initial population of Embedding Query view"""
        # Force an update to populate the view
        self.update_embedding_query()
    
    def update_question_input(self):
        """Update Question Input monitor with real data"""
        # Load operation history
        all_operations = self.load_operation_history()
        
        # Filter for question/search operations
        questions = []
        total_length = 0
        for op in all_operations:
            op_text = op.get('operation', '')
            if 'Searching:' in op_text or 'Answering' in op_text:
                # Extract question text
                if 'Searching:' in op_text:
                    question = op_text.replace('Searching:', '').strip()
                elif 'Answering (streaming):' in op_text:
                    question = op_text.replace('Answering (streaming):', '').strip()
                else:
                    question = op_text.replace('Answering:', '').strip()
                
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
        self.qi_total_label.config(text=str(len(questions)))
        
        if questions:
            avg_length = total_length / len(questions)
            self.qi_avg_length_label.config(text=f"{avg_length:.0f} chars")
            self.qi_last_label.config(text=questions[-1]['question'][:60] + "..." if len(questions[-1]['question']) > 60 else questions[-1]['question'])
        else:
            self.qi_avg_length_label.config(text="0 chars")
            self.qi_last_label.config(text="N/A")
        
        # Track what we've already displayed
        if not hasattr(self, 'qi_last_count'):
            self.qi_last_count = 0
        
        # Only update if there are new questions
        if len(questions) > self.qi_last_count:
            yview = self.qi_questions_text.yview()
            at_bottom = yview[1] >= 0.99
            
            self.qi_questions_text.config(state=tk.NORMAL)
            
            # If first time or need full refresh, clear and show last 20
            if self.qi_last_count == 0 or len(questions) < self.qi_last_count:
                self.qi_questions_text.delete(1.0, tk.END)
                for q in questions[-20:]:
                    self.qi_questions_text.insert(tk.END, f"[{q['time']}] ({q['length']} chars)\n")
                    self.qi_questions_text.insert(tk.END, f"  {q['question']}\n\n")
            else:
                # Append only new questions
                for q in questions[self.qi_last_count:]:
                    self.qi_questions_text.insert(tk.END, f"[{q['time']}] ({q['length']} chars)\n")
                    self.qi_questions_text.insert(tk.END, f"  {q['question']}\n\n")
            
            self.qi_questions_text.config(state=tk.DISABLED)
            self.qi_last_count = len(questions)
            
            # Auto-scroll only if at bottom
            if at_bottom:
                self.qi_questions_text.see(tk.END)
    
    def update_embedding_query(self):
        """Update Embedding Query monitor with real data"""
        # Load operation history
        all_operations = self.load_operation_history()
        
        # Filter for embedding operations
        embeddings = []
        total_time = 0
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
        self.eq_total_label.config(text=str(len(embeddings)))
        
        # For now, estimated time (we can add real timing later)
        if len(embeddings) > 0:
            self.eq_avg_time_label.config(text="~0.050s")
        else:
            self.eq_avg_time_label.config(text="0.000s")
        
        # Check device
        try:
            import torch
            if torch.cuda.is_available():
                self.eq_device_label.config(text="CUDA (GPU)", foreground="#4ec9b0")
            else:
                self.eq_device_label.config(text="CPU", foreground="#dcdcaa")
        except:
            self.eq_device_label.config(text="CPU", foreground="#dcdcaa")
        
        # Track what we've already displayed
        if not hasattr(self, 'eq_last_count'):
            self.eq_last_count = 0
        
        # Only update if there are new embeddings
        if len(embeddings) > self.eq_last_count:
            yview = self.eq_recent_text.yview()
            at_bottom = yview[1] >= 0.99
            
            self.eq_recent_text.config(state=tk.NORMAL)
            
            # If first time or need full refresh, clear and show last 15
            if self.eq_last_count == 0 or len(embeddings) < self.eq_last_count:
                self.eq_recent_text.delete(1.0, tk.END)
                for emb in embeddings[-15:]:
                    self.eq_recent_text.insert(tk.END, f"[{emb['time']}] {emb['operation']}\n")
            else:
                # Append only new embeddings
                for emb in embeddings[self.eq_last_count:]:
                    self.eq_recent_text.insert(tk.END, f"[{emb['time']}] {emb['operation']}\n")
            
            self.eq_recent_text.config(state=tk.DISABLED)
            self.eq_last_count = len(embeddings)
            
            # Auto-scroll only if at bottom
            if at_bottom:
                self.eq_recent_text.see(tk.END)


# ============================================================================
# MONITOR VIEW: FAISS SEARCH
# ============================================================================
    
    def show_faiss_search(self):
        """Monitor for FAISS vector search operations"""
        for widget in self.main_container.winfo_children():
            widget.destroy()
        
        self.current_view = "faiss_search"
        # Reset counter when view is created
        self.fs_last_count = 0
        
        main_frame = ttk.Frame(self.main_container, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.main_container.columnconfigure(0, weight=1)
        self.main_container.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        
        ttk.Button(main_frame, text="← Back to Menu", command=self.show_menu).grid(row=0, column=0, sticky=tk.W, pady=(0, 10))
        ttk.Label(main_frame, text="FAISS SEARCH MONITOR", style="Title.TLabel").grid(row=1, column=0, pady=(0, 15), sticky=tk.W)
        
        # Index Info
        index_frame = ttk.LabelFrame(main_frame, text="INDEX INFORMATION", padding="10")
        index_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        index_frame.columnconfigure(1, weight=1)
        
        ttk.Label(index_frame, text="Index Status:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.fs_index_status_label = ttk.Label(index_frame, text="Checking...")
        self.fs_index_status_label.grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(index_frame, text="Total Vectors:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(5, 0))
        self.fs_vector_count_label = ttk.Label(index_frame, text="0")
        self.fs_vector_count_label.grid(row=1, column=1, sticky=tk.W, pady=(5, 0))
        
        ttk.Label(index_frame, text="Index Size:").grid(row=2, column=0, sticky=tk.W, padx=(0, 10), pady=(5, 0))
        self.fs_index_size_label = ttk.Label(index_frame, text="0 KB")
        self.fs_index_size_label.grid(row=2, column=1, sticky=tk.W, pady=(5, 0))
        
        # Search Stats
        stats_frame = ttk.LabelFrame(main_frame, text="SEARCH STATISTICS", padding="10")
        stats_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        stats_frame.columnconfigure(1, weight=1)
        
        ttk.Label(stats_frame, text="Total Searches:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.fs_total_searches_label = ttk.Label(stats_frame, text="0")
        self.fs_total_searches_label.grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(stats_frame, text="K Value:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(5, 0))
        self.fs_k_value_label = ttk.Label(stats_frame, text="100")
        self.fs_k_value_label.grid(row=1, column=1, sticky=tk.W, pady=(5, 0))
        
        ttk.Label(stats_frame, text="Avg Results:").grid(row=2, column=0, sticky=tk.W, padx=(0, 10), pady=(5, 0))
        self.fs_avg_results_label = ttk.Label(stats_frame, text="0")
        self.fs_avg_results_label.grid(row=2, column=1, sticky=tk.W, pady=(5, 0))
        
        ttk.Label(stats_frame, text="Avg Search Time:").grid(row=3, column=0, sticky=tk.W, padx=(0, 10), pady=(5, 0))
        self.fs_avg_time_label = ttk.Label(stats_frame, text="0.000s")
        self.fs_avg_time_label.grid(row=3, column=1, sticky=tk.W, pady=(5, 0))
        
        # Recent Searches
        searches_frame = ttk.LabelFrame(main_frame, text="RECENT SEARCH OPERATIONS", padding="10")
        searches_frame.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        searches_frame.columnconfigure(0, weight=1)
        searches_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)
        
        self.fs_searches_text = scrolledtext.ScrolledText(searches_frame, height=10, width=80,
                                                         bg="#252526", fg="#d4d4d4",
                                                         font=("Consolas", 9), wrap=tk.WORD,
                                                         relief=tk.FLAT, borderwidth=0)
        self.fs_searches_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.fs_searches_text.config(state=tk.DISABLED)
        
        # Populate with existing data immediately
        self.root.after(50, self._populate_faiss_search)
    
    def _populate_faiss_search(self):
        """Initial population of FAISS Search view"""
        # Force an update to populate the view
        self.update_faiss_search()
    
    def update_faiss_search(self):
        """Update FAISS Search monitor with real data"""
        # Check index status
        index_path = "data/index/faiss_index/index.faiss"
        if os.path.exists(index_path):
            index_size = os.path.getsize(index_path) / 1024  # KB
            self.fs_index_status_label.config(text="Loaded", foreground="#4ec9b0")
            self.fs_index_size_label.config(text=f"{index_size:.1f} KB")
            
            # Try to get vector count
            try:
                if self.pipeline and self.pipeline.vectorstore:
                    vector_count = self.pipeline.vectorstore.index.ntotal
                    self.fs_vector_count_label.config(text=str(vector_count))
                else:
                    self.fs_vector_count_label.config(text="Unknown")
            except:
                self.fs_vector_count_label.config(text="Unknown")
        else:
            self.fs_index_status_label.config(text="Not Found", foreground="#f48771")
            self.fs_index_size_label.config(text="0 KB")
            self.fs_vector_count_label.config(text="0")
        
        # Load operation history for search operations
        all_operations = self.load_operation_history()
        
        # Filter for search operations (both Searching and Answering trigger FAISS searches)
        searches = []
        for op in all_operations:
            op_text = op.get('operation', '')
            if 'Searching:' in op_text or 'Answering' in op_text:
                timestamp = op.get('timestamp', 0)
                time_str = datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')
                
                # Extract query text
                if 'Searching:' in op_text:
                    query = op_text.replace('Searching:', '').strip()[:50]
                else:
                    query = op_text.replace('Answering (streaming):', '').replace('Answering:', '').strip()[:50]
                
                searches.append({
                    'query': query,
                    'time': time_str
                })
        
        # Update stats
        self.fs_total_searches_label.config(text=str(len(searches)))
        
        if searches:
            # Estimated average (we can add real metrics later)
            self.fs_avg_results_label.config(text="~15")
            self.fs_avg_time_label.config(text="~0.025s")
        else:
            self.fs_avg_results_label.config(text="0")
            self.fs_avg_time_label.config(text="0.000s")
        
        # Track what we've already displayed
        if not hasattr(self, 'fs_last_count'):
            self.fs_last_count = 0
        
        # Only update if there are new searches
        if len(searches) > self.fs_last_count:
            yview = self.fs_searches_text.yview()
            at_bottom = yview[1] >= 0.99
            
            self.fs_searches_text.config(state=tk.NORMAL)
            
            # If first time or need full refresh, clear and show last 15
            if self.fs_last_count == 0 or len(searches) < self.fs_last_count:
                self.fs_searches_text.delete(1.0, tk.END)
                for search in searches[-15:]:
                    self.fs_searches_text.insert(tk.END, f"[{search['time']}] Query: {search['query']}...\n")
                    self.fs_searches_text.insert(tk.END, f"  → K=100, Searching index...\n\n")
            else:
                # Append only new searches
                for search in searches[self.fs_last_count:]:
                    self.fs_searches_text.insert(tk.END, f"[{search['time']}] Query: {search['query']}...\n")
                    self.fs_searches_text.insert(tk.END, f"  → K=100, Searching index...\n\n")
            
            self.fs_searches_text.config(state=tk.DISABLED)
            self.fs_last_count = len(searches)
            
            # Auto-scroll only if at bottom
            if at_bottom:
                self.fs_searches_text.see(tk.END)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    monitor = LiveMonitorGUI()
    monitor.run()
