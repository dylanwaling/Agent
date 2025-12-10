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
        
        # Grid of monitor buttons (3 columns)
        monitors = [
            ("General Info", self.show_general_info),
            ("Operation 2", lambda: self.show_placeholder("Operation 2")),
            ("Operation 3", lambda: self.show_placeholder("Operation 3")),
            ("Operation 4", lambda: self.show_placeholder("Operation 4")),
            ("Operation 5", lambda: self.show_placeholder("Operation 5")),
            ("Operation 6", lambda: self.show_placeholder("Operation 6")),
            ("Operation 7", lambda: self.show_placeholder("Operation 7")),
            ("Operation 8", lambda: self.show_placeholder("Operation 8")),
            ("Operation 9", lambda: self.show_placeholder("Operation 9")),
            ("Operation 10", lambda: self.show_placeholder("Operation 10")),
            ("Operation 11", lambda: self.show_placeholder("Operation 11")),
            ("Operation 12", lambda: self.show_placeholder("Operation 12")),
            ("Operation 13", lambda: self.show_placeholder("Operation 13")),
            ("Operation 14", lambda: self.show_placeholder("Operation 14")),
            ("Operation 15", lambda: self.show_placeholder("Operation 15")),
            ("Operation 16", lambda: self.show_placeholder("Operation 16")),
            ("Operation 17", lambda: self.show_placeholder("Operation 17")),
            ("Operation 18", lambda: self.show_placeholder("Operation 18")),
            ("Operation 19", lambda: self.show_placeholder("Operation 19")),
            ("Operation 20", lambda: self.show_placeholder("Operation 20")),
            ("Operation 21", lambda: self.show_placeholder("Operation 21")),
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
    
    def show_general_info(self):
        """Show the general info monitor (original view)"""
        # Clear current container
        for widget in self.main_container.winfo_children():
            widget.destroy()
        
        self.current_view = "general_info"
        
        # Create the original widgets in a new frame
        self.create_general_info_widgets()
        
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
        # Only update if we're in the general info view
        if self.current_view != "general_info":
            return
        
        # Read current status
        status, last_op, timestamp, operation_id = self.read_status()
        
        # Load operation history
        all_operations = self.load_operation_history()
        self.operation_count = len(all_operations)
        
        # Update operation history deque
        if all_operations and len(all_operations) > self.last_history_size:
            for op in all_operations[self.last_history_size:]:
                op_time = datetime.fromtimestamp(op['timestamp']).strftime('%H:%M:%S')
                self.operation_history.append(f"[{op_time}] {op['operation']}")
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
        
        # Update operations text
        self.operations_text.config(state=tk.NORMAL)
        self.operations_text.delete(1.0, tk.END)
        for op in list(self.operation_history):
            self.operations_text.insert(tk.END, op + "\n")
        self.operations_text.config(state=tk.DISABLED)
        # Auto-scroll to bottom
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


if __name__ == "__main__":
    monitor = LiveMonitorGUI()
    monitor.run()
