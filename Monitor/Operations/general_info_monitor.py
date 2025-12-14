"""
General Info Monitor - System overview with operations, pipeline, and runtime stats
"""

import os
import logging
from datetime import datetime
from tkinter import ttk, scrolledtext
import tkinter as tk
from monitor.performance_monitor import BaseMonitor

logger = logging.getLogger(__name__)


class GeneralInfoMonitor(BaseMonitor):
    """
    General system information monitor.
    
    Displays comprehensive system status with operations, pipeline info, and runtime stats.
    """
    
    def show(self):
        """
        Display the General Info monitor view.
        
        Creates UI with system status, operations log, pipeline info, and runtime statistics.
        """
        self.create_frame()
        row = 0
        
        # Back button
        btn = ttk.Button(self.main_frame, text="← Back to Menu", command=self.gui.show_menu)
        btn.grid(row=row, column=0, sticky=tk.W, pady=(0, 10))
        row += 1
        
        # Title
        title = ttk.Label(self.main_frame, text="GENERAL INFO - LIVE SYSTEM MONITOR", style="Title.TLabel")
        title.grid(row=row, column=0, pady=(0, 15), sticky=tk.W)
        row += 1
        
        # Process Status Section
        row = self.add_stat_frame("PROCESS STATUS", [
            ("Status", "status", "IDLE"),
            ("Last Operation", "last_operation", "System started"),
            ("Operations Count", "operations_count", "0")
        ], row)
        
        # Recent Operations Section (Scrollable)
        self.text_widget, row = self.add_scrollable_text("RECENT OPERATIONS", 8, row)
        
        # Pipeline Status Section
        row = self.add_stat_frame("PIPELINE STATUS", [
            ("Index Status", "index_status", "Checking..."),
            ("Documents", "documents", "Checking...")
        ], row)
        
        # Runtime Info Section
        row = self.add_stat_frame("RUNTIME INFORMATION", [
            ("Uptime", "uptime", "00:00:00"),
            ("Current Time", "current_time", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        ], row)
        
        # Status bar at bottom
        self.statusbar = ttk.Label(self.main_frame, text="Monitor running | Auto-refresh", 
                                  relief=tk.SUNKEN, anchor=tk.W)
        self.statusbar.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        # Load historical data
        all_operations = self.gui.load_operation_history()
        self._load_initial_items(all_operations, self._extract_operations, lambda op: [
            f"[{op['time']}] {op['operation']}"
        ])
        self._update_stats()
        
        # Start auto-refresh to get updates from file
        self.start_auto_refresh(self._extract_operations, lambda op: [
            f"[{op['time']}] {op['operation']}"
        ])
        
        # Start periodic stats update (every second for time/status)
        self._schedule_stats_update()
    
    def on_new_operation(self, operation_data):
        """
        Handle new operation event for general info.
        
        Args:
            operation_data: Dictionary containing operation information
        """
        op = {
            'operation': operation_data.get('operation', 'Unknown'),
            'time': datetime.fromtimestamp(operation_data.get('timestamp', 0)).strftime('%H:%M:%S')
        }
        self.items.append(op)
        self._add_item_to_display(op, lambda o: [
            f"[{o['time']}] {o['operation']}"
        ])
        self._update_stats()
    
    def _update_stats(self):
        """
        Update statistics labels for general info.
        
        Updates status, operations count, pipeline status, and runtime info.
        """
        # Update operations count
        self.widgets['operations_count'].config(text=str(len(self.items)))
        
        # Read current status
        status, last_op, timestamp, operation_id = self.gui.read_status()
        
        # Update status with color
        status_colors = {
            "IDLE": "#4ec9b0",      # Green
            "THINKING": "#dcdcaa",   # Yellow
            "ERROR": "#f48771",      # Red
            "PROCESSING": "#569cd6"  # Blue
        }
        self.widgets['status'].config(text=status, foreground=status_colors.get(status, "#ffffff"))
        self.widgets['last_operation'].config(text=last_op)
        
        # Update pipeline status
        from config.settings import paths
        index_path = str(paths.FAISS_INDEX_FILE)
        docs_path = str(paths.DOCS_DIR)
        
        if os.path.exists(index_path):
            index_size = os.path.getsize(index_path) / 1024  # KB
            self.widgets['index_status'].config(text=f"Loaded ({index_size:.1f} KB)", foreground="#4ec9b0")
        else:
            self.widgets['index_status'].config(text="Not found", foreground="#f48771")
        
        if os.path.exists(docs_path):
            doc_files = [f for f in os.listdir(docs_path) if f.endswith(('.pdf', '.docx', '.txt', '.md'))]
            self.widgets['documents'].config(text=f"{len(doc_files)} files", foreground="#4ec9b0")
        else:
            self.widgets['documents'].config(text="Folder not found", foreground="#f48771")
        
        # Update runtime info
        import time
        uptime = time.time() - self.gui.start_time
        self.widgets['uptime'].config(text=self.gui.format_uptime(uptime))
        self.widgets['current_time'].config(text=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        # Update status bar
        if timestamp > 0:
            age = time.time() - timestamp
            self.statusbar.config(text=f"Monitor running | Auto-refresh | Last status: {age:.1f}s ago")
    
    def _schedule_stats_update(self):
        """Schedule periodic stats update (every second)"""
        if hasattr(self, 'stats_update_job'):
            self.gui.root.after_cancel(self.stats_update_job)
        self._update_stats()
        self.stats_update_job = self.gui.root.after(1000, self._schedule_stats_update)
    
    def stop_auto_refresh(self):
        """Stop the periodic refresh and stats update"""
        super().stop_auto_refresh()
        if hasattr(self, 'stats_update_job'):
            self.gui.root.after_cancel(self.stats_update_job)
    
    def _extract_operations(self, operations):
        """
        Extract operation data from operations.
        
        Args:
            operations: List of all operation dictionaries
            
        Returns:
            List of operation data dictionaries
        """
        ops = []
        for op in operations:
            ops.append({
                'operation': op.get('operation', 'Unknown'),
                'time': datetime.fromtimestamp(op.get('timestamp', 0)).strftime('%H:%M:%S')
            })
        return ops
