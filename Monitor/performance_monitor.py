"""
Base Monitor Class for Performance Monitor Views
Provides reusable UI components and data management for monitor screens.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


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
