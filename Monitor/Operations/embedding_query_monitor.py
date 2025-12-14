"""
Embedding Query Monitor - Tracks query embedding process
"""

import logging
from datetime import datetime
from monitor.performance_monitor import BaseMonitor

logger = logging.getLogger(__name__)


class EmbeddingQueryMonitor(BaseMonitor):
    """
    Monitor for query embedding process.
    
    Tracks embedding generation with model info, vector dimensions,
    device usage (GPU/CPU), and performance metrics.
    """
    
    def show(self):
        """
        Display the Embedding Query monitor view.
        
        Creates UI with model info, embedding statistics, and operation log.
        """
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
        
        # Start auto-refresh to get updates from file
        self.start_auto_refresh(self._extract_embeddings, lambda e: [
            f"[{e['time']}] {e['query'][:60]}...",
            f"  → Model: {e['model']}, Dim: {e['dim']}, Device: {e['device']}",
            ""
        ])
    
    def on_new_operation(self, operation_data):
        """
        Handle new operation event for embedding queries.
        
        Args:
            operation_data: Dictionary containing operation information
        """
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
        """
        Update statistics labels for embeddings.
        
        Updates device status, total embeddings, and average processing time.
        """
        self._update_device_status()
        self.widgets['total'].config(text=str(len(self.items)))
        if self.items:
            avg_time = sum(e.get('time_ms', 0) for e in self.items) / len(self.items)
            self.widgets['avg_time'].config(text=f"{avg_time/1000:.3f}s")
        else:
            self.widgets['avg_time'].config(text="0.000s")
    
    def _update_device_status(self):
        """
        Check and update GPU/CPU status.
        
        Detects CUDA availability and updates the device label with appropriate color.
        """
        try:
            import torch
            device = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
            color = "#4ec9b0" if torch.cuda.is_available() else "#dcdcaa"
            self.widgets['device'].config(text=device, foreground=color)
        except:
            self.widgets['device'].config(text="CPU", foreground="#dcdcaa")
    
    def _extract_embeddings(self, operations):
        """
        Extract embedding data from operations.
        
        Args:
            operations: List of all operation dictionaries
            
        Returns:
            List of embedding data dictionaries
        """
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
