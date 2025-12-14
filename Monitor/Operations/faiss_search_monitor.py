"""
FAISS Search Monitor - Tracks vector search operations
"""

import os
import logging
from datetime import datetime
from monitor.performance_monitor import BaseMonitor

logger = logging.getLogger(__name__)


class FAISSSearchMonitor(BaseMonitor):
    """
    Monitor for FAISS vector search operations.
    
    Displays search operations with index status, vector counts,
    K values, result counts, and search timing metrics.
    """
    
    def show(self):
        """
        Display the FAISS Search monitor view.
        
        Creates UI with index information, search statistics, and operation log.
        """
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
        
        # Start auto-refresh to get updates from file
        self.start_auto_refresh(self._extract_searches, lambda s: [
            f"[{s['time']}] Query: {s['query'][:50]}...",
            f"  → K={s['k']}, Results={s['num_results']}, Index={s['index_size']}, Time={s['time_ms']:.2f}ms",
            ""
        ])
    
    def on_new_operation(self, operation_data):
        """
        Handle new operation event for FAISS searches.
        
        Args:
            operation_data: Dictionary containing operation information
        """
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
        """
        Update statistics labels for FAISS searches.
        
        Updates index status, total searches, average results, and average search time.
        """
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
        """
        Check and update FAISS index status.
        
        Checks index file existence, size, and vector count from the pipeline.
        """
        # Import here to avoid circular dependency
        from config.settings import paths
        index_path = str(paths.FAISS_INDEX_FILE)
        
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
        """
        Extract search data from operations.
        
        Args:
            operations: List of all operation dictionaries
            
        Returns:
            List of search data dictionaries
        """
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
