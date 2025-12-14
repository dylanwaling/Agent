"""
Question Input Monitor - Tracks incoming user questions
"""

from datetime import datetime
from Monitor.performance_monitor import BaseMonitor


class QuestionInputMonitor(BaseMonitor):
    """
    Monitor for incoming user questions.
    
    Displays real-time question submissions with character counts,
    streaming indicators, and statistics.
    """
    
    def show(self):
        """
        Display the Question Input monitor view.
        
        Creates UI with question statistics and scrollable question log.
        """
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
        
        # Start auto-refresh to get updates from file
        self.start_auto_refresh(self._extract_questions, lambda q: [
            f"[{q['time']}] ({q['length']} chars) {'[STREAMING]' if q.get('streaming') else ''}",
            f"  {q['question']}",
            ""
        ])
    
    def on_new_operation(self, operation_data):
        """
        Handle new operation event for question input.
        
        Args:
            operation_data: Dictionary containing operation information
        """
        op_type = operation_data.get('operation_type', '')
        if op_type == 'question_input':
            metadata = operation_data.get('metadata', {})
            question = metadata.get('question', operation_data.get('operation', 'N/A'))
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
        """
        Update statistics labels for questions.
        
        Calculates and displays total questions, average length, and last question.
        """
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
        """
        Extract question data from operations.
        
        Args:
            operations: List of all operation dictionaries
            
        Returns:
            List of question data dictionaries
        """
        questions = []
        for op in operations:
            op_type = op.get('operation_type', '')
            if op_type == 'question_input':
                metadata = op.get('metadata', {})
                question = metadata.get('question', op.get('operation', 'N/A'))
                questions.append({
                    'question': question,
                    'time': datetime.fromtimestamp(op.get('timestamp', 0)).strftime('%H:%M:%S'),
                    'length': metadata.get('question_length', len(question)),
                    'streaming': metadata.get('streaming', False)
                })
        return questions
