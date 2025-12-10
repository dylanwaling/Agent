"""
Live System Monitoring for Document Q&A Agent
Task-manager-style real-time display showing system status and metrics
"""

import os
import sys
import time
import threading
from datetime import datetime
import psutil
from backend_logic import DocumentPipeline

# ANSI escape codes for colors and cursor control
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RED = '\033[91m'
    GRAY = '\033[90m'

class LiveMonitor:
    def __init__(self):
        self.pipeline = None
        self.status = "IDLE"
        self.last_operation = "System started"
        self.operation_count = 0
        self.start_time = time.time()
        self.running = True
        self.lock = threading.Lock()
        
    def clear_screen(self):
        """Clear the terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
        
    def set_status(self, status, operation=None):
        """Update the current status"""
        with self.lock:
            self.status = status.upper()
            if operation:
                self.last_operation = operation
                self.operation_count += 1
                
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
        
    def display(self):
        """Display the live monitoring interface"""
        while self.running:
            self.clear_screen()
            
            # Get current metrics
            with self.lock:
                status = self.status
                last_op = self.last_operation
                op_count = self.operation_count
                
            uptime = time.time() - self.start_time
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            process = psutil.Process()
            process_memory = process.memory_info().rss / (1024**2)  # MB
            
            # Header
            print(f"{Colors.BOLD}{Colors.CYAN}╔════════════════════════════════════════════════════════════════════════╗{Colors.RESET}")
            print(f"{Colors.BOLD}{Colors.CYAN}║          DOCUMENT Q&A AGENT - LIVE SYSTEM MONITOR                      ║{Colors.RESET}")
            print(f"{Colors.BOLD}{Colors.CYAN}╚════════════════════════════════════════════════════════════════════════╝{Colors.RESET}")
            print()
            
            # Process Status (Main feature requested)
            status_color = Colors.GREEN if status == "IDLE" else Colors.YELLOW
            print(f"{Colors.BOLD}Process Status:{Colors.RESET} {status_color}{status}{Colors.RESET}")
            print(f"{Colors.BOLD}Last Operation:{Colors.RESET} {last_op}")
            print(f"{Colors.BOLD}Operations Count:{Colors.RESET} {op_count}")
            print()
            
            # System Information
            print(f"{Colors.BOLD}{Colors.BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Colors.RESET}")
            print(f"{Colors.BOLD}SYSTEM METRICS{Colors.RESET}")
            print(f"{Colors.BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Colors.RESET}")
            print()
            
            # CPU and Memory
            cpu_bar = self._create_bar(cpu_percent, 100, 30)
            memory_bar = self._create_bar(memory.percent, 100, 30)
            
            print(f"{Colors.BOLD}CPU Usage:{Colors.RESET}     {cpu_bar} {cpu_percent:.1f}%")
            print(f"{Colors.BOLD}Memory Usage:{Colors.RESET}  {memory_bar} {memory.percent:.1f}% ({memory.used / (1024**3):.1f}/{memory.total / (1024**3):.1f} GB)")
            print(f"{Colors.BOLD}Process Memory:{Colors.RESET} {process_memory:.1f} MB")
            print()
            
            # GPU Info
            gpu_info = self.get_gpu_info()
            print(f"{Colors.BOLD}GPU:{Colors.RESET} {gpu_info}")
            print()
            
            # Pipeline Information
            print(f"{Colors.BOLD}{Colors.BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Colors.RESET}")
            print(f"{Colors.BOLD}PIPELINE STATUS{Colors.RESET}")
            print(f"{Colors.BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Colors.RESET}")
            print()
            
            if self.pipeline:
                # Check if index exists
                index_path = "data/index/faiss_index/index.faiss"
                if os.path.exists(index_path):
                    index_size = os.path.getsize(index_path) / 1024  # KB
                    print(f"{Colors.GREEN}● Index Status:{Colors.RESET} Loaded ({index_size:.1f} KB)")
                else:
                    print(f"{Colors.RED}● Index Status:{Colors.RESET} Not found")
                    
                # Check documents folder
                docs_path = "data/documents"
                if os.path.exists(docs_path):
                    doc_files = [f for f in os.listdir(docs_path) if f.endswith(('.pdf', '.docx', '.txt', '.md'))]
                    print(f"{Colors.GREEN}● Documents:{Colors.RESET} {len(doc_files)} files in {docs_path}")
                else:
                    print(f"{Colors.RED}● Documents:{Colors.RESET} Folder not found")
            else:
                print(f"{Colors.YELLOW}● Pipeline:{Colors.RESET} Not initialized")
            print()
            
            # Runtime Info
            print(f"{Colors.BOLD}{Colors.BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Colors.RESET}")
            print(f"{Colors.BOLD}RUNTIME INFORMATION{Colors.RESET}")
            print(f"{Colors.BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Colors.RESET}")
            print()
            print(f"{Colors.BOLD}Uptime:{Colors.RESET} {self.format_uptime(uptime)}")
            print(f"{Colors.BOLD}Current Time:{Colors.RESET} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print()
            
            # Footer
            print(f"{Colors.GRAY}Press Ctrl+C to exit monitoring{Colors.RESET}")
            
            time.sleep(1)  # Refresh every second
            
    def _create_bar(self, value, max_value, width=30):
        """Create a text progress bar"""
        filled = int((value / max_value) * width)
        bar = '█' * filled + '░' * (width - filled)
        
        # Color based on percentage
        if value < 50:
            color = Colors.GREEN
        elif value < 80:
            color = Colors.YELLOW
        else:
            color = Colors.RED
            
        return f"{color}[{bar}]{Colors.RESET}"
        
    def initialize_pipeline(self):
        """Initialize the document pipeline"""
        self.set_status("THINKING", "Initializing pipeline...")
        try:
            self.pipeline = DocumentPipeline()
            self.set_status("IDLE", "Pipeline initialized successfully")
        except Exception as e:
            self.set_status("ERROR", f"Failed to initialize pipeline: {str(e)}")
            
    def monitor_operation(self, operation_name, func, *args, **kwargs):
        """Wrap an operation with status monitoring"""
        self.set_status("THINKING", operation_name)
        try:
            result = func(*args, **kwargs)
            self.set_status("IDLE", f"Completed: {operation_name}")
            return result
        except Exception as e:
            self.set_status("ERROR", f"Failed: {operation_name} - {str(e)}")
            raise
            
    def run(self):
        """Run the live monitor"""
        print("Starting Live Monitor...")
        print("Initializing system...")
        
        # Start display in a separate thread
        display_thread = threading.Thread(target=self.display, daemon=True)
        display_thread.start()
        
        # Initialize pipeline
        time.sleep(0.5)  # Brief delay to show initial screen
        self.initialize_pipeline()
        
        # Main loop - keep running and monitoring
        try:
            while True:
                time.sleep(1)
                # Could add periodic health checks here
                
        except KeyboardInterrupt:
            print("\n\nShutting down monitor...")
            self.running = False
            display_thread.join(timeout=2)
            print("Monitor stopped.")


if __name__ == "__main__":
    monitor = LiveMonitor()
    monitor.run()
