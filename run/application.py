#!/usr/bin/env python3
"""
Tkinter Desktop Interface for Document Q&A
Secure desktop version - provides a native UI for document upload,
processing, and Q&A with streaming responses.
"""


# ============================================================================
# CONSTANTS & IMPORTS
# ============================================================================

# Standard library imports
import os
import sys
import logging
import subprocess
import time
import threading
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import shutil

# Local imports - configuration and utilities
from config.settings import paths, performance_config, file_config
from utils.helpers import get_document_files, count_document_files
from core.pipeline import DocumentPipeline


# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Global pipeline instance (lazy-loaded)
pipeline = None


# Session storage (simple in-memory for single-user)
session_data = {
    'chat_history': [],  # List of chat messages with type and content
    'status': None,      # Current status message dict with type and message
    'documents': []      # List of uploaded document filenames
}


# ============================================================================
# CORE LOGIC - PIPELINE MANAGEMENT
# ============================================================================

def clean_startup():
    """
    Clean startup routine - validates and removes incomplete index files.
    
    Checks for both FAISS index and PKL files. Only removes index directory
    if both files are completely missing to avoid partial index corruption.
    
    Returns:
        bool: True if cleanup completed successfully, False on error
    """
    try:
        index_dir = paths.INDEX_DIR
        if index_dir.exists():
            # Check if index is complete
            faiss_file = paths.FAISS_INDEX_FILE
            pkl_file = paths.PKL_INDEX_FILE
            
            logger.info(f"Checking index files:")
            logger.info(f"FAISS file exists: {faiss_file.exists()} - {faiss_file}")
            logger.info(f"PKL file exists: {pkl_file.exists()} - {pkl_file}")
            
            # Only clean if BOTH files are missing (not if one is missing)
            if not faiss_file.exists() and not pkl_file.exists():
                logger.info("🧹 No index files found - cleaning empty directory...")
                shutil.rmtree(index_dir, ignore_errors=True)
                logger.info("✅ Cleaned up empty index directory")
            elif faiss_file.exists() and pkl_file.exists():
                logger.info("✅ Complete index found - keeping existing files")
            else:
                logger.warning(f"⚠️ Partial index found - keeping files but may need reprocessing")
        
        return True
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        return False


def get_pipeline():
    """
    Get or initialize the document processing pipeline (lazy loading).
    
    Creates pipeline instance on first call, performs startup cleanup,
    and auto-processes documents if no existing index is found.
    
    Returns:
        DocumentPipeline: Initialized pipeline instance or None on failure
    """
    global pipeline
    if pipeline is None:
        try:
            # Clean startup first
            clean_startup()
            
            # Initialize pipeline (it will auto-load existing index)
            pipeline = DocumentPipeline()
            
            # Auto-process documents if no index was loaded
            if pipeline.vectorstore is None:
                logger.info("🔄 No index found - automatically processing documents...")
                success = pipeline.process_documents()
                if success:
                    logger.info("✅ Documents auto-processed successfully!")
                else:
                    logger.error("❌ Auto-processing failed")
            else:
                logger.info("✅ Pipeline initialized with existing index")
                
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            pipeline = None
    return pipeline


# ============================================================================
# UTILITIES & HELPERS
# ============================================================================

def get_documents():
    """
    Get list of uploaded document filenames from the documents directory.
    
    Returns:
        list: List of document filenames (empty list if directory doesn't exist)
    """
    doc_files = get_document_files()
    return [f.name for f in doc_files]


# ============================================================================
# TKINTER APPLICATION CLASS
# ============================================================================

class DocumentQAApp:
    """
    Main Tkinter application class for Document Q&A.
    
    Provides a desktop interface with:
    - Document upload and management
    - Real-time chat interface with streaming responses
    - Status messages and document listing
    """
    
    def __init__(self, root):
        """
        Initialize the Tkinter application.
        
        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("📚 Document Q&A")
        self.root.geometry("1200x700")
        
        # Configure max file size
        self.max_content_length = performance_config.MAX_FILE_SIZE_MB * 1024 * 1024
        self.upload_folder = str(paths.DOCS_DIR)
        
        # Setup UI
        self.setup_styles()
        self.create_ui()
        
        # Initialize pipeline in background
        self.init_pipeline_async()
        
        # Load initial documents
        self.refresh_documents()
    
    def setup_styles(self):
        """Setup modern UI styling for ttk widgets."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure custom styles
        style.configure('Title.TLabel', 
                       font=('Arial', 16, 'bold'))
        style.configure('Status.TLabel',
                       font=('Arial', 10),
                       padding=10)
        style.configure('Success.TLabel',
                       background='#d4edda',
                       foreground='#155724')
        style.configure('Error.TLabel',
                       background='#f8d7da',
                       foreground='#721c24')
        style.configure('Loading.TLabel',
                       background='#fff3cd',
                       foreground='#856404')
    
    def create_ui(self):
        """Create the main user interface."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="📚 Document Q&A", style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=2)
        main_frame.rowconfigure(1, weight=1)
        
        # Left section - Upload
        self.create_upload_section(main_frame)
        
        # Right section - Chat
        self.create_chat_section(main_frame)
    
    def create_upload_section(self, parent):
        """
        Create the document upload section.
        
        Args:
            parent: Parent widget to attach this section to
        """
        upload_frame = ttk.LabelFrame(parent, text="📄 Upload Documents", padding="15")
        upload_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Upload button
        upload_btn = ttk.Button(upload_frame, text="📤 Upload File", command=self.upload_file)
        upload_btn.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Documents label
        docs_label = ttk.Label(upload_frame, text="Documents:", font=('Arial', 10, 'bold'))
        docs_label.grid(row=1, column=0, sticky=tk.W, pady=(10, 5))
        
        # Document list with scrollbar
        list_frame = ttk.Frame(upload_frame)
        list_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.doc_listbox = tk.Listbox(list_frame, 
                                       yscrollcommand=scrollbar.set,
                                       font=('Arial', 9),
                                       height=15)
        self.doc_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.doc_listbox.yview)
        
        # Remove all button
        remove_btn = ttk.Button(upload_frame, text="🗑️ Remove All Documents", 
                               command=self.remove_all_documents)
        remove_btn.grid(row=3, column=0, sticky=(tk.W, tk.E))
        
        # Configure grid weights
        upload_frame.columnconfigure(0, weight=1)
        upload_frame.rowconfigure(2, weight=1)
    
    def create_chat_section(self, parent):
        """
        Create the chat section with message display and input.
        
        Args:
            parent: Parent widget to attach this section to
        """
        chat_frame = ttk.Frame(parent)
        chat_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Chat header
        chat_label = ttk.Label(chat_frame, text="💬 Ask Questions", font=('Arial', 12, 'bold'))
        chat_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 10))
        
        # Status message (hidden by default)
        self.status_label = ttk.Label(chat_frame, text="", style='Status.TLabel')
        self.status_label.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        self.status_label.grid_remove()  # Hide initially
        
        # Chat box with scrollbar
        chat_box_frame = ttk.Frame(chat_frame)
        chat_box_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Use Text widget instead of scrolledtext for better control
        self.chat_box = tk.Text(chat_box_frame,
                               wrap=tk.WORD,
                               font=('Segoe UI', 10),
                               height=20,
                               state=tk.DISABLED,
                               bg='white',
                               relief=tk.SOLID,
                               borderwidth=1)
        
        scrollbar = ttk.Scrollbar(chat_box_frame, command=self.chat_box.yview)
        self.chat_box.configure(yscrollcommand=scrollbar.set)
        
        self.chat_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Configure text tags for styling
        self.chat_box.tag_config('user', 
                                justify=tk.RIGHT,
                                foreground='white',
                                background='#007AFF',
                                font=('Segoe UI', 10),
                                spacing1=5,
                                spacing3=5,
                                lmargin1=200,
                                lmargin2=200,
                                rmargin=10)
        
        self.chat_box.tag_config('bot',
                                justify=tk.LEFT,
                                foreground='#333',
                                background='#F0F0F0',
                                font=('Segoe UI', 10),
                                spacing1=5,
                                spacing3=5,
                                lmargin1=10,
                                lmargin2=10,
                                rmargin=200)
        
        self.chat_box.tag_config('sources',
                                foreground='#856404',
                                background='#fff3e0',
                                font=('Segoe UI', 9),
                                lmargin1=10,
                                lmargin2=10,
                                rmargin=10)
        
        # Question input frame
        input_frame = ttk.Frame(chat_frame)
        input_frame.grid(row=3, column=0, sticky=(tk.W, tk.E))
        
        self.question_entry = ttk.Entry(input_frame, font=('Arial', 10))
        self.question_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        self.question_entry.bind('<Return>', lambda e: self.ask_streaming())
        
        self.ask_button = ttk.Button(input_frame, text="Ask", command=self.ask_streaming)
        self.ask_button.grid(row=0, column=1)
        
        # Configure grid weights
        chat_frame.columnconfigure(0, weight=1)
        chat_frame.rowconfigure(2, weight=1)
        input_frame.columnconfigure(0, weight=1)
    
    def init_pipeline_async(self):
        """Initialize pipeline in background thread."""
        def init():
            try:
                get_pipeline()
                self.update_status('success', '✅ System ready')
            except Exception as e:
                logger.error(f"Pipeline initialization failed: {e}")
                self.update_status('error', f'❌ Initialization failed: {e}')
        
        thread = threading.Thread(target=init, daemon=True)
        thread.start()
    
    def refresh_documents(self):
        """Refresh document list display."""
        session_data['documents'] = get_documents()
        
        # Update listbox
        self.doc_listbox.delete(0, tk.END)
        for doc in session_data['documents']:
            self.doc_listbox.insert(tk.END, doc)
    
    def update_status(self, status_type, message):
        """
        Update status message display.
        
        Args:
            status_type: Type of status ('success', 'error', 'loading')
            message: Status message text
        """
        session_data['status'] = {'type': status_type, 'message': message}
        
        # Update status label
        self.status_label.config(text=message)
        
        # Set style based on type
        if status_type == 'success':
            self.status_label.config(style='Success.TLabel')
        elif status_type == 'error':
            self.status_label.config(style='Error.TLabel')
        elif status_type == 'loading':
            self.status_label.config(style='Loading.TLabel')
        
        # Show status label
        self.status_label.grid()
        
        # Auto-hide after 5 seconds
        self.root.after(5000, self.status_label.grid_remove)
    
    def upload_file(self):
        """
        Handle file upload and automatic processing.
        """
        try:
            # Open file dialog
            filetypes = [
                ("All Supported", " ".join(f"*{ext}" for ext in file_config.ALL_EXTENSIONS)),
                ("PDF files", "*.pdf"),
                ("Text files", "*.txt"),
                ("Markdown files", "*.md"),
                ("Word files", "*.docx"),
                ("Excel files", "*.xlsx"),
            ]
            
            filepath = filedialog.askopenfilename(
                title="Select Document to Upload",
                filetypes=filetypes
            )
            
            if not filepath:
                return  # User cancelled
            
            file_path = Path(filepath)
            filename = file_path.name
            
            # Secure filename (basic sanitization)
            filename = "".join(c for c in filename if c.isalnum() or c in (' ', '.', '_', '-')).strip()
            
            # Check file size
            file_size = file_path.stat().st_size
            if file_size > self.max_content_length:
                messagebox.showerror("File Too Large",
                                   f"File size exceeds {performance_config.MAX_FILE_SIZE_MB} MB limit")
                return
            
            # Create upload directory
            upload_path = Path(self.upload_folder)
            upload_path.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            dest_path = upload_path / filename
            shutil.copy2(file_path, dest_path)
            
            logger.info(f"File uploaded: {filename}")
            
            # Update status and refresh list
            self.update_status('loading', f'🔄 Processing new file: {filename}...')
            self.refresh_documents()
            
            # Auto-process in background
            def process():
                current_pipeline = get_pipeline()
                if current_pipeline:
                    success = current_pipeline.process_single_document(dest_path)
                    if success:
                        self.root.after(0, lambda: self.update_status('success', 
                                       f'✅ Uploaded and processed: {filename}'))
                        logger.info(f"New document processed: {filename}")
                    else:
                        self.root.after(0, lambda: self.update_status('error',
                                       f'⚠️ Uploaded {filename} but processing failed'))
                        logger.error(f"Single document processing failed: {filename}")
                else:
                    self.root.after(0, lambda: self.update_status('error',
                                   f'⚠️ Uploaded {filename} but pipeline unavailable'))
            
            thread = threading.Thread(target=process, daemon=True)
            thread.start()
            
        except Exception as e:
            self.update_status('error', f'Upload failed: {str(e)}')
            logger.error(f"Upload error: {e}")
    
    def remove_all_documents(self):
        """
        Remove all documents and clear the vector index.
        """
        # Confirm with user
        if not messagebox.askyesno("Confirm Removal", 
                                   "Remove all documents and clear the index?"):
            return
        
        try:
            self.update_status('loading', '🗑️ Removing all documents...')
            
            # Clear in background
            def remove():
                global pipeline
                
                # Clear documents directory
                docs_dir = paths.DOCS_DIR
                if docs_dir.exists():
                    for file_path in docs_dir.iterdir():
                        if file_path.is_file():
                            file_path.unlink()
                            logger.info(f"Deleted document: {file_path.name}")
                
                # Clear index directory
                index_dir = paths.INDEX_DIR
                if index_dir.exists():
                    shutil.rmtree(index_dir, ignore_errors=True)
                    logger.info("Deleted index directory")
                
                # Reset pipeline
                pipeline = None
                
                # Clear chat history
                session_data['chat_history'] = []
                
                # Update UI
                self.root.after(0, self.refresh_documents)
                self.root.after(0, self.clear_chat_display)
                self.root.after(0, lambda: self.update_status('success', 
                               '✅ All documents removed successfully!'))
                
                logger.info("All documents and index removed successfully")
            
            thread = threading.Thread(target=remove, daemon=True)
            thread.start()
                
        except Exception as e:
            self.update_status('error', f'Remove error: {str(e)}')
            logger.error(f"Remove error: {e}")
    
    def clear_chat_display(self):
        """Clear the chat display."""
        self.chat_box.config(state=tk.NORMAL)
        self.chat_box.delete(1.0, tk.END)
        self.chat_box.config(state=tk.DISABLED)
    
    def add_message(self, msg_type, content, sources=None):
        """
        Add message to chat display.
        
        Args:
            msg_type: Type of message ('user' or 'bot')
            content: Message content text
            sources: Optional list of source documents
        """
        self.chat_box.config(state=tk.NORMAL)
        
        if msg_type == 'user':
            self.chat_box.insert(tk.END, f"\n{content}\n", 'user')
        elif msg_type == 'bot':
            self.chat_box.insert(tk.END, f"\n{content}\n", 'bot')
            
            # Add sources if provided
            if sources:
                sources_text = "\n📚 Sources:\n"
                for source in sources:
                    source_preview = source.get('content', '')[:100]
                    sources_text += f"• {source['source']}: {source_preview}...\n"
                self.chat_box.insert(tk.END, sources_text + "\n", 'sources')
        
        self.chat_box.see(tk.END)
        self.chat_box.config(state=tk.DISABLED)
    
    def ask_streaming(self):
        """
        Handle question asking with streaming responses.
        """
        question = self.question_entry.get().strip()
        if not question:
            return
        
        current_pipeline = get_pipeline()
        if not current_pipeline or not current_pipeline.vectorstore:
            messagebox.showwarning("Not Ready", "Please process documents first")
            return
        
        # Clear input
        self.question_entry.delete(0, tk.END)
        
        # Add user message
        self.add_message('user', question)
        
        # Store current chat position
        session_data['chat_history'].append({
            'type': 'user',
            'content': question,
            'sources': None
        })
        
        # Disable button during processing
        self.ask_button.config(state=tk.DISABLED, text="Thinking...")
        
        # Stream response in background
        def stream():
            try:
                # Prepare bot message area
                self.chat_box.config(state=tk.NORMAL)
                self.chat_box.insert(tk.END, "\n", 'bot')
                bot_start = self.chat_box.index("end-1c")
                
                # Stream tokens
                response_text = ""
                for token in current_pipeline.ask_streaming(question):
                    response_text += token
                    self.chat_box.insert(tk.END, token, 'bot')
                    self.chat_box.see(tk.END)
                
                self.chat_box.insert(tk.END, "\n", 'bot')
                self.chat_box.config(state=tk.DISABLED)
                
                # Store in history
                session_data['chat_history'].append({
                    'type': 'bot',
                    'content': response_text,
                    'sources': None  # Sources not available in streaming mode
                })
                
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                self.root.after(0, lambda: self.add_message('bot', f"Error: {str(e)}"))
            finally:
                # Re-enable button
                self.root.after(0, lambda: self.ask_button.config(state=tk.NORMAL, text="Ask"))
        
        thread = threading.Thread(target=stream, daemon=True)
        thread.start()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    """
    Main entry point - initializes system, launches monitoring GUI, and starts Tkinter app.
    """
    print("🚀 Document Q&A - Tkinter Desktop Interface")
    print("=" * 50)
    
    # Launch live monitoring GUI (no console window needed)
    try:
        print("📊 Starting live system monitor GUI...")
        # Get the project root directory
        project_root = Path(__file__).parent.parent.absolute()
        
        # Launch monitoring dashboard as module
        python_exe = sys.executable
        # Use pythonw.exe instead of python.exe to hide console
        if python_exe.endswith('python.exe'):
            pythonw_exe = python_exe.replace('python.exe', 'pythonw.exe')
        else:
            pythonw_exe = python_exe
        
        subprocess.Popen(
            [pythonw_exe, "-m", "monitoring.dashboard"],
            cwd=str(project_root),
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
        )
        print("✅ Live monitor GUI launched")
    except Exception as e:
        print(f"⚠️ Failed to launch monitor: {e}")
        print("   (App will continue without monitor)")
    
    # Check documents status
    doc_count = count_document_files()
    if doc_count > 0:
        print(f"📄 Found {doc_count} documents ready for processing")
    else:
        print("📂 Documents directory will be created on first upload")
    
    print("✅ Starting desktop application...")
    
    # Create and run Tkinter app
    root = tk.Tk()
    app = DocumentQAApp(root)
    root.mainloop()
