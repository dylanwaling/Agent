#!/usr/bin/env python3
"""
Lightweight Flask Web Interface for Document Q&A
Simple, fast alternative to Streamlit - provides a clean web UI for document upload,
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
from pathlib import Path

# Third-party imports
from flask import Flask, request, render_template_string, jsonify, redirect, url_for, Response
from werkzeug.utils import secure_filename

# Local imports - configuration and utilities
from config import paths, performance_config, file_config
from utils import get_document_files, count_document_files
from backend_logic import DocumentPipeline


# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Flask application configuration
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = performance_config.MAX_FILE_SIZE_MB * 1024 * 1024
app.config['UPLOAD_FOLDER'] = str(paths.DOCS_DIR)


# Global pipeline instance (lazy-loaded)
pipeline = None


# HTML template (embedded for simplicity and single-file deployment)
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Document Q&A</title>
    <meta charset="UTF-8">
    <style>
        body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
        .container { display: flex; gap: 20px; }
        .upload-section { flex: 1; }
        .chat-section { flex: 2; }
        .document-list { background: #f5f5f5; padding: 15px; border-radius: 5px; margin: 10px 0; }
        .chat-box { height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 15px; margin: 10px 0; }
        .message { 
            margin: 15px 0; 
            padding: 0;
            clear: both;
        }
        .message .content {
            max-width: 70%;
            padding: 12px 16px;
            border-radius: 18px;
            white-space: pre-wrap; 
            word-wrap: break-word; 
            line-height: 1.5; 
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            font-size: 14px;
        }
        .user-message {
            display: flex;
            justify-content: flex-end;
        }
        .user-message .content {
            background: #007AFF;
            color: white;
            border-bottom-right-radius: 5px;
        }
        .bot-message {
            display: flex;
            justify-content: flex-start;
        }
        .bot-message .content {
            background: #F0F0F0;
            color: #333;
            border-bottom-left-radius: 5px;
        }
        .sources { 
            background: #fff3e0; 
            padding: 8px 12px; 
            margin: 8px 0 0 0; 
            border-radius: 12px; 
            font-size: 0.85em; 
            border-left: 3px solid #ff9800;
            max-width: 100%;
        }
        input[type="text"] { width: 70%; padding: 10px; }
        button { padding: 10px 20px; background: #2196F3; color: white; border: none; border-radius: 3px; cursor: pointer; }
        button:hover { background: #1976D2; }
        .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
        .success { background: #d4edda; color: #155724; }
        .error { background: #f8d7da; color: #721c24; }
        .loading { background: #fff3cd; color: #856404; }
        #cursor {
            animation: blink 1s infinite;
            font-weight: bold;
            color: #333;
        }
        @keyframes blink {
            0%, 50% { opacity: 1; }
            51%, 100% { opacity: 0; }
        }
    </style>
</head>
<body>
    <h1>📚 Document Q&A</h1>
    
    <div class="container">
        <!-- Upload Section -->
        <div class="upload-section">
            <h3>📄 Upload Documents</h3>
            <form action="/upload" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept=".pdf,.txt,.md,.docx,.xlsx" required>
                <button type="submit">Upload</button>
            </form>
            
            <div class="document-list">
                <h4>Documents:</h4>
                <ul>
                {% for doc in documents %}
                    <li>{{ doc }}</li>
                {% endfor %}
                </ul>
            </div>
            
            <form action="/remove_all" method="post" style="margin-top: 20px;">
                <button type="submit">�️ Remove All Documents</button>
            </form>
        </div>
        
        <!-- Chat Section -->
        <div class="chat-section">
            <h3>💬 Ask Questions</h3>
            
            {% if status %}
            <div class="status {{ status.type }}">{{ status.message }}</div>
            {% endif %}
            
            <div class="chat-box" id="chatBox">
                {% for msg in chat_history %}
                <div class="message {{ 'user-message' if msg.type == 'user' else 'bot-message' }}">
                    <div class="content">{{ msg.content }}</div>
                    {% if msg.sources %}
                    <div class="sources">
                        <strong>📚 Sources:</strong>
                        {% for source in msg.sources %}
                        <div>• {{ source.source }}: {{ source.content[:100] }}...</div>
                        {% endfor %}
                    </div>
                    {% endif %}
                </div>
                {% endfor %}
            </div>
            
            <form onsubmit="askStreaming(); return false;">
                <input type="text" id="questionInput" name="question" placeholder="Ask a question about your documents..." required>
                <button type="submit" id="streamButton">Ask</button>
            </form>
        </div>
    </div>
    
    <script>
        // Auto-scroll chat to bottom
        function scrollToBottom() {
            var chatBox = document.getElementById('chatBox');
            chatBox.scrollTop = chatBox.scrollHeight;
        }
        scrollToBottom();
        
        // Streaming function
        function askStreaming() {
            const question = document.getElementById('questionInput').value.trim();
            if (!question) return;
            
            const button = document.getElementById('streamButton');
            const chatBox = document.getElementById('chatBox');
            
            // Generate unique IDs for this conversation
            const timestamp = Date.now();
            const textId = 'streamingText_' + timestamp;
            const cursorId = 'cursor_' + timestamp;
            
            // Add user message to chat with proper bubble styling
            const userMsg = document.createElement('div');
            userMsg.className = 'message user-message';
            userMsg.innerHTML = '<div class="content">' + question + '</div>';
            chatBox.appendChild(userMsg);
            
            // Add bot message container with proper bubble styling and unique IDs
            const botMsg = document.createElement('div');
            botMsg.className = 'message bot-message';
            botMsg.innerHTML = '<div class="content"><span id="' + textId + '"></span><span id="' + cursorId + '">█</span></div>';
            chatBox.appendChild(botMsg);
            
            const streamingText = document.getElementById(textId);
            const cursor = document.getElementById(cursorId);
            
            button.disabled = true;
            button.textContent = 'Thinking...';
            scrollToBottom();
            
            fetch('/stream?question=' + encodeURIComponent(question))
                .then(response => response.body.getReader())
                .then(reader => {
                    function readStream() {
                        return reader.read().then(({ done, value }) => {
                            if (done) {
                                cursor.style.display = 'none'; // Hide cursor when done
                                button.disabled = false;
                                button.textContent = 'Ask';
                                document.getElementById('questionInput').value = ''; // Clear input
                                return;
                            }
                            const text = new TextDecoder().decode(value);
                            streamingText.textContent += text;
                            scrollToBottom();
                            return readStream();
                        });
                    }
                    return readStream();
                });
        }
    </script>
</body>
</html>
'''


# ============================================================================
# DATA MODELS & SESSION STATE
# ============================================================================

# Session storage (simple in-memory for single-user testing)
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
                import shutil
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
# FLASK ROUTES & REQUEST HANDLERS
# ============================================================================

@app.route('/')
def index():
    """
    Main page route - displays the document Q&A interface.
    
    Returns:
        str: Rendered HTML template with current documents, chat history, and status
    """
    try:
        session_data['documents'] = get_documents()
        return render_template_string(HTML_TEMPLATE, 
                                    documents=session_data['documents'],
                                    chat_history=session_data['chat_history'],
                                    status=session_data['status'])
    except Exception as e:
        logger.error(f"Error loading index page: {e}")
        session_data['status'] = {'type': 'error', 'message': f'Error: {str(e)}'}
        return render_template_string(HTML_TEMPLATE, 
                                    documents=[],
                                    chat_history=session_data['chat_history'],
                                    status=session_data['status'])


@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle file upload and automatic processing.
    
    Accepts file uploads, saves to documents directory, and triggers
    automatic processing of the new document into the vector index.
    
    Returns:
        redirect: Redirects to main page with status message
    """
    try:
        if 'file' not in request.files:
            session_data['status'] = {'type': 'error', 'message': 'No file selected'}
            return redirect(url_for('index'))
        
        file = request.files['file']
        if file.filename == '':
            session_data['status'] = {'type': 'error', 'message': 'No file selected'}
            return redirect(url_for('index'))
        
        if file:
            filename = secure_filename(file.filename)
            upload_path = Path(app.config['UPLOAD_FOLDER'])
            upload_path.mkdir(parents=True, exist_ok=True)
            
            file_path = upload_path / filename
            file.save(str(file_path))
            
            logger.info(f"File uploaded: {filename}")
            
            # Auto-process only the new document
            session_data['status'] = {'type': 'loading', 'message': f'🔄 Processing new file: {filename}...'}
            
            current_pipeline = get_pipeline()
            if current_pipeline:
                success = current_pipeline.process_single_document(file_path)
                if success:
                    session_data['status'] = {'type': 'success', 'message': f'✅ Uploaded and processed: {filename}'}
                    logger.info(f"New document processed: {filename}")
                else:
                    session_data['status'] = {'type': 'error', 'message': f'⚠️ Uploaded {filename} but processing failed'}
                    logger.error(f"Single document processing failed: {filename}")
            else:
                session_data['status'] = {'type': 'error', 'message': f'⚠️ Uploaded {filename} but pipeline unavailable'}
        
    except Exception as e:
        session_data['status'] = {'type': 'error', 'message': f'Upload failed: {str(e)}'}
        logger.error(f"Upload error: {e}")
    
    return redirect(url_for('index'))


@app.route('/remove_all', methods=['POST'])
def remove_all_documents():
    """
    Remove all documents and clear the vector index.
    
    Deletes all files from documents directory, removes index files,
    resets the pipeline, and clears session chat history.
    
    Returns:
        redirect: Redirects to main page with status message
    """
    try:
        session_data['status'] = {'type': 'loading', 'message': '�️ Removing all documents...'}
        
        # Clear documents directory
        docs_dir = paths.DOCS_DIR
        if docs_dir.exists():
            import shutil
            for file_path in docs_dir.iterdir():
                if file_path.is_file():
                    file_path.unlink()
                    logger.info(f"Deleted document: {file_path.name}")
        
        # Clear index directory
        index_dir = paths.INDEX_DIR
        if index_dir.exists():
            import shutil
            shutil.rmtree(index_dir, ignore_errors=True)
            logger.info("Deleted index directory")
        
        # Reset pipeline
        global pipeline
        pipeline = None
        
        # Clear session data
        session_data['chat_history'] = []
        
        session_data['status'] = {'type': 'success', 'message': '✅ All documents removed successfully!'}
        logger.info("All documents and index removed successfully")
            
    except Exception as e:
        session_data['status'] = {'type': 'error', 'message': f'Remove error: {str(e)}'}
        logger.error(f"Remove error: {e}")
    
    return redirect(url_for('index'))


@app.route('/stream')
def stream():
    """
    Stream AI responses token by token for real-time display.
    
    Accepts question via query parameter and streams response using
    the pipeline's streaming capability for improved UX.
    
    Returns:
        Response: Streaming response with text/plain mimetype
        tuple: Error message and status code if validation fails
    """
    question = request.args.get('question', '').strip()
    if not question:
        return "No question provided", 400
    
    current_pipeline = get_pipeline()
    if not current_pipeline or not current_pipeline.vectorstore:
        return "Please process documents first", 400
    
    def generate():
        """Generator function for streaming tokens."""
        for token in current_pipeline.ask_streaming(question):
            yield token
    
    return Response(generate(), mimetype='text/plain')


@app.route('/clear')
def clear_chat():
    """
    Clear chat history and reset session.
    
    Returns:
        redirect: Redirects to main page with success message
    """
    session_data['chat_history'] = []
    session_data['status'] = {'type': 'success', 'message': '🗑️ Chat cleared'}
    return redirect(url_for('index'))


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    """
    Main entry point - initializes system, launches monitoring GUI, and starts Flask server.
    """
    print("🚀 Document Q&A - Flask Web Interface")
    print("=" * 50)
    
    # Launch live monitoring GUI (no console window needed)
    try:
        print("📊 Starting live system monitor GUI...")
        # Get the directory where app.py is located
        script_dir = Path(__file__).parent.absolute()
        monitor_script = script_dir / "backend_live.py"
        
        # Launch GUI without console window (pythonw.exe for Windows)
        python_exe = sys.executable
        # Use pythonw.exe instead of python.exe to hide console
        if python_exe.endswith('python.exe'):
            pythonw_exe = python_exe.replace('python.exe', 'pythonw.exe')
        else:
            pythonw_exe = python_exe
        
        subprocess.Popen(
            [pythonw_exe, str(monitor_script)],
            cwd=str(script_dir),
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
    
    print("📍 Open: http://127.0.0.1:5000")
    print("🛑 Stop: Ctrl+C")
    print("✅ Running in stable mode (no debug, no restarts)")
    
    # Run without debug mode to prevent restart loops during document processing
    app.run(debug=False, host='127.0.0.1', port=5000)
