#!/usr/bin/env python3
"""
Lightweight Flask Web Interface for Document Q&A
Simple, fast alternative to Streamlit
"""

import os
import logging
from pathlib import Path
from flask import Flask, request, render_template_string, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import time

# Import our pipeline
from backend_logic import DocumentPipeline

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'data/documents'

# Initialize pipeline on startup (with lazy loading and cleanup)
pipeline = None

def clean_startup():
    """Clean startup - remove any partial index files"""
    try:
        index_dir = Path("data/index")
        if index_dir.exists():
            # Check if index is complete
            faiss_file = index_dir / "faiss_index" / "index.faiss"
            pkl_file = index_dir / "faiss_index" / "index.pkl"
            
            # If either file is missing, clean the directory
            if not (faiss_file.exists() and pkl_file.exists()):
                logger.info("🧹 Cleaning incomplete index files...")
                import shutil
                shutil.rmtree(index_dir, ignore_errors=True)
                logger.info("✅ Cleaned up incomplete index")
        
        return True
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        return False

def get_pipeline():
    """Get or initialize pipeline (lazy loading)"""
    global pipeline
    if pipeline is None:
        try:
            # Clean startup first
            clean_startup()
            
            pipeline = DocumentPipeline()
            if pipeline.load_index():
                logger.info("✅ Loaded existing index")
            else:
                logger.info("⚠️ No existing index found - documents need processing")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            pipeline = None
    return pipeline

# HTML Template (embedded for simplicity)
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
            
            <form action="/process" method="post" style="margin-top: 20px;">
                <button type="submit">🔄 Process All Documents</button>
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

# Session storage (simple in-memory for testing)
session_data = {
    'chat_history': [],
    'status': None,
    'documents': []
}

def get_documents():
    """Get list of uploaded documents"""
    docs_dir = Path('data/documents')
    if docs_dir.exists():
        return [f.name for f in docs_dir.iterdir() if f.is_file()]
    return []

@app.route('/')
def index():
    """Main page"""
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
    """Handle file upload"""
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
            
            session_data['status'] = {'type': 'success', 'message': f'✅ Uploaded: {filename}'}
            logger.info(f"File uploaded: {filename}")
        
    except Exception as e:
        session_data['status'] = {'type': 'error', 'message': f'Upload failed: {str(e)}'}
        logger.error(f"Upload error: {e}")
    
    return redirect(url_for('index'))

@app.route('/process', methods=['POST'])
def process_documents():
    """Process all documents"""
    try:
        session_data['status'] = {'type': 'loading', 'message': '🔄 Processing documents...'}
        
        # Get pipeline
        current_pipeline = get_pipeline()
        if not current_pipeline:
            session_data['status'] = {'type': 'error', 'message': '❌ Failed to initialize pipeline'}
            return redirect(url_for('index'))
        
        # Process documents with timeout protection
        logger.info("Starting document processing...")
        success = current_pipeline.process_documents()
        
        if success:
            session_data['status'] = {'type': 'success', 'message': '✅ Documents processed successfully!'}
            logger.info("Document processing completed successfully")
        else:
            session_data['status'] = {'type': 'error', 'message': '❌ Document processing failed'}
            logger.error("Document processing failed")
            
    except Exception as e:
        session_data['status'] = {'type': 'error', 'message': f'Processing error: {str(e)}'}
        logger.error(f"Processing error: {e}")
    
    return redirect(url_for('index'))

@app.route('/stream')
def stream():
    """Stream responses word by word"""
    from flask import Response
    
    question = request.args.get('question', '').strip()
    if not question:
        return "No question provided", 400
    
    current_pipeline = get_pipeline()
    if not current_pipeline or not current_pipeline.vectorstore:
        return "Please process documents first", 400
    
    def generate():
        for token in current_pipeline.ask_streaming(question):
            yield token
    
    return Response(generate(), mimetype='text/plain')

@app.route('/clear')
def clear_chat():
    """Clear chat history"""
    session_data['chat_history'] = []
    session_data['status'] = {'type': 'success', 'message': '🗑️ Chat cleared'}
    return redirect(url_for('index'))
def clear_chat():
    """Clear chat history"""
    session_data['chat_history'] = []
    session_data['status'] = {'type': 'success', 'message': '🗑️ Chat cleared'}
    return redirect(url_for('index'))

if __name__ == '__main__':
    print("🚀 Document Q&A - Flask Web Interface")
    print("=" * 50)
    
    # Check documents
    docs_dir = Path('data/documents')
    if docs_dir.exists():
        doc_count = len([f for f in docs_dir.iterdir() if f.is_file()])
        print(f"� Found {doc_count} documents ready for processing")
    else:
        print("📂 Documents directory will be created on first upload")
    
    print("📍 Open: http://127.0.0.1:5000")
    print("🛑 Stop: Ctrl+C")
    print("✅ Running in stable mode (no debug, no restarts)")
    
    # Run without debug mode to prevent restart loops during document processing
    app.run(debug=False, host='127.0.0.1', port=5000)
