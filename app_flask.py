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
import sys
sys.path.append(str(Path(__file__).parent))

# Import DocumentPipeline directly from 5-chat.py
import importlib.util
spec = importlib.util.spec_from_file_location("chat_module", "5-chat.py")
chat_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chat_module)
DocumentPipeline = chat_module.DocumentPipeline

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'data/documents'

# Global pipeline instance
pipeline = None

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
        .message { margin: 10px 0; padding: 10px; border-radius: 5px; }
        .user-message { background: #e3f2fd; text-align: right; }
        .bot-message { background: #f1f8e9; }
        .sources { background: #fff3e0; padding: 10px; margin: 5px 0; border-radius: 3px; font-size: 0.9em; }
        input[type="text"] { width: 70%; padding: 10px; }
        button { padding: 10px 20px; background: #2196F3; color: white; border: none; border-radius: 3px; cursor: pointer; }
        button:hover { background: #1976D2; }
        .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
        .success { background: #d4edda; color: #155724; }
        .error { background: #f8d7da; color: #721c24; }
        .loading { background: #fff3cd; color: #856404; }
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
                    <strong>{{ 'You' if msg.type == 'user' else '🤖' }}:</strong> {{ msg.content }}
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
            
            <form action="/ask" method="post">
                <input type="text" name="question" placeholder="Ask a question about your documents..." required>
                <button type="submit">Ask</button>
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
    session_data['documents'] = get_documents()
    return render_template_string(HTML_TEMPLATE, 
                                documents=session_data['documents'],
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
    global pipeline
    try:
        session_data['status'] = {'type': 'loading', 'message': '🔄 Processing documents...'}
        
        # Initialize pipeline if needed
        if not pipeline:
            pipeline = DocumentPipeline()
        
        # Process documents
        success = pipeline.process_documents()
        
        if success:
            session_data['status'] = {'type': 'success', 'message': '✅ Documents processed successfully!'}
        else:
            session_data['status'] = {'type': 'error', 'message': '❌ Document processing failed'}
            
    except Exception as e:
        session_data['status'] = {'type': 'error', 'message': f'Processing error: {str(e)}'}
        logger.error(f"Processing error: {e}")
    
    return redirect(url_for('index'))

@app.route('/ask', methods=['POST'])
def ask_question():
    """Handle Q&A"""
    global pipeline
    try:
        question = request.form.get('question', '').strip()
        if not question:
            session_data['status'] = {'type': 'error', 'message': 'Please enter a question'}
            return redirect(url_for('index'))
        
        if not pipeline or not pipeline.qa_chain:
            session_data['status'] = {'type': 'error', 'message': 'Please process documents first'}
            return redirect(url_for('index'))
        
        # Add user message
        session_data['chat_history'].append({
            'type': 'user',
            'content': question,
            'sources': None
        })
        
        # Get answer
        session_data['status'] = {'type': 'loading', 'message': '🤖 Thinking...'}
        start_time = time.time()
        
        result = pipeline.ask(question)
        
        # Add bot response
        session_data['chat_history'].append({
            'type': 'bot',
            'content': result['answer'],
            'sources': result.get('sources', [])
        })
        
        elapsed = time.time() - start_time
        session_data['status'] = {'type': 'success', 'message': f'✅ Answered in {elapsed:.1f}s'}
        
    except Exception as e:
        session_data['status'] = {'type': 'error', 'message': f'Question failed: {str(e)}'}
        logger.error(f"Q&A error: {e}")
    
    return redirect(url_for('index'))

@app.route('/clear')
def clear_chat():
    """Clear chat history"""
    session_data['chat_history'] = []
    session_data['status'] = {'type': 'success', 'message': '🗑️ Chat cleared'}
    return redirect(url_for('index'))

if __name__ == '__main__':
    print("🚀 Starting Flask Document Q&A Server...")
    print("📍 Open: http://127.0.0.1:5000")
    print("🛑 Stop: Ctrl+C")
    
    app.run(debug=True, host='127.0.0.1', port=5000)
