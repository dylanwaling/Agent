# Available Commands Reference

Quick reference for all commands available in this Document Q&A Pipeline.

## 🚀 Main Applications

### Start Web Interface
```bash
python web_app.py
```
- Launches Flask web interface on http://127.0.0.1:5000
- Upload documents, process them, and ask questions
- Lightweight alternative to Streamlit

### System Check & Debugging
```bash
python system_check.py
```
- Comprehensive system health check
- Tests Docling, embeddings, Ollama, and database
- Troubleshoots common issues

## 🔧 Pipeline Operations

### Direct Pipeline Usage
```python
from document_pipeline import DocumentPipeline

# Initialize pipeline
pipeline = DocumentPipeline()

# Add documents
pipeline.add_document("path/to/document.pdf")

# Process all documents
pipeline.process_documents()

# Ask questions
result = pipeline.ask("What is this document about?")
print(result["answer"])
```

## 🤖 Ollama Commands

### Check Ollama Status
```bash
ollama list
```
- Lists all installed models

```bash
ollama serve
```
- Start Ollama service (if not running)

### Model Management
```bash
# Pull TinyLlama (recommended - fast & lightweight)
ollama pull tinyllama

# Pull Llama3 (better quality, slower)
ollama pull llama3

# Remove a model
ollama rm model_name
```

### Test Ollama
```bash
ollama run tinyllama "Hello, respond with just 'Hi there!'"
```

## 📦 Package Management

### Install Dependencies
```bash
pip install -r docs/requirements.txt
```

### Install Individual Packages
```bash
pip install flask docling langchain-ollama faiss-cpu
```

## 🗂️ File Management

### Current File Structure
```
document_pipeline.py    # Main processing pipeline
web_app.py             # Flask web interface  
system_check.py        # Debug and verification
data/documents/        # Upload documents here
data/index/           # Vector search index
docs/                 # Documentation
utils/                # Utility modules
```

### Manual Document Upload
```bash
# Copy documents to the data folder
copy "your_document.pdf" "data\documents\"
```

## 🔍 Debugging Commands

### Check System Components
```python
# In Python console
from system_check import *

# Test specific components
test_docling()
test_embeddings() 
test_ollama()
test_database()
```

### Database Operations
```python
from document_pipeline import DocumentPipeline

pipeline = DocumentPipeline()

# List current documents
docs = pipeline.list_documents()
print(docs)

# Remove a document
pipeline.remove_document("filename.pdf")

# Rebuild index
pipeline.process_documents()
```

## 🌐 Web Interface Usage

### Access Points
- **Main Interface**: http://127.0.0.1:5000
- **Upload**: Use the web form or copy to `data/documents/`
- **Process**: Click "Process All Documents" button
- **Chat**: Type questions in the chat input

### Supported File Types
- **PDF**: .pdf files
- **Text**: .txt, .md files  
- **Documents**: .docx files
- **Spreadsheets**: .xlsx files
- **Images**: .png, .jpg, .jpeg (OCR)

## ⚡ Quick Start Workflow

```bash
# 1. Start the system
python web_app.py

# 2. Open browser to http://127.0.0.1:5000

# 3. Upload documents via web interface

# 4. Click "Process All Documents"

# 5. Start asking questions!
```

## 🛠️ Troubleshooting Commands

### If Ollama won't start:
```bash
# Windows
ollama serve

# Check if service is running
curl http://localhost:11434/api/tags
```

### If documents won't process:
```bash
# Check system status
python system_check.py

# Manually check documents folder
dir "data\documents"
```

### If web app won't start:
```bash
# Check for port conflicts
netstat -an | findstr :5000

# Try different port
python web_app.py --port 5001
```

### Clear cache/index:
```bash
# Remove index to force rebuild
rmdir /s "data\index"

# Remove documents to start fresh  
rmdir /s "data\documents"
```

## 📊 Performance Commands

### Model Comparison
```bash
# Time TinyLlama response
ollama run tinyllama "What is 2+2?" --timing

# Time Llama3 response  
ollama run llama3 "What is 2+2?" --timing
```

### Memory Usage
```bash
# Check Python memory usage
python -c "import psutil; print(f'Memory: {psutil.virtual_memory().percent}%')"
```

## 🔄 Reset Commands

### Complete Reset
```bash
# Stop web app (Ctrl+C)
# Remove all data
rmdir /s "data"

# Restart and reprocess
python web_app.py
```

### Soft Reset (Keep Documents)
```bash
# Remove only index
rmdir /s "data\index"

# Reprocess via web interface
```

---

## 💡 Tips

- **Use TinyLlama** for faster testing and development
- **Use Llama3** for better answer quality in production
- **Check system_check.py first** if anything isn't working
- **Web interface is more reliable** than Streamlit for this use case
- **Smart retrieval** returns only relevant documents (1 to 500+, based on relevance)
- **Documents are processed once** - index is reused until rebuilt
- **Relevance threshold** filters out unrelated content automatically

## 🧠 Smart Retrieval Features

- **No arbitrary limits** - returns 1 document or 500+ based on relevance
- **Score-based filtering** - only shows documents above relevance threshold
- **Automatic quality control** - irrelevant documents are filtered out
- **Scalable** - works with small and large document collections

---

*Last updated: August 2025*
