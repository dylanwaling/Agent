# Available Commands Reference

Quick reference for all commands available in this Enhanced Document Q&A Pipeline.

## 🚀 Main Applications

### Start Web Interface
```bash
python app.py
```
- Launches Flask web interface on http://127.0.0.1:5000
- Upload documents, process them, and ask questions
- Enhanced search with perfect document targeting

### Comprehensive Testing & Debugging
```bash
python backend_debug.py
```
- Complete system health check and validation
- Tests all components: Docling, embeddings, Ollama, FAISS
- Performance benchmarking and accuracy validation
- Search functionality testing with real queries

## 🔧 Pipeline Operations

### Direct Pipeline Usage
```python
from backend_logic import DocumentPipeline

# Initialize pipeline
pipeline = DocumentPipeline()

# Load existing index or process documents
if not pipeline.load_index():
    pipeline.process_documents()

# Enhanced search with smart filtering
results = pipeline.search("invoice outline")

# Ask questions with enhanced retrieval
result = pipeline.ask("What is in the invoice document?")
print(result["answer"])
print("Sources:", [s["source"] for s in result["sources"]])
```

### Advanced Usage
```python
# Custom score threshold
results = pipeline.search("company handbook", score_threshold=1.5)

# Debug search to see ranking
debug_info = pipeline.debug_search("product manual")
for r in debug_info["results"]:
    print(f"{r['rank']}. {r['source']} (score: {r['score']})")
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
# Pull Llama3 (recommended for quality)
ollama pull llama3:latest

# Pull TinyLlama (faster, lower quality)
ollama pull tinyllama:latest

# Check running models
ollama ps

# Remove a model
ollama rm model_name
```

### Test Ollama
```bash
ollama run llama3:latest "Hello, respond with just 'Hi there!'"
```

## � Testing & Validation

### Full System Test
```bash
python backend_debug.py
```
- Tests all system components
- Validates document processing
- Benchmarks search performance
- Checks Q&A accuracy

### Quick Performance Check
```python
from backend_logic import DocumentPipeline
import time

pipeline = DocumentPipeline()
pipeline.load_index()

start = time.time()
results = pipeline.search("company handbook")
print(f"Search took: {time.time() - start:.3f}s")
print(f"Found: {len(results)} results")
```

## 🗂️ File Management

### Document Locations
```bash
# Add documents here
data/documents/

# Index storage
data/index/faiss_index.faiss
data/index/faiss_index.pkl
```

### Clear Index (Force Rebuild)
```bash
# Delete index files
rm data/index/faiss_index.faiss
rm data/index/faiss_index.pkl

# Or in Python
import shutil
shutil.rmtree("data/index", ignore_errors=True)
```

### Current File Structure
```
backend_logic.py       # Main processing pipeline
app.py                 # Flask web interface  
backend_debug.py       # Debug and verification
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
from backend_logic import DocumentPipeline

pipeline = DocumentPipeline()

# Check if index exists and load it
if pipeline.load_index():
    print("Index loaded successfully")
else:
    print("No index found, processing documents...")
    pipeline.process_documents()

# Search for documents
results = pipeline.search("your search query")
for result in results:
    print(f"Found: {result['source']}")

# Rebuild index (reprocess all documents)
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
python app.py

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
python backend_debug.py

# Manually check documents folder
dir "data\documents"
```

### If web app won't start:
```bash
# Check for port conflicts
netstat -an | findstr :5000

# Try different port
python app.py --port 5001
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
python app.py
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
- **Check backend_debug.py first** if anything isn't working
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
