# Setup Guide for Enhanced Document Q&A Pipeline

This guide will help you set up and run the enhanced document processing pipeline using Docling, FAISS, HuggingFace embeddings, and Ollama.

## Prerequisites

### 1. Install Ollama

Download and install Ollama from [https://ollama.ai/](https://ollama.ai/)

### 2. Start Ollama Service

After installation, start the Ollama service:

**Windows:**
```cmd
ollama serve
```

**macOS/Linux:**
```bash
ollama serve
```

### 3. Install Llama 3 Model

In a new terminal/command prompt:
```bash
ollama pull llama3:latest
```

This will download the Llama 3 model (about 4.7GB).

### 4. Verify Installation

Check that everything is working:
```bash
ollama list
```

You should see `llama3:latest` in the list of models.

## Python Environment Setup

### 1. Install Dependencies

```bash
pip install -r docs/requirements.txt
```

The requirements include:
- **docling** - Document processing and extraction
- **langchain** - Text processing and chunking  
- **langchain-huggingface** - HuggingFace embeddings integration
- **langchain-ollama** - Ollama LLM integration
- **faiss-cpu** - Fast similarity search and clustering
- **flask** - Web application framework
```

## Running the Application

### Option 1: Web Application (Recommended)
```bash
python app.py
```

**What this does:**
- Starts Flask web server on http://127.0.0.1:5000
- Automatically loads existing index or processes documents
- Provides clean web interface for document Q&A
- Uses enhanced search for perfect document targeting

**⏰ First Run:** May take a few minutes to download models:
- HuggingFace sentence-transformers model (~90MB)
- Docling AI models for document processing (~500MB)

### Option 2: Comprehensive Testing
```bash
python backend_debug.py
```

**What this does:**
- Tests all system components (imports, Docling, Ollama, embeddings)
- Validates document processing pipeline
- Benchmarks search performance (should show ~6ms average)
- Tests Q&A functionality with real queries
- Validates enhanced search accuracy

## First Time Setup

### 1. Add Your Documents
Place documents in the `data/documents/` folder:
```bash
# Supported formats
data/documents/
├── your_document.pdf
├── notes.md
├── manual.docx
└── text_file.txt
```

### 2. Process Documents
Either use the web interface or run:
```python
from backend_logic import DocumentPipeline

pipeline = DocumentPipeline()
pipeline.process_documents()  # Creates FAISS index
```

### 3. Start Asking Questions
```python
result = pipeline.ask("What is in the invoice document?")
print(result["answer"])
print("Sources:", [s["source"] for s in result["sources"]])
```
python app.py
```
This launches the interactive web interface where you can ask questions about the documents.

## Using the Web Interface

1. After running `python app.py`, open your browser to `http://localhost:5000`
2. Type your questions about the document content
3. The system will:
   - Search for relevant document chunks
   - Show you the sources it found
   - Generate an answer using Llama 3

## Troubleshooting

### Common Issues

#### "Error connecting to Ollama"
- Make sure Ollama is running: `ollama serve`
- Check that the service is accessible: `ollama list`
- Verify the model is installed: `ollama pull llama3`

#### "Import errors" in Python
- Make sure all requirements are installed: `pip install -r docs/requirements.txt`
- Check your Python environment is activated

#### "No module named 'lancedb'"
- Install LanceDB: `pip install lancedb`
- Some systems may need: `pip install --upgrade lancedb`

#### Slow performance
- The first run will be slower as it downloads embedding models
- Subsequent runs should be much faster
- Consider using a smaller Llama model if needed: `ollama pull llama3:8b`

### Performance Optimization

1. **For faster embeddings**: The sentence-transformers model will be downloaded on first use
2. **For lower memory usage**: Use `llama3:8b` instead of the full `llama3` model
3. **For better responses**: You can try other Ollama models like `llama3:70b` if your hardware supports it

## Architecture Overview

```
Documents → Docling → Chunks → Embeddings → LanceDB
                                              ↓
User Query → Semantic Search → Context → Llama 3 → Response
```

1. **Docling** extracts and structures content from various document formats
2. **Hybrid Chunker** creates intelligent chunks preserving document structure
3. **Sentence Transformers** creates embeddings for semantic search
4. **LanceDB** stores embeddings for fast vector search
5. **Ollama + Llama 3** generates contextual responses

## Customization

### Using Different Models

To use a different Ollama model, edit `5-chat.py`:
```python
response_stream = ollama.chat(
    model="your-model-name",  # Change this line
    messages=formatted_messages,
    stream=True,
)
```

### Changing Embedding Models

To use a different embedding model, edit `3-embedding.py`:
```python
func = get_registry().get("sentence-transformers").create(name="your-model-name")
```

### Processing Different Documents

To process your own documents, modify the URL in `1-extraction.py`:
```python
result = converter.convert("path/to/your/document.pdf")
```

## Next Steps

- Experiment with different document types
- Try different Ollama models
- Customize the chat interface
- Add more sophisticated retrieval strategies
- Implement document upload functionality

For more examples and documentation, visit:
- [Docling Documentation](https://github.com/DS4SD/docling)
- [Ollama Documentation](https://ollama.ai/)
- [LanceDB Documentation](https://lancedb.github.io/lancedb/)
