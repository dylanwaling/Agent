# Setup Guide for Docling + Ollama + Llama 3 Pipeline

This guide will help you set up and run the knowledge extraction pipeline using Docling, Ollama, and Llama 3.

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
ollama pull llama3
```

This will download the Llama 3 model (about 4.7GB).

### 4. Verify Installation

Check that everything is working:
```bash
ollama list
```

You should see `llama3` in the list of models.

## Python Environment Setup

### 1. Install Dependencies

The Python packages have been installed, but if you need to reinstall:

```bash
pip install -r requirements.txt
```

### 2. Environment Configuration

Copy the example environment file:
```bash
copy .env.example .env
```

The default configuration should work with a standard Ollama installation.

## Running the Pipeline

Execute the scripts in the following order:

### 1. Document Extraction
```bash
python 1-extraction.py
```
This script extracts content from documents (PDFs, HTML pages) using Docling.

**⏰ First Run Timing**: This can take 5-15 minutes on first run as it downloads AI models:
- Layout analysis models (~500MB)
- Table recognition models
- Document structure analysis

**What's happening**:
- Downloading RT-DETR layout analysis model
- Downloading TableFormer table recognition model
- Processing ArXiv PDF with AI analysis
- Extracting content from multiple HTML pages

### 2. Document Chunking
```bash
python 2-chunking.py
```
This script creates intelligent chunks from the extracted documents.

### 3. Create Embeddings Database
```bash
python 3-embedding.py
```
This script creates embeddings using sentence-transformers and stores them in LanceDB.

### 4. Test Search Functionality
```bash
python 4-search.py
```
This script tests the search functionality to ensure everything is working.

### 5. Launch Chat Interface
```bash
streamlit run 5-chat.py
```
This launches the interactive chat interface where you can ask questions about the documents.

## Using the Chat Interface

1. After running `streamlit run 5-chat.py`, open your browser to `http://localhost:8501`
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
- Make sure all requirements are installed: `pip install -r requirements.txt`
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
