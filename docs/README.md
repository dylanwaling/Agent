# Document Q&A Pipeline

A robust document processing and question-answering system using Docling, LangChain, and Streamlit.

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r docs/requirements.txt
   ```

2. **Start Ollama (if not running):**
   ```bash
   ollama serve
   ollama pull llama3:latest
   ```

3. **Run the pipeline:**
   ```bash
   python 5-chat.py
   ```

## 📁 Project Structure

### Core Pipeline Files
- **`1-extraction.py`** - Document extraction using Docling
- **`2-chunking.py`** - Text chunking and preprocessing  
- **`3-embedding.py`** - Vector embeddings generation
- **`4-search.py`** - Document search and retrieval
- **`5-chat.py`** - Complete Q&A pipeline with Streamlit UI

### Debug & Utilities
- **`debug-1-extraction.py`** - Debug extraction issues
- **`debug-2-chunking.py`** - Debug chunking problems
- **`debug-3-embedding.py`** - Debug embedding generation
- **`debug-4-search.py`** - Debug search functionality
- **`debug-5-chat.py`** - Debug chat/QA issues
- **`debug_database_content.py`** - Database content inspection
- **`debug_utils.py`** - General debugging utilities
- **`check_docling_status.py`** - Verify Docling installation

## 🛠️ System Architecture

### Pipeline Flow
```
Documents → Docling → Chunking → Embeddings → FAISS Index → LangChain QA → Streamlit UI
```

### Components

#### 1. Document Processing (Docling)
- Supports PDF, DOCX, TXT, MD, images
- Handles encrypted/protected PDFs with OCR fallback
- Extracts text, tables, and metadata
- Converts to clean markdown format

#### 2. Text Processing (LangChain)
- **Chunking**: 1000 chars with 200 char overlap
- **Embeddings**: HuggingFace all-MiniLM-L6-v2
- **Vector Store**: FAISS for fast similarity search
- **Quality Filtering**: Removes corrupted/repetitive text

#### 3. Question Answering
- **LLM**: Ollama (llama3:latest)
- **Retrieval**: Top-3 relevant chunks
- **Context**: Source attribution included
- **Chain Type**: "stuff" (concatenate contexts)

#### 4. User Interface
- **Streamlit**: Clean, responsive web interface
- **Features**: Document upload, chat history, source display
- **Management**: Add/remove documents, rebuild index

## 📋 Supported File Formats

| Format | Method | Notes |
|--------|--------|-------|
| PDF | Docling | Handles text, images, tables |
| DOCX | Docling | Full document structure |
| TXT | Direct read | Plain text files |
| MD | Direct read | Markdown files |
| Images | Docling + OCR | PNG, JPG, JPEG |

## 🔧 Configuration

### Default Settings
- **Chunk size**: 1000 characters
- **Chunk overlap**: 200 characters
- **Retrieval**: Top 3 documents
- **Model**: llama3:latest
- **Embeddings**: all-MiniLM-L6-v2

### Environment Variables
```bash
# Optional: Set custom model
OLLAMA_MODEL=llama3:latest

# Optional: Set data directories
DOCS_DIR=data/documents
INDEX_DIR=data/index
```

## 🧪 Testing & Debugging

### Run Tests
```bash
# Test individual components
python debug-1-extraction.py
python debug-2-chunking.py
python debug-3-embedding.py
python debug-4-search.py
python debug-5-chat.py

# Check system status
python check_docling_status.py

# Inspect database
python debug_database_content.py
```

### Common Issues

#### 1. Model Not Found
```bash
# Pull the required model
ollama pull llama3:latest

# Check available models
ollama list
```

#### 2. Docling Format Errors
- Check file permissions
- Verify file isn't corrupted
- Try OCR fallback for images/scanned PDFs

#### 3. Empty Search Results
- Rebuild index: Use Streamlit "Rebuild Index" button
- Check document processing logs
- Verify embeddings are generated

#### 4. Slow Performance
- Use smaller model: `tinyllama:latest`
- Reduce chunk size in configuration
- Limit retrieval documents (k parameter)

## 📊 Performance Guidelines

### Document Limits
- **Small files** (< 1MB): Process immediately
- **Medium files** (1-10MB): May take 30-60 seconds
- **Large files** (> 10MB): Consider splitting first

### Memory Usage
- **Embeddings**: ~100MB for model loading
- **FAISS Index**: ~1KB per document chunk
- **LLM**: Depends on Ollama model size

## 🔄 Version History

- **v0.0.13**: Complete pipeline rewrite with quality filtering
- **v0.0.08**: Base model working with debug tools
- **v0.0.07**: Major refactor of all components
- **v0.0.02**: Initial extraction and search functionality

## 📚 Dependencies

See `docs/requirements.txt` for complete list. Key dependencies:
- `docling` - Document processing
- `langchain` - QA framework  
- `streamlit` - Web interface
- `faiss-cpu` - Vector search
- `sentence-transformers` - Embeddings
- `ollama` - Local LLM

## 🤝 Usage Tips

1. **Start small**: Test with a few simple documents first
2. **Monitor logs**: Check console output for processing status
3. **Use debug tools**: Isolate issues with specific debug scripts
4. **Quality matters**: Clean, well-formatted documents work best
5. **Be patient**: Large documents take time to process

## 🆘 Troubleshooting

If you encounter issues:

1. Check the debug scripts for the specific component
2. Verify all dependencies are installed
3. Make sure Ollama is running with the correct model
4. Check file permissions and formats
5. Look at the console logs for detailed error messages

For persistent issues, check the individual debug files in the project root.  
- **Flexible Output**: Export to HTML, Markdown, JSON, or plain text
- **High Performance**: Efficient processing on local hardware
- **Local AI**: Uses Ollama with Llama 3 for chat responses
- **Local Embeddings**: Uses sentence-transformers for embeddings (BAAI/bge-small-en-v1.5)
- **Pydantic v1**: Compatible with Pydantic v1.x for broader compatibility

## Things They're Working on

- Metadata extraction, including title, authors, references & language
- Inclusion of Visual Language Models (SmolDocling)
- Chart understanding (Barchart, Piechart, LinePlot, etc)
- Complex chemistry understanding (Molecular structures)

## Getting Started with the Example

### Prerequisites

1. **Install Ollama** from [https://ollama.ai/](https://ollama.ai/)

2. **Start the Ollama service**:
   ```bash
   ollama serve
   ```

3. **Pull the Llama 3 model**:
   ```bash
   ollama pull llama3
   ```

4. **Install Python dependencies**:
   ```bash
   pip install -r docs/requirements.txt
   ```

### Configuration

1. Copy the environment template:
   ```bash
   cp .env.example .env
   ```

2. (Optional) Modify `.env` if you need custom Ollama settings

### Running the Example

Execute the files in order to build and query the document database:

1. Extract document content: `python 1-extraction.py`
2. Create document chunks: `python 2-chunking.py`
3. Create embeddings and store in LanceDB: `python 3-embedding.py`
4. Test basic search functionality: `python 4-search.py`
5. Launch the Streamlit chat interface: `streamlit run 5-chat.py`

Then open your browser and navigate to `http://localhost:8501` to interact with the document Q&A interface.

## Document Processing

### Supported Input Formats

| Format | Description |
|--------|-------------|
| PDF | Native PDF documents with layout preservation |
| DOCX, XLSX, PPTX | Microsoft Office formats (2007+) |
| Markdown | Plain text with markup |
| HTML/XHTML | Web documents |
| Images | PNG, JPEG, TIFF, BMP |
| USPTO XML | Patent documents |
| PMC XML | PubMed Central articles |

Check out this [page](https://ds4sd.github.io/docling/supported_formats/) for an up to date list.

### Processing Pipeline

The standard pipeline includes:

1. Document parsing with format-specific backend
2. Layout analysis using AI models
3. Table structure recognition
4. Metadata extraction
5. Content organization and structuring
6. Export formatting

## Models

Docling leverages two primary specialized AI models for document understanding. At its core, the layout analysis model is built on the `RT-DETR (Real-Time Detection Transformer)` architecture, which excels at detecting and classifying page elements. This model processes pages at 72 dpi resolution and can analyze a single page in under a second on a standard CPU, having been trained on the comprehensive `DocLayNet` dataset.

The second key model is `TableFormer`, a table structure recognition system that can handle complex table layouts including partial borders, empty cells, spanning cells, and hierarchical headers. TableFormer typically processes tables in 2-6 seconds on CPU, making it efficient for practical use. 

For documents requiring text extraction from images, Docling integrates `EasyOCR` as an optional component, which operates at 216 dpi for optimal quality but requires about 30 seconds per page. Both the layout analysis and TableFormer models were developed by IBM Research and are publicly available as pre-trained weights on Hugging Face under "ds4sd/docling-models".

For more detailed information about these models and their implementation, you can refer to the [technical documentation](https://arxiv.org/pdf/2408.09869).

## Chunking

When you're building a RAG (Retrieval Augmented Generation) application, you need to break down documents into smaller, meaningful pieces that can be easily searched and retrieved. But this isn't as simple as just splitting text every X words or characters.

What makes [Docling's chunking](https://ds4sd.github.io/docling/concepts/chunking/) unique is that it understands the actual structure of your document. It has two main approaches:

1. The [Hierarchical Chunker](https://ds4sd.github.io/docling/concepts/chunking/#hierarchical-chunker) is like a smart document analyzer - it knows where the natural "joints" of your document are. Instead of blindly cutting text into fixed-size pieces, it recognizes and preserves important elements like sections, paragraphs, tables, and lists. It maintains the relationship between headers and their content, and keeps related items together (like items in a list).

2. The [Hybrid Chunker](https://ds4sd.github.io/docling/concepts/chunking/#hybrid-chunker) takes this a step further. It starts with the hierarchical chunks but then:
   - It can split chunks that are too large for your embedding model
   - It can stitch together chunks that are too small
   - It works with your specific tokenizer, so the chunks will fit perfectly with your chosen language model

### Why is this great for RAG applications?

Imagine you're building a system to answer questions about technical documents. With basic chunking (like splitting every 500 words), you might cut right through the middle of a table, or separate a header from its content. But Docling's smart chunking:

- Keeps related information together
- Preserves document structure
- Maintains context (like headers and captions)
- Creates chunks that are optimized for your specific embedding model
- Ensures each chunk is meaningful and self-contained

This means when your RAG system retrieves chunks, they'll have the proper context and structure, leading to more accurate and coherent responses from your language model.

## Documentation

For full documentation, visit [documentation site](https://ds4sd.github.io/docling/).

For example notebooks and more detailed guides, check out [GitHub repository](https://github.com/DS4SD/docling).