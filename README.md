# Do## 🚀 Quick Start

```bash
# Install dependencies
pip install -r docs/requirements.txt

# Run the Streamlit app
streamlit run app.py

# Or test the pipeline directly
python quick_test.py
```A Pipeline

A clean, robust document processing and question-answering system using **Docling → LangChain → Streamlit**.

## � Quick Start

```bash
# Install dependencies
pip install -r docs/requirements.txt

# Run the complete pipeline
python 5-chat.py
```

## 📁 File Structure

- **`1-extraction.py`** - Document extraction
- **`2-chunking.py`** - Text chunking  
- **`3-embedding.py`** - Vector embeddings
- **`4-search.py`** - Document search
- **`5-chat.py`** - Complete Q&A pipeline ⭐

### Debug Tools
- **`debug-*.py`** - Component-specific debugging
- **`debug_utils.py`** - General utilities
- **`check_docling_status.py`** - System verification

## 📖 Documentation

**➡️ See `docs/README.md` for complete documentation, architecture, and troubleshooting.**

## 🎯 Current Version: v0.0.13

Complete pipeline rewrite with:
- Quality filtering for corrupted text
- Streamlit UI with document management
- FAISS vector search
- Ollama LLM integration
- Robust error handling

---

**📁 For complete information, please visit the [`docs/`](docs/) folder**