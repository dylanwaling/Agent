# Document Q&A Agent - Professional Edition

## 🚀 Quick Start

### Run the Application

```powershell
# Option 1: Use the interactive launcher (Recommended)
python launcher.py

# Option 2: Run specific components directly
python -m run             # Desktop GUI application with monitoring
python -m test            # System tests
```

## 📁 Professional Project Structure

```
/Agent/
│
├── 🚀 launcher.py                    # Interactive launcher (main entry point)
│
├── 📦 core/                          # Core pipeline logic
│   ├── __init__.py
│   ├── analytics.py                  # Logging & monitoring system
│   ├── components.py                 # Component initialization (GPU, LLM)
│   ├── document_processor.py         # Document processing & FAISS indexing
│   ├── pipeline.py                   # Main DocumentPipeline coordinator
│   └── search_engine.py              # Semantic search & Q&A engine
│
├── 📦 config/                        # Configuration management
│   ├── __init__.py
│   └── settings.py                   # All configuration classes & settings
│
├── 📦 utils/                         # Utility functions
│   ├── __init__.py
│   └── helpers.py                    # Helper functions (I/O, formatting, GPU)
│
├── 📦 run/                           # Main application
│   ├── __init__.py
│   ├── __main__.py                   # Package entry point
│   ├── application.py                # Desktop GUI (Tkinter)
│   └── dashboard.py                  # Live monitoring dashboard
│
├── 📦 test/                          # System tests
│   ├── __init__.py
│   ├── __main__.py                   # Package entry point
│   └── system_test.py                # Comprehensive system tests
│
├── 📂 data/                          # Runtime data (auto-generated)
│   ├── documents/                    # User uploaded documents
│   ├── index/                        # FAISS vector index
│   │   └── faiss_index/
│   ├── operation_history.jsonl       # Operation logs
│   └── pipeline_status.json          # Real-time status
│
└── 📂 docs/                          # Documentation
    ├── ARCHITECTURE.md
    ├── PROJECT_STRUCTURE.md
    ├── REFACTORING_COMPLETION.md
    └── ... (other documentation)
```

## 🎯 Package Overview

### Core (`core/`)
Main document processing pipeline components:
- **analytics.py** - Comprehensive logging and monitoring
- **components.py** - GPU/CPU detection, embeddings, LLM initialization
- **document_processor.py** - Document ingestion, chunking, FAISS indexing
- **search_engine.py** - Semantic search and Q&A generation
- **pipeline.py** - Main coordinator (DocumentPipeline class)

### Config (`config/`)
Centralized configuration management:
- **settings.py** - All paths, model configs, search params, performance settings

### Utils (`utils/`)
Reusable utility functions:
- **helpers.py** - File I/O, formatting, GPU utilities, validation

### UI (`ui/`)
User interface components:
- **application.py** - Desktop GUI with document upload and Q&A

### Monitoring (`monitoring/`)
Live system monitoring:
- **dashboard.py** - Real-time operation tracking and metrics

### Tests (`tests/`)
System validation and debugging:
- **system_test.py** - Comprehensive test suite and benchmarks

## 🔧 Running Components

### Desktop Application
```powershell
python -m ui
```
- Upload and process documents (PDF, DOCX, TXT, MD)
- Ask questions with AI-powered answers
- Real-time streaming responses

### Monitoring Dashboard
```powershell
python -m monitoring
```
- Real-time operation tracking
- Performance metrics
- GPU/CPU monitoring
- Search and embedding analytics

### System Tests
```powershell
python -m tests
```
- Validate all components
- Performance benchmarks
- Debugging utilities

## 💡 Key Features

✅ **Fully Modular** - Clean separation of concerns  
✅ **Professional Structure** - Industry-standard organization  
✅ **Backward Compatible** - Old imports still work  
✅ **Well Documented** - Comprehensive docs in `/docs`  
✅ **GPU Optimized** - Automatic GPU detection and optimization  
✅ **Type Safety** - Full type hints throughout  
✅ **Production Ready** - Error handling and logging  

## 📚 Documentation

- [Architecture Overview](docs/ARCHITECTURE.md) - System design and data flows
- [Project Structure](docs/PROJECT_STRUCTURE.md) - Detailed module descriptions
- [Refactoring Summary](docs/REFACTORING_COMPLETION.md) - Migration guide

## 🔄 Backward Compatibility

Old code continues to work without changes:
```python
# Legacy imports (still work)
from backend_logic import DocumentPipeline
from config import paths, model_config
from utils import read_jsonl
```

New code should use explicit package imports:
```python
# Recommended for new code
from core import DocumentPipeline
from config.settings import paths, model_config
from utils.helpers import read_jsonl
from run import DocumentQAApp, LiveMonitorGUI
```

## 🛠️ Technology Stack

- **Python 3.x** - Core language
- **Docling** - Document conversion (PDF, DOCX → text)
- **LangChain** - Text processing and LLM integration
- **FAISS** - Vector similarity search
- **Ollama** - Local LLM inference (qwen2.5:1.5b)
- **HuggingFace** - Embeddings (all-MiniLM-L6-v2)
- **Tkinter** - Desktop GUI framework

## 📊 Performance

- **GPU Acceleration** - CUDA support for FAISS and embeddings
- **Optimized Chunking** - Dynamic chunk sizes based on GPU memory
- **Fast Search** - Sub-second semantic search
- **Streaming Responses** - Real-time token generation

## 🎉 Summary

This is a **professional, production-ready** document Q&A system with:
- ✨ Clean, organized code structure
- 📦 Proper package hierarchy
- 🚀 Easy to run and extend
- 📖 Comprehensive documentation
- 🔧 Full backward compatibility

Everything is properly categorized and organized for easy maintenance and development!
