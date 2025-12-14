# Project Structure - Document Q&A Agent

## Overview
This document describes the refactored modular structure of the Document Q&A Agent system.

## Directory Structure

```
/Agent/
├── core/                          # Main pipeline logic
│   ├── __init__.py               # Package exports
│   ├── analytics.py              # Analytics & logging system
│   ├── components.py             # Component initialization (GPU, LLM, embeddings)
│   ├── document_processor.py    # Document processing & indexing
│   ├── pipeline.py               # Main DocumentPipeline coordinator
│   └── search_engine.py          # Semantic search & Q&A
│
├── config/                        # Configuration management
│   ├── __init__.py               # Package exports
│   └── settings.py               # All configuration classes
│
├── utils/                         # Utility functions
│   ├── __init__.py               # Package exports
│   └── helpers.py                # Helper functions (I/O, formatting, GPU)
│
├── monitoring/                    # Live monitoring (future split)
│   └── (to be organized)
│
├── ui/                           # User interfaces (future split)
│   └── (to be organized)
│
├── data/                         # Runtime data
│   ├── documents/                # User uploaded documents
│   ├── index/                    # FAISS vector index
│   ├── operation_history.jsonl  # Operation logs
│   └── pipeline_status.json     # Real-time status
│
├── docs/                         # Documentation
│   └── (various documentation files)
│
├── backend_logic.py              # Backward compatibility wrapper
├── config.py                     # Backward compatibility wrapper
├── utils.py                      # Backward compatibility wrapper
├── backend_live.py               # Live monitoring GUI (entry point)
├── app_tkinter.py                # Main Tkinter app (entry point)
└── backend_debug.py              # Debug utilities
```

## Module Descriptions

### Core Package (`core/`)

#### `analytics.py`
- **Purpose**: Comprehensive logging and monitoring
- **Key Classes**: 
  - `AnalyticsLogger` - Operation logging, status updates, event bus integration
- **Responsibilities**:
  - Operation history tracking (JSONL format)
  - Real-time status file updates
  - Event bus publishing for live monitoring

#### `components.py`
- **Purpose**: Initialize all processing components
- **Key Classes**:
  - `ComponentInitializer` - GPU detection, embeddings, LLM, text splitter
- **Responsibilities**:
  - GPU/CPU detection and optimization
  - Initialize Docling document converter
  - Initialize LangChain components (embeddings, LLM, text splitter)
  - Create prompt template

#### `document_processor.py`
- **Purpose**: Document ingestion and indexing
- **Key Classes**:
  - `DocumentProcessor` - File processing and FAISS index management
- **Responsibilities**:
  - Read and parse documents (text, PDF, DOCX)
  - Split text into chunks
  - Generate embeddings
  - Build and save FAISS index
  - Load existing index

#### `search_engine.py`
- **Purpose**: Semantic search and Q&A
- **Key Classes**:
  - `SearchEngine` - Vector search and LLM response generation
- **Responsibilities**:
  - FAISS similarity search
  - Relevance filtering with filename priority
  - Context building from search results
  - LLM-based answer generation
  - Streaming response support

#### `pipeline.py`
- **Purpose**: Main coordinator for all components
- **Key Classes**:
  - `DocumentPipeline` - Public API and component integration
- **Responsibilities**:
  - Initialize all subsystems
  - Provide unified public API
  - Coordinate document processing and search
  - Backward compatibility properties

### Config Package (`config/`)

#### `settings.py`
- **Purpose**: Centralized configuration
- **Key Classes**:
  - `Paths` - Directory paths
  - `ModelConfig` - LLM and embedding settings
  - `SearchConfig` - Search parameters
  - `PerformanceConfig` - GPU and performance settings
  - `FileConfig` - Supported file types
  - `LoggingConfig` - Operation and status types
- **Global Instances**: `paths`, `model_config`, `search_config`, etc.

### Utils Package (`utils/`)

#### `helpers.py`
- **Purpose**: Reusable utility functions
- **Key Functions**:
  - File I/O: `read_text_file`, `write_json_atomic`, `append_jsonl`, `read_jsonl`
  - Document processing: `extract_clean_content`, `normalize_filename`
  - Formatting: `format_timestamp`, `format_duration`, `format_file_size`
  - GPU utilities: `get_gpu_info`, `should_optimize_for_gpu`
  - Validation: `is_valid_question`, `truncate_text`
  - Document listing: `get_document_files`, `count_document_files`

## Entry Points

### `app_tkinter.py`
- Main desktop application
- Tkinter-based GUI
- Document upload and Q&A interface
- Imports: `backend_logic`, `config`, `utils`

### `backend_live.py`
- Live monitoring GUI
- Task-manager-style interface
- Real-time operation tracking
- Imports: `backend_logic`, `config`, `utils`

### `backend_debug.py`
- Debug utilities and tests
- Imports: `backend_logic`

## Backward Compatibility

The following wrapper files maintain backward compatibility:

- **`backend_logic.py`**: Re-exports `DocumentPipeline` from `core`
- **`config.py`**: Re-exports all configuration from `config.settings`
- **`utils.py`**: Re-exports all utilities from `utils.helpers`

This allows existing code to continue using:
```python
from backend_logic import DocumentPipeline
from config import paths, model_config
from utils import read_jsonl, format_timestamp
```

## Import Patterns

### Internal Imports (within packages)
```python
# In core modules
from config import paths, model_config
from utils import write_json_atomic, append_jsonl
```

### Package Imports (for new code)
```python
# Recommended for new code
from core import DocumentPipeline
from config import paths, model_config, search_config
from utils import read_jsonl, format_timestamp
```

### Legacy Imports (backward compatible)
```python
# Still works for existing code
from backend_logic import DocumentPipeline
from config import paths
from utils import read_jsonl
```

## Benefits of This Structure

1. **Modularity**: Each module has a clear, single purpose
2. **Maintainability**: Easier to find and modify specific functionality
3. **Testability**: Smaller modules are easier to test in isolation
4. **Scalability**: Easy to add new features without cluttering existing code
5. **Readability**: Clear separation of concerns
6. **Backward Compatibility**: Existing code continues to work without changes

## Next Steps

Future improvements could include:

1. **Split `backend_live.py`** into `monitoring/` package:
   - `event_bus.py` - Event publishing system
   - `gui.py` - Main GUI application
   - `monitors.py` - Individual monitor implementations

2. **Create `ui/` package** for app_tkinter.py:
   - `app.py` - Main application
   - `components.py` - UI components
   - `actions.py` - Event handlers

3. **Add tests/**:
   - `test_pipeline.py`
   - `test_search.py`
   - `test_document_processor.py`

4. **Add examples/**:
   - `basic_usage.py`
   - `custom_config.py`
   - `advanced_search.py`

## Migration Guide

If you have existing code using the old structure:

### No changes needed!
The refactored structure maintains full backward compatibility. Your existing imports will continue to work:

```python
# Old code - still works
from backend_logic import DocumentPipeline
from config import paths, model_config
from utils import read_jsonl

# Initialize pipeline
pipeline = DocumentPipeline()
```

### Optional: Update to new structure
For new code, you can use the more explicit package imports:

```python
# New recommended style
from core import DocumentPipeline
from core.analytics import AnalyticsLogger
from core.search_engine import SearchEngine
from config import paths, model_config
from utils import read_jsonl
```

This provides better clarity about where functionality comes from.

## Summary

The refactored structure organizes code into logical packages while maintaining full backward compatibility. The system now has:

- **Clear separation of concerns**: Analytics, components, processing, search
- **Modular architecture**: Easy to understand, modify, and extend
- **No breaking changes**: All existing code continues to work
- **Better organization**: Related code grouped together
- **Scalable foundation**: Ready for future growth

All functionality remains identical - only the organization has changed.
