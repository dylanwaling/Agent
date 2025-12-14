# Architecture Overview - Document Q&A Agent

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Entry Points                             │
├─────────────────────────────────────────────────────────────────┤
│  app_tkinter.py          backend_live.py       backend_debug.py │
│  (Desktop GUI)           (Monitoring GUI)      (Debug Utils)    │
└────────────────┬─────────────────┬─────────────────┬───────────┘
                 │                 │                 │
                 └─────────────────┴─────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              Backward Compatibility Layer                        │
├─────────────────────────────────────────────────────────────────┤
│  backend_logic.py     config.py          utils.py               │
│  (Re-exports)         (Re-exports)       (Re-exports)           │
└────────────────┬──────────────┬──────────────┬─────────────────┘
                 │              │              │
                 ▼              ▼              ▼
┌────────────────────┐  ┌─────────────┐  ┌──────────────┐
│    Core Package    │  │   Config    │  │    Utils     │
│                    │  │  Package    │  │   Package    │
├────────────────────┤  ├─────────────┤  ├──────────────┤
│ • Analytics        │  │ • Settings  │  │ • Helpers    │
│ • Components       │  │ • Paths     │  │ • I/O        │
│ • Doc Processor    │  │ • Models    │  │ • Formatting │
│ • Search Engine    │  │ • Search    │  │ • GPU Utils  │
│ • Pipeline         │  │ • Perf      │  │ • Validation │
└────────────────────┘  └─────────────┘  └──────────────┘
```

## Core Package Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    DocumentPipeline                          │
│                   (Main Coordinator)                         │
└────────┬─────────────────┬──────────────────┬───────────────┘
         │                 │                  │
         ▼                 ▼                  ▼
┌─────────────────┐ ┌─────────────────┐ ┌────────────────────┐
│ AnalyticsLogger │ │ComponentInit.   │ │ DocumentProcessor  │
│                 │ │                 │ │                    │
│ • log_operation │ │ • GPU detect    │ │ • process_all()    │
│ • update_status │ │ • Init LLM      │ │ • load_index()     │
│ • event_bus     │ │ • Init embed    │ │ • save_index()     │
└─────────────────┘ │ • Init splitter │ └────────────────────┘
                    └─────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  SearchEngine    │
                    │                  │
                    │ • search()       │
                    │ • ask()          │
                    │ • ask_streaming()│
                    └──────────────────┘
```

## Data Flow

### Document Processing Flow
```
User Upload
    │
    ▼
DocumentProcessor.process_all_documents()
    │
    ├─→ Read files (TXT, PDF, DOCX)
    │   └─→ Use Docling for non-text files
    │
    ├─→ Split into chunks
    │   └─→ RecursiveCharacterTextSplitter
    │
    ├─→ Generate embeddings
    │   └─→ HuggingFaceEmbeddings (all-MiniLM-L6-v2)
    │
    ├─→ Build FAISS index
    │   └─→ Optimize for GPU if available
    │
    └─→ Save to disk
        └─→ data/index/faiss_index/
```

### Question Answering Flow
```
User Question
    │
    ▼
SearchEngine.ask(question)
    │
    ├─→ 1. Embed query
    │   └─→ HuggingFaceEmbeddings
    │
    ├─→ 2. FAISS similarity search
    │   └─→ Top K results (K=100)
    │
    ├─→ 3. Relevance filtering
    │   ├─→ Filename matching (strong/weak)
    │   └─→ Score thresholding
    │
    ├─→ 4. Context building
    │   └─→ Top 8 sources, max 3500 chars
    │
    ├─→ 5. LLM generation
    │   └─→ Ollama (qwen2.5:1.5b)
    │
    └─→ 6. Return answer + sources
        └─→ {answer: str, sources: list}
```

## Analytics & Monitoring Flow

```
Any Operation
    │
    ▼
AnalyticsLogger.log_operation()
    │
    ├─→ Append to operation_history.jsonl
    │   └─→ JSONL format for easy parsing
    │
    ├─→ Update pipeline_status.json
    │   └─→ Atomic write for safety
    │
    └─→ Publish to event_bus
        └─→ Real-time monitoring GUI
            └─→ backend_live.py displays
```

## Component Dependencies

```
┌─────────────────────────────────────────────────────────────┐
│                         pipeline.py                          │
│                    (DocumentPipeline)                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
       ┌───────────────┼───────────────┐
       │               │               │
       ▼               ▼               ▼
┌──────────────┐ ┌─────────────┐ ┌──────────────┐
│ analytics.py │ │components.py│ │doc_processor.│
└──────┬───────┘ └──────┬──────┘ └───────┬──────┘
       │                │                 │
       │                │                 └───────┐
       │                │                         │
       ▼                ▼                         ▼
┌──────────────┐ ┌──────────────┐       ┌────────────────┐
│   config     │ │    utils     │       │search_engine.py│
└──────────────┘ └──────────────┘       └────────────────┘
```

## Technology Stack

### Core Technologies
- **Python 3.x**: Main programming language
- **Docling**: Document conversion (PDF, DOCX → text)
- **LangChain**: Text processing and LLM integration
- **FAISS**: Vector similarity search (Facebook AI)
- **Ollama**: Local LLM inference (qwen2.5:1.5b)
- **HuggingFace**: Embeddings (all-MiniLM-L6-v2)

### UI Technologies
- **Tkinter**: Desktop GUI
- **Threading**: Background processing

### Data Storage
- **JSON**: Status and configuration
- **JSONL**: Operation history
- **FAISS Index**: Vector embeddings (binary)
- **PKL**: FAISS metadata

### Hardware Support
- **GPU**: CUDA support for FAISS and embeddings
- **CPU**: Fallback for systems without GPU

## Configuration Hierarchy

```
config/settings.py
    │
    ├─→ Paths
    │   ├─→ DATA_DIR
    │   ├─→ DOCS_DIR
    │   ├─→ INDEX_DIR
    │   └─→ STATUS_FILE, HISTORY_FILE
    │
    ├─→ ModelConfig
    │   ├─→ EMBEDDING_MODEL
    │   ├─→ LLM_MODEL
    │   └─→ CHUNK_SIZE, CHUNK_OVERLAP
    │
    ├─→ SearchConfig
    │   ├─→ SEARCH_K
    │   ├─→ SCORE_THRESHOLD
    │   └─→ MAX_CONTEXT_LENGTH
    │
    ├─→ PerformanceConfig
    │   ├─→ GPU_MEMORY_THRESHOLD
    │   └─→ MAX_FILE_SIZE_MB
    │
    ├─→ FileConfig
    │   └─→ SUPPORTED_EXTENSIONS
    │
    └─→ LoggingConfig
        └─→ OPERATION_TYPES, STATUS_TYPES
```

## Error Handling Strategy

```
Try/Except Blocks
    │
    ├─→ Component Initialization
    │   └─→ Graceful fallback to CPU
    │
    ├─→ Document Processing
    │   └─→ Skip failed files, continue
    │
    ├─→ Search Operations
    │   └─→ Return empty results
    │
    ├─→ LLM Generation
    │   └─→ Return error message
    │
    └─→ Analytics Logging
        └─→ Log errors but continue
```

## Scalability Considerations

### Current Limits
- **Documents**: Hundreds of documents (tested)
- **Index Size**: Limited by RAM/GPU memory
- **Query Speed**: <1s for typical searches
- **Concurrent Users**: Single user (desktop app)

### Future Enhancements
- **Distributed FAISS**: For larger document sets
- **Multiple Indexes**: Per-category indexing
- **Caching**: Frequently asked questions
- **Async Processing**: Non-blocking operations
- **Multi-user**: Web API endpoint

## Security Considerations

- **Local Processing**: All data stays on local machine
- **No Network**: No external API calls for processing
- **File Access**: Limited to designated folders
- **Deserialization**: Marked as dangerous but local-only

## Performance Optimizations

1. **GPU Acceleration**
   - FAISS index on GPU
   - Embeddings on GPU
   - Automatic fallback to CPU

2. **Memory Management**
   - Chunked document processing
   - GPU memory limits for 6GB cards
   - Streaming responses

3. **Efficient Search**
   - FAISS for fast similarity search
   - Score-based filtering
   - Context length limits

4. **Atomic Operations**
   - Atomic file writes
   - Transaction-safe index updates

## Summary

The architecture provides:

✅ **Modular Design**: Clean separation of concerns
✅ **Scalable Structure**: Easy to extend and modify
✅ **Robust Error Handling**: Graceful degradation
✅ **Performance Optimized**: GPU acceleration where available
✅ **Well Documented**: Clear component responsibilities
✅ **Backward Compatible**: No breaking changes
✅ **Production Ready**: Tested and verified

This architecture supports both current functionality and future enhancements while maintaining code quality and developer experience.
