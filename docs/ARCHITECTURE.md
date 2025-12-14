# Architecture Overview - Document Q&A Agent

## System Architecture

The Document Q&A Agent is built using a modern, modular architecture designed for professional document processing and AI-powered question answering.

```
┌─────────────────────────────────────────────────────────────────┐
│                       Entry Points Layer                         │
├─────────────────────────────────────────────────────────────────┤
│  launcher.py         program/              tests/              │
│  (Interactive)       (Desktop GUI)         (Validation)         │
└─────────────┬─────────────────┬─────────────────┬──────────────┘
              │                 │                 │
              └─────────────────┴─────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Core Logic Layer                             │
├─────────────────────────────────────────────────────────────────┤
│  logic/rag_pipeline_orchestrator.py   (DocumentPipeline)        │
│  logic/model_component_initializer.py  (ComponentInitializer)   │
│  logic/document_ingestion_handler.py   (DocumentProcessor)      │
│  logic/semantic_search_qa_engine.py    (SearchEngine)           │
└─────────────┬─────────────────┬─────────────────┬──────────────┘
              │                 │                 │
              ▼                 ▼                 ▼
┌──────────────────┐  ┌─────────────────┐  ┌────────────────────┐
│   Configuration  │  │   Monitoring    │  │     Utilities      │
│     Package      │  │    Package      │  │     Package        │
├──────────────────┤  ├─────────────────┤  ├────────────────────┤
│ • settings.py    │  │ • analytics/    │  │ • system_io_helpers│
│ • Model configs  │  │ • monitor/      │  │ • File I/O         │
│ • Search params  │  │ • Event bus     │  │ • GPU utilities    │
│ • Performance    │  │ • Live GUI      │  │ • Formatting       │
└──────────────────┘  └─────────────────┘  └────────────────────┘
```

## Core Pipeline Architecture

### DocumentPipeline Coordinator

```
┌──────────────────────────────────────────────────────────────┐
│                    DocumentPipeline                          │
│                   (Main Coordinator)                         │
└────────┬─────────────────┬──────────────────┬───────────────┘
         │                 │                  │
         ▼                 ▼                  ▼
┌─────────────────┐ ┌─────────────────┐ ┌────────────────────┐
│ ComponentInit.  │ │DocumentProcessor│ │   SearchEngine     │
│                 │ │                 │ │                    │
│ • GPU detect    │ │ • process_all() │ │ • semantic_search()│
│ • Init LLM      │ │ • load_index()  │ │ • relevance_filter │
│ • Init embeddings│ │ • save_index() │ │ • context_building │
│ • Init splitter │ │ • Docling parse │ │ • LLM generation   │
└─────────────────┘ └─────────────────┘ └────────────────────┘
         │                 │                  │
         └─────────────────┴──────────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ AnalyticsLogger │
                  │                 │
                  │ • log_operation │
                  │ • update_status │
                  │ • event_bus     │
                  └─────────────────┘
```

## Data Flow Architecture

### Document Processing Pipeline
```
User Upload → File Validation → Document Processing → Vector Index
     │              │                    │                │
     ▼              ▼                    ▼                ▼
┌─────────┐ ┌─────────────┐ ┌─────────────────┐ ┌───────────────┐
│ File    │ │ Format      │ │ Docling         │ │ FAISS Index   │
│ Upload  │ │ Check       │ │ Conversion      │ │ Storage       │
│ (GUI)   │ │ (Utils)     │ │ (Components)    │ │ (Processor)   │
└─────────┘ └─────────────┘ └─────────────────┘ └───────────────┘
     │              │                    │                │
     └──────────────┴────────────────────┴────────────────┘
                               │
                               ▼
                    ┌─────────────────┐
                    │ Analytics       │
                    │ Logging         │
                    └─────────────────┘
```

### Question-Answer Pipeline  
```
User Query → Embedding → Vector Search → Context Build → LLM → Response
     │           │            │             │            │        │
     ▼           ▼            ▼             ▼            ▼        ▼
┌─────────┐ ┌─────────┐ ┌─────────────┐ ┌─────────┐ ┌────────┐ ┌────────┐
│ Question│ │ Query   │ │ FAISS       │ │ Relevance│ │ Ollama │ │ Stream │
│ Input   │ │ Vector  │ │ Similarity  │ │ Filter   │ │ LLM    │ │ Output │
│ (GUI)   │ │ (HF)    │ │ Search      │ │ (Custom) │ │ (API)  │ │ (GUI)  │
└─────────┘ └─────────┘ └─────────────┘ └─────────┘ └────────┘ └────────┘
```

### Real-time Monitoring Flow
```
Operation Events → Analytics Logger → Event Bus → Live GUI
       │               │                │           │
       ▼               ▼                ▼           ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ System      │ │ JSONL       │ │ Threading   │ │ Tkinter     │
│ Operations  │ │ History     │ │ Queue       │ │ Monitor     │
└─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘
```

## Technology Stack Details

### **Core AI/ML Technologies**
- **Document Processing**: Docling (IBM Research) - RT-DETR + TableFormer
- **Embeddings**: HuggingFace all-MiniLM-L6-v2 (384 dimensions)
- **Vector Search**: FAISS (Facebook AI) with GPU acceleration
- **LLM**: Ollama qwen2.5:1.5b (local inference)
- **Text Processing**: LangChain RecursiveCharacterTextSplitter

### **Desktop Application Stack**
- **GUI Framework**: Python Tkinter with custom components
- **Threading**: Background processing for non-blocking operations
- **Monitoring**: Real-time event bus with live GUI updates
- **File I/O**: Atomic operations with JSONL logging

### **Configuration & Utilities**
- **Settings Management**: Centralized dataclass-based configuration
- **Logging**: Structured operation history with event categorization
- **System Utilities**: GPU detection, memory management, file validation
- **Error Handling**: Graceful fallbacks with comprehensive logging

## Component Integration Patterns

### **Initialization Pattern**
```python
pipeline = DocumentPipeline()
    └─→ components = ComponentInitializer()
        ├─→ Detect GPU/CPU capabilities
        ├─→ Initialize embeddings model
        ├─→ Initialize LLM connection
        └─→ Setup text splitter

    └─→ analytics = AnalyticsLogger()
        ├─→ Setup operation logging
        ├─→ Initialize event bus
        └─→ Configure status tracking

    └─→ doc_processor = DocumentProcessor()
        └─→ Setup document processing pipeline

    └─→ search_engine = SearchEngine()
        └─→ Setup search and Q&A pipeline
```

### **Configuration Injection Pattern**
All components receive configuration through centralized settings:
- **Path Configuration**: Data directories, index locations
- **Model Configuration**: LLM settings, embedding parameters  
- **Search Configuration**: Relevance thresholds, context limits
- **Performance Configuration**: GPU settings, memory limits

### **Event-Driven Monitoring Pattern**
Operations are logged through a centralized analytics system:
- **Operation Tracking**: Every major operation is logged with timing
- **Status Broadcasting**: Real-time status updates via event bus
- **Performance Metrics**: Memory usage, processing times, throughput

## Scalability & Performance Architecture

### **Memory Management Strategy**
```
GPU Memory (if available)
├─→ Embeddings Model (1-2GB)
├─→ FAISS Index (scales with docs)
└─→ LLM Context (managed by Ollama)

System Memory
├─→ Document Processing (chunked)
├─→ GUI Components (minimal)
└─→ Monitoring Data (bounded queues)
```

### **Processing Optimization**
- **Document Chunking**: Dynamic sizing based on GPU availability
- **Batch Processing**: Multiple documents processed efficiently
- **Streaming Responses**: Real-time answer generation
- **Index Optimization**: FAISS GPU acceleration when available

### **Concurrency Strategy**
- **Background Processing**: Non-blocking document processing
- **GUI Threading**: Responsive interface during operations  
- **Event Bus**: Thread-safe communication between components
- **Resource Locking**: Safe access to shared resources

## Security & Data Privacy

### **Local-First Architecture**
- **No External APIs**: All processing happens locally
- **Data Isolation**: Documents never leave the local machine
- **Secure Storage**: FAISS indices stored locally
- **Process Isolation**: Ollama runs as separate service

### **File System Security**
- **Sandboxed Paths**: All operations within designated directories
- **Input Validation**: File type and size restrictions
- **Atomic Operations**: Safe file writes with rollback capability
- **Permission Checks**: Verify read/write access before operations

## Extension Points & Future Architecture

### **Modular Design Benefits**
- **Plugin Architecture**: Easy addition of new document formats
- **Model Swapping**: Simple LLM or embedding model replacement
- **UI Extensions**: Additional interfaces (web, CLI, API)
- **Storage Backends**: Alternative to FAISS for specialized use cases

### **Planned Architectural Enhancements**
- **Multi-Index Support**: Topic-based or user-based index separation  
- **Distributed Processing**: Scale beyond single-machine limits
- **API Layer**: REST API for external integrations
- **Advanced Monitoring**: Metrics collection and analysis dashboard

---

**Architecture Philosophy**: Clean separation of concerns with professional error handling, comprehensive monitoring, and extensible design patterns for enterprise-grade document processing.

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
