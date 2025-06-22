# 🔧 Debug and Testing Utilities

This directory contains comprehensive debug and testing utilities for each step of the Docling + Ollama pipeline.

## 📁 Debug Files Overview

### `debug_utils.py` - Combined Database & Search Utilities
- **Purpose**: Combined functionality from old `check_data.py` and `test_search.py`
- **Functions**: Database status, search testing, embedding visualization, file structure check
- **Usage**: `python debug_utils.py`

### `1-extraction-debug.py` - Document Extraction Testing
- **Purpose**: Debug and test document extraction (PDFs, HTML, sitemaps)
- **Features**:
  - Internet connectivity check
  - Docling installation verification
  - PDF extraction testing (ArXiv paper)
  - HTML extraction testing (GitHub page)
  - Sitemap functionality testing
  - AI model cache analysis
  - Performance timing
- **Usage**: `python 1-extraction-debug.py`

### `2-chunking-debug.py` - Document Chunking Testing
- **Purpose**: Debug and test document chunking process
- **Features**:
  - Tokenizer wrapper testing
  - Document extraction for chunking
  - HybridChunker creation and configuration
  - Chunk analysis (length, metadata, quality)
  - Performance timing
  - Content coherence checking
- **Usage**: `python 2-chunking-debug.py`

### `3-embedding-debug.py` - Embeddings & Database Testing
- **Purpose**: Debug and test embedding generation and LanceDB storage
- **Features**:
  - SentenceTransformer model testing
  - LanceDB connection and operations
  - Document processing pipeline
  - Metadata extraction verification
  - Embedding generation performance
  - Database insertion testing
  - Memory usage analysis
- **Usage**: `python 3-embedding-debug.py`

### `4-search-debug.py` - Search Functionality Testing
- **Purpose**: Debug and test search and retrieval functionality
- **Features**:
  - Database connection verification
  - Embedding model consistency
  - Database content analysis
  - Search accuracy testing
  - Performance benchmarking
  - Result quality assessment
  - Multiple query testing
- **Usage**: `python 4-search-debug.py`

### `5-chat-debug.py` - Chat Interface Testing
- **Purpose**: Debug and test chat functionality with Ollama
- **Features**:
  - Ollama service connection testing
  - Model availability checking
  - Chat API functionality
  - Context retrieval testing
  - Response generation testing
  - Streaming response testing
  - Error handling verification
  - Full chat simulation
- **Usage**: `python 5-chat-debug.py`

## 🚀 Quick Testing Commands

### Test Individual Steps:
```bash
# Test extraction
python 1-extraction-debug.py

# Test chunking  
python 2-chunking-debug.py

# Test embeddings
python 3-embedding-debug.py

# Test search
python 4-search-debug.py

# Test chat
python 5-chat-debug.py
```

### Test Database & Search:
```bash
# Combined database and search utilities
python debug_utils.py
```

### Run Full Pipeline Test:
```bash
# Run all debug files in sequence
python 1-extraction-debug.py && python 2-chunking-debug.py && python 3-embedding-debug.py && python 4-search-debug.py && python 5-chat-debug.py
```

## 🔍 What Each Debug File Tests

### Step 1: Extraction Debug
- ✅ Internet connectivity
- ✅ Docling installation
- ✅ AI model caching
- ✅ PDF processing (ArXiv paper)
- ✅ HTML processing (GitHub page)
- ✅ Sitemap extraction
- ✅ Performance timing
- ✅ Content validation

### Step 2: Chunking Debug
- ✅ Tokenizer functionality
- ✅ Document extraction
- ✅ Chunker configuration
- ✅ Chunk generation
- ✅ Content quality analysis
- ✅ Metadata preservation
- ✅ Performance metrics

### Step 3: Embedding Debug
- ✅ SentenceTransformer model
- ✅ LanceDB operations
- ✅ Embedding generation
- ✅ Metadata processing
- ✅ Database insertion
- ✅ Memory usage
- ✅ Data consistency

### Step 4: Search Debug
- ✅ Database connectivity
- ✅ Model consistency
- ✅ Content analysis
- ✅ Search accuracy
- ✅ Performance timing
- ✅ Result quality
- ✅ Query variations

### Step 5: Chat Debug
- ✅ Ollama service status
- ✅ Model availability
- ✅ API functionality
- ✅ Context retrieval
- ✅ Response generation
- ✅ Streaming responses
- ✅ Error handling

## 🔧 Troubleshooting Guide

### Common Issues & Solutions:

**❌ Internet Connection Failed**
- Check network connectivity
- Verify firewall settings
- Try different URLs

**❌ Ollama Connection Failed**
- Start Ollama service: `ollama serve`
- Check if running: `ollama list`
- Verify port 11434 is available

**❌ Model Not Found**
- Install models: `ollama pull llama3`
- Check available models: `ollama list`

**❌ Database Not Found**
- Run step 3 first: `python 3-embedding.py`
- Check data directory exists

**❌ Poor Search Results**
- Verify embeddings: `python 3-embedding-debug.py`
- Check database content: `python debug_utils.py`

**❌ Slow Performance**
- Check model caching: `python 1-extraction-debug.py`
- Monitor memory usage: `python 3-embedding-debug.py`

## 📊 Expected Output

Each debug script will show:
- ✅ **Green checkmarks**: Tests passed
- ❌ **Red X marks**: Tests failed
- ⚠️ **Yellow warnings**: Potential issues
- 📊 **Statistics**: Performance metrics
- 💡 **Tips**: Improvement suggestions

## 🎯 Best Practices

1. **Run in sequence**: Follow steps 1→2→3→4→5
2. **Check prerequisites**: Ensure Ollama is running before step 5
3. **Monitor performance**: Watch timing and memory usage
4. **Verify data**: Check database content before testing search
5. **Test variations**: Try different queries and models

## 🔄 Re-running Tests

- **Safe to re-run**: All debug scripts are non-destructive
- **Database recreation**: Step 3 debug may recreate test tables
- **Cache usage**: Subsequent runs will be faster due to caching

These debug utilities provide comprehensive testing and troubleshooting capabilities for the entire pipeline. Use them to identify issues, verify functionality, and optimize performance at each step.
