# 🔧 Debug and Testing Guide

Comprehensive debugging and testing utilities for the Enhanced Document Q&A Pipeline.

## 📁 Current Debug Tools

### `backend_debug.py` - Comprehensive Testing Suite
- **Purpose**: Complete system validation and performance testing
- **Features**:
  - System component testing (imports, Docling, Ollama, embeddings)
  - Database content inspection (FAISS index validation)
  - Pipeline initialization and document processing tests
  - Enhanced search functionality validation
  - Q&A functionality testing with real queries
  - Edge case testing (empty queries, non-existent docs, emojis)
  - Performance benchmarking
- **Usage**: `python backend_debug.py`
- **Expected Results**: All tests should pass, ~6ms search time

### `backend_logic.py` - Core Pipeline with Debug Methods
- **Purpose**: Main DocumentPipeline class with built-in debugging
- **Debug Methods**:
  - `debug_search(query)` - Returns detailed search results with scores
  - `search(query, score_threshold)` - Enhanced search with configurable thresholds
  - `ask(question)` - Q&A with source attribution
- **Usage**: Import and use directly for custom debugging

## 🧪 Testing Workflow

### 1. System Health Check
```bash
python backend_debug.py
```

**What to look for:**
- ✅ All 7 system components pass
- ✅ FAISS index exists and loads
- ✅ Documents directory has files
- ✅ Ollama responds with llama3:latest

### 2. Search Functionality Validation
The debug script tests these queries:
- `"Invoice_Outline_-_Sheet1_1.pdf"` → Should find Invoice PDF as #1
- `"product manual"` → Should find product_manual.md as #1
- `"company handbook"` → Should find company_handbook.md as #1
- `"algebra operations"` → Should find Math PDF as #1

### 3. Performance Benchmarks
Expected metrics:
## 🚀 Quick Debugging Commands

### Manual Testing Examples
```python
from backend_logic import DocumentPipeline

# Initialize and test
pipeline = DocumentPipeline()
pipeline.load_index()

# Test search with debug info
debug_info = pipeline.debug_search("invoice outline")
print(f"Query: {debug_info['query']}")
print(f"Results: {debug_info['total_results']}")
for r in debug_info["results"][:3]:
    print(f"{r['rank']}. {r['source']} (score: {r['score']:.3f})")

# Test Q&A
result = pipeline.ask("What is in the invoice document?")
print(f"Answer: {result['answer'][:100]}...")
print(f"Sources: {[s['source'] for s in result['sources']]}")
```

### Performance Testing
```python
import time
from backend_logic import DocumentPipeline

pipeline = DocumentPipeline()
pipeline.load_index()

# Test search speed
queries = ["invoice", "manual", "handbook", "math"]
times = []

for query in queries:
    start = time.time()
    results = pipeline.search(query)
    end = time.time()
    times.append(end - start)
    print(f"{query}: {len(results)} results in {(end-start)*1000:.1f}ms")

print(f"Average: {sum(times)/len(times)*1000:.1f}ms")
```

## � Troubleshooting Common Issues

### Issue: No search results
```python
# Check if index exists
from pathlib import Path
index_path = Path("data/index/faiss_index.faiss")
print(f"Index exists: {index_path.exists()}")

# Check document count
pipeline = DocumentPipeline()
if pipeline.load_index():
    test_results = pipeline.search("test", score_threshold=10.0)  # Very lenient
    print(f"Total indexed chunks: {len(test_results)}")
```

### Issue: Ollama connection problems
```python
from langchain_ollama import OllamaLLM

try:
    llm = OllamaLLM(model="llama3:latest")
    response = llm.invoke("Hello")
    print(f"Ollama working: {response[:50]}...")
except Exception as e:
    print(f"Ollama error: {e}")
    print("Run: ollama serve")
```

### Issue: Poor search results
```python
# Check score thresholds
pipeline = DocumentPipeline()
pipeline.load_index()

query = "your query here"
results = pipeline.search(query, score_threshold=5.0)  # Very lenient

print(f"Found {len(results)} results:")
for i, r in enumerate(results[:5]):
    print(f"{i+1}. {r['source']} (score: {r['relevance_score']:.3f})")
```

### Run Full Pipeline Test:
```bash
# Run all debug files in sequence (Windows PowerShell)
python debug-1-extraction.py; python debug-2-chunking.py; python debug-3-embedding.py; python debug-4-search.py; python debug-5-chat.py
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
