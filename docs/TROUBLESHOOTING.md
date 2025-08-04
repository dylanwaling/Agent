# 🔧 Troubleshooting Guide - Document Processing Issues

## 🚨 Common Issues and Solutions

### Issue 1: Server Keeps Restarting During Document Processing

**Symptoms:**
- Logs show multiple "Restarting with watchdog" messages
- Processing starts but never completes
- Server restarts every time Docling processes a PDF

**Root Cause:**
Flask debug mode detects file changes from PyTorch/Docling libraries and restarts the server.

**Solutions:**

**✅ Solution: Use the stable app.py**
```bash
python app.py
```
*Debug mode has been removed to prevent restart loops*

### Issue 2: Documents Won't Process / Processing Fails

**Symptoms:**
- Click "Process Documents" but nothing happens
- Processing starts but crashes
- No success message appears

**Solutions:**

**Step 1: Check the startup logs**
```bash
python app.py
```
Look for any error messages during pipeline initialization.

**Step 2: Clean restart**
```bash
# Stop the server (Ctrl+C)
# Remove any partial index files
rmdir /s "data\index"
# Restart
python app.py
```

**Step 3: Check individual documents**
```bash
python backend_debug.py
```
This will test each document individually and show which ones fail.

### Issue 3: Server Says "No existing index found" repeatedly

**Symptoms:**
- Processing appears to work but index isn't saved
- Every restart shows "No existing index found"

**Solutions:**

**Check file permissions:**
- Make sure you can write to the `data/index` directory
- Try running as administrator if needed

**Manual verification:**
```bash
# Check if index was created
dir "data\index\faiss_index"
# You should see: index.faiss and index.pkl
```

### Issue 4: Processing Starts But Never Finishes

**Symptoms:**
- Processing log messages appear
- But no "Documents processed successfully!" message
- Status stays on "Processing documents..."

**Solutions:**

**Force a clean restart:**
```bash
# 1. Stop server (Ctrl+C)
# 2. Clean up
rmdir /s "data\index"
# 3. Restart
python app.py
# 4. Try processing again
```

## 🎯 Best Practices

### For Stable Document Processing:
1. **Always use:** `python app.py`
2. **If stuck:** Clean restart with index removal
3. **For testing:** Use `python backend_debug.py` to test individual components

### For Development:
1. **Running the app:** Use `python app.py`
2. **Full testing:** Use `python backend_debug.py`

## 🔍 Debug Commands

### Check System Status:
```bash
python backend_debug.py
```

### Check Pipeline Manually:
```python
from backend_logic import DocumentPipeline
pipeline = DocumentPipeline()
success = pipeline.process_documents()
print(f"Success: {success}")
```

### Check Flask App:
```python
from app import get_pipeline
pipeline = get_pipeline()
print(f"Pipeline available: {pipeline is not None}")
```

### Verify Index:
```bash
# Check if index files exist
dir "data\index\faiss_index"
```

## 📋 Startup Checklist

Before processing documents:

- [ ] ✅ Ollama is running: `ollama list`
- [ ] ✅ Using stable mode: `python app.py`
- [ ] ✅ Documents are in `data/documents/` folder
- [ ] ✅ No partial index files (clean restart if needed)

Following these guidelines will prevent the restart loop and processing issues you experienced.
