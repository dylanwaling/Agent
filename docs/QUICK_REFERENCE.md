# Quick Reference Guide - New Infrastructure

## Configuration (`config.py`)

### Importing Configuration
```python
from config import (
    paths,                    # File paths
    model_config,            # Model settings
    search_config,           # Search parameters
    performance_config,      # Performance settings
    file_config,             # File type settings
    logging_config          # Logging constants
)
```

### Common Configuration Values

#### Paths
```python
paths.DOCS_DIR              # Path("data/documents")
paths.INDEX_DIR             # Path("data/index")
paths.STATUS_FILE           # Path("data/pipeline_status.json")
paths.HISTORY_FILE          # Path("data/operation_history.jsonl")
paths.FAISS_INDEX_FILE      # Path("data/index/faiss_index/index.faiss")
```

#### Model Configuration
```python
model_config.EMBEDDING_MODEL          # "sentence-transformers/all-MiniLM-L6-v2"
model_config.EMBEDDING_DIMENSION      # 384
model_config.LLM_MODEL                # "qwen2.5:1.5b"
model_config.LLM_TEMPERATURE          # 0.3
model_config.LLM_CONTEXT_WINDOW       # 4096
model_config.LLM_MAX_TOKENS           # 800
model_config.CHUNK_SIZE               # 1000
model_config.CHUNK_OVERLAP            # 200
```

#### Search Configuration
```python
search_config.SEARCH_K                      # 100
search_config.DEFAULT_SCORE_THRESHOLD       # 1.25
search_config.STRONG_MATCH_THRESHOLD        # 2.5
search_config.WEAK_MATCH_THRESHOLD          # 2.0
search_config.MAX_CONTEXT_SOURCES           # 8
search_config.MAX_CONTEXT_LENGTH            # 3500
```

#### Performance Configuration
```python
performance_config.MAX_FILE_SIZE_MB              # 16
performance_config.MAX_OPERATIONS_DISPLAY        # 50
performance_config.QUEUE_PROCESS_INTERVAL_MS     # 50
performance_config.STATUS_UPDATE_INTERVAL_MS     # 1000
```

#### Logging Configuration
```python
logging_config.OPERATION_TYPES['QUESTION_INPUT']        # 'question_input'
logging_config.OPERATION_TYPES['EMBEDDING_QUERY']       # 'embedding_query'
logging_config.OPERATION_TYPES['FAISS_SEARCH']          # 'faiss_search'
logging_config.STATUS_TYPES['IDLE']                     # 'IDLE'
logging_config.STATUS_TYPES['THINKING']                 # 'THINKING'
logging_config.STATUS_TYPES['PROCESSING']               # 'PROCESSING'
```

---

## Utilities (`utils.py`)

### File I/O Functions

#### Read Text File
```python
from utils import read_text_file

content = read_text_file(file_path)
# Returns: Optional[str] - content or None on error
```

#### Atomic JSON Write
```python
from utils import write_json_atomic

success = write_json_atomic(file_path, data_dict)
# Returns: bool - True if successful
```

#### Append to JSONL
```python
from utils import append_jsonl

success = append_jsonl(file_path, data_dict)
# Returns: bool - True if successful
```

#### Read JSONL File
```python
from utils import read_jsonl

operations = read_jsonl(file_path)
# Returns: list - list of dictionaries
```

### Document Processing Functions

#### Extract Clean Content
```python
from utils import extract_clean_content

clean = extract_clean_content(content, source_name)
# Removes filename prefix from indexed content
```

#### Create Searchable Content
```python
from utils import create_searchable_content

searchable = create_searchable_content(filename, stem, content)
# Adds filename prefix for better search matching
```

#### Normalize Filename
```python
from utils import normalize_filename

normalized = normalize_filename("My Document.pdf")
# Returns: "mydocument" (lowercase, no special chars)
```

### Time & Formatting Functions

#### Format Timestamp
```python
from utils import format_timestamp

time_str = format_timestamp(unix_timestamp)
# Returns: "HH:MM:SS" formatted string
```

#### Format Duration
```python
from utils import format_duration

duration_str = format_duration(seconds)
# Returns: "2.34s" or "1m 23s" or "1h 15m"
```

#### Format File Size
```python
from utils import format_file_size

size_str = format_file_size(bytes)
# Returns: "1.23 MB" or "456.78 KB"
```

### GPU Functions

#### Get GPU Info
```python
from utils import get_gpu_info

gpu_info = get_gpu_info()
# Returns: Dict with 'available', 'device', 'name', 'memory', 'message'
```

#### Check GPU Optimization
```python
from utils import should_optimize_for_gpu

optimize = should_optimize_for_gpu(gpu_memory_gb)
# Returns: bool - True if memory < threshold
```

### Validation Functions

#### Validate Question
```python
from utils import is_valid_question

valid = is_valid_question(question_str, min_length=3)
# Returns: bool - True if valid
```

#### Truncate Text
```python
from utils import truncate_text

truncated = truncate_text(long_text, max_length=100, suffix="...")
# Returns: str - truncated with suffix if needed
```

### Document Listing Functions

#### Get Document Files
```python
from utils import get_document_files

doc_files = get_document_files()
# Returns: List[Path] - list of document file paths
```

#### Count Document Files
```python
from utils import count_document_files

count = count_document_files()
# Returns: int - number of documents
```

---

## Migration Examples

### Example 1: Replace Hardcoded Path
```python
# Before
docs_dir = Path("data/documents")

# After
from config import paths
docs_dir = paths.DOCS_DIR
```

### Example 2: Replace File Reading
```python
# Before
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# After
from utils import read_text_file
content = read_text_file(file_path)
```

### Example 3: Replace Atomic Write
```python
# Before
temp_file = path.parent / f"temp_{time.time()}.tmp"
with open(temp_file, 'w') as f:
    json.dump(data, f)
    f.flush()
    os.fsync(f.fileno())
temp_file.rename(path)

# After
from utils import write_json_atomic
write_json_atomic(path, data)
```

### Example 4: Replace Magic Numbers
```python
# Before
results = search_results[:8]
if len(context) > 3500:
    context = context[:3500]

# After
from config import search_config
results = search_results[:search_config.MAX_CONTEXT_SOURCES]
if len(context) > search_config.MAX_CONTEXT_LENGTH:
    context = context[:search_config.MAX_CONTEXT_LENGTH]
```

### Example 5: Replace Operation Types
```python
# Before
operation_type = "question_input"
status = "PROCESSING"

# After
from config import logging_config
operation_type = logging_config.OPERATION_TYPES['QUESTION_INPUT']
status = logging_config.STATUS_TYPES['PROCESSING']
```

---

## Best Practices

### 1. Always Use Configuration
```python
# ✅ Good
from config import model_config
llm = OllamaLLM(model=model_config.LLM_MODEL)

# ❌ Bad
llm = OllamaLLM(model="qwen2.5:1.5b")
```

### 2. Use Utility Functions
```python
# ✅ Good
from utils import write_json_atomic
write_json_atomic(path, data)

# ❌ Bad
with open(path, 'w') as f:
    json.dump(data, f)
```

### 3. Import Only What You Need
```python
# ✅ Good
from config import paths, model_config
from utils import read_text_file, format_duration

# ❌ Bad
from config import *
from utils import *
```

### 4. Use Type Hints
```python
# ✅ Good
def process(query: str) -> List[Dict[str, Any]]:
    pass

# ❌ Bad
def process(query):
    pass
```

---

## Troubleshooting

### Import Errors
```python
# Problem: ImportError: cannot import name 'paths'
# Solution: Make sure config.py is in the same directory

# Check location
import sys
print(sys.path)  # Should include your project directory
```

### Configuration Not Applied
```python
# Problem: Changes to config.py not taking effect
# Solution: Restart Python interpreter or reimport

# Force reload
import importlib
import config
importlib.reload(config)
```

### Utility Function Errors
```python
# Problem: Utility returns None
# Solution: Check file permissions and paths

from pathlib import Path
file_path = Path("data/test.json")
print(f"Exists: {file_path.exists()}")
print(f"Readable: {file_path.is_file()}")
```

---

## Quick Checklist

When adding new features:

- [ ] Add configuration to appropriate dataclass in `config.py`
- [ ] Create reusable utilities in `utils.py` if needed
- [ ] Import from `config` and `utils` instead of hardcoding
- [ ] Use named constants instead of magic numbers
- [ ] Add type hints to new functions
- [ ] Update documentation if adding new config options

---

**For more details, see:**
- `docs/CODE_STRUCTURE_IMPROVEMENTS.md` - Comprehensive analysis
- `docs/REFACTORING_SUMMARY.md` - Implementation summary
- `config.py` - Configuration module source
- `utils.py` - Utilities module source
