# Document Q&A Agent - Quick Start Guide

## 🚀 Running the Application

### Option 1: Use the Launcher (Recommended)
```powershell
python launcher.py
```
This provides an interactive menu to choose which component to launch.

### Option 2: Direct Launch

#### Main Application (Desktop GUI)
```powershell
python main.py
```
- Upload and process documents
- Ask questions and get AI-powered answers
- Stream responses in real-time

#### Monitoring Dashboard
```powershell
python monitoring_dashboard.py
```
- Real-time operation tracking
- Performance metrics
- System status monitoring

#### System Tests
```powershell
python system_test.py
```
- Test all components
- Verify system health
- Performance benchmarks

## 📁 Project Structure

```
/Agent/
├── launcher.py              # Interactive launcher (NEW!)
├── main.py                  # Main desktop application (renamed from app_tkinter.py)
├── monitoring_dashboard.py  # Live monitoring GUI (renamed from backend_live.py)
├── system_test.py          # Debug and testing tools (renamed from backend_debug.py)
│
├── core/                    # Core pipeline modules
│   ├── analytics.py        # Logging and monitoring
│   ├── components.py       # Component initialization
│   ├── document_processor.py  # Document processing
│   ├── pipeline.py         # Main coordinator
│   └── search_engine.py    # Search and Q&A
│
├── config/                  # Configuration
│   └── settings.py         # All settings
│
├── utils/                   # Utilities
│   └── helpers.py          # Helper functions
│
├── data/                    # Runtime data
│   ├── documents/          # Uploaded documents
│   ├── index/              # FAISS vector index
│   ├── operation_history.jsonl
│   └── pipeline_status.json
│
├── docs/                    # Documentation
│   ├── ARCHITECTURE.md
│   ├── PROJECT_STRUCTURE.md
│   ├── REFACTORING_COMPLETION.md
│   └── ... (other docs)
│
├── backend_logic.py        # Backward compatibility
├── config.py               # Backward compatibility
└── utils.py                # Backward compatibility
```

## 🎯 What Changed

### Professional Naming
- `app_tkinter.py` → **`main.py`**
- `backend_live.py` → **`monitoring_dashboard.py`**
- `backend_debug.py` → **`system_test.py`**

### Cleanup
- ❌ Removed empty `monitoring/` folder
- ❌ Removed empty `ui/` folder
- ✅ Fixed incomplete import statements

### New Features
- ✨ Added **`launcher.py`** - Interactive menu for all components

## 📝 Notes

- **Backward compatibility maintained**: Old import statements still work
- **No functional changes**: All features work exactly as before
- **Professional structure**: Clear, organized file naming
- **Better usability**: Interactive launcher for easy access

## 🔗 Quick Links

- [Full Architecture](docs/ARCHITECTURE.md)
- [Project Structure](docs/PROJECT_STRUCTURE.md)
- [Refactoring Summary](docs/REFACTORING_COMPLETION.md)
