#!/usr/bin/env python3
"""
Run Package Entry Point
Allows running: python -m run
"""

if __name__ == '__main__':
    from .application import *
    import tkinter as tk
    import sys
    from pathlib import Path
    import subprocess
    from utils.helpers import count_document_files
    
    print("🚀 Document Q&A - Desktop Interface")
    print("=" * 50)
    
    # Launch live monitoring GUI
    try:
        print("📊 Starting live system monitor GUI...")
        project_root = Path(__file__).parent.parent.absolute()
        python_exe = sys.executable
        if python_exe.endswith('python.exe'):
            pythonw_exe = python_exe.replace('python.exe', 'pythonw.exe')
        else:
            pythonw_exe = python_exe
        
        subprocess.Popen(
            [pythonw_exe, "-m", "run.dashboard"],
            cwd=str(project_root),
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
        )
        print("✅ Live monitor GUI launched")
    except Exception as e:
        print(f"⚠️ Failed to launch monitor: {e}")
    
    # Check documents
    doc_count = count_document_files()
    if doc_count > 0:
        print(f"📄 Found {doc_count} documents ready for processing")
    else:
        print("📂 Documents directory will be created on first upload")
    
    print("✅ Starting desktop application...")
    
    # Create and run Tkinter app
    root = tk.Tk()
    app = DocumentQAApp(root)
    root.mainloop()
