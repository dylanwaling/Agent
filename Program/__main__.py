#!/usr/bin/env python3
"""
Run Package Entry Point
Allows running: python -m run
"""

if __name__ == '__main__':
    from .rag_desktop_app import *
    import tkinter as tk
    import sys
    from pathlib import Path
    import subprocess
    from utils.system_io_helpers import count_document_files
    
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
        
        # Import and run monitor from monitor package
        from monitor.performance_monitor import LiveMonitorGUI
        import threading
        
        def run_monitor():
            monitor = LiveMonitorGUI()
            monitor.run()
        
        monitor_thread = threading.Thread(target=run_monitor, daemon=False)
        monitor_thread.start()
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
