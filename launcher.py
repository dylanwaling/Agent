#!/usr/bin/env python3
"""
Document Q&A Agent - Main Entry Point
Professional launcher script for the Document Q&A application
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Launch the Document Q&A application"""
    print("╔══════════════════════════════════════════════════════════╗")
    print("║       Document Q&A Agent - Professional Edition         ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
    print("Available Commands:")
    print("  1. Run Application (GUI + Monitoring)")
    print("  2. Run System Tests")
    print("  3. Exit")
    print()
    
    while True:
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == "1":
            print("\n🚀 Running Application...")
            subprocess.run([sys.executable, "-m", "run"])
            break
        elif choice == "2":
            print("\n🔧 Running System Tests...")
            subprocess.run([sys.executable, "-m", "test"])
            break
        elif choice == "3":
            print("\n👋 Goodbye!")
            sys.exit(0)
        else:
            print("❌ Invalid choice. Please enter 1-3.")

if __name__ == "__main__":
    main()
