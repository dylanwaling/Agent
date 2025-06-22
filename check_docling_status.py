#!/usr/bin/env python3
"""
Check if Docling models are downloaded and processing is working
"""

import os
import sys
from pathlib import Path

def check_docling_cache():
    """Check if Docling models are cached"""
    print("🔍 Checking Docling model cache...")
    
    # Common cache locations
    cache_paths = [
        Path.home() / ".cache" / "docling",
        Path.home() / ".cache" / "huggingface",
        Path.home() / "AppData/Local/docling" if os.name == 'nt' else None,
    ]
    
    for cache_path in cache_paths:
        if cache_path and cache_path.exists():
            print(f"✅ Found cache directory: {cache_path}")
            
            # List contents
            try:
                contents = list(cache_path.rglob("*"))
                if contents:
                    print(f"   📁 Contains {len(contents)} files/folders")
                    
                    # Look for model files
                    model_files = [f for f in contents if f.suffix in ['.bin', '.onnx', '.pt', '.safetensors']]
                    if model_files:
                        print(f"   🤖 Found {len(model_files)} model files")
                        for model_file in model_files[:3]:  # Show first 3
                            size_mb = model_file.stat().st_size / (1024*1024)
                            print(f"      - {model_file.name} ({size_mb:.1f}MB)")
                        if len(model_files) > 3:
                            print(f"      ... and {len(model_files)-3} more")
                else:
                    print(f"   📂 Directory is empty")
            except Exception as e:
                print(f"   ❌ Error reading cache: {e}")
        else:
            print(f"❌ Cache directory not found: {cache_path}")

def test_simple_docling():
    """Test basic Docling functionality"""
    print("\n🧪 Testing basic Docling functionality...")
    
    try:
        from docling.document_converter import DocumentConverter
        print("✅ Docling imported successfully")
        
        converter = DocumentConverter()
        print("✅ DocumentConverter created")
        
        # Test with a simple HTML string using temp file method
        import tempfile
        import os
        
        simple_html = "<html><body><h1>Test</h1><p>This is a test document.</p></body></html>"
        
        print("🔄 Testing HTML processing...")
        
        # Create a temporary HTML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as temp_file:
            temp_file.write(simple_html)
            temp_file_path = temp_file.name
        
        try:
            # Use the correct convert() method
            result = converter.convert(temp_file_path)
            
            if result.document:
                print("✅ Basic HTML processing works")
                markdown = result.document.export_to_markdown()
                print(f"📄 Extracted content preview: {markdown[:100]}...")
                return True
            else:
                print("❌ No document returned")
                return False
        finally:
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("🔍 Docling Status Checker")
    print("=" * 50)
    
    check_docling_cache()
    
    if test_simple_docling():
        print("\n✅ Docling is working! The 1-extraction.py script should complete successfully.")
        print("💡 PDF processing takes longer because it uses AI models for layout analysis.")
    else:
        print("\n❌ Docling has issues. Check the error messages above.")

if __name__ == "__main__":
    main()
