"""
Debug and testing utilities for Step 1: Document Extraction
Run this to test and debug the extraction process
"""
import os
import sys
from pathlib import Path
import time
from docling.document_converter import DocumentConverter
from utils.sitemap import get_sitemap_urls

def check_internet_connection():
    """Check if we have internet connectivity"""
    print("🌐 Checking Internet Connection")
    print("=" * 40)
    
    try:
        import requests
        response = requests.get("https://httpbin.org/status/200", timeout=5)
        if response.status_code == 200:
            print("✅ Internet connection: OK")
            return True
        else:
            print(f"⚠️ Internet connection: Status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Internet connection: Failed ({e})")
        return False

def check_docling_installation():
    """Test Docling installation and basic functionality"""
    print("\n🔧 Checking Docling Installation")
    print("=" * 40)
    
    try:
        from docling.document_converter import DocumentConverter
        print("✅ Docling imported successfully")
        
        converter = DocumentConverter()
        print("✅ DocumentConverter created")
        
        # Test with a simple HTML file (create temporary file)
        import tempfile
        simple_html = "<html><body><h1>Test Document</h1><p>This is a test paragraph with some content to verify HTML processing.</p></body></html>"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as temp_file:
            temp_file.write(simple_html)
            temp_path = temp_file.name
        
        print("🔄 Testing HTML processing...")
        start_time = time.time()
        result = converter.convert(temp_path)
        end_time = time.time()
        
        # Clean up temp file
        os.unlink(temp_path)
        
        if result.document:
            print(f"✅ Basic HTML processing works ({end_time-start_time:.2f}s)")
            markdown = result.document.export_to_markdown()
            print(f"📄 Extracted content: {markdown[:100]}...")
            return True
        else:
            print("❌ No document returned")
            return False
            
    except Exception as e:
        print(f"❌ Docling error: {e}")
        return False

def test_pdf_extraction():
    """Test PDF extraction with the ArXiv paper"""
    print("\n📄 Testing PDF Extraction")
    print("=" * 40)
    
    if not check_internet_connection():
        print("⚠️ Skipping PDF test - no internet connection")
        return False
    
    try:
        converter = DocumentConverter()
        pdf_url = "https://arxiv.org/pdf/2408.09869"
        
        print(f"🔄 Extracting PDF from: {pdf_url}")
        print("⏳ This may take 1-5 minutes on first run (downloading AI models)...")
        
        start_time = time.time()
        result = converter.convert(pdf_url)
        end_time = time.time()
        
        if result.document:
            markdown_output = result.document.export_to_markdown()
            print(f"✅ PDF extraction successful ({end_time-start_time:.2f}s)")
            print(f"📊 Content length: {len(markdown_output)} characters")
            print(f"📖 Preview: {markdown_output[:300]}...")
            
            # Check for key content
            if "docling" in markdown_output.lower():
                print("✅ Document contains expected 'docling' content")
            else:
                print("⚠️ Document may not have extracted correctly")
            
            return True
        else:
            print("❌ PDF extraction failed - no document returned")
            return False
            
    except Exception as e:
        print(f"❌ PDF extraction error: {e}")
        return False

def test_html_extraction():
    """Test HTML extraction from GitHub page"""
    print("\n🌐 Testing HTML Extraction")
    print("=" * 40)
    
    if not check_internet_connection():
        print("⚠️ Skipping HTML test - no internet connection")
        return False
    
    try:
        converter = DocumentConverter()
        html_url = "https://github.com/DS4SD/docling"
        
        print(f"🔄 Extracting HTML from: {html_url}")
        
        start_time = time.time()
        result = converter.convert(html_url)
        end_time = time.time()
        
        if result.document:
            markdown_output = result.document.export_to_markdown()
            print(f"✅ HTML extraction successful ({end_time-start_time:.2f}s)")
            print(f"📊 Content length: {len(markdown_output)} characters")
            print(f"📖 Preview: {markdown_output[:300]}...")
            
            # Check for key content
            if "docling" in markdown_output.lower():
                print("✅ Document contains expected 'docling' content")
            else:
                print("⚠️ Document may not have extracted correctly")
            
            return True
        else:
            print("❌ HTML extraction failed - no document returned")
            return False
            
    except Exception as e:
        print(f"❌ HTML extraction error: {e}")
        return False

def test_sitemap_functionality():
    """Test sitemap extraction"""
    print("\n🗺️ Testing Sitemap Functionality")
    print("=" * 40)
    
    if not check_internet_connection():
        print("⚠️ Skipping sitemap test - no internet connection")
        return False
    
    try:
        from utils.sitemap import get_sitemap_urls
        
        test_url = "https://ds4sd.github.io/docling/"
        print(f"🔄 Testing sitemap for: {test_url}")
        
        start_time = time.time()
        urls = get_sitemap_urls(test_url)
        end_time = time.time()
        
        print(f"✅ Sitemap extraction successful ({end_time-start_time:.2f}s)")
        print(f"📊 Found {len(urls)} URLs")
        
        # Show first few URLs
        for i, url in enumerate(urls[:5]):
            print(f"  {i+1}. {url}")
        
        if len(urls) > 5:
            print(f"  ... and {len(urls)-5} more URLs")
        
        return True
        
    except Exception as e:
        print(f"❌ Sitemap error: {e}")
        return False

def check_model_cache():
    """Check if Docling AI models are cached"""
    print("\n🤖 Checking Model Cache")
    print("=" * 40)
    
    # Common cache locations
    cache_paths = [
        Path.home() / ".cache" / "docling",
        Path.home() / ".cache" / "huggingface",
        Path.home() / "AppData" / "Local" / "docling" if os.name == 'nt' else None,
    ]
    
    total_model_files = 0
    total_size_mb = 0
    
    for cache_path in cache_paths:
        if cache_path and cache_path.exists():
            print(f"✅ Found cache directory: {cache_path}")
            
            try:
                contents = list(cache_path.rglob("*"))
                model_files = [f for f in contents if f.suffix in ['.bin', '.onnx', '.pt', '.safetensors', '.pkl']]
                
                if model_files:
                    print(f"   🤖 Found {len(model_files)} model files")
                    for model_file in model_files[:3]:  # Show first 3
                        size_mb = model_file.stat().st_size / (1024*1024)
                        total_size_mb += size_mb
                        print(f"      - {model_file.name} ({size_mb:.1f}MB)")
                    
                    if len(model_files) > 3:
                        remaining_size = sum(f.stat().st_size for f in model_files[3:]) / (1024*1024)
                        total_size_mb += remaining_size
                        print(f"      ... and {len(model_files)-3} more ({remaining_size:.1f}MB)")
                    
                    total_model_files += len(model_files)
                else:
                    print(f"   📂 No model files found")
                    
            except Exception as e:
                print(f"   ❌ Error reading cache: {e}")
        else:
            print(f"❌ Cache directory not found: {cache_path}")
    
    if total_model_files > 0:
        print(f"\n📊 Total: {total_model_files} model files ({total_size_mb:.1f}MB)")
        if total_size_mb > 100:
            print("✅ Sufficient models cached - extractions should be fast")
        else:
            print("⚠️ Limited models cached - first extraction may be slow")
    else:
        print("\n⚠️ No models cached - first extraction will download models")

def run_full_extraction_test():
    """Run the complete extraction pipeline from 1-extraction.py"""
    print("\n🚀 Running Full Extraction Test")
    print("=" * 40)
    
    try:
        # This mirrors the logic in 1-extraction.py
        converter = DocumentConverter()
        
        print("🔄 Step 1: PDF Extraction...")
        pdf_result = converter.convert("https://arxiv.org/pdf/2408.09869")
        pdf_markdown = pdf_result.document.export_to_markdown()
        print(f"✅ PDF: {len(pdf_markdown)} characters extracted")
        
        print("🔄 Step 2: HTML Extraction...")
        html_result = converter.convert("https://github.com/DS4SD/docling")
        html_markdown = html_result.document.export_to_markdown()
        print(f"✅ HTML: {len(html_markdown)} characters extracted")
        
        print("✅ Full extraction test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Full extraction test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Step 1: Document Extraction Debug")
    print("=" * 50)
    
    # Run all tests
    check_model_cache()
    
    if check_docling_installation():
        test_pdf_extraction()
        test_html_extraction()
        test_sitemap_functionality()
        
        # Final comprehensive test
        print("\n" + "="*50)
        run_full_extraction_test()
    
    print("\n✅ Extraction debug completed!")
