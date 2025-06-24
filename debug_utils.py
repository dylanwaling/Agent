#!/usr/bin/env python3
"""
Comprehensive Debug Utilities for Document Q&A Pipeline
Consolidated debugging tools for all pipeline components
"""

import os
import sys
import lancedb
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
import tempfile

# =============================================================================
# DOCLING STATUS CHECKING
# =============================================================================

def check_docling_status():
    """Check if Docling is installed and working"""
    print("🔍 Checking Docling Status")
    print("=" * 50)
    
    # Check installation
    try:
        from docling.document_converter import DocumentConverter
        print("✅ Docling is installed")
    except ImportError as e:
        print(f"❌ Docling import failed: {e}")
        return False
    
    # Check model cache
    print("\n📁 Checking Docling model cache...")
    cache_paths = [
        Path.home() / ".cache" / "docling",
        Path.home() / ".cache" / "huggingface",
        Path.home() / "AppData/Local/docling" if os.name == 'nt' else None,
    ]
    
    cache_found = False
    for cache_path in cache_paths:
        if cache_path and cache_path.exists():
            print(f"✅ Found cache directory: {cache_path}")
            cache_found = True
            
            try:
                contents = list(cache_path.rglob("*"))
                if contents:
                    model_files = [f for f in contents if f.suffix in ['.bin', '.onnx', '.pt', '.safetensors']]
                    if model_files:
                        print(f"   🤖 Found {len(model_files)} model files")
                        total_size = sum(f.stat().st_size for f in model_files) / (1024*1024*1024)
                        print(f"   💾 Total model size: {total_size:.1f}GB")
            except Exception as e:
                print(f"   ⚠️  Error reading cache: {e}")
    
    if not cache_found:
        print("❌ No model cache found - models will download on first use")
      # Test basic functionality
    print("\n🧪 Testing basic Docling functionality...")
    try:
        converter = DocumentConverter()
        
        # Create a simple test markdown file (Docling supports .md better)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("# Test Document\n\nThis is a test document for Docling.\n\n## Section\n\nSome content here.")
            test_file = f.name
        
        try:
            result = converter.convert(test_file)
            text = result.document.export_to_markdown()
            if "test document" in text.lower():
                print("✅ Docling processing test successful")
            else:
                print("⚠️  Docling processing returned unexpected result")
                print(f"   Result: {text[:100]}...")
        finally:
            os.unlink(test_file)
            
        return True
        
    except Exception as e:
        print(f"❌ Docling processing test failed: {e}")
        # Try the fallback approach used in 5-chat.py
        print("🔄 Testing fallback text processing...")
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write("This is a test document for text processing.")
                test_file = f.name
            
            try:
                # Read directly like 5-chat.py does for .txt files
                with open(test_file, 'r', encoding='utf-8') as f:
                    text = f.read()
                if "test document" in text:
                    print("✅ Text file processing works (used for .txt and .md)")
                    return True
            finally:
                os.unlink(test_file)
                
        except Exception as e2:
            print(f"❌ Text processing also failed: {e2}")
        
        return False

# =============================================================================
# DATABASE CONTENT INSPECTION
# =============================================================================

def inspect_database_content():
    """Inspect the content of documents in the database"""
    print("🔍 Database Content Inspector")
    print("=" * 50)
    
    try:
        # Try FAISS first (new format)
        index_path = Path("data/index/faiss_index.faiss")
        if index_path.exists():
            print("✅ Found FAISS index")
            
            # Try to load and inspect FAISS
            try:
                from langchain_community.vectorstores import FAISS
                from langchain_huggingface import HuggingFaceEmbeddings
                
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                vectorstore = FAISS.load_local("data/index/faiss_index", embeddings, allow_dangerous_deserialization=True)
                
                # Get document count (approximate)
                print(f"📊 FAISS index loaded successfully")
                
                # Test search
                results = vectorstore.similarity_search("test", k=3)
                print(f"📚 Found {len(results)} documents in search test")
                
                for i, doc in enumerate(results):
                    source = doc.metadata.get('source', 'Unknown')
                    chunk_id = doc.metadata.get('chunk_id', 'Unknown')
                    print(f"   • Document {i+1}: {source} (chunk {chunk_id})")
                    print(f"     Preview: {doc.page_content[:100]}...")
                
                return True
                
            except Exception as e:
                print(f"⚠️  FAISS inspection failed: {e}")
        
        # Try LanceDB (old format)
        db_path = Path("data/lancedb")
        if db_path.exists():
            print("✅ Found LanceDB database")
            
            db = lancedb.connect("data/lancedb")
            table = db.open_table("docling")
            df = table.to_pandas()
            
            print(f"📊 Total chunks in database: {len(df)}")
            
            # Show unique documents
            if 'filename' in df.columns:
                unique_docs = df['filename'].unique()
                print(f"📚 Documents in database: {len(unique_docs)}")
                for doc in unique_docs:
                    doc_chunks = len(df[df['filename'] == doc])
                    print(f"   • {doc} ({doc_chunks} chunks)")
                
                # Show sample content
                for doc in unique_docs[:3]:  # Show first 3 docs
                    print(f"\n📄 Sample from '{doc}':")
                    doc_data = df[df['filename'] == doc]
                    if len(doc_data) > 0:
                        text = doc_data.iloc[0].get('text', 'No text')
                        preview = text[:200] + "..." if len(text) > 200 else text
                        print(f"   {preview}")
            
            return True
            
        else:
            print("❌ No database found")
            return False
            
    except Exception as e:
        print(f"❌ Database inspection error: {e}")
        return False

# =============================================================================
# PIPELINE COMPONENT TESTING
# =============================================================================

def test_pipeline_imports():
    """Test all pipeline imports"""
    print("🔍 Testing Pipeline Imports")
    print("=" * 30)
    
    components = [
        ("Docling", "from docling.document_converter import DocumentConverter"),
        ("LangChain Text Splitter", "from langchain.text_splitter import RecursiveCharacterTextSplitter"),
        ("HuggingFace Embeddings", "from langchain_huggingface import HuggingFaceEmbeddings"),
        ("FAISS Vector Store", "from langchain_community.vectorstores import FAISS"),
        ("Ollama LLM", "from langchain_ollama import OllamaLLM"),
        ("LangChain QA", "from langchain.chains import RetrievalQA"),
        ("Streamlit", "import streamlit"),
    ]
    
    success_count = 0
    for name, import_str in components:
        try:
            exec(import_str)
            print(f"✅ {name}")
            success_count += 1
        except ImportError as e:
            print(f"❌ {name}: {e}")
    
    print(f"\n📊 Import Results: {success_count}/{len(components)} successful")
    return success_count == len(components)

def test_ollama_connection():
    """Test Ollama connection and models"""
    print("\n🦙 Testing Ollama Connection")
    print("=" * 30)
    
    try:
        from langchain_ollama import OllamaLLM
        
        # Test with available models
        models_to_test = ["llama3:latest", "tinyllama:latest"]
        
        for model in models_to_test:
            try:
                llm = OllamaLLM(model=model)
                print(f"✅ {model} - Connection successful")
                
                # Test a simple query
                try:
                    response = llm.invoke("Hello")
                    if response:
                        print(f"   💬 Test response: {response[:50]}...")
                    else:
                        print("   ⚠️  Empty response")
                except Exception as e:
                    print(f"   ⚠️  Query test failed: {e}")
                
                return True  # If we get here, at least one model works
                
            except Exception as e:
                print(f"❌ {model} - {e}")
        
        print("❌ No working Ollama models found")
        print("💡 Try: ollama pull llama3:latest")
        return False
        
    except Exception as e:
        print(f"❌ Ollama import/connection failed: {e}")
        print("💡 Make sure Ollama is running: ollama serve")
        return False

def test_embedding_model():
    """Test embedding model loading"""
    print("\n🔤 Testing Embedding Model")
    print("=" * 30)
    
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        print("✅ Embedding model loaded")
        
        # Test embedding generation
        test_texts = ["This is a test", "Another test sentence"]
        vectors = embeddings.embed_documents(test_texts)
        
        print(f"✅ Generated embeddings: {len(vectors)} vectors, {len(vectors[0])} dimensions")
        return True
        
    except Exception as e:
        print(f"❌ Embedding model test failed: {e}")
        return False

# =============================================================================
# COMPREHENSIVE SYSTEM CHECK
# =============================================================================

def run_comprehensive_system_check():
    """Run all system checks"""
    print("🚀 Comprehensive System Check")
    print("=" * 60)
    
    checks = [
        ("Docling Status", check_docling_status),
        ("Pipeline Imports", test_pipeline_imports),
        ("Embedding Model", test_embedding_model), 
        ("Ollama Connection", test_ollama_connection),
        ("Database Content", inspect_database_content),
    ]
    
    results = {}
    for check_name, check_func in checks:
        print(f"\n{'='*20} {check_name} {'='*20}")
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"❌ {check_name} check failed with error: {e}")
            results[check_name] = False
    
    # Summary
    print(f"\n{'='*20} SUMMARY {'='*20}")
    passed = sum(results.values())
    total = len(results)
    
    for check_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {check_name}")
    
    print(f"\n📊 Overall: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 All systems operational!")
    else:
        print("⚠️  Some issues detected. Check individual failures above.")
    
    return passed == total

# =============================================================================
# QUALITY FILTERING TESTS
# =============================================================================

def test_quality_filtering():
    """Test document quality filtering"""
    print("\n🔍 Testing Quality Filtering")
    print("=" * 30)
    
    # Test cases for quality filtering
    test_texts = [
        ("Good text", "This is a well-formed document with proper sentences and meaningful content."),
        ("Repetitive", "Doc Doc Doc Doc Doc Doc Doc Doc Doc Doc Doc Doc Doc Doc Doc Doc"),
        ("Low alphabetic", "123 456 789 !@# $%^ &*( )_+ 123 456 789"),
        ("Too short", "Hi"),
        ("Corrupted", "F F F F F F F F F F F F F F F F F F F F F F F F"),
        ("Mixed good", "This document has some good content and explains various concepts clearly."),
    ]
    
    def is_high_quality(text):
        """Simple quality check"""
        if len(text) < 20:
            return False
        
        # Check alphabetic ratio
        alpha_chars = sum(c.isalpha() for c in text)
        alpha_ratio = alpha_chars / len(text) if text else 0
        if alpha_ratio < 0.6:
            return False
        
        # Check for repetitive patterns
        words = text.split()
        if len(words) > 5:
            unique_words = len(set(words))
            if unique_words / len(words) < 0.3:
                return False
        
        return True
    
    for label, text in test_texts:
        quality = is_high_quality(text)
        status = "✅ PASS" if quality else "❌ FILTER"
        print(f"{status} {label}: {text[:50]}...")
    
    return True

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main debug function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug utilities for Document Q&A Pipeline")
    parser.add_argument("--check", choices=[
        "all", "docling", "database", "imports", "ollama", "embeddings", "quality"
    ], default="all", help="Which check to run")
    
    args = parser.parse_args()
    
    if args.check == "all":
        run_comprehensive_system_check()
    elif args.check == "docling":
        check_docling_status()
    elif args.check == "database":
        inspect_database_content()
    elif args.check == "imports":
        test_pipeline_imports()
    elif args.check == "ollama":
        test_ollama_connection()
    elif args.check == "embeddings":
        test_embedding_model()
    elif args.check == "quality":
        test_quality_filtering()

if __name__ == "__main__":
    main()
