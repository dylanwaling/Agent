#!/usr/bin/env python3
"""
Comprehensive Testing and Debugging
Combined testing for document processing, search, Q&A functionality, and system checks
"""

import sys
import os
import tempfile
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from backend_logic import DocumentPipeline

def test_system_components():
    """Test system components and dependencies"""
    print("🔧 Testing System Components")
    print("=" * 50)
    
    # Test imports
    components = [
        ("Docling", "from docling.document_converter import DocumentConverter"),
        ("LangChain Text Splitter", "from langchain.text_splitter import RecursiveCharacterTextSplitter"),
        ("HuggingFace Embeddings", "from langchain_huggingface import HuggingFaceEmbeddings"),
        ("FAISS Vector Store", "from langchain_community.vectorstores import FAISS"),
        ("Ollama LLM", "from langchain_ollama import OllamaLLM"),
        ("LangChain Documents", "from langchain.schema import Document"),
        ("LangChain Prompts", "from langchain.prompts import PromptTemplate"),
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
    
    # Test Docling functionality
    print(f"\n🧪 Testing Docling functionality...")
    try:
        from docling.document_converter import DocumentConverter
        converter = DocumentConverter()
        
        # Create a simple test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("# Test Document\n\nThis is a test document for Docling.\n\n## Section\n\nSome content here.")
            test_file = f.name
        
        try:
            result = converter.convert(test_file)
            text = result.document.export_to_markdown()
            if "test document" in text.lower():
                print("✅ Docling processing test successful")
            else:
                print("⚠️ Docling processing returned unexpected result")
        finally:
            os.unlink(test_file)
            
    except Exception as e:
        print(f"❌ Docling test failed: {e}")
    
    # Test Ollama connection
    print(f"\n🦙 Testing Ollama connection...")
    try:
        from langchain_ollama import OllamaLLM
        
        models_to_test = ["llama3:latest", "tinyllama:latest"]
        ollama_working = False
        
        for model in models_to_test:
            try:
                llm = OllamaLLM(model=model)
                response = llm.invoke("Hello")
                if response:
                    print(f"✅ {model} - Working (response: {response[:30]}...)")
                    ollama_working = True
                    break
            except Exception as e:
                print(f"❌ {model} - {str(e)[:50]}...")
        
        if not ollama_working:
            print("⚠️ No Ollama models responding. Make sure Ollama is running.")
            
    except Exception as e:
        print(f"❌ Ollama connection test failed: {e}")
    
    # Test embeddings
    print(f"\n🔤 Testing embedding model...")
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectors = embeddings.embed_documents(["test sentence"])
        print(f"✅ Embeddings working ({len(vectors[0])} dimensions)")
    except Exception as e:
        print(f"❌ Embeddings test failed: {e}")
    
    # Test GPU availability
    print(f"\n🚀 Testing GPU support...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print(f"🔥 CUDA Version: {torch.version.cuda}")
        else:
            print("💻 CUDA not available - using CPU")
            
        # Test FAISS GPU
        import faiss
        if hasattr(faiss, 'StandardGpuResources'):
            print("✅ FAISS GPU support available")
        else:
            print("💻 FAISS CPU-only version installed")
            
    except ImportError as e:
        print(f"❌ GPU test failed: {e}")
        print("💻 Using CPU-only mode")
    
    return success_count >= len(components) - 1  # Allow one failure

def inspect_database_content():
    """Inspect the content of documents in the database"""
    print("\n🔍 Database Content Inspector")
    print("=" * 50)
    
    try:
        # Check for FAISS index
        index_path = Path("data/index/faiss_index.faiss")
        if index_path.exists():
            print("✅ Found FAISS index")
            
            try:
                from langchain_community.vectorstores import FAISS
                from langchain_huggingface import HuggingFaceEmbeddings
                
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                vectorstore = FAISS.load_local("data/index/faiss_index", embeddings, allow_dangerous_deserialization=True)
                
                print(f"📊 FAISS index loaded successfully")
                
                # Test search to see what's in there
                results = vectorstore.similarity_search("test", k=5)
                print(f"📚 Found {len(results)} documents in test search")
                
                # Show document sources
                sources = set()
                for doc in results:
                    source = doc.metadata.get('source', 'Unknown')
                    sources.add(source)
                
                print(f"📄 Unique documents in index:")
                for source in sorted(sources):
                    print(f"   • {source}")
                    
                return True
                
            except Exception as e:
                print(f"⚠️ FAISS inspection failed: {e}")
        else:
            print("❌ No FAISS index found")
        
        return False
            
    except Exception as e:
        print(f"❌ Database inspection error: {e}")
        return False

def test_initialization():
    """Test pipeline initialization"""
    print("🚀 Testing Pipeline Initialization")
    print("=" * 50)
    
    try:
        pipeline = DocumentPipeline()
        print("✅ Pipeline initialized successfully")
        
        # Check directory structure
        if pipeline.docs_dir.exists():
            doc_count = len(list(pipeline.docs_dir.iterdir()))
            print(f"📁 Documents directory: {doc_count} files found")
        else:
            print("⚠️ Documents directory not found")
            
        if pipeline.index_dir.exists():
            index_exists = (pipeline.index_dir / "faiss_index.faiss").exists()
            print(f"📊 Index directory: {'Index exists' if index_exists else 'No index found'}")
        else:
            print("⚠️ Index directory not found")
            
        return pipeline
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return None

def test_document_processing(pipeline):
    """Test document processing and indexing"""
    print("\n📄 Testing Document Processing")
    print("=" * 50)
    
    try:
        # Try to load existing index first
        if pipeline.load_index():
            print("✅ Existing index loaded successfully")
            return True
        else:
            print("⚠️ No existing index, processing documents...")
            if pipeline.process_documents():
                print("✅ Documents processed and indexed successfully")
                return True
            else:
                print("❌ Document processing failed")
                return False
                
    except Exception as e:
        print(f"❌ Document processing error: {e}")
        return False

def test_search_functionality(pipeline):
    """Test search with various queries"""
    print("\n🔍 Testing Search Functionality")
    print("=" * 50)
    
    # Test queries that cover different document types
    test_queries = [
        "Invoice_Outline_-_Sheet1_1.pdf",
        "invoice outline", 
        "product_manual",
        "product manual",
        "company handbook",
        "company_handbook.md",
        "algebra operations",
        "Math Review",
        "tmphp713yna_Math_Review_-_Algebra_Operations.pdf"
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: '{query}'")
        print("-" * 30)
        
        try:
            # Debug search
            debug_results = pipeline.debug_search(query)
            
            if "error" in debug_results:
                print(f"❌ Error: {debug_results['error']}")
                continue
            
            print(f"📊 Found {debug_results['total_results']} results")
            
            # Show top 3 results
            for result in debug_results["results"][:3]:
                print(f"  {result['rank']}. {result['source']} (score: {result['score']:.3f})")
                print(f"     Preview: {result['content_preview']}")
            
            # Test content extraction for first result
            if debug_results['total_results'] > 0:
                search_results = pipeline.search(query)
                if search_results:
                    content = search_results[0]["content"]
                    
                    # Test content cleaning
                    parts = content.split(' ', 2)
                    if len(parts) >= 3:
                        clean_content = parts[2]
                        print(f"  🧹 Clean content: {clean_content[:100]}...")
                    else:
                        print(f"  ⚠️ Content cleaning issue - parts: {len(parts)}")
            
            # Verify expected document is found
            expected_variations = [
                query.lower(),
                query.lower().replace(" ", "_"),
                query.lower().replace("-", "_"),
                query.lower().replace("_", " ")
            ]
            
            found_match = False
            if debug_results["results"]:
                found_source = debug_results["results"][0]["source"].lower()
                for variation in expected_variations:
                    if variation in found_source or found_source.replace("_", "").replace("-", "").replace(".", "") in variation.replace("_", "").replace("-", "").replace(".", ""):
                        found_match = True
                        break
            
            if found_match:
                print(f"  ✅ Expected document found")
            else:
                print(f"  ⚠️ Expected document not in top result")
                
        except Exception as e:
            print(f"❌ Search error for '{query}': {e}")

def test_qa_functionality(pipeline):
    """Test Q&A functionality"""
    print("\n🤖 Testing Q&A Functionality") 
    print("=" * 50)
    
    # Test different types of questions
    test_questions = [
        "What is in the Invoice_Outline_-_Sheet1_1.pdf?",
        "Tell me about the company handbook",
        "What does the product manual contain?",
        "What's in the Math Review document?",
        "Summarize the algebra operations document"
    ]
    
    for question in test_questions:
        print(f"\n❓ Question: '{question}'")
        print("-" * 50)
        
        try:
            result = pipeline.ask(question)
            
            print(f"🤖 Answer: {result['answer'][:300]}{'...' if len(result['answer']) > 300 else ''}")
            print(f"📚 Sources ({len(result['sources'])}):")
            
            for i, source in enumerate(result['sources'], 1):
                print(f"  {i}. {source['source']}")
                print(f"     Content: {source['content'][:150]}{'...' if len(source['content']) > 150 else ''}")
                
        except Exception as e:
            print(f"❌ Q&A error for '{question}': {e}")

def test_edge_cases(pipeline):
    """Test edge cases and error handling"""
    print("\n🔬 Testing Edge Cases")
    print("=" * 50)
    
    edge_cases = [
        "",  # Empty query
        "nonexistent document that definitely does not exist",  # Non-existent document
        "a b c d e f g h i j k l m n o p q r s t u v w x y z",  # Random letters
        "🚀🎯💡",  # Emojis only
        "What is the meaning of life, the universe, and everything?",  # Unrelated question
    ]
    
    for case in edge_cases:
        print(f"\n🧪 Edge case: '{case}'")
        print("-" * 30)
        
        try:
            # Test search
            search_results = pipeline.search(case)
            print(f"🔍 Search: {len(search_results)} results")
            
            # Test Q&A
            qa_result = pipeline.ask(case)
            print(f"🤖 Q&A: {'Success' if qa_result['answer'] else 'No answer'}")
            
        except Exception as e:
            print(f"❌ Edge case error: {e}")

def run_performance_test(pipeline):
    """Run basic performance test"""
    print("\n⚡ Performance Test")
    print("=" * 50)
    
    import time
    
    test_query = "company handbook"
    iterations = 5
    
    print(f"Running {iterations} iterations of search for '{test_query}'...")
    
    times = []
    for i in range(iterations):
        start_time = time.time()
        results = pipeline.search(test_query)
        end_time = time.time()
        
        duration = end_time - start_time
        times.append(duration)
        print(f"  Iteration {i+1}: {duration:.3f}s ({len(results)} results)")
    
    avg_time = sum(times) / len(times)
    print(f"📊 Average search time: {avg_time:.3f}s")

def main():
    """Run all tests"""
    print("🧪 COMPREHENSIVE DOCUMENT PIPELINE TESTING")
    print("=" * 60)
    
    # Test system components first
    if not test_system_components():
        print("⚠️ Some system components have issues, but continuing...")
    
    # Inspect database content
    inspect_database_content()
    
    # Initialize pipeline
    pipeline = test_initialization()
    if not pipeline:
        print("❌ Cannot continue without pipeline")
        return
    
    # Test document processing
    if not test_document_processing(pipeline):
        print("❌ Cannot continue without processed documents")
        return
    
    # Test search functionality
    test_search_functionality(pipeline)
    
    # Test Q&A functionality
    test_qa_functionality(pipeline)
    
    # Test edge cases
    test_edge_cases(pipeline)
    
    # Performance test
    run_performance_test(pipeline)
    
    print("\n🎉 All tests completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
