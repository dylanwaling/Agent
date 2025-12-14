#!/usr/bin/env python3
"""
Comprehensive Testing and Debugging
Combined testing for document processing, search, Q&A functionality, and system checks
"""

# ============================================================================
# CONSTANTS & IMPORTS
# ============================================================================

# Standard library imports
import sys
import os
import tempfile
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# Third-party imports
import psutil

# Local imports
sys.path.append(str(Path(__file__).parent))
from logic.rag_pipeline_orchestrator import DocumentPipeline


# ============================================================================
# DATA MODELS & CLASSES
# ============================================================================

@dataclass
class TimingMetric:
    """Performance timing for a specific operation"""
    operation: str
    start_time: float
    end_time: float
    duration: float
    memory_start: float
    memory_end: float
    memory_delta: float
    success: bool
    error: Optional[str] = None
    metadata: Optional[Dict] = None

@dataclass
class SectionProfile:
    """Profile for a pipeline section"""
    section_name: str
    total_time: float
    call_count: int
    avg_time: float
    min_time: float
    max_time: float
    total_memory: float
    avg_memory: float
    success_rate: float
    errors: List[str]

class PerformanceMonitor:
    """Real-time performance monitoring for pipeline operations"""
    
    def __init__(self):
        self.metrics: List[TimingMetric] = []
        self.process = psutil.Process()
        self.start_memory = self.get_memory_mb()
        self.session_start = time.time()
        
    def get_memory_mb(self) -> float:
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def start_operation(self, operation: str) -> Dict[str, Any]:
        """Start timing an operation"""
        return {
            'operation': operation,
            'start_time': time.time(),
            'memory_start': self.get_memory_mb()
        }
    
    def end_operation(self, context: Dict[str, Any], success: bool = True, 
                     error: Optional[str] = None, metadata: Optional[Dict] = None) -> TimingMetric:
        """End timing an operation and record metrics"""
        end_time = time.time()
        memory_end = self.get_memory_mb()
        
        metric = TimingMetric(
            operation=context['operation'],
            start_time=context['start_time'],
            end_time=end_time,
            duration=end_time - context['start_time'],
            memory_start=context['memory_start'],
            memory_end=memory_end,
            memory_delta=memory_end - context['memory_start'],
            success=success,
            error=error,
            metadata=metadata or {}
        )
        
        self.metrics.append(metric)
        return metric
    
    def get_section_profile(self, section_name: str) -> SectionProfile:
        """Get aggregate profile for a specific section"""
        section_metrics = [m for m in self.metrics if m.operation == section_name]
        
        if not section_metrics:
            return SectionProfile(
                section_name=section_name,
                total_time=0, call_count=0, avg_time=0,
                min_time=0, max_time=0, total_memory=0,
                avg_memory=0, success_rate=0, errors=[]
            )
        
        durations = [m.duration for m in section_metrics]
        memories = [m.memory_delta for m in section_metrics]
        successes = [m.success for m in section_metrics]
        errors = [m.error for m in section_metrics if m.error]
        
        return SectionProfile(
            section_name=section_name,
            total_time=sum(durations),
            call_count=len(section_metrics),
            avg_time=sum(durations) / len(durations),
            min_time=min(durations),
            max_time=max(durations),
            total_memory=sum(memories),
            avg_memory=sum(memories) / len(memories),
            success_rate=sum(successes) / len(successes) * 100,
            errors=errors
        )
    
    def get_all_profiles(self) -> Dict[str, SectionProfile]:
        """Get profiles for all sections"""
        operations = set(m.operation for m in self.metrics)
        return {op: self.get_section_profile(op) for op in operations}
    
    def print_summary(self):
        """Print performance summary"""
        total_time = time.time() - self.session_start
        total_memory = self.get_memory_mb() - self.start_memory
        
        print("\n" + "=" * 70)
        print("📊 PERFORMANCE SUMMARY")
        print("=" * 70)
        print(f"⏱️  Total Session Time: {total_time:.2f}s")
        print(f"💾 Total Memory Delta: {total_memory:+.2f} MB")
        print(f"📈 Total Operations: {len(self.metrics)}")
        if self.metrics:
            print(f"🎯 Success Rate: {sum(m.success for m in self.metrics) / len(self.metrics) * 100:.1f}%")
        
        profiles = self.get_all_profiles()
        
        print("\n📍 Section Breakdown:")
        print("-" * 70)
        
        for section_name, profile in sorted(profiles.items(), key=lambda x: x[1].total_time, reverse=True):
            print(f"\n🔹 {section_name}")
            print(f"   Time: {profile.total_time:.3f}s total | {profile.avg_time:.3f}s avg | {profile.min_time:.3f}s min | {profile.max_time:.3f}s max")
            print(f"   Calls: {profile.call_count}")
            print(f"   Memory: {profile.avg_memory:+.2f} MB avg")
            print(f"   Success: {profile.success_rate:.1f}%")
            if profile.errors:
                print(f"   ⚠️  Errors: {len(profile.errors)}")
    
    def export_json(self, filepath: str):
        """Export metrics to JSON file"""
        data = {
            'session_start': self.session_start,
            'session_duration': time.time() - self.session_start,
            'start_memory_mb': self.start_memory,
            'end_memory_mb': self.get_memory_mb(),
            'metrics': [asdict(m) for m in self.metrics],
            'profiles': {name: asdict(profile) for name, profile in self.get_all_profiles().items()}
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"📊 Metrics exported to: {filepath}")


# ============================================================================
# CORE LOGIC - SYSTEM TESTING
# ============================================================================

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

# ============================================================================
# CORE LOGIC - DATABASE INSPECTION
# ============================================================================


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
        

# ============================================================================
# CORE LOGIC - PIPELINE TESTING
# ============================================================================

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
    
    # Get actual documents from the system
    docs_dir = Path("data/documents")
    actual_docs = []
    if docs_dir.exists():
        actual_docs = [f.name for f in docs_dir.iterdir() if f.is_file()]
    
    if not actual_docs:
        print("⚠️ No documents found in data/documents/")
        return
    
    print(f"📁 Found {len(actual_docs)} document(s): {', '.join(actual_docs)}")
    print()
    
    # Build test queries based on actual documents
    test_queries = []
    
    # Add exact filenames
    for doc in actual_docs:
        test_queries.append(doc)
        
        # Add filename without extension
        name_without_ext = Path(doc).stem
        test_queries.append(name_without_ext)
        
        # Add filename with spaces instead of underscores
        name_with_spaces = name_without_ext.replace("_", " ").replace("-", " ")
        if name_with_spaces != name_without_ext:
            test_queries.append(name_with_spaces)
    
    # Add some generic queries
    test_queries.extend(["document", "information", "content"])
    
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
            
            # Check if query matches any actual document
            query_lower = query.lower()
            is_document_query = any(
                query_lower == doc.lower() or 
                query_lower == Path(doc).stem.lower() or
                query_lower.replace(" ", "_") in doc.lower() or
                query_lower.replace(" ", "-") in doc.lower()
                for doc in actual_docs
            )
            
            if is_document_query and debug_results["results"]:
                # Check if the top result matches the query
                top_source = debug_results["results"][0]["source"].lower()
                found_match = False
                
                for doc in actual_docs:
                    if query_lower in doc.lower() or Path(doc).stem.lower() in query_lower:
                        if doc.lower() == top_source:
                            found_match = True
                            break
                
                if found_match:
                    print(f"  ✅ Correct document found as top result")
                else:
                    print(f"  ⚠️ Expected document not in top result")
            else:
                print(f"  ℹ️  Generic query - showing most relevant results")
                
        except Exception as e:
            print(f"❌ Search error for '{query}': {e}")

def test_qa_functionality(pipeline):
    """Test Q&A functionality"""
    print("\n🤖 Testing Q&A Functionality") 
    print("=" * 50)
    
    # Get actual documents from the system
    docs_dir = Path("data/documents")
    actual_docs = []
    if docs_dir.exists():
        actual_docs = [f.name for f in docs_dir.iterdir() if f.is_file()]
    
    if not actual_docs:
        print("⚠️ No documents found in data/documents/")
        return
    
    # Build test questions based on actual documents
    test_questions = []
    
    # Add specific questions about each document
    for doc in actual_docs[:3]:  # Test first 3 documents
        name = Path(doc).stem.replace("_", " ").replace("-", " ")
        test_questions.append(f"What is in {doc}?")
        test_questions.append(f"Tell me about {name}")
    
    # Add generic questions
    test_questions.extend([
        "What documents are available?",
        "Summarize the main content"
    ])
    
    # Limit to 5 questions to avoid excessive testing
    test_questions = test_questions[:5]
    
    for question in test_questions:
        print(f"\n❓ Question: '{question}'")
        print("-" * 50)
        
        try:
            result = pipeline.ask(question)
            

# ============================================================================
# CORE LOGIC - EDGE CASE TESTING
# ============================================================================

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

# ============================================================================
# ANALYTICS / REPORTING - PERFORMANCE TESTING
# ============================================================================

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
    

# ============================================================================
# ENTRY POINT
# ============================================================================

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
    """Run all tests with performance monitoring"""
    print("🧪 COMPREHENSIVE DOCUMENT PIPELINE TESTING & ANALYSIS")
    print("=" * 60)
    
    # Initialize performance monitor
    monitor = PerformanceMonitor()
    
    try:
        # Test system components first
        ctx = monitor.start_operation("test_system_components")
        success = test_system_components()
        monitor.end_operation(ctx, success=success)
        if not success:
            print("⚠️ Some system components have issues, but continuing...")
        
        # Inspect database content
        ctx = monitor.start_operation("inspect_database_content")
        inspect_database_content()
        monitor.end_operation(ctx, success=True)
        
        # Initialize pipeline
        ctx = monitor.start_operation("test_initialization")
        pipeline = test_initialization()
        monitor.end_operation(ctx, success=pipeline is not None)
        if not pipeline:
            print("❌ Cannot continue without pipeline")
            return
        
        # Test document processing
        ctx = monitor.start_operation("test_document_processing")
        processing_success = test_document_processing(pipeline)
        monitor.end_operation(ctx, success=processing_success)
        if not processing_success:
            print("❌ Cannot continue without processed documents")
            return
        
        # Test search functionality
        ctx = monitor.start_operation("test_search_functionality")
        test_search_functionality(pipeline)
        monitor.end_operation(ctx, success=True)
        
        # Test Q&A functionality
        ctx = monitor.start_operation("test_qa_functionality")
        test_qa_functionality(pipeline)
        monitor.end_operation(ctx, success=True)
        
        # Test edge cases
        ctx = monitor.start_operation("test_edge_cases")
        test_edge_cases(pipeline)
        monitor.end_operation(ctx, success=True)
        
        # Performance test
        ctx = monitor.start_operation("run_performance_test")
        run_performance_test(pipeline)

        monitor.end_operation(ctx, success=True)
        
        print("\n🎉 All tests completed!")
        print("=" * 60)
        
        # Print performance summary
        monitor.print_summary()
        
        # Export metrics
        monitor.export_json("data/test_performance_metrics.json")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Testing interrupted by user")
        monitor.print_summary()
    except Exception as e:
        print(f"\n❌ Fatal error during testing: {e}")
        import traceback
        traceback.print_exc()
        monitor.print_summary()

if __name__ == "__main__":
    main()
