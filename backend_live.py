#!/usr/bin/env python3
"""
Live Backend Analysis & Performance Monitoring
Real-time profiling and optimization analysis for Document Q&A Pipeline

This tool provides:
- Section-by-section timing analysis
- Memory usage tracking
- Component performance benchmarking
- Bottleneck identification
- Optimization recommendations
"""

import sys
import os
import time
import psutil
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import traceback

sys.path.append(str(Path(__file__).parent))

from backend_logic import DocumentPipeline

# ============================================================================
# DATA STRUCTURES
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

# ============================================================================
# PERFORMANCE MONITOR
# ============================================================================

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
    
    def get_cpu_percent(self) -> float:
        """Get current CPU usage percentage"""
        return self.process.cpu_percent(interval=0.1)
    
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
# PIPELINE ANALYZER
# ============================================================================

class PipelineAnalyzer:
    """Comprehensive analysis of DocumentPipeline performance"""
    
    def __init__(self, pipeline: Optional[DocumentPipeline] = None):
        self.pipeline = pipeline or DocumentPipeline()
        self.monitor = PerformanceMonitor()
    
    # ========================================================================
    # SECTION 1: INITIALIZATION ANALYSIS
    # ========================================================================
    
    def analyze_initialization(self) -> Dict[str, Any]:
        """Analyze pipeline initialization performance"""
        print("\n" + "=" * 70)
        print("🚀 SECTION 1: INITIALIZATION ANALYSIS")
        print("=" * 70)
        
        results = {}
        
        # GPU Detection
        ctx = self.monitor.start_operation("initialization_gpu_detection")
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            if gpu_available:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                results['gpu'] = {'available': True, 'name': gpu_name, 'memory_gb': gpu_memory}
            else:
                results['gpu'] = {'available': False}
            
            metric = self.monitor.end_operation(ctx, success=True, metadata=results['gpu'])
            print(f"✅ GPU Detection: {metric.duration:.3f}s | {results['gpu']}")
        except Exception as e:
            metric = self.monitor.end_operation(ctx, success=False, error=str(e))
            print(f"❌ GPU Detection Failed: {e}")
            results['gpu'] = {'available': False, 'error': str(e)}
        
        # Component Initialization
        components = [
            ('docling_converter', 'Docling Converter'),
            ('text_splitter', 'Text Splitter'),
            ('embeddings', 'Embeddings Model'),
            ('llm', 'LLM (Ollama)'),
            ('prompt_template', 'Prompt Template')
        ]
        
        results['components'] = {}
        
        for comp_attr, comp_name in components:
            ctx = self.monitor.start_operation(f"initialization_{comp_attr}")
            try:
                if hasattr(self.pipeline, comp_attr):
                    component = getattr(self.pipeline, comp_attr)
                    metric = self.monitor.end_operation(ctx, success=True)
                    print(f"✅ {comp_name}: {metric.duration:.3f}s | Memory: {metric.memory_delta:+.2f} MB")
                    results['components'][comp_attr] = {'status': 'ok', 'time': metric.duration}
                else:
                    metric = self.monitor.end_operation(ctx, success=False, error="Not found")
                    print(f"⚠️  {comp_name}: Not found")
                    results['components'][comp_attr] = {'status': 'missing'}
            except Exception as e:
                metric = self.monitor.end_operation(ctx, success=False, error=str(e))
                print(f"❌ {comp_name}: {e}")
                results['components'][comp_attr] = {'status': 'error', 'error': str(e)}
        
        return results
    
    # ========================================================================
    # SECTION 2: DOCUMENT PROCESSING ANALYSIS
    # ========================================================================
    
    def analyze_document_processing(self, test_file: Optional[Path] = None) -> Dict[str, Any]:
        """Analyze document processing pipeline"""
        print("\n" + "=" * 70)
        print("📄 SECTION 2: DOCUMENT PROCESSING ANALYSIS")
        print("=" * 70)
        
        results = {}
        
        # Check for existing documents
        docs_dir = Path("data/documents")
        if docs_dir.exists():
            doc_files = [f for f in docs_dir.iterdir() if f.is_file()]
            results['document_count'] = len(doc_files)
            print(f"📁 Found {len(doc_files)} documents in data/documents/")
            
            if test_file is None and doc_files:
                test_file = doc_files[0]
        else:
            results['document_count'] = 0
            print("⚠️  No documents directory found")
            return results
        
        if test_file is None or not test_file.exists():
            print("⚠️  No test file available for processing analysis")
            return results
        
        print(f"📝 Analyzing: {test_file.name}")
        
        # Step 1: File Reading
        ctx = self.monitor.start_operation("processing_file_read")
        try:
            file_size = test_file.stat().st_size / 1024  # KB
            
            if test_file.suffix in ['.txt', '.md']:
                content = test_file.read_text(encoding='utf-8')
            else:
                # For binary files, just note the size
                content = None
            
            metric = self.monitor.end_operation(ctx, success=True, 
                                               metadata={'file_size_kb': file_size})
            print(f"✅ File Read: {metric.duration:.3f}s | Size: {file_size:.2f} KB")
            results['file_read'] = {'time': metric.duration, 'size_kb': file_size}
        except Exception as e:
            metric = self.monitor.end_operation(ctx, success=False, error=str(e))
            print(f"❌ File Read Failed: {e}")
            results['file_read'] = {'error': str(e)}
            return results
        
        # Step 2: Docling Conversion (if applicable)
        if test_file.suffix in ['.pdf', '.docx']:
            ctx = self.monitor.start_operation("processing_docling_conversion")
            try:
                result = self.pipeline.converter.convert(test_file)
                text = result.document.export_to_markdown()
                
                metric = self.monitor.end_operation(ctx, success=True,
                                                   metadata={'text_length': len(text)})
                print(f"✅ Docling Conversion: {metric.duration:.3f}s | Output: {len(text)} chars | Memory: {metric.memory_delta:+.2f} MB")
                results['docling_conversion'] = {
                    'time': metric.duration,
                    'text_length': len(text),
                    'memory_delta': metric.memory_delta
                }
            except Exception as e:
                metric = self.monitor.end_operation(ctx, success=False, error=str(e))
                print(f"❌ Docling Conversion Failed: {e}")
                results['docling_conversion'] = {'error': str(e)}
                return results
        else:
            text = content
            results['docling_conversion'] = {'skipped': 'text file'}
        
        # Step 3: Text Chunking
        ctx = self.monitor.start_operation("processing_text_chunking")
        try:
            from langchain.schema import Document
            doc = Document(page_content=text, metadata={'source': test_file.name})
            chunks = self.pipeline.text_splitter.split_documents([doc])
            
            metric = self.monitor.end_operation(ctx, success=True,
                                               metadata={'chunk_count': len(chunks)})
            print(f"✅ Text Chunking: {metric.duration:.3f}s | Chunks: {len(chunks)} | Avg size: {len(text)//len(chunks) if chunks else 0} chars")
            results['text_chunking'] = {
                'time': metric.duration,
                'chunk_count': len(chunks),
                'avg_chunk_size': len(text) // len(chunks) if chunks else 0
            }
        except Exception as e:
            metric = self.monitor.end_operation(ctx, success=False, error=str(e))
            print(f"❌ Text Chunking Failed: {e}")
            results['text_chunking'] = {'error': str(e)}
            return results
        
        # Step 4: Embedding Generation
        ctx = self.monitor.start_operation("processing_embedding_generation")
        try:
            # Embed first 5 chunks for testing
            test_chunks = chunks[:min(5, len(chunks))]
            texts = [chunk.page_content for chunk in test_chunks]
            
            embeddings = self.pipeline.embeddings.embed_documents(texts)
            
            metric = self.monitor.end_operation(ctx, success=True,
                                               metadata={
                                                   'chunks_embedded': len(test_chunks),
                                                   'embedding_dim': len(embeddings[0]) if embeddings else 0
                                               })
            print(f"✅ Embedding Generation: {metric.duration:.3f}s | {len(test_chunks)} chunks → {len(embeddings[0])}D vectors | Memory: {metric.memory_delta:+.2f} MB")
            print(f"   ⚡ Speed: {metric.duration/len(test_chunks):.3f}s per chunk")
            results['embedding_generation'] = {
                'time': metric.duration,
                'chunks_embedded': len(test_chunks),
                'time_per_chunk': metric.duration / len(test_chunks),
                'embedding_dim': len(embeddings[0]) if embeddings else 0,
                'memory_delta': metric.memory_delta
            }
        except Exception as e:
            metric = self.monitor.end_operation(ctx, success=False, error=str(e))
            print(f"❌ Embedding Generation Failed: {e}")
            results['embedding_generation'] = {'error': str(e)}
            return results
        
        # Step 5: FAISS Indexing (simulated)
        ctx = self.monitor.start_operation("processing_faiss_indexing")
        try:
            # Note: Not actually creating index to avoid side effects
            metric = self.monitor.end_operation(ctx, success=True)
            print(f"✅ FAISS Indexing (simulated): {metric.duration:.3f}s")
            results['faiss_indexing'] = {'time': metric.duration, 'simulated': True}
        except Exception as e:
            metric = self.monitor.end_operation(ctx, success=False, error=str(e))
            print(f"❌ FAISS Indexing Failed: {e}")
            results['faiss_indexing'] = {'error': str(e)}
        
        # Calculate total processing time
        total_time = sum([
            results.get('file_read', {}).get('time', 0),
            results.get('docling_conversion', {}).get('time', 0),
            results.get('text_chunking', {}).get('time', 0),
            results.get('embedding_generation', {}).get('time', 0),
            results.get('faiss_indexing', {}).get('time', 0)
        ])
        
        print(f"\n📊 Total Processing Time: {total_time:.3f}s")
        results['total_time'] = total_time
        
        return results
    
    # ========================================================================
    # SECTION 3: SEARCH ANALYSIS
    # ========================================================================
    
    def analyze_search(self, test_queries: Optional[List[str]] = None) -> Dict[str, Any]:
        """Analyze search performance"""
        print("\n" + "=" * 70)
        print("🔍 SECTION 3: SEARCH ANALYSIS")
        print("=" * 70)
        
        results = {}
        
        # Check if vectorstore exists
        if not self.pipeline.vectorstore:
            if not self.pipeline.load_index():
                print("⚠️  No vectorstore available. Process documents first.")
                return {'error': 'No vectorstore'}
        
        if test_queries is None:
            test_queries = [
                "test query",
                "invoice",
                "product manual",
                "company handbook"
            ]
        
        results['queries'] = {}
        
        for query in test_queries:
            print(f"\n📝 Query: '{query}'")
            query_results = {}
            
            # Step 1: Query Embedding
            ctx = self.monitor.start_operation("search_query_embedding")
            try:
                query_vector = self.pipeline.embeddings.embed_query(query)
                
                metric = self.monitor.end_operation(ctx, success=True,
                                                   metadata={'vector_dim': len(query_vector)})
                print(f"  ✅ Query Embedding: {metric.duration:.3f}s | {len(query_vector)}D vector")
                query_results['query_embedding'] = {
                    'time': metric.duration,
                    'vector_dim': len(query_vector)
                }
            except Exception as e:
                metric = self.monitor.end_operation(ctx, success=False, error=str(e))
                print(f"  ❌ Query Embedding Failed: {e}")
                query_results['query_embedding'] = {'error': str(e)}
                continue
            
            # Step 2: FAISS Vector Search
            ctx = self.monitor.start_operation("search_faiss_similarity")
            try:
                docs_with_scores = self.pipeline.vectorstore.similarity_search_with_score(query, k=100)
                
                metric = self.monitor.end_operation(ctx, success=True,
                                                   metadata={'results_count': len(docs_with_scores)})
                print(f"  ✅ FAISS Search: {metric.duration:.3f}s | {len(docs_with_scores)} results")
                query_results['faiss_search'] = {
                    'time': metric.duration,
                    'results_count': len(docs_with_scores)
                }
            except Exception as e:
                metric = self.monitor.end_operation(ctx, success=False, error=str(e))
                print(f"  ❌ FAISS Search Failed: {e}")
                query_results['faiss_search'] = {'error': str(e)}
                continue
            
            # Step 3: Enhanced Filtering
            ctx = self.monitor.start_operation("search_enhanced_filtering")
            try:
                filtered_results = self.pipeline.search(query)
                
                metric = self.monitor.end_operation(ctx, success=True,
                                                   metadata={'filtered_count': len(filtered_results)})
                print(f"  ✅ Enhanced Filtering: {metric.duration:.3f}s | {len(filtered_results)} after filter")
                print(f"     Filter ratio: {len(filtered_results)}/{len(docs_with_scores)} = {len(filtered_results)/len(docs_with_scores)*100:.1f}%")
                query_results['enhanced_filtering'] = {
                    'time': metric.duration,
                    'filtered_count': len(filtered_results),
                    'filter_ratio': len(filtered_results) / len(docs_with_scores) if docs_with_scores else 0
                }
            except Exception as e:
                metric = self.monitor.end_operation(ctx, success=False, error=str(e))
                print(f"  ❌ Enhanced Filtering Failed: {e}")
                query_results['enhanced_filtering'] = {'error': str(e)}
            
            # Total search time
            total_search_time = sum([
                query_results.get('query_embedding', {}).get('time', 0),
                query_results.get('faiss_search', {}).get('time', 0),
                query_results.get('enhanced_filtering', {}).get('time', 0)
            ])
            
            query_results['total_time'] = total_search_time
            print(f"  📊 Total Search Time: {total_search_time:.3f}s")
            
            results['queries'][query] = query_results
        
        # Calculate averages
        avg_times = {
            'query_embedding': [],
            'faiss_search': [],
            'enhanced_filtering': [],
            'total': []
        }
        
        for query_data in results['queries'].values():
            for key in avg_times.keys():
                if key == 'total':
                    avg_times[key].append(query_data.get('total_time', 0))
                else:
                    avg_times[key].append(query_data.get(key, {}).get('time', 0))
        
        results['averages'] = {
            key: sum(times) / len(times) if times else 0
            for key, times in avg_times.items()
        }
        
        print(f"\n📊 Average Search Times:")
        print(f"   Query Embedding: {results['averages']['query_embedding']:.3f}s")
        print(f"   FAISS Search: {results['averages']['faiss_search']:.3f}s")
        print(f"   Enhanced Filtering: {results['averages']['enhanced_filtering']:.3f}s")
        print(f"   Total: {results['averages']['total']:.3f}s")
        
        return results
    
    # ========================================================================
    # SECTION 4: Q&A ANALYSIS
    # ========================================================================
    
    def analyze_qa(self, test_questions: Optional[List[str]] = None) -> Dict[str, Any]:
        """Analyze Q&A performance"""
        print("\n" + "=" * 70)
        print("🤖 SECTION 4: Q&A ANALYSIS")
        print("=" * 70)
        
        results = {}
        
        if not self.pipeline.vectorstore:
            if not self.pipeline.load_index():
                print("⚠️  No vectorstore available. Process documents first.")
                return {'error': 'No vectorstore'}
        
        if test_questions is None:
            test_questions = [
                "What documents are available?",
                "Tell me about the invoice"
            ]
        
        results['questions'] = {}
        
        for question in test_questions:
            print(f"\n❓ Question: '{question}'")
            qa_results = {}
            
            # Full Q&A flow timing
            ctx = self.monitor.start_operation("qa_full_pipeline")
            try:
                # This includes: search + context building + LLM inference
                start = time.time()
                answer_result = self.pipeline.ask(question)
                total_time = time.time() - start
                
                metric = self.monitor.end_operation(ctx, success=True,
                                                   metadata={
                                                       'answer_length': len(answer_result['answer']),
                                                       'sources_count': len(answer_result['sources'])
                                                   })
                
                print(f"  ✅ Full Q&A: {metric.duration:.3f}s")
                print(f"     Answer: {len(answer_result['answer'])} chars")
                print(f"     Sources: {len(answer_result['sources'])} documents")
                print(f"     Memory: {metric.memory_delta:+.2f} MB")
                
                qa_results['full_pipeline'] = {
                    'time': metric.duration,
                    'answer_length': len(answer_result['answer']),
                    'sources_count': len(answer_result['sources']),
                    'memory_delta': metric.memory_delta
                }
                
                # Estimate breakdown (search is fast, most time is LLM)
                search_time = results.get('averages', {}).get('total', 0.01)
                llm_time = metric.duration - search_time
                
                print(f"  📊 Estimated Breakdown:")
                print(f"     Search: ~{search_time:.3f}s ({search_time/metric.duration*100:.1f}%)")
                print(f"     LLM: ~{llm_time:.3f}s ({llm_time/metric.duration*100:.1f}%)")
                
                qa_results['breakdown'] = {
                    'search_time': search_time,
                    'llm_time': llm_time
                }
                
            except Exception as e:
                metric = self.monitor.end_operation(ctx, success=False, error=str(e))
                print(f"  ❌ Q&A Failed: {e}")
                qa_results['full_pipeline'] = {'error': str(e)}
            
            results['questions'][question] = qa_results
        
        return results
    
    # ========================================================================
    # SECTION 5: BOTTLENECK IDENTIFICATION
    # ========================================================================
    
    def identify_bottlenecks(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Identify performance bottlenecks and optimization opportunities"""
        print("\n" + "=" * 70)
        print("🎯 SECTION 5: BOTTLENECK IDENTIFICATION & OPTIMIZATION")
        print("=" * 70)
        
        bottlenecks = []
        recommendations = []
        
        # Analyze document processing
        if 'document_processing' in analysis_results:
            proc = analysis_results['document_processing']
            
            # Docling conversion
            if 'docling_conversion' in proc and 'time' in proc['docling_conversion']:
                time_val = proc['docling_conversion']['time']
                if time_val > 2.0:
                    bottlenecks.append({
                        'section': 'Document Processing - Docling',
                        'severity': 'high' if time_val > 5.0 else 'medium',
                        'time': time_val,
                        'issue': f'Docling conversion taking {time_val:.2f}s'
                    })
                    recommendations.append({
                        'section': 'Docling Conversion',
                        'recommendation': 'Consider batch processing or caching converted documents'
                    })
            
            # Embedding generation
            if 'embedding_generation' in proc and 'time_per_chunk' in proc['embedding_generation']:
                time_per_chunk = proc['embedding_generation']['time_per_chunk']
                if time_per_chunk > 0.1:
                    bottlenecks.append({
                        'section': 'Embedding Generation',
                        'severity': 'high' if time_per_chunk > 0.2 else 'medium',
                        'time': time_per_chunk,
                        'issue': f'Embeddings taking {time_per_chunk:.3f}s per chunk'
                    })
                    recommendations.append({
                        'section': 'Embeddings',
                        'recommendation': 'Enable GPU acceleration or use batch processing'
                    })
        
        # Analyze search performance
        if 'search' in analysis_results and 'averages' in analysis_results['search']:
            avg = analysis_results['search']['averages']
            
            if avg.get('faiss_search', 0) > 0.05:
                bottlenecks.append({
                    'section': 'FAISS Search',
                    'severity': 'medium',
                    'time': avg['faiss_search'],
                    'issue': f'FAISS search taking {avg["faiss_search"]:.3f}s'
                })
                recommendations.append({
                    'section': 'FAISS',
                    'recommendation': 'Enable GPU for FAISS or use approximate search (IVF)'
                })
            
            if avg.get('enhanced_filtering', 0) > avg.get('faiss_search', 0):
                bottlenecks.append({
                    'section': 'Enhanced Filtering',
                    'severity': 'low',
                    'time': avg['enhanced_filtering'],
                    'issue': 'Filtering slower than FAISS search'
                })
                recommendations.append({
                    'section': 'Filtering',
                    'recommendation': 'Optimize filename matching logic or reduce candidates'
                })
        
        # Analyze Q&A performance
        if 'qa' in analysis_results and 'questions' in analysis_results['qa']:
            qa_times = []
            for qa_data in analysis_results['qa']['questions'].values():
                if 'full_pipeline' in qa_data and 'time' in qa_data['full_pipeline']:
                    qa_times.append(qa_data['full_pipeline']['time'])
            
            if qa_times:
                avg_qa_time = sum(qa_times) / len(qa_times)
                
                if avg_qa_time > 10.0:
                    bottlenecks.append({
                        'section': 'Q&A - LLM Inference',
                        'severity': 'high',
                        'time': avg_qa_time,
                        'issue': f'Q&A taking {avg_qa_time:.2f}s on average'
                    })
                    recommendations.append({
                        'section': 'LLM',
                        'recommendation': 'Use smaller model, reduce context size, or enable GPU'
                    })
        
        # Print bottlenecks
        print("\n🔴 Identified Bottlenecks:")
        if bottlenecks:
            for i, bottleneck in enumerate(sorted(bottlenecks, key=lambda x: x['time'], reverse=True), 1):
                severity_icon = "🔴" if bottleneck['severity'] == 'high' else "🟡" if bottleneck['severity'] == 'medium' else "🟢"
                print(f"\n{i}. {severity_icon} {bottleneck['section']}")
                print(f"   Time: {bottleneck['time']:.3f}s")
                print(f"   Issue: {bottleneck['issue']}")
        else:
            print("   ✅ No significant bottlenecks detected!")
        
        # Print recommendations
        print("\n💡 Optimization Recommendations:")
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(f"\n{i}. {rec['section']}")
                print(f"   → {rec['recommendation']}")
        else:
            print("   ✅ Performance is optimal!")
        
        return {
            'bottlenecks': bottlenecks,
            'recommendations': recommendations
        }
    
    # ========================================================================
    # FULL ANALYSIS
    # ========================================================================
    
    def run_full_analysis(self, export_path: Optional[str] = None) -> Dict[str, Any]:
        """Run complete pipeline analysis"""
        print("\n" + "=" * 70)
        print("🚀 STARTING FULL PIPELINE ANALYSIS")
        print("=" * 70)
        
        analysis_results = {}
        
        try:
            # Section 1: Initialization
            analysis_results['initialization'] = self.analyze_initialization()
            
            # Section 2: Document Processing
            analysis_results['document_processing'] = self.analyze_document_processing()
            
            # Section 3: Search
            analysis_results['search'] = self.analyze_search()
            
            # Section 4: Q&A
            analysis_results['qa'] = self.analyze_qa()
            
            # Section 5: Bottlenecks
            analysis_results['bottlenecks'] = self.identify_bottlenecks(analysis_results)
            
            # Print summary
            self.monitor.print_summary()
            
            # Export if requested
            if export_path:
                self.monitor.export_json(export_path)
            
        except Exception as e:
            print(f"\n❌ Analysis failed: {e}")
            traceback.print_exc()
        
        return analysis_results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution"""
    print("\n" + "=" * 70)
    print("📊 BACKEND LIVE ANALYSIS")
    print("Real-time Performance Monitoring & Optimization")
    print("=" * 70)
    
    try:
        # Initialize analyzer
        print("\n🔧 Initializing analyzer...")
        analyzer = PipelineAnalyzer()
        
        # Run full analysis
        results = analyzer.run_full_analysis(export_path="data/performance_analysis.json")
        
        print("\n" + "=" * 70)
        print("✅ ANALYSIS COMPLETE")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
