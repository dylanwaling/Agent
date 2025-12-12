#!/usr/bin/env python3
"""
Document Pipeline Module
Clean document processing pipeline using Docling → LangChain → Search
"""

import os
import logging
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Import event bus for live monitoring
try:
    from backend_live import event_bus
    LIVE_MONITORING_ENABLED = True
except ImportError:
    LIVE_MONITORING_ENABLED = False
    event_bus = None

# Core imports
from docling.document_converter import DocumentConverter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.schema import Document
from langchain.prompts import PromptTemplate

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentPipeline:
    """Clean document processing pipeline"""
    
    def __init__(self, 
                 docs_dir: str = "data/documents",
                 index_dir: str = "data/index",
                 model_name: str = "qwen2.5:1.5b"):
        
        self.docs_dir = Path(docs_dir)
        self.index_dir = Path(index_dir)
        self.model_name = model_name
        
        # Create directories
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Status file for live monitoring
        self.status_file = Path("data/pipeline_status.json")
        self._update_status("IDLE", "System initialized")
        
        # Initialize components
        self._init_components()
        
    def _log_operation(self, operation_type: str, operation: str, metadata: Optional[Dict] = None, status: str = "THINKING"):
        """
        Comprehensive logging system for all pipeline operations
        
        Args:
            operation_type: Type of operation (question_input, embedding_query, faiss_search, etc.)
            operation: Human-readable operation description
            metadata: Additional structured data for the operation
            status: Current pipeline status (THINKING, IDLE, ERROR, PROCESSING)
        """
        try:
            timestamp = time.time()
            operation_id = f"{operation_type}_{timestamp}"
            
            # Standardized log entry with rich metadata
            log_entry = {
                "timestamp": timestamp,
                "datetime": datetime.fromtimestamp(timestamp).isoformat(),
                "operation_type": operation_type,
                "operation": operation,
                "operation_id": operation_id,
                "status": status,
                "metadata": metadata or {}
            }
            
            # Write to operation history (JSONL format for easy parsing)
            history_file = self.status_file.parent / "operation_history.jsonl"
            history_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(history_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
                f.flush()
                os.fsync(f.fileno())
            
            # Publish to event bus for real-time monitoring (push-based)
            if LIVE_MONITORING_ENABLED and event_bus:
                event_bus.publish(log_entry)
            
            # Also update current status for real-time monitoring
            self._update_status_only(status, operation, metadata)
            
            logger.debug(f"[{operation_type.upper()}] {operation[:80]}")
            
        except Exception as e:
            logger.error(f"Failed to log operation: {e}")
    
    def _update_status_only(self, status: str, operation: str, metadata: Optional[Dict] = None):
        """Update status file only (used internally by _log_operation)"""
        try:
            timestamp = time.time()
            status_data = {
                "status": status,
                "operation": operation,
                "timestamp": timestamp,
                "operation_id": f"{status}_{timestamp}",
                "metadata": metadata or {}
            }
            
            # Ensure directory exists
            self.status_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write to temp file first, then atomic rename (prevents partial reads)
            temp_file = self.status_file.parent / f"status_{timestamp}.tmp"
            
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(status_data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            
            # Atomic rename (Windows-safe)
            try:
                if self.status_file.exists():
                    self.status_file.unlink()
            except:
                pass
            temp_file.rename(self.status_file)
            
        except Exception as e:
            logger.error(f"Failed to update status: {e}")
    
    def _update_status(self, status: str, operation: str, metadata: Optional[Dict] = None):
        """Backward compatibility wrapper - use _log_operation for new code"""
        self._log_operation(
            operation_type="general",
            operation=operation,
            metadata=metadata,
            status=status
        )
    
    def _init_components(self):
        """Initialize processing components"""
        
        # Check for GPU availability and optimize for hardware
        try:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.device == "cuda":
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info(f"🚀 GPU detected: {gpu_name}")
                logger.info(f"💾 GPU Memory: {gpu_memory_gb:.1f} GB")
                
                # Optimize for lower memory GPUs (like GTX 1060 6GB)
                if gpu_memory_gb < 8:
                    logger.info("� Optimizing for 6GB GPU...")
                    # Enable memory efficiency for smaller GPUs
                    torch.cuda.empty_cache()
                    self.gpu_optimized = True
                else:
                    self.gpu_optimized = False
            else:
                logger.info("�💻 Using CPU (no GPU detected)")
                self.gpu_optimized = False
        except ImportError:
            self.device = "cpu"
            self.gpu_optimized = False
            logger.info("💻 Using CPU (PyTorch not available)")
        
        # Document converter (Docling)
        self.converter = DocumentConverter()
        
        # Text splitter - optimized for GPU memory
        chunk_size = 800 if hasattr(self, 'gpu_optimized') and self.gpu_optimized else 1000
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=150 if chunk_size == 800 else 200,
            length_function=len,
        )
        
        # Embeddings with GPU support
        embedding_kwargs = {
            "model_name": "sentence-transformers/all-MiniLM-L6-v2"
        }
        
        # Add device specification if GPU is available
        if self.device == "cuda":
            embedding_kwargs["model_kwargs"] = {"device": self.device}
            embedding_kwargs["encode_kwargs"] = {"device": self.device}
        
        self.embeddings = HuggingFaceEmbeddings(**embedding_kwargs)
        
        # LLM optimized for qwen2.5:1.5b - excellent reasoning performance
        self.llm = OllamaLLM(
            model=self.model_name,
            temperature=0.3,      # Balanced for thoughtful but focused responses
            num_ctx=4096,        # Larger context window for more document content
            num_predict=800,     # Much longer responses for completeness
            streaming=True,      # Enable streaming for word-by-word display
        )        # Enhanced prompt template for thoughtful responses
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful assistant analyzing documents. Using ONLY the information in the documents provided below, give a complete and accurate answer to the question. 

IMPORTANT:
- Be thorough and include all relevant details from the documents
- If information is incomplete or missing, say so explicitly
- Cite which document(s) you're referring to
- Give a full, complete response - don't cut off mid-thought

Documents:
{context}

Question: {question}

Complete Answer:"""
        )
        
        # Vector store
        self.vectorstore = None
        
        # Try to load existing index on startup
        logger.info("Checking for existing document index...")
        if self.load_index():
            logger.info("✅ Existing document index loaded successfully!")
        else:
            logger.info("No existing index found - documents will need to be processed")
        
        # Set status back to idle after initialization
        self._update_status("IDLE", "System ready")
        
    def process_documents(self) -> bool:
        """Process all documents and build index"""
        process_start = time.time()
        
        self._log_operation(
            operation_type="document_processing_start",
            operation="Starting document processing pipeline",
            metadata={},
            status="PROCESSING"
        )
        
        try:
            documents = []
            doc_files = list(self.docs_dir.iterdir())
            
            if not doc_files:
                logger.warning("No documents found to process")
                return False
                
            logger.info(f"Processing {len(doc_files)} documents...")
            
            # Process each document with better error handling
            processed_count = 0
            for doc_file in doc_files:
                if not doc_file.is_file():
                    continue
                    
                try:
                    file_start = time.time()
                    # Convert document with Docling
                    logger.info(f"Processing: {doc_file.name}")
                    
                    # Log file upload
                    self._log_operation(
                        operation_type="file_upload",
                        operation=f"Processing file: {doc_file.name}",
                        metadata={
                            "filename": doc_file.name,
                            "file_size_bytes": doc_file.stat().st_size,
                            "file_type": doc_file.suffix.lower()
                        },
                        status="PROCESSING"
                    )
                    
                    # Handle different file types
                    if doc_file.suffix.lower() in ['.txt', '.md']:
                        # Read text files directly
                        with open(doc_file, 'r', encoding='utf-8') as f:
                            text = f.read()
                    else:
                        # Use Docling for other formats (PDF, DOCX, images) with timeout protection
                        try:
                            parse_start = time.time()
                            result = self.converter.convert(str(doc_file))
                            text = result.document.export_to_markdown()
                            parse_time = time.time() - parse_start
                            
                            # Log Docling parsing
                            self._log_operation(
                                operation_type="docling_parse",
                                operation=f"Parsed: {doc_file.name}",
                                metadata={
                                    "filename": doc_file.name,
                                    "parse_time_s": round(parse_time, 3),
                                    "extracted_length": len(text)
                                },
                                status="PROCESSING"
                            )
                        except Exception as docling_error:
                            logger.error(f"Docling failed for {doc_file.name}: {docling_error}")
                            continue
                    
                    if not text.strip():
                        logger.warning(f"No text extracted from {doc_file.name}")
                        continue
                    
                    # Create document chunks
                    split_start = time.time()
                    chunks = self.text_splitter.split_text(text)
                    split_time = time.time() - split_start
                    
                    # Log text splitting
                    self._log_operation(
                        operation_type="text_splitting",
                        operation=f"Split text: {doc_file.name}",
                        metadata={
                            "filename": doc_file.name,
                            "original_length": len(text),
                            "num_chunks": len(chunks),
                            "split_time_s": round(split_time, 3),
                            "chunk_size": getattr(self.text_splitter, '_chunk_size', 1000),
                            "chunk_overlap": getattr(self.text_splitter, '_chunk_overlap', 200)
                        },
                        status="PROCESSING"
                    )
                    
                    for i, chunk in enumerate(chunks):
                        if chunk.strip():  # Only add non-empty chunks
                            # Create searchable metadata
                            searchable_content = f"{doc_file.name} {doc_file.stem} {chunk}"
                            
                            doc = Document(
                                page_content=searchable_content,  # Include filename in content for better matching
                                metadata={
                                    "source": doc_file.name,
                                    "filename": doc_file.stem,  # Filename without extension
                                    "chunk_id": i,
                                    "total_chunks": len(chunks),
                                    "original_content": chunk  # Keep original for display
                                }
                            )
                            documents.append(doc)
                    
                    file_time = time.time() - file_start
                    processed_count += 1
                    logger.info(f"✅ Successfully processed {doc_file.name} in {file_time:.2f}s")
                            
                except Exception as e:
                    logger.error(f"Error processing {doc_file.name}: {e}")
                    # Continue with other documents even if one fails
                    continue
            
            if not documents:
                logger.error("No valid document chunks created")
                return False
                
            logger.info(f"Created {len(documents)} chunks from {processed_count} documents")
            
            # Build vector store with GPU support
            try:
                # Log embedding generation
                embed_gen_start = time.time()
                self._log_operation(
                    operation_type="embedding_generation",
                    operation=f"Generating embeddings for {len(documents)} chunks",
                    metadata={
                        "num_documents": len(documents),
                        "model": "all-MiniLM-L6-v2",
                        "dimensions": 384,
                        "device": getattr(self, 'device', 'cpu')
                    },
                    status="PROCESSING"
                )
                
                self.vectorstore = FAISS.from_documents(documents, self.embeddings)
                embed_gen_time = time.time() - embed_gen_start
                
                # Log FAISS indexing
                self._log_operation(
                    operation_type="faiss_indexing",
                    operation=f"Built FAISS index with {len(documents)} vectors",
                    metadata={
                        "num_vectors": len(documents),
                        "embedding_time_s": round(embed_gen_time, 3),
                        "vectors_per_second": round(len(documents) / embed_gen_time, 2) if embed_gen_time > 0 else 0,
                        "index_size": self.vectorstore.index.ntotal,
                        "device": getattr(self, 'device', 'cpu')
                    },
                    status="PROCESSING"
                )
                
                # Move to GPU if available (with memory management for 6GB GPUs)
                if hasattr(self, 'device') and self.device == "cuda":
                    try:
                        # Move FAISS index to GPU with memory optimization
                        import faiss
                        if hasattr(faiss, 'StandardGpuResources'):
                            gpu_res = faiss.StandardGpuResources()
                            
                            # For 6GB GPUs, use memory-efficient settings
                            if hasattr(self, 'gpu_optimized') and self.gpu_optimized:
                                # Limit GPU memory usage for smaller GPUs
                                gpu_res.setTempMemory(2 * 1024 * 1024 * 1024)  # 2GB temp memory
                                logger.info("🔧 Using memory-optimized GPU settings for 6GB GPU")
                            
                            self.vectorstore.index = faiss.index_cpu_to_gpu(gpu_res, 0, self.vectorstore.index)
                            logger.info("🚀 Moved FAISS index to GPU")
                    except Exception as gpu_error:
                        logger.warning(f"Failed to move FAISS to GPU: {gpu_error}")
                        logger.info("💻 Continuing with CPU FAISS")
                        
                        # Clear GPU memory if failed
                        if hasattr(self, 'device') and self.device == "cuda":
                            try:
                                import torch
                                torch.cuda.empty_cache()
                            except:
                                pass
                
                # Save index
                save_start = time.time()
                index_path = str(self.index_dir / "faiss_index")
                self.vectorstore.save_local(index_path)
                save_time = time.time() - save_start
                
                # Log index storage
                index_file = Path(index_path) / "index.faiss"
                index_size_mb = index_file.stat().st_size / (1024 * 1024) if index_file.exists() else 0
                
                self._log_operation(
                    operation_type="index_storage",
                    operation=f"Saved FAISS index to disk",
                    metadata={
                        "index_path": index_path,
                        "index_size_mb": round(index_size_mb, 2),
                        "save_time_s": round(save_time, 3),
                        "num_vectors": self.vectorstore.index.ntotal
                    },
                    status="PROCESSING"
                )
                
                process_time = time.time() - process_start
                logger.info(f"Document processing complete in {process_time:.2f}s!")
                
                self._log_operation(
                    operation_type="document_processing_complete",
                    operation=f"Processed {processed_count} documents successfully",
                    metadata={
                        "num_files": processed_count,
                        "num_chunks": len(documents),
                        "total_time_s": round(process_time, 3)
                    },
                    status="IDLE"
                )
                
                return True
                
            except Exception as vector_error:
                logger.error(f"Error creating vector store: {vector_error}")
                return False
            
        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            return False
    
    def load_index(self) -> bool:
        """Load existing index"""
        try:
            index_path = str(self.index_dir / "faiss_index")
            logger.info(f"Attempting to load index from: {index_path}")
            
            # Check for the actual file locations (inside the faiss_index directory)
            faiss_file = Path(index_path) / "index.faiss"
            pkl_file = Path(index_path) / "index.pkl"
            
            logger.info(f"Looking for FAISS file at: {faiss_file}")
            logger.info(f"Looking for PKL file at: {pkl_file}")
            
            if not faiss_file.exists():
                logger.warning(f"FAISS file not found at: {faiss_file}")
                return False
                
            if not pkl_file.exists():
                logger.warning(f"PKL file not found at: {pkl_file}")
                return False
            
            logger.info("Both index files found, loading vectorstore...")
            self.vectorstore = FAISS.load_local(
                index_path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            logger.info(f"✅ Index loaded successfully! Vectorstore has {self.vectorstore.index.ntotal} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            logger.error(f"Exception type: {type(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
    
    def search(self, query: str, score_threshold: float = 1.25, update_status: bool = True) -> List[Dict[str, Any]]:
        """Search documents with relevance-based filtering"""
        search_start = time.time()
        
        try:
            if not self.vectorstore:
                return []
            
            # Log embedding query
            embed_start = time.time()
            self._log_operation(
                operation_type="embedding_query",
                operation=f"Embedding query: {query[:80]}",
                metadata={
                    "query": query,
                    "query_length": len(query),
                    "model": "all-MiniLM-L6-v2",
                    "dimensions": 384,
                    "device": getattr(self, 'device', 'cpu')
                },
                status="PROCESSING"
            )
            
            # Use similarity search with score threshold for smart retrieval
            docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=100)
            embed_time = time.time() - embed_start
            
            # Log FAISS search
            self._log_operation(
                operation_type="faiss_search",
                operation=f"FAISS search: {query[:80]}",
                metadata={
                    "query": query,
                    "k": 100,
                    "num_results": len(docs_with_scores),
                    "index_size": self.vectorstore.index.ntotal if self.vectorstore else 0,
                    "search_time_ms": round(embed_time * 1000, 2),
                    "device": getattr(self, 'device', 'cpu')
                },
                status="PROCESSING"
            )
            
            # Smart filtering with filename priority
            relevant_docs = []
            query_lower = query.lower()
            
            for doc, score in docs_with_scores:
                source = doc.metadata.get("source", "").lower()
                filename = doc.metadata.get("filename", "").lower()
                
                # Check for exact filename matches with various transformations
                query_normalized = query_lower.replace(" ", "_").replace("-", "_")
                source_normalized = source.replace(" ", "_").replace("-", "_")
                filename_normalized = filename.replace(" ", "_").replace("-", "_")
                
                # Strong filename match (exact or very close)
                strong_filename_match = (
                    query_normalized in source_normalized or 
                    query_normalized in filename_normalized or
                    source_normalized.replace("_", "").replace(".", "") in query_normalized.replace("_", "").replace(".", "") or
                    filename_normalized in query_normalized or
                    query_lower.replace(" ", "") in source.replace("_", "").replace("-", "").replace(".", "")
                )
                
                # Weaker filename match (partial)
                weak_filename_match = (
                    any(word in source for word in query_lower.split() if len(word) > 2) or
                    any(word in filename for word in query_lower.split() if len(word) > 2)
                )
                
                # Apply different thresholds based on match strength
                if strong_filename_match:
                    # Very lenient for strong filename matches
                    if score <= 2.5:
                        relevant_docs.append((doc, score * 0.5))  # Boost score by halving it
                elif weak_filename_match:
                    # Moderate threshold for weak filename matches  
                    if score <= 2.0:
                        relevant_docs.append((doc, score * 0.8))  # Small boost
                elif score <= score_threshold:
                    # Strict threshold for content-only matches
                    relevant_docs.append((doc, score))
            
            # Sort by score (lower is better in FAISS)
            relevant_docs.sort(key=lambda x: x[1])
            
            results = []
            for doc, score in relevant_docs:
                results.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "Unknown"),
                    "chunk_id": doc.metadata.get("chunk_id", 0),
                    "relevance_score": score
                })
            
            search_time = time.time() - search_start
            
            # Log relevance filtering results
            self._log_operation(
                operation_type="relevance_filter",
                operation=f"Filtered search results: {query[:80]}",
                metadata={
                    "query": query,
                    "total_candidates": len(docs_with_scores),
                    "filtered_results": len(results),
                    "score_threshold": score_threshold,
                    "filter_time_s": round(search_time, 3)
                },
                status="IDLE" if update_status else "PROCESSING"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching: {e}")
            self._log_operation(
                operation_type="error",
                operation=f"Search error: {str(e)[:100]}",
                metadata={
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                },
                status="ERROR"
            )
            return []
    
    def ask(self, question: str) -> Dict[str, Any]:
        """Ask a question about the documents using enhanced search logic"""
        start_time = time.time()
        
        # Log the incoming question
        self._log_operation(
            operation_type="question_input",
            operation=f"Question: {question[:100]}",
            metadata={"question": question, "question_length": len(question)},
            status="THINKING"
        )
        
        try:
            if not self.vectorstore:
                return {
                    "answer": "No documents processed yet. Please process documents first.",
                    "sources": []
                }
            
            # Use our enhanced search method directly (same as debug search)
            logger.info(f"Starting search for: {question}")
            search_start = time.time()
            search_results = self.search(question, update_status=False)
            search_time = time.time() - search_start
            logger.info(f"Search completed in {search_time:.3f} seconds")
            
            if not search_results:
                return {
                    "answer": "No relevant documents found for your question.",
                    "sources": []
                }
            
            # Prepare context from search results (optimized but functional)
            logger.info(f"Preparing context from {len(search_results)} search results")
            context_start = time.time()
            context_parts = []
            sources = []
            
            for result in search_results[:8]:  # Use top 8 results for comprehensive context
                # Extract clean content (remove filename prefix)
                content = result["content"]
                source_name = result["source"]
                
                # Remove filename prefix to get clean content
                # The content format is: "filename.ext filename_stem actual_content"
                parts = content.split(' ', 2)  # Split into filename, stem, and content
                if len(parts) >= 3:
                    clean_content = parts[2]  # Get the actual content part
                else:
                    clean_content = content  # Fallback to full content
                
                # Add source context to make it clear which document this content comes from
                contextual_content = f"From document '{source_name}':\n{clean_content}"
                
                context_parts.append(contextual_content)
                sources.append({
                    "source": source_name,
                    "content": clean_content[:200] + "..." if len(clean_content) > 200 else clean_content
                })
            
            # Combine context with larger length limit for comprehensive answers
            context = "\n\n".join(context_parts)
            context_truncated = False
            if len(context) > 3500:  # Much larger context for complete information
                context = context[:3500] + "..."
                context_truncated = True
                
            context_time = time.time() - context_start
            
            # Log context building
            self._log_operation(
                operation_type="context_builder",
                operation=f"Built context for: {question[:60]}",
                metadata={
                    "question": question,
                    "num_sources": len(sources),
                    "context_length": len(context),
                    "context_truncated": context_truncated,
                    "build_time_s": round(context_time, 3)
                },
                status="PROCESSING"
            )
            
            logger.info(f"Context preparation completed in {context_time:.3f} seconds")
            
            # Generate answer using LangChain Ollama LLM
            logger.info(f"Sending prompt to LLM (context length: {len(context)} chars)")
            llm_start = time.time()
            prompt = self.prompt_template.format(context=context, question=question)
            
            # Log prompt assembly
            self._log_operation(
                operation_type="prompt_assembly",
                operation=f"Prompt assembled for: {question[:60]}",
                metadata={
                    "question": question,
                    "context_length": len(context),
                    "prompt_length": len(prompt),
                    "num_sources": len(sources),
                    "model": self.model_name
                },
                status="PROCESSING"
            )
            
            logger.info(f"Prompt formatted ({len(prompt)} chars), calling LLM...")
            answer = self.llm.invoke(prompt)
            llm_time = time.time() - llm_start
            
            # Log LLM generation
            self._log_operation(
                operation_type="llm_generation",
                operation=f"LLM response for: {question[:60]}",
                metadata={
                    "question": question,
                    "model": self.model_name,
                    "response_length": len(answer),
                    "generation_time_s": round(llm_time, 3),
                    "tokens_per_second": round(len(answer.split()) / llm_time, 2) if llm_time > 0 else 0
                },
                status="PROCESSING"
            )
            
            logger.info(f"LLM response received in {llm_time:.3f} seconds")
            
            total_time = time.time() - start_time
            logger.info(f"Total ask() time: {total_time:.3f} seconds (search: {search_time:.3f}s, context: {context_time:.3f}s, llm: {llm_time:.3f}s)")
            
            # Log complete response
            self._log_operation(
                operation_type="response_complete",
                operation=f"Response complete for: {question[:60]}",
                metadata={
                    "question": question,
                    "total_time_s": round(total_time, 3),
                    "search_time_s": round(search_time, 3),
                    "context_time_s": round(context_time, 3),
                    "llm_time_s": round(llm_time, 3),
                    "answer_length": len(answer),
                    "num_sources": len(sources)
                },
                status="IDLE"
            )
            
            return {
                "answer": answer,
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"Error asking question: {e}")
            
            # Log error
            self._log_operation(
                operation_type="error",
                operation=f"Error processing question: {str(e)[:100]}",
                metadata={
                    "question": question,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                },
                status="ERROR"
            )
            
            return {
                "answer": f"Error processing question: {e}",
                "sources": []
            }

    def ask_streaming(self, question: str):
        """Same as ask() but yields tokens as they're generated"""
        stream_start = time.time()
        
        # Log streaming question
        self._log_operation(
            operation_type="question_input",
            operation=f"Question (streaming): {question[:100]}",
            metadata={"question": question, "question_length": len(question), "streaming": True},
            status="THINKING"
        )
        
        try:
            if not self.vectorstore:
                yield "No documents processed yet. Please process documents first."
                return
            
            # Use the same search logic as ask() but don't let it change status
            search_results = self.search(question, update_status=False)
            if not search_results:
                yield "No relevant documents found for your question."
                return
            
            # Prepare context the same way as ask() - use more results for completeness
            context_parts = []
            sources = []
            
            for result in search_results[:8]:  # Match the ask() method
                content = result["content"]
                source_name = result["source"]
                parts = content.split(' ', 2)
                if len(parts) >= 3:
                    clean_content = parts[2]
                else:
                    clean_content = content
                
                contextual_content = f"From document '{source_name}':\n{clean_content}"
                context_parts.append(contextual_content)
                sources.append({
                    "source": source_name,
                    "content": clean_content[:200] + "..." if len(clean_content) > 200 else clean_content
                })
            
            context = "\n\n".join(context_parts)
            if len(context) > 3500:  # Match the ask() method's larger limit
                context = context[:3500] + "..."
            
            # Log streaming start
            self._log_operation(
                operation_type="response_stream_start",
                operation=f"Starting stream for: {question[:60]}",
                metadata={
                    "question": question,
                    "num_sources": len(sources),
                    "context_length": len(context)
                },
                status="PROCESSING"
            )
            
            # Stream the response word by word
            prompt = self.prompt_template.format(context=context, question=question)
            token_count = 0
            for token in self.llm.stream(prompt):
                token_count += 1
                yield token
            
            stream_time = time.time() - stream_start
            
            # Log streaming complete
            self._log_operation(
                operation_type="response_stream_complete",
                operation=f"Stream complete for: {question[:60]}",
                metadata={
                    "question": question,
                    "stream_time_s": round(stream_time, 3),
                    "token_count": token_count
                },
                status="IDLE"
            )
                
        except Exception as e:
            self._log_operation(
                operation_type="error",
                operation=f"Streaming error: {str(e)[:100]}",
                metadata={
                    "question": question,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                },
                status="ERROR"
            )
            yield f"Error processing question: {e}"

    def process_single_document(self, file_path: Path) -> bool:
        """Process a single document and add it to existing vectorstore"""
        try:
            if not self.vectorstore:
                logger.warning("No existing vectorstore - processing single document as new collection")
                return self.process_documents()
            
            logger.info(f"Processing single document: {file_path.name}")
            
            # Process the single document
            if file_path.suffix.lower() == '.md':
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                document = Document(
                    page_content=content,
                    metadata={"source": file_path.name}
                )
                documents = [document]
                
            elif file_path.suffix.lower() == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                document = Document(
                    page_content=content,
                    metadata={"source": file_path.name}
                )
                documents = [document]
                
            elif file_path.suffix.lower() == '.pdf':
                from docling.document_converter import DocumentConverter
                converter = DocumentConverter()
                result = converter.convert(file_path)
                
                content = result.document.export_to_markdown()
                document = Document(
                    page_content=content,
                    metadata={"source": file_path.name}
                )
                documents = [document]
                
            else:
                logger.warning(f"Unsupported file type: {file_path.suffix}")
                return False
            
            # Split into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            # Add filename prefix to chunks for better search
            processed_chunks = []
            for chunk in chunks:
                chunk_content = f"{file_path.stem} {file_path.stem} {chunk.page_content}"
                processed_chunk = Document(
                    page_content=chunk_content,
                    metadata=chunk.metadata
                )
                processed_chunks.append(processed_chunk)
            
            logger.info(f"Created {len(processed_chunks)} chunks from {file_path.name}")
            
            # Add to existing vectorstore
            self.vectorstore.add_documents(processed_chunks)
            
            # Save updated index
            index_path = str(self.index_dir / "faiss_index")
            self.vectorstore.save_local(index_path)
            logger.info(f"Updated index saved to {index_path}")
            
            logger.info(f"✅ Single document processed: {file_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing single document {file_path}: {e}")
            return False
    
    def debug_search(self, query: str) -> Dict[str, Any]:
        """Debug search to see what's being retrieved using our enhanced search logic"""
        if not self.vectorstore:
            return {"error": "No vectorstore available"}
        
        try:
            # Use our enhanced search method instead of raw FAISS
            search_results = self.search(query)
            
            results = {
                "query": query,
                "total_results": len(search_results),
                "results": []
            }
            
            for i, result in enumerate(search_results[:10]):  # Show top 10
                results["results"].append({
                    "rank": i + 1,
                    "score": result["relevance_score"],
                    "source": result["source"],
                    "chunk_id": result["chunk_id"],
                    "content_preview": result["content"][:100] + "..."
                })
            
            return results
            
        except Exception as e:
            return {"error": str(e)}
