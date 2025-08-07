#!/usr/bin/env python3
"""
Document Pipeline Module
Clean document processing pipeline using Docling → LangChain → Search
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any

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
        
        # Initialize components
        self._init_components()
        
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
            temperature=0.2,      # Slightly higher for more thoughtful responses
            num_ctx=2048,        # Good context window
            num_predict=320,     # Allow longer responses for better reasoning
            streaming=True,      # Enable streaming for word-by-word display
        )        # Enhanced prompt template for thoughtful responses
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""Based on the provided documents, give a comprehensive and helpful answer. Explain concepts clearly and provide context when useful.

Documents:
{context}

Question: {question}

Answer:"""
        )
        
        # Vector store
        self.vectorstore = None
        
        # Try to load existing index on startup
        logger.info("Checking for existing document index...")
        if self.load_index():
            logger.info("✅ Existing document index loaded successfully!")
        else:
            logger.info("No existing index found - documents will need to be processed")
        
    def process_documents(self) -> bool:
        """Process all documents and build index"""
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
                    # Convert document with Docling
                    logger.info(f"Processing: {doc_file.name}")
                    
                    # Handle different file types
                    if doc_file.suffix.lower() in ['.txt', '.md']:
                        # Read text files directly
                        with open(doc_file, 'r', encoding='utf-8') as f:
                            text = f.read()
                    else:
                        # Use Docling for other formats (PDF, DOCX, images) with timeout protection
                        try:
                            result = self.converter.convert(str(doc_file))
                            text = result.document.export_to_markdown()
                        except Exception as docling_error:
                            logger.error(f"Docling failed for {doc_file.name}: {docling_error}")
                            continue
                    
                    if not text.strip():
                        logger.warning(f"No text extracted from {doc_file.name}")
                        continue
                    
                    # Create document chunks
                    chunks = self.text_splitter.split_text(text)
                    
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
                    
                    processed_count += 1
                    logger.info(f"✅ Successfully processed {doc_file.name}")
                            
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
                self.vectorstore = FAISS.from_documents(documents, self.embeddings)
                
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
                index_path = str(self.index_dir / "faiss_index")
                self.vectorstore.save_local(index_path)
                logger.info(f"Saved index to {index_path}")
                
                logger.info("Document processing complete!")
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
    
    def search(self, query: str, score_threshold: float = 1.25) -> List[Dict[str, Any]]:
        """Search documents with relevance-based filtering"""
        if not self.vectorstore:
            return []
            
        try:
            # Use similarity search with score threshold for smart retrieval
            docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=100)
            
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
                
            return results
            
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []
    
    def ask(self, question: str) -> Dict[str, Any]:
        """Ask a question about the documents using enhanced search logic"""
        import time
        start_time = time.time()
        
        if not self.vectorstore:
            return {
                "answer": "No documents processed yet. Please process documents first.",
                "sources": []
            }
            
        try:
            # Use our enhanced search method directly (same as debug search)
            logger.info(f"Starting search for: {question}")
            search_start = time.time()
            search_results = self.search(question)
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
            
            for result in search_results[:3]:  # Use top 3 most relevant results  
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
            
            # Combine context with moderate length limit
            context = "\n\n".join(context_parts)
            if len(context) > 1200:  # Reasonable context size limit
                context = context[:1200] + "..."
                
            context_time = time.time() - context_start
            logger.info(f"Context preparation completed in {context_time:.3f} seconds")
            
            # Generate answer using LangChain Ollama LLM
            logger.info(f"Sending prompt to LLM (context length: {len(context)} chars)")
            llm_start = time.time()
            prompt = self.prompt_template.format(context=context, question=question)
            logger.info(f"Prompt formatted ({len(prompt)} chars), calling LLM...")
            answer = self.llm.invoke(prompt)
            llm_time = time.time() - llm_start
            logger.info(f"LLM response received in {llm_time:.3f} seconds")
            
            total_time = time.time() - start_time
            logger.info(f"Total ask() time: {total_time:.3f} seconds (search: {search_time:.3f}s, context: {context_time:.3f}s, llm: {llm_time:.3f}s)")
            
            return {
                "answer": answer,
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"Error asking question: {e}")
            return {
                "answer": f"Error processing question: {e}",
                "sources": []
            }

    def ask_streaming(self, question: str):
        """Same as ask() but yields tokens as they're generated"""
        if not self.vectorstore:
            yield "No documents processed yet. Please process documents first."
            return
            
        try:
            # Use the same search logic as ask()
            search_results = self.search(question)
            if not search_results:
                yield "No relevant documents found for your question."
                return
            
            # Prepare context the same way
            context_parts = []
            sources = []
            
            for result in search_results[:3]:
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
            if len(context) > 1200:
                context = context[:1200] + "..."
            
            # Stream the response word by word
            prompt = self.prompt_template.format(context=context, question=question)
            for token in self.llm.stream(prompt):
                yield token
                
        except Exception as e:
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
