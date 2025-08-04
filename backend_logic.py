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
                 model_name: str = "llama3:latest"):
        
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
        
        # Document converter (Docling)
        self.converter = DocumentConverter()
        
        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
          # LLM
        self.llm = OllamaLLM(model=self.model_name)        # Custom prompt template optimized for Llama3
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful assistant that answers questions based on the provided document content.

Use the following document excerpts to answer the question. Be direct and informative.

Document Content:
{context}

Question: {question}

Answer based on the documents above. If the question asks about a specific document or file, clearly identify which document contains the relevant information:"""
        )
        
        # Vector store
        self.vectorstore = None
        
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
            
            # Build vector store
            try:
                self.vectorstore = FAISS.from_documents(documents, self.embeddings)
                
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
            if not Path(index_path + ".faiss").exists():
                logger.warning("No existing index found")
                return False
                
            self.vectorstore = FAISS.load_local(
                index_path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            logger.info("Index loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading index: {e}")
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
        if not self.vectorstore:
            return {
                "answer": "No documents processed yet. Please process documents first.",
                "sources": []
            }
            
        try:
            # Use our enhanced search method directly (same as debug search)
            search_results = self.search(question)
            
            if not search_results:
                return {
                    "answer": "No relevant documents found for your question.",
                    "sources": []
                }
            
            # Prepare context from search results
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
            
            # Combine context
            context = "\n\n".join(context_parts)
            
            # Generate answer using LLM with context
            prompt = self.prompt_template.format(context=context, question=question)
            answer = self.llm.invoke(prompt)
            
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
