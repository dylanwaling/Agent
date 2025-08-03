#!/usr/bin/env python3
"""
Document Pipeline Module
Clean document processing pipeline using Docling → LangChain → Search
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any
import shutil

# Core imports
from docling.document_converter import DocumentConverter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
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
                 model_name: str = "tinyllama:latest"):
        
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
        self.llm = OllamaLLM(model=self.model_name)        # Custom prompt template optimized for TinyLlama
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""Answer the question using the document content below.

Documents:
{context}

Question: {question}

Answer:"""
        )
        
        # Vector store
        self.vectorstore = None
        self.qa_chain = None
        
    def add_document(self, file_path: str) -> bool:
        """Add a single document to the collection"""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return False
                
            # Copy to docs directory
            dest_path = self.docs_dir / file_path.name
            shutil.copy2(file_path, dest_path)
            logger.info(f"Added document: {file_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding document {file_path}: {e}")
            return False
    
    def remove_document(self, filename: str) -> bool:
        """Remove a document from the collection"""
        try:
            file_path = self.docs_dir / filename
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Removed document: {filename}")
                return True
            else:
                logger.warning(f"Document not found: {filename}")
                return False
                
        except Exception as e:
            logger.error(f"Error removing document {filename}: {e}")
            return False
    
    def list_documents(self) -> List[str]:
        """List all documents in the collection"""
        try:
            docs = [f.name for f in self.docs_dir.iterdir() if f.is_file()]
            return sorted(docs)
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return []
    
    def process_documents(self) -> bool:
        """Process all documents and build index"""
        try:
            documents = []
            doc_files = list(self.docs_dir.iterdir())
            
            if not doc_files:
                logger.warning("No documents found to process")
                return False
                
            logger.info(f"Processing {len(doc_files)} documents...")
            
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
                        # Use Docling for other formats (PDF, DOCX, images)
                        result = self.converter.convert(str(doc_file))
                        text = result.document.export_to_markdown()
                    
                    if not text.strip():
                        logger.warning(f"No text extracted from {doc_file.name}")
                        continue
                    
                    # Create document chunks
                    chunks = self.text_splitter.split_text(text)
                    
                    for i, chunk in enumerate(chunks):
                        if chunk.strip():  # Only add non-empty chunks
                            doc = Document(
                                page_content=chunk,
                                metadata={
                                    "source": doc_file.name,
                                    "chunk_id": i,
                                    "total_chunks": len(chunks)
                                }
                            )
                            documents.append(doc)
                            
                except Exception as e:
                    logger.error(f"Error processing {doc_file.name}: {e}")
                    continue
            
            if not documents:
                logger.error("No valid document chunks created")
                return False
                
            logger.info(f"Created {len(documents)} document chunks")
            
            # Build vector store
            self.vectorstore = FAISS.from_documents(documents, self.embeddings)
            
            # Save index
            index_path = str(self.index_dir / "faiss_index")
            self.vectorstore.save_local(index_path)
            logger.info(f"Saved index to {index_path}")
              # Create QA chain with smart relevance-based retrieval
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs={
                        "score_threshold": 0.1,  # More inclusive threshold
                        "k": 100  # Search through many candidates
                    }
                ),
                return_source_documents=True,
                chain_type_kwargs={"prompt": self.prompt_template}
            )
            
            logger.info("Document processing complete!")
            return True
            
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
              # Create QA chain with smart relevance-based retrieval
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs={
                        "score_threshold": 0.1,  # More inclusive threshold
                        "k": 100  # Search through many candidates
                    }
                ),
                return_source_documents=True,
                chain_type_kwargs={"prompt": self.prompt_template}
            )
            
            logger.info("Index loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False
    
    def search(self, query: str, score_threshold: float = 0.1) -> List[Dict[str, Any]]:
        """Search documents with relevance-based filtering"""
        if not self.vectorstore:
            return []
            
        try:
            # Use similarity search with score threshold for smart retrieval
            docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=100)
            
            # Filter by relevance score
            relevant_docs = [(doc, score) for doc, score in docs_with_scores if score >= score_threshold]
            
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
        """Ask a question about the documents"""
        if not self.qa_chain:
            return {
                "answer": "No documents processed yet. Please process documents first.",
                "sources": []
            }
            
        try:
            result = self.qa_chain.invoke({"query": question})
            
            sources = []
            if "source_documents" in result:
                for doc in result["source_documents"]:
                    sources.append({
                        "source": doc.metadata.get("source", "Unknown"),
                        "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    })
            
            return {
                "answer": result.get("result", "No answer generated"),
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"Error asking question: {e}")
            return {
                "answer": f"Error processing question: {e}",
                "sources": []
            }
