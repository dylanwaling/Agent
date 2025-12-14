# ============================================================================
# DOCUMENT PROCESSOR MODULE
# ============================================================================
# Purpose:
#   Handle document ingestion, conversion, chunking, and indexing
#   Processes documents through Docling → Text Splitting → FAISS
# ============================================================================

import logging
import time
from pathlib import Path
from typing import Optional, List

# Third-party imports
from langchain.schema import Document
from langchain_community.vectorstores import FAISS

# Local imports
from Config.settings import paths, model_config, logging_config, performance_config

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Document processing and indexing handler.
    
    Handles:
    - Document file reading and parsing
    - Text splitting and chunking
    - FAISS index creation and storage
    - GPU optimization for large document sets
    """
    
    def __init__(self, components, analytics_logger, docs_dir: Path, index_dir: Path):
        """
        Initialize document processor.
        
        Args:
            components: ComponentInitializer instance
            analytics_logger: AnalyticsLogger instance
            docs_dir: Directory containing documents to process
            index_dir: Directory for storing FAISS index
        """
        self.components = components
        self.analytics = analytics_logger
        self.docs_dir = docs_dir
        self.index_dir = index_dir
        self.vectorstore = None
    
    def process_all_documents(self) -> bool:
        """
        Process all documents in the documents directory and build FAISS index.
        
        Workflow:
        1. Scans documents directory for files
        2. Converts documents to text (using Docling for PDFs/images)
        3. Splits text into chunks
        4. Generates embeddings
        5. Builds FAISS vector store
        6. Saves index to disk
        
        Returns:
            True if processing successful, False otherwise
        """
        process_start = time.time()
        
        self.analytics.log_operation(
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
                    chunks = self._process_single_file(doc_file)
                    if chunks:
                        documents.extend(chunks)
                        processed_count += 1
                        
                except Exception as e:
                    logger.error(f"Error processing {doc_file.name}: {e}")
                    continue
            
            if not documents:
                logger.error("No valid document chunks created")
                return False
                
            logger.info(f"Created {len(documents)} chunks from {processed_count} documents")
            
            # Build vector store with GPU support
            success = self._build_vectorstore(documents)
            
            if success:
                process_time = time.time() - process_start
                logger.info(f"Document processing complete in {process_time:.2f}s!")
                
                self.analytics.log_operation(
                    operation_type="document_processing_complete",
                    operation=f"Processed {processed_count} documents successfully",
                    metadata={
                        "num_files": processed_count,
                        "num_chunks": len(documents),
                        "total_time_s": round(process_time, 3)
                    },
                    status="IDLE"
                )
                
            return success
            
        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            return False
    
    def _process_single_file(self, doc_file: Path) -> List[Document]:
        """
        Process a single document file.
        
        Args:
            doc_file: Path to document file
            
        Returns:
            List of Document chunks
        """
        file_start = time.time()
        logger.info(f"Processing: {doc_file.name}")
        
        # Log file upload
        self.analytics.log_operation(
            operation_type="file_upload",
            operation=f"Processing file: {doc_file.name}",
            metadata={
                "filename": doc_file.name,
                "file_size_bytes": doc_file.stat().st_size,
                "file_type": doc_file.suffix.lower()
            },
            status="PROCESSING"
        )
        
        # Extract text based on file type
        text = self._extract_text(doc_file)
        if not text or not text.strip():
            logger.warning(f"No text extracted from {doc_file.name}")
            return []
        
        # Split text into chunks
        chunks = self._split_text(doc_file, text)
        
        file_time = time.time() - file_start
        logger.info(f"✅ Successfully processed {doc_file.name} in {file_time:.2f}s")
        
        return chunks
    
    def _extract_text(self, doc_file: Path) -> Optional[str]:
        """
        Extract text from document file.
        
        Args:
            doc_file: Path to document file
            
        Returns:
            Extracted text or None
        """
        try:
            # Handle different file types
            if doc_file.suffix.lower() in ['.txt', '.md']:
                # Read text files directly
                with open(doc_file, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                # Use Docling for other formats (PDF, DOCX, images)
                parse_start = time.time()
                result = self.components.converter.convert(str(doc_file))
                text = result.document.export_to_markdown()
                parse_time = time.time() - parse_start
                
                # Log Docling parsing
                self.analytics.log_operation(
                    operation_type="docling_parse",
                    operation=f"Parsed: {doc_file.name}",
                    metadata={
                        "filename": doc_file.name,
                        "parse_time_s": round(parse_time, 3),
                        "extracted_length": len(text)
                    },
                    status="PROCESSING"
                )
                
                return text
                
        except Exception as e:
            logger.error(f"Text extraction failed for {doc_file.name}: {e}")
            return None
    
    def _split_text(self, doc_file: Path, text: str) -> List[Document]:
        """
        Split text into chunks and create Document objects.
        
        Args:
            doc_file: Original document file path
            text: Extracted text content
            
        Returns:
            List of Document chunks
        """
        split_start = time.time()
        chunks = self.components.text_splitter.split_text(text)
        split_time = time.time() - split_start
        
        # Log text splitting
        self.analytics.log_operation(
            operation_type="text_splitting",
            operation=f"Split text: {doc_file.name}",
            metadata={
                "filename": doc_file.name,
                "original_length": len(text),
                "num_chunks": len(chunks),
                "split_time_s": round(split_time, 3),
                "chunk_size": getattr(self.components.text_splitter, '_chunk_size', 1000),
                "chunk_overlap": getattr(self.components.text_splitter, '_chunk_overlap', 200)
            },
            status="PROCESSING"
        )
        
        # Create Document objects with metadata
        documents = []
        for i, chunk in enumerate(chunks):
            if chunk.strip():  # Only add non-empty chunks
                # Create searchable metadata
                searchable_content = f"{doc_file.name} {doc_file.stem} {chunk}"
                
                doc = Document(
                    page_content=searchable_content,
                    metadata={
                        "source": doc_file.name,
                        "filename": doc_file.stem,
                        "chunk_id": i,
                        "total_chunks": len(chunks),
                        "original_content": chunk
                    }
                )
                documents.append(doc)
        
        return documents
    
    def _build_vectorstore(self, documents: List[Document]) -> bool:
        """
        Build FAISS vectorstore from documents.
        
        Args:
            documents: List of Document chunks
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Log embedding generation
            embed_gen_start = time.time()
            self.analytics.log_operation(
                operation_type="embedding_generation",
                operation=f"Generating embeddings for {len(documents)} chunks",
                metadata={
                    "num_documents": len(documents),
                    "model": "all-MiniLM-L6-v2",
                    "dimensions": 384,
                    "device": getattr(self.components, 'device', 'cpu')
                },
                status="PROCESSING"
            )
            
            self.vectorstore = FAISS.from_documents(documents, self.components.embeddings)
            embed_gen_time = time.time() - embed_gen_start
            
            # Log FAISS indexing
            self.analytics.log_operation(
                operation_type="faiss_indexing",
                operation=f"Built FAISS index with {len(documents)} vectors",
                metadata={
                    "num_vectors": len(documents),
                    "embedding_time_s": round(embed_gen_time, 3),
                    "vectors_per_second": round(len(documents) / embed_gen_time, 2) if embed_gen_time > 0 else 0,
                    "index_size": self.vectorstore.index.ntotal,
                    "device": getattr(self.components, 'device', 'cpu')
                },
                status="PROCESSING"
            )
            
            # Move to GPU if available
            self._optimize_gpu()
            
            # Save index
            self._save_index()
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            return False
    
    def _optimize_gpu(self):
        """Move FAISS index to GPU if available."""
        if hasattr(self.components, 'device') and self.components.device == "cuda":
            try:
                import faiss
                if hasattr(faiss, 'StandardGpuResources'):
                    gpu_res = faiss.StandardGpuResources()
                    
                    # For 6GB GPUs, use memory-efficient settings
                    if hasattr(self.components, 'gpu_optimized') and self.components.gpu_optimized:
                        gpu_res.setTempMemory(performance_config.GPU_TEMP_MEMORY_GB * 1024 * 1024 * 1024)
                        logger.info("🔧 Using memory-optimized GPU settings for 6GB GPU")
                    
                    self.vectorstore.index = faiss.index_cpu_to_gpu(gpu_res, 0, self.vectorstore.index)
                    logger.info("🚀 Moved FAISS index to GPU")
            except Exception as gpu_error:
                logger.warning(f"Failed to move FAISS to GPU: {gpu_error}")
                logger.info("💻 Continuing with CPU FAISS")
                
                # Clear GPU memory if failed
                if hasattr(self.components, 'device') and self.components.device == "cuda":
                    try:
                        import torch
                        torch.cuda.empty_cache()
                    except:
                        pass
    
    def _save_index(self):
        """Save FAISS index to disk."""
        save_start = time.time()
        index_path = str(self.index_dir / "faiss_index")
        self.vectorstore.save_local(index_path)
        save_time = time.time() - save_start
        
        # Log index storage
        index_file = Path(index_path) / "index.faiss"
        index_size_mb = index_file.stat().st_size / (1024 * 1024) if index_file.exists() else 0
        
        self.analytics.log_operation(
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
    
    def load_index(self) -> bool:
        """
        Load existing FAISS index from disk.
        
        Returns:
            True if index loaded successfully, False otherwise
        """
        try:
            index_path = str(self.index_dir / "faiss_index")
            logger.info(f"Attempting to load index from: {index_path}")
            
            # Check for the actual file locations
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
                self.components.embeddings,
                allow_dangerous_deserialization=True
            )
            
            logger.info(f"✅ Index loaded successfully! Vectorstore has {self.vectorstore.index.ntotal} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
