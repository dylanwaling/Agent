# ============================================================================
# DOCUMENT PIPELINE MODULE
# ============================================================================
# Purpose:
#   Main coordinator for document processing pipeline
#   Integrates all components: analytics, processing, and search
# ============================================================================

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

# Local imports
from config.settings import paths, model_config, logging_config
from .analytics import AnalyticsLogger
from .components import ComponentInitializer
from .document_processor import DocumentProcessor
from .search_engine import SearchEngine

logger = logging.getLogger(__name__)


class DocumentPipeline:
    """
    Clean document processing pipeline using Docling → LangChain → Search.
    
    This class coordinates:
    - Document ingestion and processing
    - Vector embeddings and FAISS indexing
    - Semantic search and question answering
    - Real-time operation logging and monitoring
    """
    
    def __init__(self, 
                 docs_dir: Optional[str] = None,
                 index_dir: Optional[str] = None,
                 model_name: Optional[str] = None):
        """
        Initialize the document processing pipeline.
        
        Args:
            docs_dir: Directory containing documents (defaults to config)
            index_dir: Directory for FAISS index (defaults to config)
            model_name: LLM model name (defaults to config)
        """
        # Use config defaults if not provided
        self.docs_dir = Path(docs_dir) if docs_dir else paths.DOCS_DIR
        self.index_dir = Path(index_dir) if index_dir else paths.INDEX_DIR
        self.model_name = model_name if model_name else model_config.LLM_MODEL
        
        # Create directories
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Status file for live monitoring
        self.status_file = paths.STATUS_FILE
        
        # Initialize core components
        self._init_pipeline()
    
    def _init_pipeline(self):
        """Initialize all pipeline components."""
        # Analytics logger
        self.analytics = AnalyticsLogger(self.status_file)
        self.analytics.update_status(logging_config.STATUS_TYPES['IDLE'], "System initialized")
        
        # Component initializer (GPU, embeddings, LLM, etc.)
        self.components = ComponentInitializer(self.model_name)
        self.components.init_all()
        
        # Document processor
        self.doc_processor = DocumentProcessor(
            components=self.components,
            analytics_logger=self.analytics,
            docs_dir=self.docs_dir,
            index_dir=self.index_dir
        )
        
        # Search engine
        self.search_engine = SearchEngine(
            document_processor=self.doc_processor,
            components=self.components,
            analytics_logger=self.analytics
        )
        
        # Try to load existing index on startup
        logger.info("Checking for existing document index...")
        if self.doc_processor.load_index():
            logger.info("✅ Existing document index loaded successfully!")
        else:
            logger.info("No existing index found - documents will need to be processed")
        
        # Set status back to idle after initialization
        self.analytics.update_status("IDLE", "System ready")
    
    # ------------------------------------------------------------------------
    # Public API - Document Processing
    # ------------------------------------------------------------------------
    
    def process_documents(self) -> bool:
        """
        Process all documents in the documents directory and build FAISS index.
        
        Returns:
            True if processing successful, False otherwise
        """
        return self.doc_processor.process_all_documents()
    
    def process_single_document(self, file_path: Path) -> bool:
        """
        Process a single document and add it to existing vectorstore.
        
        Args:
            file_path: Path to the document file to process
            
        Returns:
            True if successful, False otherwise
        """
        # For now, just reprocess all documents
        # TODO: Implement true single document processing
        logger.warning("Single document processing not fully implemented - reprocessing all documents")
        return self.process_documents()
    
    def load_index(self) -> bool:
        """
        Load existing FAISS index from disk.
        
        Returns:
            True if index loaded successfully, False otherwise
        """
        return self.doc_processor.load_index()
    
    # ------------------------------------------------------------------------
    # Public API - Search & Query
    # ------------------------------------------------------------------------
    
    def search(self, query: str, score_threshold: Optional[float] = None, 
               update_status: bool = True) -> List[Dict[str, Any]]:
        """
        Search documents with relevance-based filtering.
        
        Args:
            query: Search query text
            score_threshold: Maximum distance score for relevance filtering
            update_status: Whether to update status to IDLE after search
            
        Returns:
            List of search results with content, source, chunk_id, and relevance_score
        """
        return self.search_engine.search(query, score_threshold, update_status)
    
    def ask(self, question: str) -> Dict[str, Any]:
        """
        Ask a question about the documents.
        
        Args:
            question: Natural language question to answer
            
        Returns:
            Dictionary containing 'answer' (string) and 'sources' (list of dicts)
        """
        return self.search_engine.ask(question)
    
    def ask_streaming(self, question: str):
        """
        Ask a question with streaming response.
        
        Args:
            question: Natural language question to answer
            
        Yields:
            String tokens from the LLM response as they are generated
        """
        yield from self.search_engine.ask_streaming(question)
    
    def debug_search(self, query: str) -> Dict[str, Any]:
        """
        Debug search to see what's being retrieved.
        
        Args:
            query: Search query text to debug
            
        Returns:
            Dictionary with query, total_results count, and top 10 results
        """
        return self.search_engine.debug_search(query)
    
    # ------------------------------------------------------------------------
    # Backward Compatibility - Old Methods
    # ------------------------------------------------------------------------
    
    def _log_operation(self, operation_type: str, operation: str, 
                      metadata: Optional[Dict] = None, status: str = "THINKING"):
        """Backward compatibility wrapper for analytics logging."""
        self.analytics.log_operation(operation_type, operation, metadata, status)
    
    def _update_status(self, status: str, operation: str, metadata: Optional[Dict] = None):
        """Backward compatibility wrapper for status updates."""
        self.analytics.update_status(status, operation, metadata)
    
    @property
    def vectorstore(self):
        """Backward compatibility - access vectorstore directly."""
        return self.doc_processor.vectorstore
    
    @property
    def device(self):
        """Backward compatibility - access device info."""
        return self.components.device
    
    @property
    def llm(self):
        """Backward compatibility - access LLM directly."""
        return self.components.llm
    
    @property
    def embeddings(self):
        """Backward compatibility - access embeddings directly."""
        return self.components.embeddings
    
    @property
    def converter(self):
        """Backward compatibility - access converter directly."""
        return self.components.converter
    
    @property
    def text_splitter(self):
        """Backward compatibility - access text splitter directly."""
        return self.components.text_splitter
    
    @property
    def prompt_template(self):
        """Backward compatibility - access prompt template directly."""
        return self.components.prompt_template
