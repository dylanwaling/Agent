# ============================================================================
# COMPONENT INITIALIZATION MODULE  
# ============================================================================
# Purpose:
#   Initialize all document processing components
#   Handles GPU/CPU detection, embeddings, LLM, and text splitter setup
# ============================================================================

import os
import logging
from pathlib import Path
from typing import Optional

# Force CPU mode to avoid meta tensor issues
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Additional PyTorch environment setup for stability
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['TORCH_USE_CUDA_DSA'] = '0'

# Third-party imports
from docling.document_converter import DocumentConverter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

# Local imports
from config.settings import model_config, get_gpu_optimized_chunk_size, get_gpu_optimized_chunk_overlap

logger = logging.getLogger(__name__)


def clean_pytorch_state():
    """Clean PyTorch state to prevent meta tensor issues"""
    try:
        import torch
        import gc
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Force garbage collection
        gc.collect()
        
        # Clear any lingering model state
        if hasattr(torch, '_C') and hasattr(torch._C, '_clear_cache'):
            torch._C._clear_cache()
            
    except Exception as e:
        logger.warning(f"Could not clean PyTorch state: {e}")


class ComponentInitializer:
    """
    Initialize and manage document processing components.
    
    Handles:
    - GPU/CPU detection and optimization
    - Document converter (Docling)
    - Text splitter for chunking
    - Embeddings model with hardware acceleration
    - LLM for question answering
    - Prompt template
    """
    
    # Class-level cache to prevent multiple initializations
    _shared_embeddings = None
    _shared_converter = None
    _initialization_lock = False
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize component initializer.
        
        Args:
            model_name: LLM model name (defaults to config value)
        """
        self.model_name = model_name or model_config.LLM_MODEL
        self.device = "cpu"
        self.gpu_optimized = False
        
        # Components (initialized in init_all)
        self.converter = None
        self.text_splitter = None
        self.embeddings = None
        self.llm = None
        self.prompt_template = None
    
    def init_all(self):
        """Initialize all components - force CPU mode to avoid meta tensor issues."""
        
        # Clean PyTorch state first to prevent meta tensor issues
        clean_pytorch_state()
        
        # Force CPU mode completely due to meta tensor issues
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable CUDA completely
        
        # Set device to CPU and disable GPU optimization
        self.device = "cpu"
        self.gpu_optimized = False
        logger.info("💻 Using CPU mode (forced to avoid meta tensor issues)")
        
        # Document converter (Docling) - use shared instance
        if ComponentInitializer._shared_converter is not None:
            logger.info("🔄 Using shared converter instance")
            self.converter = ComponentInitializer._shared_converter
        else:
            try:
                # Set environment variables to force CPU mode for Docling
                import os
                os.environ['DOCLING_DEVICE'] = 'cpu'
                os.environ['CUDA_VISIBLE_DEVICES'] = '' 
                
                converter = DocumentConverter()
                ComponentInitializer._shared_converter = converter
                self.converter = converter
                logger.info("✅ Docling converter initialized in CPU mode")
            except Exception as e:
                logger.error(f"❌ Docling converter initialization failed: {e}")
                raise
        
        # Text splitter - optimized for GPU memory
        gpu_optimized = hasattr(self, 'gpu_optimized') and self.gpu_optimized
        chunk_size = get_gpu_optimized_chunk_size(gpu_optimized)
        chunk_overlap = get_gpu_optimized_chunk_overlap(gpu_optimized)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        # Embeddings - use shared instance to prevent multiple initializations
        if ComponentInitializer._shared_embeddings is not None:
            logger.info("🔄 Using shared embeddings instance")
            self.embeddings = ComponentInitializer._shared_embeddings
        else:
            logger.info("🔧 Initializing embeddings (CPU only)")
            
            # Prevent concurrent initialization
            if ComponentInitializer._initialization_lock:
                logger.info("⏳ Waiting for concurrent initialization to complete...")
                import time
                while ComponentInitializer._initialization_lock:
                    time.sleep(0.1)
                if ComponentInitializer._shared_embeddings is not None:
                    self.embeddings = ComponentInitializer._shared_embeddings
                    logger.info("✅ Used embeddings from concurrent initialization")
                    # Continue with LLM and prompt template initialization
            
            ComponentInitializer._initialization_lock = True
            
            try:
                # Clear any existing PyTorch cache to avoid meta tensor issues
                import torch
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # Multiple attempts with different strategies
                embedding_kwargs = {
                    "model_name": model_config.EMBEDDING_MODEL,
                    "model_kwargs": {"device": "cpu"},
                    "encode_kwargs": {"device": "cpu"}
                }
                
                # Try multiple initialization strategies
                max_attempts = 3
                for attempt in range(max_attempts):
                    try:
                        logger.info(f"🔄 Embeddings initialization attempt {attempt + 1}/{max_attempts}")
                        
                        # Clear cache before each attempt
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        
                        # Force clean state
                        if attempt > 0:
                            import gc
                            gc.collect()
                            
                        # Initialize embeddings
                        embeddings = HuggingFaceEmbeddings(**embedding_kwargs)
                        
                        # Cache for reuse
                        ComponentInitializer._shared_embeddings = embeddings
                        self.embeddings = embeddings
                        logger.info("✅ Embeddings initialized successfully on CPU")
                        break
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Embeddings attempt {attempt + 1} failed: {str(e)[:100]}...")
                        if attempt == max_attempts - 1:
                            logger.error("❌ All embedding initialization attempts failed")
                            raise e
                        else:
                            # Wait and try again with even more aggressive cleanup
                            import time
                            time.sleep(1)
                            torch.cuda.empty_cache() if torch.cuda.is_available() else None
                            
            finally:
                ComponentInitializer._initialization_lock = False
        
        # LLM optimized for qwen2.5:1.5b - excellent reasoning performance
        self.llm = OllamaLLM(
            model=self.model_name,
            temperature=model_config.LLM_TEMPERATURE,
            num_ctx=model_config.LLM_CONTEXT_WINDOW,
            num_predict=model_config.LLM_MAX_TOKENS,
            streaming=model_config.LLM_STREAMING,
        )
        
        # Enhanced prompt template for thoughtful responses
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