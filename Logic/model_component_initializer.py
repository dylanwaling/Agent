# ============================================================================
# COMPONENT INITIALIZATION MODULE
# ============================================================================
# Purpose:
#   Initialize all document processing components
#   Handles GPU/CPU detection, embeddings, LLM, and text splitter setup
# ============================================================================

import logging
from pathlib import Path
from typing import Optional

# Third-party imports
from docling.document_converter import DocumentConverter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

# Local imports
from Config.settings import model_config, get_gpu_optimized_chunk_size, get_gpu_optimized_chunk_overlap

logger = logging.getLogger(__name__)


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
        """Initialize all components."""
        self._init_gpu_detection()
        self._init_converter()
        self._init_text_splitter()
        self._init_embeddings()
        self._init_llm()
        self._init_prompt_template()
    
    def _init_gpu_detection(self):
        """Detect GPU availability and optimize for hardware."""
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
                    logger.info("🔧 Optimizing for 6GB GPU...")
                    # Enable memory efficiency for smaller GPUs
                    torch.cuda.empty_cache()
                    self.gpu_optimized = True
                else:
                    self.gpu_optimized = False
            else:
                logger.info("💻 Using CPU (no GPU detected)")
                self.gpu_optimized = False
        except ImportError:
            self.device = "cpu"
            self.gpu_optimized = False
            logger.info("💻 Using CPU (PyTorch not available)")
    
    def _init_converter(self):
        """Initialize document converter (Docling)."""
        self.converter = DocumentConverter()
    
    def _init_text_splitter(self):
        """Initialize text splitter with GPU-optimized settings."""
        chunk_size = get_gpu_optimized_chunk_size(self.gpu_optimized)
        chunk_overlap = get_gpu_optimized_chunk_overlap(self.gpu_optimized)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    
    def _init_embeddings(self):
        """Initialize embeddings model with GPU support."""
        embedding_kwargs = {
            "model_name": model_config.EMBEDDING_MODEL
        }
        
        # Add device specification if GPU is available
        if self.device == "cuda":
            embedding_kwargs["model_kwargs"] = {"device": self.device}
            embedding_kwargs["encode_kwargs"] = {"device": self.device}
        
        self.embeddings = HuggingFaceEmbeddings(**embedding_kwargs)
    
    def _init_llm(self):
        """Initialize LLM optimized for qwen2.5:1.5b."""
        self.llm = OllamaLLM(
            model=self.model_name,
            temperature=model_config.LLM_TEMPERATURE,
            num_ctx=model_config.LLM_CONTEXT_WINDOW,
            num_predict=model_config.LLM_MAX_TOKENS,
            streaming=model_config.LLM_STREAMING,
        )
    
    def _init_prompt_template(self):
        """Initialize enhanced prompt template for thoughtful responses."""
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
