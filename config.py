#!/usr/bin/env python3
"""
Configuration Management Module
Centralized configuration for the Document Q&A Agent system
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional


# ============================================================================
# DIRECTORY PATHS
# ============================================================================

@dataclass
class Paths:
    """Centralized path configuration"""
    # Base directories
    DATA_DIR: Path = Path("data")
    DOCS_DIR: Path = Path("data/documents")
    INDEX_DIR: Path = Path("data/index")
    
    # Status and logging files
    STATUS_FILE: Path = Path("data/pipeline_status.json")
    HISTORY_FILE: Path = Path("data/operation_history.jsonl")
    
    # Index files
    INDEX_PATH: Path = Path("data/index/faiss_index")
    FAISS_INDEX_FILE: Path = Path("data/index/faiss_index/index.faiss")
    PKL_INDEX_FILE: Path = Path("data/index/faiss_index/index.pkl")


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

@dataclass
class ModelConfig:
    """LLM and embedding model configuration"""
    # Embedding model
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    
    # LLM configuration
    LLM_MODEL: str = "qwen2.5:1.5b"
    LLM_TEMPERATURE: float = 0.3
    LLM_CONTEXT_WINDOW: int = 4096
    LLM_MAX_TOKENS: int = 800
    LLM_STREAMING: bool = True
    
    # Text splitting
    CHUNK_SIZE: int = 1000
    CHUNK_SIZE_GPU_OPTIMIZED: int = 800
    CHUNK_OVERLAP: int = 200
    CHUNK_OVERLAP_GPU_OPTIMIZED: int = 150


# ============================================================================
# SEARCH CONFIGURATION
# ============================================================================

@dataclass
class SearchConfig:
    """Search and retrieval configuration"""
    # FAISS search parameters
    SEARCH_K: int = 100
    DEFAULT_SCORE_THRESHOLD: float = 1.25
    
    # Relevance filtering thresholds
    STRONG_MATCH_THRESHOLD: float = 2.5
    WEAK_MATCH_THRESHOLD: float = 2.0
    STRONG_MATCH_SCORE_BOOST: float = 0.5
    WEAK_MATCH_SCORE_BOOST: float = 0.8
    
    # Context building
    MAX_CONTEXT_SOURCES: int = 8
    MAX_CONTEXT_LENGTH: int = 3500
    SOURCE_PREVIEW_LENGTH: int = 200


# ============================================================================
# PERFORMANCE CONFIGURATION
# ============================================================================

@dataclass
class PerformanceConfig:
    """Performance and resource configuration"""
    # GPU settings
    GPU_MEMORY_THRESHOLD_GB: float = 8.0
    GPU_TEMP_MEMORY_GB: float = 2.0
    
    # File upload
    MAX_FILE_SIZE_MB: int = 16
    
    # Monitoring
    MAX_OPERATIONS_DISPLAY: int = 50
    QUEUE_PROCESS_INTERVAL_MS: int = 50
    STATUS_UPDATE_INTERVAL_MS: int = 1000


# ============================================================================
# FILE TYPE CONFIGURATION
# ============================================================================

@dataclass
class FileConfig:
    """Supported file types and extensions"""
    # Text files (read directly)
    TEXT_EXTENSIONS: tuple = ('.txt', '.md')
    
    # Document files (use Docling)
    DOCUMENT_EXTENSIONS: tuple = ('.pdf', '.docx', '.xlsx')
    
    # All supported extensions
    ALL_EXTENSIONS: tuple = ('.pdf', '.txt', '.md', '.docx', '.xlsx')


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

@dataclass
class LoggingConfig:
    """Logging configuration"""
    # Operation types for categorization
    OPERATION_TYPES = {
        'QUESTION_INPUT': 'question_input',
        'EMBEDDING_QUERY': 'embedding_query',
        'FAISS_SEARCH': 'faiss_search',
        'RELEVANCE_FILTER': 'relevance_filter',
        'CONTEXT_BUILDER': 'context_builder',
        'PROMPT_ASSEMBLY': 'prompt_assembly',
        'LLM_GENERATION': 'llm_generation',
        'RESPONSE_COMPLETE': 'response_complete',
        'RESPONSE_STREAM_START': 'response_stream_start',
        'RESPONSE_STREAM_COMPLETE': 'response_stream_complete',
        'FILE_UPLOAD': 'file_upload',
        'DOCLING_PARSE': 'docling_parse',
        'TEXT_SPLITTING': 'text_splitting',
        'EMBEDDING_GENERATION': 'embedding_generation',
        'FAISS_INDEXING': 'faiss_indexing',
        'INDEX_STORAGE': 'index_storage',
        'DOCUMENT_PROCESSING_START': 'document_processing_start',
        'DOCUMENT_PROCESSING_COMPLETE': 'document_processing_complete',
        'GENERAL': 'general',
        'ERROR': 'error'
    }
    
    # Status types
    STATUS_TYPES = {
        'IDLE': 'IDLE',
        'THINKING': 'THINKING',
        'PROCESSING': 'PROCESSING',
        'ERROR': 'ERROR'
    }


# ============================================================================
# GLOBAL CONFIGURATION INSTANCES
# ============================================================================

# Create global instances for easy access
paths = Paths()
model_config = ModelConfig()
search_config = SearchConfig()
performance_config = PerformanceConfig()
file_config = FileConfig()
logging_config = LoggingConfig()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def ensure_directories():
    """Create all required directories if they don't exist"""
    paths.DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    paths.INDEX_DIR.mkdir(parents=True, exist_ok=True)


def get_gpu_optimized_chunk_size(gpu_optimized: bool) -> int:
    """Get chunk size based on GPU optimization status"""
    return (model_config.CHUNK_SIZE_GPU_OPTIMIZED 
            if gpu_optimized 
            else model_config.CHUNK_SIZE)


def get_gpu_optimized_chunk_overlap(gpu_optimized: bool) -> int:
    """Get chunk overlap based on GPU optimization status"""
    return (model_config.CHUNK_OVERLAP_GPU_OPTIMIZED 
            if gpu_optimized 
            else model_config.CHUNK_OVERLAP)


def is_text_file(file_path: Path) -> bool:
    """Check if file is a text file (can be read directly)"""
    return file_path.suffix.lower() in file_config.TEXT_EXTENSIONS


def is_document_file(file_path: Path) -> bool:
    """Check if file is a document file (requires Docling)"""
    return file_path.suffix.lower() in file_config.DOCUMENT_EXTENSIONS


def is_supported_file(file_path: Path) -> bool:
    """Check if file type is supported"""
    return file_path.suffix.lower() in file_config.ALL_EXTENSIONS
