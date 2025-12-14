# ============================================================================
# CONFIG PACKAGE
# ============================================================================
# Purpose:
#   Centralized configuration management
#   Re-exports all configuration classes and functions
# ============================================================================

from .settings import (
    Paths, ModelConfig, SearchConfig, PerformanceConfig,
    FileConfig, LoggingConfig,
    paths, model_config, search_config, performance_config,
    file_config, logging_config,
    ensure_directories, get_gpu_optimized_chunk_size,
    get_gpu_optimized_chunk_overlap, is_text_file,
    is_document_file, is_supported_file
)

__all__ = [
    'Paths', 'ModelConfig', 'SearchConfig', 'PerformanceConfig',
    'FileConfig', 'LoggingConfig',
    'paths', 'model_config', 'search_config', 'performance_config',
    'file_config', 'logging_config',
    'ensure_directories', 'get_gpu_optimized_chunk_size',
    'get_gpu_optimized_chunk_overlap', 'is_text_file',
    'is_document_file', 'is_supported_file'
]
