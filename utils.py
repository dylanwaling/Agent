#!/usr/bin/env python3
"""
Utility Functions Module
Reusable helper functions for the Document Q&A Agent system
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Import configuration
from config import paths, file_config

logger = logging.getLogger(__name__)


# ============================================================================
# FILE I/O UTILITIES
# ============================================================================

def read_text_file(file_path: Path, encoding: str = 'utf-8') -> Optional[str]:
    """
    Read text content from a file.
    
    Args:
        file_path: Path to the text file
        encoding: Text encoding (default: utf-8)
        
    Returns:
        File content as string, or None if error
    """
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return None


def write_json_atomic(file_path: Path, data: Dict[str, Any]) -> bool:
    """
    Write JSON data to file atomically (prevents partial reads).
    
    Args:
        file_path: Target file path
        data: Dictionary to write as JSON
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to temp file first
        timestamp = time.time()
        temp_file = file_path.parent / f"{file_path.stem}_{timestamp}.tmp"
        
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        
        # Atomic rename (Windows-safe)
        try:
            if file_path.exists():
                file_path.unlink()
        except:
            pass
        
        temp_file.rename(file_path)
        return True
        
    except Exception as e:
        logger.error(f"Error writing JSON to {file_path}: {e}")
        return False


def append_jsonl(file_path: Path, data: Dict[str, Any]) -> bool:
    """
    Append a JSON line to a JSONL file.
    
    Args:
        file_path: Target JSONL file path
        data: Dictionary to append as JSON line
        
    Returns:
        True if successful, False otherwise
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
            f.flush()
            os.fsync(f.fileno())
        
        return True
        
    except Exception as e:
        logger.error(f"Error appending to JSONL {file_path}: {e}")
        return False


def read_jsonl(file_path: Path) -> list:
    """
    Read all lines from a JSONL file.
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        List of dictionaries, empty list if error
    """
    data = []
    try:
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
    except Exception as e:
        logger.error(f"Error reading JSONL {file_path}: {e}")
    
    return data


# ============================================================================
# DOCUMENT PROCESSING UTILITIES
# ============================================================================

def extract_clean_content(content: str, source_name: str) -> str:
    """
    Extract clean content by removing filename prefix added during indexing.
    
    Args:
        content: Full content string (may include filename prefix)
        source_name: Source document name
        
    Returns:
        Clean content without prefix
    """
    # The content format is: "filename.ext filename_stem actual_content"
    parts = content.split(' ', 2)  # Split into filename, stem, and content
    
    if len(parts) >= 3:
        return parts[2]  # Get the actual content part
    else:
        return content  # Fallback to full content


def create_searchable_content(filename: str, stem: str, content: str) -> str:
    """
    Create searchable content with filename prefix for better matching.
    
    Args:
        filename: Full filename with extension
        stem: Filename without extension
        content: Actual document content
        
    Returns:
        Searchable content string with filename prefix
    """
    return f"{filename} {stem} {content}"


def normalize_filename(filename: str) -> str:
    """
    Normalize filename for comparison (remove spaces, dashes, dots).
    
    Args:
        filename: Original filename
        
    Returns:
        Normalized filename string
    """
    return filename.lower().replace(" ", "_").replace("-", "_").replace(".", "")


# ============================================================================
# TIME AND FORMATTING UTILITIES
# ============================================================================

def format_timestamp(timestamp: float, format_str: str = '%H:%M:%S') -> str:
    """
    Format Unix timestamp as readable time string.
    
    Args:
        timestamp: Unix timestamp
        format_str: strftime format string
        
    Returns:
        Formatted time string
    """
    return datetime.fromtimestamp(timestamp).strftime(format_str)


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string (e.g., "2.34s" or "1m 23s")
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    
    if minutes < 60:
        return f"{minutes}m {secs}s"
    
    hours = int(minutes // 60)
    minutes = int(minutes % 60)
    return f"{hours}h {minutes}m {secs}s"


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in bytes to human-readable string.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string (e.g., "1.23 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


# ============================================================================
# GPU UTILITIES
# ============================================================================

def get_gpu_info() -> Dict[str, Any]:
    """
    Get GPU information if available.
    
    Returns:
        Dictionary with GPU details or error info
    """
    try:
        import torch
        
        if not torch.cuda.is_available():
            return {
                'available': False,
                'device': 'cpu',
                'message': 'No GPU detected'
            }
        
        gpu_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        total_memory_gb = props.total_memory / (1024**3)
        
        try:
            allocated_memory_gb = torch.cuda.memory_allocated(0) / (1024**3)
        except:
            allocated_memory_gb = 0
        
        return {
            'available': True,
            'device': 'cuda',
            'name': gpu_name,
            'total_memory_gb': total_memory_gb,
            'allocated_memory_gb': allocated_memory_gb,
            'message': f"{gpu_name} ({allocated_memory_gb:.1f}/{total_memory_gb:.1f} GB)"
        }
        
    except ImportError:
        return {
            'available': False,
            'device': 'cpu',
            'message': 'PyTorch not available'
        }


def should_optimize_for_gpu(gpu_memory_gb: float, threshold_gb: float = 8.0) -> bool:
    """
    Determine if GPU memory optimization is needed.
    
    Args:
        gpu_memory_gb: Total GPU memory in GB
        threshold_gb: Memory threshold for optimization
        
    Returns:
        True if optimization needed (memory < threshold)
    """
    return gpu_memory_gb < threshold_gb


# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def is_valid_question(question: str, min_length: int = 3) -> bool:
    """
    Validate if a question string is valid.
    
    Args:
        question: Question string to validate
        min_length: Minimum required length
        
    Returns:
        True if valid question
    """
    return bool(question and question.strip() and len(question.strip()) >= min_length)


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to maximum length with suffix.
    
    Args:
        text: Text to truncate
        max_length: Maximum length (including suffix)
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text with suffix if needed
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


# ============================================================================
# DOCUMENT LISTING UTILITIES
# ============================================================================

def get_document_files(directory: Path = None, extensions: tuple = None) -> list:
    """
    Get list of document files from directory.
    
    Args:
        directory: Directory to scan (default: from config)
        extensions: Tuple of extensions to filter (default: from config)
        
    Returns:
        List of Path objects for matching files
    """
    if directory is None:
        directory = paths.DOCS_DIR
    
    if extensions is None:
        extensions = file_config.ALL_EXTENSIONS
    
    if not directory.exists():
        return []
    
    return [
        f for f in directory.iterdir() 
        if f.is_file() and f.suffix.lower() in extensions
    ]


def count_document_files(directory: Path = None) -> int:
    """
    Count number of document files in directory.
    
    Args:
        directory: Directory to scan (default: from config)
        
    Returns:
        Number of document files
    """
    return len(get_document_files(directory))
