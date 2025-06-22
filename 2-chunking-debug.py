"""
Debug and testing utilities for Step 2: Document Chunking
Run this to test and debug the chunking process
"""
import time
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
from utils.tokenizer import OpenAITokenizerWrapper
import tiktoken

load_dotenv()

def test_tokenizer():
    """Test the custom tokenizer wrapper"""
    print("🔧 Testing Tokenizer")
    print("=" * 40)
    
    try:
        tokenizer = OpenAITokenizerWrapper()
        print("✅ OpenAITokenizerWrapper created successfully")
        
        # Test basic functionality
        test_text = "This is a test sentence to verify tokenization works correctly."
        
        print(f"📝 Test text: '{test_text}'")
        
        # Test tokenization
        tokens = tokenizer.tokenize(test_text)
        print(f"🔤 Tokens ({len(tokens)}): {tokens[:10]}..." if len(tokens) > 10 else f"🔤 Tokens: {tokens}")
        
        # Test vocabulary size
        vocab_size = len(tokenizer)
        print(f"📚 Vocabulary size: {vocab_size:,}")
        
        # Test model max length
        max_length = tokenizer.model_max_length
        print(f"📏 Max token length: {max_length:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Tokenizer error: {e}")
        return False

def test_document_extraction_for_chunking():
    """Test document extraction that will be used for chunking"""
    print("\n📄 Testing Document Extraction for Chunking")
    print("=" * 40)
    
    try:
        converter = DocumentConverter()
        print("🔄 Extracting Docling paper for chunking test...")
        
        start_time = time.time()
        result = converter.convert("https://arxiv.org/pdf/2408.09869")
        end_time = time.time()
        
        if result.document:
            print(f"✅ Document extracted successfully ({end_time-start_time:.2f}s)")
            
            # Check document structure
            markdown_output = result.document.export_to_markdown()
            print(f"📊 Document length: {len(markdown_output):,} characters")
            
            # Analyze document structure
            lines = markdown_output.split('\n')
            non_empty_lines = [line for line in lines if line.strip()]
            print(f"📄 Total lines: {len(lines):,}")
            print(f"📝 Non-empty lines: {len(non_empty_lines):,}")
            
            # Check for headers/structure
            headers = [line for line in non_empty_lines if line.startswith('#')]
            print(f"📋 Headers found: {len(headers)}")
            
            # Show first few headers
            if headers:
                print("📑 Header structure:")
                for i, header in enumerate(headers[:5]):
                    print(f"  {header[:80]}...")
                if len(headers) > 5:
                    print(f"  ... and {len(headers)-5} more headers")
            
            return result.document
        else:
            print("❌ Document extraction failed")
            return None
            
    except Exception as e:
        print(f"❌ Document extraction error: {e}")
        return None

def test_chunker_creation():
    """Test HybridChunker creation and configuration"""
    print("\n🔪 Testing Chunker Creation")
    print("=" * 40)
    
    try:
        tokenizer = OpenAITokenizerWrapper()
        MAX_TOKENS = 8191
        
        chunker = HybridChunker(
            tokenizer=tokenizer,
            max_tokens=MAX_TOKENS,
            merge_peers=True,
        )
        
        print("✅ HybridChunker created successfully")
        print(f"🔢 Max tokens: {MAX_TOKENS:,}")
        print(f"🤝 Merge peers: True")
        
        return chunker
        
    except Exception as e:
        print(f"❌ Chunker creation error: {e}")
        return None

def analyze_chunks(chunks):
    """Analyze the properties of generated chunks"""
    print(f"\n📊 Chunk Analysis ({len(chunks)} chunks)")
    print("=" * 40)
    
    if not chunks:
        print("❌ No chunks to analyze")
        return
    
    # Basic statistics
    chunk_lengths = [len(chunk.text) for chunk in chunks]
    avg_length = sum(chunk_lengths) / len(chunk_lengths)
    min_length = min(chunk_lengths)
    max_length = max(chunk_lengths)
    
    print(f"📏 Chunk lengths:")
    print(f"  Average: {avg_length:.0f} characters")
    print(f"  Min: {min_length:,} characters") 
    print(f"  Max: {max_length:,} characters")
    
    # Token count analysis (approximate)
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        token_counts = [len(encoding.encode(chunk.text)) for chunk in chunks[:5]]  # Sample first 5
        avg_tokens = sum(token_counts) / len(token_counts)
        print(f"🔤 Average tokens (sample): {avg_tokens:.0f}")
    except:
        print("🔤 Token count analysis skipped")
    
    # Metadata analysis
    chunks_with_headings = sum(1 for chunk in chunks if chunk.meta.headings)
    chunks_with_pages = sum(1 for chunk in chunks if any(
        prov.page_no for item in chunk.meta.doc_items for prov in item.prov
    ))
    
    print(f"📋 Metadata:")
    print(f"  Chunks with headings: {chunks_with_headings}/{len(chunks)}")
    print(f"  Chunks with page info: {chunks_with_pages}/{len(chunks)}")
    
    # Show sample chunks
    print(f"\n📝 Sample Chunks:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Length: {len(chunk.text)} chars")
        print(f"Heading: {chunk.meta.headings[0] if chunk.meta.headings else 'None'}")
        
        # Get page numbers
        page_numbers = sorted(set(
            prov.page_no
            for item in chunk.meta.doc_items
            for prov in item.prov
        ))
        print(f"Pages: {page_numbers if page_numbers else 'None'}")
        print(f"Text preview: {chunk.text[:150]}...")

def test_chunking_performance():
    """Test chunking performance with timing"""
    print("\n⏱️ Testing Chunking Performance")
    print("=" * 40)
    
    document = test_document_extraction_for_chunking()
    if not document:
        return False
    
    chunker = test_chunker_creation()
    if not chunker:
        return False
    
    try:
        print("🔄 Starting chunking process...")
        start_time = time.time()
        
        chunk_iter = chunker.chunk(dl_doc=document)
        chunks = list(chunk_iter)
        
        end_time = time.time()
        
        print(f"✅ Chunking completed in {end_time-start_time:.2f} seconds")
        print(f"📊 Created {len(chunks)} chunks")
        
        if chunks:
            analyze_chunks(chunks)
            return chunks
        else:
            print("❌ No chunks created")
            return None
            
    except Exception as e:
        print(f"❌ Chunking error: {e}")
        return None

def test_chunk_content_quality():
    """Test the quality and coherence of chunk content"""
    print("\n🎯 Testing Chunk Content Quality")
    print("=" * 40)
    
    chunks = test_chunking_performance()
    if not chunks:
        return False
    
    # Test for content coherence
    coherent_chunks = 0
    fragments = 0
    
    for i, chunk in enumerate(chunks):
        text = chunk.text.strip()
        
        # Check if chunk seems coherent (not just fragments)
        word_count = len(text.split())
        sentence_count = text.count('.') + text.count('!') + text.count('?')
        
        if word_count >= 10 and sentence_count >= 1:
            coherent_chunks += 1
        else:
            fragments += 1
            if i < 3:  # Show first few fragments
                print(f"⚠️ Fragment found: {text[:100]}...")
    
    print(f"📊 Content Quality:")
    print(f"  Coherent chunks: {coherent_chunks}/{len(chunks)} ({coherent_chunks/len(chunks)*100:.1f}%)")
    print(f"  Fragments: {fragments}/{len(chunks)} ({fragments/len(chunks)*100:.1f}%)")
    
    # Check for overlapping content
    if len(chunks) >= 2:
        first_words_0 = chunks[0].text.split()[:20]
        first_words_1 = chunks[1].text.split()[:20]
        overlap = set(first_words_0) & set(first_words_1)
        print(f"🔄 Adjacent chunk overlap: {len(overlap)} words")
    
    return True

def run_full_chunking_test():
    """Run the complete chunking pipeline from 2-chunking.py"""
    print("\n🚀 Running Full Chunking Test")
    print("=" * 40)
    
    try:
        # Mirror the exact logic from 2-chunking.py
        tokenizer = OpenAITokenizerWrapper()
        MAX_TOKENS = 8191
        
        converter = DocumentConverter()
        print("📄 Extracting document for chunking...")
        result = converter.convert("https://arxiv.org/pdf/2408.09869")
        print("✅ Document extraction completed")
        
        chunker = HybridChunker(
            tokenizer=tokenizer,
            max_tokens=MAX_TOKENS,
            merge_peers=True,
        )
        
        print("🔪 Starting document chunking...")
        chunk_iter = chunker.chunk(dl_doc=result.document)
        chunks = list(chunk_iter)
        
        print(f"✅ Chunking completed! Created {len(chunks)} chunks")
        
        if chunks:
            avg_size = sum(len(chunk.text) for chunk in chunks) // len(chunks)
            print(f"📊 Average chunk size: {avg_size} characters")
            
            print(f"\n📝 First chunk preview:")
            print(f"Text: {chunks[0].text[:200]}...")
            print(f"Length: {len(chunks[0].text)} characters")
        
        return True
        
    except Exception as e:
        print(f"❌ Full chunking test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Step 2: Document Chunking Debug")
    print("=" * 50)
    
    # Run all tests
    if test_tokenizer():
        test_chunk_content_quality()
        
        # Final comprehensive test
        print("\n" + "="*50)
        run_full_chunking_test()
    
    print("\n✅ Chunking debug completed!")
