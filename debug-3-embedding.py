"""
Debug and testing utilities for Step 3: Embeddings & Database
Run this to test and debug the embedding and LanceDB storage process
"""
import time
import numpy as np
import lancedb
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from sentence_transformers import SentenceTransformer
from utils.tokenizer import OpenAITokenizerWrapper
import pandas as pd

def test_embedding_model():
    """Test the sentence transformer model"""
    print("🤖 Testing Embedding Model")
    print("=" * 40)
    
    try:
        model = SentenceTransformer('BAAI/bge-small-en-v1.5')
        print("✅ SentenceTransformer model loaded successfully")
        
        # Test embedding generation
        test_texts = [
            "This is a test sentence.",
            "Docling is a document processing library.",
            "Machine learning models process text efficiently."
        ]
        
        print("🔄 Generating test embeddings...")
        start_time = time.time()
        embeddings = model.encode(test_texts)
        end_time = time.time()
        
        print(f"✅ Embeddings generated in {end_time-start_time:.3f} seconds")
        print(f"📊 Embedding shape: {embeddings.shape}")
        print(f"📏 Embedding dimension: {embeddings.shape[1]}")
        print(f"🔢 Data type: {embeddings.dtype}")
        
        # Test embedding properties
        print(f"📈 Embedding statistics:")
        print(f"  Mean: {embeddings.mean():.4f}")
        print(f"  Std: {embeddings.std():.4f}")
        print(f"  Min: {embeddings.min():.4f}")
        print(f"  Max: {embeddings.max():.4f}")
        
        # Test similarity
        similarity_01 = np.dot(embeddings[0], embeddings[1])
        similarity_02 = np.dot(embeddings[0], embeddings[2])
        similarity_12 = np.dot(embeddings[1], embeddings[2])
        
        print(f"🔍 Similarity scores:")
        print(f"  Text 1-2: {similarity_01:.4f}")
        print(f"  Text 1-3: {similarity_02:.4f}")
        print(f"  Text 2-3: {similarity_12:.4f}")
        
        return model
        
    except Exception as e:
        print(f"❌ Embedding model error: {e}")
        return None

def test_lancedb_connection():
    """Test LanceDB connection and basic operations"""
    print("\n💾 Testing LanceDB Connection")
    print("=" * 40)
    
    try:
        # Connect to database
        db = lancedb.connect("data/lancedb")
        print("✅ LanceDB connection successful")
        
        # List existing tables
        try:
            tables = db.table_names()
            print(f"📊 Existing tables: {tables}")
        except:
            print("📊 No existing tables found")
        
        # Test creating a simple table
        test_data = [
            {
                "text": "Test document 1",
                "vector": np.random.randn(384).astype(np.float32),
                "filename": "test1.txt",
                "page_numbers": [1],
                "title": "Test Title 1"
            },
            {
                "text": "Test document 2", 
                "vector": np.random.randn(384).astype(np.float32),
                "filename": "test2.txt",
                "page_numbers": [1, 2],
                "title": "Test Title 2"
            }
        ]
        
        print("🔄 Creating test table...")
        test_table = db.create_table("test_table", data=test_data, mode="overwrite")
        print(f"✅ Test table created with {len(test_table)} rows")
        
        # Test querying
        query_vector = np.random.randn(384).astype(np.float32)
        results = test_table.search(query_vector).limit(1).to_pandas()
        print(f"🔍 Test query returned {len(results)} results")
        
        # Clean up test table
        db.drop_table("test_table")
        print("🧹 Test table cleaned up")
        
        return db
        
    except Exception as e:
        print(f"❌ LanceDB error: {e}")
        return None

def test_document_processing_for_embeddings():
    """Test the full document extraction and chunking pipeline"""
    print("\n📄 Testing Document Processing Pipeline")
    print("=" * 40)
    
    try:
        # Extract document
        converter = DocumentConverter()
        print("🔄 Extracting document...")
        result = converter.convert("https://arxiv.org/pdf/2408.09869")
        print("✅ Document extraction completed")
        
        # Chunk document
        tokenizer = OpenAITokenizerWrapper()
        MAX_TOKENS = 8191
        
        chunker = HybridChunker(
            tokenizer=tokenizer,
            max_tokens=MAX_TOKENS,
            merge_peers=True,
        )
        
        print("🔪 Chunking document...")
        chunk_iter = chunker.chunk(dl_doc=result.document)
        chunks = list(chunk_iter)
        print(f"✅ Created {len(chunks)} chunks")
        
        return chunks
        
    except Exception as e:
        print(f"❌ Document processing error: {e}")
        return None

def test_metadata_extraction(chunks):
    """Test metadata extraction from chunks"""
    print("\n📋 Testing Metadata Extraction")
    print("=" * 40)
    
    if not chunks:
        print("❌ No chunks provided")
        return False
    
    try:
        metadata_stats = {
            "chunks_with_filenames": 0,
            "chunks_with_pages": 0,
            "chunks_with_titles": 0,
            "total_unique_pages": set(),
            "total_unique_filenames": set()
        }
        
        sample_metadata = []
        
        for i, chunk in enumerate(chunks):
            # Extract filename
            filename = chunk.meta.origin.filename
            if filename:
                metadata_stats["chunks_with_filenames"] += 1
                metadata_stats["total_unique_filenames"].add(filename)
            
            # Extract page numbers
            page_numbers = [
                page_no
                for page_no in sorted(
                    set(
                        prov.page_no
                        for item in chunk.meta.doc_items
                        for prov in item.prov
                    )
                )
            ] or None
            
            if page_numbers:
                metadata_stats["chunks_with_pages"] += 1
                metadata_stats["total_unique_pages"].update(page_numbers)
            
            # Extract title
            title = chunk.meta.headings[0] if chunk.meta.headings else None
            if title:
                metadata_stats["chunks_with_titles"] += 1
            
            # Collect sample for first few chunks
            if i < 5:
                sample_metadata.append({
                    "chunk_id": i,
                    "filename": filename,
                    "page_numbers": page_numbers,
                    "title": title,
                    "text_preview": chunk.text[:100]
                })
        
        # Print statistics
        print(f"📊 Metadata Statistics:")
        print(f"  Chunks with filenames: {metadata_stats['chunks_with_filenames']}/{len(chunks)}")
        print(f"  Chunks with page numbers: {metadata_stats['chunks_with_pages']}/{len(chunks)}")
        print(f"  Chunks with titles: {metadata_stats['chunks_with_titles']}/{len(chunks)}")
        print(f"  Unique filenames: {len(metadata_stats['total_unique_filenames'])}")
        print(f"  Unique pages: {len(metadata_stats['total_unique_pages'])}")
        
        # Show sample metadata
        print(f"\n📝 Sample Metadata:")
        for meta in sample_metadata:
            print(f"  Chunk {meta['chunk_id']}:")
            print(f"    File: {meta['filename']}")
            print(f"    Pages: {meta['page_numbers']}")
            print(f"    Title: {meta['title']}")
            print(f"    Text: {meta['text_preview']}...")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Metadata extraction error: {e}")
        return False

def test_embedding_generation_performance(chunks, model):
    """Test embedding generation performance and memory usage"""
    print("\n⚡ Testing Embedding Generation Performance")
    print("=" * 40)
    
    if not chunks or not model:
        print("❌ Missing chunks or model")
        return False
    
    try:
        print(f"🔄 Generating embeddings for {len(chunks)} chunks...")
        
        # Batch processing test
        batch_size = 10
        total_time = 0
        
        embeddings = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            texts = [chunk.text for chunk in batch]
            
            start_time = time.time()
            batch_embeddings = model.encode(texts)
            end_time = time.time()
            
            batch_time = end_time - start_time
            total_time += batch_time
            
            embeddings.extend(batch_embeddings)
            
            if i == 0:  # Log first batch details
                print(f"📊 First batch ({len(texts)} chunks): {batch_time:.3f}s")
                print(f"   Average per chunk: {batch_time/len(texts):.3f}s")
        
        print(f"✅ All embeddings generated in {total_time:.2f}s")
        print(f"📈 Average per chunk: {total_time/len(chunks):.3f}s")
        print(f"📏 Embedding shape: {embeddings[0].shape}")
        
        # Memory usage estimate
        total_embeddings = np.array(embeddings)
        memory_mb = total_embeddings.nbytes / (1024 * 1024)
        print(f"💾 Memory usage: {memory_mb:.1f}MB")
        
        return embeddings
        
    except Exception as e:
        print(f"❌ Embedding generation error: {e}")
        return None

def test_database_insertion(chunks, embeddings, model):
    """Test inserting data into LanceDB"""
    print("\n💾 Testing Database Insertion")
    print("=" * 40)
    
    if not chunks or not embeddings:
        print("❌ Missing chunks or embeddings")
        return False
    
    try:
        # Prepare data for LanceDB
        print("🔄 Preparing data for database...")
        processed_chunks = []
        
        for i, chunk in enumerate(chunks):
            # Extract metadata
            page_numbers = [
                page_no
                for page_no in sorted(
                    set(
                        prov.page_no
                        for item in chunk.meta.doc_items
                        for prov in item.prov
                    )
                )
            ] or None
            
            processed_chunks.append({
                "text": chunk.text,
                "vector": embeddings[i],
                "filename": chunk.meta.origin.filename,
                "page_numbers": page_numbers,
                "title": chunk.meta.headings[0] if chunk.meta.headings else None,
            })
        
        print(f"📊 Prepared {len(processed_chunks)} records")
        
        # Create database and table
        print("🔄 Creating database table...")
        db = lancedb.connect("data/lancedb")
        
        start_time = time.time()
        table = db.create_table("docling_test", data=processed_chunks, mode="overwrite")
        end_time = time.time()
        
        print(f"✅ Table created in {end_time-start_time:.2f}s")
        print(f"📊 Table contains {len(table)} rows")
        
        # Test table query
        print("🔍 Testing table query...")
        df = table.to_pandas()
        print(f"✅ Retrieved {len(df)} rows as DataFrame")
        
        # Show schema
        print(f"📋 Table schema: {list(df.columns)}")
        
        # Test search functionality
        print("🔍 Testing vector search...")
        query_embedding = model.encode("What is Docling?")
        search_results = table.search(query_embedding).limit(3).to_pandas()
        print(f"✅ Search returned {len(search_results)} results")
        
        # Clean up test table
        db.drop_table("docling_test")
        print("🧹 Test table cleaned up")
        
        return True
        
    except Exception as e:
        print(f"❌ Database insertion error: {e}")
        return False

def run_full_embedding_test():
    """Run the complete embedding pipeline from 3-embedding.py"""
    print("\n🚀 Running Full Embedding Test")
    print("=" * 40)
    
    try:
        # Mirror the exact logic from 3-embedding.py
        tokenizer = OpenAITokenizerWrapper()
        MAX_TOKENS = 8191
        
        # Extract and chunk document
        converter = DocumentConverter()
        result = converter.convert("https://arxiv.org/pdf/2408.09869")
        
        chunker = HybridChunker(
            tokenizer=tokenizer,
            max_tokens=MAX_TOKENS,
            merge_peers=True,
        )
        
        chunk_iter = chunker.chunk(dl_doc=result.document)
        chunks = list(chunk_iter)
        
        # Create database and embeddings
        db = lancedb.connect("data/lancedb")
        model = SentenceTransformer('BAAI/bge-small-en-v1.5')
        
        print(f"🔄 Processing {len(chunks)} chunks...")
        processed_chunks = []
        for chunk in chunks:
            # Create embedding
            embedding = model.encode(chunk.text)
            
            # Process metadata
            page_numbers = [
                page_no
                for page_no in sorted(
                    set(
                        prov.page_no
                        for item in chunk.meta.doc_items
                        for prov in item.prov
                    )
                )
            ] or None
            
            processed_chunks.append({
                "text": chunk.text,
                "vector": embedding,
                "filename": chunk.meta.origin.filename,
                "page_numbers": page_numbers,
                "title": chunk.meta.headings[0] if chunk.meta.headings else None,
            })
        
        # Create table
        table = db.create_table("docling", data=processed_chunks, mode="overwrite")
        
        print(f"✅ Table created with {len(table)} rows")
        
        # Verify the table
        df = table.to_pandas()
        print("📊 Table verification:")
        print(df.head())
        
        return True
        
    except Exception as e:
        print(f"❌ Full embedding test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Step 3: Embeddings & Database Debug")
    print("=" * 50)
    
    # Run all tests
    model = test_embedding_model()
    db = test_lancedb_connection()
    
    if model and db:
        chunks = test_document_processing_for_embeddings()
        if chunks:
            test_metadata_extraction(chunks)
            embeddings = test_embedding_generation_performance(chunks, model)
            if embeddings:
                test_database_insertion(chunks, embeddings, model)
        
        # Final comprehensive test
        print("\n" + "="*50)
        run_full_embedding_test()
    
    print("\n✅ Embedding debug completed!")
