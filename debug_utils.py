"""
Combined debugging and testing utilities for the Docling + Ollama pipeline
"""
import lancedb
import pandas as pd
from sentence_transformers import SentenceTransformer
import os
from pathlib import Path

def check_database_status():
    """Check the current status of the LanceDB database"""
    print("📊 Database Status Check")
    print("=" * 50)
    
    try:
        # Connect to the database
        db = lancedb.connect("data/lancedb")
        table = db.open_table("docling")

        # Get basic info about the table
        print(f"✅ Database connected successfully")
        print(f"Total rows: {len(table)}")

        # Show the first few rows to see what data we have
        df = table.to_pandas()
        print(f"\n🔍 Table Schema:")
        print(f"Columns: {list(df.columns)}")

        print(f"\n📖 Sample Data:")
        for i, row in df.head(3).iterrows():
            print(f"\n--- Row {i+1} ---")
            print(f"Filename: {row.get('filename', 'Unknown')}")
            print(f"Title: {row.get('title', 'No title')}")
            print(f"Page numbers: {row.get('page_numbers', 'No pages')}")
            print(f"Text preview: {str(row.get('text', ''))[:200]}...")

        print(f"\n📚 Unique Documents:")
        unique_files = df['filename'].unique()
        for filename in unique_files:
            count = len(df[df['filename'] == filename])
            print(f"- {filename}: {count} chunks")
            
        return True
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

def test_search_queries():
    """Test search functionality with various queries"""
    print("\n🔍 Search Query Testing")
    print("=" * 50)
    
    try:
        # Connect to the database
        db = lancedb.connect("data/lancedb")
        table = db.open_table("docling")
        model = SentenceTransformer('BAAI/bge-small-en-v1.5')

        # Test queries
        queries = [
            "what is docling?",
            "how much data is there?", 
            "what are the main features?",
            "AI models",
            "PDF processing",
            "table recognition"
        ]

        for query in queries:
            print(f"\n🔍 Query: '{query}'")
            print("-" * 30)
            
            query_embedding = model.encode(query)
            results = table.search(query_embedding).limit(2).to_pandas()
            
            for i, row in results.iterrows():
                print(f"\n📄 Result {i+1}:")
                print(f"Filename: {row['filename']}")
                print(f"Title: {row['title']}")
                print(f"Pages: {row['page_numbers']}")
                print(f"Text: {row['text'][:200]}...")
                
        return True
        
    except Exception as e:
        print(f"❌ Search test error: {e}")
        return False

def visualize_embeddings():
    """Visualize embedding distribution and similarity"""
    print("\n📊 Embedding Visualization")
    print("=" * 50)
    
    try:
        import numpy as np
        
        # Connect to the database
        db = lancedb.connect("data/lancedb")
        table = db.open_table("docling")
        df = table.to_pandas()
        
        # Get embeddings
        embeddings = np.stack(df['vector'].values)
        
        print(f"✅ Loaded {len(embeddings)} embeddings")
        print(f"Embedding dimension: {embeddings.shape[1]}")
        print(f"Embedding stats:")
        print(f"  Mean: {embeddings.mean():.4f}")
        print(f"  Std: {embeddings.std():.4f}")
        print(f"  Min: {embeddings.min():.4f}")
        print(f"  Max: {embeddings.max():.4f}")
        
        # Calculate similarity matrix for first 5 chunks
        if len(embeddings) >= 5:
            sample_embeddings = embeddings[:5]
            similarity_matrix = np.dot(sample_embeddings, sample_embeddings.T)
            
            print(f"\n🔍 Similarity Matrix (first 5 chunks):")
            for i in range(5):
                row_str = " ".join([f"{similarity_matrix[i,j]:.3f}" for j in range(5)])
                print(f"  {row_str}")
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization error: {e}")
        return False

def check_file_structure():
    """Check the file structure and data directory"""
    print("\n📁 File Structure Check")
    print("=" * 50)
    
    # Check data directory
    data_dir = Path("data")
    if data_dir.exists():
        print(f"✅ Data directory exists: {data_dir.absolute()}")
        
        # Check LanceDB directory
        lancedb_dir = data_dir / "lancedb"
        if lancedb_dir.exists():
            print(f"✅ LanceDB directory exists: {lancedb_dir.absolute()}")
            
            # List contents
            contents = list(lancedb_dir.rglob("*"))
            print(f"📁 LanceDB contents ({len(contents)} items):")
            for item in contents[:10]:  # Show first 10 items
                size = ""
                if item.is_file():
                    size_bytes = item.stat().st_size
                    if size_bytes > 1024*1024:
                        size = f" ({size_bytes/(1024*1024):.1f}MB)"
                    elif size_bytes > 1024:
                        size = f" ({size_bytes/1024:.1f}KB)"
                    else:
                        size = f" ({size_bytes}B)"
                print(f"  {'📁' if item.is_dir() else '📄'} {item.name}{size}")
            
            if len(contents) > 10:
                print(f"  ... and {len(contents)-10} more items")
        else:
            print(f"❌ LanceDB directory not found")
    else:
        print(f"❌ Data directory not found")
    
    # Check docs directory
    docs_dir = Path("docs")
    if docs_dir.exists():
        print(f"✅ Docs directory exists: {docs_dir.absolute()}")
        doc_files = list(docs_dir.rglob("*"))
        print(f"📚 Found {len(doc_files)} items in docs/")
    else:
        print(f"ℹ️ Docs directory not found (optional)")

if __name__ == "__main__":
    print("🔧 Docling Pipeline Debugging Utilities")
    print("=" * 60)
    
    # Run all checks
    check_file_structure()
    
    if check_database_status():
        test_search_queries()
        visualize_embeddings()
    
    print("\n✅ Debug check completed!")
