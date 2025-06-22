#!/usr/bin/env python3
"""
Debug script to inspect what's actually stored in the LanceDB database
"""

import lancedb
import pandas as pd

def inspect_database():
    """Inspect the content of documents in the database"""
    print("🔍 Database Content Inspector")
    print("=" * 50)
    
    try:
        # Connect to database
        db = lancedb.connect("data/lancedb")
        table = db.open_table("docling")
        
        # Get all data
        df = table.to_pandas()
        
        print(f"📊 Total chunks in database: {len(df)}")
        print()
        
        # Show unique documents
        if 'filename' in df.columns:
            unique_docs = df['filename'].unique()
            print(f"📚 Documents in database: {len(unique_docs)}")
            for doc in unique_docs:
                doc_chunks = len(df[df['filename'] == doc])
                print(f"   • {doc} ({doc_chunks} chunks)")
            print()
            
            # Show sample content from each document
            for doc in unique_docs:
                print(f"📄 Sample content from '{doc}':")
                print("-" * 40)
                
                doc_data = df[df['filename'] == doc]
                
                # Show first chunk
                if len(doc_data) > 0:
                    first_chunk = doc_data.iloc[0]
                    text = first_chunk.get('text', 'No text found')
                    
                    # Show first 200 characters
                    preview = text[:200]
                    if len(text) > 200:
                        preview += "..."
                    
                    print(f"Text preview: {preview}")
                    print(f"Text length: {len(text)} characters")
                    
                    # Check if text looks corrupted
                    if len(text) > 50:
                        # Count meaningful characters vs random characters
                        alpha_chars = sum(1 for c in text if c.isalpha())
                        total_chars = len(text.replace(' ', '').replace('\n', ''))
                        
                        if total_chars > 0:
                            alpha_ratio = alpha_chars / total_chars
                            
                            if alpha_ratio < 0.5:
                                print("⚠️  WARNING: Text appears corrupted (low alphabetic character ratio)")
                            else:
                                print("✅ Text appears normal")
                        
                print()
        
    except Exception as e:
        print(f"❌ Error inspecting database: {e}")

if __name__ == "__main__":
    inspect_database()
