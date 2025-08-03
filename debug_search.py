#!/usr/bin/env python3
"""
Debug Document Search
Quick test to see what's happening with document retrieval
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from document_pipeline import DocumentPipeline

def test_search():
    """Test search for missing documents"""
    print("🔍 Debug Document Search")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = DocumentPipeline()
    
    # Try to load existing index first
    if not pipeline.load_index():
        print("⚠️ No existing index found, processing documents...")
        if not pipeline.process_documents():
            print("❌ Failed to process documents")
            return
        print("✅ Documents processed successfully")
    
    # Test queries that are failing
    test_queries = [
        "product manual",  # Should find product_manual.md as #1
        "invoice outline", # Should find Invoice_Outline_-_Sheet1_1.pdf as #1  
        "company handbook", # Should find company_handbook.md as #1
        "algebra operations", # Should find the math PDF as #1
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: '{query}'")
        print("-" * 30)
        
        # Debug search
        debug_results = pipeline.debug_search(query)
        
        if "error" in debug_results:
            print(f"❌ Error: {debug_results['error']}")
            continue
        
        print(f"📊 Found {debug_results['total_results']} results")
        
        # Show top 3 results
        for result in debug_results["results"][:3]:
            print(f"  {result['rank']}. {result['source']} (score: {result['score']:.3f})")
            print(f"     Content: {result['content_preview']}")
            
        # Show if the expected document is in results
        expected_docs = {
            "product manual": "product_manual.md",
            "invoice outline": "Invoice_Outline_-_Sheet1_1.pdf", 
            "company handbook": "company_handbook.md",
            "algebra operations": "tmphp713yna_Math_Review_-_Algebra_Operations.pdf"
        }
        
        if query.lower() in expected_docs:
            expected = expected_docs[query.lower()]
            found_ranks = [i+1 for i, r in enumerate(debug_results["results"]) if r["source"] == expected]
            if found_ranks:
                print(f"  ✅ Expected doc '{expected}' found at rank {found_ranks[0]}")
            else:
                print(f"  ❌ Expected doc '{expected}' NOT FOUND in top 10!")

if __name__ == "__main__":
    test_search()
