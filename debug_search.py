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
        "Invoice_Outline_-_Sheet1_1.pdf",
        "invoice outline",
        "product_manual",
        "product manual",
        "company handbook"
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
            
        # For Invoice PDF, show full content extraction
        if "invoice" in query.lower() and debug_results['total_results'] > 0:
            print(f"\n🔍 Full content extraction test for Invoice PDF:")
            search_results = pipeline.search(query)
            if search_results:
                content = search_results[0]["content"]
                print(f"   Raw content: {content[:200]}...")
                
                # Test content cleaning
                parts = content.split(' ', 2)
                if len(parts) >= 3:
                    clean_content = parts[2]
                    print(f"   Clean content: {clean_content[:200]}...")
                else:
                    print(f"   Content split failed, parts: {len(parts)}")
                    for i, part in enumerate(parts):
                        print(f"     Part {i}: {part[:50]}...")
                        
            expected_doc = query.replace(" ", "_").replace("-", "_") + (".pdf" if "pdf" not in query else "")
            found_doc = debug_results["results"][0]["source"] if debug_results["results"] else "None"
            if expected_doc.lower() in found_doc.lower():
                print(f"  ✅ Expected doc '{found_doc}' found at rank 1")
            else:
                print(f"  ⚠️ Expected doc containing '{expected_doc}' but got '{found_doc}'")
        else:
            expected_doc = query.replace(" ", "_").replace("-", "_") + (".pdf" if "pdf" not in query else "")
            found_doc = debug_results["results"][0]["source"] if debug_results["results"] else "None"
            if expected_doc.lower() in found_doc.lower():
                print(f"  ✅ Expected doc '{found_doc}' found at rank 1")
            else:
                print(f"  ⚠️ Expected doc containing '{expected_doc}' but got '{found_doc}'")

if __name__ == "__main__":
    test_search()
