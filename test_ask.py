#!/usr/bin/env python3
"""
Test the ask method directly
"""

from document_pipeline import DocumentPipeline

def test_ask():
    """Test the ask method with problematic queries"""
    print("🧪 Testing Ask Method")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = DocumentPipeline()
    
    # Load existing index or process documents
    if not pipeline.load_index():
        print("⚠️ No existing index found, processing documents...")
        if not pipeline.process_documents():
            print("❌ Failed to process documents")
            return
        print("✅ Documents processed successfully")
    
    # Test queries that were failing
    test_queries = [
        "whats Invoice_Outline_-_Sheet1_1.pdf",
        "whats company_handbook",
        "whats product manual"
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: '{query}'")
        print("-" * 50)
        
        result = pipeline.ask(query)
        
        print(f"🤖 Answer: {result['answer']}")
        print(f"📚 Sources ({len(result['sources'])}):")
        for source in result['sources']:
            print(f"  • {source['source']}: {source['content'][:100]}...")

if __name__ == "__main__":
    test_ask()
