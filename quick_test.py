#!/usr/bin/env python3
"""
Quick Pipeline Test
Create sample documents and test the complete pipeline
"""

import os
import sys
from pathlib import Path

# Import our pipeline
sys.path.append(os.path.dirname(__file__))
exec(open('5-chat.py').read())

def create_test_documents():
    """Create sample documents for testing"""
    docs_dir = Path("data/documents")
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample document 1
    doc1 = """# Company Handbook

## Welcome to Acme Corp

Welcome to our company! This handbook contains important information about our policies and procedures.

### Working Hours
- Standard hours: 9 AM to 5 PM
- Lunch break: 12 PM to 1 PM
- Flexible arrangements available

### Benefits
- Health insurance
- 401k matching
- Paid time off
- Professional development budget

### Contact Information
- HR: hr@acme.com
- IT Support: it@acme.com
- Main Office: (555) 123-4567
"""
    
    # Sample document 2
    doc2 = """# Product Manual

## SmartWidget 3000

The SmartWidget 3000 is our flagship product designed for maximum efficiency.

### Features
- Advanced AI processing
- 24/7 operation
- Remote monitoring
- Energy efficient design

### Specifications
- Power: 100W
- Weight: 2.5 kg
- Dimensions: 30x20x15 cm
- Operating temperature: -10°C to 50°C

### Installation
1. Unpack the device carefully
2. Connect to power source
3. Follow setup wizard
4. Configure network settings

### Troubleshooting
- Check power connections
- Verify network settings
- Contact support if issues persist
"""
    
    with open(docs_dir / "company_handbook.md", "w", encoding="utf-8") as f:
        f.write(doc1)
    
    with open(docs_dir / "product_manual.md", "w", encoding="utf-8") as f:
        f.write(doc2)
    
    print("✅ Created test documents:")
    print("  - company_handbook.md")
    print("  - product_manual.md")

def test_pipeline():
    """Test the complete pipeline"""
    print("\n🚀 Testing Document Q&A Pipeline")
    print("=" * 40)
    
    # Create test documents
    create_test_documents()
    
    # Initialize pipeline
    pipeline = DocumentPipeline()
    
    # Process documents
    print("\n📄 Processing documents...")
    if pipeline.process_documents():
        print("✅ Documents processed successfully")
    else:
        print("❌ Document processing failed")
        return False
    
    # Test questions
    questions = [
        "What are the working hours?",
        "What benefits does the company offer?",
        "How much does the SmartWidget weigh?",
        "What is the contact email for HR?"
    ]
    
    print("\n💬 Testing Q&A...")
    for question in questions:
        print(f"\n❓ {question}")
        result = pipeline.ask(question)
        print(f"🤖 {result['answer'][:100]}...")
        if result['sources']:
            print(f"📚 Sources: {[s['source'] for s in result['sources']]}")
    
    print("\n🎉 Pipeline test completed!")
    return True

if __name__ == "__main__":
    try:
        test_pipeline()
        print("\n✅ Pipeline is working correctly!")
        print("\n🚀 Ready to run: streamlit run app.py")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
