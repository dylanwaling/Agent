#!/usr/bin/env python3
"""
Quick test to verify the prompt template is working
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

# Import our pipeline
import importlib.util
spec = importlib.util.spec_from_file_location("chat_module", "5-chat.py")
chat_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chat_module)
DocumentPipeline = chat_module.DocumentPipeline

def test_prompt():
    pipeline = DocumentPipeline()
    
    # Load existing index if available
    if pipeline.load_index():
        print("✅ Loaded existing index")
        
        # Test the prompt template directly
        print("\n🧪 Testing prompt template:")
        print("Template:", pipeline.prompt_template.template[:100] + "...")
        
        # Test a question
        print("\n❓ Testing question...")
        result = pipeline.ask("What's in Algebra_Operations.pdf?")
        print("🤖 Answer:", result['answer'][:200] + "...")
        print(f"📚 Sources: {len(result.get('sources', []))}")
        
    else:
        print("❌ No index found - run processing first")

if __name__ == "__main__":
    test_prompt()
