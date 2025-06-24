#!/usr/bin/env python3
"""
Simple test to verify LangChain integration is working
"""

def test_langchain_basic():
    """Test basic LangChain functionality"""
    print("🧪 Testing LangChain Integration")
    print("=" * 40)
    
    try:
        # Test modern imports
        from langchain.schema import Document
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_ollama import OllamaLLM
        from langchain.chains import RetrievalQA
        
        print("✅ All LangChain imports successful")
        
        # Test embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name='BAAI/bge-small-en-v1.5'
        )
        
        # Test LLM
        llm = OllamaLLM(model="tinyllama", temperature=0.1)
        
        # Create test documents
        docs = [
            Document(page_content="This is a test document about AI.", metadata={"source": "test1"}),
            Document(page_content="LangChain is a framework for building AI applications.", metadata={"source": "test2"})
        ]
        
        # Create vectorstore
        vectorstore = FAISS.from_documents(docs, embeddings)
        
        # Create QA chain
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )
        
        # Test query
        result = qa.invoke({"query": "What is LangChain?"})
        answer = result.get('result', 'No answer')
        
        print(f"✅ LangChain QA working!")
        print(f"📝 Answer: {answer[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_langchain_basic()
    if success:
        print("\n🚀 LangChain is ready! You can now run:")
        print("   streamlit run 5-chat-langchain.py")
    else:
        print("\n❌ LangChain setup needs attention")
