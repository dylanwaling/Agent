#!/usr/bin/env python3
"""
Test script for LangChain integration
Run this to verify LangChain is working with your data
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

def test_langchain_imports():
    """Test if LangChain packages are properly installed"""
    print("🔍 Testing LangChain Imports")
    print("=" * 40)
    
    try:
        from langchain.schema import Document
        print("✅ langchain.schema imported")
        
        from langchain.vectorstores import FAISS
        print("✅ langchain.vectorstores imported")
        
        from langchain.embeddings import HuggingFaceEmbeddings
        print("✅ langchain.embeddings imported")
        
        from langchain.llms import Ollama
        print("✅ langchain.llms imported")
        
        from langchain.chains import RetrievalQA
        print("✅ langchain.chains imported")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_database_connection():
    """Test connection to existing LanceDB"""
    print("\n💾 Testing Database Connection")
    print("=" * 40)
    
    try:
        import lancedb
        import pandas as pd
        
        # Connect to database
        db = lancedb.connect("data/lancedb")
        table = db.open_table("docling")
        df = table.to_pandas()
        
        print(f"✅ Connected to database")
        print(f"📊 Found {len(df)} chunks")
        
        if not df.empty and 'filename' in df.columns:
            unique_docs = df['filename'].unique()
            print(f"📚 Documents: {len(unique_docs)}")
            for doc in unique_docs:
                doc_chunks = len(df[df['filename'] == doc])
                print(f"   • {doc} ({doc_chunks} chunks)")
        
        return df
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return None

def test_langchain_embeddings():
    """Test LangChain embeddings initialization"""
    print("\n🤖 Testing LangChain Embeddings")
    print("=" * 40)
    
    try:
        from langchain.embeddings import HuggingFaceEmbeddings
        
        embeddings = HuggingFaceEmbeddings(
            model_name='BAAI/bge-small-en-v1.5',
            model_kwargs={'device': 'cpu'}
        )
        
        # Test embedding a simple text
        test_text = "This is a test sentence for embedding."
        embedding = embeddings.embed_query(test_text)
        
        print(f"✅ Embeddings initialized")
        print(f"📏 Embedding dimension: {len(embedding)}")
        
        return embeddings
        
    except Exception as e:
        print(f"❌ Embeddings error: {e}")
        return None

def test_ollama_llm():
    """Test Ollama LLM connection"""
    print("\n🦙 Testing Ollama LLM")
    print("=" * 40)
    
    try:
        from langchain.llms import Ollama
        
        llm = Ollama(
            model="tinyllama",
            temperature=0.1,
            timeout=10
        )
        
        # Test simple query
        response = llm("Hello! Please respond with just 'Hello back!'")
        print(f"✅ Ollama LLM working")
        print(f"📝 Response: {response[:100]}...")
        
        return llm
        
    except Exception as e:
        print(f"❌ Ollama error: {e}")
        return None

def test_document_conversion(df):
    """Test converting LanceDB data to LangChain Documents"""
    print("\n📄 Testing Document Conversion")
    print("=" * 40)
    
    try:
        from langchain.schema import Document
        
        if df is None or df.empty:
            print("❌ No data to convert")
            return []
        
        documents = []
        for _, row in df.head(5).iterrows():  # Test with first 5 rows
            metadata = {
                "filename": row.get("filename", "Unknown"),
                "source": row.get("filename", "Unknown")
            }
            
            doc = Document(
                page_content=row.get("text", ""),
                metadata=metadata
            )
            documents.append(doc)
        
        print(f"✅ Converted {len(documents)} documents")
        print(f"📝 Sample content: {documents[0].page_content[:100]}...")
        
        return documents
        
    except Exception as e:
        print(f"❌ Conversion error: {e}")
        return []

def test_vectorstore_creation(documents, embeddings):
    """Test FAISS vectorstore creation"""
    print("\n🔍 Testing Vectorstore Creation")
    print("=" * 40)
    
    try:
        from langchain.vectorstores import FAISS
        
        if not documents or not embeddings:
            print("❌ Missing documents or embeddings")
            return None
        
        vectorstore = FAISS.from_documents(documents, embeddings)
        print(f"✅ Vectorstore created")
        
        # Test similarity search
        results = vectorstore.similarity_search("test", k=2)
        print(f"🔍 Search test: found {len(results)} results")
        
        return vectorstore
        
    except Exception as e:
        print(f"❌ Vectorstore error: {e}")
        return None

def test_qa_chain(vectorstore, llm):
    """Test RetrievalQA chain"""
    print("\n🔗 Testing QA Chain")
    print("=" * 40)
    
    try:
        from langchain.chains import RetrievalQA
        from langchain.prompts import PromptTemplate
        
        if not vectorstore or not llm:
            print("❌ Missing vectorstore or LLM")
            return None
        
        # Create chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
        
        print(f"✅ QA chain created")
        
        # Test query
        test_query = "What is this document about?"
        result = qa_chain({"query": test_query})
        
        answer = result.get("result", "No answer")
        sources = len(result.get("source_documents", []))
        
        print(f"📝 Test answer: {answer[:100]}...")
        print(f"📚 Sources found: {sources}")
        
        return qa_chain
        
    except Exception as e:
        print(f"❌ QA chain error: {e}")
        return None

def main():
    """Run all tests"""
    print("🧪 LangChain Integration Test Suite")
    print("=" * 50)
    
    # Test imports
    if not test_langchain_imports():
        print("\n❌ LangChain imports failed. Please install: pip install langchain")
        return False
    
    # Test database
    df = test_database_connection()
    if df is None:
        print("\n❌ Database connection failed. Please run the pipeline first (1-3 scripts)")
        return False
    
    # Test embeddings
    embeddings = test_langchain_embeddings()
    if not embeddings:
        return False
    
    # Test LLM
    llm = test_ollama_llm()
    if not llm:
        print("\n❌ Ollama not working. Please start: ollama serve")
        return False
    
    # Test document conversion
    documents = test_document_conversion(df)
    if not documents:
        return False
    
    # Test vectorstore
    vectorstore = test_vectorstore_creation(documents, embeddings)
    if not vectorstore:
        return False
    
    # Test QA chain
    qa_chain = test_qa_chain(vectorstore, llm)
    if not qa_chain:
        return False
    
    print("\n✅ All LangChain tests passed!")
    print("🚀 You can now run: streamlit run 5-chat-langchain.py")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
