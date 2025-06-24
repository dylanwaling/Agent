"""
Debug and testing utilities for Step 5: Chat Interface
Run this to test and debug the chat functionality with Ollama
"""
import time
import streamlit as st
import lancedb
import ollama
from sentence_transformers import SentenceTransformer
import requests
import json

def test_ollama_connection():
    """Test connection to Ollama service"""
    print("🤖 Testing Ollama Connection")
    print("=" * 40)
    
    try:
        # Test if Ollama service is running
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        
        if response.status_code == 200:
            print("✅ Ollama service is running")
            
            # List available models
            models_data = response.json()
            models = models_data.get('models', [])
            
            print(f"📊 Available models: {len(models)}")
            for model in models:
                name = model.get('name', 'Unknown')
                size = model.get('size', 0)
                size_gb = size / (1024**3) if size > 0 else 0
                print(f"  - {name} ({size_gb:.1f}GB)")
            
            # Check for specific models
            model_names = [model.get('name', '') for model in models]
            
            if any('llama3' in name for name in model_names):
                print("✅ Llama3 model available")
            else:
                print("⚠️ Llama3 model not found")
            
            if any('tinyllama' in name for name in model_names):
                print("✅ TinyLlama model available")
            else:
                print("⚠️ TinyLlama model not found")
            
            return True
        else:
            print(f"❌ Ollama service error: Status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama - service may not be running")
        print("💡 Start Ollama with: ollama serve")
        return False
    except Exception as e:
        print(f"❌ Ollama connection error: {e}")
        return False

def test_ollama_chat_api():
    """Test Ollama chat API functionality"""
    print("\n💬 Testing Ollama Chat API")
    print("=" * 40)
    
    try:
        # Test with TinyLlama (smaller, faster)
        test_models = ["tinyllama", "llama3:latest"]
        
        for model_name in test_models:
            print(f"\n🔄 Testing model: {model_name}")
            
            try:
                # Simple test message
                messages = [
                    {"role": "user", "content": "Hello! Can you respond with just 'Hello back!' please?"}
                ]
                
                start_time = time.time()
                response = ollama.chat(model=model_name, messages=messages, stream=False)
                end_time = time.time()
                
                if response and 'message' in response:
                    content = response['message'].get('content', '')
                    print(f"✅ Model {model_name} responded in {end_time-start_time:.2f}s")
                    print(f"📝 Response: {content[:100]}...")
                    return model_name  # Return working model
                else:
                    print(f"❌ No valid response from {model_name}")
                    
            except Exception as e:
                print(f"❌ Error with {model_name}: {e}")
        
        print("❌ No working models found")
        return None
        
    except Exception as e:
        print(f"❌ Chat API test error: {e}")
        return None

def test_database_for_chat():
    """Test database connection for chat functionality"""
    print("\n💾 Testing Database for Chat")
    print("=" * 40)
    
    try:
        # Test database connection
        db = lancedb.connect("data/lancedb")
        table = db.open_table("docling")
        
        print("✅ Database connection successful")
        print(f"📊 Table contains {len(table)} records")
        
        # Test embedding model
        model = SentenceTransformer('BAAI/bge-small-en-v1.5')
        print("✅ Embedding model loaded")
        
        # Test search functionality
        test_query = "What is Docling?"
        query_embedding = model.encode(test_query)
        results = table.search(query_embedding).limit(3).to_pandas()
        
        print(f"🔍 Test search returned {len(results)} results")
        
        return table, model
        
    except Exception as e:
        print(f"❌ Database test error: {e}")
        return None, None

def test_context_retrieval(table, model):
    """Test context retrieval functionality"""
    print("\n📋 Testing Context Retrieval")
    print("=" * 40)
    
    if not table or not model:
        print("❌ Missing table or model")
        return False
    
    try:
        # Test queries that should return good context
        test_queries = [
            "What is Docling?",
            "How does document processing work?",
            "What are the main features?"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: '{query}'")
            
            # Simulate the get_context function
            query_embedding = model.encode(query)
            results = table.search(query_embedding).limit(5).to_pandas()
            
            contexts = []
            for _, row in results.iterrows():
                filename = row.get("filename")
                page_numbers = row.get("page_numbers")
                title = row.get("title")
                
                # Build source citation
                source_parts = []
                if filename:
                    source_parts.append(filename)
                
                # Handle page_numbers safely
                if page_numbers is not None:
                    try:
                        page_list = list(page_numbers) if not isinstance(page_numbers, list) else page_numbers
                        if page_list:
                            source_parts.append(f"p. {', '.join(str(p) for p in page_list)}")
                    except (TypeError, ValueError):
                        pass
                
                source = f"\nSource: {' - '.join(source_parts)}" if source_parts else ""
                if title:
                    source += f"\nTitle: {title}"
                
                contexts.append(f"{row['text']}{source}")
            
            context = "\n\n".join(contexts)
            
            print(f"📊 Context length: {len(context)} characters")
            print(f"📝 Context preview: {context[:200]}...")
            
            if len(context) > 100:
                print("✅ Good context retrieved")
            else:
                print("⚠️ Limited context retrieved")
        
        return True
        
    except Exception as e:
        print(f"❌ Context retrieval error: {e}")
        return False

def test_chat_response_generation(working_model):
    """Test chat response generation with context"""
    print("\n🎯 Testing Chat Response Generation")
    print("=" * 40)
    
    if not working_model:
        print("❌ No working model available")
        return False
    
    try:
        # Simulate a chat scenario
        context = """
        Docling is a document processing library that converts various document formats 
        into a unified format. It uses AI models for layout analysis and table structure 
        recognition. The library supports PDF, DOCX, HTML and other formats.
        
        Source: 2408.09869v5.pdf - p. 1
        Title: Docling Technical Report
        """
        
        user_query = "What is Docling?"
        
        system_prompt = f"""You are a helpful assistant that answers questions based on the provided context.
        Use only the information from the context to answer questions. If you're unsure or the context
        doesn't contain the relevant information, say so.
        
        Context:
        {context}
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
        
        print(f"🔄 Generating response with {working_model}...")
        print(f"📝 User query: '{user_query}'")
        
        start_time = time.time()
        response = ollama.chat(
            model=working_model,
            messages=messages,
            stream=False
        )
        end_time = time.time()
        
        if response and 'message' in response:
            content = response['message'].get('content', '')
            print(f"✅ Response generated in {end_time-start_time:.2f}s")
            print(f"📝 Response length: {len(content)} characters")
            print(f"💬 Response: {content}")
            
            # Check response quality
            if "docling" in content.lower():
                print("✅ Response mentions Docling")
            else:
                print("⚠️ Response doesn't mention Docling")
            
            if len(content) > 20:
                print("✅ Response has good length")
            else:
                print("⚠️ Response is very short")
            
            return True
        else:
            print("❌ No valid response generated")
            return False
            
    except Exception as e:
        print(f"❌ Response generation error: {e}")
        return False

def test_streaming_response(working_model):
    """Test streaming response functionality"""
    print("\n🌊 Testing Streaming Response")
    print("=" * 40)
    
    if not working_model:
        print("❌ No working model available")
        return False
    
    try:
        messages = [
            {"role": "user", "content": "Please count from 1 to 5, one number per line."}
        ]
        
        print(f"🔄 Testing streaming with {working_model}...")
        
        start_time = time.time()
        response_stream = ollama.chat(
            model=working_model,
            messages=messages,
            stream=True
        )
        
        full_response = ""
        chunk_count = 0
        
        for chunk in response_stream:
            if 'message' in chunk and 'content' in chunk['message']:
                content = chunk['message']['content']
                full_response += content
                chunk_count += 1
                
                if chunk_count <= 5:  # Show first few chunks
                    print(f"  Chunk {chunk_count}: '{content}'")
        
        end_time = time.time()
        
        print(f"✅ Streaming completed in {end_time-start_time:.2f}s")
        print(f"📊 Total chunks: {chunk_count}")
        print(f"📝 Full response: {full_response}")
        
        return True
        
    except Exception as e:
        print(f"❌ Streaming error: {e}")
        return False

def test_error_handling():
    """Test error handling scenarios"""
    print("\n🛡️ Testing Error Handling")
    print("=" * 40)
    
    try:
        # Test with non-existent model
        print("🔄 Testing with non-existent model...")
        try:
            response = ollama.chat(
                model="non-existent-model",
                messages=[{"role": "user", "content": "test"}],
                stream=False
            )
            print("⚠️ Expected error but got response")
        except Exception as e:
            print(f"✅ Correctly handled non-existent model: {type(e).__name__}")
        
        # Test with malformed messages
        print("🔄 Testing with malformed messages...")
        try:
            response = ollama.chat(
                model="tinyllama",
                messages=[{"invalid": "structure"}],
                stream=False
            )
            print("⚠️ Expected error but got response")
        except Exception as e:
            print(f"✅ Correctly handled malformed messages: {type(e).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def run_full_chat_simulation():
    """Simulate the full chat interface workflow"""
    print("\n🚀 Running Full Chat Simulation")
    print("=" * 40)
    
    try:
        # Check all components
        if not test_ollama_connection():
            return False
        
        working_model = test_ollama_chat_api()
        if not working_model:
            return False
        
        table, model = test_database_for_chat()
        if not table or not model:
            return False
        
        # Simulate full chat workflow
        print("\n🎭 Simulating chat interaction...")
        
        # User asks a question
        user_question = "What is Docling and how does it work?"
        print(f"👤 User: {user_question}")
        
        # Get context from database
        query_embedding = model.encode(user_question)
        results = table.search(query_embedding).limit(3).to_pandas()
        
        # Build context
        contexts = []
        for _, row in results.iterrows():
            contexts.append(row['text'])
        context = "\n\n".join(contexts)
        
        # Generate response
        system_prompt = f"""You are a helpful assistant. Answer based on this context:
        
        {context[:1000]}...
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question}
        ]
        
        response = ollama.chat(
            model=working_model,
            messages=messages,
            stream=False
        )
        
        if response and 'message' in response:
            bot_response = response['message'].get('content', '')
            print(f"🤖 Bot: {bot_response}")
            print("✅ Full chat simulation successful!")
            return True
        else:
            print("❌ Chat simulation failed")
            return False
            
    except Exception as e:
        print(f"❌ Full chat simulation error: {e}")
        return False

def test_langchain_components():
    """Test LangChain components initialization"""
    print("\n🦜 Testing LangChain Components")
    print("=" * 40)
    
    try:
        # Test LangChain imports
        from langchain_core.documents import Document
        from langchain_community.vectorstores import LanceDB as LangChainLanceDB
        from langchain_ollama import OllamaLLM
        from langchain.chains import RetrievalQA
        from langchain_community.embeddings import SentenceTransformerEmbeddings
        print("✅ LangChain imports successful")
        
        # Test Ollama LLM initialization
        try:
            llm = OllamaLLM(
                model="tinyllama",
                base_url="http://localhost:11434",
                temperature=0.1,
                num_predict=100,  # Small for testing
            )
            print("✅ OllamaLLM initialization successful")
            
            # Test simple generation
            test_response = llm.invoke("Hello, respond with just 'Hi there!'")
            print(f"✅ LLM test response: {test_response[:50]}...")
            
        except Exception as e:
            print(f"❌ OllamaLLM error: {e}")
            return False
        
        # Test embeddings
        try:
            embeddings = SentenceTransformerEmbeddings(model_name='BAAI/bge-small-en-v1.5')
            test_embedding = embeddings.embed_query("test text")
            print(f"✅ Embeddings working, dimension: {len(test_embedding)}")
            
        except Exception as e:
            print(f"❌ Embeddings error: {e}")
            return False
        
        return True
        
    except ImportError as e:
        print(f"❌ LangChain import error: {e}")
        return False
    except Exception as e:
        print(f"❌ LangChain component error: {e}")
        return False


def test_langchain_vectorstore():
    """Test LangChain vector store creation and retrieval"""
    print("\n🗃️ Testing LangChain Vector Store")
    print("=" * 40)
    
    try:
        # Check if original LanceDB exists
        import lancedb
        db = lancedb.connect("data/lancedb")
        table = db.open_table("docling")
        df = table.to_pandas()
        
        if df.empty:
            print("❌ No data in original LanceDB - run steps 1-3 first")
            return False
        
        print(f"✅ Found {len(df)} chunks in original database")
        
        # Test document conversion
        from langchain_core.documents import Document
        
        documents = []
        for _, row in df.head(5).iterrows():  # Test with first 5 for speed
            doc = Document(
                page_content=row.get('text', ''),
                metadata={
                    'filename': row.get('filename', ''),
                    'page_numbers': row.get('page_numbers', []),
                    'title': row.get('title', ''),
                    'source': f"{row.get('filename', 'Unknown')} - {row.get('title', 'No title')}"
                }
            )
            documents.append(doc)
        
        print(f"✅ Converted {len(documents)} documents for testing")
        
        # Test vector store creation
        from langchain_community.vectorstores import LanceDB as LangChainLanceDB
        from langchain_community.embeddings import SentenceTransformerEmbeddings
        
        embeddings = SentenceTransformerEmbeddings(model_name='BAAI/bge-small-en-v1.5')
        
        vectorstore = LangChainLanceDB.from_documents(
            documents=documents,
            embedding=embeddings,
            uri="data/test_langchain_db",  # Test database
        )
        
        print("✅ LangChain vector store created successfully")
        
        # Test retrieval
        test_query = "what is docling?"
        retrieved_docs = vectorstore.similarity_search(test_query, k=3)
        
        print(f"✅ Retrieved {len(retrieved_docs)} documents for test query")
        for i, doc in enumerate(retrieved_docs):
            filename = doc.metadata.get('filename', 'Unknown')
            preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            print(f"  {i+1}. {filename}: {preview}")
        
        return True
        
    except Exception as e:
        print(f"❌ Vector store test error: {e}")
        return False


def test_langchain_qa_chain():
    """Test full LangChain QA chain"""
    print("\n🔗 Testing LangChain QA Chain")
    print("=" * 40)
    
    try:
        # Initialize components (reuse test vector store if it exists)
        from langchain_ollama import OllamaLLM
        from langchain.chains import RetrievalQA
        from langchain_community.vectorstores import LanceDB as LangChainLanceDB
        from langchain_community.embeddings import SentenceTransformerEmbeddings
        from langchain_core.prompts import PromptTemplate
        
        # Check if test database exists
        import os
        if not os.path.exists("data/test_langchain_db"):
            print("❌ Test vector store not found - run test_langchain_vectorstore first")
            return False
        
        # Initialize components
        llm = OllamaLLM(
            model="tinyllama",
            base_url="http://localhost:11434",
            temperature=0.1,
            num_predict=500,  # Moderate length for testing
        )
        
        embeddings = SentenceTransformerEmbeddings(model_name='BAAI/bge-small-en-v1.5')
        
        # Load existing vector store
        vectorstore = LangChainLanceDB(
            embedding=embeddings,
            uri="data/test_langchain_db",
        )
        
        # Create prompt template
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't have that information.

        Context:
        {context}

        Question: {question}
        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        print("✅ QA chain created successfully")
        
        # Test queries
        test_queries = [
            "What is docling?",
            "What are the main features?",
            "How does it work?"
        ]
        
        for query in test_queries:
            print(f"\n📝 Testing query: '{query}'")
            
            try:
                result = qa_chain({"query": query})
                
                response = result.get("result", "No response")
                source_docs = result.get("source_documents", [])
                
                print(f"✅ Response: {response[:150]}...")
                print(f"✅ Used {len(source_docs)} source documents")
                
            except Exception as e:
                print(f"❌ Query failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ QA chain test error: {e}")
        return False


def compare_langchain_vs_legacy():
    """Compare LangChain vs legacy implementation performance"""
    print("\n⚖️ Comparing LangChain vs Legacy Implementation")
    print("=" * 50)
    
    try:
        # Test the same query with both implementations
        test_query = "What is docling and what are its main features?"
        
        print(f"🔍 Test query: '{test_query}'")
        print()
        
        # Test LangChain (if available)
        print("🦜 LangChain Implementation:")
        print("-" * 30)
        
        langchain_start = time.time()
        try:
            # This would call the actual LangChain function from 5-chat.py
            # For now, we'll simulate or call a simplified version
            from langchain_ollama import OllamaLLM
            from langchain_community.embeddings import SentenceTransformerEmbeddings
            
            llm = OllamaLLM(model="tinyllama", base_url="http://localhost:11434")
            
            # Simple test without full chain for speed
            langchain_response = llm.invoke(f"Answer briefly: {test_query}")
            langchain_time = time.time() - langchain_start
            
            print(f"✅ Response: {langchain_response[:200]}...")
            print(f"⏱️ Time: {langchain_time:.2f}s")
            
        except Exception as e:
            print(f"❌ LangChain error: {e}")
            langchain_time = None
        
        print()
        
        # Test Legacy (if available)
        print("🔧 Legacy Implementation:")
        print("-" * 30)
        
        legacy_start = time.time()
        try:
            # Simple Ollama call to simulate legacy
            import ollama
            
            legacy_response = ollama.chat(
                model="tinyllama",
                messages=[{"role": "user", "content": f"Answer briefly: {test_query}"}],
                stream=False
            )
            
            legacy_time = time.time() - legacy_start
            legacy_content = legacy_response.get('message', {}).get('content', 'No response')
            
            print(f"✅ Response: {legacy_content[:200]}...")
            print(f"⏱️ Time: {legacy_time:.2f}s")
            
        except Exception as e:
            print(f"❌ Legacy error: {e}")
            legacy_time = None
        
        # Compare results
        print()
        print("📊 Comparison Summary:")
        print("-" * 30)
        
        if langchain_time and legacy_time:
            if langchain_time < legacy_time:
                print(f"🏆 LangChain is {legacy_time/langchain_time:.1f}x faster")
            else:
                print(f"🏆 Legacy is {langchain_time/legacy_time:.1f}x faster")
        
        print("💡 LangChain provides:")
        print("  - Built-in retrieval strategies")
        print("  - Consistent prompt templates")
        print("  - Source document tracking")
        print("  - Chain type flexibility")
        
        print("💡 Legacy provides:")
        print("  - Custom context filtering")
        print("  - Streaming responses")        print("  - Fine-tuned relevance scoring")
        print("  - Professional document handling")
        
        return True
        
    except Exception as e:
        print(f"❌ Comparison error: {e}")
        return False


if __name__ == "__main__":
    print("🔍 Step 5: Chat Interface Debug (Updated with LangChain)")
    print("=" * 60)
    
    # Test standard Ollama functionality
    print("🔧 Testing Standard Ollama Functionality:")
    print("=" * 50)
    
    if test_ollama_connection():
        working_model = test_ollama_chat_api()
        
        if working_model:
            print(f"✅ Ollama is working with model: {working_model}")
            
            # Test LangChain components
            print("\n🦜 Testing LangChain Integration:")
            print("=" * 50)
            
            if test_langchain_components():
                print("✅ LangChain components working")
                
                if test_langchain_vectorstore():
                    print("✅ LangChain vector store working")
                    
                    if test_langchain_qa_chain():
                        print("✅ LangChain QA chain working")
                        
                        # Compare implementations
                        compare_langchain_vs_legacy()
                    else:
                        print("❌ LangChain QA chain failed")
                else:
                    print("❌ LangChain vector store failed")
            else:
                print("❌ LangChain components failed")
        else:
            print("❌ No working Ollama models found")
    else:
        print("❌ Ollama connection failed")
    
    print("\n" + "=" * 60)
    print("🏁 Debug Summary:")
    print("✅ Use LangChain for: Built-in RAG patterns, prompt templates, source tracking")
    print("✅ Use Legacy for: Custom filtering, streaming UI, professional document handling")
    print("💡 The refactored 5-chat.py now supports both implementations with a toggle!")
    print("✅ Chat debug completed!")
