import streamlit as st
import lancedb
import pandas as pd
from pathlib import Path
import tempfile
import os

# LangChain imports (updated to modern syntax)
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Document processing imports
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from utils.tokenizer import OpenAITokenizerWrapper

# Configuration
USE_LANGCHAIN = True  # Toggle between LangChain and legacy mode

@st.cache_resource
def init_langchain_components():
    """Initialize LangChain components"""
    try:
        # Initialize embeddings (same model as before for consistency)
        embeddings = HuggingFaceEmbeddings(
            model_name='BAAI/bge-small-en-v1.5',
            model_kwargs={'device': 'cpu'}
        )
          # Initialize Ollama LLM
        llm = OllamaLLM(
            model="tinyllama",  # Use fast model
            temperature=0.1,
            timeout=30
        )
        
        return embeddings, llm
    except Exception as e:
        st.error(f"Error initializing LangChain: {e}")
        return None, None

@st.cache_resource
def load_documents_from_lancedb():
    """Load documents from LanceDB and convert to LangChain format"""
    try:
        # Connect to existing LanceDB
        db = lancedb.connect("data/lancedb")
        table = db.open_table("docling")
        df = table.to_pandas()
        
        if df.empty:
            return []
        
        # Convert to LangChain Documents
        documents = []
        for _, row in df.iterrows():
            # Create metadata
            metadata = {
                "filename": row.get("filename", "Unknown"),
                "source": row.get("filename", "Unknown")
            }
            
            # Add page numbers if available
            if "page_numbers" in row and row["page_numbers"] is not None:
                try:
                    pages = list(row["page_numbers"]) if hasattr(row["page_numbers"], '__iter__') else [row["page_numbers"]]
                    metadata["page_numbers"] = pages
                except:
                    pass
            
            # Add title if available
            if "title" in row and row["title"]:
                metadata["title"] = row["title"]
            
            # Create Document
            doc = Document(
                page_content=row.get("text", ""),
                metadata=metadata
            )
            documents.append(doc)
        
        st.success(f"Loaded {len(documents)} documents from database")
        return documents
    
    except Exception as e:
        st.error(f"Error loading documents: {e}")
        return []

@st.cache_resource
def create_langchain_vectorstore(_documents, _embeddings):
    """Create FAISS vectorstore from documents"""
    try:
        if not _documents:
            return None
        
        # Create vectorstore
        vectorstore = FAISS.from_documents(_documents, _embeddings)
        st.success("Created LangChain vectorstore")
        return vectorstore
    
    except Exception as e:
        st.error(f"Error creating vectorstore: {e}")
        return None

def create_qa_chain(vectorstore, llm, chain_type="stuff"):
    """Create RetrievalQA chain"""
    try:
        # Custom prompt template
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context:
        {context}

        Question: {question}
        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )
        
        # Create retrieval chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type=chain_type,
            retriever=vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        return qa_chain
    
    except Exception as e:
        st.error(f"Error creating QA chain: {e}")
        return None

def process_langchain_query(query, qa_chain):
    """Process query using LangChain"""
    try:
        # Run the chain
        result = qa_chain({"query": query})
        
        # Extract answer and sources
        answer = result.get("result", "No answer generated")
        source_docs = result.get("source_documents", [])
        
        # Format sources
        sources = []
        for doc in source_docs:
            source_info = {
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "filename": doc.metadata.get("filename", "Unknown"),
                "page_numbers": doc.metadata.get("page_numbers", []),
                "title": doc.metadata.get("title", "")
            }
            sources.append(source_info)
        
        return answer, sources
    
    except Exception as e:
        st.error(f"Error processing query: {e}")
        return f"Error: {str(e)}", []

# Legacy functions (simplified versions of the original)
@st.cache_resource
def init_legacy_db():
    """Initialize legacy database connection"""
    db = lancedb.connect("data/lancedb")
    return db.open_table("docling")

def legacy_get_context(query, table, num_results=5):
    """Simplified legacy context retrieval"""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('BAAI/bge-small-en-v1.5')
        
        query_embedding = model.encode(query)
        results = table.search(query_embedding).limit(num_results).to_pandas()
        
        if results.empty:
            return ""
        
        contexts = []
        for _, row in results.iterrows():
            text = row.get("text", "")
            filename = row.get("filename", "Unknown")
            context = f"{text}\n\nSource: {filename}"
            contexts.append(context)
        
        return "\n\n".join(contexts)
    except Exception as e:
        return f"Error retrieving context: {e}"

def legacy_get_response(messages, context):
    """Simplified legacy response generation"""
    try:
        import ollama
        
        system_prompt = f"""Answer the question using the provided context.

Context:
{context}

If the context doesn't contain the answer, say "I don't have that information in the provided context." """
        
        formatted_messages = [{"role": "system", "content": system_prompt}]
        formatted_messages.extend(messages)
        
        response = ollama.chat(
            model="tinyllama",
            messages=formatted_messages,
            stream=False,
            options={"temperature": 0.1, "num_predict": 1000}
        )
        
        return response.get("message", {}).get("content", "No response generated")
    
    except Exception as e:
        return f"Error generating response: {e}"

# Streamlit App
st.title("📚 Document Q&A with LangChain")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Mode toggle
    use_langchain = st.toggle("Use LangChain", value=USE_LANGCHAIN)
    
    if use_langchain:
        st.info("🔗 Using LangChain RetrievalQA")
        chain_type = st.selectbox(
            "Chain Type",
            ["stuff", "map_reduce", "refine"],
            help="Different strategies for handling multiple documents"
        )
    else:
        st.info("🔧 Using Legacy RAG")
    
    st.divider()
    
    # Database status
    st.subheader("📊 Database Status")
    try:
        db = lancedb.connect("data/lancedb")
        table = db.open_table("docling")
        df = table.to_pandas()
        
        st.metric("Total Chunks", len(df))
        
        if 'filename' in df.columns:
            unique_docs = df['filename'].unique()
            st.metric("Documents", len(unique_docs))
            
            with st.expander("📚 View Documents"):
                for doc in unique_docs:
                    doc_chunks = len(df[df['filename'] == doc])
                    st.text(f"• {doc} ({doc_chunks} chunks)")
    except Exception as e:
        st.error(f"Database error: {e}")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize components based on mode
if use_langchain:
    # Initialize LangChain components
    embeddings, llm = init_langchain_components()
    
    if embeddings and llm:
        # Load documents and create vectorstore
        documents = load_documents_from_lancedb()
        
        if documents:
            vectorstore = create_langchain_vectorstore(documents, embeddings)
            
            if vectorstore:
                qa_chain = create_qa_chain(vectorstore, llm, chain_type)
                
                if not qa_chain:
                    st.error("Failed to create QA chain")
                    use_langchain = False
            else:
                st.error("Failed to create vectorstore")
                use_langchain = False
        else:
            st.error("No documents found in database")
            use_langchain = False
    else:
        st.error("Failed to initialize LangChain components")
        use_langchain = False

else:
    # Initialize legacy components
    try:
        table = init_legacy_db()
    except Exception as e:
        st.error(f"Failed to initialize legacy database: {e}")
        table = None

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate response
    with st.chat_message("assistant"):
        if use_langchain and 'qa_chain' in locals():
            # Use LangChain
            with st.spinner("🔗 Processing with LangChain..."):
                answer, sources = process_langchain_query(prompt, qa_chain)
            
            # Display answer
            st.markdown(answer)
            
            # Display sources
            if sources:
                st.markdown("### 📚 Sources:")
                for i, source in enumerate(sources):
                    with st.expander(f"Source {i+1}: {source['filename']}"):
                        st.text(f"Content: {source['content']}")
                        if source['page_numbers']:
                            st.text(f"Pages: {source['page_numbers']}")
                        if source['title']:
                            st.text(f"Title: {source['title']}")
            
            response_text = answer
        
        else:
            # Use legacy mode
            if table is not None:
                with st.spinner("🔧 Processing with legacy RAG..."):
                    context = legacy_get_context(prompt, table)
                    
                    if context:
                        response_text = legacy_get_response(st.session_state.messages, context)
                        st.markdown(response_text)
                        
                        # Show context used
                        with st.expander("📋 Context Used"):
                            st.text(context[:1000] + "..." if len(context) > 1000 else context)
                    else:
                        response_text = "I couldn't find relevant information to answer your question."
                        st.markdown(response_text)
            else:
                response_text = "Database connection failed. Please check the setup."
                st.markdown(response_text)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_text})

# Add helpful information
st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Tips")
st.sidebar.markdown("- **Stuff**: Best for simple questions")
st.sidebar.markdown("- **Map-reduce**: Good for complex analysis")
st.sidebar.markdown("- **Refine**: Best for detailed synthesis")
