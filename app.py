#!/usr/bin/env python3
"""
Document Q&A Streamlit App
Uses the DocumentPipeline from 5-chat.py
"""

import streamlit as st
import os
from pathlib import Path
import tempfile
import logging
import sys

# Import our pipeline
sys.path.append(os.path.dirname(__file__))
exec(open('5-chat.py').read())

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="📚 Document Q&A",
    page_icon="📚",
    layout="wide"
)

# Initialize pipeline
@st.cache_resource
def get_pipeline():
    """Initialize and cache the document pipeline"""
    return DocumentPipeline()

def main():
    st.title("📚 Document Q&A Chat")
    st.markdown("**Upload documents and ask questions about them!**")
    
    pipeline = get_pipeline()
    
    # Sidebar for document management
    with st.sidebar:
        st.header("📁 Document Management")
        
        # Current documents
        docs = pipeline.list_documents()
        st.subheader(f"Current Documents ({len(docs)})")
        
        if docs:
            for doc in docs:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"📄 {doc}")
                with col2:
                    if st.button("🗑️", key=f"delete_{doc}", help="Delete document"):
                        if pipeline.remove_document(doc):
                            st.success(f"Deleted {doc}")
                            st.rerun()
                        else:
                            st.error(f"Failed to delete {doc}")
        else:
            st.info("No documents uploaded yet")
        
        st.divider()
        
        # Upload new documents
        st.subheader("📤 Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=['pdf', 'txt', 'docx', 'md', 'png', 'jpg', 'jpeg']
        )
        
        if uploaded_files:
            if st.button("Upload & Process"):
                success_count = 0
                
                # Save uploaded files
                for uploaded_file in uploaded_files:
                    try:
                        # Save to temp file first
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name
                        
                        # Add to pipeline
                        if pipeline.add_document(tmp_path):
                            success_count += 1
                        
                        # Clean up temp file
                        os.unlink(tmp_path)
                        
                    except Exception as e:
                        st.error(f"Error uploading {uploaded_file.name}: {e}")
                
                if success_count > 0:
                    st.success(f"Uploaded {success_count} documents")
                    
                    # Process documents
                    with st.spinner("Processing documents..."):
                        if pipeline.process_documents():
                            st.success("Documents processed successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to process documents")
        
        st.divider()
        
        # System actions
        st.subheader("⚙️ System")
        
        if st.button("🔄 Rebuild Index"):
            if docs:
                with st.spinner("Rebuilding index..."):
                    if pipeline.process_documents():
                        st.success("Index rebuilt successfully!")
                    else:
                        st.error("Failed to rebuild index")
            else:
                st.warning("No documents to process")
        
        # Debug info
        with st.expander("🔧 Debug Info"):
            st.write(f"Documents directory: {pipeline.docs_dir}")
            st.write(f"Index directory: {pipeline.index_dir}")
            st.write(f"Model: {pipeline.model_name}")
            st.write(f"QA Chain ready: {pipeline.qa_chain is not None}")
    
    # Main chat interface
    st.header("💬 Chat")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available
            if message["role"] == "assistant" and "sources" in message:
                if message["sources"]:
                    with st.expander("📚 Sources"):
                        for i, source in enumerate(message["sources"], 1):
                            st.write(f"**{i}. {source['source']}**")
                            st.write(source['content'])
                            if i < len(message["sources"]):
                                st.divider()
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Check if we have documents and index
        if not docs:
            st.warning("Please upload some documents first!")
            return
            
        # Load index if needed
        if not pipeline.qa_chain:
            with st.spinner("Loading index..."):
                if not pipeline.load_index():
                    st.error("No index found. Please upload and process documents first.")
                    return
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = pipeline.ask(prompt)
                
                # Display answer
                st.markdown(result["answer"])
                
                # Display sources
                if result["sources"]:
                    with st.expander("📚 Sources"):
                        for i, source in enumerate(result["sources"], 1):
                            st.write(f"**{i}. {source['source']}**")
                            st.write(source['content'])
                            if i < len(result["sources"]):
                                st.divider()
                
                # Add to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": result["answer"],
                    "sources": result["sources"]
                })
    
    # Instructions
    if not docs:
        st.info("""
        **🚀 Get Started:**
        1. Upload documents using the sidebar
        2. Wait for processing to complete
        3. Start asking questions!
        
        **📋 Supported formats:** PDF, TXT, DOCX, MD, images (PNG, JPG)
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("*Powered by Docling + LangChain + Ollama*")

if __name__ == "__main__":
    main()
