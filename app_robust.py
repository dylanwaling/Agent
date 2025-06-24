#!/usr/bin/env python3
"""
Document Q&A Streamlit App - Robust Version
Handles model loading gracefully to prevent freezing
"""

import streamlit as st
import os
from pathlib import Path
import tempfile
import logging
import sys
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="📚 Document Q&A",
    page_icon="📚",
    layout="wide"
)

def load_pipeline():
    """Load the pipeline with progress indication"""
    try:
        # Import our pipeline
        sys.path.append(os.path.dirname(__file__))
        exec(open('5-chat.py').read(), globals())
        
        # Initialize pipeline
        pipeline = DocumentPipeline()
        return pipeline
    except Exception as e:
        st.error(f"Failed to load pipeline: {e}")
        return None

def main():
    st.title("📚 Document Q&A Chat")
    st.markdown("**Upload documents and ask questions about them!**")
    
    # Initialize pipeline with progress
    if 'pipeline' not in st.session_state:
        with st.spinner("🔄 Loading AI models... This may take a moment on first run."):
            st.session_state.pipeline = load_pipeline()
            if st.session_state.pipeline is None:
                st.error("Failed to initialize pipeline. Please check the logs.")
                return
        st.success("✅ Pipeline loaded successfully!")
    
    pipeline = st.session_state.pipeline
    
    # Sidebar for document management
    with st.sidebar:
        st.header("📁 Document Management")
        
        # Current documents
        try:
            docs = pipeline.list_documents()
        except Exception as e:
            st.error(f"Error listing documents: {e}")
            docs = []
            
        st.subheader(f"Current Documents ({len(docs)})")
        
        if docs:
            for doc in docs:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"📄 {doc}")
                with col2:
                    if st.button("🗑️", key=f"delete_{doc}", help="Delete document"):
                        try:
                            if pipeline.remove_document(doc):
                                st.success(f"Deleted {doc}")
                                time.sleep(0.5)  # Brief pause
                                st.rerun()
                            else:
                                st.error(f"Failed to delete {doc}")
                        except Exception as e:
                            st.error(f"Error deleting {doc}: {e}")
        else:
            st.info("No documents uploaded yet")
        
        st.divider()
        
        # Upload new documents
        st.subheader("📤 Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=['pdf', 'txt', 'docx', 'md', 'png', 'jpg', 'jpeg'],
            help="Supported: PDF, TXT, DOCX, MD, PNG, JPG"
        )
        
        if uploaded_files:
            if st.button("Upload & Process", type="primary"):
                success_count = 0
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Save uploaded files
                for i, uploaded_file in enumerate(uploaded_files):
                    try:
                        status_text.text(f"Uploading {uploaded_file.name}...")
                        progress_bar.progress((i + 0.5) / len(uploaded_files))
                        
                        # Save to temp file first
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name
                        
                        # Add to pipeline
                        if pipeline.add_document(tmp_path):
                            success_count += 1
                        
                        # Clean up temp file
                        os.unlink(tmp_path)
                        progress_bar.progress((i + 1) / len(uploaded_files))
                        
                    except Exception as e:
                        st.error(f"Error uploading {uploaded_file.name}: {e}")
                
                if success_count > 0:
                    st.success(f"Uploaded {success_count} documents")
                    
                    # Process documents
                    status_text.text("Processing documents with AI...")
                    progress_bar.progress(0)
                    
                    try:
                        if pipeline.process_documents():
                            progress_bar.progress(1.0)
                            status_text.text("Processing complete!")
                            st.success("Documents processed successfully!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("Failed to process documents")
                    except Exception as e:
                        st.error(f"Processing error: {e}")
                else:
                    st.warning("No documents were uploaded successfully")
        
        st.divider()
        
        # System actions
        st.subheader("⚙️ System")
        
        if st.button("🔄 Rebuild Index"):
            if docs:
                with st.spinner("Rebuilding index..."):
                    try:
                        if pipeline.process_documents():
                            st.success("Index rebuilt successfully!")
                        else:
                            st.error("Failed to rebuild index")
                    except Exception as e:
                        st.error(f"Rebuild error: {e}")
            else:
                st.warning("No documents to process")
        
        # Debug info
        with st.expander("🔧 Debug Info"):
            try:
                st.write(f"Documents directory: {pipeline.docs_dir}")
                st.write(f"Index directory: {pipeline.index_dir}")
                st.write(f"Model: {pipeline.model_name}")
                st.write(f"QA Chain ready: {pipeline.qa_chain is not None}")
                st.write(f"Vector store ready: {pipeline.vectorstore is not None}")
            except Exception as e:
                st.write(f"Debug info error: {e}")
    
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
        # Check if we have documents
        try:
            current_docs = pipeline.list_documents()
        except Exception as e:
            st.error(f"Error checking documents: {e}")
            return
            
        if not current_docs:
            st.warning("Please upload some documents first!")
            return
            
        # Load index if needed
        if not pipeline.qa_chain:
            with st.spinner("Loading search index..."):
                try:
                    if not pipeline.load_index():
                        st.error("No search index found. Please upload and process documents first.")
                        return
                except Exception as e:
                    st.error(f"Error loading index: {e}")
                    return
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            try:
                with st.spinner("🤖 Thinking..."):
                    result = pipeline.ask(prompt)
                
                # Display answer
                if result["answer"]:
                    st.markdown(result["answer"])
                else:
                    st.warning("No answer generated. Please try rephrasing your question.")
                
                # Display sources
                if result.get("sources"):
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
                    "sources": result.get("sources", [])
                })
                
            except Exception as e:
                error_msg = f"Error generating response: {e}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": []
                })
    
    # Instructions
    if not st.session_state.get('pipeline') or not pipeline.list_documents():
        st.info("""
        **🚀 Get Started:**
        1. Upload documents using the sidebar
        2. Wait for processing to complete (may take a few minutes)
        3. Start asking questions!
        
        **📋 Supported formats:** PDF, TXT, DOCX, MD, images (PNG, JPG)
        
        **💡 Tips:**
        - First run may be slower due to model downloads
        - Larger documents take more time to process
        - Be specific in your questions for better results
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("*Powered by Docling + LangChain + Ollama*")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {e}")
        st.write("Please check the console for detailed error information.")
