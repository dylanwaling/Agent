import streamlit as st
import lancedb
import ollama
from sentence_transformers import SentenceTransformer
import tempfile
import os
from pathlib import Path
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from utils.tokenizer import OpenAITokenizerWrapper

# Initialize the embedding model (same as used for creating embeddings)
model = SentenceTransformer('BAAI/bge-small-en-v1.5')


# Initialize LanceDB connection
@st.cache_resource
def init_db():
    """Initialize database connection.

    Returns:
        LanceDB table object
    """
    db = lancedb.connect("data/lancedb")
    return db.open_table("docling")


# Initialize document converter and chunker
@st.cache_resource
def init_document_processor():
    """Initialize document processing components"""
    converter = DocumentConverter()
    tokenizer = OpenAITokenizerWrapper()
    chunker = HybridChunker(
        tokenizer=tokenizer,
        max_tokens=8191,
        merge_peers=True,
    )
    return converter, chunker


def get_context(query: str, table, num_results: int = 5) -> str:
    """Search the database for relevant context.

    Args:
        query: User's question
        table: LanceDB table object
        num_results: Number of results to return

    Returns:
        str: Concatenated context from relevant chunks with source information
    """
    # Create query embedding
    query_embedding = model.encode(query)
    
    # Search the table
    results = table.search(query_embedding).limit(num_results).to_pandas()
    contexts = []

    for _, row in results.iterrows():
        # Extract metadata directly from row (no nested metadata object)
        filename = row.get("filename")
        page_numbers = row.get("page_numbers")
        title = row.get("title")

        # Build source citation
        source_parts = []
        if filename:
            source_parts.append(filename)
        
        # Handle page_numbers safely - check if it's not None and has content
        if page_numbers is not None:
            try:
                # Convert to list if it's not already, then check if it has content
                page_list = list(page_numbers) if not isinstance(page_numbers, list) else page_numbers
                if page_list:  # Check if the list is not empty
                    source_parts.append(f"p. {', '.join(str(p) for p in page_list)}")
            except (TypeError, ValueError):
                # If there's any issue with page_numbers, just skip it
                pass

        source = f"\nSource: {' - '.join(source_parts)}" if source_parts else ""
        if title:
            source += f"\nTitle: {title}"

        contexts.append(f"{row['text']}{source}")

    return "\n\n".join(contexts)


def get_chat_response(messages, context: str) -> str:
    """Get streaming response from Ollama API.

    Args:
        messages: Chat history
        context: Retrieved context from database

    Returns:
        str: Model's response
    """
    system_prompt = f"""You are a helpful assistant that answers questions based on the provided context.
    Use only the information from the context to answer questions. If you're unsure or the context
    doesn't contain the relevant information, say so.
    
    Context:
    {context}
    """

    # Format messages for Ollama
    formatted_messages = [{"role": "system", "content": system_prompt}]
    formatted_messages.extend(messages)

    # Create the streaming response using Ollama
    try:
        response_stream = ollama.chat(
            model="tinyllama",  # Use the smaller TinyLlama model
            messages=formatted_messages,
            stream=True,
        )
        
        # Collect the response
        full_response = ""
        response_placeholder = st.empty()
        
        for chunk in response_stream:
            if 'message' in chunk and 'content' in chunk['message']:
                full_response += chunk['message']['content']
                response_placeholder.markdown(full_response + "▌")
        
        response_placeholder.markdown(full_response)
        return full_response
        
    except Exception as e:
        st.error(f"Error connecting to Ollama: {str(e)}")
        st.info("Make sure Ollama is running on http://localhost:11434 and you have the llama3 model installed.")
        return "Sorry, I couldn't process your request. Please check if Ollama is running."


def process_uploaded_document(uploaded_file, converter, chunker, table):
    """Process an uploaded document and add it to the database
    
    Args:
        uploaded_file: Streamlit uploaded file object
        converter: DocumentConverter instance
        chunker: HybridChunker instance
        table: LanceDB table
        
    Returns:
        bool: Success status
    """
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            # Convert the document
            st.write("🔄 Processing document...")
            result = converter.convert(tmp_file_path)
            
            if not result.document:
                st.error("Failed to extract content from document")
                return False
            
            # Chunk the document
            st.write("🔪 Creating chunks...")
            chunk_iter = chunker.chunk(dl_doc=result.document)
            chunks = list(chunk_iter)
            
            if not chunks:
                st.error("No chunks created from document")
                return False
            
            # Process chunks for database
            st.write("🤖 Creating embeddings...")
            processed_chunks = []
            for chunk in chunks:
                # Create embedding for the chunk text
                embedding = model.encode(chunk.text)
                
                # Process metadata
                page_numbers = [
                    page_no
                    for page_no in sorted(
                        set(
                            prov.page_no
                            for item in chunk.meta.doc_items
                            for prov in item.prov
                        )
                    )
                ] or None
                
                processed_chunks.append({
                    "text": chunk.text,
                    "vector": embedding,
                    "filename": uploaded_file.name,
                    "page_numbers": page_numbers,
                    "title": chunk.meta.headings[0] if chunk.meta.headings else None,
                })
            
            # Add to database
            st.write("💾 Adding to database...")
            
            # Get current data
            existing_data = table.to_pandas()
            
            # Create new table with combined data
            db = lancedb.connect("data/lancedb")
            all_data = existing_data.to_dict('records') + processed_chunks
            
            # Recreate table with all data
            db.create_table("docling", data=all_data, mode="overwrite")
            
            st.success(f"✅ Successfully processed {uploaded_file.name} and added {len(processed_chunks)} chunks to the database!")
            return True
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
                
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        return False


# Initialize Streamlit app
st.title("📚 Document Q&A")

# Sidebar for document management
with st.sidebar:
    st.header("📄 Document Management")
    
    # Document upload section
    st.subheader("Upload New Documents")
    uploaded_files = st.file_uploader(
        "Choose files to add to knowledge base",
        type=['pdf', 'docx', 'txt', 'md', 'html', 'xlsx', 'pptx'],
        accept_multiple_files=True,
        help="Supported formats: PDF, DOCX, TXT, Markdown, HTML, Excel, PowerPoint"
    )
    
    if uploaded_files:
        if st.button("🚀 Process Documents", type="primary"):
            converter, chunker = init_document_processor()
            table = init_db()
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")
                progress_bar.progress((i) / len(uploaded_files))
                
                success = process_uploaded_document(uploaded_file, converter, chunker, table)
                if not success:
                    break
                    
            progress_bar.progress(1.0)
            status_text.text("✅ All documents processed!")
            
            # Refresh the table connection
            st.cache_resource.clear()
            st.rerun()
    
    st.divider()
    
    # Database status
    st.subheader("📊 Database Status")
    try:
        table = init_db()
        df = table.to_pandas()
        
        st.metric("Total Chunks", len(df))
        
        # Show unique documents
        if 'filename' in df.columns:
            unique_docs = df['filename'].unique()
            st.metric("Documents", len(unique_docs))
            
            with st.expander("📚 View Documents"):
                for doc in unique_docs:
                    doc_chunks = len(df[df['filename'] == doc])
                    st.text(f"• {doc} ({doc_chunks} chunks)")
    except Exception as e:
        st.error(f"Database error: {e}")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize database connection
table = init_db()

# Sidebar for document upload
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt"])
    
    if uploaded_file:
        # Display file details
        st.write(f"Uploaded file: {uploaded_file.name}")
        st.write(f"File type: {uploaded_file.type}")
        st.write(f"File size: {uploaded_file.size / (1024 * 1024):.1f} MB")
        
        # Initialize document processing components
        converter, chunker = init_document_processor()
        
        # Process document button
        if st.button("Process Document"):
            # Process the uploaded document
            with st.spinner(f"Processing {uploaded_file.name}..."):
                success = process_uploaded_document(uploaded_file, converter, chunker, table)
                
                if success:
                    st.success(f"✅ {uploaded_file.name} processed and added to the database!")
                else:
                    st.error(f"❌ Failed to process {uploaded_file.name}")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the document"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get relevant context
    with st.status("Searching document...", expanded=False) as status:
        context = get_context(prompt, table)
        st.markdown(
            """
            <style>
            .search-result {
                margin: 10px 0;
                padding: 10px;
                border-radius: 4px;
                background-color: #f0f2f6;
            }
            .search-result summary {
                cursor: pointer;
                color: #0f52ba;
                font-weight: 500;
            }
            .search-result summary:hover {
                color: #1e90ff;
            }
            .metadata {
                font-size: 0.9em;
                color: #666;
                font-style: italic;
            }
            </style>
        """,
            unsafe_allow_html=True,
        )

        st.write("Found relevant sections:")
        for chunk in context.split("\n\n"):
            # Split into text and metadata parts
            parts = chunk.split("\n")
            text = parts[0]
            metadata = {
                line.split(": ")[0]: line.split(": ")[1]
                for line in parts[1:]
                if ": " in line
            }

            source = metadata.get("Source", "Unknown source")
            title = metadata.get("Title", "Untitled section")

            st.markdown(
                f"""
                <div class="search-result">
                    <details>
                        <summary>{source}</summary>
                        <div class="metadata">Section: {title}</div>
                        <div style="margin-top: 8px;">{text}</div>
                    </details>
                </div>
            """,
                unsafe_allow_html=True,
            )

    # Display assistant response
    with st.chat_message("assistant"):
        # Get model response with streaming
        response = get_chat_response(st.session_state.messages, context)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
