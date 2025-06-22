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


def get_context(query: str, table, num_results: int = 5, relevance_threshold: float = 0.3) -> str:
    """Search the database for relevant context.

    Args:
        query: User's question
        table: LanceDB table object
        num_results: Number of results to return
        relevance_threshold: Minimum similarity score to consider relevant (0.0-1.0)

    Returns:
        str: Concatenated context from relevant chunks with source information, or empty string if no relevant docs
    """
    # Create query embedding
    query_embedding = model.encode(query)
    
    # Search the table
    results = table.search(query_embedding).limit(num_results).to_pandas()
    
    # Check if query mentions a specific document filename using fuzzy matching
    target_filename = None
    query_lower = query.lower()
    
    # Get all available filenames from the database - need to search all data, not just top results
    db = lancedb.connect("data/lancedb")
    table_full = db.open_table("docling")
    all_data = table_full.to_pandas()
    
    if not all_data.empty and 'filename' in all_data.columns:
        all_filenames = all_data['filename'].unique()
        
        # Extract key parts from query that might be filename components
        import re
        
        # Remove common words and extract meaningful terms
        stop_words = {'my', 'the', 'a', 'an', 'is', 'what', 'whats', 'show', 'me', 'document', 'file', 'pdf', 'docx', 'txt'}
        query_words = [word for word in re.findall(r'\b\w+\b', query_lower) if word not in stop_words]
        
        best_match = None
        best_score = 0
        
        print(f"Debug - Query words after filtering: {query_words}")
        
        for filename in all_filenames:
            if not filename:
                continue
                
            filename_lower = filename.lower()
            # Remove file extensions and extract meaningful parts
            filename_clean = re.sub(r'\.(pdf|docx|txt|html|xlsx|pptx)$', '', filename_lower)
            filename_parts = re.findall(r'\b\w+\b', filename_clean)
            
            print(f"Debug - Checking filename: {filename} -> parts: {filename_parts}")
            
            # Count how many filename parts appear in the query
            matches = 0
            total_parts = len(filename_parts)
            
            for part in filename_parts:
                # Check for exact word match
                if part in query_words:
                    matches += 1
                # Check for partial matches (for things like "jan" matching "january")
                else:
                    for query_word in query_words:
                        if len(query_word) >= 3 and (part.startswith(query_word) or query_word in part):
                            matches += 0.7  # Partial match gets less weight
                            break
            
            # Special bonus for exact filename substring in query
            if filename_lower in query_lower or filename_clean in query_lower:
                matches += 1
            
            # Calculate match score
            if total_parts > 0:
                score = matches / total_parts
                print(f"Debug - {filename}: {matches}/{total_parts} = {score:.2f}")
                
                # Lower threshold for better matching, especially for short filenames
                threshold = 0.4 if total_parts <= 2 else 0.5
                
                if score >= threshold and score > best_score:
                    best_match = filename
                    best_score = score
        
        if best_match:
            target_filename = best_match
            print(f"Debug - Detected specific document request: {target_filename} (score: {best_score:.2f})")
    
    # Check if we have any results and if they meet the relevance threshold
    if results.empty:
        return ""
    
    # Set up distance threshold for relevance filtering
    max_distance = 0.75  # Only accept results with distance < 0.75 (balanced threshold)
    
    # Debug: print distance values to understand the scale
    if '_distance' in results.columns:
        print(f"Debug - Query: '{query}'")
        print(f"Debug - Distance values: {results['_distance'].tolist()}")
        
    # If a specific document was detected, search within that document only
    if target_filename:
        print(f"Debug - Searching specifically within document: {target_filename}")
        
        # For document-specific queries, get ALL chunks from that document
        # rather than limiting to semantic similarity
        document_data = all_data[all_data['filename'] == target_filename]
        if not document_data.empty:
            print(f"Debug - Found {len(document_data)} total chunks in document '{target_filename}'")
            
            # Convert document_data to the same format as search results
            # We'll create fake distances (all set to 0.1 to pass relevance check)
            results = document_data.copy()
            results['_distance'] = 0.1  # Set low distance so all chunks pass relevance check
            
            print(f"Debug - Using all {len(results)} chunks from target document")
        else:
            print(f"Debug - Document '{target_filename}' not found in database")
            return ""
        
        # Apply a very lenient relevance threshold (basically accept all chunks from the target doc)
        if not results.empty and '_distance' in results.columns:
            # Use a very high threshold to accept all chunks from the specific document
            doc_specific_threshold = 2.0  # Accept virtually all chunks from specific document
            relevant_mask = results['_distance'] < doc_specific_threshold
            print(f"Debug - Relevant mask for document-specific results (threshold {doc_specific_threshold}): accepting all {relevant_mask.sum()} chunks")
            
            results = results[relevant_mask]
            print(f"Debug - Using all {len(results)} chunks from target document")
    
    else:
        # No specific document detected, apply normal relevance filtering
        if '_distance' in results.columns:
            relevant_mask = results['_distance'] < max_distance
            print(f"Debug - Relevant mask: {relevant_mask.tolist()}")
            
            if not relevant_mask.any():
                print("Debug - No relevant results found based on distance threshold")
                return ""
            
            results = results[relevant_mask]
            print(f"Debug - Filtered to {len(results)} relevant results")
    
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
        type=['pdf', 'docx', 'txt', 'md', 'html', 'xlsx', 'pptx', 'jpg', 'jpeg', 'png', 'tiff', 'bmp'],
        accept_multiple_files=True,
        help="Supported formats: PDF, DOCX, TXT, Markdown, HTML, Excel, PowerPoint, Images (JPG, PNG, TIFF, BMP)"
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
    # Show a temporary status message
    search_placeholder = st.empty()
    search_placeholder.info("🔍 Searching documents...")
    
    context = get_context(prompt, table)
    
    # Clear the search status
    search_placeholder.empty()
    
    if not context:
        # No relevant documents found
        with st.chat_message("assistant"):
            no_context_response = "I couldn't find any relevant documents to answer your question. Please try asking about topics covered in your uploaded documents, or consider uploading more documents that might contain the information you're looking for."
            st.markdown(no_context_response)
            
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": no_context_response})
        
    else:
        # Found relevant context - show search results
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

        # Display assistant response with context
        with st.chat_message("assistant"):
            # Get model response with streaming
            response = get_chat_response(st.session_state.messages, context)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
