import streamlit as st
import lancedb
import ollama
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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


# Initialize Streamlit app
st.title("📚 Document Q&A")

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
