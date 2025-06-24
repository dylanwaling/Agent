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

# LangChain imports
from langchain_core.documents import Document
from langchain_community.vectorstores import LanceDB as LangChainLanceDB
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate

# Initialize the embedding model (same as used for creating embeddings)
model = SentenceTransformer('BAAI/bge-small-en-v1.5')

# LangChain Configuration
USE_LANGCHAIN = True  # Toggle between LangChain and legacy implementation
LANGCHAIN_MODEL = "tinyllama"  # Default model for LangChain


# Initialize LangChain components
@st.cache_resource
def init_langchain_components():
    """Initialize LangChain components for RAG pipeline"""
    try:
        # Initialize LLM
        llm = OllamaLLM(
            model=LANGCHAIN_MODEL,
            base_url="http://localhost:11434",
            temperature=0.1,
            num_predict=2000,
        )
        
        # Initialize embeddings
        embeddings = SentenceTransformerEmbeddings(model_name='BAAI/bge-small-en-v1.5')
        
        # Custom prompt template for better responses
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
        
        return llm, embeddings, PROMPT
        
    except Exception as e:
        st.error(f"Failed to initialize LangChain components: {e}")
        return None, None, None


@st.cache_resource
def init_langchain_vectorstore():
    """Initialize LangChain vector store from existing LanceDB data"""
    try:
        # Connect to existing LanceDB
        db = lancedb.connect("data/lancedb")
        table = db.open_table("docling")
        
        # Convert existing data to LangChain Documents
        df = table.to_pandas()
        
        documents = []
        for _, row in df.iterrows():
            # Create Document with text and metadata
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
        
        # Initialize embeddings
        _, embeddings, _ = init_langchain_components()
        if embeddings is None:
            return None
            
        # Create vector store (this will create embeddings if needed)
        vectorstore = LangChainLanceDB.from_documents(
            documents=documents,
            embedding=embeddings,
            uri="data/lancedb_langchain",  # Separate database for LangChain
        )
        
        return vectorstore
        
    except Exception as e:
        st.error(f"Failed to initialize LangChain vector store: {e}")
        return None


def get_langchain_response(query: str, model_name: str = None) -> str:
    """Get response using LangChain RetrievalQA chain"""
    try:
        # Use provided model or default
        current_model = model_name or LANGCHAIN_MODEL
        
        # Initialize components
        llm = OllamaLLM(
            model=current_model,
            base_url="http://localhost:11434",
            temperature=0.1,
            num_predict=2000,
        )
        
        vectorstore = init_langchain_vectorstore()
        
        if vectorstore is None:
            return "Failed to initialize LangChain vector store. Please check your data."
        
        # Initialize embeddings and prompt
        _, embeddings, prompt = init_langchain_components()
        
        # Create retrieval QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # "stuff", "map_reduce", or "refine"
            retriever=vectorstore.as_retriever(
                search_kwargs={"k": 5}  # Number of documents to retrieve
            ),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        
        # Get response
        result = qa_chain({"query": query})
        
        # Format response with sources
        response = result.get("result", "No response generated")
        source_docs = result.get("source_documents", [])
        
        if source_docs:
            response += "\n\n**Sources:**\n"
            for i, doc in enumerate(source_docs[:3], 1):  # Show top 3 sources
                filename = doc.metadata.get('filename', 'Unknown')
                title = doc.metadata.get('title', 'No title')
                response += f"{i}. {filename} - {title}\n"
        
        return response
        
    except Exception as e:
        return f"Error generating LangChain response: {str(e)}"


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
    """Search the database for relevant context with professional-level large document handling.

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
        
    # PROFESSIONAL-LEVEL LARGE DOCUMENT HANDLING
    if target_filename:
        print(f"Debug - PROFESSIONAL MODE: Large document handling for '{target_filename}'")
        return handle_large_document_query(query, target_filename, all_data, query_embedding)
    
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
    total_length = 0
    
    # Standard context length limit for general queries
    max_context_length = 4000  # Increased from 2500 to include more chunks
    print(f"Debug - Standard context limit: {max_context_length} characters")
    
    # First pass: collect and filter chunks, prioritizing clean text
    good_chunks = []
    
    for i, (_, row) in enumerate(results.iterrows()):
        # Extract metadata directly from row (no nested metadata object)
        filename = row.get("filename")
        page_numbers = row.get("page_numbers")
        title = row.get("title")
        text = row.get("text", "")
        
        # Clean and assess chunk quality (same as before)
        import re
        cleaned_text = re.sub(r'[ \t]+', ' ', text)
        cleaned_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_text)
        cleaned_text = cleaned_text.strip()
        cleaned_text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', cleaned_text)
        
        # Quality assessment
        printable_ratio = sum(1 for c in cleaned_text if c.isprintable() or c in '\n\r\t') / len(cleaned_text) if cleaned_text else 0
        words = re.findall(r'\b[a-zA-Z]{3,}\b', cleaned_text)
        word_ratio = len(' '.join(words)) / len(cleaned_text) if cleaned_text else 0
        uppercase_ratio = sum(1 for c in cleaned_text if c.isupper()) / len(cleaned_text) if cleaned_text else 0
        random_char_count = sum(1 for c in cleaned_text if c in '<>{}[]|\\~/`^@#$%&*+=_') 
        space_ratio = sum(1 for c in cleaned_text if c == ' ') / len(cleaned_text) if cleaned_text else 0
        
        quality_score = min(printable_ratio, 1.0) * min(word_ratio * 2, 1.0)
        
        if random_char_count > len(cleaned_text) * 0.1:
            quality_score *= 0.3
        if uppercase_ratio > 0.8:
            quality_score *= 0.4
        if space_ratio < 0.05 and len(cleaned_text) > 50:
            quality_score *= 0.5
        if len(cleaned_text) < 10:
            quality_score *= 0.1
            
        if quality_score < 0.5:
            continue

        # Build source citation
        source_parts = []
        if filename:
            source_parts.append(filename)
        
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

        chunk_content = f"{cleaned_text}{source}"
        good_chunks.append(chunk_content)
    
    print(f"Debug - Found {len(good_chunks)} high-quality chunks after filtering")
    
    # Debug: show what chunks we actually have
    for i, chunk in enumerate(good_chunks[:3]):  # Show first 3 chunks
        print(f"Debug - Chunk {i+1} preview: '{chunk[:100]}...'")
    
    # Build context from good chunks with standard length limit
    for i, chunk_content in enumerate(good_chunks):
        if max_context_length is not None and total_length + len(chunk_content) > max_context_length:
            print(f"Debug - Stopping at chunk {i+1} to avoid context overflow (would be {total_length + len(chunk_content)} chars)")
            break
            
        contexts.append(chunk_content)
        total_length += len(chunk_content)
        print(f"Debug - Added chunk {i+1}, total length now: {total_length}")

    final_context = "\n\n".join(contexts)
    print(f"Debug - Final context length: {len(final_context)} characters")
    print(f"Debug - Final context sections: {len(contexts)}")
    return final_context


def handle_large_document_query(query: str, filename: str, all_data, query_embedding) -> str:
    """Professional-level handling for large document queries using hierarchical retrieval and summarization.
    
    Args:
        query: User's question
        filename: Target document filename  
        all_data: Full database as pandas DataFrame
        query_embedding: Query embedding vector
        
    Returns:
        str: Comprehensive context using professional techniques
    """
    print(f"Debug - PROFESSIONAL: Handling large document query for '{filename}'")
    
    # Get all chunks from the target document
    document_data = all_data[all_data['filename'] == filename].copy()
    if document_data.empty:
        return ""
    
    print(f"Debug - PROFESSIONAL: Found {len(document_data)} chunks in document")
    
    # Step 1: HIERARCHICAL RETRIEVAL - Find most relevant sections first
    print("Debug - PROFESSIONAL: Step 1 - Hierarchical retrieval")
    
    # Calculate relevance scores for all chunks
    import numpy as np
    from sentence_transformers import util
    
    chunk_embeddings = []
    valid_chunks = []
    
    model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    
    for idx, (_, row) in enumerate(document_data.iterrows()):
        if 'vector' in row and row['vector'] is not None:
            chunk_embeddings.append(row['vector'])
            valid_chunks.append(idx)
    
    if not chunk_embeddings:
        print("Debug - PROFESSIONAL: No valid embeddings found")
        return ""
    
    # Compute similarities
    chunk_embeddings = np.array(chunk_embeddings)
    similarities = util.cos_sim(query_embedding, chunk_embeddings)[0]
    
    # Get top relevant chunks (more than before)
    top_k = min(15, len(similarities))  # Get top 15 most relevant chunks
    top_indices = similarities.argsort(descending=True)[:top_k]
    
    print(f"Debug - PROFESSIONAL: Selected top {top_k} most relevant chunks")
    
    # Step 2: SMART SUMMARIZATION STRATEGY
    print("Debug - PROFESSIONAL: Step 2 - Smart summarization")
    
    relevant_chunks = []
    for idx in top_indices:
        chunk_idx = valid_chunks[idx]
        row = document_data.iloc[chunk_idx]
        
        text = row.get('text', '')
        # Clean the text
        import re
        cleaned_text = re.sub(r'[ \t]+', ' ', text)
        cleaned_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_text)
        cleaned_text = cleaned_text.strip()
        cleaned_text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', cleaned_text)
        
        # Quality check
        if len(cleaned_text) < 20:
            continue
            
        # Check quality
        words = re.findall(r'\b[a-zA-Z]{3,}\b', cleaned_text)
        word_ratio = len(' '.join(words)) / len(cleaned_text) if cleaned_text else 0
        
        if word_ratio < 0.3:  # Skip low-quality chunks
            continue
            
        page_numbers = row.get("page_numbers")
        # Handle page_numbers safely - could be None, list, or numpy array
        if page_numbers is not None:
            try:
                # Convert to list if it's a numpy array or other iterable
                if hasattr(page_numbers, '__len__') and len(page_numbers) > 0:
                    # It's an array-like object with elements
                    page_list = list(page_numbers)
                    source_info = f" (Page {', '.join(str(p) for p in page_list)})"
                elif hasattr(page_numbers, '__iter__') and not isinstance(page_numbers, str):
                    # It's iterable but not a string
                    page_list = list(page_numbers)
                    if page_list:
                        source_info = f" (Page {', '.join(str(p) for p in page_list)})"
                    else:
                        source_info = " (Page unknown)"
                else:
                    # It's a single value
                    source_info = f" (Page {page_numbers})"
            except (TypeError, ValueError):
                source_info = " (Page unknown)"
        else:
            source_info = " (Page unknown)"
        
        relevant_chunks.append({
            'text': cleaned_text,
            'source': source_info,
            'similarity': similarities[idx].item()
        })
    
    print(f"Debug - PROFESSIONAL: Found {len(relevant_chunks)} high-quality relevant chunks")
    
    # Step 3: ADAPTIVE CONTEXT ASSEMBLY
    print("Debug - PROFESSIONAL: Step 3 - Adaptive context assembly")
    
    if not relevant_chunks:
        return ""
    
    # Sort by relevance score
    relevant_chunks.sort(key=lambda x: x['similarity'], reverse=True)
    
    # Build comprehensive context with smart truncation
    contexts = []
    total_length = 0
    max_context = 6000  # Larger context for document-specific queries
    
    # Include a brief document summary first
    doc_summary = f"Document: {filename}\nFound {len(relevant_chunks)} relevant sections.\n\n"
    contexts.append(doc_summary)
    total_length += len(doc_summary)
    
    # Add most relevant chunks with smart prioritization
    for i, chunk in enumerate(relevant_chunks):
        chunk_text = chunk['text']
        source_info = f"\n[Relevance: {chunk['similarity']:.2f}] Source: {filename}{chunk['source']}"
        full_chunk = f"{chunk_text}{source_info}"
        
        # Check if we can fit this chunk
        if total_length + len(full_chunk) > max_context:
            # Try to fit a truncated version of highly relevant chunks
            if chunk['similarity'] > 0.7 and i < 3:  # Very relevant and in top 3
                available_space = max_context - total_length - len(source_info) - 50
                if available_space > 200:
                    truncated_text = chunk_text[:available_space] + "..."
                    full_chunk = f"{truncated_text}{source_info}"
                    contexts.append(full_chunk) 
                    total_length += len(full_chunk)
            break
        else:
            contexts.append(full_chunk)
            total_length += len(full_chunk)
    
    final_context = "\n\n".join(contexts)
    
    print(f"Debug - PROFESSIONAL: Final context: {len(final_context)} chars from {len(contexts)-1} chunks")
    print(f"Debug - PROFESSIONAL: Coverage: {len(contexts)-1}/{len(relevant_chunks)} relevant chunks included")
    
    return final_context
def get_chat_response(messages, context: str) -> str:
    """Get streaming response from Ollama API.

    Args:
        messages: Chat history
        context: Retrieved context from database

    Returns:
        str: Model's response
    """
    # Dynamic context limits based on query type
    is_large_doc_query = len(context) > 4000 and ("Source:" in context and context.count("Source:") >= 3)
    
    if is_large_doc_query:
        max_safe_context = 6000  # Much larger for document-specific queries
        print(f"Debug - PROFESSIONAL mode: using large context limit of {max_safe_context}")
    else:
        max_safe_context = 4000  # Increased from 2500 to match new context limit
        print(f"Debug - Standard mode: using context limit of {max_safe_context}")
    
    if len(context) > max_safe_context:
        print(f"Debug - Context too large ({len(context)} chars), truncating to {max_safe_context}")
        
        # Smart truncation - try to keep complete sections
        truncated_context = context[:max_safe_context]
        
        # Try to end at a section boundary first, then sentence
        last_section = truncated_context.rfind('\n\nSource:')
        last_paragraph = truncated_context.rfind('\n\n')
        last_sentence = truncated_context.rfind('. ')
        
        if last_section > max_safe_context - 800:  # Keep complete sections when possible
            context = truncated_context[:last_section]
        elif last_paragraph > max_safe_context - 400:
            context = truncated_context[:last_paragraph]
        elif last_sentence > max_safe_context - 200:
            context = truncated_context[:last_sentence + 1]
        else:
            context = truncated_context
        
        print(f"Debug - Context truncated to {len(context)} characters")

    # Use a universal system prompt for all queries
    system_prompt = f"""Answer the user's question using only the provided context. Be clear and informative.

Context:
{context}

If the context doesn't contain the answer, say "I don't have that information in the provided context." """

    # Format messages for Ollama
    formatted_messages = [{"role": "system", "content": system_prompt}]
    formatted_messages.extend(messages)

    # Create the streaming response using Ollama with very conservative settings
    try:
        print(f"Debug - Sending {len(system_prompt)} characters to model with conservative settings")
        
        
        # Universal response limits for all queries - optimized for speed
        is_large_doc_query = len(context) > 4000 and ("Source:" in context and context.count("Source:") >= 3)
        
        if is_large_doc_query:
            # Professional large document mode - use TinyLlama for speed
            max_response_length = 4000
            stop_after_sentences = 8
            max_chunks = 500
            model_name = "tinyllama"  # Use TinyLlama to avoid timeouts
            print(f"Debug - PROFESSIONAL large document mode: max_response={max_response_length}, sentences={stop_after_sentences}, model={model_name}")
        elif len(context) > 2000:
            # Medium context mode
            max_response_length = 3000  # Increased for better responses
            stop_after_sentences = 6    # More sentences allowed
            max_chunks = 400           # More chunks allowed
            model_name = "tinyllama"
            print(f"Debug - Medium context mode: max_response={max_response_length}, sentences={stop_after_sentences}, model={model_name}")
        else:
            # Fast response mode
            max_response_length = 2000  # Increased for better responses
            stop_after_sentences = 4    # More sentences allowed
            max_chunks = 300           # More chunks allowed
            model_name = "tinyllama"
            print(f"Debug - Fast mode: max_response={max_response_length}, sentences={stop_after_sentences}, model={model_name}")
        
        # Create enhanced prompt for large document queries
        if is_large_doc_query:
            system_prompt = f"""You are analyzing a large document with multiple relevant sections. Based on the comprehensive context below, provide a thorough and well-structured answer to the user's question.

IMPORTANT: The context contains multiple sections with relevance scores and page numbers. Synthesize information across all sections to provide a complete answer.

Context:
{context}

Please provide a comprehensive answer that:
1. Addresses the question thoroughly
2. References specific sections/pages when relevant  
3. Synthesizes information from multiple sources if applicable
4. Is well-organized and easy to follow"""
        else:
            system_prompt = f"""Answer the user's question using only the provided context. Be clear and informative.

Context:
{context}

If the context doesn't contain the answer, say "I don't have that information in the provided context." """

        # Format messages for Ollama
        formatted_messages = [{"role": "system", "content": system_prompt}]
        formatted_messages.extend(messages)

        # Send request to Ollama with appropriate model and settings
        response_stream = ollama.chat(
            model=model_name,
            messages=formatted_messages,
            stream=True,
            options={
                "temperature": 0.1,
                "top_p": 0.9,  # Slightly more flexible
                "num_predict": max_response_length,  # Allow full response length
                "stop": ["Context:", "User:"] if not is_large_doc_query else [],  # Fewer restrictive stops
            }
        )
        
        # Universal response limits for all queries - optimized for speed
        full_response = ""
        response_placeholder = st.empty()
        chunk_count = 0
        
        print(f"Debug - Using optimized limits: max chunks: {max_chunks}, max response: {max_response_length}")
        print(f"Debug - Using model: {model_name}")
        
        try:
            for chunk in response_stream:
                chunk_count += 1
                if chunk_count > max_chunks:
                    print(f"Debug - Response chunk limit reached ({max_chunks}), stopping")
                    break
                    
                if 'message' in chunk and 'content' in chunk['message']:
                    chunk_content = chunk['message']['content']
                    full_response += chunk_content
                    response_placeholder.markdown(full_response + "▌")
                    
                    # Check response length limit
                    if len(full_response) > max_response_length:
                        print(f"Debug - Response length limit reached ({max_response_length}), stopping")
                        break
                    
                    # Check for natural stopping points
                    if full_response.endswith(('.', '!', '?')) and len(full_response) > 200:
                        sentences = full_response.count('.')
                        if sentences >= stop_after_sentences:  # Variable sentence limit
                            print(f"Debug - Natural stopping point reached after {sentences} sentences")
                            break
        
        except Exception as stream_error:
            print(f"Debug - Stream error: {stream_error}")
            # Continue to fallback logic below
        
        response_placeholder.markdown(full_response)
        
        # If we got no response, provide a simple fallback
        if not full_response.strip():
            # Extract key information directly from context for fallback
            context_preview = context[:500] + "..." if len(context) > 500 else context
            fallback_response = f"I found relevant information but had a technical issue. Here's what I found:\n\n{context_preview}"
            response_placeholder.markdown(fallback_response)
            return fallback_response
            
        return full_response
        
    except Exception as e:
        error_msg = str(e)
        print(f"Debug - Ollama error: {error_msg}")
        
        # Simple fallback without complex logic
        context_preview = context[:300] + "..." if len(context) > 300 else context
        fallback_response = f"Technical issue with AI model. Here's the relevant content I found:\n\n{context_preview}"
        return fallback_response


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


def reprocess_document_if_corrupted(filename, converter, chunker, table):
    """Re-process a document if it appears to be corrupted in the database
    
    Args:
        filename: Name of the document to reprocess
        converter: DocumentConverter instance
        chunker: HybridChunker instance
        table: LanceDB table
        
    Returns:
        bool: Success status
    """
    try:
        # Check if the document exists in the filesystem
        # For now, we'll just print a message - in a full system, you'd want to
        # store original file paths or allow re-upload
        st.warning(f"Document '{filename}' appears to have extraction issues. Please re-upload the document for better results.")
        return False
        
    except Exception as e:
        st.error(f"Error checking document: {str(e)}")
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
    
# Add LangChain toggle in sidebar
with st.sidebar:
    st.divider()
    st.subheader("🔧 Settings")
    
    # LangChain toggle
    use_langchain = st.checkbox(
        "🦜 Use LangChain RAG", 
        value=USE_LANGCHAIN,
        help="Toggle between LangChain and legacy implementation"
    )
    
    if use_langchain:
        langchain_model = st.selectbox(
            "Model",
            options=["tinyllama", "llama3"],
            index=0 if LANGCHAIN_MODEL == "tinyllama" else 1,
            help="Select the Ollama model to use"
        )
        
        chain_type = st.selectbox(
            "Chain Type",
            options=["stuff", "map_reduce", "refine"],
            index=0,
            help="stuff: Simple context stuffing\nmap_reduce: Parallel processing\nrefine: Iterative refinement"
        )
    
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

    # Show processing status
    search_placeholder = st.empty()
    
    # Choose between LangChain and legacy implementation
    if use_langchain:
        search_placeholder.info("🦜 Processing with LangChain...")
        
        # Display assistant response
        with st.chat_message("assistant"):
            # Get LangChain response with selected model
            current_model = langchain_model if 'langchain_model' in locals() else LANGCHAIN_MODEL
            response = get_langchain_response(prompt, current_model)
            st.markdown(response)
            
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
    else:
        # Legacy implementation
        search_placeholder.info("🔍 Searching documents (Legacy)...")
        
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
            # Found relevant context - show search results (legacy display)
            print(f"Debug - Context preview for UI display: {context[:500]}...")
            print(f"Debug - Context sections count: {len(context.split('\n\n'))}")
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
            sections = context.split("\n\n")
            print(f"Debug - Raw sections: {len(sections)}")
            
            for i, chunk in enumerate(sections):
                if not chunk.strip():
                    continue
                    
                print(f"Debug - Processing section {i+1}: '{chunk[:100]}...'")
                
                # Split into text and metadata parts
                lines = chunk.split("\n")
                text = lines[0] if lines else ""
                
                # Find source and title lines - look for lines containing source info
                source_line = ""
                title_line = ""
                
                for line in lines[1:]:
                    if "Source:" in line:
                        # Extract everything after "Source:"
                        source_start = line.find("Source:")
                        source_line = line[source_start + 7:].strip()  # +7 to skip "Source:"
                    elif line.startswith("Title:"):
                        title_line = line.replace("Title: ", "")
                    elif line.startswith("[Relevance:") and "Source:" in line:
                        # Handle lines with both relevance and source
                        source_start = line.find("Source:")
                        source_line = line[source_start + 7:].strip()
                
                # Use source info or create a meaningful default
                if source_line:
                    display_source = source_line
                else:
                    # Try to extract filename from text if it looks like a document reference
                    if "1 Jan.pdf" in chunk or "2408.09869v5.pdf" in chunk:
                        if "1 Jan.pdf" in chunk:
                            display_source = "1 Jan.pdf"
                        else:
                            display_source = "2408.09869v5.pdf"
                    else:
                        display_source = f"Section {i+1}"
                
                display_title = title_line if title_line else "Content"
                
                print(f"Debug - Section {i+1}: source='{display_source}', title='{display_title}'")
                
                st.markdown(
                    f"""
                    <div class="search-result">
                        <details>
                            <summary>{display_source}</summary>
                            <div class="metadata">Section: {display_title}</div>
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
    
    # Clear the search status
    search_placeholder.empty()
