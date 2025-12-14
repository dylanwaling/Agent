# ============================================================================
# SEARCH ENGINE MODULE
# ============================================================================
# Purpose:
#   Semantic search and question answering functionality
#   Handles FAISS search, relevance filtering, and LLM response generation
# ============================================================================

import logging
import time
from typing import List, Dict, Any, Optional

# Local imports
from Config.settings import search_config, model_config, logging_config
from Utils.system_io_helpers import normalize_filename

logger = logging.getLogger(__name__)


class SearchEngine:
    """
    Semantic search and question answering handler.
    
    Handles:
    - FAISS vector similarity search
    - Relevance-based filtering with filename priority
    - Context building from search results
    - LLM-based question answering
    - Streaming response generation
    """
    
    def __init__(self, document_processor, components, analytics_logger):
        """
        Initialize search engine.
        
        Args:
            document_processor: DocumentProcessor instance (provides vectorstore)
            components: ComponentInitializer instance (provides LLM and prompt template)
            analytics_logger: AnalyticsLogger instance
        """
        self.doc_processor = document_processor
        self.components = components
        self.analytics = analytics_logger
    
    @property
    def vectorstore(self):
        """Get vectorstore from document processor."""
        return self.doc_processor.vectorstore
    
    def search(self, query: str, score_threshold: Optional[float] = None, 
               update_status: bool = True) -> List[Dict[str, Any]]:
        """
        Search documents with relevance-based filtering.
        
        Args:
            query: Search query text
            score_threshold: Maximum distance score for relevance filtering (lower is better)
            update_status: Whether to update status to IDLE after search
            
        Returns:
            List of search results with content, source, chunk_id, and relevance_score
        """
        search_start = time.time()
        
        # Use default threshold from config if not provided
        if score_threshold is None:
            score_threshold = search_config.DEFAULT_SCORE_THRESHOLD
        
        try:
            if not self.vectorstore:
                return []
            
            # Log embedding query
            embed_start = time.time()
            self.analytics.log_operation(
                operation_type=logging_config.OPERATION_TYPES['EMBEDDING_QUERY'],
                operation=f"Embedding query: {query[:80]}",
                metadata={
                    "query": query,
                    "query_length": len(query),
                    "model": model_config.EMBEDDING_MODEL,
                    "dimensions": model_config.EMBEDDING_DIMENSION,
                    "device": getattr(self.components, 'device', 'cpu')
                },
                status=logging_config.STATUS_TYPES['PROCESSING']
            )
            
            # Use similarity search with score threshold for smart retrieval
            k_value = search_config.SEARCH_K
            docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k_value)
            embed_time = time.time() - embed_start
            
            # Log FAISS search
            self.analytics.log_operation(
                operation_type=logging_config.OPERATION_TYPES['FAISS_SEARCH'],
                operation=f"FAISS search: {query[:80]}",
                metadata={
                    "query": query,
                    "k": k_value,
                    "num_results": len(docs_with_scores),
                    "index_size": self.vectorstore.index.ntotal if self.vectorstore else 0,
                    "search_time_ms": round(embed_time * 1000, 2),
                    "device": getattr(self.components, 'device', 'cpu')
                },
                status=logging_config.STATUS_TYPES['PROCESSING']
            )
            
            # Smart filtering with filename priority
            relevant_docs = self._filter_by_relevance(query, docs_with_scores, score_threshold)
            
            search_time = time.time() - search_start
            
            # Log relevance filtering results
            self.analytics.log_operation(
                operation_type=logging_config.OPERATION_TYPES['RELEVANCE_FILTER'],
                operation=f"Filtered search results: {query[:80]}",
                metadata={
                    "query": query,
                    "total_candidates": len(docs_with_scores),
                    "filtered_results": len(relevant_docs),
                    "score_threshold": score_threshold,
                    "filter_time_s": round(search_time, 3)
                },
                status=logging_config.STATUS_TYPES['IDLE'] if update_status else logging_config.STATUS_TYPES['PROCESSING']
            )
            
            return relevant_docs
            
        except Exception as e:
            logger.error(f"Error searching: {e}")
            self.analytics.log_operation(
                operation_type="error",
                operation=f"Search error: {str(e)[:100]}",
                metadata={
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                },
                status="ERROR"
            )
            return []
    
    def _filter_by_relevance(self, query: str, docs_with_scores: List, 
                            score_threshold: float) -> List[Dict[str, Any]]:
        """
        Filter search results by relevance with filename priority.
        
        Args:
            query: Search query text
            docs_with_scores: List of (Document, score) tuples from FAISS
            score_threshold: Base threshold for filtering
            
        Returns:
            List of filtered and sorted results
        """
        relevant_docs = []
        query_lower = query.lower()
        
        for doc, score in docs_with_scores:
            source = doc.metadata.get("source", "").lower()
            filename = doc.metadata.get("filename", "").lower()
            
            # Check for filename matches
            query_normalized = normalize_filename(query)
            source_normalized = normalize_filename(source)
            filename_normalized = normalize_filename(filename)
            
            # Strong filename match (exact or very close)
            strong_filename_match = (
                query_normalized in source_normalized or 
                query_normalized in filename_normalized or
                source_normalized in query_normalized or
                filename_normalized in query_normalized or
                query_lower.replace(" ", "") in source.replace("_", "").replace("-", "").replace(".", "")
            )
            
            # Weaker filename match (partial)
            weak_filename_match = (
                any(word in source for word in query_lower.split() if len(word) > 2) or
                any(word in filename for word in query_lower.split() if len(word) > 2)
            )
            
            # Apply different thresholds based on match strength
            if strong_filename_match:
                if score <= search_config.STRONG_MATCH_THRESHOLD:
                    relevant_docs.append((doc, score * search_config.STRONG_MATCH_SCORE_BOOST))
            elif weak_filename_match:
                if score <= search_config.WEAK_MATCH_THRESHOLD:
                    relevant_docs.append((doc, score * search_config.WEAK_MATCH_SCORE_BOOST))
            elif score <= score_threshold:
                relevant_docs.append((doc, score))
        
        # Sort by score (lower is better in FAISS)
        relevant_docs.sort(key=lambda x: x[1])
        
        results = []
        for doc, score in relevant_docs:
            results.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "chunk_id": doc.metadata.get("chunk_id", 0),
                "relevance_score": score
            })
        
        return results
    
    def ask(self, question: str) -> Dict[str, Any]:
        """
        Ask a question about the documents using enhanced search logic.
        
        Args:
            question: Natural language question to answer
            
        Returns:
            Dictionary containing 'answer' (string) and 'sources' (list of dicts)
        """
        start_time = time.time()
        
        # Log the incoming question
        self.analytics.log_operation(
            operation_type="question_input",
            operation=f"Question: {question[:100]}",
            metadata={"question": question, "question_length": len(question)},
            status="THINKING"
        )
        
        try:
            if not self.vectorstore:
                return {
                    "answer": "No documents processed yet. Please process documents first.",
                    "sources": []
                }
            
            # Search for relevant documents
            logger.info(f"Starting search for: {question}")
            search_start = time.time()
            search_results = self.search(question, update_status=False)
            search_time = time.time() - search_start
            logger.info(f"Search completed in {search_time:.3f} seconds")
            
            if not search_results:
                return {
                    "answer": "No relevant documents found for your question.",
                    "sources": []
                }
            
            # Build context from search results
            logger.info(f"Preparing context from {len(search_results)} search results")
            context_start = time.time()
            context, sources = self._build_context(search_results)
            context_time = time.time() - context_start
            
            # Log context building
            self.analytics.log_operation(
                operation_type="context_builder",
                operation=f"Built context for: {question[:60]}",
                metadata={
                    "question": question,
                    "num_sources": len(sources),
                    "context_length": len(context),
                    "context_truncated": len(context) >= search_config.MAX_CONTEXT_LENGTH,
                    "build_time_s": round(context_time, 3)
                },
                status="PROCESSING"
            )
            
            logger.info(f"Context preparation completed in {context_time:.3f} seconds")
            
            # Generate answer using LLM
            logger.info(f"Sending prompt to LLM (context length: {len(context)} chars)")
            llm_start = time.time()
            prompt = self.components.prompt_template.format(context=context, question=question)
            
            # Log prompt assembly
            self.analytics.log_operation(
                operation_type="prompt_assembly",
                operation=f"Prompt assembled for: {question[:60]}",
                metadata={
                    "question": question,
                    "context_length": len(context),
                    "prompt_length": len(prompt),
                    "num_sources": len(sources),
                    "model": self.components.model_name
                },
                status="PROCESSING"
            )
            
            logger.info(f"Prompt formatted ({len(prompt)} chars), calling LLM...")
            answer = self.components.llm.invoke(prompt)
            llm_time = time.time() - llm_start
            
            # Log LLM generation
            self.analytics.log_operation(
                operation_type="llm_generation",
                operation=f"LLM response for: {question[:60]}",
                metadata={
                    "question": question,
                    "model": self.components.model_name,
                    "response_length": len(answer),
                    "generation_time_s": round(llm_time, 3),
                    "tokens_per_second": round(len(answer.split()) / llm_time, 2) if llm_time > 0 else 0
                },
                status="PROCESSING"
            )
            
            logger.info(f"LLM response received in {llm_time:.3f} seconds")
            
            total_time = time.time() - start_time
            logger.info(f"Total ask() time: {total_time:.3f} seconds")
            
            # Log complete response
            self.analytics.log_operation(
                operation_type="response_complete",
                operation=f"Response complete for: {question[:60]}",
                metadata={
                    "question": question,
                    "total_time_s": round(total_time, 3),
                    "search_time_s": round(search_time, 3),
                    "context_time_s": round(context_time, 3),
                    "llm_time_s": round(llm_time, 3),
                    "answer_length": len(answer),
                    "num_sources": len(sources)
                },
                status="IDLE"
            )
            
            return {
                "answer": answer,
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"Error asking question: {e}")
            
            self.analytics.log_operation(
                operation_type="error",
                operation=f"Error processing question: {str(e)[:100]}",
                metadata={
                    "question": question,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                },
                status="ERROR"
            )
            
            return {
                "answer": f"Error processing question: {e}",
                "sources": []
            }
    
    def ask_streaming(self, question: str):
        """
        Same as ask() but yields tokens as they're generated.
        
        Args:
            question: Natural language question to answer
            
        Yields:
            String tokens from the LLM response as they are generated
        """
        stream_start = time.time()
        
        # Log streaming question
        self.analytics.log_operation(
            operation_type="question_input",
            operation=f"Question (streaming): {question[:100]}",
            metadata={"question": question, "question_length": len(question), "streaming": True},
            status="THINKING"
        )
        
        try:
            if not self.vectorstore:
                yield "No documents processed yet. Please process documents first."
                return
            
            # Search for relevant documents
            search_results = self.search(question, update_status=False)
            if not search_results:
                yield "No relevant documents found for your question."
                return
            
            # Build context
            context, sources = self._build_context(search_results)
            
            # Log streaming start
            self.analytics.log_operation(
                operation_type="response_stream_start",
                operation=f"Starting stream for: {question[:60]}",
                metadata={
                    "question": question,
                    "num_sources": len(sources),
                    "context_length": len(context)
                },
                status="PROCESSING"
            )
            
            # Stream the response
            prompt = self.components.prompt_template.format(context=context, question=question)
            token_count = 0
            for token in self.components.llm.stream(prompt):
                token_count += 1
                yield token
            
            stream_time = time.time() - stream_start
            
            # Log streaming complete
            self.analytics.log_operation(
                operation_type="response_stream_complete",
                operation=f"Stream complete for: {question[:60]}",
                metadata={
                    "question": question,
                    "stream_time_s": round(stream_time, 3),
                    "token_count": token_count
                },
                status="IDLE"
            )
                
        except Exception as e:
            self.analytics.log_operation(
                operation_type="error",
                operation=f"Streaming error: {str(e)[:100]}",
                metadata={
                    "question": question,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                },
                status="ERROR"
            )
            yield f"Error processing question: {e}"
    
    def _build_context(self, search_results: List[Dict[str, Any]]) -> tuple:
        """
        Build context string and sources list from search results.
        
        Args:
            search_results: List of search result dictionaries
            
        Returns:
            Tuple of (context_string, sources_list)
        """
        context_parts = []
        sources = []
        
        # Use top results for context
        max_sources = search_config.MAX_CONTEXT_SOURCES
        for result in search_results[:max_sources]:
            # Extract clean content (remove filename prefix)
            content = result["content"]
            source_name = result["source"]
            
            # Remove filename prefix to get clean content
            parts = content.split(' ', 2)
            if len(parts) >= 3:
                clean_content = parts[2]
            else:
                clean_content = content
            
            # Add source context
            contextual_content = f"From document '{source_name}':\n{clean_content}"
            context_parts.append(contextual_content)
            
            # Add to sources with preview
            preview_length = search_config.SOURCE_PREVIEW_LENGTH
            sources.append({
                "source": source_name,
                "content": clean_content[:preview_length] + "..." if len(clean_content) > preview_length else clean_content
            })
        
        # Combine context with length limit
        context = "\n\n".join(context_parts)
        max_length = search_config.MAX_CONTEXT_LENGTH
        if len(context) > max_length:
            context = context[:max_length] + "..."
        
        return context, sources
    
    def debug_search(self, query: str) -> Dict[str, Any]:
        """
        Debug search to see what's being retrieved.
        
        Args:
            query: Search query text to debug
            
        Returns:
            Dictionary with query, total_results count, and top 10 results with details
        """
        if not self.vectorstore:
            return {"error": "No vectorstore available"}
        
        try:
            search_results = self.search(query)
            
            results = {
                "query": query,
                "total_results": len(search_results),
                "results": []
            }
            
            for i, result in enumerate(search_results[:10]):
                results["results"].append({
                    "rank": i + 1,
                    "score": result["relevance_score"],
                    "source": result["source"],
                    "chunk_id": result["chunk_id"],
                    "content_preview": result["content"][:100] + "..."
                })
            
            return results
            
        except Exception as e:
            return {"error": str(e)}
