"""
Agent tools for document search and retrieval.
"""
from typing import Annotated, Optional, Dict, Any
from langchain.tools import tool
from src.pgvector_manager import PGVectorManager
from src.embeddings_manager import EmbeddingsManager

# Global variable to store last search results for UI display
_last_search_results = []

def get_last_search_results():
    """Get the last search results for display."""
    return _last_search_results

def create_search_tool(document_id: Optional[int] = None, embedding_model: str = "text-embedding-3-small"):
    """
    Create a search tool that retrieves from PostgreSQL pgvector database.

    Args:
        document_id: Optional document ID to filter search results
        embedding_model: Embedding model to use for search

    Returns:
        A tool function that can search the documents
    """

    @tool
    def search_documents(
        query: Annotated[str, "Search query to look up information inside the uploaded documents"]
    ) -> str:
        """
        Search the uploaded documents for relevant information.

        Use this tool when you need to find specific information from the uploaded documents
        to answer user questions.
        """

        try:
            # Compute embedding for the query using the specified model
            embeddings = EmbeddingsManager.create_embeddings(embedding_model)
            query_embedding = embeddings.embed_query(query)
            
            # Search directly in PostgreSQL pgvector
            pgvector_mgr = PGVectorManager()
            results = pgvector_mgr.search_similar(
                query_embedding, 
                k=8, 
                document_id=document_id,
                embedding_model=embedding_model
            )

            if not results:
                return "No relevant information found."

            context_parts = []
            chunks_info = []  # Store structured info for UI display

            for i, item in enumerate(results, 1):

                # Manejar (doc, score) o solo doc
                if isinstance(item, tuple) and len(item) == 2:
                    doc, score = item
                else:
                    doc = item
                    score = (
                        doc.metadata.get("distance")
                        or doc.metadata.get("similarity")
                        or 0
                    )

                metadata = doc.metadata or {}

                source = metadata.get(
                    "filename",
                    metadata.get("source", "Unknown source")
                )

                # página
                page = metadata.get("page", "?")

                content = doc.page_content.strip()

                if not content:
                    continue

                context_parts.append(
                    f"**{source} (pág. {page})** | sim: {round(float(score),3)}\n"
                    f"{content}"
                )
                
                # Store structured info for UI
                chunks_info.append({
                    "rank": i,
                    "source": source,
                    "page": page,
                    "similarity": round(float(score), 3),
                    "content": content,
                    "chunk_id": metadata.get("chunk_id"),
                    "model_name": metadata.get("model_name", embedding_model)
                })

            if not context_parts:
                return "No relevant information found."
            
            # Store results globally for UI display
            global _last_search_results
            _last_search_results = chunks_info

            return "\n\n---\n\n".join(context_parts)

        except Exception as e:
            print(f"Error during document search: {str(e)}")
            return f"Search error: {str(e)}"

    return search_documents