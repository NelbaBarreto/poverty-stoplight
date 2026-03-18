"""
Agent tools for document search and retrieval.
"""
from typing import Annotated, Optional
import requests
from langchain.tools import tool
from src.pgvector_manager import PGVectorManager

OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "bge-m3"

EMBED_MODEL_TABLE = {
    "bge-m3":                  "embeddings_bge_m3",
    "nomic-embed-text":        "embeddings_nomic",
    "mxbai-embed-large":       "embeddings_mxbai",
    "all-minilm":              "embeddings_minilm",
    "snowflake-arctic-embed":  "embeddings_snowflake",
}


def get_query_embedding(embed_model: str, query: str) -> list:
    """Get embedding vector for a query using Ollama."""
    resp = requests.post(OLLAMA_URL, json={"model": embed_model, "prompt": query})
    resp.raise_for_status()
    return resp.json()["embedding"]

def create_search_tool(document_id: Optional[int] = None):
    """
    Create a search tool that retrieves from PostgreSQL pgvector database.

    Args:
        document_id: Optional document ID to filter search results

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
            # Compute embedding for the query using Ollama bge-m3
            resp = requests.post(OLLAMA_URL, json={"model": EMBED_MODEL, "prompt": query})
            resp.raise_for_status()
            query_embedding = resp.json()["embedding"]
            
            # Search directly in PostgreSQL pgvector
            pgvector_mgr = PGVectorManager()
            results = pgvector_mgr.search_similar(query_embedding, k=8, document_id=document_id)

            if not results:
                return "No relevant information found."

            context_parts = []

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

            if not context_parts:
                return "No relevant information found."

            return "\n\n---\n\n".join(context_parts)

        except Exception as e:
            print(f"Error during document search: {str(e)}")
            return f"Search error: {str(e)}"

    return search_documents