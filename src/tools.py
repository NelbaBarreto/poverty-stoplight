"""
Agent tools for document search and retrieval.
"""
from typing import Annotated
from langchain.tools import tool
from langchain_openai import OpenAIEmbeddings


def create_search_tool(vectorstore):
    """
    Create a search tool that has access to the vector store.

    Args:
        vectorstore: The Chroma vector store containing documents

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
            # Compute embedding for the query then perform similarity search
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            query_embedding = embeddings.embed_query(query)
            results = vectorstore.search_similar(query_embedding, k=8)

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
                    f"📄 **{source} (pág. {page})** | sim: {round(float(score),3)}\n"
                    f"{content}"
                )

            if not context_parts:
                return "No relevant information found."

            return "\n\n---\n\n".join(context_parts)

        except Exception as e:
            print(f"Error during document search: {str(e)}")
            return f"Search error: {str(e)}"

    return search_documents