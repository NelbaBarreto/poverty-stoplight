"""
Agent tools for document search and retrieval.
"""
from typing import Annotated
from langchain.tools import tool


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
            # Perform similarity search
            results = vectorstore.search_similar(query, k=8)

            if not results:
                return "No relevant information found."

            context_parts = []

            for i, (doc, score) in enumerate(results, 1):
                source = doc.metadata.get(
                    "filename",
                    doc.metadata.get("source", "Unknown source")
                )

                content = doc.page_content.strip()

                # Evita chunks vacíos
                if not content:
                    continue

                context_parts.append(
                    f"[Source {i}: {source} | similarity: {round(score, 3)}]\n"
                    f"{content}"
                )

            if not context_parts:
                return "No relevant information found."

            return "\n\n---\n\n".join(context_parts)

        except Exception as e:
            print(f"Error during document search: {str(e)}")
            return f"Search error: {str(e)}"

    return search_documents
