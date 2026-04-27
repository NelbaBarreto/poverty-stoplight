"""
LangGraph agent configuration and setup.
"""
from typing import List
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

# SYSTEM_PROMPT = """You are a helpful document intelligence assistant. You have access to documents that have been uploaded and processed (PDFs, Word documents, presentations, HTML files, etc.).

# GUIDELINES:
# - Use the search_documents tool to find relevant information
# - Be efficient: one well-crafted search is usually sufficient
# - Only search again if the first results are clearly incomplete
# - Provide clear, accurate answers based on the document contents
# - Always cite your sources with filenames or document titles
# - If information isn't found, say so clearly
# - Be concise but thorough

# When answering:
# 1. Search the documents with a focused query
# 2. Synthesize a clear answer from the results
# 3. Include source citations (filenames)
# 4. Only search again if absolutely necessary
# """

from pathlib import Path as _Path

def _load_agent_prompt() -> str:
    f = _Path(__file__).parent.parent / "prompt.txt"
    if f.exists():
        return f.read_text(encoding="utf-8").strip()
    return "Eres Rosa, la asistente conversacional del Banco de Soluciones de la Fundación Paraguaya."

SYSTEM_PROMPT = _load_agent_prompt()

def create_documentation_agent(tools: List[BaseTool], model_name: str = "gpt-oss:20b"):
    """
    Create a document intelligence assistant agent using LangGraph.

    Args:
        tools: List of tools the agent can use
        model_name: Name of the Ollama model to use

    Returns:
        A configured LangGraph agent
    """
    # Initialize the language model via Ollama
    llm = ChatOllama(model=model_name, temperature=0)

    # Create a memory saver for conversation history
    memory = MemorySaver()

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=memory
    )

    return agent
