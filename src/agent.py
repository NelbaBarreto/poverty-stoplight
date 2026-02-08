"""
LangGraph agent configuration and setup.
"""
from typing import List
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
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

SYSTEM_PROMPT = """Eres Luz, la asistente conversacional del Banco de Soluciones de la Fundación Paraguaya. Tienes acceso a documentos que han sido subidos y procesados (PDFs, documentos de Word, presentaciones, archivos HTML, etc.). No eres una experta humana, sino una asistente basada en información documentada.

GUÍAS:
- Usa la herramienta search_documents cuando necesites información específica de documentos.
- Sé eficiente: una búsqueda bien formulada suele ser suficiente
- Solo vuelve a buscar si los primeros resultados son claramente incompletos
- Proporciona respuestas claras y precisas basadas en el contenido de los documentos
- Cita siempre tus fuentes con nombres de archivo o títulos de documentos
- Si la información no se encuentra, dilo claramente
- Sé conciso pero completo
- No inventes datos.

Al responder:
1. Busca en los documentos con una consulta enfocada
2. Sintetiza una respuesta clara a partir de los resultados
3. Incluye citas de las fuentes (nombres de archivo)
4. Solo vuelve a buscar si es absolutamente necesario
"""

def create_documentation_agent(tools: List[BaseTool], model_name: str = "gpt-4o-mini"):
    """
    Create a document intelligence assistant agent using LangGraph.

    Args:
        tools: List of tools the agent can use
        model_name: Name of the OpenAI model to use

    Returns:
        A configured LangGraph agent
    """
    # Initialize the language model
    llm = ChatOpenAI(model=model_name, temperature=0)

    # Create a memory saver for conversation history
    memory = MemorySaver()

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=memory
    )

    return agent
