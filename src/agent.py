"""
LangGraph agent configuration and setup.
"""
import os
import re
from typing import List, Literal, Optional
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver


def get_hf_llm_endpoint_for_model(model_name: str) -> Optional[str]:
    """
    Get the HuggingFace LLM endpoint URL for a specific model.
    
    Looks for endpoint in the following order:
    1. Model-specific env var: HF_LLM_ENDPOINT_{SAFE_MODEL_NAME}
    2. Generic env var: HF_LLM_ENDPOINT_URL
    3. Returns None if not found
    
    Args:
        model_name: Full model name (e.g., "meta-llama/Llama-3.1-8B-Instruct")
    
    Returns:
        Endpoint URL or None
    
    Examples:
        For model "meta-llama/Llama-3.1-8B-Instruct":
        - Looks for: HF_LLM_ENDPOINT_META_LLAMA_LLAMA_3_1_8B_INSTRUCT
        - Then looks for: HF_LLM_ENDPOINT_URL
    """
    # Convert model name to safe env var name
    safe_model_name = re.sub(r'[^a-zA-Z0-9]', '_', model_name).upper()
    
    # Try model-specific endpoint first
    specific_var = f"HF_LLM_ENDPOINT_{safe_model_name}"
    endpoint = os.getenv(specific_var)
    if endpoint:
        return endpoint
    
    # Try generic endpoint
    endpoint = os.getenv("HF_LLM_ENDPOINT_URL")
    if endpoint:
        return endpoint
    
    return None

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

def create_documentation_agent(
    tools: List[BaseTool], 
    model_name: str = "gpt-4o-mini",
    provider: Literal["openai", "huggingface"] = "openai",
    endpoint_url: str = None
):
    """
    Create a document intelligence assistant agent using LangGraph.

    Args:
        tools: List of tools the agent can use
        model_name: Name of the model to use (e.g., "gpt-4o-mini" for OpenAI or model path for HF)
        provider: Model provider - "openai" or "huggingface"
        endpoint_url: Custom HuggingFace inference endpoint URL 
                     (e.g., "https://xxx.us-east-1.aws.endpoints.huggingface.cloud")
                     Only used when provider="huggingface". If None, will look for model-specific
                     or generic endpoint in env vars

    Returns:
        A configured LangGraph agent
    """
    # Initialize the language model based on provider
    if provider == "openai":
        llm = ChatOpenAI(model=model_name, temperature=0)
    elif provider == "huggingface":
        # Get endpoint URL with priority: parameter > model-specific env > generic env
        if endpoint_url:
            hf_endpoint = endpoint_url
        else:
            hf_endpoint = get_hf_llm_endpoint_for_model(model_name)
        
        hf_token = os.getenv("HUGGINGFACE_API_TOKEN") or os.getenv("HF_TOKEN")
        
        if not hf_token:
            raise ValueError("HUGGINGFACE_API_TOKEN or HF_TOKEN required for HuggingFace provider")
        
        if hf_endpoint:
            # Use custom inference endpoint
            print(f"Using HuggingFace LLM Endpoint for {model_name}: {hf_endpoint[:50]}...")
            llm_endpoint = HuggingFaceEndpoint(
                endpoint_url=hf_endpoint,
                huggingfacehub_api_token=hf_token,
                task="text-generation",
                temperature=0.1,
                max_new_tokens=2048,
            )
        else:
            # Use public Inference API with model name
            print(f"Using HuggingFace public API for model: {model_name}")
            llm_endpoint = HuggingFaceEndpoint(
                repo_id=model_name,
                huggingfacehub_api_token=hf_token,
                task="text-generation",
                temperature=0.1,
                max_new_tokens=2048,
            )
        
        # Wrap with ChatHuggingFace for chat completion
        llm = ChatHuggingFace(llm=llm_endpoint)
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'huggingface'")

    # Create a memory saver for conversation history
    memory = MemorySaver()

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=memory
    )

    return agent
