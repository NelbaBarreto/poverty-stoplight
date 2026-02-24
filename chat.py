"""
Chat interface for querying the pre-loaded knowledge base.
Uses Ollama (configurable LLM, bge-m3 for embeddings) and pgvector.
Supports two modes: RAG (direct context injection) and Agent (tool-calling).
"""

import requests
import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from src.tools import create_search_tool
from src.agent import create_documentation_agent
from src.pgvector_manager import PGVectorManager

st.set_page_config(
    page_title="Luz - Asistente del Semaforo de Pobreza",
    page_icon="💡",
    layout="centered",
)

OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "bge-m3"

AVAILABLE_MODELS = [
    "gpt-oss:20b",
    "deepseek-r1:32b",
    "deepseek-r1:14b",
    "deepseek-r1:8b",
    "llama3.2:latest",
    "llama3.2:3b",
    "llama3.1:8b",
    "llama3.1:latest",
    "llama3:8b",
    "llama3:latest",
    "llama2:latest",
    "mistral:latest",
    "gemma:latest",
    "codellama:latest",
    "codegemma:latest",
    "deepcoder:latest",
]

RAG_SYSTEM_PROMPT = """Eres Luz, la asistente conversacional del Banco de Soluciones de la Fundacion Paraguaya.
Responde UNICAMENTE con la informacion proporcionada en el CONTEXTO a continuacion.
Si la informacion no esta en el contexto, dilo claramente.
Cita siempre las fuentes (nombres de archivo) cuando respondas.
No inventes datos. Se concisa pero completa. Responde en espanol."""


def initialize():
    """Initialize session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_model" not in st.session_state:
        st.session_state.current_model = AVAILABLE_MODELS[0]
    if "use_agent_mode" not in st.session_state:
        st.session_state.use_agent_mode = False
    if "agent" not in st.session_state:
        st.session_state.agent = None


def search_context(query: str, k: int = 8) -> str:
    """Search pgvector for relevant chunks and return formatted context."""
    try:
        resp = requests.post(OLLAMA_URL, json={"model": EMBED_MODEL, "prompt": query})
        resp.raise_for_status()
        query_embedding = resp.json()["embedding"]

        pgvector_mgr = PGVectorManager()
        results = pgvector_mgr.search_similar(query_embedding, k=k)

        if not results:
            return ""

        context_parts = []
        for doc in results:
            metadata = doc.metadata or {}
            source = metadata.get("filename", metadata.get("source", "Desconocido"))
            page = metadata.get("page", "?")
            distance = metadata.get("distance", 0)
            content = doc.page_content.strip()
            if content:
                context_parts.append(
                    f"[Fuente: {source}, pag. {page}, sim: {1 - float(distance):.3f}]\n{content}"
                )

        return "\n\n---\n\n".join(context_parts)
    except Exception as e:
        return f"Error en busqueda: {str(e)}"


def get_agent(model_name: str):
    """Create the agent, recreating if the model changed."""
    if (
        st.session_state.agent is not None
        and st.session_state.current_model == model_name
    ):
        return st.session_state.agent

    search_tool = create_search_tool()
    agent = create_documentation_agent([search_tool], model_name=model_name)
    st.session_state.agent = agent
    st.session_state.current_model = model_name
    return agent


def render_sidebar():
    """Sidebar with model selector, mode toggle, document info and controls."""
    with st.sidebar:
        st.title("💡 Luz")
        st.caption("Asistente del Banco de Soluciones")

        st.divider()

        # Model selector
        st.subheader("Modelo LLM")
        selected_model = st.selectbox(
            "Seleccionar modelo:",
            AVAILABLE_MODELS,
            index=AVAILABLE_MODELS.index(st.session_state.current_model),
            key="model_selector",
        )

        if selected_model != st.session_state.current_model:
            st.session_state.agent = None
            st.session_state.current_model = selected_model

        st.caption(f"Activo: **{st.session_state.current_model}**")

        # Mode selector
        st.divider()
        st.subheader("Modo RAG")
        use_agent = st.toggle(
            "Modo Agente (tool-calling)",
            value=st.session_state.use_agent_mode,
            help="Activado: el LLM decide cuando buscar (requiere modelos grandes). "
                 "Desactivado: siempre busca contexto antes de responder (mas confiable).",
        )
        st.session_state.use_agent_mode = use_agent

        if use_agent:
            st.caption("El LLM decide cuando usar la herramienta de busqueda")
        else:
            st.caption("Busqueda automatica en cada pregunta (recomendado)")

        st.divider()

        # Show loaded documents
        st.subheader("Documentos cargados")
        try:
            pgvector_mgr = PGVectorManager()
            docs = pgvector_mgr.get_all_documents()
            if docs:
                total_chunks = sum(d.get("chunk_count", 0) for d in docs)
                st.metric("Documentos", len(docs))
                st.metric("Fragmentos indexados", total_chunks)
                with st.expander("Ver documentos"):
                    for doc in docs:
                        chunks = doc.get("chunk_count", 0)
                        st.markdown(f"**{doc['filename']}**  \n{chunks} fragmentos")
            else:
                st.warning("No hay documentos cargados.")
        except Exception as e:
            st.error(f"Error de conexion: {e}")

        st.divider()

        if st.button("Limpiar conversacion", use_container_width=True):
            st.session_state.messages = []
            st.session_state.agent = None
            st.rerun()

    return selected_model


def stream_rag_response(prompt: str, model_name: str):
    """RAG mode: search first, then stream LLM response with context."""
    yield "**Buscando en documentos...**\n\n"

    context = search_context(prompt)

    if not context:
        yield "No se encontraron documentos relevantes para tu pregunta."
        return

    full_prompt = (
        f"{RAG_SYSTEM_PROMPT}\n\n"
        f"## CONTEXTO (documentos encontrados):\n\n{context}\n\n"
        f"## PREGUNTA DEL USUARIO:\n{prompt}\n\n"
        f"## RESPUESTA:"
    )

    llm = ChatOllama(model=model_name, temperature=0)

    # Build message history for context
    messages = []
    for msg in st.session_state.messages[-6:]:  # Last 3 exchanges for context
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            from langchain_core.messages import AIMessage
            messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=full_prompt))

    for chunk in llm.stream(messages):
        if chunk.content:
            yield chunk.content


def stream_agent_response(prompt: str, model_name: str):
    """Agent mode: let the LLM decide when to use tools."""
    agent = get_agent(model_name)
    config = {"configurable": {"thread_id": "chat_session"}}

    tool_call_seen = False
    first_token = True

    for msg, metadata in agent.stream(
        {"messages": [HumanMessage(content=prompt)]},
        config=config,
        stream_mode="messages",
    ):
        node = metadata.get("langgraph_node", "")

        if "tool" in node.lower():
            if not tool_call_seen:
                yield "**Buscando en documentos...**\n\n"
                tool_call_seen = True
            continue

        if hasattr(msg, "content") and msg.content:
            if first_token:
                first_token = False
            yield msg.content

    if first_token:
        yield "No pude generar una respuesta. Intenta reformular tu pregunta."


def render_chat(model_name: str):
    """Main chat interface."""
    st.title("💡 Luz - Asistente del Semaforo de Pobreza")
    mode_label = "Agente" if st.session_state.use_agent_mode else "RAG directo"
    st.caption(
        f"Modelo: **{model_name}** | Modo: **{mode_label}**"
    )

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "model" in message:
                st.caption(f"Modelo: {message['model']}")

    # Chat input
    prompt = st.chat_input("Escribe tu pregunta...")

    if not prompt:
        return

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        try:
            if st.session_state.use_agent_mode:
                full_response = st.write_stream(
                    stream_agent_response(prompt, model_name)
                )
            else:
                full_response = st.write_stream(
                    stream_rag_response(prompt, model_name)
                )
        except Exception as e:
            full_response = f"Error: {str(e)}"
            st.markdown(full_response)

        st.caption(f"Modelo: {model_name} | Modo: {mode_label}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "model": model_name,
    })


def main():
    initialize()
    selected_model = render_sidebar()
    render_chat(selected_model)


if __name__ == "__main__":
    main()
