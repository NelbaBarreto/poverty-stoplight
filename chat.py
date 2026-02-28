"""
Chat interface for querying the pre-loaded knowledge base.
Uses Ollama (configurable LLM + embedding model) and pgvector.
Supports two modes: RAG (direct context injection) and Agent (tool-calling).
"""

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from src.tools import create_search_tool, get_query_embedding, EMBED_MODEL_TABLE
from src.agent import create_documentation_agent
from src.pgvector_manager import PGVectorManager

st.set_page_config(
    page_title="Luz - Asistente del Semaforo de Pobreza",
    page_icon="💡",
    layout="centered",
)

# ── LLM models available locally ─────────────────────────────
AVAILABLE_LLM_MODELS = [
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

# ── Embedding model display labels → Ollama model name ───────
EMBED_MODEL_OPTIONS = {
    "bge-m3  (1024d)":               "bge-m3",
    "nomic-embed-text  (768d)":       "nomic-embed-text",
    "mxbai-embed-large  (1024d)":     "mxbai-embed-large",
    "all-minilm  (384d)":             "all-minilm",
    "snowflake-arctic-embed  (1024d)": "snowflake-arctic-embed",
}

# ── Chunk config display labels → config name ────────────────
CHUNK_CONFIG_OPTIONS = {
    "small  (512 chars)":   "small",
    "medium  (1024 chars)": "medium",
    "large  (2048 chars)":  "large",
}

RAG_SYSTEM_PROMPT = """Eres Luz, la asistente conversacional del Banco de Soluciones de la Fundacion Paraguaya.
Responde UNICAMENTE con la informacion proporcionada en el CONTEXTO a continuacion.
Si la informacion no esta en el contexto, dilo claramente.
Cita siempre las fuentes (nombres de archivo) cuando respondas.
No inventes datos. Se concisa pero completa. Responde en espanol."""


def initialize():
    """Initialize session state."""
    defaults = {
        "messages": [],
        "current_llm": AVAILABLE_LLM_MODELS[0],
        "embed_model": "bge-m3",
        "chunk_config": "medium",
        "use_agent_mode": False,
        "agent": None,
        # track last agent config to detect when recreation is needed
        "_agent_llm": None,
        "_agent_embed": None,
        "_agent_chunk": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def search_context(query: str, embed_model: str, chunk_config: str, k: int = 8) -> str:
    """Search pgvector for relevant chunks and return formatted context."""
    try:
        query_embedding = get_query_embedding(embed_model, query)
        embed_table = EMBED_MODEL_TABLE.get(embed_model, "embeddings_bge_m3")

        pgvector_mgr = PGVectorManager()
        results = pgvector_mgr.search_similar_new_schema(
            embedding=query_embedding,
            embed_table=embed_table,
            chunk_config_name=chunk_config,
            k=k,
        )

        if not results:
            return ""

        context_parts = []
        for doc in results:
            meta = doc.metadata or {}
            source = meta.get("filename", "Desconocido")
            fmt = meta.get("format", "")
            dist = meta.get("distance", 0)
            sim = round(1 - float(dist), 3)
            content = doc.page_content.strip()
            if content:
                context_parts.append(
                    f"[Fuente: {source} | formato: {fmt} | sim: {sim}]\n{content}"
                )

        return "\n\n---\n\n".join(context_parts)
    except Exception as e:
        return f"Error en búsqueda: {e}"


def get_agent(llm_name: str, embed_model: str, chunk_config: str):
    """Return cached agent or recreate if any config changed."""
    if (
        st.session_state.agent is not None
        and st.session_state._agent_llm == llm_name
        and st.session_state._agent_embed == embed_model
        and st.session_state._agent_chunk == chunk_config
    ):
        return st.session_state.agent

    search_tool = create_search_tool(embed_model=embed_model, chunk_config=chunk_config)
    agent = create_documentation_agent([search_tool], model_name=llm_name)
    st.session_state.agent = agent
    st.session_state._agent_llm = llm_name
    st.session_state._agent_embed = embed_model
    st.session_state._agent_chunk = chunk_config
    return agent


def render_sidebar():
    """Sidebar: LLM selector, embedding model, chunk size, mode, documents."""
    with st.sidebar:
        st.title("💡 Luz")
        st.caption("Asistente del Banco de Soluciones")

        # ── LLM ──────────────────────────────────────────────
        st.divider()
        st.subheader("Modelo LLM")
        selected_llm = st.selectbox(
            "Modelo de lenguaje:",
            AVAILABLE_LLM_MODELS,
            index=AVAILABLE_LLM_MODELS.index(st.session_state.current_llm),
            key="llm_selector",
        )
        if selected_llm != st.session_state.current_llm:
            st.session_state.current_llm = selected_llm
            st.session_state.agent = None

        # ── Embedding model ───────────────────────────────────
        st.divider()
        st.subheader("Modelo de Embedding")
        embed_labels = list(EMBED_MODEL_OPTIONS.keys())
        current_embed_label = next(
            (lbl for lbl, val in EMBED_MODEL_OPTIONS.items()
             if val == st.session_state.embed_model),
            embed_labels[0],
        )
        selected_embed_label = st.selectbox(
            "Modelo de embedding:",
            embed_labels,
            index=embed_labels.index(current_embed_label),
            key="embed_selector",
            help="Cada modelo genera vectores de distinta dimensión. "
                 "Cambiarlo afecta la calidad y velocidad de búsqueda.",
        )
        selected_embed = EMBED_MODEL_OPTIONS[selected_embed_label]
        if selected_embed != st.session_state.embed_model:
            st.session_state.embed_model = selected_embed
            st.session_state.agent = None

        # ── Chunk size ────────────────────────────────────────
        st.subheader("Tamaño de chunk")
        chunk_labels = list(CHUNK_CONFIG_OPTIONS.keys())
        current_chunk_label = next(
            (lbl for lbl, val in CHUNK_CONFIG_OPTIONS.items()
             if val == st.session_state.chunk_config),
            chunk_labels[1],  # default: medium
        )
        selected_chunk_label = st.selectbox(
            "Configuración de chunk:",
            chunk_labels,
            index=chunk_labels.index(current_chunk_label),
            key="chunk_selector",
            help="Chunks más pequeños son más precisos; "
                 "chunks más grandes dan más contexto por resultado.",
        )
        selected_chunk = CHUNK_CONFIG_OPTIONS[selected_chunk_label]
        if selected_chunk != st.session_state.chunk_config:
            st.session_state.chunk_config = selected_chunk
            st.session_state.agent = None

        # ── Mode ──────────────────────────────────────────────
        st.divider()
        st.subheader("Modo RAG")
        use_agent = st.toggle(
            "Modo Agente (tool-calling)",
            value=st.session_state.use_agent_mode,
            help="Activado: el LLM decide cuándo buscar (requiere modelos grandes). "
                 "Desactivado: siempre busca contexto antes de responder (más confiable).",
        )
        st.session_state.use_agent_mode = use_agent
        if use_agent:
            st.caption("El LLM decide cuándo usar la herramienta de búsqueda")
        else:
            st.caption("Búsqueda automática en cada pregunta (recomendado)")

        # ── Documents ─────────────────────────────────────────
        st.divider()
        st.subheader("Documentos cargados")
        try:
            pgvector_mgr = PGVectorManager()
            docs = pgvector_mgr.get_all_documents()
            if docs:
                st.metric("Documentos", len(docs))
                with st.expander("Ver documentos"):
                    for doc in docs:
                        st.markdown(f"**{doc['filename']}**  \n_{doc.get('file_type','')}_")
            else:
                st.warning("No hay documentos cargados.")
        except Exception as e:
            st.error(f"Error de conexión: {e}")

        st.divider()
        if st.button("Limpiar conversación", use_container_width=True):
            st.session_state.messages = []
            st.session_state.agent = None
            st.rerun()

    return selected_llm, selected_embed, selected_chunk


def stream_rag_response(prompt: str, llm_name: str, embed_model: str, chunk_config: str):
    """RAG mode: search first, then stream LLM response with context."""
    yield "**Buscando en documentos...**\n\n"

    context = search_context(prompt, embed_model, chunk_config)

    if not context:
        yield "No se encontraron documentos relevantes para tu pregunta."
        return

    full_prompt = (
        f"{RAG_SYSTEM_PROMPT}\n\n"
        f"## CONTEXTO (documentos encontrados):\n\n{context}\n\n"
        f"## PREGUNTA DEL USUARIO:\n{prompt}\n\n"
        f"## RESPUESTA:"
    )

    llm = ChatOllama(model=llm_name, temperature=0)

    messages = []
    for msg in st.session_state.messages[-6:]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=full_prompt))

    for chunk in llm.stream(messages):
        if chunk.content:
            yield chunk.content


def stream_agent_response(prompt: str, llm_name: str, embed_model: str, chunk_config: str):
    """Agent mode: let the LLM decide when to use tools."""
    agent = get_agent(llm_name, embed_model, chunk_config)
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


def render_chat(llm_name: str, embed_model: str, chunk_config: str):
    """Main chat interface."""
    st.title("💡 Luz - Asistente del Semaforo de Pobreza")
    mode_label = "Agente" if st.session_state.use_agent_mode else "RAG directo"
    st.caption(
        f"LLM: **{llm_name}** | Embedding: **{embed_model}** | "
        f"Chunk: **{chunk_config}** | Modo: **{mode_label}**"
    )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Escribe tu pregunta...")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            if st.session_state.use_agent_mode:
                full_response = st.write_stream(
                    stream_agent_response(prompt, llm_name, embed_model, chunk_config)
                )
            else:
                full_response = st.write_stream(
                    stream_rag_response(prompt, llm_name, embed_model, chunk_config)
                )
        except Exception as e:
            full_response = f"Error: {e}"
            st.markdown(full_response)

        st.caption(f"LLM: {llm_name} | Embedding: {embed_model} | Chunk: {chunk_config} | Modo: {mode_label}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "llm": llm_name,
        "embed_model": embed_model,
        "chunk_config": chunk_config,
    })


def main():
    initialize()
    llm_name, embed_model, chunk_config = render_sidebar()
    render_chat(llm_name, embed_model, chunk_config)


if __name__ == "__main__":
    main()
