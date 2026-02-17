"""
Chat interface for querying the pre-loaded knowledge base.
Uses Ollama (gpt-oss:20b for LLM, bge-m3 for embeddings) and pgvector.
"""

import streamlit as st
from langchain_core.messages import HumanMessage
from src.tools import create_search_tool
from src.agent import create_documentation_agent
from src.pgvector_manager import PGVectorManager

st.set_page_config(
    page_title="Luz - Asistente del Semaforo de Pobreza",
    page_icon="💡",
    layout="centered",
)


def initialize():
    """Initialize session state and agent."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent" not in st.session_state:
        st.session_state.agent = None


def get_agent():
    """Create the agent once and cache it in session state."""
    if st.session_state.agent is not None:
        return st.session_state.agent

    search_tool = create_search_tool()
    agent = create_documentation_agent([search_tool])
    st.session_state.agent = agent
    return agent


def render_sidebar():
    """Sidebar with document info and controls."""
    with st.sidebar:
        st.title("💡 Luz")
        st.caption("Asistente del Banco de Soluciones")

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
            st.rerun()


def render_chat():
    """Main chat interface."""
    st.title("💡 Luz - Asistente del Semaforo de Pobreza")
    st.caption(
        "Preguntame sobre el Semaforo de Eliminacion de Pobreza, "
        "el Banco de Soluciones o cualquier documento cargado."
    )

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    prompt = st.chat_input("Escribe tu pregunta...")

    if not prompt:
        return

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent and generate response
    with st.chat_message("assistant"):
        status = st.empty()
        message_area = st.empty()

        try:
            agent = get_agent()
            config = {"configurable": {"thread_id": "chat_session"}}

            def stream_response():
                status.markdown("**Pensando...**")
                tool_call_seen = False
                first_token = True

                for msg, metadata in agent.stream(
                    {"messages": [HumanMessage(content=prompt)]},
                    config=config,
                    stream_mode="messages",
                ):
                    node = metadata.get("langgraph_node", "")

                    # Skip tool execution messages
                    if "tool" in node.lower():
                        if not tool_call_seen:
                            status.markdown("**Buscando en documentos...**")
                            tool_call_seen = True
                        continue

                    # Stream LLM content tokens
                    if hasattr(msg, "content") and msg.content:
                        if first_token:
                            status.empty()
                            first_token = False
                        yield msg.content

                if first_token:
                    status.empty()
                    yield "No pude generar una respuesta. Intenta reformular tu pregunta."

            with message_area.container():
                full_response = st.write_stream(stream_response())

        except Exception as e:
            status.empty()
            full_response = f"Error: {str(e)}"
            message_area.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})


def main():
    initialize()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
