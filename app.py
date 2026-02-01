"""
Streamlit app for converting documents into a chatbot using Docling and LangGraph.
"""

import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from streamlit_extras.bottom_container import bottom

# Load environment variables
load_dotenv()

# Import our modules
from src.document_processor import DocumentProcessor
from src.vectorstore import VectorStoreManager
from src.tools import create_search_tool
from src.agent import create_documentation_agent
from src.structure_visualizer import DocumentStructureVisualizer


# Page configuration
st.set_page_config(
    page_title="Asistente de Documentos", page_icon="📄", layout="wide"
)


def initialize_session_state():
    """Initialize all session state variables."""
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processing_status" not in st.session_state:
        st.session_state.processing_status = "not_started"
    if "docling_docs" not in st.session_state:
        st.session_state.docling_docs = []


def process_and_index(uploaded_files):
    """Process uploaded documents and create vector store."""
    try:
        # Step 1: Process documents with Docling
        with st.spinner(
            f"Procesando {len(uploaded_files)} documento(s) con Docling..."
        ):
            processor = DocumentProcessor()
            documents, docling_docs = processor.process_uploaded_files(uploaded_files)
            st.session_state.docling_docs = docling_docs

        if not documents:
            st.error(
                "No se procesaron documentos. Verifica los archivos e inténtalo de nuevo."
            )
            return

        # Step 2: Chunk and create vector store (with pgvector persistence)
        with st.spinner("✂️ Dividiendo documentos en fragmentos..."):
            vs_manager = VectorStoreManager(use_pgvector=True)
            chunks = vs_manager.chunk_documents(documents)

        with st.spinner("🔢 Creando vector store y guardando en pgvector-db..."):
            vectorstore = vs_manager.create_vectorstore(chunks)
            st.session_state.vectorstore = vectorstore

        # Step 3: Crear agente
        with st.spinner("🤖 Creando agente..."):
            search_tool = create_search_tool(vectorstore)
            agent = create_documentation_agent([search_tool])
            st.session_state.agent = agent

        st.session_state.processing_status = "completed"
        st.success("Documentos indexados en pgvector-db. Ya puedes chatear con ellos abajo.")

    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.session_state.processing_status = "error"


def render_sidebar():
    """Render the sidebar with setup controls."""
    with st.sidebar:
        st.title("⚙️ Configuración")

        # File uploader
        uploaded_files = st.file_uploader(
            "Subir documentos",
            type=["pdf", "docx", "pptx", "html"],
            accept_multiple_files=True,
            help="Sube archivos PDF, Word (DOCX), PowerPoint (PPTX) o HTML",
        )

        # Show uploaded files count
        if uploaded_files:
            st.info(f"{len(uploaded_files)} archivo(s) subido(s)")

            # List uploaded files
            with st.expander("📁 Archivos subidos"):
                for file in uploaded_files:
                    st.write(f"- {file.name} ({file.type})")

            # Process button
            if st.button("🚀 Procesar e indexar", use_container_width=True):
                st.session_state.uploaded_files = uploaded_files
                process_and_index(uploaded_files)

        # Status indicator
        st.divider()
        st.subheader("Estado")

        if st.session_state.processing_status == "not_started":
            st.info("Listo para iniciar")
        elif st.session_state.processing_status == "completed":
            st.success("Listo para chatear")
        elif st.session_state.processing_status == "error":
            st.error("Ocurrió un error")

        # Tips
        with st.expander("💡 Consejos"):
            st.markdown(
                """
            **Formatos compatibles:**
            - Documentos PDF
            - Documentos de Word (.docx)
            - Presentaciones PowerPoint (.pptx)
            - Archivos HTML

            **Buenas prácticas:**
            - Sube documentos relacionados juntos
            - Comienza con pocos documentos para pruebas
            - Los documentos se procesan con OCR para contenido escaneado
            - Se preservan tablas y estructura

            **Para producción:**
            - Agrega almacenamiento persistente de vectores
            - Implementa procesamiento por lotes
            - Usa aceleración por GPU para mayor rapidez
            - Añade autenticación y controles de acceso
            """
            )


def render_structure_viz():
    """Render document structure visualization."""
    st.title("Estructura del documento")

    if not st.session_state.docling_docs:
        st.info("👈 Por favor sube y procesa tus documentos primero para ver su estructura.")
        return

    # Document selector
    doc_names = [doc['filename'] for doc in st.session_state.docling_docs]
    selected_doc_name = st.selectbox("Selecciona el documento a analizar:", doc_names)

    # Get selected document
    selected_doc_data = next(
        (doc for doc in st.session_state.docling_docs if doc['filename'] == selected_doc_name),
        None
    )

    if not selected_doc_data:
        return

    # Create visualizer
    visualizer = DocumentStructureVisualizer(selected_doc_data['doc'])

    # Display structure in tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📑 Resumen", "🏗️ Jerarquía", "Tablas", "🖼️ Imágenes"])

    with tab1:
        st.subheader("Resumen del documento")
        summary = visualizer.get_document_summary()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Páginas", summary['num_pages'])
        with col2:
            st.metric("Tablas", summary['num_tables'])
        with col3:
            st.metric("Imágenes", summary['num_pictures'])
        with col4:
            st.metric("Elementos de texto", summary['num_texts'])

        st.subheader("Tipos de contenido")
        text_types_df = pd.DataFrame([
            {'Type': k, 'Count': v}
            for k, v in sorted(summary['text_types'].items(), key=lambda x: -x[1])
        ])
        st.dataframe(text_types_df, use_container_width=True)

    with tab2:
        st.subheader("Jerarquía del documento")
        hierarchy = visualizer.get_document_hierarchy()

        if hierarchy:
            for item in hierarchy:
                indent = "  " * (item['level'] - 1)
                st.markdown(f"{indent}**{item['text']}** _(Page {item['page']})_")
        else:
            st.info("No se detectó estructura jerárquica")

    with tab3:
        st.subheader("Tablas")
        tables_info = visualizer.get_tables_info()

        if tables_info:
            for table_data in tables_info:
                st.markdown(f"### Tabla {table_data['table_number']} (Página {table_data['page']})")

                if table_data['caption']:
                    st.caption(table_data['caption'])

                if not table_data['is_empty']:
                    st.dataframe(table_data['dataframe'], use_container_width=True)
                else:
                    st.info("La tabla está vacía")

                st.divider()
        else:
            st.info("No se encontraron tablas en este documento")

    with tab4:
        st.subheader("Imágenes")
        pictures_info = visualizer.get_pictures_info()

        if pictures_info:
            for pic_data in pictures_info:
                st.markdown(f"**Imagen {pic_data['picture_number']}** (Página {pic_data['page']})")

                if pic_data['caption']:
                    st.caption(pic_data['caption'])

                # Display the actual image if available
                if pic_data['pil_image'] is not None:
                    st.image(pic_data['pil_image'], use_container_width=True)
                else:
                    st.info("Datos de la imagen no disponibles")

                # Show bounding box info
                if pic_data['bounding_box']:
                    bbox = pic_data['bounding_box']
                    with st.expander("📐 Detalles de posición"):
                        st.text(f"Posición: ({bbox['left']:.1f}, {bbox['top']:.1f}) - ({bbox['right']:.1f}, {bbox['bottom']:.1f})")

                st.divider()
        else:
            st.info("No se encontraron imágenes en este documento")


def render_chat():
    """Render the chat interface."""
    # Check if agent is ready
    if st.session_state.agent is None:
        st.info("Por favor sube y procesa tus documentos en la barra lateral primero!")
        st.markdown(
            """
        ### Cómo usar:
        1. Sube tus documentos en la barra lateral (PDF, DOCX, PPTX o HTML)
        2. Haz clic en "Procesar e indexar" y espera a que termine el procesamiento
        3. ¡Comienza a hacer preguntas sobre tus documentos!

        ### Qué puedes hacer:
        - Hacer preguntas sobre el contenido de los documentos
        - Comparar información entre varios documentos
        - Extraer datos o insights específicos
        - Resumir secciones de documentos
        """
        )
        return

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input in bottom container (attempt to fix positioning in tabs)
    with bottom():
        prompt = st.chat_input("Haz una pregunta sobre tus documentos...")

    if prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get agent response
        with st.chat_message("assistant"):
            # Create status and message placeholders
            status_placeholder = st.empty()
            message_placeholder = st.empty()

            try:
                # Create config with thread ID for conversation memory
                config = {"configurable": {"thread_id": "document_chat"}}

                # Generator function for real-time streaming
                def generate_response():
                    """Generator that yields tokens from LangGraph stream."""
                    status_placeholder.markdown("🤔 **Pensando...**")
                    first_content_token = True
                    tool_call_detected = False
                    final_answer_started = False

                    # Stream with "messages" mode for real LLM tokens
                    for msg, metadata in st.session_state.agent.stream(
                        {"messages": [HumanMessage(content=prompt)]},
                        config=config,
                        stream_mode="messages",
                    ):
                        # Check which node is streaming
                        langgraph_node = metadata.get("langgraph_node", "")

                        # Skip tool outputs entirely (they contain the search results, not answer tokens)
                        if (
                            "tools" in langgraph_node.lower()
                            or "tool" in langgraph_node.lower()
                        ):
                            if not tool_call_detected:
                                status_placeholder.markdown(
                                    "🔍 **Buscando en documentos...**"
                                )
                                tool_call_detected = True
                            continue  # Skip all tool messages

                        # Only stream content from the "agent" node (the LLM's response)
                        if "agent" in langgraph_node.lower() and hasattr(
                            msg, "content"
                        ):
                            content = msg.content

                            # Only yield non-empty content tokens
                            if content:
                                # Update status on first content token
                                if first_content_token:
                                    status_placeholder.markdown(
                                        "**Generando respuesta...**"
                                    )
                                    first_content_token = False
                                    final_answer_started = True

                                # Yield the token only if we're in final answer mode
                                if final_answer_started:
                                    yield content

                    # Clear status when streaming is complete
                    status_placeholder.empty()

                # Use st.write_stream for automatic token-by-token display
                with message_placeholder.container():
                    full_response = st.write_stream(generate_response())

            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                print(f"Chat error: {error_details}")  # Log to console
                error_msg = f"Error: {str(e)}"
                status_placeholder.empty()
                message_placeholder.markdown(error_msg)
                full_response = error_msg

        # Add assistant response to history
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )


def main():
    """Main application function."""
    initialize_session_state()
    render_sidebar()

    # Create tabs for different views
    tab1, tab2 = st.tabs(["Conversación", "Estructura del documento"])

    with tab1:
        render_chat()

    with tab2:
        render_structure_viz()


if __name__ == "__main__":
    main()