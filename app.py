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
from src.pgvector_manager import PGVectorManager
from src.embeddings_manager import EmbeddingsManager
from src.ragas_evaluator import RAGASEvaluator


# Page configuration
st.set_page_config(
    page_title="Asistente de Documentos", page_icon="💡", layout="wide"
)


def initialize_session_state():
    """Initialize all session state variables."""
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processing_status" not in st.session_state:
        st.session_state.processing_status = "not_started"
    if "docling_docs" not in st.session_state:
        st.session_state.docling_docs = []
    if "selected_embedding_models" not in st.session_state:
        st.session_state.selected_embedding_models = ["text-embedding-3-small"]
    if "selected_query_model" not in st.session_state:
        st.session_state.selected_query_model = "text-embedding-3-small"
    if "selected_llm_provider" not in st.session_state:
        st.session_state.selected_llm_provider = "openai"
    if "selected_llm_model" not in st.session_state:
        st.session_state.selected_llm_model = "gpt-4o-mini"


def process_and_index(uploaded_files, embedding_models=None, replace_existing=None):
    """Process uploaded documents and create vector store."""
    try:
        # Use selected models or default
        if embedding_models is None:
            embedding_models = st.session_state.selected_embedding_models
        
        if not embedding_models:
            st.error("No hay modelos de embeddings seleccionados")
            return
        
        st.info(f"Usando {len(embedding_models)} modelo(s) de embeddings: **{', '.join(embedding_models)}**")
        
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

        # Step 2: Chunk documents (only once)
        with st.spinner("Dividiendo documentos en fragmentos..."):
            # Use first model just for chunking
            vs_manager = VectorStoreManager(embedding_model=embedding_models[0])
            chunks = vs_manager.chunk_documents(documents)
        
        # Step 2.5: Check for existing chunks before generating embeddings
        pgvector_mgr = PGVectorManager()
        duplicates_found = []
        
        for uploaded_file in uploaded_files:
            filename = uploaded_file.name
            # Get or create document to get its ID
            file_type = uploaded_file.type
            document_id = pgvector_mgr.get_or_create_document(filename, file_type)
            
            for embedding_model in embedding_models:
                # Get embedding model ID
                embedding_model_id = pgvector_mgr.get_or_create_embedding_model(embedding_model)
                
                # Check if chunks already exist
                existing = pgvector_mgr.check_existing_chunks(document_id, embedding_model_id)
                
                if existing['exists']:
                    duplicates_found.append({
                        'filename': filename,
                        'model': embedding_model,
                        'count': existing['count'],
                        'document_id': document_id,
                        'embedding_model_id': embedding_model_id
                    })
        
        # If duplicates found and user hasn't decided yet, ask
        if duplicates_found and replace_existing is None:
            st.warning("Se encontraron chunks existentes para las siguientes combinaciones documento-modelo:")
            
            for dup in duplicates_found:
                st.write(f"- **{dup['filename']}** con modelo **{dup['model']}** ({dup['count']} chunks)")
            
            st.info("¿Deseas reemplazar los chunks existentes?")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Sí, reemplazar", type="primary"):
                    st.session_state.replace_decision = True
                    st.rerun()
            with col2:
                if st.button("No, omitir duplicados"):
                    st.session_state.replace_decision = False
                    st.rerun()
            
            return  # Stop here and wait for user decision
        
        # Use the decision from session state if available
        if replace_existing is None and 'replace_decision' in st.session_state:
            replace_existing = st.session_state.replace_decision
            # Clear the decision after using it
            del st.session_state.replace_decision
        
        # Step 3: Generate embeddings with each selected model
        results_summary = []
        for embedding_model in embedding_models:
            with st.spinner(f"Generando embeddings con {embedding_model}..."):
                # Create a new manager for this model
                vs_manager = VectorStoreManager(embedding_model=embedding_model)
                # Generate embeddings and save to database
                result = vs_manager.create_vectorstore(chunks, replace_existing=replace_existing or False)
                
                if result['already_existed'] and not replace_existing:
                    st.info(f"⏭Omitiendo {embedding_model}: ya existe ({result['existing_count']} chunks)")
                elif result['chunks_saved'] > 0:
                    st.success(f"{result['chunks_saved']} chunks guardados con modelo {embedding_model}")
                    results_summary.append({
                        'model': embedding_model,
                        'chunks': result['chunks_saved']
                    })

        # Step 4: Extract and save document structure for each document
        with st.spinner("Guardando estructura de documentos..."):
            for docling_doc_data in docling_docs:
                try:
                    # Extract structure
                    visualizer = DocumentStructureVisualizer(docling_doc_data['doc'])
                    structure = visualizer.export_full_structure()
                    
                    # Get document ID from database by filename
                    all_docs = pgvector_mgr.get_all_documents()
                    doc_record = next(
                        (d for d in all_docs if d['filename'] == docling_doc_data['filename']),
                        None
                    )
                    
                    if doc_record:
                        # Save structure
                        pgvector_mgr.save_document_structure(doc_record['id'], structure)
                        st.info(f"Estructura guardada para: {docling_doc_data['filename']}")
                except Exception as e:
                    st.warning(f"No se pudo guardar estructura para {docling_doc_data['filename']}: {str(e)}")

        # Step 5: Crear agente
        with st.spinner("Creando agente..."):
            # Use the selected query model for the search tool
            query_model = st.session_state.selected_query_model
            # Ensure query model is in the list of selected models
            if query_model not in embedding_models:
                query_model = embedding_models[0]
                st.session_state.selected_query_model = query_model
            
            search_tool = create_search_tool(embedding_model=query_model)
            agent = create_documentation_agent(
                [search_tool],
                model_name=st.session_state.selected_llm_model,
                provider=st.session_state.selected_llm_provider
            )
            st.session_state.agent = agent
            st.info(f"🔍 Agente creado con modelo de consulta: {query_model}")

        st.session_state.processing_status = "completed"
        
        if results_summary:
            st.success(f"Indexación completada: {len(results_summary)} modelo(s) procesado(s)")
        else:
            st.info("Indexación completada (no se guardaron nuevos chunks)")

    except Exception as e:
        st.error(f"Error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        st.session_state.processing_status = "error"


def render_sidebar():
    """Render the sidebar with setup controls."""
    with st.sidebar:
        st.title("Configuración")
        
        # Embedding model selector
        st.subheader("Modelos de Embeddings")
        
        # Get all available models
        all_models = EmbeddingsManager.list_all_models()
        
        st.write("Selecciona uno o más modelos:")
        
        # Initialize selected models in session state if needed
        if "selected_embedding_models" not in st.session_state:
            st.session_state.selected_embedding_models = ["text-embedding-3-small"]
        
        # Create checkboxes for each model
        selected_models = []
        for model in all_models:
            # Check if model should be checked by default
            is_checked = model['name'] in st.session_state.selected_embedding_models
            
            if st.checkbox(
                f"{model['name']}",
                value=is_checked,
                key=f"embed_model_{model['name']}",
                help=f"{model['description']} - {model['provider']} ({model['dimension']} dim)"
            ):
                selected_models.append(model['name'])
        
        # Update session state
        st.session_state.selected_embedding_models = selected_models
        
        # Show count of selected models
        if selected_models:
            st.success(f"{len(selected_models)} modelo(s) seleccionado(s)")
            
            # Show details of selected models in expander
            with st.expander("Ver detalles de modelos seleccionados"):
                for model_name in selected_models:
                    model_info = next((m for m in all_models if m['name'] == model_name), None)
                    if model_info:
                        st.write(f"**{model_name}**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"Proveedor: {model_info['provider']}")
                        with col2:
                            st.write(f"Dimensión: {model_info['dimension']}")
                        
                        # Show endpoint info for HuggingFace models
                        if model_info['provider'] == 'huggingface':
                            from src.embeddings_manager import EMBEDDING_MODELS
                            model_config = EMBEDDING_MODELS['huggingface'].get(model_info['name'], {})
                            endpoint = model_config.get('endpoint_url')
                            if endpoint:
                                st.write(f"Endpoint: {endpoint[:40]}...")
                        st.divider()
        else:
            st.warning("Selecciona al menos un modelo")
        
        # Query model selector
        st.subheader("Modelo para Consultas")
        st.write("Selecciona el modelo a usar para búsquedas en el chat:")
        
        # Get models that have been indexed in the database
        try:
            pgvector_mgr = PGVectorManager()
            indexed_models = pgvector_mgr.get_all_embedding_models()
            indexed_model_names = [m['model_name'] for m in indexed_models]
            
            # Filter to only show models that are both selected and indexed
            available_query_models = [m for m in selected_models if m in indexed_model_names]
            
            if not available_query_models:
                # If no overlap, show all indexed models
                available_query_models = indexed_model_names
            
            if available_query_models:
                # Ensure selected_query_model is in the list
                if st.session_state.selected_query_model not in available_query_models:
                    st.session_state.selected_query_model = available_query_models[0]
                
                current_index = available_query_models.index(st.session_state.selected_query_model) if st.session_state.selected_query_model in available_query_models else 0
                
                selected_query_model = st.selectbox(
                    "Modelo de búsqueda:",
                    options=available_query_models,
                    index=current_index,
                    key="query_model_selector",
                    help="Este modelo se usará para generar embeddings de tus consultas y buscar documentos relevantes"
                )
                
                # Check if model changed and invalidate agent
                if selected_query_model != st.session_state.selected_query_model:
                    st.session_state.selected_query_model = selected_query_model
                    # Reset agent to force recreation with new model
                    st.session_state.agent = None
                    st.info(f"Modelo de consulta cambiado a: {selected_query_model}")
                
                st.session_state.selected_query_model = selected_query_model
                
                # Show info about selected query model
                query_model_info = next((m for m in all_models if m['name'] == selected_query_model), None)
                if query_model_info:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Proveedor", query_model_info['provider'])
                    with col2:
                        st.metric("Dimensión", query_model_info['dimension'])
            else:
                st.warning("No hay modelos indexados. Procesa documentos primero.")
        except Exception as e:
            st.warning(f"No se pudieron cargar modelos indexados: {str(e)}")
            # Fallback to first selected model
            if selected_models:
                st.session_state.selected_query_model = selected_models[0]

        st.divider()
        
        # LLM model selector
        st.subheader("Modelo LLM (Chat)")
        
        # Provider selector
        llm_provider = st.radio(
            "Proveedor:",
            options=["openai", "huggingface"],
            index=0 if st.session_state.selected_llm_provider == "openai" else 1,
            horizontal=True,
            help="Selecciona el proveedor del modelo de lenguaje para el chat"
        )
        st.session_state.selected_llm_provider = llm_provider
        
        # Model selector based on provider
        if llm_provider == "openai":
            llm_models = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
            selected_llm = st.selectbox(
                "Modelo:",
                options=llm_models,
                index=llm_models.index(st.session_state.selected_llm_model) if st.session_state.selected_llm_model in llm_models else 1,
                help="Modelo OpenAI para el chat"
            )
            st.session_state.selected_llm_model = selected_llm
        else:  # huggingface
            # Common LLM models
            llm_models = [
                "Qwen/Qwen2.5-72B-Instruct",
                "meta-llama/Llama-3.1-8B-Instruct", 
                "meta-llama/Llama-3.1-70B-Instruct",
                "mistralai/Mistral-7B-Instruct-v0.3",
                "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "HuggingFaceH4/zephyr-7b-beta"
            ]
            
            selected_llm = st.selectbox(
                "Modelo:",
                options=llm_models,
                index=0,
                help="Modelo HuggingFace para el chat"
            )
            st.session_state.selected_llm_model = selected_llm
            
            # Show endpoint info
            from src.agent import get_hf_llm_endpoint_for_model
            endpoint = get_hf_llm_endpoint_for_model(selected_llm)
            if endpoint:
                st.success(f"Endpoint: {endpoint[:40]}...")
            else:
                st.info("API pública")

        st.divider()

        # Show saved documents from database
        st.subheader("Documentos guardados")
        try:
            pgvector_mgr = PGVectorManager()
            all_docs = pgvector_mgr.get_all_documents()
            
            if all_docs:
                st.success(f"{len(all_docs)} documento(s) disponible(s)")
                with st.expander("Ver documentos"):
                    for doc in all_docs:
                        st.write(f"📄 **{doc['filename']}**")
                        
                        # Show embedding models for this document
                        if doc.get('embedding_models'):
                            st.write("   Modelos de embeddings:")
                            for model_info in doc['embedding_models']:
                                if model_info:  # Check if not None
                                    st.write(f"   - {model_info['model_name']} ({model_info['provider']}) - {model_info['chunk_count']} chunks")
                        else:
                            st.write("Sin modelos de embeddings asociados")
                        
                        st.write(f"   Total chunks: {doc.get('chunk_count', 0)}")
                        st.divider()
            else:
                st.info("No hay documentos en la BD")
        except Exception as e:
            st.warning(f"Error al cargar documentos: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

        st.divider()

        # File uploader for new documents
        st.subheader("Subir nuevos documentos")
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
            with st.expander("Archivos a procesar"):
                for file in uploaded_files:
                    st.write(f"- {file.name} ({file.type})")

            # Process button
            if st.button("Procesar e indexar"):
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
        with st.expander("Consejos y Configuración"):
            st.markdown(
                """
            **Cómo usar:**
            1. Selecciona uno o más modelos de embeddings (con checkboxes)
            2. Selecciona el modelo a usar para consultas en el chat
            3. Selecciona el modelo LLM (chat)
            4. Sube documentos nuevos
            5. Se generarán embeddings con TODOS los modelos seleccionados
            6. Los datos se guardan en PostgreSQL (pgvector-db)
            7. El chat busca usando el modelo de consulta seleccionado

            **Formatos compatibles:**
            - Documentos PDF
            - Documentos de Word (.docx)
            - Presentaciones PowerPoint (.pptx)
            - Archivos HTML
            
            **Modelos de embeddings:**
            - OpenAI: Modelos comerciales de alta calidad
            - HuggingFace: Modelos open-source gratuitos vía Inference API
            - **Indexación:** Puedes seleccionar múltiples modelos para generar embeddings en paralelo
            - **Consultas:** Selecciona el modelo específico para búsquedas en el chat
            - **Flexibilidad:** Cambia el modelo de consulta en cualquier momento
            
            **Múltiples Inference Endpoints:**
            
            Puedes configurar diferentes endpoints para cada modelo:
            
            1. **Endpoint genérico** (para todos los modelos):
               ```
               HF_EMBEDDING_ENDPOINT_URL=https://xxx.aws.endpoints.huggingface.cloud
               HF_LLM_ENDPOINT_URL=https://yyy.aws.endpoints.huggingface.cloud
               ```
            
            2. **Endpoint específico por modelo** (prioridad sobre el genérico):
               ```
               HF_EMBEDDING_ENDPOINT_SENTENCE_TRANSFORMERS_ALL_MINILM_L6_V2=https://...
               HF_LLM_ENDPOINT_QWEN_QWEN2_5_72B_INSTRUCT=https://...
               ```
            
            Ver **HUGGINGFACE_ENDPOINTS.md** para documentación completa.
            """
            )


def render_structure_viz():
    """Render document structure visualization."""
    st.title("Estructura del documento")
    manager = PGVectorManager()

    # Select source: session (docling) or persisted (Postgres)
    source = st.radio("Fuente de documentos:", ["Sesión", "Base de datos"], horizontal=True)

    if source == "Sesión":
        if not st.session_state.docling_docs:
            st.info("Por favor sube y procesa tus documentos primero para ver su estructura.")
            return

        # Document selector from session
        doc_names = [doc['filename'] for doc in st.session_state.docling_docs]
        selected_doc_name = st.selectbox("Selecciona el documento a analizar:", doc_names)

        # Get selected document
        selected_doc_data = next(
            (doc for doc in st.session_state.docling_docs if doc['filename'] == selected_doc_name),
            None
        )

        if not selected_doc_data:
            return

        # Create visualizer (full feature set available for session docs)
        visualizer = DocumentStructureVisualizer(selected_doc_data['doc'])

        # Display structure in tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Resumen", "Jerarquía", "Tablas", "Imágenes"])

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
            st.dataframe(text_types_df)

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
                        st.dataframe(table_data['dataframe'])
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
                        st.image(pic_data['pil_image'])
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

    else:
        # Base de datos: show persisted documents and their structure
        try:
            saved_docs = manager.get_all_documents()
        except Exception as e:
            st.error(f"Error al obtener documentos desde la base de datos: {str(e)}")
            return

        if not saved_docs:
            st.info("No hay documentos guardados en la base de datos.")
            return

        # Build selection options with id and filename
        options = [f"{d['id']} - {d['filename']}" for d in saved_docs]
        selected = st.selectbox("Selecciona un documento guardado:", options)
        # Extract id
        doc_id = int(selected.split(" - ")[0])
        doc_meta = next((d for d in saved_docs if d['id'] == doc_id), None)

        if not doc_meta:
            st.error("Documento no encontrado")
            return

        st.subheader(f"Documento: {doc_meta['filename']}")
        st.write(f"ID: {doc_meta['id']} — Tipo: {doc_meta.get('file_type')} — Creado: {doc_meta.get('created_at')} — Chunks: {doc_meta.get('chunk_count')}")

        # Fetch structure from database
        try:
            structure = manager.get_document_structure(doc_id)
        except Exception as e:
            st.error(f"Error al obtener estructura: {str(e)}")
            return

        # Display structure in tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Resumen", "Jerarquía", "Tablas", "Imágenes", "Fragmentos"])

        with tab1:
            st.subheader("Resumen del documento")
            if 'summary' in structure:
                summary = structure['summary']
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Páginas", summary.get('num_pages', 0))
                with col2:
                    st.metric("Elementos de texto", summary.get('num_texts', 0))
                with col3:
                    st.metric("Tablas", summary.get('num_tables', 0))
                with col4:
                    st.metric("Imágenes", summary.get('num_pictures', 0))

                st.subheader("Tipos de contenido")
                text_types = summary.get('text_types', {})
                if text_types:
                    text_types_df = pd.DataFrame([
                        {'Type': k, 'Count': v}
                        for k, v in sorted(text_types.items(), key=lambda x: -x[1])
                    ])
                    st.dataframe(text_types_df)
            else:
                st.info("No hay datos de resumen disponibles")

        with tab2:
            st.subheader("Jerarquía del documento")
            hierarchy = structure.get('hierarchy', [])
            if hierarchy:
                for item in hierarchy:
                    indent = "  " * (item.get('level', 0) - 1)
                    page = item.get('page')
                    st.markdown(f"{indent}**{item.get('text')}** _(Page {page})_")
            else:
                st.info("No se encontró estructura jerárquica")

        with tab3:
            st.subheader("Tablas")
            tables = structure.get('tables', [])
            if tables:
                for table_data in tables:
                    st.markdown(f"### Tabla {table_data.get('table_number')} (Página {table_data.get('page')})")

                    if table_data.get('caption'):
                        st.caption(table_data['caption'])

                    if 'dataframe' in table_data:
                        st.dataframe(table_data['dataframe'])
                    else:
                        st.info(f"Tabla con forma {table_data.get('shape')}")

                    st.divider()
            else:
                st.info("No se encontraron tablas en este documento")

        with tab4:
            st.subheader("Imágenes")
            pictures = structure.get('pictures', [])
            if pictures:
                for pic_data in pictures:
                    st.markdown(f"**Imagen {pic_data.get('picture_number')}** (Página {pic_data.get('page')})")

                    if pic_data.get('caption'):
                        st.caption(pic_data['caption'])

                    # Show bounding box info
                    if pic_data.get('bounding_box'):
                        bbox = pic_data['bounding_box']
                        with st.expander("📐 Detalles de posición"):
                            st.text(f"Posición: ({bbox.get('left', 0):.1f}, {bbox.get('top', 0):.1f}) - ({bbox.get('right', 0):.1f}, {bbox.get('bottom', 0):.1f})")

                    st.divider()
            else:
                st.info("No se encontraron imágenes en este documento")

        with tab5:
            st.subheader("Fragmentos de texto")
            # Fetch chunks for document
            try:
                chunks = manager.get_chunks_by_document(doc_id)
            except Exception as e:
                st.error(f"Error al obtener fragmentos: {str(e)}")
                return

            st.markdown(f"**Fragmentos encontrados:** {len(chunks)}")

            if chunks:
                # Prepare a simple dataframe preview
                rows = []
                for c in chunks:
                    meta = c.metadata or {}
                    page = meta.get('page') if isinstance(meta, dict) else None
                    rows.append({
                        'chunk_id': meta.get('chunk_id', ''),
                        'chunk_index': meta.get('chunk_index', ''),
                        'page': page,
                        'preview': (c.page_content[:300] + '...') if len(c.page_content) > 300 else c.page_content
                    })

                try:
                    df = pd.DataFrame(rows)
                    st.dataframe(df)
                except Exception:
                    for r in rows:
                        st.write(r)

                # Expanders to view full chunk text
                for i, c in enumerate(chunks):
                    with st.expander(f"Fragmento {i} (ID: {c.metadata.get('chunk_id', '')})"):
                        st.write(c.page_content)
            else:
                st.info("No hay fragmentos disponibles")



def render_chat():
    """Render the chat interface."""
    # Check if agent is ready
    if st.session_state.agent is None:
        # Try to create agent if there are documents in BD
        try:
            pgvector_mgr = PGVectorManager()
            all_docs = pgvector_mgr.get_all_documents()
            
            if all_docs:
                # Crear agente automáticamente si hay documentos
                # Usar el modelo de consulta seleccionado
                query_model = st.session_state.selected_query_model
                search_tool = create_search_tool(embedding_model=query_model)
                agent = create_documentation_agent(
                    [search_tool],
                    model_name=st.session_state.selected_llm_model,
                    provider=st.session_state.selected_llm_provider
                )
                st.session_state.agent = agent
            else:
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
        except Exception as e:
            st.error(f"Error al verificar documentos: {str(e)}")
            return

    # Crear agente si no existe y hay documentos en BD
    if st.session_state.agent is None:
        try:
            query_model = st.session_state.selected_query_model
            search_tool = create_search_tool(embedding_model=query_model)
            agent = create_documentation_agent(
                [search_tool],
                model_name=st.session_state.selected_llm_model,
                provider=st.session_state.selected_llm_provider
            )
            st.session_state.agent = agent
        except Exception as e:
            st.error(f"Error al crear agente: {str(e)}")
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
                    status_placeholder.markdown("**Pensando...**")
                    first_content_token = True
                    tool_call_detected = False
                    final_answer_started = False
                    token_count = 0
                    
                    try:
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
                                        "**Buscando en documentos...**"
                                    )
                                    tool_call_detected = True
                                continue  # Skip all tool messages

                            # Only stream content from the "agent" node (the LLM's response)
                            # if "agent" in langgraph_node.lower() and hasattr(
                            #     msg, "content"
                            # ):
                            if hasattr(
                                msg, "content"
                            ):
                                content = msg.content

                                # Only yield non-empty content tokens
                                if content:
                                    token_count += 1
                                    
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
                        
                        if token_count == 0:
                            print(f"*El agente no generó respuesta. Verifica los logs.*")
                            yield "Lo siento, no pude generar una respuesta en este momento."

                    except Exception as e:
                        print(f"[DEBUG ERROR] Excepción en generate_response: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        yield f"Error en stream: {str(e)}"

                # Use st.write_stream for automatic token-by-token display
                with message_placeholder.container():
                    full_response = st.write_stream(generate_response())

            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                print(f"[DEBUG ERROR] Chat error completo:\n{error_details}")
                error_msg = f"Error: {str(e)}\n\n*Revisa la consola (terminal) para ver logs detallados.*"
                status_placeholder.empty()
                message_placeholder.markdown(error_msg)
                full_response = error_msg

        # Add assistant response to history
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )

def render_model_comparison():
    """Render the model comparison interface using RAGAS."""
    st.title("🔬 Comparación de Modelos de Embeddings")
    
    st.markdown("""
    Esta herramienta permite comparar el rendimiento de diferentes modelos de embeddings 
    usando métricas RAGAS (Retrieval-Augmented Generation Assessment).
    """)
    
    # Get available models
    all_models = EmbeddingsManager.list_all_models()
    pgvector_mgr = PGVectorManager()
    
    # Get documents from database
    try:
        all_docs = pgvector_mgr.get_all_documents()
    except Exception as e:
        st.error(f"Error al cargar documentos: {str(e)}")
        return
    
    if not all_docs:
        st.warning("⚠️ No hay documentos en la base de datos. Sube y procesa documentos primero.")
        return
    
    # Configuration section
    st.subheader("⚙️ Configuración de Evaluación")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Select models to compare
        st.write("**Selecciona modelos para comparar:**")
        
        selected_models = []
        for model in all_models:
            if st.checkbox(
                f"{model['name']} ({model['provider']})",
                key=f"model_{model['name']}"
            ):
                selected_models.append(model['name'])
    
    with col2:
        # Select document (optional)
        st.write("**Documento a evaluar (opcional):**")
        doc_options = ["Todos los documentos"] + [f"{d['id']} - {d['filename']}" for d in all_docs]
        selected_doc_option = st.selectbox(
            "Documento:",
            options=doc_options,
            help="Puedes evaluar contra un documento específico o todos"
        )
        
        document_id = None
        if selected_doc_option != "Todos los documentos":
            document_id = int(selected_doc_option.split(" - ")[0])
        
        # Number of test questions
        num_questions = st.slider(
            "Número de preguntas de prueba:",
            min_value=3,
            max_value=20,
            value=5,
            help="Más preguntas = evaluación más precisa pero más lenta"
        )
        
        # Number of context chunks to retrieve
        k_value = st.slider(
            "Chunks a recuperar (k):",
            min_value=2,
            max_value=10,
            value=4
        )
    
    # Run evaluation button
    st.divider()
    
    if st.button("🚀 Ejecutar Evaluación", type="primary", disabled=len(selected_models) == 0):
        if len(selected_models) == 0:
            st.error("Por favor selecciona al menos un modelo para evaluar")
            return
        
        st.info(f"Evaluando {len(selected_models)} modelo(s) con {num_questions} preguntas...")
        
        # Create evaluator
        try:
            evaluator = RAGASEvaluator()
            
            # Run comparison
            with st.spinner("Ejecutando evaluaciones... Esto puede tomar varios minutos."):
                results = evaluator.compare_embedding_models(
                    models=selected_models,
                    document_id=document_id,
                    test_cases=None,  # Will generate synthetic questions
                    k=k_value
                )
            
            # Display results
            st.success("✅ Evaluación completada!")
            
            # Create comparison table
            st.subheader("📊 Resultados de Comparación")
            
            comparison_data = []
            for model_name, metrics in results.items():
                if "error" not in metrics:
                    comparison_data.append({
                        "Modelo": model_name,
                        "Faithfulness": f"{metrics.get('faithfulness', 0):.3f}",
                        "Answer Relevancy": f"{metrics.get('answer_relevancy', 0):.3f}",
                        "Context Precision": f"{metrics.get('context_precision', 0):.3f}",
                        "Context Recall": f"{metrics.get('context_recall', 0):.3f}",
                        "Tiempo Promedio (s)": f"{metrics.get('average_retrieval_time', 0):.3f}",
                    })
                else:
                    st.error(f"Error en {model_name}: {metrics['error']}")
            
            if comparison_data:
                df = pd.DataFrame(comparison_data)
                st.dataframe(df, use_container_width=True)
                
                # Visualizations
                st.subheader("📈 Visualizaciones")
                
                # Create metrics comparison chart
                import plotly.graph_objects as go
                
                metrics_to_plot = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
                
                fig = go.Figure()
                
                for metric in metrics_to_plot:
                    values = []
                    models = []
                    for model_name, metrics in results.items():
                        if "error" not in metrics and metric in metrics:
                            values.append(metrics[metric])
                            models.append(model_name.split("/")[-1][:20])  # Shorten name
                    
                    if values:
                        fig.add_trace(go.Bar(
                            name=metric.replace("_", " ").title(),
                            x=models,
                            y=values
                        ))
                
                fig.update_layout(
                    title="Comparación de Métricas RAGAS",
                    xaxis_title="Modelo",
                    yaxis_title="Score",
                    barmode='group',
                    yaxis=dict(range=[0, 1])
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Retrieval time comparison
                fig2 = go.Figure()
                
                times = []
                models = []
                for model_name, metrics in results.items():
                    if "error" not in metrics:
                        times.append(metrics.get('average_retrieval_time', 0))
                        models.append(model_name.split("/")[-1][:20])
                
                fig2.add_trace(go.Bar(
                    x=models,
                    y=times,
                    marker_color='lightblue'
                ))
                
                fig2.update_layout(
                    title="Tiempo Promedio de Recuperación",
                    xaxis_title="Modelo",
                    yaxis_title="Tiempo (segundos)"
                )
                
                st.plotly_chart(fig2, use_container_width=True)
                
                # Best model recommendation
                try:
                    best_model = evaluator.get_best_model(results, "answer_relevancy")
                    st.success(f"🏆 **Modelo Recomendado (mejor Answer Relevancy):** {best_model}")
                except Exception as e:
                    st.warning(f"No se pudo determinar el mejor modelo: {str(e)}")
        
        except Exception as e:
            st.error(f"Error durante la evaluación: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    # Show historical evaluations
    st.divider()
    st.subheader("📜 Evaluaciones Históricas")
    
    try:
        evaluations = pgvector_mgr.get_ragas_evaluations()
        
        if evaluations:
            eval_data = []
            for ev in evaluations:
                eval_data.append({
                    "Fecha": ev.get('evaluation_date'),
                    "Modelo": ev.get('model_name'),
                    "Documento": ev.get('filename'),
                    "Faithfulness": f"{ev.get('faithfulness', 0):.3f}" if ev.get('faithfulness') else "N/A",
                    "Answer Relevancy": f"{ev.get('answer_relevancy', 0):.3f}" if ev.get('answer_relevancy') else "N/A",
                    "Context Precision": f"{ev.get('context_precision', 0):.3f}" if ev.get('context_precision') else "N/A",
                    "Context Recall": f"{ev.get('context_recall', 0):.3f}" if ev.get('context_recall') else "N/A",
                })
            
            df_hist = pd.DataFrame(eval_data)
            st.dataframe(df_hist, use_container_width=True)
        else:
            st.info("No hay evaluaciones históricas disponibles.")
    
    except Exception as e:
        st.warning(f"No se pudieron cargar evaluaciones históricas: {str(e)}")

def main():
    """Main application function."""
    initialize_session_state()
    render_sidebar()

    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["💬 Conversación", "📊 Estructura del documento", "🔬 Comparación de Modelos"])

    with tab1:
        render_chat()

    with tab2:
        render_structure_viz()
    
    with tab3:
        render_model_comparison()


if __name__ == "__main__":
    main()