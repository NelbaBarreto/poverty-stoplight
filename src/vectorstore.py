"""
Vector store management for document storage and retrieval.
"""
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.pgvector_manager import PGVectorManager
from src.embeddings_manager import EmbeddingsManager


class VectorStoreManager:
    """Manages document chunking, embedding, and vector storage."""

    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        """
        Initialize the vector store manager.
        
        Args:
            embedding_model: Name of the embedding model to use
        """
        # Create embeddings provider
        self.embedding_model = embedding_model
        self.embeddings = EmbeddingsManager.create_embeddings(embedding_model)
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
        )
        self.pgvector_manager = PGVectorManager()
        self.vectorstore = None

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks for better retrieval.

        Args:
            documents: List of documents to chunk

        Returns:
            List of chunked documents
        """
        chunks = self.text_splitter.split_documents(documents)
        return chunks

    def create_vectorstore(self, chunks: List[Document], replace_existing: bool = False) -> dict:
        """
        Create a vector store from document chunks.
        Supports both PostgreSQL pgvector and in-memory Chroma.

        Args:
            chunks: List of document chunks
            replace_existing: If True, replace existing chunks for this document-model combination

        Returns:
            Dictionary with save results including 'document_id', 'chunks_saved', 'already_existed'
        """
        print(f"Creating vector store with {len(chunks)} chunks using model: {self.embedding_model}...")

        try:
            # Add embeddings to chunks
            print("Generating embeddings...")
            chunks_with_embeddings = []
            for chunk in chunks:
                embedding = self.embeddings.embed_query(chunk.page_content)
                chunk.metadata["embedding"] = embedding
                chunks_with_embeddings.append(chunk)

            # Save to pgvector
            print("Saving to PostgreSQL pgvector...")
            filename = chunks[0].metadata.get("filename", "unknown")
            file_type = chunks[0].metadata.get("file_type", "unknown")
            print(f"Document: {filename}, Type: {file_type}, Model: {self.embedding_model}")
            result = self.pgvector_manager.save_chunks(
                chunks_with_embeddings, 
                filename, 
                file_type, 
                self.embedding_model,
                replace_existing=replace_existing
            )
            
            if result['already_existed']:
                print(f"Chunks already exist for this document-model combination ({result['existing_count']} chunks)")
            else:
                print(f"Vector store created successfully in PostgreSQL ({result['chunks_saved']} chunks saved)")
            
            return result

        except Exception as e:
            print(f"Error creating vector store: {str(e)}")
            raise

    def search_similar(self, vectorstore: any, query: str, k: int = 4, embedding_model: str = None) -> List[Document]:
        """
        Perform semantic similarity search.

        Args:
            vectorstore: The vector store (PGVectorManager)
            query: Search query
            k: Number of results to return
            embedding_model: Specific embedding model to use for search (optional)

        Returns:
            List of similar documents
        """
        try:
            # Use specified model or default to instance model
            model_to_use = embedding_model or self.embedding_model
            
            # If searching with a different model, create temporary embeddings
            if model_to_use != self.embedding_model:
                temp_embeddings = EmbeddingsManager.create_embeddings(model_to_use)
                query_embedding = temp_embeddings.embed_query(query)
            else:
                query_embedding = self.embeddings.embed_query(query)
            
            # Search in pgvector with model filter
            results = vectorstore.search_similar(
                query_embedding, 
                k=k, 
                embedding_model=model_to_use
            )            
            return results
        except Exception as e:
            print(f"Error searching vector store: {str(e)}")
            return []
    
    def get_embedding_model(self) -> str:
        """Get the current embedding model name."""
        return self.embedding_model
