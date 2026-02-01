"""
Vector store management for document storage and retrieval.
"""
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from src.pgvector_manager import PGVectorManager


class VectorStoreManager:
    """Manages document chunking, embedding, and vector storage."""

    def __init__(self, use_pgvector: bool = True):
        """
        Initialize the vector store manager.
        
        Args:
            use_pgvector: If True, use PostgreSQL pgvector for storage.
                         If False, use in-memory Chroma.
        """
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
        )
        self.use_pgvector = use_pgvector
        if use_pgvector:
            self.pgvector_manager = PGVectorManager()
        self.vectorstore = None  # For Chroma (fallback)

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks for better retrieval.

        Args:
            documents: List of documents to chunk

        Returns:
            List of chunked documents
        """
        print(f"Chunking {len(documents)} documents...")
        chunks = self.text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks")
        return chunks

    def create_vectorstore(self, chunks: List[Document]) -> any:
        """
        Create a vector store from document chunks.
        Supports both PostgreSQL pgvector and in-memory Chroma.

        Args:
            chunks: List of document chunks

        Returns:
            Vector store instance (Chroma or PGVectorManager)
        """
        print(f"🔢 Creating vector store with {len(chunks)} chunks...")

        try:
            if self.use_pgvector:
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
                self.pgvector_manager.save_chunks(chunks_with_embeddings, filename, file_type)
                
                print("Vector store created successfully in PostgreSQL")
                return self.pgvector_manager

            else:
                # Use Chroma (in-memory)
                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=self.embeddings,
                    collection_name="documents"
                )
                self.vectorstore = vectorstore
                print("Vector store created successfully with Chroma")
                return vectorstore

        except Exception as e:
            print(f"Error creating vector store: {str(e)}")
            raise

    def search_similar(self, vectorstore: any, query: str, k: int = 4) -> List[Document]:
        """
        Perform semantic similarity search.

        Args:
            vectorstore: The vector store (Chroma or PGVectorManager)
            query: Search query
            k: Number of results to return

        Returns:
            List of similar documents
        """
        try:
            if self.use_pgvector and isinstance(vectorstore, PGVectorManager):
                # Generate embedding for the query
                query_embedding = self.embeddings.embed_query(query)
                # Search in pgvector
                results = vectorstore.search_similar(query_embedding, k=k)
            else:
                # Use Chroma
                results = vectorstore.similarity_search(query, k=k)
            
            return results
        except Exception as e:
            print(f"Error searching vector store: {str(e)}")
            return []

    def get_vectorstore_type(self) -> str:
        """Get the type of vector store being used."""
        return "pgvector" if self.use_pgvector else "chroma"
