"""
PostgreSQL pgvector integration for storing and retrieving document chunks.
"""
import os
from typing import List, Optional
from langchain_core.documents import Document
import psycopg2
from psycopg2.extras import execute_values, RealDictCursor
import json


class PGVectorManager:
    """Manages document chunk storage and retrieval in PostgreSQL with pgvector."""

    def __init__(self):
        """Initialize connection to PostgreSQL."""
        self.host = os.getenv("DB_HOST", "localhost")
        self.port = os.getenv("DB_PORT", "5432")
        self.database = os.getenv("DB_NAME", "fundacion_py_db")
        self.user = os.getenv("DB_USER", "postgres")
        self.password = os.getenv("DB_PASSWORD", "12356789")

    def get_connection(self):
        """Get a connection to the PostgreSQL database."""
        try:
            conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
            return conn
        except psycopg2.OperationalError as e:
            print(f"❌ Error connecting to database: {str(e)}")
            raise

    def save_chunks(self, chunks: List[Document], filename: str, file_type: str) -> int:
        """
        Save document chunks with embeddings to PostgreSQL.

        Args:
            chunks: List of document chunks with embeddings
            filename: Name of the original document
            file_type: Type of the document (pdf, docx, etc.)

        Returns:
            Document ID in the database
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            # Step 1: Insert document metadata
            cursor.execute(
                """
                INSERT INTO documents (filename, file_type)
                VALUES (%s, %s)
                RETURNING id
                """,
                (filename, file_type)
            )
            document_id = cursor.fetchone()[0]
            print(f"📝 Created document record with ID: {document_id}")

            # Step 2: Prepare chunk data
            chunk_data = []
            for i, chunk in enumerate(chunks):
                # Extract embedding if it exists in the document metadata
                embedding = chunk.metadata.get("embedding", None)
                
                chunk_data.append((
                    document_id,
                    chunk.page_content,
                    embedding,
                    i,
                    json.dumps({
                        "filename": chunk.metadata.get("filename", filename),
                        "source": chunk.metadata.get("source", ""),
                        "page": chunk.metadata.get("page", 0),
                    })
                ))

            # Step 3: Insert chunks
            if chunk_data:
                execute_values(
                    cursor,
                    """
                    INSERT INTO chunks (document_id, chunk_text, embedding, chunk_index, metadata)
                    VALUES %s
                    ON CONFLICT DO NOTHING
                    """,
                    chunk_data,
                    template=None
                )
                conn.commit()
                print(f"✅ Saved {len(chunk_data)} chunks for document '{filename}'")

            return document_id

        except Exception as e:
            conn.rollback()
            print(f"❌ Error saving chunks: {str(e)}")
            raise
        finally:
            cursor.close()
            conn.close()

    def save_chunks_batch(self, all_chunks: List[tuple], files_metadata: List[dict]) -> List[int]:
        """
        Save multiple document chunks in batch.

        Args:
            all_chunks: List of tuples (chunks, filename, file_type)
            files_metadata: List of file metadata dicts

        Returns:
            List of document IDs
        """
        document_ids = []
        for chunks, filename, file_type in zip(all_chunks, files_metadata, files_metadata):
            doc_id = self.save_chunks(chunks, filename, file_type.get("file_type"))
            document_ids.append(doc_id)
        return document_ids

    def search_similar(self, embedding: List[float], k: int = 4, document_id: Optional[int] = None) -> List[Document]:
        """
        Search for similar chunks using vector similarity.

        Args:
            embedding: Query embedding vector
            k: Number of results to return
            document_id: Optional filter by document ID

        Returns:
            List of similar documents
        """
        conn = self.get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        try:
            # Format embedding as pgvector format
            embedding_str = "[" + ",".join(str(e) for e in embedding) + "]"

            query = """
            SELECT 
                id,
                document_id,
                chunk_text,
                metadata,
                embedding <-> %s::vector AS distance
            FROM chunks
            """
            
            params = [embedding_str]
            
            if document_id:
                query += " WHERE document_id = %s"
                params.append(document_id)
            
            query += """
            ORDER BY distance
            LIMIT %s
            """
            params.append(k)

            cursor.execute(query, params)
            results = cursor.fetchall()

            # Convert results to Document objects
            documents = []
            for row in results:
                metadata = json.loads(row["metadata"]) if row["metadata"] else {}
                doc = Document(
                    page_content=row["chunk_text"],
                    metadata={
                        **metadata,
                        "chunk_id": row["id"],
                        "document_id": row["document_id"],
                        "distance": float(row["distance"])
                    }
                )
                documents.append(doc)

            return documents

        except Exception as e:
            print(f"❌ Error searching similar chunks: {str(e)}")
            return []
        finally:
            cursor.close()
            conn.close()

    def get_chunks_by_document(self, document_id: int) -> List[Document]:
        """
        Retrieve all chunks for a specific document.

        Args:
            document_id: ID of the document

        Returns:
            List of document chunks
        """
        conn = self.get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        try:
            cursor.execute(
                """
                SELECT id, document_id, chunk_text, metadata, chunk_index
                FROM chunks
                WHERE document_id = %s
                ORDER BY chunk_index
                """,
                (document_id,)
            )
            results = cursor.fetchall()

            documents = []
            for row in results:
                metadata = json.loads(row["metadata"]) if row["metadata"] else {}
                doc = Document(
                    page_content=row["chunk_text"],
                    metadata={
                        **metadata,
                        "chunk_id": row["id"],
                        "chunk_index": row["chunk_index"]
                    }
                )
                documents.append(doc)

            return documents

        except Exception as e:
            print(f"❌ Error retrieving chunks: {str(e)}")
            return []
        finally:
            cursor.close()
            conn.close()

    def delete_document(self, document_id: int) -> bool:
        """
        Delete a document and all its chunks.

        Args:
            document_id: ID of the document to delete

        Returns:
            True if successful, False otherwise
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(
                "DELETE FROM documents WHERE id = %s",
                (document_id,)
            )
            conn.commit()
            print(f"✅ Deleted document {document_id} and its chunks")
            return True
        except Exception as e:
            conn.rollback()
            print(f"❌ Error deleting document: {str(e)}")
            return False
        finally:
            cursor.close()
            conn.close()

    def get_all_documents(self) -> List[dict]:
        """
        Get all stored documents.

        Returns:
            List of document metadata
        """
        conn = self.get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        try:
            cursor.execute(
                """
                SELECT d.id, d.filename, d.file_type, d.created_at,
                       COUNT(c.id) as chunk_count
                FROM documents d
                LEFT JOIN chunks c ON d.id = c.document_id
                GROUP BY d.id, d.filename, d.file_type, d.created_at
                ORDER BY d.created_at DESC
                """
            )
            return cursor.fetchall()
        except Exception as e:
            print(f"❌ Error retrieving documents: {str(e)}")
            return []
        finally:
            cursor.close()
            conn.close()
