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
            print(f"Error connecting to database: {str(e)}")
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
            print(f"Created document record with ID: {document_id}")

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
                print(f"Saved {len(chunk_data)} chunks for document '{filename}'")

            return document_id

        except Exception as e:
            conn.rollback()
            print(f"Error saving chunks: {str(e)}")
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
                try:
                    raw_meta = row.get("metadata")

                    metadata = raw_meta.copy() if isinstance(raw_meta, dict) else {}

                    doc = Document(
                        page_content=row.get("chunk_text", ""),
                        metadata={
                            **metadata,
                            "chunk_id": row.get("id"),
                            "document_id": row.get("document_id"),
                            "distance": float(row.get("distance", 0)),
                        },
                    )

                    documents.append(doc)

                except Exception as e:
                    print("Error building Document:", e)
                    print("Row:", row)

            return documents
        
        except Exception as e:
            print(f"Error searching similar chunks: {str(e)}")
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
                raw_meta = row["metadata"]
                if not raw_meta:
                    metadata = {}
                elif isinstance(raw_meta, str):
                    try:
                        metadata = json.loads(raw_meta)
                    except Exception:
                        metadata = {}
                elif isinstance(raw_meta, dict):
                    metadata = raw_meta
                else:
                    metadata = {}
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
            print(f"Error retrieving chunks: {str(e)}")
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
            print(f"Deleted document {document_id} and its chunks")
            return True
        except Exception as e:
            conn.rollback()
            print(f"Error deleting document: {str(e)}")
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
            print(f"Error retrieving documents: {str(e)}")
            return []
        finally:
            cursor.close()
            conn.close()
    def save_document_structure(self, document_id: int, structure: dict) -> bool:
        """
        Save complete document structure (summary, hierarchy, tables, pictures).

        Args:
            document_id: ID of the document
            structure: Dictionary with 'summary', 'hierarchy', 'tables', 'pictures' keys

        Returns:
            True if successful, False otherwise
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            # Save summary
            summary = structure.get('summary', {})
            cursor.execute(
                """
                INSERT INTO document_summary (document_id, num_pages, num_texts, num_tables, num_pictures, text_types)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (document_id) DO UPDATE SET
                    num_pages = EXCLUDED.num_pages,
                    num_texts = EXCLUDED.num_texts,
                    num_tables = EXCLUDED.num_tables,
                    num_pictures = EXCLUDED.num_pictures,
                    text_types = EXCLUDED.text_types
                """,
                (
                    document_id,
                    summary.get('num_pages', 0),
                    summary.get('num_texts', 0),
                    summary.get('num_tables', 0),
                    summary.get('num_pictures', 0),
                    json.dumps(summary.get('text_types', {}))
                )
            )
            print(f"Saved document summary for document {document_id}")

            # Save hierarchy
            hierarchy = structure.get('hierarchy', [])
            for item in hierarchy:
                cursor.execute(
                    """
                    INSERT INTO document_hierarchy (document_id, type, text, page_no, level)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        document_id,
                        item.get('type', ''),
                        item.get('text', ''),
                        item.get('page', None),
                        item.get('level', 0)
                    )
                )
            print(f"Saved {len(hierarchy)} hierarchy items for document {document_id}")

            # Save tables metadata
            tables = structure.get('tables', [])
            for table in tables:
                # Convert dataframe to JSON for storage
                table_json = None
                if 'dataframe' in table:
                    try:
                        table_json = table['dataframe'].to_json(orient='split')
                    except:
                        table_json = None

                cursor.execute(
                    """
                    INSERT INTO document_tables (document_id, table_number, page_no, caption, table_data, shape)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        document_id,
                        table.get('table_number', 0),
                        table.get('page', None),
                        table.get('caption', ''),
                        table_json,
                        str(table.get('shape', ''))
                    )
                )
            print(f"Saved {len(tables)} tables for document {document_id}")

            # Save pictures metadata
            pictures = structure.get('pictures', [])
            for pic in pictures:
                cursor.execute(
                    """
                    INSERT INTO document_pictures (document_id, picture_number, page_no, caption, bounding_box)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        document_id,
                        pic.get('picture_number', 0),
                        pic.get('page', None),
                        pic.get('caption', ''),
                        json.dumps(pic.get('bounding_box', {})) if pic.get('bounding_box') else None
                    )
                )
            print(f"Saved {len(pictures)} pictures for document {document_id}")

            conn.commit()
            return True

        except Exception as e:
            conn.rollback()
            print(f"Error saving document structure: {str(e)}")
            return False
        finally:
            cursor.close()
            conn.close()

    def get_document_structure(self, document_id: int) -> dict:
        """
        Retrieve complete document structure from database.

        Args:
            document_id: ID of the document

        Returns:
            Dictionary with structure data or empty dict if not found
        """
        conn = self.get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        try:
            structure = {}

            # Get summary
            cursor.execute(
                "SELECT * FROM document_summary WHERE document_id = %s",
                (document_id,)
            )
            summary_row = cursor.fetchone()
            if summary_row:
                structure['summary'] = {
                    'num_pages': summary_row['num_pages'],
                    'num_texts': summary_row['num_texts'],
                    'num_tables': summary_row['num_tables'],
                    'num_pictures': summary_row['num_pictures'],
                    'text_types': summary_row['text_types'] or {}
                }

            # Get hierarchy
            cursor.execute(
                "SELECT type, text, page_no, level FROM document_hierarchy WHERE document_id = %s ORDER BY id",
                (document_id,)
            )
            hierarchy = []
            for row in cursor.fetchall():
                hierarchy.append({
                    'type': row['type'],
                    'text': row['text'],
                    'page': row['page_no'],
                    'level': row['level']
                })
            structure['hierarchy'] = hierarchy

            # Get tables
            cursor.execute(
                "SELECT table_number, page_no, caption, table_data, shape FROM document_tables WHERE document_id = %s ORDER BY table_number",
                (document_id,)
            )
            tables = []
            for row in cursor.fetchall():
                table_dict = {
                    'table_number': row['table_number'],
                    'page': row['page_no'],
                    'caption': row['caption'],
                    'shape': row['shape']
                }
                if row['table_data']:
                    try:
                        import pandas as pd
                        table_dict['dataframe'] = pd.read_json(row['table_data'], orient='split')
                    except:
                        pass
                tables.append(table_dict)
            structure['tables'] = tables

            # Get pictures
            cursor.execute(
                "SELECT picture_number, page_no, caption, bounding_box FROM document_pictures WHERE document_id = %s ORDER BY picture_number",
                (document_id,)
            )
            pictures = []
            for row in cursor.fetchall():
                pictures.append({
                    'picture_number': row['picture_number'],
                    'page': row['page_no'],
                    'caption': row['caption'],
                    'bounding_box': row['bounding_box'] or {}
                })
            structure['pictures'] = pictures

            return structure

        except Exception as e:
            print(f"Error retrieving document structure: {str(e)}")
            return {}
        finally:
            cursor.close()
            conn.close()