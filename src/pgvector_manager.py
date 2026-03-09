"""
PostgreSQL pgvector integration for storing and retrieving document chunks.
"""
import os
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
import psycopg2
from psycopg2.extras import execute_values, RealDictCursor
import json
from src.embeddings_manager import EmbeddingsManager


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

    def save_ragas_test_questions(self, test_cases: List[Any], document_id: int = None) -> int:
        """
        Persist RAGAS test questions to database.

        Args:
            test_cases: List of question strings or dicts containing 'question' and optional 'ground_truth'
            document_id: Optional document ID scope (None means global/shared)

        Returns:
            Number of inserted rows
        """
        if not test_cases:
            return 0

        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            rows = []
            for tc in test_cases:
                if isinstance(tc, str):
                    question = tc.strip()
                    ground_truth = None
                else:
                    question = (tc.get("question") or "").strip()
                    ground_truth = tc.get("ground_truth")

                if not question:
                    continue

                rows.append((
                    document_id,
                    question,
                    ground_truth
                ))

            if not rows:
                return 0

            execute_values(
                cursor,
                """
                INSERT INTO ragas_test_questions (
                    document_id, question, ground_truth
                )
                VALUES %s
                ON CONFLICT (document_id, question)
                DO UPDATE SET ground_truth = COALESCE(EXCLUDED.ground_truth, ragas_test_questions.ground_truth)
                """,
                rows,
                template=None
            )

            inserted = cursor.rowcount
            conn.commit()
            return inserted

        except Exception as e:
            conn.rollback()
            print(f"Error saving RAGAS test questions: {str(e)}")
            raise
        finally:
            cursor.close()
            conn.close()

    def get_ragas_test_questions(self, document_id: int = None, limit: int = 5) -> List[dict]:
        """
        Retrieve persisted RAGAS test questions from database.

        Args:
            document_id: Optional document ID scope. If provided, prioritize that document.
            limit: Maximum number of questions to return

        Returns:
            List of dictionaries with question and ground_truth
        """
        conn = self.get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        try:
            if document_id is not None:
                cursor.execute(
                    """
                    SELECT id, question, ground_truth, created_at
                    FROM ragas_test_questions
                    WHERE document_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (document_id, limit)
                )
                rows = cursor.fetchall()

                if rows and len(rows) >= limit:
                    return [dict(r) for r in rows]

                remaining = limit - len(rows)
                if remaining > 0:
                    cursor.execute(
                        """
                        SELECT id, question, ground_truth, created_at
                        FROM ragas_test_questions
                        WHERE document_id IS NULL
                        ORDER BY created_at DESC
                        LIMIT %s
                        """,
                        (remaining,)
                    )
                    global_rows = cursor.fetchall()
                    rows.extend(global_rows)

                return [dict(r) for r in rows[:limit]]

            cursor.execute(
                """
                SELECT id, question, ground_truth, created_at
                FROM ragas_test_questions
                WHERE document_id IS NULL
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (limit,)
            )
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

        except Exception as e:
            print(f"Error getting RAGAS test questions: {str(e)}")
            return []
        finally:
            cursor.close()
            conn.close()

    def get_ragas_test_question_embeddings(
        self,
        embedding_model: str,
        questions: List[str],
        document_id: int = None
    ) -> Dict[str, List[float]]:
        """
        Get persisted question embeddings for a model and question set.

        Args:
            embedding_model: Embedding model name
            questions: Questions to fetch embeddings for
            document_id: Optional document scope

        Returns:
            Mapping question -> embedding vector
        """
        if not questions:
            return {}

        conn = self.get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        try:
            cursor.execute(
                """
                SELECT q.question, qem.embedding
                FROM ragas_test_question_embeddings qem
                INNER JOIN ragas_test_questions q ON q.id = qem.test_question_id
                INNER JOIN embedding_models em ON em.id = qem.embedding_model_id
                WHERE em.model_name = %s
                  AND q.document_id IS NOT DISTINCT FROM %s
                  AND q.question = ANY(%s)
                """,
                (embedding_model, document_id, questions)
            )

            rows = cursor.fetchall()
            result = {}
            for row in rows:
                embedding = row.get("embedding")
                if isinstance(embedding, str):
                    try:
                        embedding = json.loads(embedding)
                    except Exception:
                        continue
                result[row["question"]] = embedding
            return result

        except Exception as e:
            print(f"Error getting RAGAS test question embeddings: {str(e)}")
            return {}
        finally:
            cursor.close()
            conn.close()

    def save_ragas_test_question_embeddings(
        self,
        embedding_model: str,
        question_embeddings: Dict[str, List[float]],
        document_id: int = None
    ) -> int:
        """
        Persist question embeddings for a specific embedding model.

        Args:
            embedding_model: Embedding model name
            question_embeddings: Mapping question -> embedding vector
            document_id: Optional document scope

        Returns:
            Number of rows inserted/updated
        """
        if not question_embeddings:
            return 0

        conn = self.get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        try:
            embedding_model_id = self.get_or_create_embedding_model(embedding_model)

            questions = list(question_embeddings.keys())
            cursor.execute(
                """
                SELECT id, question
                FROM ragas_test_questions
                WHERE document_id IS NOT DISTINCT FROM %s
                  AND question = ANY(%s)
                """,
                (document_id, questions)
            )
            question_rows = cursor.fetchall()

            if not question_rows:
                return 0

            rows_to_save = []
            for row in question_rows:
                question = row["question"]
                embedding = question_embeddings.get(question)
                if embedding is None:
                    continue
                rows_to_save.append(
                    (row["id"], embedding_model_id, json.dumps(embedding))
                )

            if not rows_to_save:
                return 0

            execute_values(
                cursor,
                """
                INSERT INTO ragas_test_question_embeddings (
                    test_question_id, embedding_model_id, embedding
                )
                VALUES %s
                ON CONFLICT (test_question_id, embedding_model_id)
                DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    created_at = CURRENT_TIMESTAMP
                """,
                rows_to_save,
                template=None
            )

            affected = cursor.rowcount
            conn.commit()
            return affected

        except Exception as e:
            conn.rollback()
            print(f"Error saving RAGAS test question embeddings: {str(e)}")
            raise
        finally:
            cursor.close()
            conn.close()

    def get_saved_question_embedding(
        self,
        question: str,
        embedding_model: str,
        document_id: int = None
    ) -> Optional[List[float]]:
        """
        Get one saved embedding for an exact question and model.

        Priority:
        1) Exact question in provided document scope
        2) Exact question in global scope (document_id IS NULL)
        3) Exact question in any scope (most recent)
        """
        normalized_question = (question or "").strip()
        if not normalized_question:
            return None

        conn = self.get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        try:
            # 1) Scoped lookup
            if document_id is not None:
                cursor.execute(
                    """
                    SELECT qem.embedding
                    FROM ragas_test_question_embeddings qem
                    INNER JOIN ragas_test_questions q ON q.id = qem.test_question_id
                    INNER JOIN embedding_models em ON em.id = qem.embedding_model_id
                    WHERE em.model_name = %s
                      AND q.question = %s
                      AND q.document_id = %s
                    LIMIT 1
                    """,
                    (embedding_model, normalized_question, document_id)
                )
                row = cursor.fetchone()
                if row:
                    embedding = row.get("embedding")
                    if isinstance(embedding, str):
                        embedding = json.loads(embedding)
                    return embedding

            # 2) Global fallback
            cursor.execute(
                """
                SELECT qem.embedding
                FROM ragas_test_question_embeddings qem
                INNER JOIN ragas_test_questions q ON q.id = qem.test_question_id
                INNER JOIN embedding_models em ON em.id = qem.embedding_model_id
                WHERE em.model_name = %s
                  AND q.question = %s
                  AND q.document_id IS NULL
                LIMIT 1
                """,
                (embedding_model, normalized_question)
            )
            row = cursor.fetchone()
            if row:
                embedding = row.get("embedding")
                if isinstance(embedding, str):
                    embedding = json.loads(embedding)
                return embedding

            # 3) Any-scope fallback
            cursor.execute(
                """
                SELECT qem.embedding
                FROM ragas_test_question_embeddings qem
                INNER JOIN ragas_test_questions q ON q.id = qem.test_question_id
                INNER JOIN embedding_models em ON em.id = qem.embedding_model_id
                WHERE em.model_name = %s
                  AND q.question = %s
                ORDER BY q.created_at DESC
                LIMIT 1
                """,
                (embedding_model, normalized_question)
            )
            row = cursor.fetchone()
            if row:
                embedding = row.get("embedding")
                if isinstance(embedding, str):
                    embedding = json.loads(embedding)
                return embedding

            return None
        except Exception as e:
            print(f"Error getting saved question embedding: {str(e)}")
            return None
        finally:
            cursor.close()
            conn.close()
    
    def get_or_create_embedding_model(self, model_name: str) -> int:
        """
        Get or create an embedding model record in the database.
        
        Args:
            model_name: Name of the embedding model
            
        Returns:
            Model ID in the database
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Try to get existing model
            cursor.execute(
                "SELECT id FROM embedding_models WHERE model_name = %s",
                (model_name,)
            )
            result = cursor.fetchone()
            
            if result:
                return result[0]
            
            # Create new model record
            model_info = EmbeddingsManager.get_model_info(model_name)
            cursor.execute(
                """
                INSERT INTO embedding_models (model_name, provider, dimension, description)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (
                    model_name,
                    model_info["provider"],
                    model_info["dimension"],
                    model_info["description"]
                )
            )
            model_id = cursor.fetchone()[0]
            conn.commit()
            print(f"Created embedding model record: {model_name} (ID: {model_id})")
            return model_id
            
        except Exception as e:
            conn.rollback()
            print(f"Error managing embedding model: {str(e)}")
            raise
        finally:
            cursor.close()
            conn.close()
    
    def get_or_create_document(self, filename: str, file_type: str) -> int:
        """
        Get or create a document record by filename.
        Documents are stored only once, independent of embedding models.
        
        Args:
            filename: Name of the document
            file_type: Type of the document (pdf, docx, etc.)
            
        Returns:
            Document ID in the database
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Try to get existing document
            cursor.execute(
                "SELECT id FROM documents WHERE filename = %s",
                (filename,)
            )
            result = cursor.fetchone()
            
            if result:
                return result[0]
            
            # Create new document record
            cursor.execute(
                """
                INSERT INTO documents (filename, file_type)
                VALUES (%s, %s)
                RETURNING id
                """,
                (filename, file_type)
            )
            document_id = cursor.fetchone()[0]
            conn.commit()
            print(f"Created document record: {filename} (ID: {document_id})")
            return document_id
            
        except Exception as e:
            conn.rollback()
            print(f"Error managing document: {str(e)}")
            raise
        finally:
            cursor.close()
            conn.close()
    
    def check_existing_chunks(self, document_id: int, embedding_model_id: int) -> dict:
        """
        Check if chunks already exist for a document-model combination.
        
        Args:
            document_id: ID of the document
            embedding_model_id: ID of the embedding model
            
        Returns:
            Dict with 'exists' (bool) and 'count' (int) keys
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                """
                SELECT COUNT(*) FROM chunks 
                WHERE document_id = %s AND embedding_model_id = %s
                """,
                (document_id, embedding_model_id)
            )
            count = cursor.fetchone()[0]
            return {
                'exists': count > 0,
                'count': count
            }
        finally:
            cursor.close()
            conn.close()
    
    def delete_chunks_for_document_model(self, document_id: int, embedding_model_id: int) -> int:
        """
        Delete all chunks for a specific document-model combination.
        
        Args:
            document_id: ID of the document
            embedding_model_id: ID of the embedding model
            
        Returns:
            Number of chunks deleted
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                """
                DELETE FROM chunks 
                WHERE document_id = %s AND embedding_model_id = %s
                """,
                (document_id, embedding_model_id)
            )
            deleted_count = cursor.rowcount
            
            # Also update document_embeddings
            cursor.execute(
                """
                DELETE FROM document_embeddings
                WHERE document_id = %s AND embedding_model_id = %s
                """,
                (document_id, embedding_model_id)
            )
            
            conn.commit()
            print(f"Deleted {deleted_count} chunks for document {document_id} with model {embedding_model_id}")
            return deleted_count
            
        except Exception as e:
            conn.rollback()
            print(f"Error deleting chunks: {str(e)}")
            raise
        finally:
            cursor.close()
            conn.close()

    def save_chunks(self, chunks: List[Document], filename: str, file_type: str, embedding_model: str = "text-embedding-3-small", replace_existing: bool = False) -> dict:
        """
        Save document chunks with embeddings to PostgreSQL.

        Args:
            chunks: List of document chunks with embeddings
            filename: Name of the original document
            file_type: Type of the document (pdf, docx, etc.)
            embedding_model: Name of the embedding model used
            replace_existing: If True, replace existing chunks for this document-model combination

        Returns:
            Dict with 'document_id', 'chunks_saved', and 'already_existed' keys
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            # Get or create embedding model record
            embedding_model_id = self.get_or_create_embedding_model(embedding_model)
            
            # Get or create document record (without duplication)
            document_id = self.get_or_create_document(filename, file_type)
            
            # Check if chunks already exist
            existing = self.check_existing_chunks(document_id, embedding_model_id)
            
            if existing['exists'] and not replace_existing:
                print(f"Chunks already exist for document '{filename}' with model '{embedding_model}' ({existing['count']} chunks)")
                return {
                    'document_id': document_id,
                    'chunks_saved': 0,
                    'already_existed': True,
                    'existing_count': existing['count'],
                    'embedding_model_id': embedding_model_id
                }
            
            # Delete existing chunks if replace_existing is True
            if existing['exists'] and replace_existing:
                self.delete_chunks_for_document_model(document_id, embedding_model_id)
                print(f"Replacing {existing['count']} existing chunks")

            # Step 2: Prepare chunk data
            chunk_data = []
            for i, chunk in enumerate(chunks):
                # Extract embedding if it exists in the document metadata
                embedding = chunk.metadata.get("embedding", None)
                
                chunk_data.append((
                    document_id,
                    chunk.page_content,
                    embedding,
                    embedding_model_id,
                    i,
                    json.dumps({
                        "filename": chunk.metadata.get("filename", filename),
                        "source": chunk.metadata.get("source", ""),
                        "page": chunk.metadata.get("page", 0),
                    })
                ))

            # Step 3: Insert chunks
            chunks_saved = 0
            if chunk_data:
                execute_values(
                    cursor,
                    """
                    INSERT INTO chunks (document_id, chunk_text, embedding, embedding_model_id, chunk_index, metadata)
                    VALUES %s
                    ON CONFLICT (document_id, embedding_model_id, chunk_index) DO NOTHING
                    """,
                    chunk_data,
                    template=None
                )
                chunks_saved = cursor.rowcount
                
                # Update document_embeddings table
                cursor.execute(
                    """
                    INSERT INTO document_embeddings (document_id, embedding_model_id, chunk_count)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (document_id, embedding_model_id) 
                    DO UPDATE SET chunk_count = %s, updated_at = CURRENT_TIMESTAMP
                    """,
                    (document_id, embedding_model_id, len(chunk_data), len(chunk_data))
                )
                
                conn.commit()
                print(f"Saved {chunks_saved} chunks for document '{filename}' using model '{embedding_model}'")

            return {
                'document_id': document_id,
                'chunks_saved': chunks_saved,
                'already_existed': False,
                'embedding_model_id': embedding_model_id
            }

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

    def search_similar(self, embedding: List[float], k: int = 4, document_id: Optional[int] = None, embedding_model: str = None) -> List[Document]:
        """
        Search for similar chunks using vector similarity.

        Args:
            embedding: Query embedding vector
            k: Number of results to return
            document_id: Optional filter by document ID
            embedding_model: Optional filter by embedding model name

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
                c.id,
                c.document_id,
                c.chunk_text,
                c.metadata,
                c.embedding <=> %s::vector AS distance,
                em.model_name,
                em.provider
            FROM chunks c
            LEFT JOIN embedding_models em ON c.embedding_model_id = em.id
            WHERE 1=1
            """
            
            params = [embedding_str]
            
            if document_id:
                query += " AND c.document_id = %s"
                params.append(document_id)
            
            if embedding_model:
                query += " AND em.model_name = %s"
                params.append(embedding_model)
            
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
                            "embedding_model": row.get("model_name"),
                            "embedding_provider": row.get("provider"),
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

    def get_chunks_by_document(self, document_id: int, embedding_model_id: Optional[int] = None) -> List[Document]:
        """
        Retrieve all chunks for a specific document, optionally filtered by embedding model.

        Args:
            document_id: ID of the document
            embedding_model_id: Optional ID of the embedding model to filter by

        Returns:
            List of document chunks
        """
        conn = self.get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        try:
            if embedding_model_id is not None:
                # Filter by embedding model
                cursor.execute(
                    """
                    SELECT c.id, c.document_id, c.chunk_text, c.metadata, c.chunk_index, 
                           c.embedding_model_id, em.model_name, em.provider
                    FROM chunks c
                    LEFT JOIN embedding_models em ON c.embedding_model_id = em.id
                    WHERE c.document_id = %s AND c.embedding_model_id = %s
                    ORDER BY c.chunk_index
                    """,
                    (document_id, embedding_model_id)
                )
            else:
                # Get all chunks regardless of model
                cursor.execute(
                    """
                    SELECT c.id, c.document_id, c.chunk_text, c.metadata, c.chunk_index,
                           c.embedding_model_id, em.model_name, em.provider
                    FROM chunks c
                    LEFT JOIN embedding_models em ON c.embedding_model_id = em.id
                    WHERE c.document_id = %s
                    ORDER BY c.chunk_index
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
                        "chunk_index": row["chunk_index"],
                        "embedding_model_id": row.get("embedding_model_id"),
                        "model_name": row.get("model_name"),
                        "provider": row.get("provider")
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
        Get all stored documents with their associated embedding models.

        Returns:
            List of document metadata with embedding model information
        """
        conn = self.get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        try:
            cursor.execute(
                """
                SELECT d.id, d.filename, d.file_type, d.created_at,
                       COUNT(DISTINCT c.id) as chunk_count,
                       json_agg(DISTINCT jsonb_build_object(
                           'model_id', em.id,
                           'model_name', em.model_name,
                           'provider', em.provider,
                           'chunk_count', de.chunk_count
                       )) FILTER (WHERE em.id IS NOT NULL) as embedding_models
                FROM documents d
                LEFT JOIN document_embeddings de ON d.id = de.document_id
                LEFT JOIN embedding_models em ON de.embedding_model_id = em.id
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
        Save complete document structure (summary, hierarchy, tables).

        Args:
            document_id: ID of the document
            structure: Dictionary with 'summary', 'hierarchy', 'tables' keys

        Returns:
            True if successful, False otherwise
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            # Remove image metadata for this document (deprecated in current pipeline)
            cursor.execute(
                "DELETE FROM document_pictures WHERE document_id = %s",
                (document_id,)
            )

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
                    0,
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
                    'num_pictures': 0,
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
            structure['pictures'] = []

            return structure

        except Exception as e:
            print(f"Error retrieving document structure: {str(e)}")
            return {}
        finally:
            cursor.close()
            conn.close()

    def purge_all_picture_metadata(self) -> int:
        """
        Delete all image/geometric metadata from database.

        Returns:
            Number of deleted rows
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("DELETE FROM document_pictures")
            deleted_rows = cursor.rowcount or 0
            cursor.execute("UPDATE document_summary SET num_pictures = 0")
            conn.commit()
            return deleted_rows
        except Exception as e:
            conn.rollback()
            print(f"Error deleting picture metadata: {str(e)}")
            return 0
        finally:
            cursor.close()
            conn.close()
    
    def get_all_embedding_models(self) -> List[dict]:
        """
        Get all embedding models stored in the database.
        
        Returns:
            List of embedding model records
        """
        conn = self.get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            cursor.execute(
                """
                SELECT id, model_name, provider, dimension, description, created_at
                FROM embedding_models
                ORDER BY created_at DESC
                """
            )
            models = cursor.fetchall()
            return [dict(m) for m in models]
        except Exception as e:
            print(f"Error getting embedding models: {str(e)}")
            return []
        finally:
            cursor.close()
            conn.close()
    
    def save_ragas_evaluation(self, embedding_model_id: int, document_id: int, 
                             metrics: dict) -> int:
        """
        Save RAGAS evaluation metrics to the database.
        
        Args:
            embedding_model_id: ID of the embedding model
            document_id: ID of the document
            metrics: Dictionary with RAGAS metrics
            
        Returns:
            Evaluation ID
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                """
                INSERT INTO ragas_evaluations (
                    embedding_model_id, document_id,
                    faithfulness, answer_relevancy, context_precision,
                    context_recall, context_relevancy,
                    test_questions_count, average_retrieval_time, metadata
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    embedding_model_id,
                    document_id,
                    metrics.get('faithfulness'),
                    metrics.get('answer_relevancy'),
                    metrics.get('context_precision'),
                    metrics.get('context_recall'),
                    metrics.get('context_relevancy'),
                    metrics.get('test_questions_count'),
                    metrics.get('average_retrieval_time'),
                    json.dumps(metrics.get('metadata', {}))
                )
            )
            evaluation_id = cursor.fetchone()[0]
            conn.commit()
            print(f"Saved RAGAS evaluation (ID: {evaluation_id})")
            return evaluation_id
            
        except Exception as e:
            conn.rollback()
            print(f"Error saving RAGAS evaluation: {str(e)}")
            raise
        finally:
            cursor.close()
            conn.close()
    
    def get_ragas_evaluations(self, embedding_model_id: int = None, 
                              document_id: int = None) -> List[dict]:
        """
        Get RAGAS evaluation metrics.
        
        Args:
            embedding_model_id: Optional filter by embedding model
            document_id: Optional filter by document
            
        Returns:
            List of evaluation records
        """
        conn = self.get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            query = """
            SELECT 
                re.*,
                em.model_name,
                em.provider,
                d.filename
            FROM ragas_evaluations re
            LEFT JOIN embedding_models em ON re.embedding_model_id = em.id
            LEFT JOIN documents d ON re.document_id = d.id
            WHERE 1=1
            """
            params = []
            
            if embedding_model_id:
                query += " AND re.embedding_model_id = %s"
                params.append(embedding_model_id)
            
            if document_id:
                query += " AND re.document_id = %s"
                params.append(document_id)
            
            query += " ORDER BY re.evaluation_date DESC"
            
            cursor.execute(query, params)
            evaluations = cursor.fetchall()
            return [dict(e) for e in evaluations]
            
        except Exception as e:
            print(f"Error getting RAGAS evaluations: {str(e)}")
            return []
        finally:
            cursor.close()
            conn.close()