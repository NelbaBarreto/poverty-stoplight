"""
Ingest all files from /data into PostgreSQL pgvector.
- PDFs: extract with Docling, chunk, embed with Ollama bge-m3, store
- CSV (Banco_de_soluciones_processed.csv): one chunk per row, embed and store
"""

import os
import sys
import json
import glob
import requests
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "bge-m3"

# DB config
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "db_rag",
    "user": "sgadmin",
    "password": "sg4dm1n!",
}

# Chunking params (same as src/vectorstore.py)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
)


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def get_embedding(text: str, retries: int = 3) -> list[float]:
    """Get embedding from Ollama bge-m3 with retry logic."""
    import time

    for attempt in range(retries):
        try:
            # Truncate very long text that might cause Ollama issues
            if len(text) > 8000:
                text = text[:8000]
            resp = requests.post(OLLAMA_URL, json={"model": EMBED_MODEL, "prompt": text})
            resp.raise_for_status()
            return resp.json()["embedding"]
        except Exception as e:
            if attempt < retries - 1:
                print(f"    Retry {attempt+1}/{retries} for embedding (error: {e})")
                time.sleep(2)
            else:
                raise


def file_already_ingested(conn, filename: str) -> bool:
    """Check if a file has already been ingested."""
    cur = conn.cursor()
    cur.execute("SELECT id FROM documents WHERE filename = %s", (filename,))
    result = cur.fetchone()
    cur.close()
    return result is not None


def insert_document(conn, filename: str, file_type: str) -> int:
    """Insert document record and return its ID."""
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO documents (filename, file_type) VALUES (%s, %s) RETURNING id",
        (filename, file_type),
    )
    doc_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    return doc_id


def insert_chunks(conn, doc_id: int, chunks: list[dict]):
    """Insert chunks with embeddings into the chunks table."""
    cur = conn.cursor()
    data = [
        (
            doc_id,
            c["text"],
            c["embedding"],
            c["index"],
            json.dumps(c.get("metadata", {})),
        )
        for c in chunks
    ]
    execute_values(
        cur,
        """
        INSERT INTO chunks (document_id, chunk_text, embedding, chunk_index, metadata)
        VALUES %s
        """,
        data,
    )
    conn.commit()
    cur.close()


def process_pdf(filepath: str, conn):
    """Process a single PDF file: extract, chunk, embed, store."""
    filename = os.path.basename(filepath)

    if file_already_ingested(conn, filename):
        print(f"  SKIP (already ingested): {filename}")
        return

    print(f"  Extracting text with Docling: {filename}...")
    # Import Docling here to avoid slow import at startup if not needed
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    result = converter.convert(filepath)
    markdown_content = result.document.export_to_markdown()

    if not markdown_content.strip():
        print(f"  WARNING: No text extracted from {filename}")
        return

    # Chunk
    from langchain_core.documents import Document

    doc = Document(page_content=markdown_content, metadata={"filename": filename})
    text_chunks = text_splitter.split_documents([doc])
    print(f"  {len(text_chunks)} chunks created")

    # Embed and prepare for insertion
    chunks_to_insert = []
    skipped = 0
    for i, chunk in enumerate(text_chunks):
        if i % 20 == 0:
            print(f"    Embedding chunk {i+1}/{len(text_chunks)}...")
        try:
            embedding = get_embedding(chunk.page_content)
        except Exception as e:
            print(f"    SKIP chunk {i} (embedding failed): {e}")
            skipped += 1
            continue
        # Format as pgvector string
        emb_str = "[" + ",".join(str(v) for v in embedding) + "]"
        chunks_to_insert.append(
            {
                "text": chunk.page_content,
                "embedding": emb_str,
                "index": i,
                "metadata": {"filename": filename, "source": filename, "page": 0},
            }
        )

    # Insert
    doc_id = insert_document(conn, filename, "pdf")
    insert_chunks(conn, doc_id, chunks_to_insert)
    print(f"  Stored {len(chunks_to_insert)} chunks for document ID {doc_id} (skipped {skipped})")


def process_csv(filepath: str, conn):
    """Process the solutions CSV: one chunk per row."""
    filename = os.path.basename(filepath)

    if file_already_ingested(conn, filename):
        print(f"  SKIP (already ingested): {filename}")
        return

    print(f"  Reading CSV: {filename}...")
    df = pd.read_csv(filepath)
    print(f"  {len(df)} rows loaded")

    doc_id = insert_document(conn, filename, "csv")

    chunks_to_insert = []
    for i, row in df.iterrows():
        # Combine text fields
        parts = []
        for field in ["title", "description", "content_text", "content_rich"]:
            val = row.get(field)
            if isinstance(val, str) and val.strip():
                parts.append(val.strip())
        text = "\n".join(parts)

        if not text.strip():
            continue

        # Truncate very long texts to avoid embedding issues
        if len(text) > 8000:
            text = text[:8000]

        if i % 100 == 0:
            print(f"    Embedding row {i+1}/{len(df)}...")

        embedding = get_embedding(text)
        emb_str = "[" + ",".join(str(v) for v in embedding) + "]"

        metadata = {
            "filename": filename,
            "source": filename,
            "row_index": int(i),
            "title": str(row.get("title", ""))[:200],
        }
        # Add useful CSV metadata
        for meta_field in ["dimension", "indicators_code_names", "indicators_names", "country", "original_lang"]:
            val = row.get(meta_field)
            if isinstance(val, str) and val.strip():
                metadata[meta_field] = val

        chunks_to_insert.append(
            {
                "text": text,
                "embedding": emb_str,
                "index": int(i),
                "metadata": metadata,
            }
        )

        # Batch insert every 200 rows to avoid memory buildup
        if len(chunks_to_insert) >= 200:
            insert_chunks(conn, doc_id, chunks_to_insert)
            print(f"    Inserted batch of {len(chunks_to_insert)} chunks")
            chunks_to_insert = []

    # Insert remaining
    if chunks_to_insert:
        insert_chunks(conn, doc_id, chunks_to_insert)
        print(f"    Inserted final batch of {len(chunks_to_insert)} chunks")

    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM chunks WHERE document_id = %s", (doc_id,))
    total = cur.fetchone()[0]
    cur.close()
    print(f"  Stored {total} chunks for document ID {doc_id}")


def main():
    conn = get_connection()
    print("Connected to database\n")

    # Collect files
    pdf_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.pdf")))
    csv_files = [os.path.join(DATA_DIR, "Banco_de_soluciones_processed.csv")]

    print(f"Found {len(pdf_files)} PDFs and {len(csv_files)} CSV files\n")

    # Process PDFs
    for pdf_path in pdf_files:
        print(f"\n[PDF] {os.path.basename(pdf_path)}")
        try:
            process_pdf(pdf_path, conn)
        except Exception as e:
            print(f"  ERROR: {e}")

    # Process CSV
    for csv_path in csv_files:
        if os.path.exists(csv_path):
            print(f"\n[CSV] {os.path.basename(csv_path)}")
            try:
                process_csv(csv_path, conn)
            except Exception as e:
                print(f"  ERROR: {e}")
        else:
            print(f"\n[CSV] MISSING: {csv_path}")

    # Summary
    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    cur = conn.cursor()
    cur.execute("""
        SELECT d.filename, d.file_type, COUNT(c.id) as chunk_count
        FROM documents d
        LEFT JOIN chunks c ON d.id = c.document_id
        GROUP BY d.id, d.filename, d.file_type
        ORDER BY d.id
    """)
    for row in cur.fetchall():
        print(f"  {row[0]} ({row[1]}): {row[2]} chunks")
    cur.execute("SELECT COUNT(*) FROM documents")
    doc_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM chunks")
    chunk_count = cur.fetchone()[0]
    print(f"\nTotal: {doc_count} documents, {chunk_count} chunks")
    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
