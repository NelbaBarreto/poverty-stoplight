-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create documents table to store document metadata
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    file_type VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create chunks table to store document chunks with embeddings
CREATE TABLE chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    embedding vector(1024),
    chunk_index INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Create indexes for faster retrieval
CREATE INDEX idx_chunks_document_id ON chunks(document_id);
CREATE INDEX idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_chunks_metadata ON chunks USING gin (metadata);
-- Create document_summary table to store document statistics
CREATE TABLE document_summary (
    id SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL UNIQUE REFERENCES documents(id) ON DELETE CASCADE,
    num_pages INTEGER DEFAULT 0,
    num_texts INTEGER DEFAULT 0,
    num_tables INTEGER DEFAULT 0,
    num_pictures INTEGER DEFAULT 0,
    text_types JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create document_hierarchy table to store document structure (headings, sections, etc.)
CREATE TABLE document_hierarchy (
    id SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    type VARCHAR(100),
    text TEXT,
    page_no INTEGER,
    level INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create document_tables table to store table metadata and content
CREATE TABLE document_tables (
    id SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    table_number INTEGER,
    page_no INTEGER,
    caption TEXT,
    table_data JSONB,
    shape VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create document_pictures table to store image metadata
CREATE TABLE document_pictures (
    id SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    picture_number INTEGER,
    page_no INTEGER,
    caption TEXT,
    bounding_box JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for the new tables
CREATE INDEX idx_document_summary_doc_id ON document_summary(document_id);
CREATE INDEX idx_document_hierarchy_doc_id ON document_hierarchy(document_id);
CREATE INDEX idx_document_tables_doc_id ON document_tables(document_id);
CREATE INDEX idx_document_pictures_doc_id ON document_pictures(document_id);