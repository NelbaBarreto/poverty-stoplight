-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create embedding_models table to track different embedding models used
CREATE TABLE embedding_models (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(255) NOT NULL UNIQUE,
    provider VARCHAR(50) NOT NULL,
    dimension INTEGER NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create documents table to store document metadata
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    file_type VARCHAR(50),
    embedding_model_id INTEGER REFERENCES embedding_models(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create chunks table to store document chunks with embeddings
-- Using vector type without fixed dimension to support different models
CREATE TABLE chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    embedding vector,  -- Dynamic dimension based on model
    embedding_model_id INTEGER REFERENCES embedding_models(id),
    chunk_index INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Create indexes for faster retrieval
CREATE INDEX idx_chunks_document_id ON chunks(document_id);
CREATE INDEX idx_chunks_embedding_model ON chunks(embedding_model_id);
-- Note: ivfflat index will be created dynamically per model dimension
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

-- Create ragas_evaluations table to store RAGAS evaluation metrics
CREATE TABLE ragas_evaluations (
    id SERIAL PRIMARY KEY,
    embedding_model_id INTEGER REFERENCES embedding_models(id),
    document_id INTEGER REFERENCES documents(id),
    evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- RAGAS metrics
    faithfulness FLOAT,
    answer_relevancy FLOAT,
    context_precision FLOAT,
    context_recall FLOAT,
    context_relevancy FLOAT,
    
    -- Additional metadata
    test_questions_count INTEGER,
    average_retrieval_time FLOAT,
    metadata JSONB,
    
    -- Composite indexes for querying
    UNIQUE(embedding_model_id, document_id, evaluation_date)
);

CREATE INDEX idx_ragas_model ON ragas_evaluations(embedding_model_id);
CREATE INDEX idx_ragas_document ON ragas_evaluations(document_id);
CREATE INDEX idx_ragas_date ON ragas_evaluations(evaluation_date);