-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================
-- CATALOG TABLES
-- ============================================================

-- Catalog of embedding models
CREATE TABLE embedding_models (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL UNIQUE,
    table_name VARCHAR(100) NOT NULL UNIQUE,
    dimensions INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Chunk size configurations
CREATE TABLE chunk_configs (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) NOT NULL UNIQUE,  -- small / medium / large
    chunk_size INTEGER NOT NULL,
    overlap INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- DOCUMENTS
-- ============================================================

CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL UNIQUE,
    file_type VARCHAR(50) NOT NULL,  -- pdf / csv / md
    file_path TEXT NOT NULL,
    titulo VARCHAR(500),             -- human-readable title shown in chat (nullable)
    link TEXT,                       -- URL de referencia para el documento (nullable)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- CHUNKS (text only, no embedding stored here)
-- ============================================================

CREATE TABLE chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_config_id INTEGER NOT NULL REFERENCES chunk_configs(id),
    format VARCHAR(20) NOT NULL,  -- markdown / plaintext
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_chunks_document_id ON chunks(document_id);
CREATE INDEX idx_chunks_config_format ON chunks(chunk_config_id, format);

-- ============================================================
-- EMBEDDING TABLES (one per model)
-- IVFFlat indexes are created by ingest_all.py after data load
-- ============================================================

CREATE TABLE embeddings_bge_m3 (
    id SERIAL PRIMARY KEY,
    chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE UNIQUE,
    embedding vector(1024) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE embeddings_nomic (
    id SERIAL PRIMARY KEY,
    chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE UNIQUE,
    embedding vector(768) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE embeddings_mxbai (
    id SERIAL PRIMARY KEY,
    chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE UNIQUE,
    embedding vector(1024) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE embeddings_minilm (
    id SERIAL PRIMARY KEY,
    chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE UNIQUE,
    embedding vector(384) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE embeddings_snowflake (
    id SERIAL PRIMARY KEY,
    chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE UNIQUE,
    embedding vector(1024) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- KNOWLEDGE BASE (model Q&A from base_conocimientos.pdf)
-- ============================================================

CREATE TABLE knowledge_base (
    id SERIAL PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    category VARCHAR(100),               -- thematic grouping
    phase VARCHAR(20) DEFAULT 'fase_1',  -- fase_1 | fase_futura
    source_document VARCHAR(255) DEFAULT 'base_conocimientos.pdf',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_kb_category ON knowledge_base(category);
CREATE INDEX idx_kb_phase    ON knowledge_base(phase);

-- ============================================================
-- SEED CATALOG DATA
-- ============================================================

INSERT INTO embedding_models (model_name, table_name, dimensions) VALUES
    ('bge-m3',                 'embeddings_bge_m3',   1024),
    ('nomic-embed-text',       'embeddings_nomic',     768),
    ('mxbai-embed-large',      'embeddings_mxbai',    1024),
    ('all-minilm',             'embeddings_minilm',    384),
    ('snowflake-arctic-embed', 'embeddings_snowflake', 1024);

INSERT INTO chunk_configs (name, chunk_size, overlap) VALUES
    ('small',  512,  64),
    ('medium', 1024, 128),
    ('large',  2048, 256);

-- ============================================================
-- ADMIN USERS (panel de administración — port 8502)
-- ============================================================

CREATE TABLE IF NOT EXISTS admin_users (
    id            SERIAL PRIMARY KEY,
    username      VARCHAR(80)  NOT NULL UNIQUE,
    email         VARCHAR(200),
    password_hash VARCHAR(64)  NOT NULL,
    salt          VARCHAR(32)  NOT NULL,
    role          VARCHAR(20)  NOT NULL DEFAULT 'viewer'
                      CHECK (role IN ('admin', 'viewer')),
    is_active     BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at    TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_admin_users_username ON admin_users(username);
-- Default admin is seeded automatically by admin/app.py on first startup.

-- ============================================================
-- PROMPT HISTORY (historial del system prompt del agente Rosa)
-- ============================================================

CREATE TABLE IF NOT EXISTS prompt_history (
    id          SERIAL       PRIMARY KEY,
    prompt_text TEXT         NOT NULL,
    edited_by   VARCHAR(100) NOT NULL DEFAULT 'admin',
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE prompt_history IS 'Historial de versiones del system prompt del agente Rosa';
