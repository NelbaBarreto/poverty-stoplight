-- ============================================================
-- Migration 001: RAG evaluation tables
-- ============================================================

-- LLM models catalog
CREATE TABLE IF NOT EXISTS llm_models (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO llm_models (model_name) VALUES
    ('qwen3:8b'),
    ('llama3.2:latest'),
    ('llama3.2:3b'),
    ('gpt-oss:20b'),
    ('llama2:latest'),
    ('deepseek-r1:14b')
ON CONFLICT (model_name) DO NOTHING;

-- Evaluation runs (one row per question × llm × embedding × chunk_config)
CREATE TABLE IF NOT EXISTS eval_runs (
    id                 SERIAL PRIMARY KEY,
    llm_model_id       INTEGER NOT NULL REFERENCES llm_models(id),
    embedding_model_id INTEGER NOT NULL REFERENCES embedding_models(id),
    chunk_config_id    INTEGER NOT NULL REFERENCES chunk_configs(id),
    knowledge_base_id  INTEGER NOT NULL REFERENCES knowledge_base(id),

    -- Retrieval output stored as JSON array of dicts:
    -- [{chunk_id, chunk_text, filename, format, distance}, ...]
    retrieved_contexts JSONB,
    k_retrieved        INTEGER DEFAULT 8,

    generated_answer   TEXT,

    -- Status lifecycle: pending → success | error
    status             VARCHAR(20) DEFAULT 'pending',
    error_message      TEXT,

    -- Timing (milliseconds)
    retrieval_time_ms  INTEGER,
    generation_time_ms INTEGER,
    total_time_ms      INTEGER,

    created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Each combination is unique; ON CONFLICT DO NOTHING used for resumability
    UNIQUE (llm_model_id, embedding_model_id, chunk_config_id, knowledge_base_id)
);

CREATE INDEX IF NOT EXISTS idx_eval_runs_llm    ON eval_runs(llm_model_id);
CREATE INDEX IF NOT EXISTS idx_eval_runs_embed  ON eval_runs(embedding_model_id);
CREATE INDEX IF NOT EXISTS idx_eval_runs_chunk  ON eval_runs(chunk_config_id);
CREATE INDEX IF NOT EXISTS idx_eval_runs_kb     ON eval_runs(knowledge_base_id);
CREATE INDEX IF NOT EXISTS idx_eval_runs_status ON eval_runs(status);

-- ============================================================
-- RAGAS column mapping (documentation comment)
-- ============================================================
-- question     → knowledge_base.question        (JOIN via knowledge_base_id)
-- answer       → eval_runs.generated_answer
-- contexts     → [c["chunk_text"] for c in retrieved_contexts]
-- ground_truth → knowledge_base.answer          (JOIN via knowledge_base_id)
-- ============================================================
