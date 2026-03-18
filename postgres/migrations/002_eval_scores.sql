-- ============================================================
-- Migration 002: RAGAS evaluation scores table
-- ============================================================

CREATE TABLE IF NOT EXISTS eval_scores (
    id                SERIAL PRIMARY KEY,
    eval_run_id       INTEGER NOT NULL REFERENCES eval_runs(id) UNIQUE,
    faithfulness      FLOAT,
    answer_relevancy  FLOAT,
    context_precision FLOAT,
    context_recall    FLOAT,
    judge_llm         VARCHAR(100) NOT NULL DEFAULT 'gpt-oss:20b',
    status            VARCHAR(20)  NOT NULL DEFAULT 'pending',
    error_message     TEXT,
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_eval_scores_run    ON eval_scores(eval_run_id);
CREATE INDEX IF NOT EXISTS idx_eval_scores_status ON eval_scores(status);
