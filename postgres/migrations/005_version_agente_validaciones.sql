-- ============================================================
-- Migration 005: Agent versions and human validation
-- Applied: 2026-04-15
-- ============================================================

-- ── version_agente ────────────────────────────────────────────────────────────
-- Tracks named versions of the best agent combination (llm + embed + chunk).
CREATE TABLE IF NOT EXISTS version_agente (
    id              SERIAL      PRIMARY KEY,
    numero          INTEGER     NOT NULL UNIQUE,
    descripcion     VARCHAR(200),
    llm_model_id    INTEGER     NOT NULL REFERENCES llm_models(id),
    embed_model_id  INTEGER     NOT NULL REFERENCES embedding_models(id),
    chunk_config_id INTEGER     NOT NULL REFERENCES chunk_configs(id),
    activa          BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Seed version 1: qwen3:8b | bge-m3 | medium (llm=1, embed=1, chunk=2)
INSERT INTO version_agente (numero, descripcion, llm_model_id, embed_model_id, chunk_config_id, activa)
VALUES (1, 'Versión inicial: qwen3:8b | bge-m3 | medium', 1, 1, 2, TRUE)
ON CONFLICT (numero) DO NOTHING;

-- ── validaciones ─────────────────────────────────────────────────────────────
-- Manual human evaluations of agent responses.
-- One row per (eval_run, user) → a user can re-evaluate (UPDATE), not duplicate.
CREATE TABLE IF NOT EXISTS validaciones (
    id                SERIAL      PRIMARY KEY,
    version_agente_id INTEGER     NOT NULL REFERENCES version_agente(id),
    eval_run_id       INTEGER     NOT NULL REFERENCES eval_runs(id),
    user_id           INTEGER     NOT NULL REFERENCES admin_users(id),
    calificacion      SMALLINT    NOT NULL CHECK (calificacion BETWEEN 1 AND 10),
    observacion       TEXT,
    alucinacion       BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at        TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at        TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (eval_run_id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_validaciones_version  ON validaciones(version_agente_id);
CREATE INDEX IF NOT EXISTS idx_validaciones_run      ON validaciones(eval_run_id);
CREATE INDEX IF NOT EXISTS idx_validaciones_user     ON validaciones(user_id);
