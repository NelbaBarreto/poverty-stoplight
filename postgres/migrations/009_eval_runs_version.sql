-- Migration 009: Agregar version_agente_id a eval_runs
-- Fecha: 2026-04-29
-- Permite tener ejecuciones separadas por versión del agente con la misma config técnica

BEGIN;

-- 1. Agregar columna (nullable para compatibilidad con filas existentes)
ALTER TABLE eval_runs
    ADD COLUMN IF NOT EXISTS version_agente_id INTEGER REFERENCES version_agente(id);

-- 2. Asignar versión 1 a todos los runs existentes
UPDATE eval_runs SET version_agente_id = 1 WHERE version_agente_id IS NULL;

-- 3. Hacer la columna NOT NULL ahora que todos tienen valor
ALTER TABLE eval_runs ALTER COLUMN version_agente_id SET NOT NULL;
ALTER TABLE eval_runs ALTER COLUMN version_agente_id SET DEFAULT 1;

-- 4. Reemplazar el constraint UNIQUE para incluir version_agente_id
ALTER TABLE eval_runs
    DROP CONSTRAINT eval_runs_llm_model_id_embedding_model_id_chunk_config_id_k_key;

ALTER TABLE eval_runs
    ADD CONSTRAINT eval_runs_unique_per_version
    UNIQUE (llm_model_id, embedding_model_id, chunk_config_id, knowledge_base_id, version_agente_id);

-- 5. Índice para búsquedas por versión
CREATE INDEX IF NOT EXISTS idx_eval_runs_version ON eval_runs(version_agente_id);

SELECT 'Migration 009 OK — eval_runs ahora tiene version_agente_id' AS resultado;
SELECT version_agente_id, COUNT(*) FROM eval_runs GROUP BY version_agente_id ORDER BY 1;

COMMIT;
