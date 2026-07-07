-- Migration 013: Versión v4 — Qwen/Qwen3.5-27B-FP8 via vLLM | bge-m3 | medium
-- Fecha: 2026-07-07

BEGIN;

-- Registrar nuevo modelo LLM
INSERT INTO llm_models (model_name)
VALUES ('Qwen/Qwen3.5-27B-FP8')
ON CONFLICT DO NOTHING;

-- Desactivar versión anterior
UPDATE version_agente SET activa = FALSE WHERE activa = TRUE;

-- Insertar v4
INSERT INTO version_agente (numero, descripcion, llm_model_id, embed_model_id, chunk_config_id, activa, created_at)
VALUES (
    4,
    'v4: Qwen/Qwen3.5-27B-FP8 (vLLM) | bge-m3 | medium',
    (SELECT id FROM llm_models WHERE model_name = 'Qwen/Qwen3.5-27B-FP8'),
    (SELECT id FROM embedding_models WHERE model_name = 'bge-m3'),
    (SELECT id FROM chunk_configs WHERE name = 'medium'),
    TRUE,
    NOW()
);

SELECT id, numero, descripcion, activa FROM version_agente ORDER BY id;

COMMIT;
