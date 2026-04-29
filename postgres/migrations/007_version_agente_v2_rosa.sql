-- Migration 007: Crear versión v2 "Versión Rosa" en version_agente
-- Fecha: 2026-04-29
-- Nueva versión del agente con nombre Rosa, prompt actualizado y nuevos documentos

BEGIN;

-- Desactivar la versión anterior
UPDATE version_agente SET activa = FALSE WHERE activa = TRUE;

-- Insertar v2 — Versión Rosa
INSERT INTO version_agente (numero, descripcion, llm_model_id, embed_model_id, chunk_config_id, activa, created_at)
VALUES (
    2,
    'Versión Rosa: qwen3:8b | bge-m3 | medium',
    1,  -- qwen3:8b
    1,  -- bge-m3
    2,  -- medium
    TRUE,
    NOW()
);

SELECT id, numero, descripcion, activa, created_at FROM version_agente ORDER BY id;

COMMIT;
