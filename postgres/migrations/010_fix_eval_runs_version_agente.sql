-- Migration 010: Limpiar version_agente_id en eval_runs
-- Fecha: 2026-04-29
-- Problema: Migration 009 asignó version_agente_id = 1 a los 9.720 runs
-- (90 combinaciones × 108 preguntas), pero solo los 108 runs con la config
-- canónica de v1 (qwen3:8b | bge-m3 | medium) deben tenerlo.
--
-- Solución: poner version_agente_id = NULL en todos los runs cuya config
-- de modelo NO coincide con la de la versión asignada.
-- Esto deja que version_agente.llm_model_id / embed_model_id / chunk_config_id
-- sea la única fuente de verdad de la "mejor combinación" de cada versión.

BEGIN;

-- 1. Quitar NOT NULL y DEFAULT (migration 009 los puso; ahora necesitamos
--    permitir NULL para runs que no corresponden a ninguna versión canónica)
ALTER TABLE eval_runs ALTER COLUMN version_agente_id DROP NOT NULL;
ALTER TABLE eval_runs ALTER COLUMN version_agente_id DROP DEFAULT;

-- 2. NULL-ificar runs cuya config no corresponde a su versión asignada
UPDATE eval_runs er
SET version_agente_id = NULL
WHERE version_agente_id IS NOT NULL
  AND NOT EXISTS (
      SELECT 1 FROM version_agente va
      WHERE va.id              = er.version_agente_id
        AND va.llm_model_id    = er.llm_model_id
        AND va.embed_model_id  = er.embedding_model_id
        AND va.chunk_config_id = er.chunk_config_id
  );

-- 2. Verificar resultado
SELECT
    CASE WHEN version_agente_id IS NULL THEN 'sin_version' ELSE version_agente_id::text END AS version,
    COUNT(*) AS runs
FROM eval_runs
GROUP BY version_agente_id
ORDER BY version_agente_id NULLS LAST;

-- 3. Confirmar que solo quedan las 108 filas canónicas con version_agente_id = 1
SELECT
    va.numero,
    va.descripcion,
    lm.model_name AS llm,
    em.model_name AS embed,
    cc.name       AS chunk,
    COUNT(er.id)  AS runs_con_version
FROM version_agente va
JOIN llm_models     lm ON lm.id = va.llm_model_id
JOIN embedding_models em ON em.id = va.embed_model_id
JOIN chunk_configs  cc ON cc.id = va.chunk_config_id
LEFT JOIN eval_runs er
       ON er.version_agente_id  = va.id
      AND er.llm_model_id       = va.llm_model_id
      AND er.embedding_model_id = va.embed_model_id
      AND er.chunk_config_id    = va.chunk_config_id
GROUP BY va.numero, va.descripcion, lm.model_name, em.model_name, cc.name
ORDER BY va.numero;

SELECT 'Migration 010 OK — version_agente_id limpio' AS resultado;

COMMIT;
