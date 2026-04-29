-- Migration 008: Asignar version_agente_id = 1 a todas las validaciones existentes
-- Fecha: 2026-04-29
-- Contexto: todas las evaluaciones registradas hasta ahora corresponden a
--           la v1 "Versión inicial: qwen3:8b | bge-m3 | medium"

BEGIN;

UPDATE validaciones
SET version_agente_id = 1
WHERE version_agente_id IS NULL
   OR version_agente_id != 1;

DO $$
DECLARE
    n INTEGER;
BEGIN
    SELECT COUNT(*) INTO n FROM validaciones WHERE version_agente_id = 1;
    RAISE NOTICE 'validaciones actualizadas a version_agente_id=1: %', n;
END;
$$;

COMMIT;
