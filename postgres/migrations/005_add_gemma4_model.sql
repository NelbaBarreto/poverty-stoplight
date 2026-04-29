-- Migration 005: Agregar gemma4:latest a llm_models
-- Fecha: 2026-04-29

INSERT INTO llm_models (model_name)
VALUES ('gemma4:latest')
ON CONFLICT DO NOTHING;
