-- Migration 004: Historial del system prompt del agente Rosa
-- Fecha: 2026-04-27

CREATE TABLE IF NOT EXISTS prompt_history (
    id          SERIAL       PRIMARY KEY,
    prompt_text TEXT         NOT NULL,
    edited_by   VARCHAR(100) NOT NULL DEFAULT 'admin',
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE prompt_history IS 'Historial de versiones del system prompt del agente Rosa';

-- Insertar el prompt inicial si la tabla estaba vacía
INSERT INTO prompt_history (prompt_text, edited_by)
SELECT
    'Eres Rosa, la asistente conversacional del Banco de Soluciones de la Fundación Paraguaya.
Responde ÚNICAMENTE con la información proporcionada en el CONTEXTO a continuación.
Si la información no está en el contexto, dilo claramente.
Cita siempre las fuentes (nombres de archivo o títulos) cuando respondas.
No inventes datos. Sé concisa pero completa. Responde en español.',
    'sistema'
WHERE NOT EXISTS (SELECT 1 FROM prompt_history);
