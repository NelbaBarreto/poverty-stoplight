-- ============================================================
-- Migration 004: Webchat session tracking
-- Applied: 2026-04-15
-- ============================================================

-- Enable pgcrypto for gen_random_uuid() if not already enabled
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ── Chat sessions ─────────────────────────────────────────────────────────
-- One row per browser session. session_id is stored in localStorage.
CREATE TABLE IF NOT EXISTS chat_sessions (
    id          UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    origin      VARCHAR(255),                    -- referrer / WordPress page URL
    created_at  TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_active TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_chat_sessions_created ON chat_sessions(created_at DESC);

-- ── Chat messages ─────────────────────────────────────────────────────────
-- Every user message and assistant response, in order.
CREATE TABLE IF NOT EXISTS chat_messages (
    id               SERIAL       PRIMARY KEY,
    session_id       UUID         NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role             VARCHAR(20)  NOT NULL CHECK (role IN ('user', 'assistant')),
    content          TEXT         NOT NULL,
    llm_model        VARCHAR(100),
    embed_model      VARCHAR(100),
    chunk_config     VARCHAR(50),
    response_time_ms INTEGER,                    -- NULL for user messages
    created_at       TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_session  ON chat_messages(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_chat_messages_created  ON chat_messages(created_at DESC);
