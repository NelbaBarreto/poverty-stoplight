-- ============================================================
-- Migration 011: Add link column to documents
-- Applied: 2026-06-03
-- ============================================================

-- Adds a reference URL column to documents table for citations in chat responses.
-- Used by webchat API to provide source links in responses.

ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS link TEXT;

CREATE INDEX IF NOT EXISTS idx_documents_link ON documents(link);
