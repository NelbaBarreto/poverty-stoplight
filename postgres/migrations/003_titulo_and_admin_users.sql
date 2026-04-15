-- ============================================================
-- Migration 003: Document title + admin users
-- Applied: 2026-04-15
-- ============================================================

-- ── 1. Título de documento ────────────────────────────────────────────────
-- Adds a human-readable title to documents.
-- When set, this value is displayed in chat citations instead of the filename.
-- NULL means "no title set" — code falls back to filename.
ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS titulo VARCHAR(500);

-- ── 2. Admin users table ──────────────────────────────────────────────────
-- Stores credentials for the admin panel (Streamlit app on port 8502).
-- Passwords are hashed with pbkdf2_hmac(sha256, iterations=200_000) + random salt.
-- Roles: 'admin' (full access) | 'viewer' (read-only, cannot manage users).
CREATE TABLE IF NOT EXISTS admin_users (
    id            SERIAL PRIMARY KEY,
    username      VARCHAR(80)  NOT NULL UNIQUE,
    email         VARCHAR(200),
    password_hash VARCHAR(64)  NOT NULL,
    salt          VARCHAR(32)  NOT NULL,
    role          VARCHAR(20)  NOT NULL DEFAULT 'viewer'
                      CHECK (role IN ('admin', 'viewer')),
    is_active     BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at    TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_admin_users_username ON admin_users(username);

-- Default admin user (password: admin1234).
-- Hash generated with: pbkdf2_hmac('sha256', b'admin1234', salt.encode(), 200_000)
-- Change this password immediately after first login.
INSERT INTO admin_users (username, email, password_hash, salt, role)
VALUES (
    'admin',
    'admin@localhost',
    'placeholder_change_on_first_run',
    'placeholder_salt',
    'admin'
)
ON CONFLICT (username) DO NOTHING;

-- NOTE: The real hash is seeded automatically by the admin app (admin/app.py)
-- via ensure_users_table() on first startup — this INSERT is only a marker.
-- If running this migration manually on a fresh DB, start the admin app once
-- to generate the correctly hashed default credentials.
