-- ============================================================
-- Migration 003b: Fix default admin password
-- Applied: 2026-04-15
--
-- Updates the admin user created by 003_titulo_and_admin_users.sql
-- with the correct password hash for 'sg4dm1n!'.
-- Algorithm: pbkdf2_hmac(sha256, iterations=200_000) + random salt
-- ============================================================

UPDATE admin_users
SET password_hash = 'efc1b94352c2fa1b2b3d6b73ad8230d08b70dfc1d262b4f784066ef7df8590b6',
    salt          = '124c3e08123b3b2ba233175c1128a3d7'
WHERE username = 'admin'
  AND password_hash = 'placeholder_change_on_first_run';
