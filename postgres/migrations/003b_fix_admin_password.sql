-- ============================================================
-- Migration 003b: Fix default admin password
-- Applied: 2026-04-15
--
-- Updates the admin user created by 003_titulo_and_admin_users.sql
-- with the correct password hash for 'sg4dm1n!'.
-- Algorithm: pbkdf2_hmac(sha256, iterations=200_000) + random salt
-- ============================================================

UPDATE admin_users
SET password_hash = 'c6761286e7fe5812e6fb73f22d84d7487e77dda565159cb382c4a576702fbfbc',
    salt          = 'c4848c28f98f009699e36765ff4199f6'
WHERE username = 'admin'
  AND password_hash = 'placeholder_change_on_first_run';
