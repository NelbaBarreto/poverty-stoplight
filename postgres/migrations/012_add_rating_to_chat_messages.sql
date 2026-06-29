-- Add rating column to chat_messages for thumbs up/down feedback
ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS rating SMALLINT CHECK (rating IN (1, -1));
