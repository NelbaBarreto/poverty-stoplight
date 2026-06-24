#!/usr/bin/env python3
"""
Webchat API — Poverty Stoplight RAG
FastAPI backend: chat endpoint, session tracking, history.

Endpoints:
  POST /api/session          → create session, returns session_id
  POST /api/chat             → send message, get full response
  GET  /api/session/{id}     → get session message history
  GET  /api/health           → health check

Configuration (via .env):
  LLM_MODEL      default: qwen3:8b
  EMBED_MODEL    default: bge-m3
  CHUNK_CONFIG   default: medium
  VLLM_BASE_URL  default: http://localhost:8800
  API_PORT       default: 8000
  ALLOWED_ORIGINS  default: * (CORS)
"""

import hashlib
import os
import sys
import time
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import psycopg2
import psycopg2.extras
import requests
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("webchat.api")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LLM_MODEL       = os.getenv("LLM_MODEL",       "qwen3:8b")
EMBED_MODEL     = os.getenv("EMBED_MODEL",     "bge-m3")
CHUNK_CONFIG    = os.getenv("CHUNK_CONFIG",    "medium")
VLLM_URL        = os.getenv("VLLM_BASE_URL",  "http://localhost:8800")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
WEBCHAT_SECRET  = os.getenv("WEBCHAT_SECRET",  "W3bCh4tFup4")

# Single shared LLM instance — avoids re-initialization on every request
_llm: "ChatOpenAI | None" = None

def get_llm() -> "ChatOpenAI":
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0,
            base_url=f"{VLLM_URL}/v1",
            api_key="EMPTY",
            max_tokens=4096,
        )
    return _llm

# ---------------------------------------------------------------------------
# Token rotativo diario — sha512(SECRET + DD/MM/YYYY)
# ---------------------------------------------------------------------------

def _daily_token(dt: datetime) -> str:
    date_str = dt.strftime("%d/%m/%Y")
    raw = f"{WEBCHAT_SECRET}{date_str}"
    token = hashlib.sha512(raw.encode()).hexdigest()
    return token


def require_token(x_auth_token: str = Header(..., alias="X-Auth-Token")):
    expected = _daily_token(datetime.now())
    log.debug(f"[Token] Comparing: expected={expected[:16]}... vs received={x_auth_token[:16] if x_auth_token else 'NONE'}...")
    if x_auth_token != expected:
        log.warning(f"[Token] INVALID: received '{x_auth_token[:32]}...' vs expected '{expected[:32]}...'")
        raise HTTPException(status_code=403, detail="Token inválido o expirado")
    log.debug("[Token] Valid token accepted")

EMBED_TABLE = {
    "bge-m3":                 "embeddings_bge_m3",
    "nomic-embed-text":       "embeddings_nomic",
    "mxbai-embed-large":      "embeddings_mxbai",
    "all-minilm":             "embeddings_minilm",
    "snowflake-arctic-embed": "embeddings_snowflake",
}

_PROMPT_FILE = PROJECT_ROOT / "prompt.txt"

def load_system_prompt() -> str:
    if _PROMPT_FILE.exists():
        return _PROMPT_FILE.read_text(encoding="utf-8").strip()
    return "Eres Rosa, la asistente conversacional del Banco de Soluciones de la Fundación Paraguaya."

DB_CONFIG = dict(
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", "5432")),
    dbname=os.getenv("DB_NAME", "db_rag"),
    user=os.getenv("DB_USER", "sgadmin"),
    password=os.getenv("DB_PASSWORD", "sg4dm1n!"),
)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Webchat API — Poverty Stoplight",
    description="RAG chat backend for the WordPress widget",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def get_conn():
    return psycopg2.connect(**DB_CONFIG, cursor_factory=psycopg2.extras.RealDictCursor)


def create_session(origin: Optional[str] = None) -> str:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO chat_sessions (origin) VALUES (%s) RETURNING id",
                (origin,),
            )
            session_id = str(cur.fetchone()["id"])
        conn.commit()
        return session_id
    finally:
        conn.close()


def touch_session(session_id: str):
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE chat_sessions SET last_active = NOW() WHERE id = %s",
                (session_id,),
            )
        conn.commit()
    finally:
        conn.close()


def session_exists(session_id: str) -> bool:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM chat_sessions WHERE id = %s", (session_id,))
            return cur.fetchone() is not None
    finally:
        conn.close()


def save_message(session_id: str, role: str, content: str,
                 llm_model=None, embed_model=None, chunk_config=None,
                 response_time_ms=None):
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO chat_messages
                   (session_id, role, content, llm_model, embed_model, chunk_config, response_time_ms)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (session_id, role, content, llm_model, embed_model, chunk_config, response_time_ms),
            )
        conn.commit()
    finally:
        conn.close()


def get_history(session_id: str, limit: int = 20) -> list:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT role, content, created_at, response_time_ms
                   FROM chat_messages
                   WHERE session_id = %s
                   ORDER BY created_at
                   LIMIT %s""",
                (session_id, limit),
            )
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# RAG helpers
# ---------------------------------------------------------------------------

def get_embedding(query: str) -> list:
    t0 = time.monotonic()
    resp = requests.post(
        f"{VLLM_URL}/v1/embeddings",
        json={"model": EMBED_MODEL, "input": query},
        headers={"Authorization": "Bearer EMPTY"},
        timeout=60,
    )
    resp.raise_for_status()
    log.info(f"[timing] embed={int((time.monotonic()-t0)*1000)}ms")
    return resp.json()["data"][0]["embedding"]


def search_context(query: str, k: int = 8) -> tuple:
    """Return (context_str, sources_list) where sources_list is [{titulo, link}]."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.pgvector_manager import PGVectorManager

    embedding = get_embedding(query)
    embed_table = EMBED_TABLE.get(EMBED_MODEL, "embeddings_bge_m3")

    t1 = time.monotonic()
    mgr = PGVectorManager()
    results = mgr.search_similar_new_schema(
        embedding=embedding,
        embed_table=embed_table,
        chunk_config_name=CHUNK_CONFIG,
        k=k,
    )
    log.info(f"[timing] pgvector={int((time.monotonic()-t1)*1000)}ms")

    parts = []
    seen: set = set()
    sources: list = []
    for doc in results:
        meta = doc.metadata or {}
        titulo = meta.get("titulo") or meta.get("filename", "Desconocido")
        link   = meta.get("link") or None
        content = doc.page_content.strip()
        if content:
            parts.append(f"[Fuente: {titulo}]\n{content}")
        if titulo not in seen:
            seen.add(titulo)
            sources.append({"titulo": titulo, "link": link})
    return "\n\n---\n\n".join(parts), sources


def build_messages(history: list, prompt: str, context: str):
    """Build LangChain message list from DB history + new RAG prompt."""
    messages = []
    # Include last 6 turns for context
    for msg in history[-6:]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    full_prompt = (
        f"{load_system_prompt()}\n\n"
        f"## CONTEXTO:\n\n{context}\n\n"
        f"## PREGUNTA:\n{prompt}\n\n"
        f"## RESPUESTA:"
    )
    messages.append(HumanMessage(content=full_prompt))
    return messages


# ---------------------------------------------------------------------------
# LLM metadata helper
# ---------------------------------------------------------------------------

def _log_llm_meta(meta: dict):
    """Log token usage from vLLM/OpenAI response metadata."""
    if not meta:
        return
    usage         = meta.get("token_usage") or {}
    prompt_tokens = usage.get("prompt_tokens", 0)
    gen_tokens    = usage.get("completion_tokens", 0)
    model_name    = meta.get("model_name", LLM_MODEL)
    log.info(
        f"[vllm] prompt_tokens={prompt_tokens} "
        f"gen_tokens={gen_tokens} "
        f"model={model_name}"
    )


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class SessionCreateRequest(BaseModel):
    origin: Optional[str] = None


class ChatRequest(BaseModel):
    session_id: str
    message: str


class Source(BaseModel):
    titulo: str
    link: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    response: str
    sources: list[Source] = []
    response_time_ms: int
    llm_model: str
    embed_model: str
    chunk_config: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health(_: None = Depends(require_token)):
    try:
        conn = get_conn()
        conn.close()
        db = "ok"
    except Exception as e:
        db = f"error: {e}"
    return {
        "status":       "ok",
        "db":           db,
        "llm_model":    LLM_MODEL,
        "embed_model":  EMBED_MODEL,
        "chunk_config": CHUNK_CONFIG,
        "vllm_url":     VLLM_URL,
    }


@app.get("/api/debug/ping")
def debug_ping():
    """Debug endpoint - no token required. Used to test CORS and connectivity."""
    return {
        "status": "ok",
        "message": "API is accessible and CORS is working",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/debug/token")
def debug_token():
    """Debug endpoint - returns today's expected token without requiring authentication.
    Used to verify token generation on client side matches server expectations."""
    today = datetime.now()
    expected_token = _daily_token(today)
    date_str = today.strftime("%d/%m/%Y")
    return {
        "status": "ok",
        "date": date_str,
        "secret": WEBCHAT_SECRET,
        "expected_token": expected_token,
        "token_length": len(expected_token),
    }


@app.post("/api/session")
def new_session(req: SessionCreateRequest, _: None = Depends(require_token)):
    session_id = create_session(origin=req.origin)
    log.info(f"New session: {session_id} from {req.origin}")
    return {"session_id": session_id}


@app.get("/api/session/{session_id}")
def get_session(session_id: str, _: None = Depends(require_token)):
    if not session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    history = get_history(session_id)
    return {"session_id": session_id, "messages": history}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest, _: None = Depends(require_token)):
    # Validate / auto-create session
    if not req.session_id or not session_exists(req.session_id):
        raise HTTPException(status_code=404, detail="Session not found. POST /api/session first.")

    touch_session(req.session_id)

    # Save user message
    save_message(req.session_id, "user", req.message)

    t0 = time.monotonic()
    try:
        # Retrieve context + sources
        context, sources = search_context(req.message)
        if not context:
            context = "No se encontró contexto relevante en los documentos."
            sources = []

        # Get conversation history for multi-turn context
        history = get_history(req.session_id, limit=20)
        # Exclude the message we just saved (last item)
        history = history[:-1]

        # Build messages and call LLM
        messages = build_messages(history, req.message, context)
        response_msg = get_llm().invoke(messages)
        response_text = response_msg.content.strip()
        _log_llm_meta(response_msg.response_metadata)

    except Exception as e:
        log.error(f"Chat error: {e}")
        response_text = f"Lo siento, ocurrió un error al procesar tu pregunta: {e}"
        sources = []

    elapsed_ms = int((time.monotonic() - t0) * 1000)

    # Save assistant response
    save_message(
        req.session_id, "assistant", response_text,
        llm_model=LLM_MODEL, embed_model=EMBED_MODEL,
        chunk_config=CHUNK_CONFIG, response_time_ms=elapsed_ms,
    )

    log.info(f"session={req.session_id} time={elapsed_ms}ms")

    return ChatResponse(
        session_id=req.session_id,
        response=response_text,
        sources=[Source(**s) for s in sources],
        response_time_ms=elapsed_ms,
        llm_model=LLM_MODEL,
        embed_model=EMBED_MODEL,
        chunk_config=CHUNK_CONFIG,
    )


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest, _: None = Depends(require_token)):
    if not req.session_id or not session_exists(req.session_id):
        raise HTTPException(status_code=404, detail="Session not found. POST /api/session first.")

    touch_session(req.session_id)
    save_message(req.session_id, "user", req.message)

    t0 = time.monotonic()

    try:
        context, sources = search_context(req.message)
        if not context:
            context = "No se encontró contexto relevante en los documentos."
            sources = []
    except Exception as e:
        log.error(f"Context error: {e}")
        context = "No se encontró contexto relevante en los documentos."
        sources = []

    history = get_history(req.session_id, limit=20)[:-1]
    messages = build_messages(history, req.message, context)

    def generate():
        import json
        full_response: list[str] = []
        last_meta: dict = {}
        try:
            for chunk in get_llm().stream(messages):
                token = chunk.content
                if token:
                    full_response.append(token)
                    yield f"data: {json.dumps({'token': token})}\n\n"
                if chunk.response_metadata:
                    last_meta = chunk.response_metadata
        except Exception as e:
            log.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

        response_text = "".join(full_response)
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        _log_llm_meta(last_meta)

        save_message(
            req.session_id, "assistant", response_text,
            llm_model=LLM_MODEL, embed_model=EMBED_MODEL,
            chunk_config=CHUNK_CONFIG, response_time_ms=elapsed_ms,
        )
        log.info(f"stream session={req.session_id} time={elapsed_ms}ms")
        yield f"data: {json.dumps({'done': True, 'sources': sources, 'response_time_ms': elapsed_ms})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)
