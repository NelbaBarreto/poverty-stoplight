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
  LLM_BACKEND    default: vllm           ("vllm" or "ollama")
  LLM_MODEL      default: per-backend    (vllm: Qwen/Qwen2.5-7B-Instruct | ollama: qwen3:8b)
  EMBED_MODEL    default: bge-m3         (always via Ollama)
  CHUNK_CONFIG   default: medium
  VLLM_BASE_URL  default: http://localhost:8800
  OLLAMA_BASE_URL default: http://localhost:11434
  API_PORT       default: 8000
  ALLOWED_ORIGINS  default: * (CORS)
"""

import hashlib
import itertools
import os
import sys
import time
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import psycopg2
import psycopg2.extras
import requests
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from sentence_transformers import SentenceTransformer
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

LLM_BACKEND     = os.getenv("LLM_BACKEND",    "vllm")   # "vllm" or "ollama"
# Comma-separated vLLM URLs — round-robin load balancing
# e.g. VLLM_BASE_URLS=http://localhost:8800,http://localhost:8801
_vllm_urls_raw  = os.getenv("VLLM_BASE_URLS", os.getenv("VLLM_BASE_URL", "http://localhost:8800"))
VLLM_URLS       = [u.strip() for u in _vllm_urls_raw.split(",") if u.strip()]
VLLM_URL        = VLLM_URLS[0]  # kept for display / embedding fallback
OLLAMA_URL      = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# Per-instance max_tokens — must match order of VLLM_BASE_URLS
# e.g. VLLM_MAX_TOKENS_LIST=2048,1024
_max_tokens_raw = os.getenv("VLLM_MAX_TOKENS_LIST", "")
_max_tokens_list = [int(x.strip()) for x in _max_tokens_raw.split(",") if x.strip()]
LLM_MAX_TOKENS  = int(os.getenv("LLM_MAX_TOKENS", "2048"))  # default / Ollama
RAG_K           = int(os.getenv("RAG_K", "5"))
THINKING_BUDGET = int(os.getenv("THINKING_BUDGET", "512"))

LLM_MODEL       = os.getenv("LLM_MODEL", "")   # if empty, auto-detected per vLLM instance
EMBED_MODEL     = os.getenv("EMBED_MODEL",     "bge-m3")
CHUNK_CONFIG    = os.getenv("CHUNK_CONFIG",    "medium")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
WEBCHAT_SECRET  = os.getenv("WEBCHAT_SECRET",  "W3bCh4tFup4")

# ---------------------------------------------------------------------------
# LLM pool — one ChatOpenAI/ChatOllama per backend URL, round-robin
# ---------------------------------------------------------------------------

_llm_pool: list = []
_llm_cycle = None

def _detect_vllm_model(url: str) -> str:
    """Query /v1/models to get the loaded model name for this vLLM instance."""
    try:
        resp = requests.get(f"{url}/v1/models",
                            headers={"Authorization": "Bearer EMPTY"}, timeout=10)
        resp.raise_for_status()
        models = resp.json().get("data", [])
        if models:
            return models[0]["id"]
    except Exception as e:
        log.warning(f"[llm] could not detect model at {url}: {e}")
    return "unknown"

def _build_llm_pool():
    """Build one LLM instance per vLLM URL (or single Ollama instance)."""
    global _llm_pool, _llm_cycle
    pool = []
    if LLM_BACKEND == "ollama":
        model = LLM_MODEL or "qwen3:8b"
        pool.append(ChatOllama(model=model, temperature=0, base_url=OLLAMA_URL,
                               num_ctx=LLM_MAX_TOKENS, think=False))
        log.info(f"[llm] ollama pool: 1 instance model={model}")
    else:
        # THINKING_BUDGET: >0 = enable with token budget, 0 = disable, -1 = enable unlimited
        if THINKING_BUDGET > 0:
            extra = {"chat_template_kwargs": {"enable_thinking": True,
                                              "thinking_budget": THINKING_BUDGET}}
        elif THINKING_BUDGET == 0:
            extra = {"chat_template_kwargs": {"enable_thinking": False}}
        else:
            extra = {}
        for i, url in enumerate(VLLM_URLS):
            model = LLM_MODEL or _detect_vllm_model(url)
            max_tok = _max_tokens_list[i] if i < len(_max_tokens_list) else LLM_MAX_TOKENS
            instance = ChatOpenAI(model=model, temperature=0,
                                  base_url=f"{url}/v1", api_key="EMPTY",
                                  max_tokens=max_tok,
                                  model_kwargs={"extra_body": extra} if extra else {})
            pool.append(instance)
            log.info(f"[llm] vllm pool: {url} model={model} max_tokens={max_tok}")
    _llm_pool = pool
    _llm_cycle = itertools.cycle(pool)

def get_llm():
    """Return next LLM instance in round-robin rotation."""
    if not _llm_pool:
        _build_llm_pool()
    return next(_llm_cycle)

# ---------------------------------------------------------------------------
# Token rotativo diario — sha512(SECRET + DD/MM/YYYY)
# ---------------------------------------------------------------------------

def _daily_token(dt: datetime) -> str:
    date_str = dt.strftime("%d/%m/%Y")
    raw = f"{WEBCHAT_SECRET}{date_str}"
    token = hashlib.sha512(raw.encode()).hexdigest()
    return token


def require_token(x_auth_token: str = Header(..., alias="X-Auth-Token")):
    expected = _daily_token(datetime.now(timezone.utc))
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
_FALLBACK_PROMPT = "Eres Rosa, la asistente conversacional del Banco de Soluciones de la Fundación Paraguaya."

def load_system_prompt() -> str:
    """Read prompt from prompt.txt on every request (fast file read, OS-cached)."""
    if _PROMPT_FILE.exists():
        return _PROMPT_FILE.read_text(encoding="utf-8").strip()
    log.warning("[prompt] prompt.txt not found — using fallback")
    return _FALLBACK_PROMPT

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

# Middleware must be registered before event handlers
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _sync_prompt_from_db():
    """Pull latest prompt from DB and write to prompt.txt so every request reads the file."""
    try:
        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT prompt_text FROM prompt_history ORDER BY created_at DESC LIMIT 1")
            row = cur.fetchone()
        conn.close()
        if row and row["prompt_text"]:
            _PROMPT_FILE.write_text(row["prompt_text"].strip(), encoding="utf-8")
            log.info(f"[prompt] synced from DB → {_PROMPT_FILE} ({len(row['prompt_text'])} chars)")
            return
    except Exception as e:
        log.warning(f"[prompt] DB sync failed: {e}")
    log.info(f"[prompt] using existing {_PROMPT_FILE}")

@app.on_event("startup")
def startup_events():
    _sync_prompt_from_db()
    _build_llm_pool()
    if LLM_BACKEND != "ollama":
        _get_st_model()  # preload embedding model — avoids 6s cold start on first request

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
                 response_time_ms=None) -> Optional[int]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO chat_messages
                   (session_id, role, content, llm_model, embed_model, chunk_config, response_time_ms)
                   VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id""",
                (session_id, role, content, llm_model, embed_model, chunk_config, response_time_ms),
            )
            row = cur.fetchone()
        conn.commit()
        return row["id"] if row else None
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

_EMBED_MODEL_MAP = {
    "bge-m3":                 "BAAI/bge-m3",
    "nomic-embed-text":       "nomic-ai/nomic-embed-text-v1",
    "mxbai-embed-large":      "mixedbread-ai/mxbai-embed-large-v1",
    "all-minilm":             "sentence-transformers/all-MiniLM-L6-v2",
    "snowflake-arctic-embed": "Snowflake/snowflake-arctic-embed-m",
}
_st_model: SentenceTransformer | None = None

def _get_st_model() -> SentenceTransformer:
    global _st_model
    if _st_model is None:
        hf_name = _EMBED_MODEL_MAP.get(EMBED_MODEL, "BAAI/bge-m3")
        log.info(f"[embed] loading sentence-transformers model: {hf_name}")
        # local_files_only avoids HuggingFace network calls on every startup
        try:
            _st_model = SentenceTransformer(hf_name, trust_remote_code=True,
                                             local_files_only=True)
        except Exception:
            log.info(f"[embed] model not cached locally, downloading...")
            _st_model = SentenceTransformer(hf_name, trust_remote_code=True)
        log.info(f"[embed] model loaded")
    return _st_model

def get_embedding(query: str) -> list:
    t0 = time.monotonic()
    if LLM_BACKEND == "ollama":
        resp = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": query},
            timeout=60,
        )
        resp.raise_for_status()
        vector = resp.json()["embedding"]
    else:
        model = _get_st_model()
        vector = model.encode(query, normalize_embeddings=True).tolist()
    log.info(f"[timing] embed={int((time.monotonic()-t0)*1000)}ms")
    return vector


def search_context(query: str, k: int = RAG_K) -> tuple:
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
    system_prompt = load_system_prompt()
    system_content = (
        f"{system_prompt}\n\n"
        f"IMPORTANTE: No incluyas la lista de fuentes al final de tu respuesta. "
        f"Las fuentes se muestran automáticamente en la interfaz.\n\n"
        f"## CONTEXTO RELEVANTE:\n\n{context}"
    )
    messages = [SystemMessage(content=system_content)]

    # Include last 6 turns for multi-turn context
    for msg in history[-6:]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=prompt))

    log.debug(f"[prompt] system_prompt ({len(system_prompt)} chars): {system_prompt[:120]}...")
    log.debug(f"[prompt] context ({len(context)} chars), history_turns={len(history)}, question={prompt[:80]}")
    log.info(f"[prompt] messages={len(messages)} system={len(system_content)}chars question={len(prompt)}chars")

    return messages


# ---------------------------------------------------------------------------
# LLM metadata helper
# ---------------------------------------------------------------------------

def _log_llm_meta(meta: dict):
    """Log token usage — handles both Ollama and vLLM/OpenAI response metadata."""
    if not meta:
        return
    if LLM_BACKEND == "ollama":
        eval_count  = meta.get("eval_count", 0)
        eval_dur_ns = meta.get("eval_duration", 0)
        tok_per_sec = eval_count / (eval_dur_ns / 1e9) if eval_dur_ns else 0
        log.info(f"[ollama] prompt_tokens={meta.get('prompt_eval_count', 0)} "
                 f"gen_tokens={eval_count} speed={tok_per_sec:.1f}tok/s")
    else:
        usage = meta.get("token_usage") or {}
        log.info(f"[vllm] prompt_tokens={usage.get('prompt_tokens', 0)} "
                 f"gen_tokens={usage.get('completion_tokens', 0)} "
                 f"model={meta.get('model_name', LLM_MODEL)}")


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
    message_id: Optional[int] = None


class RateRequest(BaseModel):
    message_id: int
    rating: int  # 1 = thumbs up, -1 = thumbs down


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
        "status":         "ok",
        "db":             db,
        "llm_backend":    LLM_BACKEND,
        "llm_model":      LLM_MODEL or "auto-detected",
        "vllm_urls":      VLLM_URLS,
        "pool_size":      len(_llm_pool),
        "embed_model":    EMBED_MODEL,
        "chunk_config":   CHUNK_CONFIG,
        "llm_max_tokens": LLM_MAX_TOKENS,
        "ollama_url":     OLLAMA_URL,
    }


@app.post("/api/reload-prompt")
def reload_prompt(_: None = Depends(require_token)):
    """Sync latest prompt from DB to prompt.txt. Call from admin UI after saving a new prompt."""
    _sync_prompt_from_db()
    prompt = load_system_prompt()
    return {"status": "ok", "prompt_length": len(prompt)}


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
    from fastapi.responses import JSONResponse
    today = datetime.now(timezone.utc)
    expected_token = _daily_token(today)
    date_str = today.strftime("%d/%m/%Y")
    return JSONResponse(
        content={
            "status": "ok",
            "date": date_str,
            "secret": WEBCHAT_SECRET,
            "expected_token": expected_token,
            "token_length": len(expected_token),
        },
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


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
    msg_id = save_message(
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
        message_id=msg_id,
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

        msg_id = save_message(
            req.session_id, "assistant", response_text,
            llm_model=LLM_MODEL, embed_model=EMBED_MODEL,
            chunk_config=CHUNK_CONFIG, response_time_ms=elapsed_ms,
        )
        log.info(f"stream session={req.session_id} time={elapsed_ms}ms")
        yield f"data: {json.dumps({'done': True, 'sources': sources, 'response_time_ms': elapsed_ms, 'message_id': msg_id})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/chat/rate")
def rate_message(req: RateRequest, _: None = Depends(require_token)):
    if req.rating not in (1, -1):
        raise HTTPException(status_code=400, detail="rating must be 1 or -1")
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE chat_messages SET rating = %s WHERE id = %s",
                (req.rating, req.message_id),
            )
            if cur.rowcount == 0:
                raise HTTPException(status_code=404, detail="Message not found")
        conn.commit()
    finally:
        conn.close()
    log.info(f"[rating] message_id={req.message_id} rating={req.rating}")
    return {"ok": True}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)
