#!/usr/bin/env python3
"""
Webchat test client — Poverty Stoplight
Streamlit app that consumes the webchat API (api.py).
Mirrors what the WordPress widget does: stores session_id, calls /api/chat.

Run: streamlit run webchat/app.py --server.port 8503
"""

import os
import time
import requests
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / ".env")

API_BASE = os.getenv("WEBCHAT_API_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Luz — Webchat Test",
    page_icon="💡",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Session bootstrap
# ---------------------------------------------------------------------------

def init_session():
    """Create API session on first load and store in st.session_state."""
    if "session_id" not in st.session_state or not st.session_state.session_id:
        try:
            resp = requests.post(
                f"{API_BASE}/api/session",
                json={"origin": "streamlit-test"},
                timeout=10,
            )
            resp.raise_for_status()
            st.session_state.session_id = resp.json()["session_id"]
        except Exception as e:
            st.error(f"No se pudo conectar con la API: {e}")
            st.session_state.session_id = None

    if "messages" not in st.session_state:
        st.session_state.messages = []


def load_history_from_api():
    """Load existing messages from the API for this session."""
    sid = st.session_state.get("session_id")
    if not sid:
        return
    try:
        resp = requests.get(f"{API_BASE}/api/session/{sid}", timeout=10)
        if resp.status_code == 200:
            msgs = resp.json().get("messages", [])
            st.session_state.messages = [
                {"role": m["role"], "content": m["content"]} for m in msgs
            ]
    except Exception:
        pass


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

init_session()

st.title("💡 Luz — Webchat (Test Client)")
st.caption(
    f"API: `{API_BASE}` | "
    f"Session: `{st.session_state.get('session_id', 'sin sesión')}`"
)

with st.sidebar:
    st.header("Sesión")
    if st.session_state.get("session_id"):
        st.success(f"**ID:** `{st.session_state.session_id}`")
    else:
        st.error("Sin sesión activa")

    if st.button("↺ Nueva sesión"):
        st.session_state.session_id = None
        st.session_state.messages = []
        st.rerun()

    if st.button("📂 Recargar historial"):
        load_history_from_api()
        st.rerun()

    st.divider()

    # API health check
    st.header("API Status")
    try:
        h = requests.get(f"{API_BASE}/api/health", timeout=5).json()
        st.success("API online")
        st.json(h)
    except Exception as e:
        st.error(f"API offline: {e}")

# ── Chat messages ─────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Input ─────────────────────────────────────────────────────────────────

prompt = st.chat_input("Escribe tu pregunta...")
if prompt and st.session_state.get("session_id"):
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call API
    with st.chat_message("assistant"):
        with st.spinner("Luz está pensando..."):
            try:
                t0 = time.monotonic()
                resp = requests.post(
                    f"{API_BASE}/api/chat",
                    json={"session_id": st.session_state.session_id, "message": prompt},
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
                answer = data["response"]
                elapsed = data.get("response_time_ms", int((time.monotonic()-t0)*1000))
            except requests.HTTPError as e:
                answer = f"Error de API ({e.response.status_code}): {e.response.text}"
                elapsed = 0
            except Exception as e:
                answer = f"Error de conexión: {e}"
                elapsed = 0

        st.markdown(answer)
        st.caption(f"⏱ {elapsed} ms · LLM: {data.get('llm_model','?')} · "
                   f"Embed: {data.get('embed_model','?')} · "
                   f"Chunk: {data.get('chunk_config','?')}")

    st.session_state.messages.append({"role": "assistant", "content": answer})

elif prompt and not st.session_state.get("session_id"):
    st.warning("Sin sesión activa. Recarga la página.")
