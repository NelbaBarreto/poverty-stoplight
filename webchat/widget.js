/**
 * Luz Webchat Widget — Poverty Stoplight
 * Compact vanilla JS widget for WordPress (no dependencies).
 *
 * Usage in WordPress (HTML block or header):
 *   <script src="https://your-server/widget.js"
 *           data-api-url="https://your-server"
 *           data-title="Luz"
 *           data-subtitle="Asistente del Banco de Soluciones"
 *           data-placeholder="Escribe tu pregunta..."
 *           data-accent="#00C06B">
 *   </script>
 */
(function () {
  "use strict";

  /* ── Config from script tag ────────────────────────────────────────────── */
  const scriptTag =
    document.currentScript ||
    document.querySelector("script[data-api-url]");

  const API_URL     = (scriptTag && scriptTag.getAttribute("data-api-url"))     || "http://localhost:8000";
  const TITLE       = (scriptTag && scriptTag.getAttribute("data-title"))        || "Rosa";
  const SUBTITLE    = (scriptTag && scriptTag.getAttribute("data-subtitle"))     || "Asistente del Banco de Soluciones";
  const PLACEHOLDER = (scriptTag && scriptTag.getAttribute("data-placeholder"))  || "Escribe tu pregunta...";
  const ACCENT      = (scriptTag && scriptTag.getAttribute("data-accent"))       || "#00C06B";
  const STORAGE_KEY = "luz_session_id";

  /* ── Session management ────────────────────────────────────────────────── */
  function getSessionId() {
    return localStorage.getItem(STORAGE_KEY);
  }

  function setSessionId(id) {
    localStorage.setItem(STORAGE_KEY, id);
  }

  async function ensureSession() {
    let sid = getSessionId();
    if (sid) return sid;

    const res = await fetch(API_URL + "/api/session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ origin: window.location.href }),
    });
    const data = await res.json();
    setSessionId(data.session_id);
    return data.session_id;
  }

  async function loadHistory(sid) {
    try {
      const res = await fetch(API_URL + "/api/session/" + sid);
      if (!res.ok) return [];
      const data = await res.json();
      return data.messages || [];
    } catch {
      return [];
    }
  }

  /* ── CSS ───────────────────────────────────────────────────────────────── */
  const css = `
  #luz-btn {
    position: fixed; bottom: 24px; right: 24px; z-index: 99999;
    width: 56px; height: 56px; border-radius: 50%;
    background: ${ACCENT}; border: none; cursor: pointer;
    box-shadow: 0 4px 16px rgba(0,0,0,.25);
    display: flex; align-items: center; justify-content: center;
    transition: transform .2s, box-shadow .2s;
  }
  #luz-btn:hover { transform: scale(1.08); box-shadow: 0 6px 22px rgba(0,0,0,.32); }
  #luz-btn svg { width: 26px; height: 26px; fill: #fff; }

  #luz-panel {
    position: fixed; bottom: 92px; right: 24px; z-index: 99998;
    width: 360px; max-height: 560px;
    background: #fff; border-radius: 16px;
    box-shadow: 0 8px 40px rgba(0,0,0,.18);
    display: flex; flex-direction: column;
    overflow: hidden; font-family: system-ui, sans-serif;
    transition: opacity .2s, transform .2s;
  }
  #luz-panel.luz-hidden { opacity: 0; pointer-events: none; transform: translateY(12px); }

  #luz-header {
    background: ${ACCENT}; color: #fff;
    padding: 14px 16px; display: flex; align-items: center; gap: 10px;
  }
  #luz-header-icon { font-size: 1.4rem; }
  #luz-header-text .luz-title { font-weight: 700; font-size: .95rem; }
  #luz-header-text .luz-sub   { font-size: .72rem; opacity: .85; margin-top: 1px; }
  #luz-close {
    margin-left: auto; background: none; border: none; color: #fff;
    cursor: pointer; font-size: 1.2rem; padding: 0 4px; line-height: 1;
  }

  #luz-messages {
    flex: 1; overflow-y: auto; padding: 14px 12px;
    display: flex; flex-direction: column; gap: 10px;
    background: #f7f9fb;
  }
  .luz-msg {
    max-width: 84%; padding: 9px 12px; border-radius: 12px;
    font-size: .84rem; line-height: 1.5; word-break: break-word;
  }
  .luz-msg.user {
    align-self: flex-end; background: ${ACCENT}; color: #fff;
    border-bottom-right-radius: 3px;
  }
  .luz-msg.assistant {
    align-self: flex-start; background: #fff; color: #1a1a1a;
    border: 1px solid #e5e7eb; border-bottom-left-radius: 3px;
  }
  .luz-msg.typing { opacity: .6; font-style: italic; }

  #luz-footer {
    padding: 10px 12px; background: #fff;
    border-top: 1px solid #e5e7eb;
    display: flex; gap: 8px;
  }
  #luz-input {
    flex: 1; border: 1px solid #d1d5db; border-radius: 8px;
    padding: 8px 12px; font-size: .84rem; outline: none;
    font-family: inherit; resize: none; line-height: 1.4;
  }
  #luz-input:focus { border-color: ${ACCENT}; }
  #luz-send {
    background: ${ACCENT}; border: none; border-radius: 8px;
    width: 38px; height: 38px; cursor: pointer; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    transition: opacity .15s;
  }
  #luz-send:disabled { opacity: .45; cursor: not-allowed; }
  #luz-send svg { width: 18px; height: 18px; fill: #fff; }
  `;

  /* ── HTML ──────────────────────────────────────────────────────────────── */
  function buildPanel() {
    return `
    <div id="luz-header">
      <span id="luz-header-icon">💡</span>
      <div id="luz-header-text">
        <div class="luz-title">${TITLE}</div>
        <div class="luz-sub">${SUBTITLE}</div>
      </div>
      <button id="luz-close" aria-label="Cerrar">✕</button>
    </div>
    <div id="luz-messages"></div>
    <div id="luz-footer">
      <textarea id="luz-input" rows="1" placeholder="${PLACEHOLDER}"></textarea>
      <button id="luz-send" aria-label="Enviar">
        <svg viewBox="0 0 24 24"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>
      </button>
    </div>`;
  }

  /* ── DOM bootstrap ─────────────────────────────────────────────────────── */
  function inject() {
    // Style
    const style = document.createElement("style");
    style.textContent = css;
    document.head.appendChild(style);

    // FAB button
    const btn = document.createElement("button");
    btn.id = "luz-btn";
    btn.setAttribute("aria-label", "Abrir chat");
    btn.innerHTML = `<svg viewBox="0 0 24 24"><path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2z"/></svg>`;
    document.body.appendChild(btn);

    // Panel
    const panel = document.createElement("div");
    panel.id = "luz-panel";
    panel.className = "luz-hidden";
    panel.innerHTML = buildPanel();
    document.body.appendChild(panel);

    return { btn, panel };
  }

  /* ── Message rendering ─────────────────────────────────────────────────── */
  function appendMessage(container, role, text) {
    const div = document.createElement("div");
    div.className = "luz-msg " + role;
    div.textContent = text;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
    return div;
  }

  /* ── Send message ──────────────────────────────────────────────────────── */
  async function sendMessage(sessionId, text, msgContainer, sendBtn, input) {
    sendBtn.disabled = true;
    appendMessage(msgContainer, "user", text);

    const typing = appendMessage(msgContainer, "assistant typing", "Rosa está pensando...");

    try {
      const res = await fetch(API_URL + "/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, message: text }),
      });

      if (!res.ok) {
        const err = await res.text();
        throw new Error(err);
      }

      const data = await res.json();
      typing.remove();
      appendMessage(msgContainer, "assistant", data.response);
    } catch (e) {
      typing.remove();
      appendMessage(msgContainer, "assistant", "Lo siento, ocurrió un error. Intenta de nuevo.");
      console.error("[Luz widget]", e);
    } finally {
      sendBtn.disabled = false;
      input.value = "";
      input.style.height = "auto";
    }
  }

  /* ── Init ──────────────────────────────────────────────────────────────── */
  async function init() {
    const { btn, panel } = inject();
    const msgContainer = panel.querySelector("#luz-messages");
    const input        = panel.querySelector("#luz-input");
    const sendBtn      = panel.querySelector("#luz-send");
    const closeBtn     = panel.querySelector("#luz-close");

    let open = false;
    let sessionId = null;
    let historyLoaded = false;

    // Toggle panel
    btn.addEventListener("click", async () => {
      open = !open;
      panel.classList.toggle("luz-hidden", !open);

      if (open && !historyLoaded) {
        historyLoaded = true;
        sessionId = await ensureSession();

        // Load previous messages
        const history = await loadHistory(sessionId);
        history.forEach((m) => appendMessage(msgContainer, m.role, m.content));

        if (history.length === 0) {
          appendMessage(
            msgContainer,
            "assistant",
            `¡Hola! Soy Rosa, la asistente del Banco de Soluciones. ¿En qué puedo ayudarte hoy?`
          );
        }
        input.focus();
      }
    });

    closeBtn.addEventListener("click", () => {
      open = false;
      panel.classList.add("luz-hidden");
    });

    // Auto-resize textarea
    input.addEventListener("input", () => {
      input.style.height = "auto";
      input.style.height = Math.min(input.scrollHeight, 100) + "px";
    });

    // Send on Enter (Shift+Enter = new line)
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        const text = input.value.trim();
        if (text && sessionId) {
          sendMessage(sessionId, text, msgContainer, sendBtn, input);
        }
      }
    });

    sendBtn.addEventListener("click", () => {
      const text = input.value.trim();
      if (text && sessionId) {
        sendMessage(sessionId, text, msgContainer, sendBtn, input);
      }
    });
  }

  // Run after DOM is ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
