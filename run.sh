#!/bin/sh
# Arranca todos los servicios del proyecto:
#   - Ollama (si no está ya corriendo)
#   - Webchat API      → http://localhost:8000
#   - Webchat UI       → http://localhost:8503
#   - Admin Panel      → http://localhost:8502
#
# Uso: sh run.sh [--no-admin] [--no-ollama]
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

# Cargar variables de entorno
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    . "$SCRIPT_DIR/.env"
    set +a
fi

# ── Parse flags ───────────────────────────────────────────────────────────────
START_ADMIN=1
START_OLLAMA=1
for arg in "$@"; do
    case "$arg" in
        --no-admin)  START_ADMIN=0 ;;
        --no-ollama) START_OLLAMA=0 ;;
    esac
done

# ── Resolver python ───────────────────────────────────────────────────────────
PYTHON=""
if [ -f "$VENV_DIR/bin/python" ]; then
    PYTHON="$VENV_DIR/bin/python"
else
    for candidate in python3 python; do
        if command -v "$candidate" >/dev/null 2>&1; then
            if "$candidate" -m uvicorn --version >/dev/null 2>&1; then
                PYTHON="$candidate"
                break
            fi
        fi
    done
fi

if [ -z "$PYTHON" ]; then
    echo "ERROR: Python con dependencias no encontrado. Corre primero: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi

echo "Python: $PYTHON"

# ── Cleanup al salir ──────────────────────────────────────────────────────────
PIDS=""
cleanup() {
    echo ""
    echo "Deteniendo servicios..."
    for pid in $PIDS; do
        kill "$pid" 2>/dev/null || true
    done
    exit 0
}
trap cleanup INT TERM

# ── Ollama ────────────────────────────────────────────────────────────────────
if [ "$START_OLLAMA" = "1" ]; then
    if ! curl -s "http://localhost:11434" >/dev/null 2>&1; then
        if command -v ollama >/dev/null 2>&1; then
            echo "Iniciando Ollama..."
            ollama serve > "$SCRIPT_DIR/logs/ollama.log" 2>&1 &
            PIDS="$PIDS $!"
            # Esperar hasta 15 s
            i=0
            while [ $i -lt 15 ]; do
                if curl -s "http://localhost:11434" >/dev/null 2>&1; then
                    echo "Ollama lista."
                    break
                fi
                sleep 1
                i=$((i + 1))
            done
        else
            echo "AVISO: ollama no encontrado en PATH. Asegúrate de iniciarlo manualmente."
        fi
    else
        echo "Ollama ya está corriendo."
    fi
fi

# ── Webchat API (uvicorn, background) ─────────────────────────────────────────
API_PORT="${API_PORT:-8000}"
echo "Iniciando Webchat API en http://localhost:${API_PORT}"
mkdir -p "$SCRIPT_DIR/logs"
"$PYTHON" -m uvicorn webchat.api:app \
    --host 0.0.0.0 \
    --port "$API_PORT" \
    --reload \
    --reload-dir "$SCRIPT_DIR/webchat" \
    --reload-dir "$SCRIPT_DIR/src" \
    --app-dir "$SCRIPT_DIR" \
    > "$SCRIPT_DIR/logs/webchat_api.log" 2>&1 &
PIDS="$PIDS $!"

# Esperar a que la API esté lista
i=0
while [ $i -lt 20 ]; do
    if "$PYTHON" -c "import urllib.request; urllib.request.urlopen('http://localhost:${API_PORT}/api/health')" >/dev/null 2>&1; then
        echo "Webchat API lista."
        break
    fi
    sleep 1
    i=$((i + 1))
done

# ── Webchat Streamlit UI (background) ─────────────────────────────────────────
WEBCHAT_PORT="${WEBCHAT_STREAMLIT_PORT:-8503}"
echo "Iniciando Webchat UI en http://localhost:${WEBCHAT_PORT}"
"$PYTHON" -m streamlit run "$SCRIPT_DIR/webchat/app.py" \
    --server.port "$WEBCHAT_PORT" \
    --server.headless true \
    --server.address 0.0.0.0 \
    --server.fileWatcherType none \
    > "$SCRIPT_DIR/logs/webchat_ui.log" 2>&1 &
PIDS="$PIDS $!"

# ── Admin Panel (background, optional) ────────────────────────────────────────
if [ "$START_ADMIN" = "1" ]; then
    ADMIN_PORT="${ADMIN_STREAMLIT_PORT:-8502}"
    echo "Iniciando Admin Panel en http://localhost:${ADMIN_PORT}"
    "$PYTHON" -m streamlit run "$SCRIPT_DIR/admin/app.py" \
        --server.port "$ADMIN_PORT" \
        --server.headless true \
        --server.address 0.0.0.0 \
        --server.fileWatcherType none \
        > "$SCRIPT_DIR/logs/admin.log" 2>&1 &
    PIDS="$PIDS $!"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Servicios activos:"
echo "   Webchat API  → http://localhost:${API_PORT}"
echo "   Webchat API docs → http://localhost:${API_PORT}/docs"
echo "   Webchat UI   → http://localhost:${WEBCHAT_PORT}"
[ "$START_ADMIN" = "1" ] && echo "   Admin Panel  → http://localhost:${ADMIN_PORT}"
[ "$START_OLLAMA" = "1" ] && echo "   Ollama       → http://localhost:11434"
echo ""
echo " Logs en: $SCRIPT_DIR/logs/"
echo " Ctrl+C para detener todo."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Mantener vivo
wait
