#!/bin/sh
# Lanza el backend webchat (uvicorn :8000) y el cliente de prueba Streamlit (:8503).
# Crea y usa un virtualenv en .venv/ si las dependencias no están disponibles globalmente.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_ROOT/.venv"

# Cargar variables de entorno si existe .env
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    . "$PROJECT_ROOT/.env"
    set +a
fi

# ── Resolver python a usar ────────────────────────────────────────────────────
PYTHON=""

# 1. Usar el venv del proyecto si existe
if [ -f "$VENV_DIR/bin/python" ]; then
    PYTHON="$VENV_DIR/bin/python"
fi

# 2. Si no, buscar python global con uvicorn y streamlit
if [ -z "$PYTHON" ]; then
    for candidate in python3 python; do
        if command -v "$candidate" >/dev/null 2>&1; then
            if "$candidate" -m uvicorn --version >/dev/null 2>&1 && \
               "$candidate" -m streamlit --version >/dev/null 2>&1; then
                PYTHON="$candidate"
                break
            fi
        fi
    done
fi

# 3. Si tampoco, crear el venv e instalar dependencias
if [ -z "$PYTHON" ]; then
    echo "Dependencias no encontradas. Creando virtualenv en $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install --quiet --upgrade pip
    "$VENV_DIR/bin/pip" install --quiet -r "$PROJECT_ROOT/requirements.txt"
    PYTHON="$VENV_DIR/bin/python"
    echo "Virtualenv listo."
fi

echo "Python: $PYTHON"

# ── Kill any existing process on API_PORT ────────────────────────────────────
API_PORT="${API_PORT:-8000}"
echo "Verificando si el puerto $API_PORT está en uso..."
if lsof -Pi :${API_PORT} -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "Matando proceso existente en puerto $API_PORT..."
    kill $(lsof -t -i:${API_PORT}) 2>/dev/null || true
    sleep 1
fi
echo "Iniciando Webchat API en http://0.0.0.0:${API_PORT}"

"$PYTHON" -m uvicorn webchat.api:app \
    --host 0.0.0.0 \
    --port "$API_PORT" \
    --reload \
    --reload-dir "$SCRIPT_DIR" \
    --reload-dir "$PROJECT_ROOT/src" \
    --app-dir "$PROJECT_ROOT" &

API_PID=$!
echo "API PID: $API_PID"

# Esperar a que la API esté lista (hasta 15 s)
i=0
while [ $i -lt 15 ]; do
    if "$PYTHON" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${API_PORT}/api/health')" >/dev/null 2>&1; then
        echo "API lista."
        break
    fi
    sleep 1
    i=$((i + 1))
done

# ── Arrancar Streamlit (foreground) ──────────────────────────────────────────
STREAMLIT_PORT="${WEBCHAT_STREAMLIT_PORT:-8503}"
echo "Iniciando Webchat Test Client en http://localhost:${STREAMLIT_PORT}"

# Asegurarse de terminar uvicorn al salir
trap 'kill $API_PID 2>/dev/null; exit 0' INT TERM

"$PYTHON" -m streamlit run "$SCRIPT_DIR/app.py" \
    --server.port "$STREAMLIT_PORT" \
    --server.headless true \
    --server.address 0.0.0.0 \
    --server.fileWatcherType none
