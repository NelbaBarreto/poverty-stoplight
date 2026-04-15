#!/bin/sh
# Lanza el panel de administración en el puerto 8502.
# Crea y usa un virtualenv en .venv/ si streamlit no está disponible globalmente.
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
# Prioridad: venv propio → python global con streamlit → error
PYTHON=""

# 1. Usar el venv del proyecto si existe
if [ -f "$VENV_DIR/bin/python" ]; then
    PYTHON="$VENV_DIR/bin/python"
fi

# 2. Si no, buscar python global con streamlit
if [ -z "$PYTHON" ]; then
    for candidate in python3 python; do
        if command -v "$candidate" >/dev/null 2>&1; then
            if "$candidate" -m streamlit --version >/dev/null 2>&1; then
                PYTHON="$candidate"
                break
            fi
        fi
    done
fi

# 3. Si tampoco, crear el venv y instalar dependencias
if [ -z "$PYTHON" ]; then
    echo "streamlit no encontrado. Creando virtualenv en $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install --quiet --upgrade pip
    "$VENV_DIR/bin/pip" install --quiet -r "$PROJECT_ROOT/requirements.txt"
    PYTHON="$VENV_DIR/bin/python"
    echo "Virtualenv listo."
fi

echo "Iniciando Admin Panel en http://localhost:8502"
echo "Python: $PYTHON"

"$PYTHON" -m streamlit run "$SCRIPT_DIR/app.py" \
    --server.port 8502 \
    --server.headless true \
    --server.address 0.0.0.0 \
    --server.fileWatcherType none
