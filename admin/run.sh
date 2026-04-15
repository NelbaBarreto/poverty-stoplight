#!/bin/sh
# Lanza el panel de administración en el puerto 8502
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Cargar variables de entorno si existe .env
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    . "$PROJECT_ROOT/.env"
    set +a
fi

# Detectar python con streamlit disponible
PYTHON=""
for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        if "$candidate" -m streamlit --version >/dev/null 2>&1; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "ERROR: No se encontró python con streamlit instalado." >&2
    echo "Instalá streamlit con: pip install streamlit" >&2
    exit 1
fi

echo "Iniciando Admin Panel en http://localhost:8502"
"$PYTHON" -m streamlit run "$SCRIPT_DIR/app.py" \
    --server.port 8502 \
    --server.headless true \
    --server.address 0.0.0.0 \
    --server.fileWatcherType none
