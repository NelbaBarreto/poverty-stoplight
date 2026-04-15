#!/bin/bash
# Lanza el panel de administración en el puerto 8502
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Cargar variables de entorno si existe .env
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

echo "Iniciando Admin Panel en http://localhost:8502"
streamlit run "$SCRIPT_DIR/app.py" \
    --server.port 8502 \
    --server.headless true \
    --server.address 0.0.0.0 \
    --server.fileWatcherType none
