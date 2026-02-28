#!/usr/bin/env bash
# run_ingestion.sh — Orquestación completa del pipeline de ingesta
# Uso: bash scripts/run_ingestion.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "========================================================"
echo "  Poverty Stoplight — Pipeline de ingesta"
echo "========================================================"

# ── 1. Bajar contenedor y borrar volumen ─────────────────────
echo ""
echo "[1/5] Bajando contenedor y borrando volumen anterior..."
docker compose down -v

# ── 2. Pull de modelos Ollama faltantes ──────────────────────
echo ""
echo "[2/5] Descargando modelos Ollama (puede tardar si son nuevos)..."
ollama pull nomic-embed-text
ollama pull mxbai-embed-large
ollama pull all-minilm
ollama pull snowflake-arctic-embed
echo "  bge-m3 se asume ya descargado."

# ── 3. Levantar BD fresca ────────────────────────────────────
echo ""
echo "[3/5] Levantando PostgreSQL con esquema nuevo..."
docker compose up -d pgvector-db

# ── 4. Esperar que la BD esté lista ──────────────────────────
echo ""
echo "[4/5] Esperando que la BD esté disponible..."
until docker exec pgvector-db pg_isready -U sgadmin -d db_rag 2>/dev/null; do
    echo "  Esperando DB..."
    sleep 2
done
echo "  DB lista."

# ── 5. Ejecutar pipeline de ingesta ──────────────────────────
echo ""
echo "[5/5] Ejecutando pipeline de ingesta..."
python scripts/ingest_all.py

echo ""
echo "========================================================"
echo "  Ingesta completada exitosamente."
echo "========================================================"
echo ""
echo "Verificación rápida:"
echo "  docker exec -it pgvector-db psql -U sgadmin -d db_rag -c 'SELECT COUNT(*) FROM chunks;'"
echo "  docker exec -it pgvector-db psql -U sgadmin -d db_rag -c 'SELECT COUNT(*) FROM embeddings_bge_m3;'"
