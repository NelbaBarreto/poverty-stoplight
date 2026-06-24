#!/bin/bash
# Configura el servicio Ollama para 2x RTX 3090 en Ubuntu.
# Ejecutar en el servidor con: sudo bash scripts/setup_ollama_server.sh
set -e

SERVICE_FILE="/etc/systemd/system/ollama.service"
BACKUP_FILE="/etc/systemd/system/ollama.service.bak"

if [ "$EUID" -ne 0 ]; then
    echo "ERROR: Ejecutar con sudo: sudo bash $0"
    exit 1
fi

# Backup del servicio actual
if [ -f "$SERVICE_FILE" ]; then
    cp "$SERVICE_FILE" "$BACKUP_FILE"
    echo "Backup guardado en $BACKUP_FILE"
fi

cat > "$SERVICE_FILE" <<'EOF'
[Unit]
Description=Ollama Service
After=network-online.target

[Service]
ExecStart=/usr/local/bin/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
Environment="PATH=/home/aiserver/.local/bin:/usr/local/cuda-12.8/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/home/aiserver/.local/bin:/home/aiserver/.local/bin"
Environment="OLLAMA_NUM_GPU=99"
Environment="OLLAMA_NUM_PARALLEL=4"
Environment="OLLAMA_MAX_LOADED_MODELS=3"
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_KEEP_ALIVE=30m"

[Install]
WantedBy=default.target
EOF

echo "Servicio actualizado."

systemctl daemon-reload
systemctl restart ollama
sleep 3

if systemctl is-active --quiet ollama; then
    echo "Ollama corriendo OK."
    echo ""
    echo "Modelos cargados:"
    ollama ps 2>/dev/null || true
    echo ""
    echo "Uso de GPU:"
    nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu \
               --format=csv,noheader,nounits 2>/dev/null | \
        awk -F',' '{printf "  GPU: %s | VRAM usada: %s MB | libre: %s MB | uso: %s%%\n",$1,$2,$3,$4}' || true
else
    echo "ERROR: Ollama no arrancó. Revisar con: journalctl -u ollama -n 50"
    exit 1
fi
