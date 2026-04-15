#!/bin/sh
# Detiene el Webchat API (puerto 8000) y el Webchat UI (puerto 8503).
echo "Deteniendo Webchat..."

stop_port() {
    pids=$(lsof -ti :"$1" 2>/dev/null)
    if [ -n "$pids" ]; then
        echo "  Puerto $1 → PID $pids"
        kill $pids 2>/dev/null || true
    fi
}

stop_port 8000   # Webchat API
stop_port 8503   # Webchat UI

pkill -f "webchat.api"    2>/dev/null || true
pkill -f "webchat/app.py" 2>/dev/null || true

echo "Listo."
