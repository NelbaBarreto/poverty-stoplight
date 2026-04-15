#!/bin/sh
# Detiene todos los servicios del proyecto.
echo "Deteniendo servicios..."

stop_port() {
    pids=$(lsof -ti :"$1" 2>/dev/null)
    if [ -n "$pids" ]; then
        echo "  Puerto $1 → PID $pids"
        kill $pids 2>/dev/null || true
    fi
}

# Por puerto
stop_port 8000   # Webchat API
stop_port 8502   # Admin Panel
stop_port 8503   # Webchat UI

# Por nombre de proceso (cubre casos donde el puerto ya fue liberado)
pkill -f "webchat.api"    2>/dev/null || true
pkill -f "webchat/app.py" 2>/dev/null || true
pkill -f "admin/app.py"   2>/dev/null || true
# Ollama se deja corriendo intencionalmente

echo "Listo."
