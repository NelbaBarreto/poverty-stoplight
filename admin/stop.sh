#!/bin/sh
# Detiene el Admin Panel (puerto 8502).
echo "Deteniendo Admin Panel..."

pids=$(lsof -ti :8502 2>/dev/null)
if [ -n "$pids" ]; then
    echo "  Puerto 8502 → PID $pids"
    kill $pids 2>/dev/null || true
fi

pkill -f "admin/app.py" 2>/dev/null || true

echo "Listo."
