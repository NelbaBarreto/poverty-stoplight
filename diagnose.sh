#!/bin/sh
# Diagnóstico de conectividad para el webchat API

echo "=== Diagnóstico Webchat API ==="
echo ""

# 1. Verificar si hay proceso escuchando en puerto 8000
echo "1. Verificando puerto 8000..."
if command -v ss >/dev/null 2>&1; then
    ss -tlnp 2>/dev/null | grep :8000 || echo "   No hay proceso escuchando en :8000"
elif command -v netstat >/dev/null 2>&1; then
    netstat -tlnp 2>/dev/null | grep :8000 || echo "   No hay proceso escuchando en :8000"
elif command -v lsof >/dev/null 2>&1; then
    lsof -i :8000 2>/dev/null || echo "   No hay proceso escuchando en :8000"
else
    echo "   No se pueden verificar puertos (falta ss, netstat o lsof)"
fi
echo ""

# 2. Test localhost
echo "2. Probando conexión a localhost:8000..."
if command -v curl >/dev/null 2>&1; then
    curl -s http://127.0.0.1:8000/api/health >/dev/null 2>&1 && echo "   ✓ localhost:8000 respondiendo" || echo "   ✗ localhost:8000 no responde"
fi
echo ""

# 3. Test IP local
echo "3. Probando conexión a 10.1.50.50:8000..."
if command -v curl >/dev/null 2>&1; then
    curl -s -m 2 http://10.1.50.50:8000/api/health >/dev/null 2>&1 && echo "   ✓ 10.1.50.50:8000 respondiendo" || echo "   ✗ 10.1.50.50:8000 no responde (verifica que sea la IP correcta del servidor)"
fi
echo ""

# 4. IP del servidor
echo "4. IPs del servidor:"
if command -v hostname >/dev/null 2>&1; then
    hostname -I 2>/dev/null || echo "   No se pudo obtener IPs"
fi
echo ""

# 5. Variables de entorno
echo "5. Variables de entorno (.env):"
if [ -f ".env" ]; then
    grep -E "API_PORT|DB_HOST|WEBCHAT_API" .env 2>/dev/null || echo "   No hay variables relevantes en .env"
else
    echo "   No existe archivo .env"
fi
