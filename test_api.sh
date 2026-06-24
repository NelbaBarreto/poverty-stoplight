#!/bin/sh
# Test completo del API con token correcto

API_URL="http://10.1.50.50:8000"
SECRET="W3bCh4tFup4"

# Generar token SHA512(SECRET + DD/MM/YYYY)
TODAY=$(date +"%d/%m/%Y")
TOKEN=$(echo -n "${SECRET}${TODAY}" | shasum -a 512 | cut -d' ' -f1)

echo "=== Test Webchat API ==="
echo "API URL: $API_URL"
echo "Date: $TODAY"
echo "Token: $TOKEN"
echo ""

echo "1. Testing /api/health..."
curl -s -H "X-Auth-Token: $TOKEN" "$API_URL/api/health" | python3 -m json.tool 2>/dev/null || echo "Error en health check"
echo ""

echo "2. Creating session..."
SESSION_RESPONSE=$(curl -s -H "X-Auth-Token: $TOKEN" -H "Content-Type: application/json" -d '{"origin":"test"}' "$API_URL/api/session")
echo "$SESSION_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "Error creating session"

SESSION_ID=$(echo "$SESSION_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('session_id', ''))" 2>/dev/null)
echo "Session ID: $SESSION_ID"
echo ""

if [ -n "$SESSION_ID" ]; then
    echo "3. Testing /api/chat..."
    curl -s -H "X-Auth-Token: $TOKEN" -H "Content-Type: application/json" \
      -d "{\"session_id\":\"$SESSION_ID\",\"message\":\"Hola\"}" \
      "$API_URL/api/chat" | python3 -m json.tool 2>/dev/null || echo "Error en chat"
fi
