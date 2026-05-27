#!/usr/bin/env bash
# Adaptive Minds — vLLM endpoint examples (OpenAI-compatible /v1).
#
# Prereq: start a vLLM server using the catalog adapters, e.g.
#     $(adaptive-minds serve --catalog catalogs/qwen25_30.yaml) &
# Then either source .env or:
#     export VLLM_BASE=http://localhost:8000/v1

set -euo pipefail
VLLM_BASE="${VLLM_BASE:-http://localhost:8000/v1}"

post() {
    local model="$1"; shift
    local user="$1"; shift
    curl -sS "${VLLM_BASE}/chat/completions" \
        -H "Content-Type: application/json" \
        -d @- <<EOF | python3 -m json.tool
{
  "model": "${model}",
  "temperature": 0.3,
  "max_tokens": 256,
  "messages": [
    {"role": "user", "content": ${user}}
  ]
}
EOF
}

echo "== sanity: list models served by vLLM =="
curl -sS "${VLLM_BASE}/models" | python3 -m json.tool | head -40

echo
echo "== call the base model directly =="
post "base" '"What is the molecular formula of caffeine?"'

echo
echo "== call the chemistry adapter directly =="
post "chemistry" '"What is the molecular formula of caffeine?"'

echo
echo "== call the sql adapter directly =="
post "sql" '"Write SQL: top 5 customers by revenue in 2023."'
