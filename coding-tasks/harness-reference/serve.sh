#!/usr/bin/env bash
# serve.sh start original|lora|x2 | stop | status  -- ONE llama-server for the edit experiment on :8110 (never two at once; the Mac has 48 GB and an OOM killed a session before).
# Same flags as the lab's serve.sh / lora_feasibility/variant.sh; only the GGUF differs between the arms, the alias (= the `model` field of every request) is the same.
#   original = Google QAT Q4_0 GGUF of Gemma 4 E4B                                              (~/models/gemma-4/gemma-4-E4B_q4_0-it.gguf)
#   lora     = the same GGUF with the rank-128 adapter file merged into per_layer_model_proj.weight   (merge_adapter_gguf.py; merged-weight proxy of the LoRA, see REPORT.md)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; PORT=8110
M="$HOME/models/gemma-4"
case "${2:-}" in
  original) GGUF="$M/gemma-4-E4B_q4_0-it.gguf";;
  lora)     GGUF="$M/gemma-4-E4B_q4_0-it-lora-r128-merged.gguf";;
  x2)       GGUF="$M/gemma-4-E4B_q4_0-it-plproj-x2.gguf";;      # supplementary control: the exact x2 tensor (the lab's patched GGUF), not the rank-128 approximation
esac
case "${1:-}" in
  start)
    for p in 8101 8102 $PORT; do lsof -nP -iTCP:$p -sTCP:LISTEN >/dev/null 2>&1 && { echo "a server is already listening on :$p; stop it first" >&2; exit 1; }; done
    [ -f "$GGUF" ] || { echo "missing $GGUF" >&2; exit 1; }
    mkdir -p "$HERE/logs"
    perl -e 'use POSIX; POSIX::setsid(); exec @ARGV or die "exec failed: $!"' llama-server -m "$GGUF" --alias gemma-4-e4b --host 127.0.0.1 --port $PORT -ngl 99 -c 32768 -np 1 \
      --flash-attn on --jinja --reasoning-budget 0 > "$HERE/logs/server_$2.log" 2>&1 < /dev/null &
    for i in $(seq 1 120); do curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; sleep 2; done
    curl -sf http://127.0.0.1:$PORT/health >/dev/null || { echo "server did not come up; see logs/server_$2.log" >&2; exit 1; }
    echo "$2 up on :$PORT, serving $(curl -s http://127.0.0.1:$PORT/props | python3 -c 'import sys,json; print(json.load(sys.stdin)["model_path"])')";;
  stop)
    pid=$(lsof -ti tcp:$PORT -sTCP:LISTEN 2>/dev/null | head -1 || true)
    [ -n "$pid" ] && { kill "$pid"; for i in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done; echo "stopped pid $pid"; } || echo "nothing listening on :$PORT";;
  status)
    curl -sf http://127.0.0.1:$PORT/props | python3 -c 'import sys,json; print("up:", json.load(sys.stdin)["model_path"])' || echo "down";;
  *) echo "usage: $0 start original|lora|x2 | stop | status" >&2; exit 2;;
esac
