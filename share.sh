#!/usr/bin/env bash
# Launch Ember AND put it on a public link so other people can use it.
#
#   1. starts the Ember backend (frees the port first)
#   2. waits until it's ready
#   3. opens a public tunnel and prints a shareable URL
#
# Requirements: Ollama running (ollama serve) with llama3.1:8b pulled,
# and either cloudflared or ngrok installed.
#
# Usage:  ./share.sh      (keep this terminal open; Ctrl+C stops everything)

cd "$(dirname "$0")" || exit 1

cleanup() { echo; echo "Stopping Ember..."; kill "$BPID" 2>/dev/null; lsof -ti tcp:5050 | xargs kill 2>/dev/null; exit 0; }
trap cleanup INT TERM

echo "Freeing port 5050..."
lsof -ti tcp:5050 | xargs kill 2>/dev/null; sleep 1

echo "Starting Ember backend..."
venv/bin/python rag/backend_server_production.py > /tmp/ember_backend.log 2>&1 &
BPID=$!

echo "Loading models / index (first run can take 30-60s)..."
until curl -s -o /dev/null http://127.0.0.1:5050/health 2>/dev/null; do
  if ! kill -0 "$BPID" 2>/dev/null; then echo "Backend failed to start. See /tmp/ember_backend.log"; exit 1; fi
  sleep 2
done
echo "Ember is running locally at http://127.0.0.1:5050"
echo
echo "Opening a public link. Share the https URL it prints below."
echo "(Anyone with the link can use it while this stays running.)"
echo "------------------------------------------------------------"

if command -v cloudflared >/dev/null 2>&1; then
  cloudflared tunnel --url http://localhost:5050
elif command -v ngrok >/dev/null 2>&1; then
  ngrok http 5050
else
  echo "No tunnel tool found. Install one:  brew install cloudflared"
  echo "Meanwhile the app works locally at http://127.0.0.1:5050"
  wait "$BPID"
fi

cleanup
