#!/usr/bin/env bash
# Start the Ember backend cleanly.
#
# Frees port 5050 first (kills any stale/old instance) so you never hit
# "Address already in use", then launches the backend with the project venv.
#
# Usage:  ./run.sh      (run from anywhere; it cd's to its own folder)

cd "$(dirname "$0")" || exit 1

echo "Freeing port 5050 (stopping any stale backend)..."
lsof -ti tcp:5050 | xargs kill 2>/dev/null || true
sleep 1

echo "Starting Ember on http://127.0.0.1:5050"
echo "(Ctrl+C to stop)"
exec venv/bin/python rag/backend_server_production.py
