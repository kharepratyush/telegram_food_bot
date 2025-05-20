#!/usr/bin/env bash
# Always run from the script's directory
cd "$(dirname "$0")"

# Run the Telegram bot script, activating the .venv if present
if [ -z "$VIRTUAL_ENV" ] && [ -f "./.venv/bin/activate" ]; then
  source "./.venv/bin/activate"
fi
exec python3 bot_v2.py "$@"
