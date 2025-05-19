#!/usr/bin/env bash
# Run the Telegram bot script, activating the .venv if present
if [ -z "$VIRTUAL_ENV" ] && [ -f "./.venv/bin/activate" ]; then
  source "./.venv/bin/activate"
fi
exec python bot_v2.py "$@"