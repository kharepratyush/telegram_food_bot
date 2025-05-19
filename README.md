# Telegram Food Agent Bot

A Telegram bot for diabetic-friendly meal planning, food logging, and shopping list management, with a focus on North Indian cuisine for pregnant individuals managing diabetes.

## Features
- **Conversational meal planning** powered by LLMs (OpenAI or Ollama)
- **Food log**: Save and retrieve meals
- **Shopping list**: Add, retrieve, and delete items
- **Markdown and table rendering** for easy-to-read responses
- **Supervisor integration** for auto-restart and startup

## Setup

### 1. Clone the repository
```
git clone <repo-url>
cd TelegramBots
```

### 2. Create a virtual environment
```
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

### 4. Environment Variables
Create a `.env` file in the project root:
```
TELEGRAM_TOKEN=your_telegram_bot_token
LOG_LEVEL=INFO
```

### 5. Run the bot
```
python bot_v2.py
```

Or use Supervisor for auto-restart:
```
supervisord -c supervisord.conf
```

## Usage
- `/start` — Start the bot
- `/help` — Show help
- `/save [meal] [dish name]` — Log a meal
- `/add_to_shopping_list [item]` — Add item to shopping list
- `/retrieve_shopping_list` — Show shopping list
- `/delete_shopping_list` — Delete shopping list

Send any message to get meal recommendations and nutrition info.

## Project Structure
- `bot_v2.py` — Main entrypoint
- `src/agents/` — LLM agent logic
- `src/telegram_bot/` — Telegram handlers and utilities
- `data/` — SQLite databases
- `supervisord.conf` — Supervisor config for process management

## Security
- **Never commit your `.env` file or secrets.**
- The `.gitignore` is set up to exclude sensitive files and data.

## TODO

- **Add knowledge base for Indian meals**: Integrate a structured knowledge base or database for Indian meal recipes, nutrition, and cultural context. Should be modular and easy to extend.
- **GI API integration**: Integrate an external or custom API for Glycemic Index (GI) values. Design the integration so it can be easily swapped or extended.

## Testing

Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

Run the test suite:

```bash
pytest
```

Run static analysis (formatters, linters, type checks) via pre-commit:

```bash
pre-commit run --all-files
```

## License
MIT
