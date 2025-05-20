import logging

from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters

from src.telegram_bot.config import LOG_LEVEL, TELEGRAM_TOKEN
from src.telegram_bot.handlers.help_handler import help_command
from src.telegram_bot.handlers.message_handler import handle_message
from src.telegram_bot.handlers.save_handler import save
from src.telegram_bot.handlers.shopping_list_handler import (
    add_to_shopping_list,
    delete_shopping_list,
    retrieve_shopping_list,
)
from src.telegram_bot.handlers.start_handler import start

logging.basicConfig(level=LOG_LEVEL)


def main() -> None:
    if not TELEGRAM_TOKEN:
        logging.error("Missing TELEGRAM_TOKEN environment variable")
        return
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("save", save))
    app.add_handler(CommandHandler("add_to_shopping_list", add_to_shopping_list))
    app.add_handler(CommandHandler("retrieve_shopping_list", retrieve_shopping_list))
    app.add_handler(CommandHandler("delete_shopping_list", delete_shopping_list))
    app.add_handler(CommandHandler("help", help_command))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    app.run_polling()


if __name__ == "__main__":
    main()
