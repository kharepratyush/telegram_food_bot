from telegram import Update
from telegram.ext import ContextTypes


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    commands = [
        "/start - Start the bot",
        "/help - Show this help message",
        "/save - Log your Food",
        "/add_to_shopping_list - Add items to your shopping list",
        "/retrieve_shopping_list - Retrieve your shopping list",
        "/delete_shopping_list - Delete your shopping list",
    ]
    help_text = "Available commands:\n" + "\n".join(commands)
    await update.message.reply_text(help_text)
