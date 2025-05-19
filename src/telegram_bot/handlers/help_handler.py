

def help_command(update, context):
    commands = [
        "/start - Start the bot",
        "/help - Show this help message",
        "/save - Log your Food",
        "/add_to_shopping_list - Add items to your shopping list",
        "/retrieve_shopping_list - Retrieve your shopping list",
        "/delete_shopping_list - Delete your shopping list",
    ]
    help_text = "Available commands:\n" + "\n".join(commands)
    return update.message.reply_text(help_text)