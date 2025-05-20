from telegram import Update
from telegram.ext import ContextTypes

from src.telegram_bot.utils.db import execute_sql, fetch_all, init_db, insert
from src.telegram_bot.utils.telegram_utils import send_markdown

SHOP_TABLE_SQL = (
    "CREATE TABLE IF NOT EXISTS shopping_list ("
    "id INTEGER PRIMARY KEY AUTOINCREMENT, time TIMESTAMP, item_name TEXT)"
)


async def add_to_shopping_list(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    parts = update.message.text.split(maxsplit=1)
    if len(parts) < 2:
        await update.message.reply_text("Usage: /add_to_shopping_list [item]")
        return
    item = parts[1].strip().lower()
    await init_db("shopping_list.db", SHOP_TABLE_SQL)
    await insert(
        "shopping_list.db",
        "INSERT INTO shopping_list (time, item_name) VALUES (?, ?)",
        (update.message.date.timestamp(), item),
    )
    await update.message.reply_text("Added to shopping list.")


async def retrieve_shopping_list(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    rows = await fetch_all(
        "shopping_list.db",
        "SELECT distinct item_name FROM shopping_list ORDER BY time DESC",
    )
    if not rows:
        await update.message.reply_text("Shopping List is empty.")
        return
    text = "**Shopping List:**\n" + "\n".join(
        f"{i+1}. {r[0]}" for i, r in enumerate(rows)
    )
    await send_markdown(update, text)


async def delete_shopping_list(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    await init_db("shopping_list.db", SHOP_TABLE_SQL)
    await execute_sql("shopping_list.db", "DELETE FROM shopping_list", ())
    await update.message.reply_text("Shopping list deleted.")
