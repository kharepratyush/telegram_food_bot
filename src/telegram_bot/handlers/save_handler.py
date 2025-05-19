from telegram import Update
from telegram.ext import ContextTypes
from src.telegram_bot.utils.db import init_db, insert

SAVE_TABLE_SQL = (
    "CREATE TABLE IF NOT EXISTS food_log ("
    "id INTEGER PRIMARY KEY AUTOINCREMENT, time TIMESTAMP, meal TEXT, dish_name TEXT)"
)

async def save(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    parts = update.message.text.split(maxsplit=2)
    if len(parts) < 3:
        await update.message.reply_text("Usage: /save [meal] [dish name]")
        return
    meal, dish = parts[1].lower(), parts[2].strip()
    await init_db("food_log.db", SAVE_TABLE_SQL)
    await insert("food_log.db",
                 "INSERT INTO food_log (time, meal, dish_name) VALUES (?, ?, ?)",
                 (update.message.date.timestamp(), meal, dish))
    await update.message.reply_text("Food Logged")
