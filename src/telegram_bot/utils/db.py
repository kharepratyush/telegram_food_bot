import os
import aiosqlite
from src.telegram_bot.config import DATABASE_DIR


async def init_db(db_name: str, create_table_sql: str):
    os.makedirs(DATABASE_DIR, exist_ok=True)
    db_path = os.path.join(DATABASE_DIR, db_name)
    async with aiosqlite.connect(db_path) as db:
        await db.execute(create_table_sql)
        await db.commit()


async def insert(db_name: str, sql: str, params: tuple):
    db_path = os.path.join(DATABASE_DIR, db_name)
    async with aiosqlite.connect(db_path) as db:
        await db.execute(sql, params)
        await db.commit()


async def fetch_all(db_name: str, sql: str):
    db_path = os.path.join(DATABASE_DIR, db_name)
    async with aiosqlite.connect(db_path) as db:
        async with db.execute(sql) as cursor:
            return await cursor.fetchall()
