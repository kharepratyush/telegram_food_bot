import os

import aiosqlite

from src.telegram_bot.config import DATABASE_DIR

_initialized_dbs = set()


async def init_db(db_name: str, create_table_sql: str):
    os.makedirs(DATABASE_DIR, exist_ok=True)
    db_path = os.path.join(DATABASE_DIR, db_name)
    if db_name not in _initialized_dbs:
        async with aiosqlite.connect(db_path) as db:
            await db.execute(create_table_sql)
            await db.commit()
        _initialized_dbs.add(db_name)


async def execute_sql(db_name: str, sql: str, params: tuple):
    """Execute a SQL statement with the given parameters."""
    db_path = os.path.join(DATABASE_DIR, db_name)
    async with aiosqlite.connect(db_path) as db:
        await db.execute(sql, params)
        await db.commit()


async def insert(db_name: str, sql: str, params: tuple):
    """Execute an INSERT SQL statement."""
    return await execute_sql(db_name, sql, params)


async def fetch_all(db_name: str, sql: str):
    db_path = os.path.join(DATABASE_DIR, db_name)
    async with aiosqlite.connect(db_path) as db:
        async with db.execute(sql) as cursor:
            return await cursor.fetchall()
