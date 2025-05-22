import os
from datetime import datetime, timedelta
import logging # Added
import aiosqlite

# from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__) # Added

async def get_recent_food_log(days: int = 3):
    """
    Fetch food log entries from the last `days` days.
    Returns a list of (time, meal, dish_name) tuples.
    Returns an empty list if a database error occurs.
    """
    db_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "data", "food_log.db"
    )
    since = (datetime.now() - timedelta(days=days)).timestamp()
    # print(since)
    try: # Added try block
        async with aiosqlite.connect(db_path) as db:
            async with db.execute(
                "SELECT time, meal, dish_name FROM food_log WHERE time >= ? ORDER BY time DESC",
                (since,),
            ) as cursor:
                rows = await cursor.fetchall()
        return rows
    except aiosqlite.Error as e: # Added except block
        logger.error(f"Database error fetching food log: {e!s}")
        return []
