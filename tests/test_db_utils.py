import pytest

# skip if aiosqlite not available
pytest.importorskip("aiosqlite")

from src.telegram_bot.utils import db as db_utils


@pytest.mark.asyncio
async def test_init_insert_fetch(tmp_path, monkeypatch):
    # Override database directory to temporary path
    monkeypatch.setattr(db_utils, "DATABASE_DIR", str(tmp_path))
    # Reset initialized DBs tracking
    monkeypatch.setattr(db_utils, "_initialized_dbs", set())
    table_sql = "CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY, name TEXT)"
    # Initialize DB and insert a row
    await db_utils.init_db("test.db", table_sql)
    await db_utils.insert(
        "test.db", "INSERT INTO test (id, name) VALUES (?, ?)", (1, "Alice")
    )
    rows = await db_utils.fetch_all("test.db", "SELECT id, name FROM test")
    assert rows == [(1, "Alice")]
