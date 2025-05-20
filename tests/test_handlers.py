import pytest

# skip if python-telegram-bot or aiosqlite not available
pytest.importorskip("telegram")
pytest.importorskip("aiosqlite")

from types import SimpleNamespace

from src.telegram_bot.handlers.help_handler import help_command
from src.telegram_bot.handlers.save_handler import save
from src.telegram_bot.handlers.shopping_list_handler import (
    add_to_shopping_list,
    delete_shopping_list,
    retrieve_shopping_list,
)
from src.telegram_bot.handlers.start_handler import start


class DummyMessage:
    def __init__(self, text):
        self.text = text
        self.date = SimpleNamespace(timestamp=lambda: 0)
        self.replies = []

    async def reply_text(self, text):
        self.replies.append(text)
        return SimpleNamespace(chat_id=1, message_id=1)


class DummyUpdate:
    def __init__(self, text):
        self.message = DummyMessage(text)


@pytest.mark.asyncio
async def test_save_usage():
    update = DummyUpdate("/save")
    context = SimpleNamespace()
    await save(update, context)
    assert update.message.replies[-1] == "Usage: /save [meal] [dish name]"


@pytest.mark.asyncio
async def test_save_success(monkeypatch):
    called = {}

    async def fake_init_db(name, sql):
        called["init_db"] = True

    async def fake_insert(name, sql, params):
        called["insert"] = params

    monkeypatch.setattr("src.telegram_bot.handlers.save_handler.init_db", fake_init_db)
    monkeypatch.setattr("src.telegram_bot.handlers.save_handler.insert", fake_insert)
    update = DummyUpdate("/save meal dish")
    context = SimpleNamespace()
    await save(update, context)
    assert called.get("init_db")
    assert called.get("insert")
    assert update.message.replies[-1] == "Food Logged"


@pytest.mark.asyncio
async def test_add_to_shopping_list_usage():
    update = DummyUpdate("/add_to_shopping_list")
    context = SimpleNamespace()
    await add_to_shopping_list(update, context)
    assert update.message.replies[-1] == "Usage: /add_to_shopping_list [item]"


@pytest.mark.asyncio
async def test_add_to_shopping_list_success(monkeypatch):
    called = {}

    async def fake_init_db(name, sql):
        called["init_db"] = True

    async def fake_insert(name, sql, params):
        called["insert"] = params

    monkeypatch.setattr(
        "src.telegram_bot.handlers.shopping_list_handler.init_db", fake_init_db
    )
    monkeypatch.setattr(
        "src.telegram_bot.handlers.shopping_list_handler.insert", fake_insert
    )
    update = DummyUpdate("/add_to_shopping_list item")
    context = SimpleNamespace()
    await add_to_shopping_list(update, context)
    assert called.get("init_db")
    assert called.get("insert")
    assert update.message.replies[-1] == "Added to shopping list."


@pytest.mark.asyncio
async def test_retrieve_shopping_list_empty(monkeypatch):
    async def fake_fetch_all(name, sql):
        return []

    monkeypatch.setattr(
        "src.telegram_bot.handlers.shopping_list_handler.fetch_all", fake_fetch_all
    )
    update = DummyUpdate("/retrieve_shopping_list")
    context = SimpleNamespace()
    await retrieve_shopping_list(update, context)
    assert update.message.replies[-1] == "Shopping List is empty."


@pytest.mark.asyncio
async def test_retrieve_shopping_list_nonempty(monkeypatch):
    async def fake_fetch_all(name, sql):
        return [("apple",), ("banana",)]

    monkeypatch.setattr(
        "src.telegram_bot.handlers.shopping_list_handler.fetch_all", fake_fetch_all
    )
    sent = {}

    async def fake_send_markdown(update, text):
        sent["text"] = text

    monkeypatch.setattr(
        "src.telegram_bot.handlers.shopping_list_handler.send_markdown",
        fake_send_markdown,
    )
    update = DummyUpdate("/retrieve_shopping_list")
    context = SimpleNamespace()
    await retrieve_shopping_list(update, context)
    assert "**Shopping List:**" in sent["text"]
    assert "1. apple" in sent["text"]
    assert "2. banana" in sent["text"]


@pytest.mark.asyncio
async def test_delete_shopping_list(monkeypatch):
    called = {}

    async def fake_init_db(name, sql):
        called["init_db"] = True

    async def fake_execute_sql(name, sql, params):
        called["execute_sql"] = True

    monkeypatch.setattr(
        "src.telegram_bot.handlers.shopping_list_handler.init_db", fake_init_db
    )
    monkeypatch.setattr(
        "src.telegram_bot.handlers.shopping_list_handler.execute_sql", fake_execute_sql
    )
    update = DummyUpdate("/delete_shopping_list")
    context = SimpleNamespace()
    await delete_shopping_list(update, context)
    assert called.get("init_db")
    assert called.get("execute_sql")
    assert update.message.replies[-1] == "Shopping list deleted."


@pytest.mark.asyncio
async def test_help_command():
    update = DummyUpdate("/help")
    context = SimpleNamespace()
    await help_command(update, context)
    text = update.message.replies[-1]
    assert "Available commands:" in text
    assert "/start" in text


@pytest.mark.asyncio
async def test_start_command():
    update = DummyUpdate("/start")
    context = SimpleNamespace()
    await start(update, context)
    assert "Hi!" in update.message.replies[-1]
