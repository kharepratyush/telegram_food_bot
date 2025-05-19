import pytest
# skip if aiosqlite not available
pytest.importorskip("aiosqlite")
from datetime import datetime

import src.agents.prompts as prompts_module


@pytest.mark.asyncio
async def test_update_prompt_no_logs(monkeypatch):
    monkeypatch.setattr(
        prompts_module,
        "get_recent_food_log",
        lambda *args, **kwargs: [],
    )
    base = "base_prompt"
    result = await prompts_module.update_prompt(base)
    assert result == base


@pytest.mark.asyncio
async def test_update_prompt_with_logs(monkeypatch):
    timestamp = datetime.now().timestamp()
    logs = [(timestamp, "breakfast", "eggs"), (timestamp, "lunch", "salad")]

    async def fake_get_recent(days):
        return logs

    monkeypatch.setattr(
        prompts_module,
        "get_recent_food_log",
        fake_get_recent,
    )
    base = "base"
    result = await prompts_module.update_prompt(base)
    assert "Recent food log:" in result
    assert "breakfast" in result
    assert "salad" in result