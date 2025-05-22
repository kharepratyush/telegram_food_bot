import pytest
from unittest.mock import AsyncMock

# skip if aiosqlite not available
pytest.importorskip("aiosqlite")
from datetime import datetime

import src.agents.prompts as prompts_module
from src.agents.prompts import FOOD_PROMPT, update_prompt  # Import specific names


# Test for FOOD_PROMPT structure
def test_food_prompt_basic_structure():
    assert isinstance(FOOD_PROMPT, str)
    assert len(FOOD_PROMPT) > 0

    key_phrases = [
        "You are a knowledgeable food assistant",
        "North Indian cuisine",
        "pregnant individuals managing diabetes",
        "Always respond in Markdown",
        "no non-vegetarian options on Tuesdays and Thursdays",
        "Example Output:",
        "json",  # For the code block
        "Meal",  # As a header in the JSON example
        "GI Value",  # As a header in the JSON example
    ]

    for phrase in key_phrases:
        assert (
            phrase in FOOD_PROMPT
        ), f"Expected phrase '{phrase}' not found in FOOD_PROMPT."


@pytest.mark.asyncio
async def test_update_prompt_no_logs(monkeypatch):
    mock_get_recent = AsyncMock(return_value=[])
    # Patch at both the source and the usage location
    monkeypatch.setattr("src.agents.utils.db.get_recent_food_log", mock_get_recent)
    monkeypatch.setattr("src.agents.prompts.get_recent_food_log", mock_get_recent)
    base = "base_prompt"
    result = await prompts_module.update_prompt(base)
    assert base in result
    assert "Recent food log:" not in result
    assert "Dont repeat the dish in food log for future meals." not in result
    assert "(No recent food log entries)" not in result


@pytest.mark.asyncio
async def test_update_prompt_with_logs_formatting_and_multiple_entries(monkeypatch):
    ts1 = datetime(2023, 1, 1, 8, 0, 0).timestamp()
    ts2 = datetime(2023, 1, 1, 13, 30, 0).timestamp()
    ts3 = datetime(2023, 1, 2, 19, 45, 0).timestamp()
    logs = [
        (ts1, "breakfast", "oatmeal with berries"),
        (ts2, "lunch", "lentil soup and a small apple"),
        (str(ts3), "dinner", "grilled chicken with quinoa and steamed vegetables"),
    ]
    mock_get_recent = AsyncMock(return_value=logs)
    monkeypatch.setattr("src.agents.utils.db.get_recent_food_log", mock_get_recent)
    monkeypatch.setattr("src.agents.prompts.get_recent_food_log", mock_get_recent)
    base_prompt_content = "This is the base prompt."
    result = await prompts_module.update_prompt(base_prompt_content)

    # Check the structure
    assert "Recent food log:" in result
    assert "Dont repeat the dish in food log for future meals." in result
    assert base_prompt_content in result

    # Check the entries
    expected_log_entry1 = f"{datetime.fromtimestamp(float(ts1))} - breakfast: oatmeal with berries"
    expected_log_entry2 = f"{datetime.fromtimestamp(float(ts2))} - lunch: lentil soup and a small apple"
    expected_log_entry3 = f"{datetime.fromtimestamp(float(ts3))} - dinner: grilled chicken with quinoa and steamed vegetables"
    assert expected_log_entry1 in result
    assert expected_log_entry2 in result
    assert expected_log_entry3 in result


@pytest.mark.asyncio
async def test_update_prompt_varied_log_content(monkeypatch):
    timestamp1 = datetime(2023, 3, 10, 9, 0).timestamp()
    timestamp2 = datetime(2023, 3, 10, 12, 0).timestamp()
    logs_varied = [
        (timestamp1, "Snack", "Almonds"),
        (str(timestamp2), "Post-Workout", "Protein Shake with Banana & Spinach!"),
    ]
    mock_get_recent = AsyncMock(return_value=logs_varied)
    monkeypatch.setattr("src.agents.utils.db.get_recent_food_log", mock_get_recent)
    monkeypatch.setattr("src.agents.prompts.get_recent_food_log", mock_get_recent)
    base = "base_for_varied_logs"
    result = await prompts_module.update_prompt(base)

    assert "Recent food log:" in result
    assert "Dont repeat the dish in food log for future meals." in result
    assert base in result

    expected_log1 = f"{datetime.fromtimestamp(timestamp1)} - Snack: Almonds"
    expected_log2 = f"{datetime.fromtimestamp(timestamp2)} - Post-Workout: Protein Shake with Banana & Spinach!"
    assert expected_log1 in result
    assert expected_log2 in result


# Renaming the old test to be more specific about its original scope,
# though the new test `test_update_prompt_with_logs_formatting_and_multiple_entries` largely supersedes it.
# Or, we can remove it if the new one covers all aspects.
# For now, I'll keep it and rename, but it might be redundant.
@pytest.mark.asyncio
async def test_update_prompt_with_logs_original_check(monkeypatch):
    timestamp = datetime.now().timestamp()
    logs = [(timestamp, "breakfast", "eggs"), (timestamp, "lunch", "salad")]
    mock_get_recent = AsyncMock(return_value=logs)
    monkeypatch.setattr("src.agents.utils.db.get_recent_food_log", mock_get_recent)
    monkeypatch.setattr("src.agents.prompts.get_recent_food_log", mock_get_recent)
    base = "base"
    result = await prompts_module.update_prompt(base)
    assert "Recent food log:" in result
    assert "Dont repeat the dish in food log for future meals." in result
    assert "breakfast" in result
    assert "salad" in result
    expected_log_entry_eggs = f"{datetime.fromtimestamp(float(timestamp))} - breakfast: eggs"
    assert expected_log_entry_eggs in result
    assert result.startswith(base)
