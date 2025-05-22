import pytest

# skip if aiosqlite not available
pytest.importorskip("aiosqlite")
from datetime import datetime

import src.agents.prompts as prompts_module
from src.agents.prompts import FOOD_PROMPT, update_prompt # Import specific names

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
        "json", # For the code block
        "Meal", # As a header in the JSON example
        "GI Value" # As a header in the JSON example
    ]

    for phrase in key_phrases:
        assert phrase in FOOD_PROMPT, f"Expected phrase '{phrase}' not found in FOOD_PROMPT."

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
    assert "Recent food log:" not in result # Explicitly check that log section is not added

@pytest.mark.asyncio
async def test_update_prompt_with_logs_formatting_and_multiple_entries(monkeypatch):
    # Using distinct timestamps to ensure order and formatting are tested for each
    ts1 = datetime(2023, 1, 1, 8, 0, 0).timestamp()  # Jan 1, 2023 08:00:00
    ts2 = datetime(2023, 1, 1, 13, 30, 0).timestamp() # Jan 1, 2023 13:30:00
    ts3 = datetime(2023, 1, 2, 19, 45, 0).timestamp() # Jan 2, 2023 19:45:00

    logs = [
        (ts1, "breakfast", "oatmeal with berries"),
        (ts2, "lunch", "lentil soup and a small apple"),
        (str(ts3), "dinner", "grilled chicken with quinoa and steamed vegetables") # Test with timestamp as string
    ]

    async def fake_get_recent(days):
        return logs

    monkeypatch.setattr(
        prompts_module,
        "get_recent_food_log",
        fake_get_recent,
    )
    base_prompt_content = "This is the base prompt."
    result = await prompts_module.update_prompt(base_prompt_content)

    assert "Recent food log:" in result
    assert "Dont repeat the dish in food log for future meals." in result
    
    # Verify the position of the "Dont repeat" message
    assert result.index("Dont repeat the dish in food log for future meals.") < result.index("Recent food log:")

    # Expected log entry strings
    # The format in prompts.py is: f"{datetime.fromtimestamp(float(t))} - {m}: {d}"
    expected_log_entry1 = f"{datetime.fromtimestamp(float(ts1))} - breakfast: oatmeal with berries"
    expected_log_entry2 = f"{datetime.fromtimestamp(float(ts2))} - lunch: lentil soup and a small apple"
    expected_log_entry3 = f"{datetime.fromtimestamp(float(ts3))} - dinner: grilled chicken with quinoa and steamed vegetables"
    
    assert expected_log_entry1 in result
    assert expected_log_entry2 in result
    assert expected_log_entry3 in result

    # Ensure they are on separate lines and correctly formatted as part of the log section
    log_section_start = result.find("Recent food log:") + len("Recent food log:")
    log_section = result[log_section_start:].strip()
    
    logged_lines = [line.strip() for line in log_section.split('\n') if line.strip()]
    
    assert len(logged_lines) == len(logs) # Check if all logs are present
    assert logged_lines[0] == expected_log_entry1
    assert logged_lines[1] == expected_log_entry2
    assert logged_lines[2] == expected_log_entry3
    
    # Ensure the base prompt is still at the beginning
    assert result.startswith(base_prompt_content)

@pytest.mark.asyncio
async def test_update_prompt_varied_log_content(monkeypatch):
    timestamp1 = datetime(2023, 3, 10, 9, 0).timestamp()
    timestamp2 = datetime(2023, 3, 10, 12, 0).timestamp()
    logs_varied = [
        (timestamp1, "Snack", "Almonds"),
        (str(timestamp2), "Post-Workout", "Protein Shake with Banana & Spinach!"),
    ]

    async def fake_get_recent_varied(days):
        return logs_varied

    monkeypatch.setattr(
        prompts_module,
        "get_recent_food_log",
        fake_get_recent_varied,
    )
    base = "base_for_varied_logs"
    result = await prompts_module.update_prompt(base)

    assert "Recent food log:" in result
    assert "Dont repeat the dish in food log for future meals." in result

    expected_log1 = f"{datetime.fromtimestamp(timestamp1)} - Snack: Almonds"
    expected_log2 = f"{datetime.fromtimestamp(timestamp2)} - Post-Workout: Protein Shake with Banana & Spinach!"
    
    assert expected_log1 in result
    assert expected_log2 in result

    log_section_start = result.find("Recent food log:") + len("Recent food log:")
    log_section = result[log_section_start:].strip()
    logged_lines = [line.strip() for line in log_section.split('\n') if line.strip()]

    assert len(logged_lines) == len(logs_varied)
    assert logged_lines[0] == expected_log1
    assert logged_lines[1] == expected_log2
    assert result.startswith(base)

# Renaming the old test to be more specific about its original scope,
# though the new test `test_update_prompt_with_logs_formatting_and_multiple_entries` largely supersedes it.
# Or, we can remove it if the new one covers all aspects.
# For now, I'll keep it and rename, but it might be redundant.
@pytest.mark.asyncio
async def test_update_prompt_with_logs_original_check(monkeypatch): # Renamed from test_update_prompt_with_logs
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
    assert "Dont repeat the dish in food log for future meals." in result # Added this check here too
    assert "breakfast" in result # Generic check, specific formatting tested above
    assert "salad" in result   # Generic check

    # Check formatting for at least one entry to ensure the spirit of original test
    expected_log_entry_eggs = f"{datetime.fromtimestamp(float(timestamp))} - breakfast: eggs"
    assert expected_log_entry_eggs in result
    assert result.startswith(base)
