import pytest

# skip if aiosqlite not available
pytest.importorskip("aiosqlite")

from src.agents.food_agent import _clean_deepseek_response


def test_clean_deepseek_response_removes_think_tags():
    input_str = "Hello <think>secret</think> World"
    output = _clean_deepseek_response(input_str)
    assert "<think>" not in output
    assert "secret" not in output
    assert output.startswith("Hello") and output.endswith("World")
