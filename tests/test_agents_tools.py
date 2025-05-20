import pytest

# skip if agents.tools dependencies are not available
try:
    pass
except ImportError:
    pytest.skip(
        "Skipping agents.tools tests due to missing dependencies",
        allow_module_level=True,
    )

import re

from src.agents.tools import get_future_date, get_today_date, get_tomorrow_date


def test_get_today_date_format():
    result = get_today_date()
    assert result.startswith("Today is ")
    assert re.match(r"Today is \d{4}-\d{2}-\d{2}, \w+", result)


def test_get_tomorrow_date_format():
    result = get_tomorrow_date()
    assert result.startswith("Tomorrow is ")
    assert re.match(r"Tomorrow is \d{4}-\d{2}-\d{2}, \w+", result)


def test_get_future_date_offset_one():
    result = get_future_date(1)
    assert result.startswith("Tomorrow is ")


def test_get_future_date_offset_multiple():
    offset = 5
    result = get_future_date(offset)
    assert result.startswith(f"In {offset} days it will be ")
