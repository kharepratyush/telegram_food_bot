import pytest
# skip if pandas or matplotlib not available
pytest.importorskip("pandas")
pytest.importorskip("matplotlib")

import matplotlib
matplotlib.use("Agg")
from io import BytesIO
from src.telegram_bot.utils.table_renderer import table_image_with_colored_header


@pytest.mark.asyncio
async def test_table_image_minimal_table():
    table = {"headers": ["A", "B"], "rows": [[1, "foo"]]}
    buf = await table_image_with_colored_header(
        table,
        header_color="#000000",
        header_text_color="#FFFFFF",
        font_size=10,
        dpi=50,
        pad=1,
        max_col_width=1,
        max_row_chars=10,
    )
    assert isinstance(buf, BytesIO)
    data = buf.getvalue()
    assert data