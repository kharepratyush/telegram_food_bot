import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

async def table_image_with_colored_header(
    table: dict,
    header_color: str = "#4F81BD",
    header_text_color: str = "white",
    font_size: int = 18,
    dpi: int = 300,
    pad: float = 1.5,
    max_col_width: float = 3.5,
    max_row_chars: int = 20,
) -> BytesIO:
    """
    Render a table dict (with 'headers' and 'rows') to a PNG buffer,
    applying a colored header row for visual clarity.
    Tries to wrap long cell values and optimize for mobile readability.

    Args:
        table: Dict containing 'headers': list[str] and 'rows': list[list].
        header_color: Hex code or name of header background.
        header_text_color: Text color for header row.
        font_size: Base font size for table text.
        dpi: Resolution of output image.
        pad: Scaling factor for table cells.
        max_col_width: Maximum width per column (inches).
        max_row_chars: Max characters per cell before wrapping.

    Returns:
        BytesIO buffer containing the PNG image.
    """
    import textwrap

    def wrap_cell(val):
        if isinstance(val, str) and len(val) > max_row_chars:
            return "\n".join(textwrap.wrap(val, max_row_chars))
        return val

    headers = table["headers"]
    rows = [[wrap_cell(cell) for cell in row] for row in table["rows"]]
    df = pd.DataFrame(rows, columns=headers)

    # Compute proportional column widths based on max text lengths, capped for mobile
    max_lens = [
        min(max(df[col].astype(str).map(lambda x: len(str(x))).max(), len(str(col))), max_row_chars)
        for col in df.columns
    ]
    total_len = sum(max_lens)
    col_widths = [min(length / total_len * len(headers) * max_col_width, max_col_width) for length in max_lens]

    # For one-row tables, increase height for readability
    n_rows = len(df)
    fig_width = sum(col_widths)
    # Adjust fig_height to account for wrapped lines in cells
    max_lines_per_row = [
        max([str(cell).count("\n") + 1 for cell in row]) for row in df.values
    ] if len(df) > 0 else [1]
    avg_lines = sum(max_lines_per_row) / len(max_lines_per_row) if max_lines_per_row else 1
    fig_height = max(1.5, (n_rows + 1) * avg_lines * 0.7)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    ax.axis("off")

    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
        colWidths=[w / fig_width for w in col_widths],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(font_size)
    tbl.scale(pad, pad * avg_lines)  # scale y by avg_lines to avoid overlap

    # Style header row (row 0)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color=header_text_color, weight="bold")
        cell.set_linewidth(0.7)

    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf
