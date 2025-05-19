import telegramify_markdown
from telegramify_markdown import customize

customize.strict_markdown = True
customize.cite_expandable = True


async def send_markdown(update, text: str):
    # split into 4000-char chunks
    while text:
        chunk, text = text[:4000], text[4000:]
        md = telegramify_markdown.markdownify(
            chunk, max_line_length=None, normalize_whitespace=False
        )
        await update.message.reply_text(md, parse_mode="MarkdownV2")


def split_json_response(data: str) -> tuple[str, str]:
    """
    Split the text into narrative and JSON block. Returns (narrative, json_text).

    If the data does not contain a JSON block marked by ```json, returns (data, "").
    """
    try:
        narrative, json_block = data.split("```json", 1)
        narrative = narrative.strip()
        json_text = json_block.strip().rstrip("```")
    except ValueError:
        narrative, json_text = data.strip(), ""
    return narrative, json_text
