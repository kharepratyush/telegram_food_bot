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
