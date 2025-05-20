import pytest

# skip if telegramify_markdown not available
pytest.importorskip("telegramify_markdown")

from src.telegram_bot.utils import telegram_utils


class DummyMessage:
    def __init__(self):
        self.replies = []

    async def reply_text(self, text, parse_mode=None):
        self.replies.append((text, parse_mode))


class DummyUpdate:
    def __init__(self):
        self.message = DummyMessage()


def test_split_json_response_no_block():
    narrative, json_text = telegram_utils.split_json_response("Hello world")
    assert narrative == "Hello world"
    assert json_text == ""


def test_split_json_response_with_block():
    text = 'Some narrative\n```json\n{"a":1}\n```'
    narrative, json_text = telegram_utils.split_json_response(text)
    assert narrative == "Some narrative"
    assert json_text == '{"a":1}'


@pytest.mark.asyncio
async def test_send_markdown_splits_and_formats(monkeypatch):
    # prepare a message longer than 4000 chars
    long_text = "x" * 5000
    update = DummyUpdate()
    # monkeypatch markdownify to identity function
    monkeypatch.setattr(
        telegram_utils.telegramify_markdown,
        "markdownify",
        lambda s, max_line_length, normalize_whitespace: s,
    )
    await telegram_utils.send_markdown(update, long_text)
    # two chunks: 4000 and 1000
    assert len(update.message.replies) == 2
    assert update.message.replies[0][0] == "x" * 4000
    assert update.message.replies[1][0] == "x" * 1000
    assert update.message.replies[0][1] == "MarkdownV2"
