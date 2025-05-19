from telegram import Update
from telegram.ext import ContextTypes
import json
import logging

from src.telegram_bot.utils.telegram_utils import send_markdown
from src.telegram_bot.utils.table_renderer import table_image_with_colored_header
from src.agents.food_agent import expose_agent

logger = logging.getLogger(__name__)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_input = update.message.text.strip()
    sent = await update.message.reply_text("🤖 Thinking...")
    if not user_input:
        await update.message.reply_text("⚠️ Please send a non-empty message.")
        return
    try:
        agent = await expose_agent()
        response = await agent.ainvoke(
            {"messages": user_input, "error": False},
            config={"thread_id": update.message.chat.id},
        )
        data = response["messages"][-1].content or "🤖 (empty response)"

        # print(data)
        narrative, json_text = (
            data.split("```json", 1) if "```json" in data else (data, "")
        )

        await context.bot.deleteMessage(
            chat_id=sent.chat_id, message_id=sent.message_id
        )

        await send_markdown(update, narrative)
        if json_text:
            try:
                table = json.loads(json_text.strip().rstrip("```"))
                buf = await table_image_with_colored_header(table)
                await update.message.reply_photo(
                    photo=buf, caption="Here's the information in tabular format"
                )
            except:
                pass
    except Exception as e:
        logger.error("Agent error", exc_info=e)
        await update.message.reply_text(
            "⚠️ Oops, couldn’t reach Agent. Please check the server."
        )
