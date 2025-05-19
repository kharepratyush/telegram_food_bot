import asyncio
import logging
import os
import pprint
import re
import uuid
from typing import Annotated, List, TypedDict

import aiosqlite
from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

# from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from src.agents.prompts import FOOD_PROMPT, update_prompt
from src.agents.tools import get_tools

# Load environment variables from .env file
load_dotenv()

# Configure root logger for informational output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Retrieve external tools for agent
tools = get_tools()

class GraphState(TypedDict):
    """
    TypedDict defining the agent's conversation state.

    Attributes:
        messages: List of AnyMessage entries tracking the dialogue history.
        error: Boolean flag for error state.
    """

    messages: Annotated[List[AnyMessage], add_messages]
    error: bool

# Instantiate the StateGraph workflow with initial and end markers
workflow = StateGraph(GraphState)


async def extract_input(state: GraphState) -> GraphState:
    """
    Ensure the conversation begins with a system prompt if none exists.

    Args:
        state: Current GraphState of the workflow.

    Returns:
        Updated GraphState with system prompt prepended if needed.
    """
    # If no messages or missing system prompt, prepend the base prompt
    if not state.get("messages") or not any(
        isinstance(m, SystemMessage) for m in state["messages"]
    ):
        UPDATED_PROMPT = await update_prompt(FOOD_PROMPT)
        initial = [SystemMessage(content=UPDATED_PROMPT)]
        initial.extend(state.get("messages", []))
        state["messages"] = initial

    return state


def check_input(state: GraphState) -> str:
    """
    Determine next workflow node based on last human input.

    Args:
        state: Current GraphState.

    Returns:
        "send_empty_warning" if input is empty, else "call_Agent".
    """
    # Find the most recent human message
    last_human = next(
        (m for m in reversed(state.get("messages", [])) if isinstance(m, HumanMessage)),
        None,
    )

    # Route to warning if no content, otherwise proceed
    if not last_human or not last_human.content.strip():
        return "send_empty_warning"
    return "call_Agent"


async def send_empty_warning(state: GraphState) -> GraphState:
    """
    Node: Handle empty input by sending a warning.

    Args:
        state: Workflow state before warning.

    Returns:
        State updated with error flag and warning message.
    """
    state["error"] = True
    state["messages"] = [SystemMessage(content="⚠️ Please provide a non-empty prompt.")]
    return state


def _clean_deepseek_response(response: str) -> str:
    """
    Clean model output by removing <think> blocks and extra whitespace.

    Args:
        response: Raw LLM response string.

    Returns:
        Cleaned string without think tags.
    """
    # Remove any <think>...</think> sections (including newlines)
    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    return cleaned.strip()


async def call_Agent(state: GraphState) -> GraphState:
    """
    Invoke the language model agent to generate a response.

    - Optionally summarize history if too long.
    - Handle errors gracefully and log issues.

    Args:
        state: Current GraphState containing message history.

    Returns:
        Updated GraphState including the model's reply or error.
    """
    # Summarize if history exceeds threshold
    if len(state.get("messages", [])) > 10:
        state = await summarize_history(state)

    # Initialize ChatOpenAI (or ChatOllama)
    llm = ChatOpenAI(model="gpt-4o").bind_tools(tools)
    # llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_URL)
    try:
        # Send the entire message history
        try:
            result: AnyMessage = await llm.ainvoke(state["messages"])
        except Exception:
            result: AnyMessage = await llm.ainvoke(state["messages"])

        # Clean unwanted think tags
        result.content = _clean_deepseek_response(result.content)
        state["messages"].append(result)
    except Exception:
        # On error, flag and notify user
        state["error"] = True
        logger.exception("Agent invocation failed")
        state["messages"].append(
            SystemMessage(
                content="⚠️ Agent encountered an error. Please try again later."
            )
        )

    return state


async def summarize_history(state: GraphState) -> GraphState:
    """
    Condense long dialogue history into a concise summary.

    Steps:
      1. Format messages into a summarization prompt.
      2. Call the LLM for summarization.
      3. Rebuild state with summary and recent context.

    Args:
        state: Current GraphState with full history.

    Returns:
        Updated GraphState with shortened history.
    """
    # Prepare summarization LLM
    llm = ChatOpenAI(model="gpt-4o")
    # llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_URL)

    # Build a summary prompt of all messages
    history_text = "\n\n".join(
        f"{type(m).__name__}: {m.content}" for m in state["messages"]
    )
    summary_prompt = SystemMessage(
        content=(
            "You are a helpful assistant that summarizes conversation history into bullet points. "
            "Here is the dialogue: \n\n" + history_text
        )
    )

    # Invoke summarization
    summary_msg = await llm.ainvoke([summary_prompt])
    summary = summary_msg.content.strip()

    # Rebuild message list: system prompt + recent + summary
    UPDATED_PROMPT = await update_prompt(FOOD_PROMPT)
    new_messages = [SystemMessage(content=UPDATED_PROMPT)]
    new_messages.extend(state["messages"][-3:])
    new_messages.append(SystemMessage(content="Conversation summary:\n" + summary))
    state["messages"] = new_messages
    return state


async def send_response(state: GraphState) -> GraphState:
    """
    Final node: Return the state as-is, completing the workflow.

    Args:
        state: GraphState after agent invocation.

    Returns:
        Unmodified state (end of workflow).
    """
    return state


# Workflow graph setup: nodes and transitions
workflow.add_node("extract_input", extract_input)
workflow.add_node("send_empty_warning", send_empty_warning)
workflow.add_node("call_Agent", call_Agent)
workflow.add_node("summarize_history", summarize_history)
workflow.add_node("send_response", send_response)
workflow.add_node("search_tools", ToolNode(tools))

# Start -> extract_input
workflow.add_edge(START, "extract_input")

# Conditional flow: empty input vs. agent call
workflow.add_conditional_edges(
    "extract_input",
    check_input,
    {"send_empty_warning": "send_empty_warning", "call_Agent": "call_Agent"},
)

# Warning -> end
workflow.add_edge("send_empty_warning", END)

# Agent call -> tool search or response
workflow.add_conditional_edges(
    "call_Agent", tools_condition, {"tools": "search_tools", END: "send_response"}
)
workflow.add_edge("search_tools", "call_Agent")
workflow.add_edge("send_response", END)

# Compile the top-level agent
food_agent = workflow.compile()


async def expose_agent() -> StateGraph:
    """
    Build an async agent with persistent SQLite memory for integration.

    Returns:
        Compiled agent ready for async invocations.
    """
    conn = await aiosqlite.connect("data/food_agent.db")
    memory = AsyncSqliteSaver(conn)
    agent = workflow.compile(checkpointer=memory)
    return agent


def run_agent() -> None:
    """
    Example entry point for running the agent synchronously.

    - Connects to SQLite database.
    - Compiles workflow with memory.
    - Invokes agent with a sample message.
    """

    async def main():
        # Database path relative to project root
        db_path = os.path.join(os.path.dirname(__file__), "../..", "data", "food_agent.db")
        conn = await aiosqlite.connect(db_path)
        memory = AsyncSqliteSaver(conn)
        agent = workflow.compile(checkpointer=memory)

        # Example invocation
        sample = [HumanMessage(content="meal plan for today")]
        result = await agent.ainvoke(
            {"messages": sample, "error": False},
            config=RunnableConfig({"thread_id": uuid.uuid4()}),
        )
        pprint.pprint(result)

    asyncio.run(main())


if __name__ == "__main__":
    run_agent()
