"""Food Agent Workflow.

This module defines the asynchronous Food Agent workflow using StateGraph.
It includes:
- Input extraction node
- Input validation node
- Agent invocation node
- History summarization node
- Response sending node
- Workflow compilation and exposure functions
"""

import asyncio
import logging
import os
import re
import uuid
from typing import Annotated, List, Optional, TypedDict

import aiosqlite
from dotenv import load_dotenv
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from src.agents.llm import llm_selector
from src.agents.prompts import FOOD_PROMPT, update_prompt
from src.agents.tools import get_tools

# from langchain_ollama import ChatOllama


# Configuration
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DB_PATH = os.getenv(
    "FOOD_AGENT_DB_PATH",
    os.path.join(os.path.dirname(__file__), "../..", "data", "food_agent.db"),
)

# Initialize tools and workflow
tools = get_tools()


class GraphState(TypedDict):
    """State of the graph containing messages and error flag."""

    messages: Annotated[List[AnyMessage], add_messages]
    error: Optional[bool]
    human_query: Optional[str]
    interim_response: Optional[str]


def _clean_deepseek_response(response: str) -> str:
    """Remove <think> tags from the model response."""
    return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()


# === Workflow node functions ===
async def extract_input(state: GraphState) -> GraphState:
    """
    Ensure system control message and initial prompt are prepended before agent call.

    Args:
        state: Current GraphState with incoming messages.

    Returns:
        Updated GraphState with prepended control and prompt messages.
    """
    # print("-" * 50)
    # print("extract_input")
    # print(state)
    # print("-" * 50)

    last_user = next(
        (m for m in reversed(state.get("messages", [])) if isinstance(m, HumanMessage)),
        None,
    )
    human_query = last_user.content
    updated_prompt = await update_prompt(FOOD_PROMPT)
    messages = [SystemMessage(content=updated_prompt)]
    return {"messages": messages, "human_query": human_query}


def check_input(state: GraphState) -> str:
    """
    Validate that the last human message is non-empty.

    Args:
        state: Current GraphState.

    Returns:
        Transition node name based on validation.
    """
    # print("-" * 50)
    # print("check_input")
    # print(state)
    # print("-" * 50)

    human_query = state.get("human_query", "")
    if not human_query or not human_query.strip():
        return "send_empty_warning"
    if len(state.get("messages", [])) > 1:
        return "summarize_history"
    return "call_agent"


async def send_empty_warning(state: GraphState) -> GraphState:
    """
    Return an error state when user input is empty.

    Args:
        state: Current GraphState.

    Returns:
        New GraphState with error flag and warning message.
    """
    warning = SystemMessage(content="⚠️ Please provide a non-empty prompt.")
    return {"error": True, "messages": [warning]}


async def call_agent(state: GraphState) -> GraphState:
    """
    Invoke the LLM with tools, handling errors and summarizing history if needed.

    Steps:
      1. Summarize history if messages exceed limit.
      2. Bind tools to the LLM.
      3. Invoke LLM and clean response.
      4. Handle invocation errors gracefully.

    Args:
        state: Current GraphState with prepared messages.

    Returns:
        New GraphState with LLM result or error flag.
    """
    # print("-" * 50)
    # print("call_agent")
    # print(state)
    # print("Message Length", len(state.get("messages", [])))
    # print("-" * 50)

    llm = llm_selector().bind_tools(tools)
    try:
        try:
            prompt_text = (
                f"Only respond to the user's query: {state['human_query']}\n\n"
                f"Use Context only if required: \n{state["messages"]}\n\n"
            )
            result = await llm.ainvoke(prompt_text)

        except Exception:
            # Retry with 'tool' roles converted to 'function'
            for msg in state["messages"]:
                if getattr(msg, "role", None) == "tool":
                    msg.role = "function"

            prompt_text = (
                f"Only respond to the user's query: {state['human_query']}\n\n"
                f"Use Context only if required: \n{state["messages"]}\n\n"
            )
            result = await llm.ainvoke(prompt_text)

        result.content = _clean_deepseek_response(result.content)
        return {
            "messages": [result],
            "error": False,
            "interim_response": result.content,
        }
    except Exception:
        logger.exception("Agent invocation failed")
        error_msg = SystemMessage(
            content="⚠️ Agent encountered an error. Please try again later."
        )
        return {"error": True, "messages": [error_msg]}


async def summarize_history(state: GraphState) -> GraphState:
    """
    Summarize long conversation history into bullet points to reduce context size.

    Args:
        state: Current GraphState with full history.

    Returns:
        Updated GraphState with summary and recent messages.
    """
    # print("-" * 50)
    # print("call_agent")
    # print(state)
    # print("Message Length", len(state.get("messages", [])))
    # print("-" * 50)

    llm = llm_selector()
    messages = state.get("messages", [])

    history = "\n\n".join(
        f"{type(m).__name__}: {getattr(m, 'content', getattr(m, 'text', ''))}"
        for m in messages
    )

    if len(history.strip()) == 0:
        return state

    prompt = SystemMessage(
        content=(
            "You are a helpful assistant that summarizes conversation history into bullet points. "
            f"Here is the dialogue:\n\n{history}"
        )
    )
    summary_msg = await llm.ainvoke([prompt])

    to_delete_messages = [RemoveMessage(id=m.id) for m in state["messages"]]
    to_add_messages = [
        SystemMessage(content=f"Conversation summary:\n\n{summary_msg.content}"),
    ]

    summary_messages = to_delete_messages.copy()
    summary_messages.extend(to_add_messages)

    return {"messages": summary_messages}


async def send_response(state: GraphState) -> GraphState:
    """
    Generate the final response using the full context and updated prompt.

    Args:
        state: GraphState after agent invocation.

    Returns:
        New GraphState with final LLM response or error state.
    """
    # Convert any 'tool' role back to 'function'
    for msg in state.get("messages", []):
        if getattr(msg, "role", None) == "tool":
            msg.role = "function"

    intermediate_response = state["interim_response"]
    llm = llm_selector("gpt-4o")
    try:
        updated_prompt = await update_prompt(FOOD_PROMPT)
        prompt_text = (
            f"Only respond to the user's query: {state['human_query']}\n\n"
            f"Use Context only if required: \n{updated_prompt}\n\n"
            f"Please use intermediate response only if required:\n{intermediate_response}"
        )
        response = await llm.ainvoke([prompt_text])
        return {"messages": response, "error": False}
    except Exception:
        logger.exception("Failed to send final response")
        return state


def setup_workflow() -> StateGraph:
    """
    Configure nodes and transitions for the Food Agent workflow.

    Returns:
        Configured StateGraph instance.
    """
    # Create the workflow graph instance
    workflow = StateGraph(GraphState)
    workflow.add_node("extract_input", extract_input)
    workflow.add_node("send_empty_warning", send_empty_warning)
    workflow.add_node("call_agent", call_agent)
    workflow.add_node("summarize_history", summarize_history)
    workflow.add_node("send_response", send_response)
    workflow.add_node("search_tools", ToolNode(tools))

    workflow.add_edge(START, "extract_input")

    workflow.add_conditional_edges(
        "extract_input",
        check_input,
        {
            "send_empty_warning": "send_empty_warning",
            "call_agent": "call_agent",
            "summarize_history": "summarize_history",
        },
    )

    workflow.add_edge("send_empty_warning", END)
    workflow.add_conditional_edges(
        "call_agent",
        tools_condition,
        {"tools": "search_tools", END: "send_response"},
    )
    workflow.add_edge("search_tools", "call_agent")
    workflow.add_edge("summarize_history", "call_agent")
    workflow.add_edge("send_response", END)

    return workflow


food_agent = setup_workflow().compile()


async def expose_agent() -> StateGraph:
    """
    Build and return an async agent with persistent SQLite memory.

    Returns:
        Compiled StateGraph ready for async invocations.
    """
    conn = await aiosqlite.connect(DB_PATH)
    memory = AsyncSqliteSaver(conn)
    food_agent = setup_workflow().compile(checkpointer=memory)
    return food_agent


def run_agent() -> None:
    """
    Entry point for synchronous execution of the Food Agent.

    - Connects to SQLite database.
    - Compiles workflow with memory.
    - Invokes agent with a sample HumanMessage.
    """

    async def main():
        conn = await aiosqlite.connect(DB_PATH)
        memory = AsyncSqliteSaver(conn)
        agent = setup_workflow().compile(checkpointer=memory)
        sample = [HumanMessage(content="meal plan for tomorrow")]
        result = await agent.ainvoke(
            {"messages": sample, "error": False},
            config=RunnableConfig({"thread_id": uuid.uuid4()}),
        )
        print(result)

    asyncio.run(main())


if __name__ == "__main__":
    run_agent()
