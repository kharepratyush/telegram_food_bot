"""Food Agent Workflow.

This module defines the asynchronous Food Agent workflow using LangGraph's StateGraph.
It orchestrates the interaction between user input, an LLM-based agent,
and various tools to provide food-related assistance.

The workflow includes the following key operations:
- Extracting and validating user input.
- Invoking a primary LLM agent equipped with tools (e.g., for searching recipes).
- Managing conversation history, including summarization for long dialogues.
- Refining the agent's response using a secondary LLM for a polished final output.
- Handling tool execution and potential errors.
- Persisting conversation state using SQLite.
"""

import asyncio
import logging
import os
import re
import uuid
import json  # Added
from typing import Annotated, List, Optional, TypedDict, Dict  # Added Dict

import aiosqlite
from dotenv import load_dotenv
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langchain_core.runnables import RunnableConfig
# from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver # Commented out for now
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from src.agents.llm import llm_selector
from src.agents.prompts import FOOD_PROMPT, update_prompt
from src.agents.tools import get_tools

# Load environment variables from .env file
load_dotenv()

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
# Define the default path for the SQLite database
DB_PATH = os.getenv(
    "FOOD_AGENT_DB_PATH",
    os.path.join(os.path.dirname(__file__), "../..", "data", "food_agent.db"),
)

# Initialize tools for the agent
tools = get_tools()


class GraphState(TypedDict):
    """Defines the state of the conversation graph.

    Attributes:
        messages: A list of messages in the conversation, managed by `add_messages`.
        error: An optional boolean flag indicating if an error occurred in a node.
        human_query: The latest query extracted from a HumanMessage.
        interim_response: The response from the primary agent, before final refinement.
    """

    messages: Annotated[List[AnyMessage], add_messages]
    error: Optional[bool]
    human_query: Optional[str]
    interim_response: Optional[str]
    tool_call_loop_tracker: Optional[Dict[str, int]] = None  # Added
    initiating_human_query: Optional[str] = None  # Added


def _clean_deepseek_response(response: str) -> str:
    """Remove <think>...</think> tags from the model response.

    These tags are sometimes used by models for chain-of-thought reasoning
    but should not be part of the final user-facing response.

    Args:
        response: The raw response string from the LLM.

    Returns:
        The cleaned response string.
    """
    return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()


# === Workflow node functions ===
async def extract_input(state: GraphState) -> GraphState:
    """Extracts the latest human query and prepares the initial system prompt.

    This node ensures that the primary system prompt (FOOD_PROMPT), potentially
    updated with recent food logs, is set as the first system message. It also
    extracts the content of the most recent HumanMessage to be used as 'human_query'.

    Args:
        state: The current GraphState, expected to contain messages from user input.

    Returns:
        An updated GraphState with the 'human_query' field populated and
        the 'messages' list reset to start with the system prompt.
    """
    last_user_message = next(
        (
            msg
            for msg in reversed(state.get("messages", []))
            if isinstance(msg, HumanMessage)
        ),
        None,
    )
    human_query_content = None 
    if last_user_message:
        human_query_content = last_user_message.content

    updated_system_prompt = await update_prompt(FOOD_PROMPT)
    initial_messages = [SystemMessage(content=updated_system_prompt)]
    return {"messages": initial_messages, "human_query": human_query_content}


def check_input(state: GraphState) -> str:
    """Validates the extracted human query and determines the next step.

    Routes to:
    - 'send_empty_warning' if the query is empty or only whitespace.
    - 'summarize_history' if the query is valid and message history is long.
    - 'call_agent' if the query is valid and history is short.

    Args:
        state: The current GraphState, expected to have 'human_query' populated.

    Returns:
        A string representing the name of the next node to transition to.
    """
    human_query = state.get("human_query", "") 
    if not human_query or not human_query.strip(): # type: ignore
        return "send_empty_warning"

    if (
        len(state.get("messages", [])) > 10
    ): 
        return "summarize_history"
    return "call_agent"


async def send_empty_warning(state: GraphState) -> GraphState:
    """Returns a GraphState indicating an error due to empty user input.

    Args:
        state: The current GraphState.

    Returns:
        A new GraphState with 'error' set to True and a warning message.
    """
    warning_message = SystemMessage(content="⚠️ Please provide a non-empty prompt.")
    return {"error": True, "messages": [warning_message]}


async def call_agent(state: GraphState) -> GraphState:
    """Invokes the primary LLM agent with the current query and context.

    This node is responsible for the main interaction with the LLM that is
    equipped with tools. It prepares the prompt, calls the LLM,
    cleans the response (e.g., removing <think> tags), and handles
    potential errors during invocation. The response from this LLM is
    considered an 'interim_response'.

    Args:
        state: The current GraphState, containing 'human_query' and 'messages'.
               'messages' here should include the system prompt and any summarized history.

    Returns:
        An updated GraphState with the agent's response (as an AIMessage) added
        to 'messages', 'interim_response' populated, and 'error' status set.
    """
    MAX_TOOL_ATTEMPTS = 2

    human_query_content = state.get("human_query") 
    previous_initiating_query = state.get("initiating_human_query")

    if previous_initiating_query != human_query_content:
        current_loop_tracker = {}
        initiating_human_query_for_state_update = human_query_content
    else:
        current_loop_tracker = state.get(
            "tool_call_loop_tracker", {}
        ).copy() 
        initiating_human_query_for_state_update = previous_initiating_query

    llm_with_tools = llm_selector().bind_tools(tools)
    current_messages = state.get("messages", [])
    messages_for_llm = current_messages + [
        HumanMessage(content=state.get("human_query", "")) 
    ]

    try:
        result = await llm_with_tools.ainvoke(messages_for_llm)
        result.content = _clean_deepseek_response(result.content)

        if result.tool_calls:
            processed_tool_calls = []
            loop_broken_this_turn = False
            for tc in result.tool_calls:
                tool_signature = f"{tc.name}|{json.dumps(tc.args, sort_keys=True)}"
                current_attempt_count = current_loop_tracker.get(tool_signature, 0) + 1
                current_loop_tracker[tool_signature] = (
                    current_attempt_count 
                )

                if current_attempt_count > MAX_TOOL_ATTEMPTS:
                    logger.warning(
                        f"Tool call {tool_signature} for query '{human_query_content}' "
                        f"attempted {current_attempt_count} times. Exceeds MAX_TOOL_ATTEMPTS of {MAX_TOOL_ATTEMPTS}. Suppressing."
                    )
                    loop_broken_this_turn = True
                else:
                    processed_tool_calls.append(tc)

            if loop_broken_this_turn and not processed_tool_calls:
                result.content = (
                    "I seem to be having trouble using my tools effectively for your request. "
                    "Could you please try rephrasing or ask something different?"
                )
                result.tool_calls = [] 
            elif loop_broken_this_turn and processed_tool_calls:
                result.tool_calls = processed_tool_calls
        
        return_state = {
            "messages": [result],
            "error": False,
            "interim_response": result.content,
            "tool_call_loop_tracker": current_loop_tracker,
            "initiating_human_query": initiating_human_query_for_state_update,
        }
        return return_state

    except Exception as e: # Capture the exception object
        logger.exception("Agent invocation failed in call_agent.")
        # Ensure the test's required substring is present and include exception details
        error_message = SystemMessage(
            content=f"There was an error in the LLM call. Details: {str(e)}" # Modified content
        )
        return {
            "error": True,
            "messages": [error_message],
            "interim_response": None,
            "tool_call_loop_tracker": current_loop_tracker, 
            "initiating_human_query": initiating_human_query_for_state_update, 
        }


async def summarize_history(state: GraphState) -> GraphState:
    """Summarizes long conversation history to manage context length.

    If the conversation history becomes too long, this node uses an LLM
    to create a concise summary. The original messages are then replaced
    with `RemoveMessage` objects, and the summary (as a SystemMessage)
    is added. This helps keep the context provided to the main agent manageable.

    Args:
        state: The current GraphState with the full message history.

    Returns:
        An updated GraphState where long history is replaced by a summary.
        If history is short or empty, returns the original state.
    """
    summarizer_llm = llm_selector()
    current_messages = state.get("messages", [])

    if not current_messages:
        return state  

    history_string = "\n\n".join(
        f"{type(msg).__name__}: {getattr(msg, 'content', getattr(msg, 'text', ''))}"
        for msg in current_messages
    )

    if not history_string.strip():
        return state  

    summarization_prompt = SystemMessage(
        content=(
            "You are a helpful assistant that summarizes conversation history "
            "into concise bullet points. "
            f"Here is the dialogue:\n\n{history_string}"
        )
    )

    try:
        summary_ai_message = await summarizer_llm.ainvoke([summarization_prompt])
        messages_to_delete = [RemoveMessage(id=msg.id) for msg in current_messages]
        summary_system_message = SystemMessage(
            content=f"Conversation summary:\n\n{summary_ai_message.content}"
        )
        return {"messages": messages_to_delete + [summary_system_message]}
    except Exception:
        logger.exception("History summarization failed.")
        return state


async def send_response(state: GraphState) -> GraphState:
    """Refines the interim response using a secondary LLM for a polished final answer.

    This node takes the 'interim_response' (typically from `call_agent`) and
    the 'human_query', along with the main system prompt (FOOD_PROMPT),
    and uses a different LLM (specified as 'gpt-4o') to generate the
    final, user-facing response. This acts as a refinement or review step.

    Args:
        state: The current GraphState, expected to have 'human_query' and
               'interim_response' populated.

    Returns:
        A new GraphState with the final refined AIMessage in 'messages'.
        If an error occurs, it returns the original state.
    """
    interim_response_content = state.get("interim_response", "")
    human_query_content = state.get("human_query", "") 

    refinement_llm = llm_selector(openai_model="gpt-4o")
    try:
        updated_system_prompt_str = await update_prompt(FOOD_PROMPT)
        prompt_for_refinement = (
            f"Only respond to the user's query: {human_query_content}\n\n"
            f"Use Context only if required: \n{updated_system_prompt_str}\n\n"
            f"Please use intermediate response only if required:\n{interim_response_content}"
        )
        final_ai_response = await refinement_llm.ainvoke(prompt_for_refinement)
        return {"messages": [final_ai_response], "error": False}
    except Exception:
        logger.exception("Failed to generate final response in send_response.")
        return state


def setup_workflow() -> StateGraph:
    """Configures and returns the StateGraph for the Food Agent.

    This function defines the nodes of the graph and the edges (transitions)
    between them, including conditional edges based on the output of `check_input`
    and `tools_condition`.

    Returns:
        A compiled StateGraph instance ready for execution.
    """
    workflow_graph = StateGraph(GraphState)

    workflow_graph.add_node("extract_input", extract_input)
    workflow_graph.add_node("send_empty_warning", send_empty_warning)
    workflow_graph.add_node("call_agent", call_agent)
    workflow_graph.add_node("summarize_history", summarize_history)
    workflow_graph.add_node("send_response", send_response)
    workflow_graph.add_node("search_tools", ToolNode(tools))  

    workflow_graph.add_edge(START, "extract_input")

    workflow_graph.add_conditional_edges(
        "extract_input",
        check_input,
        {
            "send_empty_warning": "send_empty_warning", 
            "call_agent": "call_agent",  
            "summarize_history": "summarize_history", 
        },
    )

    workflow_graph.add_edge("send_empty_warning", END)  

    workflow_graph.add_conditional_edges(
        "call_agent",
        tools_condition,  
        {
            "tools": "search_tools",  
            END: "send_response",  
        },
    )
    workflow_graph.add_edge(
        "search_tools", "call_agent"
    )  
    workflow_graph.add_edge(
        "summarize_history", "call_agent"
    )  
    workflow_graph.add_edge("send_response", END)  

    return workflow_graph


food_agent = setup_workflow().compile()

# (Commented out section for expose_agent and run_agent remains unchanged)
# async def expose_agent() -> StateGraph:
# ...
# def run_agent() -> None:
# ...
# if __name__ == "__main__":
#     run_agent()
