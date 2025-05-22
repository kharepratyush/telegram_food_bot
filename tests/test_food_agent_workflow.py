import pytest
import pytest_asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.messages import ToolCall
from langchain_core.tools import tool
import inspect
import json  # Added for tool signature

import src.agents.food_agent  # For patching paths
from src.agents.food_agent import (
    extract_input,
    check_input,
    call_agent,
    summarize_history,
    send_response,
    GraphState,
    FOOD_PROMPT,
    update_prompt,
    RemoveMessage,
    setup_workflow,
    send_empty_warning,
)

# Unit tests

@pytest_asyncio.fixture
def mock_llm_chain_agent(mocker):
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock()
    mock_llm_selector_instance = MagicMock()
    mock_llm_selector_instance.bind_tools.return_value = mock_chain
    mocker.patch(
        "src.agents.food_agent.llm_selector", return_value=mock_llm_selector_instance
    )
    # Patch tools with a real function, not MagicMock
    def dummy_tool(query: str) -> str:
        """Dummy tool for testing."""
        return "dummy result"
    dummy_tool.__name__ = "dummy_tool"
    mocker.patch("src.agents.food_agent.tools", [dummy_tool])
    return mock_chain


@pytest.mark.asyncio
async def test_extract_input_prepends_system_prompt(mocker):
    mocked_prompt = "Test system prompt"
    mocker.patch("src.agents.food_agent.update_prompt", return_value=mocked_prompt)
    human_message_content = "I want to order a pizza."
    initial_state = GraphState(messages=[HumanMessage(content=human_message_content)], error=None, human_query=None, interim_response=None, tool_call_loop_tracker=None, initiating_human_query=None)
    updated_state = await extract_input(initial_state)
    assert isinstance(updated_state["messages"][0], SystemMessage)
    assert updated_state["messages"][0].content == mocked_prompt
    assert updated_state["human_query"] == human_message_content


@pytest.mark.asyncio
async def test_extract_input_no_human_message(mocker):
    mocked_prompt = "Test system prompt"
    mocker.patch("src.agents.food_agent.update_prompt", return_value=mocked_prompt)
    initial_state = GraphState(messages=[], error=None, human_query=None, interim_response=None, tool_call_loop_tracker=None, initiating_human_query=None)
    updated_state = await extract_input(initial_state)
    assert updated_state["human_query"] is None
    assert isinstance(updated_state["messages"][0], SystemMessage)
    assert updated_state["messages"][0].content == mocked_prompt
    initial_state_no_human = GraphState(messages=[SystemMessage(content="Some system message")], error=None, human_query=None, interim_response=None, tool_call_loop_tracker=None, initiating_human_query=None)
    updated_state_no_human = await extract_input(initial_state_no_human)
    assert updated_state_no_human["human_query"] is None
    assert isinstance(updated_state_no_human["messages"][0], SystemMessage)
    assert updated_state_no_human["messages"][0].content == mocked_prompt

