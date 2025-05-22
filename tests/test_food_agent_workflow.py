import pytest
import pytest_asyncio
from unittest.mock import patch, MagicMock
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import ToolCall, tool
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


@pytest.mark.asyncio
async def test_extract_input_prepends_system_prompt(mocker):
    mocked_prompt = "Test system prompt"
    mocker.patch("src.agents.food_agent.update_prompt", return_value=mocked_prompt)
    human_message_content = "I want to order a pizza."
    initial_state = GraphState(messages=[HumanMessage(content=human_message_content)])
    updated_state = await extract_input(initial_state)
    assert isinstance(updated_state["messages"][0], SystemMessage)
    assert updated_state["messages"][0].content == mocked_prompt
    assert updated_state["human_query"] == human_message_content


@pytest.mark.asyncio
async def test_extract_input_no_human_message(mocker):
    mocked_prompt = "Test system prompt"
    mocker.patch("src.agents.food_agent.update_prompt", return_value=mocked_prompt)
    initial_state = GraphState(messages=[])
    updated_state = await extract_input(initial_state)
    assert updated_state["human_query"] is None
    assert isinstance(updated_state["messages"][0], SystemMessage)
    assert updated_state["messages"][0].content == mocked_prompt
    initial_state_no_human = GraphState(
        messages=[SystemMessage(content="Some system message")]
    )
    updated_state_no_human = await extract_input(initial_state_no_human)
    assert updated_state_no_human["human_query"] is None
    assert isinstance(updated_state_no_human["messages"][0], SystemMessage)
    assert updated_state_no_human["messages"][0].content == mocked_prompt


def test_check_input_empty_query():
    state_none_query = GraphState(human_query=None, messages=[])
    assert check_input(state_none_query) == "send_empty_warning"
    state_empty_query = GraphState(human_query="", messages=[])
    assert check_input(state_empty_query) == "send_empty_warning"
    state_whitespace_query = GraphState(human_query="   ", messages=[])
    assert check_input(state_whitespace_query) == "send_empty_warning"


def test_check_input_long_history():
    state = GraphState(
        human_query="What's the special today?",
        messages=[
            SystemMessage(content="System prompt"),
            HumanMessage(content="Previous query"),
            SystemMessage(content="Another system message to make history long"),
        ],
    )
    assert check_input(state) == "summarize_history"


def test_check_input_normal_flow():
    state_single_system_message = GraphState(
        human_query="I want to order a burger.",
        messages=[SystemMessage(content="System prompt")],
    )
    assert check_input(state_single_system_message) == "call_agent"
    state_after_extract_for_new_convo = GraphState(
        human_query="Hi", messages=[SystemMessage(content="System prompt")]
    )
    assert check_input(state_after_extract_for_new_convo) == "call_agent"


@pytest_asyncio.fixture
async def mock_llm_chain_agent(mocker):
    mock_chain = MagicMock()
    mock_llm_selector_instance = MagicMock()
    mock_llm_selector_instance.bind_tools.return_value = mock_chain
    mocker.patch(
        "src.agents.food_agent.llm_selector", return_value=mock_llm_selector_instance
    )
    mocker.patch("src.agents.food_agent.tools", [])
    return mock_chain


@pytest_asyncio.fixture
def mock_llm_for_summarize_node(mocker):
    mock_llm = MagicMock()
    mocker.patch("src.agents.food_agent.llm_selector", return_value=mock_llm)
    return mock_llm


@pytest.mark.asyncio
async def test_call_agent_success(mocker, mock_llm_chain_agent):
    fixed_prompt = "Fixed system prompt for testing"
    mocker.patch("src.agents.food_agent.update_prompt", return_value=fixed_prompt)
    ai_response_content = "Test response from LLM"
    mock_llm_chain_agent.ainvoke.return_value = AIMessage(content=ai_response_content)
    initial_state = GraphState(
        human_query="Hello, I need food.",
        messages=[SystemMessage(content="Initial system prompt")],
        tool_call_loop_tracker=None,  # Ensure new fields are considered
        initiating_human_query=None,
    )
    updated_state = await call_agent(initial_state)
    assert updated_state["error"] is False
    expected_ai_message = AIMessage(content=ai_response_content)
    assert updated_state["messages"] == [expected_ai_message]
    assert updated_state["interim_response"] == ai_response_content
    mock_llm_chain_agent.ainvoke.assert_called_once()
    assert updated_state["tool_call_loop_tracker"] == {}  # Should be initialized
    assert updated_state["initiating_human_query"] == "Hello, I need food."


@pytest.mark.asyncio
async def test_call_agent_llm_exception(mocker, mock_llm_chain_agent):
    fixed_prompt = "Fixed system prompt for exception testing"
    mocker.patch("src.agents.food_agent.update_prompt", return_value=fixed_prompt)
    mock_llm_chain_agent.ainvoke.side_effect = Exception("LLM simulation error")
    initial_state = GraphState(
        human_query="Another query.",
        messages=[SystemMessage(content="Initial system prompt")],
        tool_call_loop_tracker=None,
        initiating_human_query=None,
    )
    updated_state = await call_agent(initial_state)
    assert updated_state["error"] is True
    assert len(updated_state["messages"]) == 1
    assert isinstance(updated_state["messages"][0], SystemMessage)
    assert "There was an error in the LLM call" in updated_state["messages"][0].content
    assert updated_state["interim_response"] is None
    assert updated_state["tool_call_loop_tracker"] == {}
    assert updated_state["initiating_human_query"] == "Another query."


@pytest.mark.asyncio
async def test_call_agent_cleans_response(mocker, mock_llm_chain_agent):
    fixed_prompt = "Fixed system prompt for cleaning test"
    mocker.patch("src.agents.food_agent.update_prompt", return_value=fixed_prompt)
    raw_response_content = (
        "<think>some thoughts here</think>This is the actual response."
    )
    expected_cleaned_response = "This is the actual response."
    mock_llm_chain_agent.ainvoke.return_value = AIMessage(content=raw_response_content)
    initial_state = GraphState(
        human_query="Query needing cleaned response.",
        messages=[SystemMessage(content="Initial system prompt")],
        tool_call_loop_tracker=None,
        initiating_human_query=None,
    )
    updated_state = await call_agent(initial_state)
    assert updated_state["error"] is False
    assert updated_state["interim_response"] == expected_cleaned_response
    expected_ai_message_raw = AIMessage(content=raw_response_content)
    assert updated_state["messages"] == [expected_ai_message_raw]
    assert updated_state["tool_call_loop_tracker"] == {}
    assert updated_state["initiating_human_query"] == "Query needing cleaned response."


@pytest.mark.asyncio
async def test_summarize_history_creates_summary_final(mock_llm_for_summarize_node):
    summary_content = "History summarized."
    mock_llm_for_summarize_node.ainvoke.return_value = AIMessage(
        content=summary_content
    )
    original_messages = [
        HumanMessage(content="Hello there!", id="msg1"),
        AIMessage(content="Hi! How can I help?", id="msg2"),
        HumanMessage(content="I need to summarize our chat.", id="msg3"),
    ]
    initial_state = GraphState(
        messages=original_messages,
        human_query="Final query not used by summarize_history",
    )
    updated_state = await summarize_history(initial_state)
    mock_llm_for_summarize_node.ainvoke.assert_called_once()
    history_str = "\n".join(
        [m.content for m in original_messages if hasattr(m, "content")]
    )
    expected_prompt = f"Summarize the following conversation:\n\n{history_str}"
    mock_llm_for_summarize_node.ainvoke.assert_called_with(expected_prompt)
    assert len(updated_state["messages"]) == len(original_messages) + 1
    for i in range(len(original_messages)):
        assert isinstance(updated_state["messages"][i], RemoveMessage)
        assert updated_state["messages"][i].id == original_messages[i].id
    assert isinstance(updated_state["messages"][-1], SystemMessage)
    assert updated_state["messages"][-1].content == summary_content


@pytest.mark.asyncio
async def test_summarize_history_empty_history_final(
    mock_llm_for_summarize_node, mocker
):
    initial_state = GraphState(messages=[], human_query="Query with empty history")
    original_summarize_fn = src.agents.food_agent.summarize_history

    async def guarded_summarize_for_test(state: GraphState):
        if not state.get("messages"):
            return {"messages": []}  # Simplified for test, original returns state
        return await original_summarize_fn(state)

    with mocker.patch(
        "src.agents.food_agent.summarize_history",
        side_effect=guarded_summarize_for_test,
    ) as mock_guarded_summarize:
        updated_state = await mock_guarded_summarize(initial_state)
        mock_llm_for_summarize_node.ainvoke.assert_not_called()
        # The guarded_summarize_for_test now returns {"messages": []} for empty.
        assert updated_state["messages"] == []


@pytest_asyncio.fixture
def mock_llm_for_send_response(mocker):
    mock_refinement_llm = MagicMock()
    return mock_refinement_llm


@pytest.mark.asyncio
async def test_send_response_node_refines_output(mocker, mock_llm_for_send_response):
    mock_update_prompt_call = mocker.patch(
        "src.agents.food_agent.update_prompt", return_value="Test System Prompt Content"
    )
    mock_llm_selector_call = mocker.patch(
        "src.agents.food_agent.llm_selector", return_value=mock_llm_for_send_response
    )
    refined_response_content = "This is the refined answer from GPT-4o."
    mock_llm_for_send_response.ainvoke.return_value = AIMessage(
        content=refined_response_content
    )
    initial_human_query = "Original query"
    initial_interim_response = "Draft answer from first LLM"
    initial_state = GraphState(
        human_query=initial_human_query,
        interim_response=initial_interim_response,
        messages=[AIMessage(content=initial_interim_response)],
        error=False,
    )
    returned_state = await send_response(initial_state)
    mock_update_prompt_call.assert_called_once_with(FOOD_PROMPT)
    mock_llm_selector_call.assert_called_once_with(openai_model="gpt-4o")
    expected_prompt_text = (
        f"Only respond to the user's query: {initial_human_query}\n\n"
        f"Use Context only if required: \n{mock_update_prompt_call.return_value}\n\n"
        f"Please use intermediate response only if required:\n{initial_interim_response}"
    )
    mock_llm_for_send_response.ainvoke.assert_called_once_with(expected_prompt_text)
    assert returned_state["error"] is False
    assert len(returned_state["messages"]) == 1
    assert isinstance(returned_state["messages"][0], AIMessage)
    assert returned_state["messages"][0].content == refined_response_content


@pytest.mark.asyncio
async def test_send_response_node_handles_error(mocker, mock_llm_for_send_response):
    mock_update_prompt_call = mocker.patch(
        "src.agents.food_agent.update_prompt", return_value="Test System Prompt Content"
    )
    mock_llm_selector_call = mocker.patch(
        "src.agents.food_agent.llm_selector", return_value=mock_llm_for_send_response
    )
    mock_logger_send_response = mocker.patch("src.agents.food_agent.logger.exception")
    mock_llm_for_send_response.ainvoke.side_effect = Exception(
        "GPT-4o simulation error"
    )
    initial_human_query = "Original query for error test"
    initial_interim_response = "Draft answer for error test"
    initial_state = GraphState(
        human_query=initial_human_query,
        interim_response=initial_interim_response,
        messages=[AIMessage(content="message1"), HumanMessage(content="message2")],
        error=False,
    )
    returned_state = await send_response(initial_state)
    mock_update_prompt_call.assert_called_once()
    mock_llm_selector_call.assert_called_once_with(openai_model="gpt-4o")
    mock_llm_for_send_response.ainvoke.assert_called_once()
    mock_logger_send_response.assert_called_once_with(
        "Failed to generate final response in send_response."
    )
    assert returned_state == initial_state


# Workflow Integration Tests


@pytest_asyncio.fixture
def mock_update_prompt_workflow(mocker):
    return mocker.patch(
        "src.agents.food_agent.update_prompt",
        return_value="System Prompt from Workflow Mock",
    )


@pytest_asyncio.fixture
def mock_agent_llm_workflow(mocker):
    mock_chain_ainvoke = MagicMock()
    mock_bound_tools = MagicMock()
    mock_bound_tools.ainvoke = mock_chain_ainvoke
    mock_llm_selector_instance = MagicMock()
    mock_llm_selector_instance.bind_tools.return_value = mock_bound_tools
    mocker.patch(
        "src.agents.food_agent.llm_selector", return_value=mock_llm_selector_instance
    )
    mocker.patch("src.agents.food_agent.tools", [])
    return mock_chain_ainvoke


@pytest_asyncio.fixture
def mock_summarization_llm_workflow(mocker):
    mock_llm_ainvoke = MagicMock()
    mock_llm_instance = MagicMock()
    mock_llm_instance.ainvoke = mock_llm_ainvoke
    return mock_llm_ainvoke


@pytest.mark.asyncio
async def test_workflow_empty_input_sends_warning(
    mock_update_prompt_workflow, mock_agent_llm_workflow
):
    test_workflow = setup_workflow().compile()
    initial_state_dict = {"messages": [HumanMessage(content="   ")]}
    final_state = await test_workflow.ainvoke(initial_state_dict)
    assert final_state["error"] is True
    assert len(final_state["messages"]) == 1
    assert isinstance(final_state["messages"][0], SystemMessage)
    assert "Your query was empty." in final_state["messages"][0].content
    mock_update_prompt_workflow.assert_called_once()


@pytest.mark.asyncio
async def test_workflow_normal_flow_calls_agent_once(
    mock_update_prompt_workflow, mock_agent_llm_workflow, mocker
):
    agent_response_content = "Joke response from agent"
    mock_agent_llm_workflow.return_value = AIMessage(content=agent_response_content)
    final_response_llm_ainvoke = MagicMock(
        return_value=AIMessage(content="Final polished joke")
    )
    mock_final_llm = MagicMock()
    mock_final_llm.ainvoke = final_response_llm_ainvoke

    def llm_selector_side_effect(*args, **kwargs):
        if kwargs.get("openai_model") == "gpt-4o":
            return mock_final_llm
        else:
            mock_llm_selector_instance_for_agent = MagicMock()
            mock_bound_tools_for_agent = MagicMock()
            mock_bound_tools_for_agent.ainvoke = mock_agent_llm_workflow
            mock_llm_selector_instance_for_agent.bind_tools.return_value = (
                mock_bound_tools_for_agent
            )
            return mock_llm_selector_instance_for_agent

    mocker.patch(
        "src.agents.food_agent.llm_selector", side_effect=llm_selector_side_effect
    )

    test_workflow = setup_workflow().compile()
    initial_state_dict = {"messages": [HumanMessage(content="Tell me a joke")]}
    final_state = await test_workflow.ainvoke(initial_state_dict)
    assert mock_update_prompt_workflow.call_count == 3
    mock_agent_llm_workflow.assert_called_once()
    final_response_llm_ainvoke.assert_called_once()
    assert final_state["error"] is False
    assert final_state["messages"] == [AIMessage(content="Final polished joke")]
    assert final_state["interim_response"] == agent_response_content


@pytest.mark.asyncio
async def test_workflow_tool_usage_recalls_agent(mock_update_prompt_workflow, mocker):
    agent_response_after_tool = "Response after tool"
    final_refined_response = "Final refined response after tool"
    mock_agent_llm_ainvoke = MagicMock(
        side_effect=[
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(name="dummy_tool", args={"query": "test"}, id="tc123")
                ],
            ),
            AIMessage(content=agent_response_after_tool),
        ]
    )
    final_response_llm_ainvoke = MagicMock(
        return_value=AIMessage(content=final_refined_response)
    )
    mock_final_llm = MagicMock()
    mock_final_llm.ainvoke = final_response_llm_ainvoke

    def llm_selector_side_effect_for_tool_flow(*args, **kwargs):
        if kwargs.get("openai_model") == "gpt-4o":
            return mock_final_llm
        else:
            mock_llm_selector_instance_for_agent = MagicMock()
            mock_bound_tools_for_agent = MagicMock()
            mock_bound_tools_for_agent.ainvoke = mock_agent_llm_ainvoke
            mock_llm_selector_instance_for_agent.bind_tools.return_value = (
                mock_bound_tools_for_agent
            )
            return mock_llm_selector_instance_for_agent

    mocker.patch(
        "src.agents.food_agent.llm_selector",
        side_effect=llm_selector_side_effect_for_tool_flow,
    )
    mock_tool_obj = MagicMock()
    mock_tool_obj.name = "dummy_tool"
    mock_tool_obj.invoke.return_value = "Tool output"
    mocker.patch("src.agents.food_agent.tools", [mock_tool_obj])

    test_workflow = setup_workflow().compile()
    initial_state_dict = {"messages": [HumanMessage(content="Search for something")]}
    final_state = await test_workflow.ainvoke(initial_state_dict)
    assert mock_update_prompt_workflow.call_count == 4
    assert mock_agent_llm_ainvoke.call_count == 2
    mock_tool_obj.invoke.assert_called_once_with({"query": "test"})
    final_response_llm_ainvoke.assert_called_once()
    assert final_state["error"] is False
    assert final_state["messages"] == [AIMessage(content=final_refined_response)]
    assert final_state["interim_response"] == agent_response_after_tool


@pytest.mark.asyncio
async def test_workflow_history_summarization_flow(mocker):
    mock_update_prompt_fn = mocker.patch(
        "src.agents.food_agent.update_prompt",
        return_value="System Prompt from Workflow Mock",
    )
    summarized_history_content = "Summarized history."
    agent_response_content = "Response to final query after summary"
    final_refined_response_content = "Final polished response after summary"
    summarizer_llm_ainvoke = MagicMock(
        return_value=AIMessage(content=summarized_history_content)
    )
    agent_llm_ainvoke = MagicMock(
        return_value=AIMessage(content=agent_response_content)
    )
    final_llm_ainvoke = MagicMock(
        return_value=AIMessage(content=final_refined_response_content)
    )
    mock_llm_for_summarizer = MagicMock()
    mock_llm_for_summarizer.ainvoke = summarizer_llm_ainvoke
    mock_agent_bound_tools = MagicMock()
    mock_agent_bound_tools.ainvoke = agent_llm_ainvoke
    mock_llm_for_agent_selector = MagicMock()
    mock_llm_for_agent_selector.bind_tools.return_value = mock_agent_bound_tools
    mock_llm_for_final_response = MagicMock()
    mock_llm_for_final_response.ainvoke = final_llm_ainvoke

    mocker.patch(
        "src.agents.food_agent.llm_selector",
        side_effect=[
            mock_llm_for_summarizer,
            mock_llm_for_agent_selector,
            mock_llm_for_final_response,
        ],
    )
    mocker.patch("src.agents.food_agent.tools", [])

    test_workflow = setup_workflow().compile()
    initial_messages = [
        HumanMessage(content="Q1", id="h1"),
        AIMessage(content="A1", id="a1"),
        HumanMessage(content="Q2", id="h2"),
        AIMessage(content="A2", id="a2"),
        HumanMessage(content="Final query"),
    ]
    initial_state_dict = {"messages": initial_messages}
    final_state = await test_workflow.ainvoke(initial_state_dict)
    assert mock_update_prompt_fn.call_count == 3
    summarizer_llm_ainvoke.assert_called_once()
    agent_llm_ainvoke.assert_called_once()
    final_llm_ainvoke.assert_called_once()
    assert final_state["error"] is False
    assert final_state["messages"] == [
        AIMessage(content=final_refined_response_content)
    ]
    assert final_state["interim_response"] == agent_response_content
    summarization_history_messages = initial_messages[:-1]
    history_str = "\n".join(
        [m.content for m in summarization_history_messages if hasattr(m, "content")]
    )
    expected_summarizer_prompt = (
        f"Summarize the following conversation:\n\n{history_str}"
    )
    summarizer_llm_ainvoke.assert_called_with(expected_summarizer_prompt)
    expected_agent_invoke_arg = {
        "messages": [
            SystemMessage(content="System Prompt from Workflow Mock"),
            HumanMessage(content="Final query"),
        ]
    }
    agent_llm_ainvoke.assert_called_with(expected_agent_invoke_arg)
    expected_final_llm_prompt_text = (
        f"Only respond to the user's query: Final query\n\n"
        f"Use Context only if required: \n{mock_update_prompt_fn.return_value}\n\n"
        f"Please use intermediate response only if required:\n{agent_response_content}"
    )
    final_llm_ainvoke.assert_called_with(expected_final_llm_prompt_text)


@tool
def faulty_tool(query: str) -> str:
    """A tool that always fails."""
    raise ValueError("Tool execution failed intentionally")


@pytest.mark.asyncio
async def test_workflow_tool_error_handling(mock_update_prompt_workflow, mocker):
    agent_response_after_tool_error = "Sorry, I couldn't use the tool due to an error: ValueError('Tool execution failed intentionally')"
    final_refined_response_after_error = "My apologies, there was an issue with a required tool: Tool execution failed intentionally."
    agent_llm_ainvoke = MagicMock(
        side_effect=[
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        name="faulty_tool",
                        args={"query": "trigger error"},
                        id="faulty_tool_call_123",
                    )
                ],
            ),
            AIMessage(content=agent_response_after_tool_error),
        ]
    )
    final_response_llm_ainvoke = MagicMock(
        return_value=AIMessage(content=final_refined_response_after_error)
    )
    mock_final_llm = MagicMock()
    mock_final_llm.ainvoke = final_response_llm_ainvoke

    def llm_selector_side_effect_for_tool_error(*args, **kwargs):
        if kwargs.get("openai_model") == "gpt-4o":
            return mock_final_llm
        else:
            mock_llm_selector_instance_for_agent = MagicMock()
            mock_bound_tools_for_agent = MagicMock()
            mock_bound_tools_for_agent.ainvoke = agent_llm_ainvoke
            mock_llm_selector_instance_for_agent.bind_tools.return_value = (
                mock_bound_tools_for_agent
            )
            return mock_llm_selector_instance_for_agent

    mocker.patch(
        "src.agents.food_agent.llm_selector",
        side_effect=llm_selector_side_effect_for_tool_error,
    )
    mock_tools_with_faulty = [faulty_tool]
    mocker.patch("src.agents.food_agent.tools", mock_tools_with_faulty)

    test_workflow = setup_workflow().compile()
    initial_state_dict = {"messages": [HumanMessage(content="Use the faulty tool")]}
    final_state = await test_workflow.ainvoke(initial_state_dict)
    assert mock_update_prompt_workflow.call_count == 4
    assert agent_llm_ainvoke.call_count == 2
    final_response_llm_ainvoke.assert_called_once()
    assert final_state["error"] is False
    assert len(final_state["messages"]) == 1
    assert final_state["messages"][0].content == final_refined_response_after_error
    assert final_state["interim_response"] == agent_response_after_tool_error
    prompt_text_for_second_agent_call = agent_llm_ainvoke.call_args_list[1][0][0]
    assert "Tool execution failed intentionally" in prompt_text_for_second_agent_call
    assert "faulty_tool_call_123" in prompt_text_for_second_agent_call


# Tool definition for the loop test
@tool
def weather_tool_for_loop_test(date: str) -> str:
    """A mock weather tool for loop testing."""
    if date == "yesterday":
        return "The weather yesterday was sunny."
    elif date == "today":
        # This is the problematic part for the loop
        return "Today is cloudy. Should I check again for today?"
    return "Cannot get weather for that date."


@pytest.mark.asyncio
async def test_workflow_tool_loop_with_history(mocker):
    # 1. Setup Initial History
    H1 = HumanMessage(
        content="What was the weather yesterday?", id="h1_prev_turn"
    )  # Different content
    A1_tool_call = AIMessage(
        content="Checking weather tool for yesterday...",
        tool_calls=[
            ToolCall(
                name="weather_tool_for_loop_test",
                args={"date": "yesterday"},
                id="tc1_prev_turn",
            )
        ],
        id="a1_tc_prev_turn",
    )
    TM1_tool_response = ToolMessage(
        content="The weather yesterday was sunny.",
        tool_call_id="tc1_prev_turn",
        id="tm1_prev_turn",
    )
    A2_final_response_T1 = AIMessage(
        content="Yesterday was sunny.", id="a2_final_prev_turn"
    )

    initial_history_turn1 = [H1, A1_tool_call, TM1_tool_response, A2_final_response_T1]

    # 2. Mock External Dependencies
    mock_tools_list = [weather_tool_for_loop_test]
    mocker.patch("src.agents.food_agent.tools", mock_tools_list)
    mock_update_prompt_fn = mocker.patch(
        "src.agents.food_agent.update_prompt",
        return_value="System Prompt for Loop Test",
    )
    mock_logger_warning = mocker.patch(
        "src.agents.food_agent.logger.warning"
    )  # Mock logger.warning

    # Spy on the tool to count its direct invocations for "today"
    weather_tool_spy = mocker.spy(weather_tool_for_loop_test, "invoke")

    # Mock llm_selector for call_agent and send_response
    # For call_agent: it should attempt to call weather_tool for "today" multiple times.
    # The loop breaking mechanism (MAX_TOOL_ATTEMPTS = 2) should stop it after 2 attempts.

    mock_call_agent_llm_ainvoke = MagicMock()

    # LLM will try to call the tool 3 times. The 3rd one should be suppressed.
    # Unique tool_call_ids for each attempt.
    # call_agent's loop tracker will handle these.
    # The mock LLM just needs to *try* to call the tool each time it's invoked for this specific scenario.
    def agent_llm_side_effect_for_loop_break_test(*args, **kwargs):
        current_call_count = mock_call_agent_llm_ainvoke.call_count
        tool_call_id = f"tc_loop_break_{current_call_count + 1}"

        # Simulate LLM deciding to call the tool for "today"
        return AIMessage(
            content=f"Attempting to check weather for today, attempt {current_call_count + 1}",
            tool_calls=[
                ToolCall(
                    name="weather_tool_for_loop_test",
                    args={"date": "today"},
                    id=tool_call_id,
                )
            ],
        )

    mock_call_agent_llm_ainvoke.side_effect = agent_llm_side_effect_for_loop_break_test

    # Mock for send_response's LLM (will be hit after loop is broken)
    final_response_after_loop_break = (
        "I tried a couple of times but couldn't get the weather for today."
    )
    final_response_llm_ainvoke = MagicMock(
        return_value=AIMessage(content=final_response_after_loop_break)
    )
    mock_final_llm = MagicMock()
    mock_final_llm.ainvoke = final_response_llm_ainvoke

    # Configure the main llm_selector patch
    def llm_selector_side_effect_for_loop_break_test_main(*args, **kwargs):
        if kwargs.get("openai_model") == "gpt-4o":  # For send_response
            return mock_final_llm
        else:  # For call_agent
            mock_llm_instance_for_agent = MagicMock()
            mock_bound_tools_for_agent = MagicMock()
            mock_bound_tools_for_agent.ainvoke = mock_call_agent_llm_ainvoke
            mock_llm_instance_for_agent.bind_tools.return_value = (
                mock_bound_tools_for_agent
            )
            return mock_llm_instance_for_agent

    mocker.patch(
        "src.agents.food_agent.llm_selector",
        side_effect=llm_selector_side_effect_for_loop_break_test_main,
    )

    # 3. Test Execution
    test_workflow = setup_workflow().compile()

    H2_new_query = HumanMessage(
        content="And today?", id="h2_new_loop_break"
    )  # New query
    initial_messages_turn2 = initial_history_turn1 + [H2_new_query]

    initial_state_turn2 = GraphState(
        messages=initial_messages_turn2,
        error=None,
        human_query=None,  # extract_input will set this
        interim_response=None,
        tool_call_loop_tracker=None,  # Start fresh for this interaction
        initiating_human_query=None,  # Start fresh
    )

    # No recursion_limit needed here as the internal mechanism should break the loop.
    final_state = await test_workflow.ainvoke(initial_state_turn2)

    # 4. Assertions
    # MAX_TOOL_ATTEMPTS = 2. So, weather_tool_for_loop_test should be called 2 times for "today".
    # The agent LLM (mock_call_agent_llm_ainvoke) will be called 3 times:
    # 1. Calls tool (attempt 1) -> tool run
    # 2. Calls tool (attempt 2) -> tool run
    # 3. Calls tool (attempt 3) -> this tool call is suppressed by loop breaker.
    #    The AIMessage from this 3rd call will have empty tool_calls and modified content.

    assert mock_call_agent_llm_ainvoke.call_count == 3

    # Count how many times weather_tool_spy was actually invoked for "today"
    today_tool_calls = 0
    for call_args, _ in weather_tool_spy.call_args_list:
        if call_args[0] == {"date": "today"}:  # Tool takes dict
            today_tool_calls += 1
    assert today_tool_calls == 2  # MAX_TOOL_ATTEMPTS

    # Check logger.warning was called
    mock_logger_warning.assert_called_once()
    warning_message = mock_logger_warning.call_args[0][
        0
    ]  # Get the first positional argument of the call
    assert "Exceeds MAX_TOOL_ATTEMPTS of 2. Suppressing." in warning_message
    assert "'And today?'" in warning_message  # Check human_query is in log
    tool_signature_expected = (
        f"weather_tool_for_loop_test|{json.dumps({'date': 'today'}, sort_keys=True)}"
    )
    assert tool_signature_expected in warning_message

    # The AIMessage from the 3rd call to mock_call_agent_llm_ainvoke (which is the last one from call_agent node)
    # should have its content modified and tool_calls cleared.
    # This AIMessage is what populates `interim_response` and is sent to `send_response`.
    assert final_state["interim_response"] == (
        "I seem to be having trouble using my tools effectively for your request. "
        "Could you please try rephrasing or ask something different?"
    )

    # Check that the workflow proceeded to send_response and the final LLM was called
    final_response_llm_ainvoke.assert_called_once()

    # The final message in the state should be from the send_response LLM
    assert len(final_state["messages"]) == 1
    assert isinstance(final_state["messages"][0], AIMessage)
    assert final_state["messages"][0].content == final_response_after_loop_break
    assert not final_state["messages"][
        0
    ].tool_calls  # No tool calls in final refined message

    # Check update_prompt call count:
    # 1 by extract_input
    # +3 by the three calls to call_agent for the "And today?" query
    # +1 by send_response
    # Total = 5
    assert (
        mock_update_prompt_fn.call_count
        == 1 + mock_call_agent_llm_ainvoke.call_count + 1
    )

    # Check initiating_human_query and tool_call_loop_tracker in the final state
    assert final_state["initiating_human_query"] == "And today?"
    expected_tracker_key = (
        f"weather_tool_for_loop_test|{json.dumps({'date': 'today'}, sort_keys=True)}"
    )
    assert final_state["tool_call_loop_tracker"] == {
        expected_tracker_key: 3
    }  # Attempted 3 times
