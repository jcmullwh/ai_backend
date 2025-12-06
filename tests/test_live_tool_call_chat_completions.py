import json

import pytest

from ai_backend import TextAI

pytestmark = pytest.mark.live_api


def add(a: int, b: int) -> int:
    return a + b


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two integers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
            },
        },
    }
]


def _run_add_tool_from_call(tool_call) -> int:
    """Execute the local add function using arguments from the model's tool call."""
    assert tool_call.type == "function"
    assert tool_call.function.name == "add"

    args = json.loads(tool_call.function.arguments or "{}")
    return add(**args)


def test_chat_completions_returns_tool_call():
    text_ai = TextAI()

    messages = [
        {
            "role": "system",
            "content": (
                "You have access to a function called 'add' that adds two integers. "
                "Whenever the user asks you to add numbers, you MUST call this tool "
                "instead of calculating yourself."
            ),
        },
        {"role": "user", "content": "What is 2 + 3? Use your tool."},
    ]

    choice = text_ai.text_chat(
        messages,
        tools=TOOLS,
        tool_choice="auto",
        response_type="full",
        use_responses=False,
        temperature=0,
    )

    msg = choice.message
    assert msg is not None

    tool_calls = msg.tool_calls
    assert tool_calls is not None and len(tool_calls) == 1

    tc = tool_calls[0]
    assert tc.function.name == "add"

    args = json.loads(tc.function.arguments or "{}")
    assert args == {"a": 2, "b": 3}


def test_tool_round_trip_with_chat_completions():
    text_ai = TextAI()

    messages = [
        {
            "role": "system",
            "content": (
                "You have a function 'add(a, b)' that adds two integers. "
                "Always call it when asked to add numbers, then explain the result."
            ),
        },
        {"role": "user", "content": "Please use your tool to add 7 and 5."},
    ]

    first_choice = text_ai.text_chat(
        messages,
        tools=TOOLS,
        tool_choice="auto",
        response_type="full",
        use_responses=False,
        temperature=0,
    )

    first_msg = first_choice.message
    assert first_msg is not None
    assert first_msg.tool_calls is not None and len(first_msg.tool_calls) == 1

    tool_call = first_msg.tool_calls[0]

    messages.append(
        {
            "role": "assistant",
            "content": first_msg.content or "",
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
            ],
        }
    )

    result = _run_add_tool_from_call(tool_call)
    assert result == 12

    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": "add",
            "content": json.dumps({"result": result}),
        }
    )

    second_choice = text_ai.text_chat(
        messages,
        tools=TOOLS,
        tool_choice="auto",
        response_type="full",
        use_responses=False,
        temperature=0,
    )

    final_msg = second_choice.message
    assert final_msg is not None
    assert not final_msg.tool_calls

    final_text = (final_msg.content or "").lower()
    assert "12" in final_text
    assert "7" in final_text and "5" in final_text
