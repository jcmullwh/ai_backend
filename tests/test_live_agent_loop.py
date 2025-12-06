import pytest

from ai_backend import TextAI, ToolRegistry, run_agentic_chat

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


def test_agent_loop_live_round_trip() -> None:
    text_ai = TextAI()
    messages = [
        {
            "role": "system",
            "content": (
                "You can call an 'add' tool that sums two integers. Always call it instead of doing the math yourself."
            ),
        },
        {"role": "user", "content": "Use your tool to add 2 and 3."},
    ]
    tool_registry: ToolRegistry = {"add": add}
    initial_len = len(messages)

    result = run_agentic_chat(
        text_ai,
        messages,
        TOOLS,
        tool_registry,
        max_steps=3,
        temperature=0,
    )

    assert result.stop_reason == "done"
    assert result.final_content is not None
    assert "5" in result.final_content
    assert len(messages) > initial_len
