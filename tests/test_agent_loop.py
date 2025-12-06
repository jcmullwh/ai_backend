from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import pytest

from ai_backend import ToolRegistry, run_agentic_chat


@dataclass
class FakeFunction:
    name: str
    arguments: Optional[str]


@dataclass
class FakeToolCall:
    id: str
    type: str
    function: FakeFunction


@dataclass
class FakeMessage:
    content: Optional[str]
    tool_calls: Optional[List[FakeToolCall]] = None


@dataclass
class FakeChoice:
    message: FakeMessage


class FakeTextAI:
    def __init__(
        self,
        responses: Optional[List[FakeMessage]] = None,
        *,
        response_fn: Optional[Callable[[List[Dict[str, Any]]], FakeMessage]] = None,
    ) -> None:
        self._responses = responses or []
        self._response_fn = response_fn
        self.calls = 0

    def text_chat(self, messages: List[Dict[str, Any]], **kwargs: Dict[str, Any]) -> FakeChoice:
        if self._response_fn is not None:
            msg = self._response_fn(messages)
        else:
            if self.calls >= len(self._responses):
                error_message = "No more fake responses configured"
                raise AssertionError(error_message)
            msg = self._responses[self.calls]

        self.calls += 1
        return FakeChoice(message=msg)


def make_tool_call(call_id: str, name: str, arguments: str) -> FakeToolCall:
    return FakeToolCall(id=call_id, type="function", function=FakeFunction(name=name, arguments=arguments))


def add(a: int, b: int) -> int:
    return a + b


def multiply(a: int, b: int) -> int:
    return a * b


def test_single_tool_call_then_final_answer() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "add",
                "parameters": {"type": "object", "properties": {"a": {}, "b": {}}, "required": ["a", "b"]},
            },
        }
    ]
    registry: ToolRegistry = {"add": add}

    first = FakeMessage(content="", tool_calls=[make_tool_call("call-1", "add", '{"a": 2, "b": 3}')])
    second = FakeMessage(content="The result is 5.", tool_calls=None)
    text_ai = FakeTextAI(responses=[first, second])

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": "Use tools"},
        {"role": "user", "content": "Add 2 and 3"},
    ]
    start_len = len(messages)

    result = run_agentic_chat(text_ai, messages, tools, registry, temperature=0)

    assert result.stop_reason == "done"
    assert result.final_content == "The result is 5."
    assert result.steps == 2

    tool_call_msg = messages[start_len]
    tool_result_msg = messages[start_len + 1]
    final_msg = messages[start_len + 2]

    assert tool_call_msg["tool_calls"][0]["function"]["name"] == "add"
    assert json.loads(tool_call_msg["tool_calls"][0]["function"]["arguments"]) == {"a": 2, "b": 3}
    assert json.loads(tool_result_msg["content"]) == 5
    assert final_msg["content"] == "The result is 5."


def test_multiple_tool_calls_in_one_step() -> None:
    tools = [
        {
            "type": "function",
            "function": {"name": "add", "parameters": {"type": "object", "properties": {}, "required": []}},
        },
        {
            "type": "function",
            "function": {"name": "multiply", "parameters": {"type": "object", "properties": {}, "required": []}},
        },
    ]
    registry: ToolRegistry = {"add": add, "multiply": multiply}

    first = FakeMessage(
        content="",
        tool_calls=[
            make_tool_call("call-add", "add", '{"a": 1, "b": 2}'),
            make_tool_call("call-mul", "multiply", '{"a": 3, "b": 4}'),
        ],
    )
    second = FakeMessage(content="done", tool_calls=None)
    text_ai = FakeTextAI(responses=[first, second])

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": "chain tools"},
        {"role": "user", "content": "add then multiply"},
    ]
    start_len = len(messages)

    result = run_agentic_chat(text_ai, messages, tools, registry)

    assert result.stop_reason == "done"
    assert result.steps == 2
    assert result.final_content == "done"

    assistant_tool_msg = messages[start_len]
    first_tool_msg = messages[start_len + 1]
    second_tool_msg = messages[start_len + 2]

    assert len(assistant_tool_msg["tool_calls"]) == 2
    assert first_tool_msg["name"] == "add"
    assert json.loads(first_tool_msg["content"]) == 3
    assert second_tool_msg["name"] == "multiply"
    assert json.loads(second_tool_msg["content"]) == 12


def test_direct_answer_no_tools_used() -> None:
    tools: List[Dict[str, Any]] = []
    registry: ToolRegistry = {}
    final_message = FakeMessage(content="Hi!", tool_calls=None)
    text_ai = FakeTextAI(responses=[final_message])

    messages: List[Dict[str, Any]] = [{"role": "user", "content": "Say hi"}]

    result = run_agentic_chat(text_ai, messages, tools, registry)

    assert result.stop_reason == "done"
    assert result.steps == 1
    assert result.final_content == "Hi!"
    assert messages[-1] == {"role": "assistant", "content": "Hi!"}


def test_unknown_tool_name_is_reported() -> None:
    tools = [
        {
            "type": "function",
            "function": {"name": "missing_tool", "parameters": {"type": "object", "properties": {}, "required": []}},
        }
    ]
    registry: ToolRegistry = {}

    first = FakeMessage(content=None, tool_calls=[make_tool_call("call-1", "missing_tool", '{"a": 1}')])
    second = FakeMessage(content="Final answer.", tool_calls=None)
    text_ai = FakeTextAI(responses=[first, second])

    messages: List[Dict[str, Any]] = [{"role": "user", "content": "call missing"}]
    result = run_agentic_chat(text_ai, messages, tools, registry)

    assert result.stop_reason == "done"
    assert result.final_content == "Final answer."

    tool_result = json.loads(messages[-2]["content"])
    assert tool_result["error"] == "unknown_tool"
    assert tool_result["tool"] == "missing_tool"
    assert tool_result["arguments"] == {"a": 1}


def test_tool_execution_error_is_captured() -> None:
    def boom() -> None:
        raise RuntimeError("kaboom")

    tools = [
        {
            "type": "function",
            "function": {"name": "bad_tool", "parameters": {"type": "object", "properties": {}, "required": []}},
        }
    ]
    registry: ToolRegistry = {"bad_tool": boom}

    first = FakeMessage(content=None, tool_calls=[make_tool_call("call-1", "bad_tool", "{}")])
    second = FakeMessage(content="done", tool_calls=None)
    text_ai = FakeTextAI(responses=[first, second])

    messages: List[Dict[str, Any]] = [{"role": "user", "content": "trigger"}]
    result = run_agentic_chat(text_ai, messages, tools, registry)

    assert result.stop_reason == "done"
    assert result.final_content == "done"

    tool_result = json.loads(messages[-2]["content"])
    assert tool_result["error"] == "tool_execution_error"
    assert tool_result["tool"] == "bad_tool"
    assert "kaboom" in tool_result["message"]


def test_invalid_json_arguments_are_handled() -> None:
    tools = [
        {
            "type": "function",
            "function": {"name": "add", "parameters": {"type": "object", "properties": {}, "required": []}},
        }
    ]
    registry: ToolRegistry = {"add": add}

    first = FakeMessage(content=None, tool_calls=[make_tool_call("call-1", "add", "not json")])
    second = FakeMessage(content="all done", tool_calls=None)
    text_ai = FakeTextAI(responses=[first, second])

    messages: List[Dict[str, Any]] = [{"role": "user", "content": "bad args"}]
    result = run_agentic_chat(text_ai, messages, tools, registry)

    assert result.stop_reason == "done"
    assert result.final_content == "all done"

    tool_result = json.loads(messages[-2]["content"])
    assert tool_result["error"] == "invalid_tool_arguments"
    assert tool_result["raw_arguments"] == "not json"


def test_repeated_tool_call_safeguard_triggers() -> None:
    tools = [
        {
            "type": "function",
            "function": {"name": "add", "parameters": {"type": "object", "properties": {}, "required": []}},
        }
    ]
    registry: ToolRegistry = {"add": add}
    repeated_message = FakeMessage(content=None, tool_calls=[make_tool_call("call-1", "add", '{"a": 1, "b": 1}')])

    text_ai = FakeTextAI(response_fn=lambda _messages: repeated_message)

    messages: List[Dict[str, Any]] = [{"role": "user", "content": "loop"}]
    result = run_agentic_chat(text_ai, messages, tools, registry, max_same_tool_calls=2)

    assert result.stop_reason == "repeated_tool_call"
    assert result.final_content is None
    assert result.steps == 2


def test_max_steps_safeguard_triggers() -> None:
    tools = [
        {
            "type": "function",
            "function": {"name": "loop", "parameters": {"type": "object", "properties": {}, "required": []}},
        }
    ]
    registry: ToolRegistry = {"loop": lambda **_: None}

    counter = {"n": 0}

    def response_fn(_messages: List[Dict[str, Any]]) -> FakeMessage:
        counter["n"] += 1
        call_id = f"call-{counter['n']}"
        raw_args = json.dumps({"step": counter["n"]})
        return FakeMessage(content=None, tool_calls=[make_tool_call(call_id, "loop", raw_args)])

    text_ai = FakeTextAI(response_fn=response_fn)

    messages: List[Dict[str, Any]] = [{"role": "user", "content": "never stop"}]
    result = run_agentic_chat(text_ai, messages, tools, registry, max_steps=3, max_same_tool_calls=99)

    assert result.stop_reason == "max_steps"
    assert result.final_content is None
    assert result.steps == 3
    assert len(messages) == 1 + (result.steps * 2)
