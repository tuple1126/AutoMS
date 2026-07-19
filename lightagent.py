"""Minimal, local OpenAI-compatible agent runtime for AutoMS.

This is an original implementation for this repository.  It intentionally
registers only callable tools supplied by the application and never writes,
imports, or executes model-generated Python code.
"""

from __future__ import annotations

import copy
import inspect
import json
import logging
import os
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Union

from openai import OpenAI


ToolCallable = Callable[..., Any]


class ToolRegistry:
    """Keep trusted application callables and their OpenAI tool schemas."""

    def __init__(self) -> None:
        self.function_mappings: Dict[str, ToolCallable] = {}
        self.function_info: Dict[str, Dict[str, Any]] = {}
        self.openai_function_schemas: List[Dict[str, Any]] = []

    def register_tool(self, func: ToolCallable) -> bool:
        info = getattr(func, "tool_info", None)
        if not isinstance(info, dict):
            raise ValueError("A tool must be a callable with a dictionary tool_info attribute")

        name = info.get("tool_name")
        parameters = info.get("tool_params", [])
        if not isinstance(name, str) or not name:
            raise ValueError("tool_info.tool_name must be a non-empty string")
        if not isinstance(parameters, list):
            raise ValueError(f"tool_info.tool_params for {name!r} must be a list")

        properties: Dict[str, Any] = {}
        required: List[str] = []
        for parameter in parameters:
            if not isinstance(parameter, dict) or not isinstance(parameter.get("name"), str):
                raise ValueError(f"tool {name!r} has an invalid parameter description")
            parameter_name = parameter["name"]
            schema: Dict[str, Any] = {
                "type": parameter.get("type", "string"),
                "description": parameter.get("description", ""),
            }
            if schema["type"] == "array":
                schema["items"] = parameter.get("items", {"type": "string"})
            if schema["type"] == "object" and "properties" in parameter:
                schema["properties"] = parameter["properties"]
            properties[parameter_name] = schema
            if parameter.get("required", False):
                required.append(parameter_name)

        schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": info.get("tool_description", ""),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        }
        self.function_mappings[name] = func
        self.function_info[name] = copy.deepcopy(info)
        self.openai_function_schemas = [
            item for item in self.openai_function_schemas if item["function"]["name"] != name
        ]
        self.openai_function_schemas.append(schema)
        return True

    def get_tools(self) -> List[Dict[str, Any]]:
        return copy.deepcopy(self.openai_function_schemas)

    def validate_arguments(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        info = self.function_info[tool_name]
        parameters = info.get("tool_params", [])
        allowed = {parameter["name"] for parameter in parameters}
        unexpected = sorted(set(arguments) - allowed)
        if unexpected:
            raise ValueError(f"unexpected arguments for {tool_name}: {', '.join(unexpected)}")

        missing = [
            parameter["name"]
            for parameter in parameters
            if parameter.get("required", False)
            and (parameter["name"] not in arguments or arguments[parameter["name"]] is None)
        ]
        if missing:
            raise ValueError(f"missing required arguments for {tool_name}: {', '.join(missing)}")


class LightAgent:
    """Synchronous OpenAI-compatible agent with trusted local tool dispatch."""

    __version__ = "1.0.0"

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        instructions: Optional[str] = None,
        role: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        tools: Optional[Iterable[ToolCallable]] = None,
        debug: bool = False,
        log_level: str = "INFO",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        **_unused: Any,
    ) -> None:
        self.name = name or "LightAgent"
        self.instructions = instructions or "You are a helpful assistant."
        self.role = role or "assistant"
        self.model = model or os.environ.get("AUTOMS_MAIN_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        if not self.api_key:
            raise ValueError("Set OPENAI_API_KEY or pass api_key when creating a LightAgent")

        client_options: Dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            client_options["base_url"] = self.base_url
        self.client = OpenAI(**client_options)

        self.tool_registry = ToolRegistry()
        self.loaded_tools: Dict[str, ToolCallable] = {}
        self.tools: List[ToolCallable] = []
        self.chat_params: Dict[str, Any] = {}
        self.default_request_options: Dict[str, Any] = {}
        if temperature is not None:
            self.default_request_options["temperature"] = temperature
        if max_tokens is not None:
            self.default_request_options["max_tokens"] = max_tokens
        self.default_tool_choice = tool_choice if tool_choice is not None else "auto"
        self.debug = bool(debug)
        self.logger = logging.getLogger(f"automS.{self.name}")
        self.logger.setLevel(log_level.upper())

        if tools:
            self.load_tools(list(tools))

    def log(self, level: str, action: str, data: Any) -> None:
        if self.debug:
            getattr(self.logger, level.lower(), self.logger.info)("%s: %s", action, data)

    def load_tools(self, tools: Iterable[Union[str, ToolCallable]], **_unused: Any) -> None:
        for tool in tools:
            if isinstance(tool, str):
                raise ValueError(
                    "String-based dynamic tool loading is disabled. Pass a trusted callable with tool_info."
                )
            if not callable(tool):
                raise TypeError("Each tool must be callable")
            self.tool_registry.register_tool(tool)
            tool_name = tool.tool_info["tool_name"]
            self.loaded_tools[tool_name] = tool
            self.tools.append(tool)

    def get_tools(self) -> List[Dict[str, Any]]:
        return self.tool_registry.get_tools()

    def get_tool(self, tool_name: str) -> ToolCallable:
        try:
            return self.loaded_tools[tool_name]
        except KeyError as exc:
            raise ValueError(f"Tool {tool_name!r} is not loaded") from exc

    def get_history(self) -> List[Dict[str, Any]]:
        return copy.deepcopy(self.chat_params.get("messages", []))

    def _system_message(self) -> Dict[str, str]:
        return {
            "role": "system",
            "content": (
                f"Agent name: {self.name}\n"
                f"Role: {self.role}\n\n"
                f"{self.instructions}"
            ),
        }

    @staticmethod
    def _safe_history(history: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        accepted_roles = {"user", "assistant", "tool"}
        safe: List[Dict[str, Any]] = []
        for message in history or []:
            if not isinstance(message, dict) or message.get("role") not in accepted_roles:
                continue
            if "content" not in message:
                continue
            if message["role"] == "tool" and not message.get("tool_call_id"):
                continue
            safe.append(copy.deepcopy(message))
        return safe

    @staticmethod
    def _json_text(value: Any) -> str:
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(value)

    def _call_tool(self, tool_name: str, raw_arguments: str) -> str:
        if tool_name not in self.tool_registry.function_mappings:
            return self._json_text({"ok": False, "error": f"unknown tool: {tool_name}"})
        try:
            arguments = json.loads(raw_arguments or "{}")
            if not isinstance(arguments, dict):
                raise ValueError("tool arguments must be a JSON object")
            self.tool_registry.validate_arguments(tool_name, arguments)
            result = self.tool_registry.function_mappings[tool_name](**arguments)
            if inspect.isawaitable(result):
                raise RuntimeError("asynchronous tools are not supported by the synchronous AutoMS runtime")
            if inspect.isgenerator(result):
                result = list(result)
            return self._json_text({"ok": True, "result": result})
        except Exception as exc:
            self.log("error", "tool_call_failed", {"tool": tool_name, "error": str(exc)})
            return self._json_text({"ok": False, "error": f"{type(exc).__name__}: tool execution failed"})

    def _create_completion(self, request: Dict[str, Any]) -> Any:
        return self.client.chat.completions.create(**request)

    @staticmethod
    def _assistant_tool_message(message: Any) -> Dict[str, Any]:
        return {
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in message.tool_calls or []
            ],
        }

    def _run_non_streaming(
        self,
        query: str,
        history: Optional[List[Dict[str, Any]]],
        max_retry: int,
        metadata: Optional[Dict[str, Any]],
    ) -> str:
        messages = [self._system_message(), *self._safe_history(history), {"role": "user", "content": query}]
        request: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            **self.default_request_options,
        }
        tools = self.tool_registry.get_tools()
        if tools:
            request["tools"] = tools
            request["tool_choice"] = self.default_tool_choice
        for key in ("temperature", "max_tokens", "top_p", "seed", "response_format", "tool_choice"):
            if metadata and key in metadata:
                if key != "tool_choice" or tools:
                    request[key] = metadata[key]

        for _ in range(max_retry + 1):
            self.chat_params = copy.deepcopy(request)
            response = self._create_completion(request)
            if not getattr(response, "choices", None):
                raise RuntimeError("The model returned no choices")
            message = response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None) or []
            if not tool_calls:
                return message.content or ""

            messages.append(self._assistant_tool_message(message))
            for call in tool_calls:
                tool_result = self._call_tool(call.function.name, call.function.arguments)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": tool_result})
            request["messages"] = messages

        raise RuntimeError(f"Tool-call limit exceeded for agent {self.name}")

    def run(
        self,
        query: str,
        light_swarm: Any = None,
        stream: bool = False,
        max_retry: int = 10,
        user_id: str = "default_user",
        history: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Union[Generator[str, None, None], str]:
        del light_swarm, user_id
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if max_retry < 0:
            raise ValueError("max_retry must be non-negative")

        if stream:
            def response_generator() -> Generator[str, None, None]:
                yield self._run_non_streaming(query, history, max_retry, metadata)

            return response_generator()
        return self._run_non_streaming(query, history, max_retry, metadata)


class LightSwarm:
    """Small compatibility container for named AutoMS agents."""

    def __init__(self) -> None:
        self.agents: Dict[str, LightAgent] = {}

    def register_agent(self, *agents: LightAgent) -> None:
        for agent in agents:
            if not isinstance(agent, LightAgent):
                raise TypeError("LightSwarm accepts LightAgent instances")
            self.agents[agent.name] = agent

    def run(self, agent: LightAgent, query: str, stream: bool = False) -> Union[Generator[str, None, None], str]:
        if agent.name not in self.agents:
            raise ValueError(f"Agent {agent.name!r} is not registered")
        return agent.run(query, stream=stream)


__all__ = ["LightAgent", "LightSwarm", "ToolRegistry"]
