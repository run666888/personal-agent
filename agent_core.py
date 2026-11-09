from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from openai import APIError, OpenAI

DEFAULT_MODEL = "gpt-5.4"
DEFAULT_REASONING_EFFORT = "low"
REASONING_CHOICES = ("none", "low", "medium", "high", "xhigh")
APP_DATA_DIR = Path.home() / ".my_personal_agent"
INTERNAL_SUMMARY_MARKERS = (
    "CONTEXT_COMPACTION_SUMMARY_V1",
    "(CONTEXT_COMPACTION_SUMMARY_V1",
)

DEFAULT_PROMPT = """You are my personal AI assistant.

Rules:
1. Default to replying in Simplified Chinese unless I ask for another language.
2. Be direct, accurate, and practical.
3. When writing code, prefer complete, runnable examples.
4. If information is missing, make the safest reasonable assumption and state it briefly.
5. Keep answers clear and avoid unnecessary complexity.
"""

EventCallback = Callable[[str], None]
TextCallback = Callable[[str], None]
ConfirmCallback = Callable[[str, str], bool]


class ConversationStateError(RuntimeError):
    """Raised when a saved response chain is no longer available."""


class ToolExecutionError(RuntimeError):
    """Raised when a local tool cannot complete successfully."""


@dataclass
class AgentOptions:
    model: str = DEFAULT_MODEL
    reasoning_effort: str = DEFAULT_REASONING_EFFORT
    use_web_search: bool = False
    enable_local_tools: bool = False
    auto_confirm_tools: bool = False
    workspace: Path = Path.cwd()
    session_file: Path = Path(".agent_session.json")
    system_prompt: str = DEFAULT_PROMPT


def ensure_app_data_dir() -> Path:
    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return APP_DATA_DIR


def load_local_env(candidates: list[Path] | None = None) -> None:
    if candidates is None:
        candidates = [Path(".env"), ensure_app_data_dir() / ".env"]

    for env_path in candidates:
        if not env_path.exists():
            continue

        for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if key and value and key not in os.environ:
                os.environ[key] = value


def load_system_prompt(prompt_file: Path | None = None) -> str:
    if prompt_file is not None:
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        return prompt_file.read_text(encoding="utf-8-sig").strip()

    default_file = Path("system_prompt.txt")
    if default_file.exists():
        return default_file.read_text(encoding="utf-8-sig").strip()

    return DEFAULT_PROMPT


def create_client(api_key: str | None = None, base_url: str | None = None) -> OpenAI:
    resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_api_key:
        raise SystemExit(
            "OPENAI_API_KEY is missing.\n"
            "PowerShell example: $env:OPENAI_API_KEY='your-api-key'"
        )

    resolved_base_url = base_url or os.getenv("OPENAI_BASE_URL")
    if resolved_base_url:
        return OpenAI(api_key=resolved_api_key, base_url=resolved_base_url)

    return OpenAI(api_key=resolved_api_key)


def default_gui_paths() -> tuple[Path, Path]:
    app_dir = ensure_app_data_dir()
    return app_dir / "gui_settings.json", app_dir / "gui_session.json"


class LocalToolExecutor:
    def __init__(
        self,
        workspace: Path,
        auto_confirm: bool = False,
        confirm_callback: ConfirmCallback | None = None,
        event_callback: EventCallback | None = None,
    ) -> None:
        self.workspace = workspace.resolve()
        self.auto_confirm = auto_confirm
        self.confirm_callback = confirm_callback
        self.event_callback = event_callback

    def tool_definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": "list_files",
                "description": "List files and folders inside the current workspace.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "recursive": {"type": "boolean"},
                        "max_entries": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 500,
                        },
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "read_text_file",
                "description": "Read a UTF-8 text file from the workspace.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "max_chars": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 50000,
                        },
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "write_text_file",
                "description": "Create or overwrite a UTF-8 text file inside the workspace.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "append_text_file",
                "description": "Append UTF-8 text to a file inside the workspace.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "make_directory",
                "description": "Create a directory inside the workspace.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "run_shell_command",
                "description": "Run a PowerShell command inside the workspace and return stdout and stderr.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "cwd": {"type": "string"},
                        "timeout_seconds": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 120,
                        },
                    },
                    "required": ["command"],
                    "additionalProperties": False,
                },
            },
        ]

    def execute(self, name: str, arguments_json: str) -> str:
        try:
            arguments = json.loads(arguments_json or "{}")
        except json.JSONDecodeError as exc:
            raise ToolExecutionError(f"Invalid tool arguments for {name}: {exc}") from exc

        if name == "list_files":
            return self._list_files(**arguments)
        if name == "read_text_file":
            return self._read_text_file(**arguments)
        if name == "write_text_file":
            return self._write_text_file(**arguments)
        if name == "append_text_file":
            return self._append_text_file(**arguments)
        if name == "make_directory":
            return self._make_directory(**arguments)
        if name == "run_shell_command":
            return self._run_shell_command(**arguments)

        raise ToolExecutionError(f"Unknown tool: {name}")

    def _emit(self, message: str) -> None:
        if self.event_callback:
            self.event_callback(message)

    def _confirm(self, action: str, detail: str) -> bool:
        if self.auto_confirm:
            return True
        if self.confirm_callback is None:
            return False
        return self.confirm_callback(action, detail)

    def _resolve_path(self, user_path: str | None) -> Path:
        raw_path = user_path or "."
        candidate = Path(raw_path)
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            resolved = (self.workspace / candidate).resolve()

        if resolved != self.workspace and self.workspace not in resolved.parents:
            raise ToolExecutionError(
                f"Path is outside the workspace and was blocked: {resolved}"
            )

        return resolved

    def _json(self, payload: dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _list_files(
        self,
        path: str = ".",
        recursive: bool = False,
        max_entries: int = 200,
    ) -> str:
        target = self._resolve_path(path)
        if not target.exists():
            raise ToolExecutionError(f"Path does not exist: {target}")
        if not target.is_dir():
            raise ToolExecutionError(f"Path is not a directory: {target}")

        entries: list[dict[str, Any]] = []
        iterator = target.rglob("*") if recursive else target.iterdir()
        for index, entry in enumerate(iterator):
            if index >= max_entries:
                break
            info = {
                "path": str(entry.relative_to(self.workspace)),
                "type": "dir" if entry.is_dir() else "file",
            }
            if entry.is_file():
                try:
                    info["size"] = entry.stat().st_size
                except OSError:
                    info["size"] = None
            entries.append(info)

        return self._json(
            {
                "workspace": str(self.workspace),
                "path": str(target.relative_to(self.workspace)),
                "recursive": recursive,
                "entries": entries,
                "count": len(entries),
                "truncated": len(entries) >= max_entries,
            }
        )

    def _read_text_file(self, path: str, max_chars: int = 20000) -> str:
        target = self._resolve_path(path)
        if not target.exists():
            raise ToolExecutionError(f"File does not exist: {target}")
        if not target.is_file():
            raise ToolExecutionError(f"Path is not a file: {target}")

        text = target.read_text(encoding="utf-8")
        truncated = len(text) > max_chars
        return self._json(
            {
                "path": str(target.relative_to(self.workspace)),
                "content": text[:max_chars],
                "truncated": truncated,
                "full_length": len(text),
            }
        )

    def _write_text_file(self, path: str, content: str) -> str:
        target = self._resolve_path(path)
        if not self._confirm("Write file", str(target)):
            raise ToolExecutionError("User declined the file write operation.")

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        self._emit(f"Tool wrote file: {target}")
        return self._json(
            {
                "status": "ok",
                "path": str(target.relative_to(self.workspace)),
                "bytes_written": len(content.encode("utf-8")),
            }
        )

    def _append_text_file(self, path: str, content: str) -> str:
        target = self._resolve_path(path)
        if not self._confirm("Append file", str(target)):
            raise ToolExecutionError("User declined the file append operation.")

        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(content)
        self._emit(f"Tool appended file: {target}")
        return self._json(
            {
                "status": "ok",
                "path": str(target.relative_to(self.workspace)),
                "bytes_appended": len(content.encode("utf-8")),
            }
        )

    def _make_directory(self, path: str) -> str:
        target = self._resolve_path(path)
        if not self._confirm("Create directory", str(target)):
            raise ToolExecutionError("User declined the directory creation.")

        target.mkdir(parents=True, exist_ok=True)
        self._emit(f"Tool created directory: {target}")
        return self._json(
            {
                "status": "ok",
                "path": str(target.relative_to(self.workspace)),
            }
        )

    def _run_shell_command(
        self,
        command: str,
        cwd: str = ".",
        timeout_seconds: int = 30,
    ) -> str:
        working_dir = self._resolve_path(cwd)
        detail = f"{command}\n\ncwd={working_dir}"
        if not self._confirm("Run command", detail):
            raise ToolExecutionError("User declined the shell command.")

        self._emit(f"Tool running command in {working_dir}: {command}")
        completed = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                command,
            ],
            cwd=str(working_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout_seconds,
        )
        stdout = completed.stdout[:12000]
        stderr = completed.stderr[:12000]
        return self._json(
            {
                "status": "ok",
                "command": command,
                "cwd": str(working_dir),
                "returncode": completed.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "stdout_truncated": len(completed.stdout) > len(stdout),
                "stderr_truncated": len(completed.stderr) > len(stderr),
            }
        )


class ChatAgent:
    def __init__(
        self,
        client: OpenAI,
        options: AgentOptions,
        confirm_callback: ConfirmCallback | None = None,
    ) -> None:
        self.client = client
        self.options = options
        self.previous_response_id: str | None = None
        self.confirm_callback = confirm_callback

    def load_session(self) -> bool:
        session_file = self.options.session_file
        if not session_file.exists():
            return False

        try:
            data = json.loads(session_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False

        self.previous_response_id = data.get("previous_response_id")
        if isinstance(data.get("model"), str):
            self.options.model = data["model"]
        if data.get("reasoning_effort") in REASONING_CHOICES:
            self.options.reasoning_effort = data["reasoning_effort"]
        if isinstance(data.get("use_web_search"), bool):
            self.options.use_web_search = data["use_web_search"]
        if isinstance(data.get("enable_local_tools"), bool):
            self.options.enable_local_tools = data["enable_local_tools"]
        if isinstance(data.get("auto_confirm_tools"), bool):
            self.options.auto_confirm_tools = data["auto_confirm_tools"]
        if isinstance(data.get("workspace"), str):
            self.options.workspace = Path(data["workspace"])
        return True

    def save_session(self) -> None:
        self.options.session_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.options.model,
            "reasoning_effort": self.options.reasoning_effort,
            "use_web_search": self.options.use_web_search,
            "enable_local_tools": self.options.enable_local_tools,
            "auto_confirm_tools": self.options.auto_confirm_tools,
            "workspace": str(self.options.workspace),
            "previous_response_id": self.previous_response_id,
        }
        self.options.session_file.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    def reset(self) -> None:
        self.previous_response_id = None
        self.save_session()

    def ask(
        self,
        user_text: str,
        text_callback: TextCallback | None = None,
        event_callback: EventCallback | None = None,
    ) -> str:
        return self._ask_with_recovery(
            user_text=user_text,
            text_callback=text_callback,
            event_callback=event_callback,
            allow_retry=True,
        )

    def _ask_with_recovery(
        self,
        user_text: str,
        text_callback: TextCallback | None,
        event_callback: EventCallback | None,
        allow_retry: bool,
    ) -> str:
        if self.options.enable_local_tools:
            text = self._ask_with_local_tools(
                user_text=user_text,
                text_callback=text_callback,
                event_callback=event_callback,
            )
        else:
            text = self._ask_streaming_simple(
                user_text=user_text,
                text_callback=text_callback,
                event_callback=event_callback,
            )

        if self._looks_like_internal_summary(text):
            if allow_retry:
                previous_response_id = self.previous_response_id
                self.reset()
                if event_callback:
                    event_callback(
                        "Detected an internal context summary from the provider. "
                        "The session was reset and the request is being retried."
                    )
                if previous_response_id is not None:
                    return self._ask_with_recovery(
                        user_text=user_text,
                        text_callback=text_callback,
                        event_callback=event_callback,
                        allow_retry=False,
                    )

            raise ConversationStateError(
                "The provider returned an internal context summary instead of a normal answer. "
                "The session has been reset. Please try again."
            )

        return text

    def _base_request(self) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.options.model,
            "reasoning": {"effort": self.options.reasoning_effort},
            "store": True,
        }
        if self.options.use_web_search:
            request["tools"] = [{"type": "web_search"}]
        return request

    def _ask_streaming_simple(
        self,
        user_text: str,
        text_callback: TextCallback | None = None,
        event_callback: EventCallback | None = None,
        use_web_search: bool | None = None,
    ) -> str:
        request = self._base_request()
        if use_web_search is not None:
            if use_web_search:
                request["tools"] = [{"type": "web_search"}]
            else:
                request.pop("tools", None)
        request["input"] = user_text
        starting_response_id = self.previous_response_id

        if self.previous_response_id is None:
            request["instructions"] = self.options.system_prompt
        else:
            request["previous_response_id"] = self.previous_response_id

        streamed_text = ""
        displayed_chars = 0

        try:
            try:
                with self.client.responses.stream(**request) as stream:
                    for event in stream:
                        if getattr(event, "type", None) != "response.output_text.delta":
                            continue
                        delta = getattr(event, "delta", "")
                        if not delta:
                            continue
                        streamed_text += delta
                        if (
                            text_callback
                            and not self._looks_like_internal_summary_prefix(streamed_text)
                        ):
                            safe_chunk = streamed_text[displayed_chars:]
                            if safe_chunk:
                                text_callback(safe_chunk)
                                displayed_chars = len(streamed_text)
                    response = stream.get_final_response()
            except Exception:
                response = self.client.responses.create(**request)
        except APIError as exc:
            self._handle_possible_stale_session(exc)
            raise

        self.previous_response_id = response.id
        self.save_session()
        text = self._extract_text(response).strip() or streamed_text.strip()

        if (
            not text
            and "tools" in request
            and use_web_search is not False
        ):
            self.previous_response_id = starting_response_id
            self.save_session()
            if event_callback:
                event_callback(
                    "No visible text came back with web search enabled. "
                    "Retrying once without web search for compatibility."
                )
            return self._ask_streaming_simple(
                user_text=user_text,
                text_callback=text_callback,
                event_callback=event_callback,
                use_web_search=False,
            )

        if text_callback and text and displayed_chars == 0:
            text_callback(text)
        return text

    def _ask_with_local_tools(
        self,
        user_text: str,
        text_callback: TextCallback | None = None,
        event_callback: EventCallback | None = None,
    ) -> str:
        tool_executor = LocalToolExecutor(
            workspace=self.options.workspace,
            auto_confirm=self.options.auto_confirm_tools,
            confirm_callback=self.confirm_callback,
            event_callback=event_callback,
        )
        tools = tool_executor.tool_definitions()
        if self.options.use_web_search:
            tools = [{"type": "web_search"}] + tools

        request = self._base_request()
        request["input"] = user_text
        request["tools"] = tools
        request["parallel_tool_calls"] = True
        starting_response_id = self.previous_response_id
        tool_calls_used = False

        if self.previous_response_id is None:
            request["instructions"] = self._tool_enabled_system_prompt()
        else:
            request["previous_response_id"] = self.previous_response_id

        try:
            response = self.client.responses.create(**request)
            max_rounds = 8
            for _ in range(max_rounds):
                function_calls = self._extract_function_calls(response)
                if not function_calls:
                    break
                tool_calls_used = True

                tool_outputs: list[dict[str, str]] = []
                for tool_call in function_calls:
                    call_name = getattr(tool_call, "name", "unknown")
                    if event_callback:
                        event_callback(f"Model requested tool: {call_name}")
                    try:
                        output = tool_executor.execute(
                            name=call_name,
                            arguments_json=getattr(tool_call, "arguments", "{}"),
                        )
                    except Exception as exc:
                        output = json.dumps(
                            {"status": "error", "message": str(exc)},
                            ensure_ascii=False,
                        )
                    tool_outputs.append(
                        {
                            "type": "function_call_output",
                            "call_id": getattr(tool_call, "call_id"),
                            "output": output,
                        }
                    )

                response = self.client.responses.create(
                    model=self.options.model,
                    previous_response_id=response.id,
                    input=tool_outputs,
                    tools=tools,
                    reasoning={"effort": self.options.reasoning_effort},
                    parallel_tool_calls=True,
                    store=True,
                )
            else:
                raise RuntimeError("Tool loop exceeded the maximum number of rounds.")
        except APIError as exc:
            self._handle_possible_stale_session(exc)
            raise

        self.previous_response_id = response.id
        self.save_session()
        text = self._extract_text(response).strip()
        if not text and not tool_calls_used:
            self.previous_response_id = starting_response_id
            self.save_session()
            if event_callback:
                event_callback(
                    "Tool mode returned no visible text. "
                    "Retrying once without local tools for compatibility."
                )
            return self._ask_streaming_simple(
                user_text=user_text,
                text_callback=text_callback,
                event_callback=event_callback,
                use_web_search=False if self.options.use_web_search else None,
            )
        if text_callback and text:
            text_callback(text)
        return text

    def _handle_possible_stale_session(self, exc: APIError) -> None:
        message = str(exc).lower()
        if "previous_response_id" in message or "conversation" in message:
            self.reset()
            raise ConversationStateError(
                "The saved conversation state is no longer available. "
                "The session has been reset. Please send your message again."
            ) from exc

    def _tool_enabled_system_prompt(self) -> str:
        return (
            f"{self.options.system_prompt}\n\n"
            "You may use local tools when they help. "
            f"Only access paths inside this workspace: {self.options.workspace.resolve()}. "
            "Before running risky actions, inspect files first."
        )

    @staticmethod
    def _extract_function_calls(response: Any) -> list[Any]:
        items = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) == "function_call":
                items.append(item)
        return items

    @staticmethod
    def _extract_text(response: Any) -> str:
        pieces: list[str] = []
        direct_output_text = getattr(response, "output_text", "")
        if isinstance(direct_output_text, str) and direct_output_text.strip():
            if not ChatAgent._looks_like_internal_summary(direct_output_text):
                pieces.append(direct_output_text)

        if pieces:
            return "\n".join(pieces)

        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) == "function_call":
                continue
            for text in ChatAgent._extract_text_from_item(item):
                if ChatAgent._looks_like_internal_summary(text):
                    continue
                pieces.append(text)
        return "\n".join(pieces)

    @staticmethod
    def _looks_like_internal_summary(text: str) -> bool:
        normalized = text.strip()
        return any(normalized.startswith(marker) for marker in INTERNAL_SUMMARY_MARKERS)

    @staticmethod
    def _looks_like_internal_summary_prefix(text: str) -> bool:
        normalized = text.lstrip()
        if not normalized:
            return False
        return any(marker.startswith(normalized) for marker in INTERNAL_SUMMARY_MARKERS)

    @staticmethod
    def _extract_text_from_item(item: Any) -> list[str]:
        pieces: list[str] = []
        content_list = getattr(item, "content", None)
        if content_list is None and isinstance(item, dict):
            content_list = item.get("content")

        if content_list:
            for content in content_list:
                text = ChatAgent._extract_text_value(content)
                if text:
                    pieces.append(text)

        if pieces:
            return pieces

        fallback_text = ChatAgent._extract_text_value(item)
        if fallback_text:
            pieces.append(fallback_text)
        return pieces

    @staticmethod
    def _extract_text_value(value: Any) -> str:
        if isinstance(value, str):
            return value

        if isinstance(value, dict):
            value_type = value.get("type")
            if value_type in {"output_text", "text"}:
                nested_text = value.get("text")
                if isinstance(nested_text, str):
                    return nested_text
                if isinstance(nested_text, dict):
                    return str(nested_text.get("value", "") or "")
            for key in ("text", "value"):
                nested = value.get(key)
                if isinstance(nested, str):
                    return nested
                if isinstance(nested, dict):
                    nested_value = nested.get("value")
                    if isinstance(nested_value, str):
                        return nested_value
            return ""

        value_type = getattr(value, "type", None)
        if value_type not in {None, "output_text", "text"}:
            return ""

        text = getattr(value, "text", None)
        if isinstance(text, str):
            return text
        if text is not None:
            nested_value = getattr(text, "value", None)
            if isinstance(nested_value, str):
                return nested_value

        raw_value = getattr(value, "value", None)
        if isinstance(raw_value, str):
            return raw_value

        return ""
