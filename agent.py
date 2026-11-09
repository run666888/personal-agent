from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from openai import APIError, AuthenticationError, RateLimitError

from agent_core import (
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    REASONING_CHOICES,
    AgentOptions,
    ChatAgent,
    ConversationStateError,
    create_client,
    load_local_env,
    load_system_prompt,
)

DEFAULT_SESSION_FILE = ".agent_session.json"

HELP_TEXT = """Commands:
/help                  Show help
/status                Show current agent settings
/reset                 Clear the current conversation state
/model <name>          Switch model and clear conversation state
/reasoning <level>     Set reasoning effort: none|low|medium|high|xhigh
/web on                Enable web search
/web off               Disable web search
/tools on              Enable local file and shell tools
/tools off             Disable local file and shell tools
/quit                  Exit the program
"""


def ensure_utf8_stdio() -> None:
    for name in ("stdin", "stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8")
        except ValueError:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A command-line OpenAI agent powered by the Responses API."
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_AGENT_MODEL", DEFAULT_MODEL),
        help=f"Model name. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=REASONING_CHOICES,
        default=os.getenv("OPENAI_REASONING_EFFORT", DEFAULT_REASONING_EFFORT),
        help=f"Reasoning effort. Default: {DEFAULT_REASONING_EFFORT}",
    )
    parser.add_argument(
        "--system-file",
        type=Path,
        help="Optional custom system prompt file.",
    )
    parser.add_argument(
        "--web-search",
        action="store_true",
        help="Start with OpenAI web search enabled.",
    )
    parser.add_argument(
        "--local-tools",
        action="store_true",
        help="Enable local file and shell tools inside the workspace.",
    )
    parser.add_argument(
        "--auto-confirm-tools",
        action="store_true",
        help="Automatically approve local tool actions without prompting.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Workspace root for local tools. Default: current directory.",
    )
    parser.add_argument(
        "--session-file",
        type=Path,
        default=Path(DEFAULT_SESSION_FILE),
        help=f"Session state file. Default: {DEFAULT_SESSION_FILE}",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore any saved session and start a new one.",
    )
    return parser.parse_args()


def print_banner(agent: ChatAgent, resumed: bool) -> None:
    print("=== My Personal Agent ===")
    print(f"Model: {agent.options.model}")
    print(f"Reasoning: {agent.options.reasoning_effort}")
    print(f"Web search: {'on' if agent.options.use_web_search else 'off'}")
    print(f"Local tools: {'on' if agent.options.enable_local_tools else 'off'}")
    print(f"Workspace: {agent.options.workspace}")
    print(f"Session file: {agent.options.session_file}")
    print(f"Resume state: {'loaded' if resumed else 'new session'}")
    print("Type /help to see commands.")


def print_status(agent: ChatAgent) -> None:
    print(f"Model: {agent.options.model}")
    print(f"Reasoning: {agent.options.reasoning_effort}")
    print(f"Web search: {'on' if agent.options.use_web_search else 'off'}")
    print(f"Local tools: {'on' if agent.options.enable_local_tools else 'off'}")
    print(f"Auto confirm tools: {'on' if agent.options.auto_confirm_tools else 'off'}")
    print(f"Workspace: {agent.options.workspace}")
    print(f"Session file: {agent.options.session_file}")
    print(
        "Conversation state: "
        f"{'saved response chain active' if agent.previous_response_id else 'empty'}"
    )


def confirm_tool_action(action: str, detail: str) -> bool:
    print(f"\nTool approval requested: {action}")
    print(detail)
    answer = input("Allow? [y/N]: ").strip().lower()
    return answer in {"y", "yes"}


def main() -> int:
    ensure_utf8_stdio()
    load_local_env()
    args = parse_args()
    system_prompt = load_system_prompt(args.system_file)
    client = create_client()

    options = AgentOptions(
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        use_web_search=args.web_search,
        enable_local_tools=args.local_tools,
        auto_confirm_tools=args.auto_confirm_tools,
        workspace=args.workspace,
        session_file=args.session_file,
        system_prompt=system_prompt,
    )
    agent = ChatAgent(
        client=client,
        options=options,
        confirm_callback=confirm_tool_action,
    )

    resumed = False
    if not args.fresh:
        resumed = agent.load_session()

    print_banner(agent, resumed=resumed)

    while True:
        try:
            user_text = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return 0

        if not user_text:
            continue

        if user_text == "/help":
            print(HELP_TEXT)
            continue

        if user_text == "/status":
            print_status(agent)
            continue

        if user_text == "/reset":
            agent.reset()
            print("Conversation state cleared.")
            continue

        if user_text.startswith("/model "):
            new_model = user_text.split(maxsplit=1)[1].strip()
            if not new_model:
                print("Usage: /model gpt-5.4")
                continue

            agent.options.model = new_model
            agent.reset()
            print(f"Model switched to: {new_model}")
            print("Conversation state cleared.")
            continue

        if user_text.startswith("/reasoning "):
            new_effort = user_text.split(maxsplit=1)[1].strip().lower()
            if new_effort not in REASONING_CHOICES:
                print("Valid values: none, low, medium, high, xhigh")
                continue

            agent.options.reasoning_effort = new_effort
            agent.save_session()
            print(f"Reasoning effort set to: {new_effort}")
            continue

        if user_text == "/web on":
            agent.options.use_web_search = True
            agent.save_session()
            print("Web search enabled.")
            continue

        if user_text == "/web off":
            agent.options.use_web_search = False
            agent.save_session()
            print("Web search disabled.")
            continue

        if user_text == "/tools on":
            agent.options.enable_local_tools = True
            agent.save_session()
            print("Local tools enabled.")
            continue

        if user_text == "/tools off":
            agent.options.enable_local_tools = False
            agent.save_session()
            print("Local tools disabled.")
            continue

        if user_text == "/quit":
            print("Bye.")
            return 0

        try:
            print("\nAgent> ", end="", flush=True)
            reply = agent.ask(
                user_text,
                text_callback=lambda chunk: print(chunk, end="", flush=True),
                event_callback=lambda message: print(
                    f"\n[{message}]\nAgent> ",
                    end="",
                    flush=True,
                ),
            )
            if not reply:
                print("(empty response)", end="", flush=True)
            print()
        except ConversationStateError as exc:
            print(f"\nSession reset: {exc}")
        except AuthenticationError:
            print("\nAuthentication failed. Check OPENAI_API_KEY.")
        except RateLimitError as exc:
            print(f"\nRate limit or quota issue: {exc}")
        except APIError as exc:
            print(f"\nOpenAI API error: {exc}")
        except Exception as exc:
            print(f"\nUnexpected error: {exc}")


if __name__ == "__main__":
    sys.exit(main())
