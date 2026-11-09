from __future__ import annotations

import json
import threading
import tkinter as tk
from pathlib import Path
from queue import Empty, Queue
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from openai import APIError, AuthenticationError, RateLimitError

from agent_core import (
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    REASONING_CHOICES,
    AgentOptions,
    ChatAgent,
    ConversationStateError,
    create_client,
    default_gui_paths,
    load_local_env,
    load_system_prompt,
)

PRESETS = {
    "Balanced": {"model": "gpt-5.4", "reasoning_effort": "low"},
    "Smart": {"model": "gpt-5.4", "reasoning_effort": "medium"},
    "Deep": {"model": "gpt-5.4", "reasoning_effort": "high"},
    "Fast": {"model": "gpt-5.4-mini", "reasoning_effort": "low"},
    "Custom": {},
}


class AgentDesktopApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Personal Agent App")
        self.root.geometry("1180x760")
        self.root.minsize(980, 620)

        self.settings_file, self.session_file = default_gui_paths()
        load_local_env()

        self.event_queue: Queue[tuple[str, str]] = Queue()
        self.current_agent: ChatAgent | None = None
        self.worker_thread: threading.Thread | None = None

        self.api_key_var = tk.StringVar()
        self.base_url_var = tk.StringVar()
        self.workspace_var = tk.StringVar(value=str(Path.cwd()))
        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        self.reasoning_var = tk.StringVar(value=DEFAULT_REASONING_EFFORT)
        self.preset_var = tk.StringVar(value="Balanced")
        self.web_search_var = tk.BooleanVar(value=False)
        self.local_tools_var = tk.BooleanVar(value=False)
        self.auto_confirm_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Ready")

        self._build_ui()
        self._load_settings()
        self.root.after(80, self._drain_queue)

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill="both", expand=True)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        sidebar = ttk.Frame(main, padding=(0, 0, 12, 0))
        sidebar.grid(row=0, column=0, sticky="nsw")

        chat_panel = ttk.Frame(main)
        chat_panel.grid(row=0, column=1, sticky="nsew")
        chat_panel.columnconfigure(0, weight=1)
        chat_panel.rowconfigure(0, weight=1)

        self._build_sidebar(sidebar)
        self._build_chat_panel(chat_panel)

    def _build_sidebar(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Connection", font=("Segoe UI", 11, "bold")).pack(anchor="w")

        ttk.Label(parent, text="API Key").pack(anchor="w", pady=(10, 2))
        ttk.Entry(parent, textvariable=self.api_key_var, show="*", width=38).pack(fill="x")

        ttk.Label(parent, text="Base URL").pack(anchor="w", pady=(10, 2))
        ttk.Entry(parent, textvariable=self.base_url_var, width=38).pack(fill="x")

        ttk.Label(parent, text="Workspace").pack(anchor="w", pady=(10, 2))
        workspace_row = ttk.Frame(parent)
        workspace_row.pack(fill="x")
        ttk.Entry(workspace_row, textvariable=self.workspace_var, width=30).pack(side="left", fill="x", expand=True)
        ttk.Button(workspace_row, text="Browse", command=self._browse_workspace).pack(side="left", padx=(6, 0))

        ttk.Separator(parent).pack(fill="x", pady=14)
        ttk.Label(parent, text="Behavior", font=("Segoe UI", 11, "bold")).pack(anchor="w")

        ttk.Label(parent, text="Preset").pack(anchor="w", pady=(10, 2))
        preset_box = ttk.Combobox(
            parent,
            textvariable=self.preset_var,
            values=list(PRESETS.keys()),
            state="readonly",
            width=35,
        )
        preset_box.pack(fill="x")
        preset_box.bind("<<ComboboxSelected>>", self._apply_preset)

        ttk.Label(parent, text="Model").pack(anchor="w", pady=(10, 2))
        ttk.Entry(parent, textvariable=self.model_var, width=38).pack(fill="x")

        ttk.Label(parent, text="Reasoning").pack(anchor="w", pady=(10, 2))
        ttk.Combobox(
            parent,
            textvariable=self.reasoning_var,
            values=REASONING_CHOICES,
            state="readonly",
            width=35,
        ).pack(fill="x")

        ttk.Checkbutton(parent, text="Enable web search", variable=self.web_search_var).pack(anchor="w", pady=(12, 4))
        ttk.Checkbutton(parent, text="Enable local file and shell tools", variable=self.local_tools_var).pack(anchor="w", pady=4)
        ttk.Checkbutton(parent, text="Auto-approve local tools", variable=self.auto_confirm_var).pack(anchor="w", pady=4)

        ttk.Separator(parent).pack(fill="x", pady=14)

        button_row = ttk.Frame(parent)
        button_row.pack(fill="x")
        ttk.Button(button_row, text="Save Settings", command=self._save_settings).pack(side="left")
        ttk.Button(button_row, text="New Session", command=self._new_session).pack(side="left", padx=(8, 0))

        ttk.Label(parent, textvariable=self.status_var, foreground="#245bdb", wraplength=280).pack(anchor="w", pady=(14, 0))

    def _build_chat_panel(self, parent: ttk.Frame) -> None:
        self.transcript = ScrolledText(parent, wrap="word", font=("Consolas", 10))
        self.transcript.grid(row=0, column=0, sticky="nsew")
        self.transcript.configure(state="disabled")

        input_frame = ttk.Frame(parent, padding=(0, 10, 0, 0))
        input_frame.grid(row=1, column=0, sticky="ew")
        input_frame.columnconfigure(0, weight=1)

        self.input_box = tk.Text(input_frame, height=5, wrap="word", font=("Segoe UI", 10))
        self.input_box.grid(row=0, column=0, sticky="ew")
        self.input_box.bind("<Control-Return>", self._send_from_shortcut)

        send_column = ttk.Frame(input_frame)
        send_column.grid(row=0, column=1, padx=(10, 0), sticky="ns")
        ttk.Button(send_column, text="Send", command=self.send_message).pack(fill="x")
        ttk.Button(send_column, text="Clear Chat", command=self._clear_transcript).pack(fill="x", pady=(8, 0))

    def _append_transcript(self, speaker: str, text: str, newline: bool = True) -> None:
        self.transcript.configure(state="normal")
        if self.transcript.index("end-1c") != "1.0":
            self.transcript.insert("end", "")
        prefix = f"{speaker}: " if speaker else ""
        self.transcript.insert("end", prefix + text)
        if newline:
            self.transcript.insert("end", "\n\n")
        self.transcript.see("end")
        self.transcript.configure(state="disabled")

    def _append_agent_chunk(self, text: str) -> None:
        self.transcript.configure(state="normal")
        if self.transcript.index("end-1c") == "1.0" or not self.transcript.get("end-2c", "end-1c").strip():
            self.transcript.insert("end", "Agent: ")
        self.transcript.insert("end", text)
        self.transcript.see("end")
        self.transcript.configure(state="disabled")

    def _finish_agent_message(self) -> None:
        self.transcript.configure(state="normal")
        self.transcript.insert("end", "\n\n")
        self.transcript.see("end")
        self.transcript.configure(state="disabled")

    def _browse_workspace(self) -> None:
        selected = filedialog.askdirectory(initialdir=self.workspace_var.get() or str(Path.cwd()))
        if selected:
            self.workspace_var.set(selected)

    def _apply_preset(self, *_args: object) -> None:
        preset = PRESETS.get(self.preset_var.get(), {})
        model = preset.get("model")
        reasoning = preset.get("reasoning_effort")
        if model:
            self.model_var.set(model)
        if reasoning:
            self.reasoning_var.set(reasoning)

    def _save_settings(self) -> None:
        payload = {
            "base_url": self.base_url_var.get().strip(),
            "workspace": self.workspace_var.get().strip(),
            "model": self.model_var.get().strip(),
            "reasoning_effort": self.reasoning_var.get().strip(),
            "preset": self.preset_var.get().strip(),
            "use_web_search": self.web_search_var.get(),
            "enable_local_tools": self.local_tools_var.get(),
            "auto_confirm_tools": self.auto_confirm_var.get(),
        }
        self.settings_file.parent.mkdir(parents=True, exist_ok=True)
        self.settings_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.status_var.set(f"Settings saved to {self.settings_file}")

    def _load_settings(self) -> None:
        if not self.settings_file.exists():
            self._append_transcript(
                "System",
                "Enter your API key or set OPENAI_API_KEY in the environment, then start chatting.",
            )
            return

        try:
            data = json.loads(self.settings_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.status_var.set("Could not read saved settings.")
            return

        self.base_url_var.set(data.get("base_url", ""))
        self.workspace_var.set(data.get("workspace", str(Path.cwd())))
        self.model_var.set(data.get("model", DEFAULT_MODEL))
        self.reasoning_var.set(data.get("reasoning_effort", DEFAULT_REASONING_EFFORT))
        self.preset_var.set(data.get("preset", "Balanced"))
        self.web_search_var.set(bool(data.get("use_web_search", False)))
        self.local_tools_var.set(bool(data.get("enable_local_tools", False)))
        self.auto_confirm_var.set(bool(data.get("auto_confirm_tools", False)))
        self.status_var.set("Loaded saved settings.")

    def _new_session(self) -> None:
        if self.current_agent is not None:
            self.current_agent.reset()
        elif self.session_file.exists():
            try:
                self.session_file.unlink()
            except OSError:
                pass
        self._append_transcript("System", "Started a new conversation session.")
        self.status_var.set("New session started.")

    def _clear_transcript(self) -> None:
        self.transcript.configure(state="normal")
        self.transcript.delete("1.0", "end")
        self.transcript.configure(state="disabled")

    def _send_from_shortcut(self, _event: tk.Event) -> str:
        self.send_message()
        return "break"

    @staticmethod
    def _is_official_openai_base_url(base_url: str | None) -> bool:
        if not base_url:
            return True
        lowered = base_url.strip().lower()
        return "openai.com" in lowered

    def _apply_compatibility_overrides(self) -> None:
        base_url = self.base_url_var.get().strip() or None
        if self._is_official_openai_base_url(base_url):
            return

        notes: list[str] = []

        if self.web_search_var.get():
            self.web_search_var.set(False)
            notes.append(
                "Web search has been turned off because third-party compatible endpoints "
                "usually do not support it reliably."
            )

        if self.local_tools_var.get():
            self.local_tools_var.set(False)
            self.auto_confirm_var.set(False)
            notes.append(
                "Local tools have been turned off for faster third-party chat. "
                "Enable them again only when you need file or shell actions."
            )

        if (
            self.preset_var.get().strip() == "Balanced"
            and self.reasoning_var.get().strip().lower() != DEFAULT_REASONING_EFFORT
        ):
            self.reasoning_var.set(DEFAULT_REASONING_EFFORT)
            notes.append("Reasoning was reset to low for faster third-party responses.")

        if notes:
            self._append_transcript("System", " ".join(notes))
            self.status_var.set("Applied third-party compatibility settings.")

    def _build_agent(self) -> ChatAgent:
        self._apply_compatibility_overrides()
        api_key = self.api_key_var.get().strip() or None
        base_url = self.base_url_var.get().strip() or None
        system_prompt = load_system_prompt()
        client = create_client(api_key=api_key, base_url=base_url)

        options = AgentOptions(
            model=self.model_var.get().strip() or DEFAULT_MODEL,
            reasoning_effort=self.reasoning_var.get().strip() or DEFAULT_REASONING_EFFORT,
            use_web_search=self.web_search_var.get(),
            enable_local_tools=self.local_tools_var.get(),
            auto_confirm_tools=self.auto_confirm_var.get(),
            workspace=Path(self.workspace_var.get().strip() or Path.cwd()),
            session_file=self.session_file,
            system_prompt=system_prompt,
        )
        agent = ChatAgent(
            client=client,
            options=options,
            confirm_callback=self._confirm_tool_action,
        )
        agent.load_session()
        return agent

    def send_message(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("Busy", "The agent is still working on the previous request.")
            return

        user_text = self.input_box.get("1.0", "end").strip()
        if not user_text:
            return

        self.input_box.delete("1.0", "end")
        self._append_transcript("You", user_text)
        self.status_var.set("Sending request...")

        try:
            self.current_agent = self._build_agent()
        except Exception as exc:
            self._append_transcript("System", f"Could not create the agent: {exc}")
            self.status_var.set("Failed to start.")
            return

        self.worker_thread = threading.Thread(
            target=self._run_agent_request,
            args=(user_text,),
            daemon=True,
        )
        self.worker_thread.start()

    def _run_agent_request(self, user_text: str) -> None:
        assert self.current_agent is not None
        try:
            self.event_queue.put(("agent_start", ""))
            reply = self.current_agent.ask(
                user_text=user_text,
                text_callback=lambda chunk: self.event_queue.put(("agent_chunk", chunk)),
                event_callback=lambda message: self.event_queue.put(("tool_event", message)),
            )
            if not reply:
                self.event_queue.put(("agent_chunk", "(empty response)"))
            self.event_queue.put(("agent_done", ""))
            self.event_queue.put(("status", "Ready"))
        except ConversationStateError as exc:
            self.event_queue.put(("system", f"Session reset: {exc}"))
            self.event_queue.put(("status", "Session reset."))
        except AuthenticationError:
            self.event_queue.put(("system", "Authentication failed. Check the API key and base URL."))
            self.event_queue.put(("status", "Authentication failed."))
        except RateLimitError as exc:
            self.event_queue.put(("system", f"Rate limit or quota issue: {exc}"))
            self.event_queue.put(("status", "Rate limited."))
        except APIError as exc:
            self.event_queue.put(("system", f"OpenAI API error: {exc}"))
            self.event_queue.put(("status", "API error."))
        except Exception as exc:
            self.event_queue.put(("system", f"Unexpected error: {exc}"))
            self.event_queue.put(("status", "Unexpected error."))

    def _drain_queue(self) -> None:
        try:
            while True:
                kind, payload = self.event_queue.get_nowait()
                if kind == "agent_start":
                    self.status_var.set("Agent is thinking...")
                elif kind == "agent_chunk":
                    self._append_agent_chunk(payload)
                elif kind == "agent_done":
                    self._finish_agent_message()
                elif kind == "tool_event":
                    self._append_transcript("System", payload)
                    self.status_var.set(payload)
                elif kind == "system":
                    self._append_transcript("System", payload)
                elif kind == "status":
                    self.status_var.set(payload)
        except Empty:
            pass
        self.root.after(80, self._drain_queue)

    def _confirm_tool_action(self, action: str, detail: str) -> bool:
        return messagebox.askyesno(
            title=action,
            message=f"{action}\n\n{detail}",
            parent=self.root,
        )


def main() -> None:
    root = tk.Tk()
    AgentDesktopApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
