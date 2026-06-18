"""
=============================================================================
  Claude Agent SDK — End-to-End Production Reference (2026)
=============================================================================

  Covers EVERY feature and pattern from the production guide:
    • Installation guard + env validation
    • query()         — one-shot headless agent loop
    • ClaudeSDKClient — multi-turn interactive client
    • ClaudeAgentOptions — full configuration surface
    • Built-in tools  — Read, Write, Edit, Bash, Glob, Grep, WebSearch, WebFetch
    • Permission modes — default / acceptEdits / plan / dontAsk / bypassPermissions
    • can_use_tool     — runtime path-jail callback
    • Custom in-process tools via @tool + create_sdk_mcp_server
    • External MCP servers — stdio (npx) + HTTP/SSE remote
    • MCP health check from the SystemMessage init
    • Agent Skills     — filesystem SKILL.md, setting_sources, skills filter
    • Subagents        — programmatic AgentDefinition, per-subagent model
    • Hooks            — PreToolUse, PostToolUse, SubagentStart safety guards
    • Sessions         — resume, continue_conversation, fork_session
    • Cost control     — max_turns, max_budget_usd, task_budget, ResultMessage logging
    • Model selection  — pinned IDs, fallback_model, alias reference
    • Provider routing — Bedrock, Vertex, Foundry via env vars
    • Structured output — output_format schema
    • Streaming message types — AssistantMessage, ToolUseBlock, ResultMessage, SystemMessage
    • FastAPI service   — /agent endpoint, /stream endpoint (SSE), /health
    • Dynamic permission change mid-session
    • Observability    — structured JSON logging, cost tracking, trace hook

  Quick start:
    pip install claude-agent-sdk fastapi uvicorn python-dotenv
    export ANTHROPIC_API_KEY="sk-ant-..."
    python claude_agent_sdk_production.py demo          # run standalone demo
    python claude_agent_sdk_production.py serve         # start FastAPI service

  Python 3.10+ required.
=============================================================================
"""

from __future__ import annotations

# ─────────────────────────────────────────────
#  stdlib
# ─────────────────────────────────────────────
import asyncio
import json
import logging
import os
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Literal, Optional

# ─────────────────────────────────────────────
#  third-party — checked at runtime below
# ─────────────────────────────────────────────
try:
    from dotenv import load_dotenv          # pip install python-dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional; env vars can come from the shell

# ─────────────────────────────────────────────
#  Claude Agent SDK
#  pip install claude-agent-sdk
# ─────────────────────────────────────────────
try:
    from claude_agent_sdk import (          # type: ignore[import]
        # Entry points
        query,
        ClaudeSDKClient,

        # Options & definitions
        ClaudeAgentOptions,
        AgentDefinition,

        # Message types returned by the stream
        AssistantMessage,
        HumanMessage,
        ResultMessage,
        SystemMessage,

        # Block types inside AssistantMessage.content
        TextBlock,
        ToolUseBlock,
        ToolResultBlock,

        # Custom-tool plumbing (in-process MCP)
        tool,
        create_sdk_mcp_server,

        # Hook machinery
        HookMatcher,
    )
except ImportError as exc:
    print(
        "\n[ERROR] claude-agent-sdk is not installed.\n"
        "  pip install claude-agent-sdk\n"
        f"  Detail: {exc}\n"
    )
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 1 — Structured logging
#  Production agents fan out into many turns/subagents; plain print() is
#  not enough.  We emit JSON lines so any log aggregator can parse them.
# ═══════════════════════════════════════════════════════════════════════════

class JsonFormatter(logging.Formatter):
    """Emit one JSON object per log record."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts":    datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "msg":   record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        for key in ("session_id", "run_id", "tool", "cost_usd", "turns"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        return json.dumps(payload)


def build_logger(name: str = "agent") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


log = build_logger()


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 2 — Environment / API key validation
# ═══════════════════════════════════════════════════════════════════════════

def validate_environment() -> None:
    """
    Raise early with a clear message if required env vars are missing.
    Supports three provider routes:
      1. Direct Anthropic API  → ANTHROPIC_API_KEY
      2. Amazon Bedrock        → CLAUDE_CODE_USE_ANTHROPIC_AWS=1
                                 ANTHROPIC_AWS_WORKSPACE_ID
      3. Google Vertex AI      → CLAUDE_CODE_USE_VERTEX=1
                                 (credentials from gcloud)
    """
    use_aws     = os.getenv("CLAUDE_CODE_USE_ANTHROPIC_AWS")
    use_vertex  = os.getenv("CLAUDE_CODE_USE_VERTEX")
    api_key     = os.getenv("ANTHROPIC_API_KEY")

    if use_aws:
        if not os.getenv("ANTHROPIC_AWS_WORKSPACE_ID"):
            raise EnvironmentError(
                "ANTHROPIC_AWS_WORKSPACE_ID must be set when using Claude Platform on AWS."
            )
        log.info("Provider: Claude Platform on AWS (Claude Code SDK routes to Bedrock)")
    elif use_vertex:
        log.info("Provider: Google Vertex AI")
    elif api_key:
        log.info("Provider: Anthropic API (direct)")
    else:
        raise EnvironmentError(
            "Set ANTHROPIC_API_KEY, or configure CLAUDE_CODE_USE_ANTHROPIC_AWS / "
            "CLAUDE_CODE_USE_VERTEX for cloud-provider routing."
        )


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 3 — Cost + usage tracker
#  Accumulates per-session spend from ResultMessage; feeds structured logs.
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SessionStats:
    session_id:   str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    turns:        int = 0
    tool_calls:   int = 0
    cost_usd:     float = 0.0
    started_at:   float = field(default_factory=time.monotonic)
    final_result: Optional[str] = None
    subtype:      str = "pending"

    def elapsed(self) -> float:
        return round(time.monotonic() - self.started_at, 2)

    def absorb_result(self, msg: ResultMessage) -> None:
        self.subtype = getattr(msg, "subtype", "unknown")
        self.cost_usd = getattr(msg, "total_cost_usd", 0.0) or self.cost_usd
        self.final_result = getattr(msg, "result", None)

    def log_summary(self) -> None:
        log.info(
            "Session complete",
            extra={
                "session_id": self.session_id,
                "turns":      self.turns,
                "tool_calls": self.tool_calls,
                "cost_usd":   round(self.cost_usd, 6),
                "elapsed_s":  self.elapsed(),
                "subtype":    self.subtype,
            },
        )


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 4 — Custom in-process tools  (@tool + create_sdk_mcp_server)
#
#  These are MCP servers running INSIDE your Python process — no subprocess,
#  no network.  When Claude invokes one, the CLI sends a control request and
#  the SDK calls your function directly.  Faster than external MCP servers.
# ═══════════════════════════════════════════════════════════════════════════

# ── 4a. Simulated order-lookup tool (replace with your real DB call) ──────

@tool(
    "order_status",
    "Look up the status and details of an order by its order ID.",
    {"order_id": str},
)
async def tool_order_status(args: dict) -> dict:
    """In a real app this would hit your database / API."""
    order_id = args.get("order_id", "UNKNOWN")
    # Simulated data
    db: dict[str, dict] = {
        "ORD-001": {"status": "shipped",   "eta": "2026-06-20", "items": 3},
        "ORD-002": {"status": "pending",   "eta": "2026-06-25", "items": 1},
        "ORD-003": {"status": "delivered", "eta": "2026-06-17", "items": 7},
    }
    row = db.get(order_id, {"status": "not_found"})
    return {"content": [{"type": "text", "text": json.dumps(row)}]}


# ── 4b. Environment metadata tool ─────────────────────────────────────────

@tool(
    "get_environment",
    "Return current runtime environment metadata (hostname, Python version, CWD).",
    {},
)
async def tool_get_environment(_args: dict) -> dict:
    import platform
    info = {
        "python":   platform.python_version(),
        "hostname": platform.node(),
        "cwd":      str(Path.cwd()),
        "ts_utc":   datetime.now(timezone.utc).isoformat(),
    }
    return {"content": [{"type": "text", "text": json.dumps(info)}]}


# ── 4c. Register both tools in a single in-process MCP server ─────────────

IN_PROCESS_MCP = create_sdk_mcp_server(
    name="internal",
    version="1.0.0",
    tools=[tool_order_status, tool_get_environment],
)


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 5 — Permission callbacks
#
#  can_use_tool fires for any tool call not already resolved by allow/deny
#  rules or the permission mode.  Use it for runtime decisions: path jails,
#  argument inspection, per-user ACLs, rate limiting.
# ═══════════════════════════════════════════════════════════════════════════

# Configurable root that Read/Write/Edit are allowed to touch.
ALLOWED_ROOT: str = str(Path.cwd())


async def can_use_tool_callback(
    tool_name: str,
    tool_input: dict,
    context: Any,
) -> dict:
    """
    Runtime permission gate.  Returns:
      {"behavior": "allow"}  — proceed
      {"behavior": "deny", "message": "..."} — block, surface reason to Claude
    """

    # ── path jail for file tools ──────────────────────────────────────────
    if tool_name in ("Read", "Write", "Edit"):
        raw_path = tool_input.get("file_path") or tool_input.get("path", "")
        try:
            resolved = str(Path(raw_path).resolve())
        except Exception:
            resolved = raw_path

        if not resolved.startswith(ALLOWED_ROOT):
            log.warning(
                "can_use_tool: path outside jail",
                extra={"tool": tool_name, "path": resolved},
            )
            return {
                "behavior": "deny",
                "message":  f"Path '{resolved}' is outside the allowed root '{ALLOWED_ROOT}'.",
            }

        # Block reads of secret files wherever they live
        for secret_pat in (".env", ".pem", "id_rsa", "credentials.json", ".netrc"):
            if secret_pat in resolved:
                return {"behavior": "deny", "message": f"Blocked: secret file pattern '{secret_pat}'."}

    # ── Bash: argument inspection (belt-and-braces; hooks also guard this) ─
    if tool_name == "Bash":
        cmd = tool_input.get("command", "")
        dangerous = ["rm -rf /", "mkfs", "dd if=/dev/", "> /dev/sd"]
        for pattern in dangerous:
            if pattern in cmd:
                return {"behavior": "deny", "message": f"Blocked dangerous pattern: '{pattern}'."}

    return {"behavior": "allow"}


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 6 — Hooks
#
#  Hooks run FIRST in the permission evaluation order and can deny a call
#  before allow/deny rules are consulted.  They also provide observability
#  (audit logging) and can rewrite tool output via PostToolUse.
# ═══════════════════════════════════════════════════════════════════════════

async def hook_pre_tool_bash(
    input_data: dict,
    tool_use_id: Optional[str],
    context: Any,
) -> dict:
    """
    PreToolUse hook scoped to Bash.
    Blocks the call if destructive patterns are detected.
    Returns {} to allow, or the deny structure to block.
    """
    cmd = input_data.get("tool_input", {}).get("command", "")
    blocked = ["rm -rf", "| sh", "| bash", "curl | ", "dd if=", "mkfs", "> /dev/sd"]
    for pattern in blocked:
        if pattern in cmd:
            log.warning(
                "Hook blocked destructive bash",
                extra={"tool": "Bash", "pattern": pattern},
            )
            return {
                "hookSpecificOutput": {
                    "hookEventName":          "PreToolUse",
                    "permissionDecision":     "deny",
                    "permissionDecisionReason": f"Blocked: pattern '{pattern}' is disallowed in production.",
                }
            }
    # Log all bash commands for the audit trail
    log.debug("Bash command approved", extra={"tool": "Bash", "cmd": cmd[:200]})
    return {}


async def hook_post_tool_redact(
    input_data: dict,
    tool_use_id: Optional[str],
    context: Any,
) -> dict:
    """
    PostToolUse hook (all tools).
    Redacts any output that looks like an API key before the model sees it.
    Return updatedToolOutput to override what Claude receives.
    """
    output: str = input_data.get("tool_result", {}).get("content", "") or ""
    if isinstance(output, list):
        # content can be a list of blocks
        output = " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in output)

    # Simple heuristic: 40+ char hex-ish tokens surrounded by boundaries
    import re
    redacted = re.sub(r"sk-ant-[A-Za-z0-9\-_]{20,}", "[REDACTED_KEY]", output)
    if redacted != output:
        log.warning("PostToolUse: redacted potential API key in tool output")
        return {"updatedToolOutput": redacted}
    return {}


async def hook_subagent_start(
    input_data: dict,
    tool_use_id: Optional[str],
    context: Any,
) -> dict:
    """
    SubagentStart hook — enforce a spawn budget.
    In production you'd store spawn count per session_id in Redis / DB.
    """
    agent_name = input_data.get("agent_name", "unknown")
    log.info("Subagent starting", extra={"agent": agent_name})
    # (Budget enforcement would go here)
    return {}


async def hook_audit_any(
    input_data: dict,
    tool_use_id: Optional[str],
    context: Any,
) -> dict:
    """Universal PreToolUse audit logger — fires for every tool call."""
    log.info(
        "Tool call",
        extra={
            "tool": input_data.get("tool_name", "unknown"),
            "id":   tool_use_id,
        },
    )
    return {}  # allow — audit only


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 7 — options factories
#
#  We build ClaudeAgentOptions through factory functions so you can see
#  every field and its intent.  Programmatic options always override any
#  filesystem settings when both are present.
# ═══════════════════════════════════════════════════════════════════════════

# ── 7a. Headless / locked-down options (CI, background jobs) ──────────────

def make_headless_options(
    *,
    cwd: str = str(Path.cwd()),
    max_turns: int = 20,
    max_budget_usd: float = 1.00,
    model: str = "claude-sonnet-4-6",           # pinned
    fallback_model: str = "claude-haiku-4-5-20251001",
    system_prompt: str = "You are a careful production agent. Be concise, precise, and never destructive.",
) -> ClaudeAgentOptions:
    """
    Locked-down options for headless / unattended use.
    • dontAsk  → any tool not pre-approved is DENIED (never hangs waiting)
    • allowlist explicitly lists every permitted tool
    • disallowlist removes truly dangerous tools entirely
    • can_use_tool enforces a path jail and argument inspection
    • Hooks provide an audit trail and a second layer of destructive-bash blocking
    """
    return ClaudeAgentOptions(
        # ── identity ──────────────────────────────────────────────────────
        model=model,
        fallback_model=fallback_model,
        system_prompt=system_prompt,

        # ── filesystem scope ──────────────────────────────────────────────
        cwd=cwd,
        # add_dirs=["/other/dir"]  # extra dirs the agent may touch

        # ── tool surface ──────────────────────────────────────────────────
        # allowed_tools = PRE-APPROVAL list (unlisted → dontAsk = deny)
        allowed_tools=[
            "Read",
            "Glob",
            "Grep",
            "Skill",                            # filesystem Skills
            "Agent",                            # invoke subagents
            "mcp__internal__order_status",      # custom in-process tool
            "mcp__internal__get_environment",   # custom in-process tool
        ],
        # disallowed_tools = HARD DISABLE (model can't use these AT ALL)
        disallowed_tools=["Write", "Edit", "Bash", "WebFetch"],

        # ── permission mode ───────────────────────────────────────────────
        # dontAsk: anything not pre-approved above → DENIED, never prompts
        permission_mode="dontAsk",

        # ── runtime permission callback ───────────────────────────────────
        can_use_tool=can_use_tool_callback,

        # ── custom + external MCP servers ─────────────────────────────────
        mcp_servers={
            "internal": IN_PROCESS_MCP,         # in-process, no subprocess
            # External HTTP MCP server (commented out — enable when available):
            # "docs": {
            #     "type": "http",
            #     "url":  "https://mcp.docs.example.com",
            #     "headers": {"Authorization": f"Bearer {os.getenv('DOCS_TOKEN', '')}"},
            # },
            # External stdio MCP server (npx example):
            # "github": {
            #     "command": "npx",
            #     "args": ["-y", "@modelcontextprotocol/server-github"],
            #     "env": {"GITHUB_TOKEN": os.getenv("GITHUB_TOKEN", "")},
            # },
        },

        # ── subagents ─────────────────────────────────────────────────────
        agents={
            "researcher": AgentDefinition(
                description="Read-only agent that searches the codebase and docs to answer factual questions.",
                prompt=(
                    "You are a read-only research assistant. "
                    "Answer only from what you find in the repository. "
                    "Never write or modify files."
                ),
                tools=["Read", "Grep", "Glob"],
                model="haiku",                  # cheaper model for subtasks
            ),
            "summariser": AgentDefinition(
                description="Summarises a set of files or search results into a concise report.",
                prompt="You are a technical writer. Produce a clear, structured summary.",
                tools=["Read"],
                model="sonnet",
            ),
        },

        # ── skills ────────────────────────────────────────────────────────
        # setting_sources is REQUIRED to load SKILL.md files from disk.
        # Without it, no skills, no CLAUDE.md, no filesystem hooks load.
        setting_sources=["project"],            # loads .claude/skills/ & CLAUDE.md
        skills="all",                           # or ["pdf", "code-review"]

        # ── bounds ────────────────────────────────────────────────────────
        max_turns=max_turns,
        max_budget_usd=max_budget_usd,

        # ── hooks ─────────────────────────────────────────────────────────
        hooks={
            "PreToolUse": [
                HookMatcher(hooks=[hook_audit_any]),           # fires on ALL tools
                HookMatcher(matcher="Bash", hooks=[hook_pre_tool_bash]),  # Bash-only guard
            ],
            "PostToolUse": [
                HookMatcher(hooks=[hook_post_tool_redact]),    # secret redaction
            ],
            "SubagentStart": [
                HookMatcher(hooks=[hook_subagent_start]),      # spawn budget
            ],
        },

        # ── structured output ─────────────────────────────────────────────
        # Uncomment to receive a JSON-schema-conformant result alongside text.
        # output_format={
        #     "type": "object",
        #     "properties": {
        #         "summary":  {"type": "string"},
        #         "issues":   {"type": "array", "items": {"type": "string"}},
        #         "severity": {"type": "string", "enum": ["low", "medium", "high"]},
        #     },
        #     "required": ["summary", "severity"],
        # },

        # ── env vars forwarded to the CLI subprocess ──────────────────────
        env={
            "CLAUDE_AGENT_LOG_LEVEL": "info",
        },
    )


# ── 7b. Interactive / permissive options (local development only) ─────────

def make_interactive_options(
    *,
    cwd: str = str(Path.cwd()),
    max_turns: int = 30,
) -> ClaudeAgentOptions:
    """
    Looser options for local interactive sessions.
    acceptEdits auto-approves file edits; Bash is allowed (with the hook guard).
    NOT suitable for production / customer-facing code.
    """
    return ClaudeAgentOptions(
        model="claude-sonnet-4-6",
        system_prompt="You are a senior software engineer helping with local development.",
        cwd=cwd,
        allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep", "WebSearch"],
        permission_mode="acceptEdits",
        can_use_tool=can_use_tool_callback,
        max_turns=max_turns,
        hooks={
            "PreToolUse": [
                HookMatcher(hooks=[hook_audit_any]),
                HookMatcher(matcher="Bash", hooks=[hook_pre_tool_bash]),
            ],
        },
    )


# ── 7c. Plan-only options (proposal / review, no file changes) ────────────

def make_plan_options(*, cwd: str = str(Path.cwd())) -> ClaudeAgentOptions:
    """
    Plan mode: Claude can read and propose, but NEVER writes or executes.
    File edits are never auto-approved even if allow rules match.
    """
    return ClaudeAgentOptions(
        model="claude-opus-4-8",        # use the most capable model for planning
        system_prompt="Propose a detailed implementation plan. Do NOT make any changes.",
        cwd=cwd,
        allowed_tools=["Read", "Glob", "Grep"],
        permission_mode="plan",
        max_turns=15,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 8 — Message-stream consumer
#
#  Every SDK call returns an async iterator.  This function shows how to
#  handle every message type correctly.
# ═══════════════════════════════════════════════════════════════════════════

async def consume_stream(
    stream: AsyncIterator,
    stats: SessionStats,
    *,
    verbose: bool = True,
) -> str:
    """
    Drain the message stream, print/log each meaningful event, and return
    the final plain-text result.  Handles:
      SystemMessage  — init (MCP health), billing metadata
      AssistantMessage — TextBlock, ToolUseBlock, ToolResultBlock
      HumanMessage   — echoed user turns in multi-turn sessions
      ResultMessage  — terminal event; subtype success / error_during_execution
    """
    result_text = ""

    async for message in stream:

        # ── System messages (init, MCP status, billing) ───────────────────
        if isinstance(message, SystemMessage):
            if getattr(message, "subtype", None) == "init":
                # Check MCP server health immediately
                mcp_servers = getattr(message, "mcp_servers", []) or []
                for srv in mcp_servers:
                    status = srv.get("status") if isinstance(srv, dict) else getattr(srv, "status", None)
                    name   = srv.get("name")   if isinstance(srv, dict) else getattr(srv, "name", "?")
                    if status != "connected":
                        log.error(f"MCP server not connected: {name!r} → {status!r}")
                    else:
                        log.info(f"MCP server connected: {name!r}")
            continue

        # ── Assistant messages (Claude's text, tool calls, tool results) ──
        if isinstance(message, AssistantMessage):
            stats.turns += 1
            for block in getattr(message, "content", []):

                if isinstance(block, TextBlock):
                    if verbose:
                        print(f"\n  [Claude] {block.text}")

                elif isinstance(block, ToolUseBlock):
                    stats.tool_calls += 1
                    tool_name = getattr(block, "name", "?")
                    tool_args = getattr(block, "input", {})
                    log.info(
                        "Tool invocation",
                        extra={"tool": tool_name, "args_keys": list(tool_args.keys())},
                    )
                    if verbose:
                        print(f"\n  [Tool →] {tool_name}({json.dumps(tool_args, indent=2)[:160]}…)")

                elif isinstance(block, ToolResultBlock):
                    if verbose:
                        content = getattr(block, "content", "")
                        if isinstance(content, list):
                            content = " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
                        print(f"\n  [Tool ←] {str(content)[:200]}")
            continue

        # ── Echoed human turns (visible in multi-turn ClaudeSDKClient) ────
        if isinstance(message, HumanMessage):
            log.debug("Human turn echoed", extra={"session_id": stats.session_id})
            continue

        # ── Terminal message — always present exactly once at the end ──────
        if isinstance(message, ResultMessage):
            stats.absorb_result(message)
            result_text = stats.final_result or ""
            if stats.subtype == "success":
                if verbose:
                    print(f"\n  ✓ Done ({stats.turns} turns, ${stats.cost_usd:.5f})")
            else:
                log.error(
                    "Agent run failed",
                    extra={"subtype": stats.subtype, "session_id": stats.session_id},
                )
            stats.log_summary()
            continue

    return result_text


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 9 — query() one-shot headless runner
#
#  Use this for background jobs, CI, request/response HTTP handlers.
# ═══════════════════════════════════════════════════════════════════════════

async def run_headless(
    prompt: str,
    *,
    cwd: str = str(Path.cwd()),
    max_turns: int = 20,
    max_budget_usd: float = 1.00,
    verbose: bool = True,
) -> SessionStats:
    """
    Runs one prompt through the locked-down headless options.
    Returns a populated SessionStats so callers can branch on subtype / cost.
    """
    stats = SessionStats()
    options = make_headless_options(
        cwd=cwd,
        max_turns=max_turns,
        max_budget_usd=max_budget_usd,
    )

    log.info(
        "Starting headless run",
        extra={"session_id": stats.session_id, "prompt": prompt[:120]},
    )

    try:
        stream = query(prompt=prompt, options=options)
        await consume_stream(stream, stats, verbose=verbose)
    except Exception as exc:
        log.error(f"Headless run crashed: {exc}", exc_info=True)
        stats.subtype = "crashed"

    return stats


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 10 — ClaudeSDKClient multi-turn interactive runner
#
#  Use this for conversational flows that need context across turns:
#  chatbots, REPL-style agents, guided workflows.
#  Also demonstrates:
#    • dynamic permission mode change mid-session (start cautious, loosen)
#    • session resume / continue_conversation
# ═══════════════════════════════════════════════════════════════════════════

async def run_interactive_session(
    turns: list[str],
    *,
    cwd: str = str(Path.cwd()),
    verbose: bool = True,
) -> list[SessionStats]:
    """
    Sends multiple prompts through a single ClaudeSDKClient, preserving context.
    Returns one SessionStats per turn.
    """
    all_stats: list[SessionStats] = []
    options = make_interactive_options(cwd=cwd)

    async with ClaudeSDKClient(options=options) as client:
        for i, prompt in enumerate(turns):
            stats = SessionStats()
            log.info(
                "Interactive turn",
                extra={"session_id": stats.session_id, "turn": i + 1, "prompt": prompt[:80]},
            )

            # ── Dynamically loosen permissions after the first turn ────────
            # Real use-case: start in read-only mode, switch to acceptEdits
            # only after Claude has proposed its plan and the user approves.
            if i == 1:
                await client.set_permission_mode("acceptEdits")
                log.info("Permission mode changed to acceptEdits mid-session")

            await client.query(prompt)

            stream = client.receive_response()
            await consume_stream(stream, stats, verbose=verbose)
            all_stats.append(stats)

    return all_stats


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 11 — Session continuity helpers
#  resume / continue_conversation / fork_session
# ═══════════════════════════════════════════════════════════════════════════

async def run_resumed_session(
    prompt: str,
    session_id: str,
    *,
    fork: bool = False,
    cwd: str = str(Path.cwd()),
) -> SessionStats:
    """
    Continue or fork a prior session.
    Sessions are JSONL files local to the machine that created them.
    For cross-host resumption, ship the session file or capture results
    as application state and start fresh.

    fork=True  → creates a new branch (fork_session); original is untouched
    fork=False → continues in-place (continue_conversation / resume)
    """
    stats = SessionStats()
    base = make_headless_options(cwd=cwd)
    # Patch the session options in-place
    if fork:
        base.resume             = session_id
        base.fork_session       = True
        log.info("Forking session", extra={"parent_session": session_id})
    else:
        base.resume             = session_id
        base.continue_conversation = True
        log.info("Resuming session", extra={"session_id": session_id})

    stream = query(prompt=prompt, options=base)
    await consume_stream(stream, stats)
    return stats


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 12 — Structured output
#  When output_format is set, the ResultMessage includes a parsed object
#  alongside the plain text.  Ideal for machine-readable pipelines.
# ═══════════════════════════════════════════════════════════════════════════

async def run_structured_output(prompt: str, *, cwd: str = str(Path.cwd())) -> dict:
    """
    Ask Claude to return a JSON object conforming to a schema.
    The parsed result is in message.structured_output (when available).
    """
    stats = SessionStats()
    options = ClaudeAgentOptions(
        model="claude-sonnet-4-6",
        system_prompt="Return ONLY valid JSON conforming to the provided schema.",
        cwd=cwd,
        allowed_tools=["Read", "Glob", "Grep"],
        permission_mode="dontAsk",
        max_turns=8,
        output_format={
            "type": "object",
            "properties": {
                "summary":  {"type": "string"},
                "findings": {"type": "array", "items": {"type": "string"}},
                "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
            },
            "required": ["summary", "severity"],
        },
    )

    structured: dict = {}
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, ResultMessage):
            stats.absorb_result(message)
            # structured output sits alongside the plain result
            structured = getattr(message, "structured_output", {}) or {}
            break

    stats.log_summary()
    return structured


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 13 — Demo runner (standalone, no HTTP server)
#  Run with:  python claude_agent_sdk_production.py demo
# ═══════════════════════════════════════════════════════════════════════════

DEMO_PROMPTS = [
    # query() / headless — uses the in-process tool
    (
        "run_headless",
        "Fetch the environment metadata using the get_environment tool, "
        "then list the top-level Python files in the current directory.",
    ),
    # query() / plan mode — read-only proposal
    (
        "run_plan",
        "Read all .py files in this directory and propose a refactoring plan "
        "to improve import organisation. Do NOT make any changes.",
    ),
    # Structured output
    (
        "run_structured",
        "Read all .py files in the current directory and return a JSON report "
        "with a one-sentence summary, a list of findings, and a severity rating.",
    ),
    # Order-status in-process tool demo
    (
        "run_headless",
        "What is the status of order ORD-001? Use the order_status tool.",
    ),
]


async def _demo() -> None:
    print("\n" + "═" * 68)
    print("  Claude Agent SDK — Production Reference Demo (2026)")
    print("═" * 68 + "\n")

    for kind, prompt in DEMO_PROMPTS:
        sep = "─" * 64
        print(f"\n{sep}\n  {kind.upper()}\n  {prompt[:80]}\n{sep}")

        if kind == "run_headless":
            stats = await run_headless(prompt, verbose=True)
            print(f"\n  [stats] turns={stats.turns}  cost=${stats.cost_usd:.5f}  subtype={stats.subtype}")

        elif kind == "run_plan":
            stats = SessionStats()
            options = make_plan_options()
            stream = query(prompt=prompt, options=options)
            result = await consume_stream(stream, stats, verbose=True)
            print(f"\n  [plan result preview]\n  {result[:400]}")

        elif kind == "run_structured":
            result = await run_structured_output(prompt)
            print(f"\n  [structured output]\n  {json.dumps(result, indent=2)}")


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 14 — FastAPI production service
#
#  Three endpoints:
#    POST /agent       — one-shot JSON request/response
#    POST /stream      — Server-Sent Events (SSE) streaming
#    GET  /health      — readiness probe
#
#  Run with:  python claude_agent_sdk_production.py serve
#         or: uvicorn claude_agent_sdk_production:app --port 8080
# ═══════════════════════════════════════════════════════════════════════════

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import StreamingResponse, JSONResponse
    from pydantic import BaseModel as PydanticModel, Field
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False


if _FASTAPI_AVAILABLE:

    app = FastAPI(
        title="Claude Agent SDK — Production Service",
        description="Headless agent endpoint with full feature coverage.",
        version="1.0.0",
    )

    # ── Request / response models ─────────────────────────────────────────

    class AgentRequest(PydanticModel):
        prompt:         str   = Field(..., min_length=1, max_length=4000)
        max_turns:      int   = Field(default=20, ge=1, le=60)
        max_budget_usd: float = Field(default=1.00, gt=0, le=10.0)
        mode:           Literal["headless", "plan"] = "headless"
        session_id:     Optional[str] = None   # resume a prior session

    class AgentResponse(PydanticModel):
        result:     Optional[str]
        subtype:    str
        turns:      int
        tool_calls: int
        cost_usd:   float
        elapsed_s:  float
        session_id: str

    # ── POST /agent ───────────────────────────────────────────────────────

    @app.post("/agent", response_model=AgentResponse)
    async def endpoint_agent(req: AgentRequest) -> AgentResponse:
        """
        One-shot agent endpoint.  Returns when the agent loop finishes.
        Use /stream for progressive output.
        """
        stats = SessionStats()

        if req.mode == "plan":
            options = make_plan_options()
        else:
            options = make_headless_options(
                max_turns=req.max_turns,
                max_budget_usd=req.max_budget_usd,
            )

        if req.session_id:
            options.resume = req.session_id
            options.continue_conversation = True

        try:
            stream = query(prompt=req.prompt, options=options)
            await consume_stream(stream, stats, verbose=False)
        except Exception as exc:
            log.error("Agent endpoint failed", exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

        return AgentResponse(
            result=stats.final_result,
            subtype=stats.subtype,
            turns=stats.turns,
            tool_calls=stats.tool_calls,
            cost_usd=round(stats.cost_usd, 6),
            elapsed_s=stats.elapsed(),
            session_id=stats.session_id,
        )

    # ── POST /stream  (SSE) ───────────────────────────────────────────────

    @app.post("/stream")
    async def endpoint_stream(req: AgentRequest) -> StreamingResponse:
        """
        Server-Sent Events endpoint.  Clients receive each assistant text
        chunk, tool-call events, and a final [DONE] event with cost data.
        """
        options = make_headless_options(
            max_turns=req.max_turns,
            max_budget_usd=req.max_budget_usd,
        )

        async def sse_generator() -> AsyncIterator[str]:
            stats = SessionStats()
            try:
                async for message in query(prompt=req.prompt, options=options):

                    if isinstance(message, AssistantMessage):
                        for block in getattr(message, "content", []):
                            if isinstance(block, TextBlock):
                                payload = json.dumps({"type": "text", "text": block.text})
                                yield f"data: {payload}\n\n"
                            elif isinstance(block, ToolUseBlock):
                                payload = json.dumps({
                                    "type": "tool_call",
                                    "name": getattr(block, "name", "?"),
                                })
                                yield f"data: {payload}\n\n"

                    elif isinstance(message, ResultMessage):
                        stats.absorb_result(message)
                        done_payload = json.dumps({
                            "type":     "done",
                            "subtype":  stats.subtype,
                            "cost_usd": round(stats.cost_usd, 6),
                            "turns":    stats.turns,
                        })
                        yield f"data: {done_payload}\n\n"
                        break

            except Exception as exc:
                err = json.dumps({"type": "error", "message": str(exc)})
                yield f"data: {err}\n\n"

        return StreamingResponse(
            sse_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # ── GET /health ───────────────────────────────────────────────────────

    @app.get("/health")
    async def endpoint_health() -> JSONResponse:
        """Kubernetes / ECS readiness probe."""
        return JSONResponse({
            "status":    "ok",
            "ts_utc":    datetime.now(timezone.utc).isoformat(),
            "api_key":   bool(os.getenv("ANTHROPIC_API_KEY")),
            "provider":  (
                "aws"    if os.getenv("CLAUDE_CODE_USE_ANTHROPIC_AWS") else
                "vertex" if os.getenv("CLAUDE_CODE_USE_VERTEX")        else
                "direct"
            ),
        })


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 15 — Provider routing reference (env-var only, no code changes)
# ═══════════════════════════════════════════════════════════════════════════
#
#  Switch providers by setting env vars BEFORE running the script:
#
#  ┌──────────────────────────────────┬─────────────────────────────────────┐
#  │ Provider                         │ Required env vars                   │
#  ├──────────────────────────────────┼─────────────────────────────────────┤
#  │ Anthropic API (direct)           │ ANTHROPIC_API_KEY                   │
#  │ Claude Platform on AWS           │ CLAUDE_CODE_USE_ANTHROPIC_AWS=1     │
#  │                                  │ ANTHROPIC_AWS_WORKSPACE_ID          │
#  │                                  │ + AWS credentials (env/profile)     │
#  │ Google Vertex AI                 │ CLAUDE_CODE_USE_VERTEX=1            │
#  │                                  │ + gcloud credentials                │
#  │ Amazon Bedrock (standalone)      │ Per Claude Code Bedrock docs        │
#  │ Microsoft Foundry                │ Per Foundry docs                    │
#  └──────────────────────────────────┴─────────────────────────────────────┘
#
#  Model IDs differ per provider.  Pin them explicitly:
#    ANTHROPIC_DEFAULT_OPUS_MODEL   = "claude-opus-4-8"
#    ANTHROPIC_DEFAULT_SONNET_MODEL = "claude-sonnet-4-6"
#    ANTHROPIC_DEFAULT_HAIKU_MODEL  = "claude-haiku-4-5-20251001"
#
#  On Bedrock / Vertex / Foundry the 'opus' alias may lag to an older model
#  — always pin a full ID for production.


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 16 — Model selection reference
# ═══════════════════════════════════════════════════════════════════════════
#
#  Model            │ API ID                        │ $/Mtok in│ out │ Use for
#  ─────────────────┼───────────────────────────────┼──────────┼─────┼──────────────────────────
#  Claude Opus 4.8  │ claude-opus-4-8               │ $5       │ $25 │ Complex reasoning, long agentic tasks, 1M ctx
#  Claude Sonnet 4.6│ claude-sonnet-4-6             │ $3       │ $15 │ DEFAULT: ~90% of tasks, best economics
#  Claude Haiku 4.5 │ claude-haiku-4-5-20251001     │ $1       │ $5  │ High-volume, latency-sensitive, subtasks
#  Claude Opus 4.7  │ claude-opus-4-7               │ —        │ —   │ Prior Opus; AWS alias 'opus' resolves here
#  Claude Fable 5   │ (frontier tier)               │ $10      │ $50 │ Hardest long-horizon jobs
#
#  Aliases  (resolve per-provider, change over time — PIN for prod):
#    "opus"     → Opus 4.8  (Anthropic API) / Opus 4.7 (AWS) / Opus 4.6 (Bedrock/Vertex)
#    "sonnet"   → Sonnet 4.6 (Anthropic API, AWS)
#    "haiku"    → Haiku 4.5
#    "opusplan" → Opus for planning, Sonnet for execution (hybrid cost saving)
#
#  Per-subagent models are set via AgentDefinition(model="sonnet"|"haiku"|"opus"|"inherit")


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 17 — Entrypoint
# ═══════════════════════════════════════════════════════════════════════════

def _print_usage() -> None:
    print(
        "\nUsage:\n"
        "  python claude_agent_sdk_production.py demo     # run standalone feature demo\n"
        "  python claude_agent_sdk_production.py serve    # start FastAPI HTTP service\n"
        "  python claude_agent_sdk_production.py check    # validate env only\n"
    )


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"

    # Always validate env first
    try:
        validate_environment()
    except EnvironmentError as e:
        print(f"\n[ENV ERROR] {e}\n")
        sys.exit(1)

    if cmd == "check":
        print("\n✓ Environment OK — ready to run.\n")

    elif cmd == "demo":
        asyncio.run(_demo())

    elif cmd == "serve":
        if not _FASTAPI_AVAILABLE:
            print("[ERROR] FastAPI not installed.  pip install fastapi uvicorn")
            sys.exit(1)
        try:
            import uvicorn  # type: ignore[import]
        except ImportError:
            print("[ERROR] uvicorn not installed.  pip install uvicorn")
            sys.exit(1)

        print("\n  Claude Agent SDK — Production Service")
        print("  Listening on http://0.0.0.0:8080")
        print("  Docs:          http://0.0.0.0:8080/docs\n")
        uvicorn.run(
            "claude_agent_sdk_production:app",
            host="0.0.0.0",
            port=8080,
            reload=False,       # never True in production
            log_level="info",
        )

    else:
        print(f"[ERROR] Unknown command: {cmd!r}")
        _print_usage()
        sys.exit(1)
