"""Persistent MCP client sessions driven from synchronous code.

The gate loop alternates between blocking GPU generation and concurrent MCP tool
calls. Rather than reconnecting per call (the mistake that makes MCP-Atlas 2.9 s
per call, recorded in MCP_FEASIBILITY §4.3), one asyncio loop is kept alive on a
background thread and all sessions live in it for the whole run.
"""

from __future__ import annotations

import asyncio
import threading
from contextlib import AsyncExitStack

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


class ToolRuntime:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        self._sessions = {}
        self._stacks = {}

    def _run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def submit(self, coro, timeout=600):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    # ---- sessions --------------------------------------------------------
    async def _open(self, key, url):
        stack = AsyncExitStack()
        r, w, _ = await stack.enter_async_context(streamablehttp_client(url))
        sess = await stack.enter_async_context(ClientSession(r, w))
        await sess.initialize()
        self._sessions[key] = sess
        self._stacks[key] = stack

    def open(self, key, url, timeout=120):
        self.submit(self._open(key, url), timeout=timeout)

    async def _close(self, key):
        stack = self._stacks.pop(key, None)
        self._sessions.pop(key, None)
        if stack is not None:
            try:
                await stack.aclose()
            except Exception:
                pass

    def close(self, key):
        try:
            self.submit(self._close(key), timeout=60)
        except Exception:
            pass

    def reopen(self, key, url, timeout=120):
        self.close(key)
        self.open(key, url, timeout=timeout)

    # ---- calls -----------------------------------------------------------
    def caller(self, key, max_chars=1200):
        """Return an async `call(tool_name, args) -> text` bound to one session."""
        async def call(name, args):
            sess = self._sessions.get(key)
            if sess is None:
                return "error: session closed"
            try:
                res = await asyncio.wait_for(sess.call_tool(name, args or {}), timeout=60)
            except Exception as exc:
                return f"error: {type(exc).__name__}: {exc}"[:400]
            txt = res.content[0].text if res.content else ""
            if len(txt) > max_chars:
                txt = txt[:max_chars] + f"\n...[truncated, {len(txt)} chars total]"
            return txt
        return call

    def run_many(self, coros, timeout=600):
        async def _gather():
            return await asyncio.gather(*coros, return_exceptions=True)
        return self.submit(_gather(), timeout=timeout)

    def shutdown(self):
        for k in list(self._stacks):
            self.close(k)
        self.loop.call_soon_threadsafe(self.loop.stop)
