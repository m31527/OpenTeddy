"""
OpenTeddy's MCP server over real stdio, against a stand-in runtime.

  A. the tools Claude Code sees (names + descriptions)
  B. each tool sends the right request: ask / home / weather → POST /tasks
     in the "Claude Code" session (wait, auto_lane, triggered_by=mcp);
     print_file uploads then asks to print with copies / duplex / b&w /
     pages; notify_me → /notify; self_review → /review; list_schedules
  C. runtime unreachable → a clear message

    .venv/bin/python tests/test_mcp_server.py
"""
from __future__ import annotations

import asyncio, json, logging, os, sys, tempfile, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEN: list = []


class _Runtime(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _send(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        SEEN.append(("GET", self.path, None))
        if self.path.startswith("/review"):
            return self._send({"text": f"🩺 report {self.path.split('=')[-1]} days"})
        if self.path == "/schedules":
            return self._send({"data": [{"id": "68a3633c-xx", "cron": "40 7 * * 1-5", "goal": "叫小愛說：早安",
                                          "enabled": True, "next_run_at": "2026-10-27T07:40:00+08:00"}]})
        self._send({}, 404)

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if self.path == "/upload":
            SEEN.append(("POST", "/upload", raw))
            return self._send({"rel_path": "uploads/report.pdf", "size_bytes": 2048})
        body = json.loads(raw or b"{}")
        SEEN.append(("POST", self.path, body))
        if self.path == "/tasks":
            return self._send({"task_id": "t1", "session_id": body["session_id"], "status": "completed",
                               "message": f"done: {body['intent'][-30:]}"})
        if self.path == "/notify":
            return self._send({"success": True, "error": None})
        self._send({}, 404)


async def test_stdio(port: int) -> None:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    params = StdioServerParameters(command=sys.executable, args=[os.path.join(ROOT, "mcp_server.py")],
                                   env={**os.environ, "OPENTEDDY_URL": f"http://127.0.0.1:{port}"})
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            init = await s.initialize()
            assert init.serverInfo.name == "openteddy"
            tools = {t.name: t for t in (await s.list_tools()).tools}
            assert set(tools) == {"ask_openteddy", "home_control", "print_file", "notify_me", "weather",
                                  "self_review", "list_schedules"}, set(tools)
            assert "小愛" in tools["home_control"].description and "Telegram" in tools["notify_me"].description
            print("  ✓ initialize over stdio; 7 tools with descriptions Claude Code can choose from")

            text = lambda res: res.content[0].text
            out = text(await s.call_tool("home_control", {"command": "關客廳燈"}))
            m, path, body = SEEN[-1]
            assert (m, path) == ("POST", "/tasks") and body["intent"] == "關客廳燈" and out == "done: 關客廳燈"
            assert body["session_id"] == "claude-code-mcp" and body["wait"] and body["auto_lane"]
            assert body["context"] == {"triggered_by": "mcp"}
            text(await s.call_tool("ask_openteddy", {"request": "每天早上 8 點查營收"}))
            assert SEEN[-1][2]["intent"] == "每天早上 8 點查營收"
            text(await s.call_tool("weather", {"question": "最近有颱風靠近台灣嗎"}))
            assert SEEN[-1][2]["intent"] == "最近有颱風靠近台灣嗎"
            print("  ✓ ask / home / weather → POST /tasks in the Claude Code session, waiting for the answer")

            with tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False) as f:
                f.write(b"%PDF-1.4 hello")
                pdf = f.name
            text(await s.call_tool("print_file", {"path": pdf, "copies": 2, "two_sided": True,
                                                  "black_and_white": True, "pages": "1-3"}))
            up = [x for x in SEEN if x[1] == "/upload"][-1][2]
            assert b"%PDF-1.4 hello" in up and b"claude-code-mcp" in up
            goal = SEEN[-1][2]["intent"]
            assert goal.startswith("[Attachments in workspace") and "uploads/report.pdf (2.0 KB)" in goal
            assert goal.endswith("列印 2 份 雙面 黑白 第 1 到 3 頁"), goal
            missing = text(await s.call_tool("print_file", {"path": "/no/such.pdf"}))
            assert missing.startswith("❌ File not found")
            os.unlink(pdf)
            print("  ✓ print_file uploads the file, then asks OpenTeddy to print with the options")

            assert text(await s.call_tool("notify_me", {"message": "部署完成"})) == "✅ sent to Telegram"
            assert SEEN[-1] == ("POST", "/notify", {"text": "部署完成"})
            assert text(await s.call_tool("self_review", {"days": 3})) == "🩺 report 3 days"
            sch = text(await s.call_tool("list_schedules", {}))
            assert "40 7 * * 1-5 → 叫小愛說：早安" in sch and "id 68a3633c" in sch
            print("  ✓ notify_me, self_review, list_schedules")


async def test_unreachable() -> None:
    import mcp_server
    mcp_server.RUNTIME = "http://127.0.0.1:9"
    out = await mcp_server.ask_openteddy("hi")
    assert out.startswith("❌ OpenTeddy isn't reachable at http://127.0.0.1:9"), out
    print("  ✓ runtime down → a clear message, not a stack trace")


async def main() -> None:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Runtime)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        print("A/B. stdio"); await test_stdio(srv.server_address[1])
        print("C. unreachable"); await test_unreachable()
    finally:
        srv.shutdown()
    print("\nALL MCP SERVER TESTS PASS")


asyncio.run(main())
