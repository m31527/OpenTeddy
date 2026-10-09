"""
Secrets never reach a log line.

    .venv/bin/python tests/test_log_redact.py
"""
from __future__ import annotations

import io, logging, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import log_redact

TOKEN = "8853061357:AAG-fake_TOKEN_for_tests_0123456789"


def main() -> None:
    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    h.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [h]
    root.setLevel(logging.INFO)
    log_redact.install()
    log_redact.install()                                  # idempotent
    assert h.filters.count(log_redact._FILTER) == 1

    log = logging.getLogger("telegram_bridge")
    log.info('HTTP Request: POST https://api.telegram.org/bot%s/sendMessage "HTTP/1.1 200 OK"', TOKEN)
    log.warning("download failed: https://api.telegram.org/file/bot%s/documents/f.xlsx", TOKEN)
    log.info("models: https://generativelanguage.googleapis.com/v1beta/models?key=AIzaSyFAKE123&pageSize=1000")
    log.info("headers {'Authorization': 'Bearer sk-ant-api03-FAKEFAKEFAKE'}")
    try:
        raise RuntimeError(f"Client error '404' for url 'https://api.telegram.org/bot{TOKEN}/getFile'")
    except RuntimeError:
        log.exception("getFile crashed")
    log.info("plain line with %d%% done and a /path?page=2", 50)
    out = buf.getvalue()

    assert "AAG-fake" not in out and "8853061357" not in out, out
    assert out.count("bot<redacted>") == 3, out
    assert "key=<redacted>&pageSize=1000" in out and "AIza" not in out
    assert "Bearer <redacted>" in out and "sk-ant" not in out
    assert "Traceback" in out and "getFile crashed" in out
    assert "plain line with 50% done and a /path?page=2" in out
    print("  ✓ bot token (URLs, file URLs, tracebacks), ?key=, Bearer masked; other text untouched")

    assert logging.getLogger("httpx").getEffectiveLevel() == logging.WARNING
    assert logging.getLogger("httpcore").getEffectiveLevel() == logging.WARNING
    buf.truncate(0); buf.seek(0)
    logging.getLogger("httpx").info("HTTP Request: GET https://api.telegram.org/bot%s/getUpdates", TOKEN)
    assert buf.getvalue() == ""
    print("  ✓ httpx per-request INFO lines no longer logged")

    print("\nALL LOG REDACTION TESTS PASS")


main()
