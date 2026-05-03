"""OpenTeddy sidecar entry point — for the bundled-binary use case.

This file is intentionally separate from main.py so the dev path
(`uvicorn main:app --reload` from a developer's checkout) and the
production path (PyInstaller-bundled binary spawned by the Tauri
desktop shell) can each have their own startup behaviour without
hidden mode-flag branches sprinkled across main.py.

What this script does, in order:

  1. Detect we're running as a frozen PyInstaller bundle
     (`sys.frozen == True` once PyInstaller has packaged us). When
     true, override the data paths inside config so we write to the
     user's Application Support folder instead of the .app bundle
     (which is read-only on macOS once the app is signed).

  2. Pick a free TCP port on 127.0.0.1 — bind(0) lets the kernel
     hand us one nobody else is using, sidesteps the user's
     existing nginx/jenkins/etc. fights with hardcoded ports.

  3. Print a single line to stdout the moment the port is chosen:
        OPENTEDDY_SIDECAR_PORT=<port>
     The Tauri Rust shell parses this line to know which URL to
     iframe. Doing it BEFORE uvicorn boots means the shell can
     start preparing the iframe while uvicorn warms up — saves a
     visible second on cold start.

  4. Boot uvicorn programmatically (no `--reload` since the bundle
     has no source files to watch; reload would just thrash).

Logs go to stderr so the Tauri side can `read_line` cleanly on
stdout for the port handshake without mixing in uvicorn's chatter.
"""

from __future__ import annotations

import logging
import os
import socket
import sys
from pathlib import Path


# ── Step 1: redirect data paths when frozen ─────────────────────────────
# PyInstaller sets sys.frozen = True on the bundled executable. We use
# that as our "running inside a .dmg" signal — when true, every path
# the backend wants to write to (SQLite, ChromaDB, skills/, workspace)
# moves to ~/Library/Application Support/OpenTeddy/ on macOS, the
# equivalent on Linux/Windows. Without this the backend would try to
# write inside the .app bundle, which is read-only after macOS
# signing/notarisation rejects the write outright.

def _user_data_dir() -> Path:
    """Cross-platform "where does this app's mutable state live"."""
    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / "OpenTeddy"
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA", home / "AppData" / "Roaming")) / "OpenTeddy"
    # Linux / *BSD — XDG_DATA_HOME or ~/.local/share
    xdg = os.environ.get("XDG_DATA_HOME")
    return Path(xdg) / "OpenTeddy" if xdg else home / ".local" / "share" / "OpenTeddy"


def _patch_paths_if_frozen() -> None:
    if not getattr(sys, "frozen", False):
        return  # dev mode — keep cwd-relative paths
    data_dir = _user_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    # Override BEFORE config is imported by main.py — env vars take
    # precedence over the dataclass defaults.
    os.environ.setdefault("DB_PATH",            str(data_dir / "openteddy.db"))
    os.environ.setdefault("MEMORY_DB_PATH",     str(data_dir / "memory_db"))
    os.environ.setdefault("SKILLS_DIR",         str(data_dir / "skills"))
    os.environ.setdefault("AGENT_WORKSPACE_DIR", str(data_dir / "agent-workspace"))
    # Make sure the dirs exist so first-write doesn't fail.
    for sub in ("memory_db", "skills", "agent-workspace"):
        (data_dir / sub).mkdir(parents=True, exist_ok=True)


# ── Step 2: pick a free port ────────────────────────────────────────────
def _pick_free_port() -> int:
    """Ask the kernel for a free local port and immediately release it.
    There's a tiny race window between this socket close and uvicorn
    bind, but in practice nothing else grabs it that fast on a single-
    user machine. If we ever hit it we can extend to retry-on-bind-fail."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ── Step 3: handshake line to stdout ────────────────────────────────────
def _emit_port_handshake(port: int) -> None:
    """Single line to stdout, immediately flushed. Tauri's spawn code
    reads this synchronously to learn the iframe URL. Format kept
    grep-friendly (`OPENTEDDY_SIDECAR_PORT=NNNNN`) so debugging from
    a manual shell run is trivial."""
    sys.stdout.write(f"OPENTEDDY_SIDECAR_PORT={port}\n")
    sys.stdout.flush()


# ── Step 4: boot uvicorn ────────────────────────────────────────────────
def main() -> None:
    _patch_paths_if_frozen()

    # Configure logging early so any import-time warnings from the
    # backend modules go to stderr (out of stdout's handshake line).
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    port = _pick_free_port()
    _emit_port_handshake(port)
    logging.info("OpenTeddy sidecar starting on 127.0.0.1:%d", port)

    # Late import: importing main.py earlier would touch config + DB
    # + chromadb at module-load time, which we want to happen AFTER
    # the path-patch + AFTER the handshake. Moving the import here
    # keeps cold-start telemetry honest (handshake → ~0 ms; uvicorn
    # ready → ~2-5 s depending on chromadb warm-up).
    import uvicorn
    from main import app

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=port,
        log_level="info",
        access_log=False,   # noisy; runtime logger inside main.py is enough
        # No reload — bundle has no source files to watch.
    )


if __name__ == "__main__":
    main()
