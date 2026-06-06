#!/usr/bin/env python3
"""
Fetch / refresh the cyber-skills index from the upstream repo.

Source: https://github.com/mukul975/Anthropic-Cybersecurity-Skills

What it does:

  1. Lists every directory under upstream `skills/` via the GitHub API
  2. Downloads each `SKILL.md` (skips `scripts/` and `references/` —
     we want guidance, not executable code, in the local index)
  3. Parses YAML frontmatter + markdown body
  4. Writes a flat JSON array to `cyber_skills/index.json`

Once a day rate-limit considerations:

  - Public GitHub API allows 60 unauthenticated requests per hour
  - With `gh auth login` the limit is 5 000 / hour — well above what
    this script needs (~755 requests for the full set, one per skill)
  - Default mode is "use `gh` CLI if available, fall back to public
    API otherwise". Set `OPENTEDDY_GH_TOKEN` env var to pass a token
    explicitly.

This is a maintenance script. It's NOT loaded by the server at runtime
— the server only reads `index.json`. So `update.py` doesn't need to
be production-grade async; sync `urllib` keeps it dependency-free.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import ssl
import urllib.request
import urllib.error
from pathlib import Path

# macOS system Python (3.12 esp) ships without a configured cert store.
# Use certifi's bundle when available — it's a transitive dep of
# anything httpx-touching so it's always present in our venv. Falls
# back to the default context outside that env.
try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()
from typing import Any, Dict, List, Optional

REPO = "mukul975/Anthropic-Cybersecurity-Skills"
API_BASE = "https://api.github.com"
HEADERS = {
    "Accept":         "application/vnd.github.v3+json",
    "User-Agent":     "OpenTeddy-cyber-skills-updater",
}

HERE = Path(__file__).parent
INDEX_PATH = HERE / "index.json"


# ── HTTP helpers ────────────────────────────────────────────────────────────

def _gh_token() -> Optional[str]:
    """Prefer an explicit env var, fall back to whatever gh CLI has
    cached (so the script works with the same auth as the rest of the
    repo without copy-pasting tokens around)."""
    if t := os.environ.get("OPENTEDDY_GH_TOKEN"):
        return t.strip()
    try:
        out = subprocess.check_output(
            ["gh", "auth", "token"], stderr=subprocess.DEVNULL,
        ).decode().strip()
        return out or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _api_get(path: str, *, raw: bool = False) -> Any:
    """One GET against api.github.com. `raw=True` returns the response
    body bytes; default decodes JSON."""
    headers = dict(HEADERS)
    if tok := _gh_token():
        headers["Authorization"] = f"token {tok}"
    url = path if path.startswith("http") else f"{API_BASE}{path}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30, context=_SSL_CTX) as resp:
            data = resp.read()
    except urllib.error.HTTPError as exc:
        # 403 with X-RateLimit-Remaining: 0 = rate limited. Surface a
        # clear hint rather than the raw stack trace.
        if exc.code == 403 and exc.headers.get("X-RateLimit-Remaining") == "0":
            reset = exc.headers.get("X-RateLimit-Reset", "?")
            raise SystemExit(
                f"GitHub API rate limit hit. Reset at unix-ts {reset}. "
                "Run `gh auth login` then re-run this script, or set "
                "OPENTEDDY_GH_TOKEN=ghp_..."
            ) from exc
        raise
    return data if raw else json.loads(data.decode("utf-8"))


# ── SKILL.md parsing ────────────────────────────────────────────────────────

def _parse_skill_md(text: str) -> Dict[str, Any]:
    """Split YAML frontmatter from the markdown body. Frontmatter is
    between two `---` lines at the top of the file; body is everything
    after the second `---`.

    The YAML uses a tiny subset (list values, simple keys) so we parse
    it by hand to avoid a runtime PyYAML dep — keeps the updater self-
    contained. If the format ever uses block scalars / anchors we'll
    swap in PyYAML, but for now bespoke parsing is fine.
    """
    if not text.startswith("---"):
        return {"frontmatter": {}, "body": text}

    end = text.find("\n---", 4)
    if end == -1:
        return {"frontmatter": {}, "body": text}

    raw_fm = text[4:end].strip("\n")
    body = text[end + 4:].lstrip("\n")

    fm: Dict[str, Any] = {}
    current_key: Optional[str] = None
    current_list: Optional[List[str]] = None

    for raw_line in raw_fm.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        # List item under the current key
        if line.lstrip().startswith("- "):
            if current_list is not None:
                current_list.append(line.lstrip()[2:].strip().strip("'\""))
            continue
        # Continuation line (indented value)
        if line.startswith("  ") and current_key:
            existing = fm.get(current_key) or ""
            fm[current_key] = (existing + " " + line.strip()).strip()
            current_list = None
            continue
        # New top-level key
        if ":" in line:
            k, _, v = line.partition(":")
            key = k.strip()
            val = v.strip()
            if val == "":
                # next non-blank line(s) are list items
                current_key = key
                current_list = []
                fm[key] = current_list
            else:
                # scalar value
                fm[key] = val.strip("'\"")
                current_key = key
                current_list = None
    return {"frontmatter": fm, "body": body}


# ── Index build ─────────────────────────────────────────────────────────────

def build_index(limit: Optional[int] = None, verbose: bool = True) -> List[Dict[str, Any]]:
    """List skills/ subdirs, fetch SKILL.md from each, build the index."""
    if verbose:
        print(f"Listing skills/ in {REPO}…")
    listing = _api_get(f"/repos/{REPO}/contents/skills")
    subdirs = [item["name"] for item in listing if item.get("type") == "dir"]
    if verbose:
        print(f"  found {len(subdirs)} skills")
    if limit:
        subdirs = subdirs[:limit]
        if verbose:
            print(f"  (limited to first {limit} via --limit)")

    out: List[Dict[str, Any]] = []
    for i, name in enumerate(subdirs, 1):
        try:
            skill_md_meta = _api_get(
                f"/repos/{REPO}/contents/skills/{name}/SKILL.md"
            )
            download_url = skill_md_meta.get("download_url")
            if not download_url:
                if verbose:
                    print(f"  [{i:3}] {name:55} SKIP (no download_url)")
                continue
            body_bytes = _api_get(download_url, raw=True)
            text = body_bytes.decode("utf-8", errors="replace")
            parsed = _parse_skill_md(text)
            fm = parsed["frontmatter"]
            out.append({
                "name":         fm.get("name") or name,
                "description":  fm.get("description") or "",
                "domain":       fm.get("domain") or "",
                "subdomain":    fm.get("subdomain") or "",
                "tags":         fm.get("tags") or [],
                "mitre_attack": fm.get("mitre_attack") or [],
                "nist_csf":     fm.get("nist_csf") or [],
                "atlas":        fm.get("atlas_techniques") or [],
                "d3fend":       fm.get("d3fend") or [],
                "version":      fm.get("version") or "",
                "license":      fm.get("license") or "Apache-2.0",
                "upstream_path": f"skills/{name}/SKILL.md",
                "body":         parsed["body"],
            })
            if verbose and (i % 50 == 0 or i == len(subdirs)):
                print(f"  [{i:3}/{len(subdirs)}] indexed")
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"  [{i:3}] {name:55} ERROR: {exc}")
            continue

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the cyber-skills index")
    ap.add_argument("--limit", type=int, default=None,
                    help="Index only the first N skills (smoke testing)")
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress per-skill progress logs")
    ap.add_argument("--output", default=str(INDEX_PATH),
                    help=f"Where to write the index (default {INDEX_PATH})")
    args = ap.parse_args()

    started = time.monotonic()
    index = build_index(limit=args.limit, verbose=not args.quiet)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "_meta": {
            "source":      f"https://github.com/{REPO}",
            "fetched_at":  time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "skill_count": len(index),
            "license":     "Apache-2.0",
        },
        "skills": index,
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    size_kb = out_path.stat().st_size / 1024
    print(f"\nDone — wrote {len(index)} skills to {out_path} "
          f"({size_kb:.1f} KB) in {time.monotonic() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
