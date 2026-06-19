#!/usr/bin/env python3
"""
file_diff_report.py — daily "what changed in this folder" report (hybrid).

Digital-employee job for watching a NAS share or an rclone-mounted Google
Drive. Each run snapshots a manifest of the folder (path + size + mtime),
diffs it against the previous snapshot, and produces a report of
added / modified / deleted files. Run it daily (cron / systemd timer /
OpenTeddy scheduler) and "the two most recent snapshots" = yesterday vs
the day before — exactly the diff you asked for.

HYBRID (per your choice): the LOCAL machine does all the file I/O and the
diff — your data never leaves the box for that part. Only a compact,
already-computed CHANGE LIST is sent to your selected commercial provider
(Settings → Cloud LLM Provider) to write the human-readable summary. No
provider / no key / provider error → it still emits a clean plain report
(the LLM summary is a bonus, never a dependency).

  Google Drive: mount it first with rclone so it looks like a folder:
      rclone mount gdrive:SomeFolder /mnt/gdrive --daemon
  then point --dir at /mnt/gdrive. NAS: just point --dir at the mount.

Examples:
    # First run on a folder = establishes the baseline (no diff yet)
    python scripts/file_diff_report.py --dir /mnt/nas/share --name nas-share

    # Daily run: diff vs the previous snapshot, summarise, send to Telegram
    python scripts/file_diff_report.py --dir /mnt/nas/share --name nas-share --notify

    # Plain report only (skip the provider summary)
    python scripts/file_diff_report.py --dir /mnt/gdrive --name gdrive --no-summary

Change signal: size + mtime (rsync-style quick check — fast on large/many
files, and mtime is exactly the "changed yesterday" signal). Pass --hash to
also content-hash files so a same-size, same-mtime edit is still caught.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from datetime import datetime

# Make the OpenTeddy package importable when run from anywhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_DEFAULT_EXCLUDES = {".git", "node_modules", ".DS_Store", "__pycache__", ".rclone"}


# ── Manifest + diff (PURE — no network, fully unit-testable) ──────────────────

def build_manifest(
    root: str, *, do_hash: bool = False, excludes: set[str] | None = None,
) -> dict[str, dict]:
    """Walk `root`, return {relpath: {size, mtime[, sha256]}} for every file.

    Excluded directory / file names are skipped at any depth. Symlinks are
    not followed (avoids loops + counting the link target twice)."""
    excludes = excludes or _DEFAULT_EXCLUDES
    manifest: dict[str, dict] = {}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [d for d in dirnames if d not in excludes]
        for fn in filenames:
            if fn in excludes:
                continue
            full = os.path.join(dirpath, fn)
            try:
                st = os.stat(full)
            except OSError:
                continue  # vanished mid-walk / permission — skip
            rel = os.path.relpath(full, root)
            entry = {"size": st.st_size, "mtime": int(st.st_mtime)}
            if do_hash:
                entry["sha256"] = _hash_file(full)
            manifest[rel] = entry
    return manifest


def _hash_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            while True:
                b = fh.read(chunk)
                if not b:
                    break
                h.update(b)
    except OSError:
        return ""
    return h.hexdigest()


def _changed(a: dict, b: dict) -> bool:
    """Two manifest entries differ? Compare hash when both have one,
    else fall back to size + mtime."""
    if "sha256" in a and "sha256" in b and a["sha256"] and b["sha256"]:
        return a["sha256"] != b["sha256"]
    return a["size"] != b["size"] or a["mtime"] != b["mtime"]


def diff_manifests(old: dict[str, dict], new: dict[str, dict]) -> dict:
    """Return {added, modified, deleted} lists of relpaths."""
    old_keys, new_keys = set(old), set(new)
    added    = sorted(new_keys - old_keys)
    deleted  = sorted(old_keys - new_keys)
    modified = sorted(k for k in (old_keys & new_keys) if _changed(old[k], new[k]))
    return {"added": added, "modified": modified, "deleted": deleted}


# ── Snapshot storage ──────────────────────────────────────────────────────────

def _snap_dir(state_dir: str, name: str) -> str:
    d = os.path.join(os.path.expanduser(state_dir), name)
    os.makedirs(d, exist_ok=True)
    return d


def save_snapshot(state_dir: str, name: str, manifest: dict, stamp: str) -> str:
    path = os.path.join(_snap_dir(state_dir, name), f"{stamp}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"stamp": stamp, "manifest": manifest}, fh)
    return path


def previous_snapshot(state_dir: str, name: str, before: str) -> dict | None:
    """Most recent snapshot strictly older than `before` (by filename)."""
    d = _snap_dir(state_dir, name)
    snaps = sorted(f for f in os.listdir(d) if f.endswith(".json"))
    prior = [s for s in snaps if s[:-5] < before]
    if not prior:
        return None
    with open(os.path.join(d, prior[-1]), encoding="utf-8") as fh:
        return json.load(fh)


# ── Report ────────────────────────────────────────────────────────────────────

def format_plain(name: str, diff: dict, prev_stamp: str, cur_stamp: str,
                 sample: int = 40) -> str:
    """Deterministic report — the always-available fallback."""
    a, m, d = diff["added"], diff["modified"], diff["deleted"]
    lines = [
        f"📁 檔案變動報告：{name}",
        f"   區間：{prev_stamp} → {cur_stamp}",
        f"   新增 {len(a)}　修改 {len(m)}　刪除 {len(d)}",
    ]
    for label, items in (("新增", a), ("修改", m), ("刪除", d)):
        if not items:
            continue
        lines.append(f"\n[{label}] ({len(items)})")
        lines.extend(f"  • {p}" for p in items[:sample])
        if len(items) > sample:
            lines.append(f"  … 還有 {len(items) - sample} 筆（已截斷）")
    return "\n".join(lines)


async def provider_summary(plain: str, name: str) -> str:
    """Hybrid step: hand the already-computed change list to the SELECTED
    provider for a readable summary. Any failure → return the plain report
    (the LLM is a bonus, never a hard dependency)."""
    try:
        from llm_provider import get_default_provider
        provider = get_default_provider()
        if not provider.is_configured():
            return plain + "\n\n(未設定商業 provider，以上為純文字報告)"
        system = (
            "你是一位系統維運助理。下面是某資料夾兩天之間的檔案變動清單"
            "（已在本機計算好）。請用繁體中文寫一段精簡的重點摘要：哪些"
            "變動值得注意（大量新增/刪除、關鍵設定檔被改、異常的批次變動），"
            "哪些是例行。不要逐條複述清單，只點出重點與潛在風險。"
        )
        resp = await provider.complete_text(
            user_message=f"資料夾：{name}\n\n{plain}",
            system=system,
            max_tokens=800,
        )
        summary = (resp.text or "").strip()
        if not summary:
            return plain
        return f"{plain}\n\n— — — AI 摘要 — — —\n{summary}"
    except Exception as exc:  # noqa: BLE001
        return plain + f"\n\n(AI 摘要略過：{exc})"


async def maybe_notify(text: str) -> None:
    try:
        from tools.notify_tool import telegram_send
        await telegram_send(text)
        print("✓ 已透過 Telegram 送出報告")
    except Exception as exc:  # noqa: BLE001
        print(f"⚠ Telegram 送出失敗（報告仍已在上方印出）：{exc}")


# ── Main ────────────────────────────────────────────────────────────────────

async def _amain(args) -> int:
    if not os.path.isdir(args.dir):
        print(f"✗ --dir 不是有效資料夾：{args.dir}")
        return 1

    excludes = _DEFAULT_EXCLUDES | set(
        e.strip() for e in (args.exclude or "").split(",") if e.strip()
    )
    stamp = args.stamp or datetime.now().strftime("%Y-%m-%d")

    print(f"… 掃描 {args.dir} （hash={'on' if args.hash else 'off'}）")
    manifest = build_manifest(args.dir, do_hash=args.hash, excludes=excludes)
    print(f"  共 {len(manifest)} 個檔案")

    prev = previous_snapshot(args.state_dir, args.name, before=stamp)
    save_snapshot(args.state_dir, args.name, manifest, stamp)

    if prev is None:
        msg = (f"📁 {args.name}: 已建立基準快照（{len(manifest)} 個檔案）。"
               f"明天起就會有差異報告。")
        print("\n" + msg)
        if args.notify:
            await maybe_notify(msg)
        return 0

    diff = diff_manifests(prev["manifest"], manifest)
    plain = format_plain(args.name, diff, prev["stamp"], stamp)

    total = sum(len(diff[k]) for k in ("added", "modified", "deleted"))
    if total == 0:
        report = f"📁 {args.name}（{prev['stamp']} → {stamp}）：無檔案變動。"
    elif args.summary:
        report = await provider_summary(plain, args.name)
    else:
        report = plain

    print("\n" + report)
    if args.notify:
        await maybe_notify(report)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="folder to watch (NAS / rclone mount)")
    ap.add_argument("--name", required=True, help="label; also namespaces the snapshots")
    ap.add_argument("--state-dir", default="~/.openteddy/file-snapshots",
                    help="where daily snapshots are stored")
    ap.add_argument("--exclude", default="",
                    help="comma-separated extra names to skip (dirs/files)")
    ap.add_argument("--hash", action="store_true",
                    help="content-hash files (catches same-size+mtime edits; slower)")
    ap.add_argument("--stamp", default=None,
                    help="override the snapshot date label YYYY-MM-DD "
                         "(for testing the diff today, or backfilling a missed day)")
    summ = ap.add_mutually_exclusive_group()
    summ.add_argument("--summary", dest="summary", action="store_true", default=True,
                      help="use the selected provider to write the summary (default)")
    summ.add_argument("--no-summary", dest="summary", action="store_false",
                      help="plain report only — no provider call")
    ap.add_argument("--notify", action="store_true", help="send the report to Telegram")
    args = ap.parse_args()
    try:
        return asyncio.run(_amain(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
