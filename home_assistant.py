"""
Home Assistant bridge: OpenTeddy controls the house through HA.

Home Assistant speaks to every brand (米家 via Xiaomi Miot Auto, Google TV,
…) and turns them into entities; OpenTeddy talks to one REST API with a
long-lived access token (Settings → Home Assistant).

    configured()                       URL + token set
    status()                           HA reachable? version, entity count, 小愛
    list_entities(query, domain)       compact device list for the model
    get_state(entity_id)
    control(entity_id, service, data)  with the safety policy below
    assist(text)                       HA's own Assist: "關客廳燈" → done, no LLM
    xiaoai(text, execute)              小愛音箱 says `text`, or runs it as a
                                       spoken command (IR air-con, TV, …)
    confirm(code)                      run a held sensitive action

Safety. Locks, alarms, valves, sirens, water heaters and garage / gate /
door covers (Settings → ha_confirm_domains) are never run directly: the
call is held and a 4-digit code comes back. Only a new message from the
person — "確認 1234" — runs it (orchestrator fast path), so the model can't
confirm on its own and Telegram's auto-approval doesn't apply. Codes
expire after 5 minutes and work once.
"""
from __future__ import annotations

import asyncio
import logging
import re
import secrets
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from config import config

logger = logging.getLogger(__name__)

DEFAULT_CONFIRM_DOMAINS = "lock,alarm_control_panel,valve,siren,water_heater"
_SENSITIVE_COVER_CLASSES = {"garage", "gate", "door"}
_CONFIRM_TTL_S = 300
_pending: Dict[str, Dict[str, Any]] = {}


class HAError(Exception):
    """A Home Assistant request that failed, with a message for the user."""


def _url() -> str:
    return (getattr(config, "ha_url", "") or "").strip().rstrip("/")


def _token() -> str:
    return (getattr(config, "ha_token", "") or "").strip()


def configured() -> bool:
    return bool(_url() and _token())


async def _request(method: str, path: str, json: Any = None, timeout: float = 10.0) -> Any:
    if not configured():
        raise HAError("還沒連上 Home Assistant：請在 Settings → Home Assistant 填網址和存取權杖。")
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.request(method, _url() + path, json=json,
                                     headers={"Authorization": f"Bearer {_token()}"})
    except httpx.HTTPError as exc:
        raise HAError(f"連不到 Home Assistant（{_url()}）：{type(exc).__name__}") from exc
    if r.status_code == 401:
        raise HAError("Home Assistant 拒絕了存取權杖，請重新產生一組貼到 Settings。")
    if r.status_code >= 400:
        raise HAError(f"Home Assistant 回應 {r.status_code}：{r.text[:200]}")
    try:
        return r.json()
    except ValueError:
        return r.text


# ── Reading ──────────────────────────────────────────────────────────────────

def _compact(s: Dict[str, Any]) -> Dict[str, Any]:
    attrs = s.get("attributes") or {}
    out = {"entity_id": s.get("entity_id"), "name": attrs.get("friendly_name") or s.get("entity_id"),
           "state": s.get("state")}
    for k in ("device_class", "unit_of_measurement", "current_temperature", "temperature",
              "hvac_modes", "brightness", "volume_level", "source", "media_title"):
        if k in attrs:
            out[k] = attrs[k]
    return out


async def list_entities(query: Optional[str] = None, domain: Optional[str] = None,
                        limit: int = 60) -> List[Dict[str, Any]]:
    states = await _request("GET", "/api/states")
    q = (query or "").strip().lower()
    out = []
    for s in states or []:
        eid = s.get("entity_id", "")
        if domain and not eid.startswith(domain.rstrip(".") + "."):
            continue
        if eid.split(".", 1)[0] in {"sun", "zone", "person", "update", "automation", "tts", "conversation", "event"}:
            continue
        c = _compact(s)
        if q and q not in eid.lower() and q not in str(c["name"]).lower():
            continue
        out.append(c)
    return out[:limit]


async def get_state(entity_id: str) -> Dict[str, Any]:
    s = await _request("GET", f"/api/states/{entity_id}")
    return _compact(s) | {"attributes": (s or {}).get("attributes", {})}


async def status() -> Dict[str, Any]:
    if not configured():
        return {"configured": False, "reachable": False}
    cfg = await _request("GET", "/api/config")
    states = await _request("GET", "/api/states")
    domains: Dict[str, int] = {}
    for s in states or []:
        d = s.get("entity_id", "").split(".", 1)[0]
        domains[d] = domains.get(d, 0) + 1
    return {"configured": True, "reachable": True, "version": cfg.get("version"),
            "location": cfg.get("location_name"), "language": cfg.get("language"),
            "entities": len(states or []), "domains": domains,
            "xiaoai": await _find_xiaoai(states)}


# ── Devices (grouped) ───────────────────────────────────────────────────────

_SKIP_DOMAINS = {"sun", "zone", "person", "update", "automation", "script", "scene", "conversation",
                 "tts", "event", "weather", "stt", "wake_word", "assist_satellite", "input_boolean",
                 "input_number", "input_select", "input_text", "input_datetime", "input_button",
                 "todo", "calendar", "device_tracker", "image", "notify"}

# One template call: every entity with its device, area and last change —
# HA's template engine knows the device registry; REST /api/states doesn't.
_DEVICES_TEMPLATE = (
    "[{% for s in states %}{% set d = device_id(s.entity_id) %}"
    "{\"e\":{{ s.entity_id|tojson }},\"st\":{{ s.state|tojson }},"
    "\"lc\":{{ s.last_changed.isoformat()|tojson }},\"d\":{{ (d or '')|tojson }},"
    "\"dn\":{{ ((device_attr(d, 'name_by_user') or device_attr(d, 'name')) if d else s.name)|tojson }},"
    "\"et\":{{ ((device_attr(d, 'entry_type') or '')|string if d else '')|tojson }},"
    "\"a\":{{ (area_name(s.entity_id) or '')|tojson }}}"
    "{% if not loop.last %},{% endif %}{% endfor %}]"
)


def _ago(iso: str) -> str:
    from datetime import datetime, timezone
    try:
        t = datetime.fromisoformat(iso)
        secs = (datetime.now(timezone.utc) - t).total_seconds()
    except (TypeError, ValueError):
        return ""
    if secs < 3600:
        return f"{max(1, int(secs // 60))} 分鐘"
    if secs < 86400:
        return f"{int(secs // 3600)} 小時"
    return f"{int(secs // 86400)} 天"


async def devices() -> List[Dict[str, Any]]:
    """[{name, area, online, entities, offline_for}] — one row per physical
    device. Service "devices" (Sun, Backup, weather, TTS) and entities with
    no device are left out. A device is offline when none of its entities
    has a real state and at least one is unavailable: buttons sit at
    "unknown" forever, and cloud integrations leave single unsupported
    features unavailable on devices that are fine."""
    import json as _json
    raw = await _request("POST", "/api/template", json={"template": _DEVICES_TEMPLATE})
    rows = _json.loads(raw) if isinstance(raw, str) else raw
    groups: Dict[str, Dict[str, Any]] = {}
    for r in rows or []:
        if (r["e"].split(".", 1)[0] in _SKIP_DOMAINS or not r.get("d")
                or "service" in str(r.get("et") or "").lower()):
            continue
        key = r["d"]
        g = groups.setdefault(key, {"name": r.get("dn") or r["e"], "area": r.get("a") or "",
                                    "states": [], "changed": []})
        g["states"].append(r.get("st"))
        if r.get("st") == "unavailable":          # when it went unavailable
            g["changed"].append(r.get("lc") or "")
        if not g["area"] and r.get("a"):
            g["area"] = r["a"]
    out = []
    for g in groups.values():
        real = any(st not in ("unavailable", "unknown", None) for st in g["states"])
        online = real or "unavailable" not in g["states"]
        out.append({"name": g["name"], "area": g["area"], "online": online,
                    "entities": len(g["states"]),
                    "offline_for": "" if online or not g["changed"] else _ago(max(g["changed"]))})
    out.sort(key=lambda d: (d["online"], d["area"], d["name"]))
    return out


def describe_devices(devs: List[Dict[str, Any]], only_offline: bool) -> str:
    """Plain-language answer for "哪些設備離線" / "家裡有哪些設備"."""
    offline = [d for d in devs if not d["online"]]
    online = [d for d in devs if d["online"]]
    lines: List[str] = []
    if only_offline:
        if not offline:
            return f"✅ 所有 {len(devs)} 台設備都在線。"
        lines.append(f"📴 離線的設備（{len(offline)} 台）：")
        for d in offline:
            where = f"{d['area']} · " if d["area"] else ""
            since = f"，已離線 {d['offline_for']}" if d["offline_for"] else ""
            lines.append(f"• {where}{d['name']}{since}")
        lines.append(f"\n✅ 其他 {len(online)} 台在線。")
        return "\n".join(lines)
    lines.append(f"🏠 家裡共有 {len(devs)} 台設備，{len(online)} 台在線、{len(offline)} 台離線：")
    by_area: Dict[str, List[Dict[str, Any]]] = {}
    for d in sorted(devs, key=lambda d: (not d["online"], d["name"])):
        by_area.setdefault(d["area"] or "未分區", []).append(d)
    for area, ds in by_area.items():
        lines.append(f"\n**{area}**")
        for d in ds:
            lines.append(f"• {d['name']}" + ("" if d["online"] else " 📴 離線"))
    return "\n".join(lines)


# ── Acting ───────────────────────────────────────────────────────────────────

def _confirm_domains() -> set:
    raw = getattr(config, "ha_confirm_domains", None)
    raw = DEFAULT_CONFIRM_DOMAINS if raw is None else raw
    return {d.strip() for d in str(raw).split(",") if d.strip()}


async def _needs_confirmation(entity_id: str) -> bool:
    domain = entity_id.split(".", 1)[0]
    if domain in _confirm_domains():
        return True
    if domain == "cover":
        try:
            st = await get_state(entity_id)
            return st.get("attributes", {}).get("device_class") in _SENSITIVE_COVER_CLASSES
        except HAError:
            return True                    # unknown cover → be careful
    return False


async def control(entity_id: str, service: str, data: Optional[Dict[str, Any]] = None,
                  confirmed: bool = False) -> Dict[str, Any]:
    """Call <domain>.<service> on one entity. Sensitive entities come back
    as {"needs_confirmation": code} unless `confirmed` (only confirm())."""
    entity_id = (entity_id or "").strip()
    if "." not in entity_id:
        raise HAError(f"不是有效的設備 ID：{entity_id}")
    domain = entity_id.split(".", 1)[0]
    service = (service or "").strip().split(".")[-1]
    if not service:
        raise HAError("缺少動作，例如 turn_on / turn_off / set_temperature。")
    if not confirmed and await _needs_confirmation(entity_id):
        code = f"{secrets.randbelow(9000) + 1000}"
        _pending[code] = {"entity_id": entity_id, "service": service, "data": data or {},
                          "expires": time.monotonic() + _CONFIRM_TTL_S}
        logger.info("Home: holding %s.%s on %s for confirmation", domain, service, entity_id)
        return {"needs_confirmation": code, "entity_id": entity_id, "service": service,
                "message": f"這是需要確認的設備。請使用者自己回覆「確認 {code}」才會執行，5 分鐘內有效。"}
    body = {"entity_id": entity_id, **(data or {})}
    changed = await _request("POST", f"/api/services/{domain}/{service}", json=body)
    logger.info("Home: %s.%s on %s %s", domain, service, entity_id, data or "")
    return {"done": True, "entity_id": entity_id, "service": f"{domain}.{service}",
            "changed": [c.get("entity_id") for c in (changed or []) if isinstance(c, dict)]}


async def confirm(code: str) -> Dict[str, Any]:
    now = time.monotonic()
    for k in [k for k, v in _pending.items() if v["expires"] < now]:
        _pending.pop(k, None)
    held = _pending.pop((code or "").strip(), None)
    if not held:
        raise HAError("這個確認碼不存在或已過期（5 分鐘），請重新下指令。")
    return await control(held["entity_id"], held["service"], held["data"], confirmed=True)


# ── Names the way people say them ────────────────────────────────────────────
# Assist matches names and aliases exactly, so "幫我把皓房間檯燈打開" missed a
# lamp called 皓檯燈 and the agent spent eight minutes writing shell scripts.
# For a plain on / off, match the spoken name ourselves: every character of
# a device's name (or alias), in order, inside what was said — 皓檯燈 in
# 皓房間檯燈 — and pick it only when exactly one device fits best.

_ON = re.compile(r"(打開|開啟|啟動|開|turn\s+on|switch\s+on)", re.I)
_OFF = re.compile(r"(關掉|關閉|關上|關|turn\s+off|switch\s+off)", re.I)
_ONOFF_DOMAINS = ("light", "switch", "fan", "humidifier", "input_boolean", "media_player")
_FILLER = re.compile(r"(幫我|幫忙|請|麻煩|把|將|給我|一下|吧|啦|喔|哦|呢|的|全部|可以|嗎|"
                     r"打開|開啟|啟動|關掉|關閉|關上|開|關|turn\s+on|turn\s+off|switch\s+on|switch\s+off|"
                     r"please|the|[\s，,。！!？?])", re.I)
_names_cache: Tuple[float, Dict[str, List[str]]] = (0.0, {})


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", (s or "").lower().replace("臺", "台"))


def _in_order(small: str, big: str) -> bool:
    it = iter(big)
    return all(ch in it for ch in small)


def _name_score(name: str, said: str) -> int:
    if len(name) < 2 or len(said) < 2:
        return 0
    if name == said:
        return 100
    if _in_order(name, said):          # 皓檯燈 inside 皓房間檯燈
        return 50 + len(name)
    if _in_order(said, name):          # 房間燈 inside 皓房間檯燈
        return 40 + len(said)
    return 0


async def _registry_names(entity_ids: List[str]) -> Dict[str, List[str]]:
    """Registry names and aliases (WebSocket; REST doesn't expose aliases).
    Cached five minutes; {} when unavailable — friendly names still work."""
    global _names_cache
    if _names_cache[1] and time.monotonic() - _names_cache[0] < 300:
        return _names_cache[1]
    import json as _json
    try:
        import websockets
        url = re.sub(r"^http", "ws", _url()) + "/api/websocket"
        async with websockets.connect(url, open_timeout=5, close_timeout=2, max_size=2 ** 24) as ws:
            await asyncio.wait_for(ws.recv(), 5)
            await ws.send(_json.dumps({"type": "auth", "access_token": _token()}))
            if _json.loads(await asyncio.wait_for(ws.recv(), 5)).get("type") != "auth_ok":
                return {}
            await ws.send(_json.dumps({"id": 1, "type": "config/entity_registry/get_entries",
                                       "entity_ids": entity_ids}))
            while True:
                msg = _json.loads(await asyncio.wait_for(ws.recv(), 10))
                if msg.get("id") == 1:
                    break
    except Exception as exc:  # noqa: BLE001
        logger.debug("Home: registry names unavailable: %s", exc)
        return {}
    out: Dict[str, List[str]] = {}
    for eid, e in ((msg.get("result") or {}).items() if isinstance(msg.get("result"), dict) else []):
        if isinstance(e, dict):
            out[eid] = [n for n in [e.get("name"), e.get("original_name"), *(e.get("aliases") or [])] if n]
    _names_cache = (time.monotonic(), out)
    return out


async def find_on_off(text: str) -> Optional[Dict[str, Any]]:
    """{entity_id, name, service} for a plain on / off of one named device,
    or None (not on / off, a number in it, no clear single match)."""
    t = (text or "").strip()
    verbs = t.replace("開關", "")                       # the noun 開關 is not 開 + 關
    on, off = bool(_ON.search(verbs)), bool(_OFF.search(verbs))
    if on == off or re.search(r"\d", t):
        return None
    said = _norm(_FILLER.sub("", t))
    if len(said) < 2:
        return None
    states = [s for s in (await _request("GET", "/api/states") or [])
              if s.get("entity_id", "").split(".", 1)[0] in _ONOFF_DOMAINS]
    extra = await _registry_names([s["entity_id"] for s in states])
    scored: List[Tuple[int, str, str]] = []
    for s in states:
        eid = s["entity_id"]
        names = [(s.get("attributes") or {}).get("friendly_name") or ""] + extra.get(eid, [])
        best = max(((_name_score(_norm(n), said), n) for n in names if n), default=(0, ""))
        if best[0] and "燈" in said and eid.startswith("light."):
            best = (best[0] + 3, best[1])
        if best[0]:
            scored.append((best[0], eid, best[1]))
    scored.sort(reverse=True)
    if not scored or scored[0][0] < 42 or (len(scored) > 1 and scored[1][0] == scored[0][0]):
        return None
    return {"entity_id": scored[0][1], "name": scored[0][2], "service": "turn_on" if on else "turn_off"}


async def assist(text: str, language: Optional[str] = None) -> Dict[str, Any]:
    """Home Assistant Assist (its own sentence matcher): {ok, kind, speech, targets}."""
    lang = language or (getattr(config, "ha_language", "") or "").strip() or None
    body: Dict[str, Any] = {"text": text}
    if lang:
        body["language"] = lang
    res = await _request("POST", "/api/conversation/process", json=body)
    r = (res or {}).get("response") or {}
    kind = r.get("response_type") or "error"
    speech = (((r.get("speech") or {}).get("plain") or {}).get("speech") or "").strip()
    data = r.get("data") or {}
    return {"ok": kind in ("action_done", "query_answer"), "kind": kind, "speech": speech,
            "targets": [t.get("name") for t in data.get("success", []) if isinstance(t, dict)],
            "failed": [t.get("name") for t in data.get("failed", []) if isinstance(t, dict)],
            "code": data.get("code")}


async def _find_xiaoai(states: Optional[List[Dict[str, Any]]] = None) -> Optional[str]:
    configured_id = (getattr(config, "xiaoai_entity", "") or "").strip()
    if configured_id:
        return configured_id
    if states is None:
        states = await _request("GET", "/api/states")
    for s in states or []:
        eid = s.get("entity_id", "")
        if eid.startswith("media_player.") and any(k in eid for k in ("lx06", "wifispeaker", "xiaoai")):
            return eid
    return None


async def xiaoai(text: str, execute: bool = False) -> Dict[str, Any]:
    """小愛音箱: say `text` aloud, or (execute=True) run it as if spoken —
    "打開冷氣" reaches IR devices only 小愛 knows about."""
    speaker = await _find_xiaoai()
    if not speaker:
        raise HAError("Home Assistant 裡找不到小愛音箱。請確認 Xiaomi Miot Auto 已登入、音箱已加入，"
                      "或在 Settings 指定小愛音箱的 entity。")
    try:
        await _request("POST", "/api/services/xiaomi_miot/intelligent_speaker",
                       json={"entity_id": speaker, "text": text, "execute": bool(execute), "silent": False})
    except HAError as exc:
        # Older integration versions expose text entities instead of the service.
        suffix = "execute_text_directive" if execute else "play_text"
        target = speaker.replace("media_player.", "text.") + f"_{suffix}"
        try:
            await _request("POST", "/api/services/text/set_value", json={"entity_id": target, "value": text})
        except HAError:
            raise exc
    logger.info("Home: 小愛 %s %r", "command" if execute else "say", text)
    return {"done": True, "speaker": speaker, "mode": "command" if execute else "say", "text": text}
