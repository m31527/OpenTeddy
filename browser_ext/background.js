// OpenTeddy browser extension — background service worker.
//
// Holds one WebSocket to the person's own OpenTeddy (browser_bridge.py) and
// runs what it asks on this browser's tabs: list tabs, open a page, read a
// page as numbered text, click / type by number (content.js does the page
// side). Pairing, the pause switch and the server address live in
// chrome.storage.local; the connection state in chrome.storage.session,
// where the popup reads it.

const VERSION = chrome.runtime.getManifest().version;
const PAUSED_MSG = '你在擴充功能按了「暫停」，OpenTeddy 現在不能操作瀏覽器。要繼續請在擴充功能按「繼續」。';
let ws = null;
let retry = 0;
let retryTimer = null;
const confirmTabs = new Map();          // question id → tab it is shown in

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function stored() {
  return chrome.storage.local.get({ server: '', token: '', label: '', paused: false });
}

async function setState(state, detail = '') {
  await chrome.storage.session.set({ state, detail });
  const { paused, token } = await stored();
  const text = !token ? '' : paused ? '停' : state === 'connected' ? '' : '!';
  await chrome.action.setBadgeText({ text });
  await chrome.action.setBadgeBackgroundColor({ color: paused ? '#9a6700' : '#cf222e' });
}

function defaultLabel() {
  const ua = navigator.userAgentData;
  const p = (ua && ua.platform) || navigator.platform || '';
  const os = /mac/i.test(p) ? 'Mac' : /win/i.test(p) ? 'Windows' : /linux|cros/i.test(p) ? 'Linux' : '電腦';
  const brands = ((ua && ua.brands) || []).map((b) => b.brand).join(' ');
  const app = /Edge/.test(brands) ? 'Edge' : /Brave/.test(brands) ? 'Brave' : 'Chrome';
  return `${os} 的 ${app}`;
}

function normServer(s) {
  s = String(s || '').trim().replace(/\/+$/, '');
  if (s && !/^https?:\/\//i.test(s)) s = `http://${s}`;
  return s;
}

function hostOf(url) {
  try { return new URL(url).host; } catch { return ''; }
}

function friendly(e) {
  const m = String((e && e.message) || e || '');
  if (/Cannot access|cannot be scripted|chrome:\/\/|extensions gallery|Web Store|edge:\/\//i.test(m)) {
    return '這個頁面（瀏覽器設定頁、擴充功能商店、新分頁等）擴充功能不能讀取或操作。';
  }
  if (/No tab with id/i.test(m)) return '那個分頁已經關掉了。用 browser_tabs 看現在開著哪些。';
  if (/error page/i.test(m)) {
    return "This address didn't open (the site doesn't exist or can't be reached). Check the address — "
      + "don't guess one; look it up with web_search if you're not sure.";
  }
  if (/frame.*removed|Frame with ID/i.test(m)) return 'The page changed while it was being read — call browser_read again.';
  return m || '瀏覽器回報失敗';
}

// ── The connection ───────────────────────────────────────────────────────────

function send(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
}

function later() {
  clearTimeout(retryTimer);
  const wait = Math.min(30000, 1000 * 2 ** Math.min(retry++, 5));
  retryTimer = setTimeout(connect, wait);
}

async function connect() {
  const s = await stored();
  if (!s.server || !s.token) { await setState('unpaired'); return; }
  if (ws && ws.readyState <= WebSocket.OPEN) return;          // connecting or open
  clearTimeout(retryTimer);
  await setState('connecting');
  let sock;
  try {
    const u = new URL(s.server);
    u.protocol = u.protocol === 'https:' ? 'wss:' : 'ws:';
    u.pathname = '/browser/ws';
    u.search = '';
    u.hash = '';
    sock = new WebSocket(u.toString());
  } catch (e) {
    await setState('offline', friendly(e));
    later();
    return;
  }
  ws = sock;
  sock.onopen = () => sock.send(JSON.stringify({
    type: 'hello', token: s.token, label: s.label || defaultLabel(), version: VERSION, paused: s.paused,
  }));
  sock.onmessage = (ev) => onMessage(ev.data);
  sock.onclose = async () => {
    if (ws === sock) ws = null;
    if ((await stored()).token) {
      await setState('offline', '連不上 OpenTeddy，會自動重試。');
      later();
    }
  };
}

async function onMessage(raw) {
  let m;
  try { m = JSON.parse(raw); } catch { return; }
  if (m.type === 'welcome') {
    retry = 0;
    await setState('connected');
    // Loaded from a folder that has since been updated: pick the new files
    // up (once per version — a stale zip copy would loop otherwise).
    if (m.latest && m.latest !== VERSION) {
      const { reloadedFor } = await chrome.storage.local.get({ reloadedFor: '' });
      if (reloadedFor !== m.latest) {
        await chrome.storage.local.set({ reloadedFor: m.latest });
        chrome.runtime.reload();
      }
    }
  } else if (m.type === 'ping') {
    send({ type: 'pong' });
  } else if (m.type === 'unpaired') {
    await chrome.storage.local.set({ token: '' });
    await setState('unpaired', '配對已解除；需要的話請重新配對。');
  } else if (m.type === 'call') {
    let reply;
    try {
      reply = { type: 'result', id: m.id, ok: true, result: await run(m.op, m.params || {}) };
    } catch (e) {
      reply = { type: 'result', id: m.id, ok: false, error: friendly(e) };
    }
    send(reply);
  } else if (m.type === 'confirm') {
    await showConfirm(m);
  } else if (m.type === 'confirm_done') {
    await hideConfirm(m.id);
  }
}

// ── Operations ───────────────────────────────────────────────────────────────

async function run(op, p) {
  if ((await stored()).paused) throw new Error(PAUSED_MSG);
  if (op === 'tabs') return listTabs();
  if (op === 'open') return openTab(p);
  const tabId = await pickTab(p.tab);
  if (op === 'read' || op === 'describe' || op === 'describe_at') return inTab(tabId, op, p);
  if (op === 'click' || op === 'type' || op === 'click_at') return act(tabId, op, p);
  if (op === 'screenshot') return screenshot(tabId, p);
  throw new Error(`unknown operation ${op}`);
}

// What the page looks like, for the vision model (icon-only buttons,
// canvas). Only the visible part of the tab can be captured, so the tab is
// brought to the front first; OpenTeddy's own bar is hidden meanwhile.
async function screenshot(tabId, p) {
  const t = await chrome.tabs.get(tabId);
  if (!t.active) {
    await chrome.tabs.update(tabId, { active: true });
    await sleep(400);
  }
  const vp = await inPage(tabId, 'viewport', { scroll_down: p.scroll_down || 0 });
  let image;
  try {
    image = await chrome.tabs.captureVisibleTab(t.windowId, { format: 'jpeg', quality: 80 });
  } finally {
    await inPage(tabId, 'unhide', {}).catch(() => {});
  }
  return {
    tab: tabId, url: t.url, title: t.title, host: hostOf(t.url),
    image: image.split(',')[1], width: vp.width, height: vp.height,
  };
}

async function listTabs() {
  const tabs = await chrome.tabs.query({});
  const [front] = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
  return {
    tabs: tabs.filter((t) => /^https?:/.test(t.url || '')).map((t) => ({
      tab: t.id, title: t.title, url: t.url, active: !!front && t.id === front.id,
    })),
  };
}

async function openTab(p) {
  const url = String(p.url || '');
  if (!/^https?:\/\//i.test(url)) throw new Error('只能打開 http(s) 網址。');
  const t = await chrome.tabs.create({ url, active: false });
  await loaded(t.id, 20000);
  await inPage(t.id, 'settle', {}).catch(() => {});
  return inTab(t.id, 'read', { max_chars: p.max_chars });
}

async function pickTab(tab) {
  if (Number.isInteger(tab) && tab > 0) return (await chrome.tabs.get(tab)).id;
  const [front] = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
  if (!front) throw new Error('找不到你正在看的分頁。');
  return front.id;
}

// Ready to read: fully loaded, or the document is usable (readyState past
// "loading") for over a second. Shops like momo keep "loading" ads and
// trackers for 25 s+ after their content is there.
async function loaded(tabId, ms) {
  const t0 = Date.now();
  while (Date.now() - t0 < ms) {
    try {
      const t = await chrome.tabs.get(tabId);
      if (t.status === 'complete') return;
      if (Date.now() - t0 > 1200) {
        const [r] = await chrome.scripting.executeScript({ target: { tabId }, func: () => document.readyState });
        if (r && r.result && r.result !== 'loading') return;
      }
    } catch {
      // between pages, or the tab is gone — keep waiting until the cap
    }
    await sleep(250);
  }
}

async function inPage(tabId, op, p) {
  await chrome.scripting.executeScript({ target: { tabId }, files: ['content.js'] });
  const [res] = await chrome.scripting.executeScript({
    target: { tabId },
    func: (o, a) => window.__openteddy.run(o, a),
    args: [op, p],
  });
  const r = res && res.result;
  if (!r) throw new Error('頁面沒有回應。');
  if (r.error) throw new Error(r.error);
  return r;
}

async function inTab(tabId, op, p) {
  const r = await inPage(tabId, op, p);
  const t = await chrome.tabs.get(tabId);
  return { ...r, tab: tabId, url: t.url, title: t.title, host: hostOf(t.url) };
}

// Click / type, then wait for the page to settle (or the next page to load)
// and read it again, so OpenTeddy can see whether it worked.
async function act(tabId, op, p) {
  const before = await chrome.tabs.get(tabId);
  let opened = null;
  const onCreated = (t) => { if (t.openerTabId === tabId) opened = t.id; };
  chrome.tabs.onCreated.addListener(onCreated);
  let r;
  try {
    r = await inPage(tabId, op, p);                 // checks, then acts a moment later
    if (r.open_url) {
      opened = (await chrome.tabs.create({ url: r.open_url, active: false, openerTabId: tabId })).id;
    }
    await sleep(500);
  } finally {
    chrome.tabs.onCreated.removeListener(onCreated);
  }
  const target = opened || tabId;
  await loaded(target, 12000);
  await inPage(target, 'settle', {}).catch(() => {});
  const after = await chrome.tabs.get(target);
  let page = '';
  try {
    page = (await inPage(target, 'read', { max_chars: p.after_chars || 3000 })).page;
  } catch (e) {
    page = `(${friendly(e)})`;
  }
  const { open_url, ...rest } = r;
  return {
    ...rest, tab: target, new_tab: !!opened, navigated: after.url !== before.url,
    url: after.url, title: after.title, host: hostOf(after.url), page,
  };
}

// ── Asking the person first ──────────────────────────────────────────────────

async function showConfirm(m) {
  try {
    const tabId = await pickTab(m.tab);
    await chrome.scripting.executeScript({ target: { tabId }, files: ['content.js'] });
    await chrome.scripting.executeScript({
      target: { tabId }, func: (id, text) => window.__openteddy.ask(id, text), args: [m.id, m.text],
    });
    confirmTabs.set(m.id, tabId);
    await chrome.tabs.update(tabId, { active: true });
  } catch {
    // Not showable here (closed tab, chrome:// page) — Telegram still asks.
  }
}

async function hideConfirm(id) {
  const tabId = confirmTabs.get(id);
  confirmTabs.delete(id);
  if (!tabId) return;
  try {
    await chrome.scripting.executeScript({
      target: { tabId }, func: (i) => window.__openteddy && window.__openteddy.unask(i), args: [id],
    });
  } catch {
    // tab gone
  }
}

// ── Popup and page messages ──────────────────────────────────────────────────

async function pair(server, code) {
  server = normServer(server);
  if (!server) throw new Error('請填 OpenTeddy 網址。');
  if (!/^\d{6}$/.test(String(code || '').replace(/\D/g, ''))) throw new Error('配對碼是 6 位數字。');
  const label = defaultLabel();
  let res;
  try {
    res = await fetch(`${server}/browser/pair`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ code, label }),
    });
  } catch {
    throw new Error(`連不到 ${server}。請確認網址，以及這台電腦連得到 OpenTeddy。`);
  }
  const body = await res.json().catch(() => ({}));
  if (!res.ok || !body.token) throw new Error(body.detail || `配對失敗（${res.status}）`);
  await chrome.storage.local.set({ server, token: body.token, label, paused: false });
  retry = 0;
  if (ws) { try { ws.close(); } catch { /* already closed */ } ws = null; }
  await connect();
  return { ok: true };
}

async function onRuntimeMessage(msg, sender) {
  // The popup (an extension page) may pair / unpair; content scripts, which
  // run inside web pages, may only answer a question or pause.
  const fromPopup = String(sender.url || '').startsWith(chrome.runtime.getURL(''));
  if (msg.type === 'confirm') {
    send({ type: 'confirm', id: String(msg.id || ''), allow: !!msg.allow });
    confirmTabs.delete(msg.id);
    return {};
  }
  if (msg.type === 'pause') {
    await chrome.storage.local.set({ paused: !!msg.value });
    send({ type: 'paused', value: !!msg.value });
    await setState(ws && ws.readyState === WebSocket.OPEN ? 'connected' : 'offline');
    return {};
  }
  if (!fromPopup) return { error: 'not allowed' };
  if (msg.type === 'status') {
    const s = await stored();
    const st = await chrome.storage.session.get({ state: 'unpaired', detail: '' });
    return { state: st.state, detail: st.detail, server: s.server, paused: s.paused, paired: !!s.token,
             label: s.label || defaultLabel() };
  }
  if (msg.type === 'pair') return pair(msg.server, msg.code);
  if (msg.type === 'unpair') {
    send({ type: 'forget' });
    await sleep(200);
    await chrome.storage.local.set({ token: '' });
    if (ws) { try { ws.close(); } catch { /* already closed */ } ws = null; }
    await setState('unpaired');
    return {};
  }
  return { error: `unknown message ${msg.type}` };
}

chrome.runtime.onMessage.addListener((msg, sender, reply) => {
  onRuntimeMessage(msg || {}, sender).then(reply, (e) => reply({ error: friendly(e) }));
  return true;
});

// Wake up and reconnect: on browser start, on install, and every 30 s (a
// worker that went to sleep comes back through the alarm).
chrome.alarms.create('openteddy-keepalive', { periodInMinutes: 0.5 });
chrome.alarms.onAlarm.addListener((a) => { if (a.name === 'openteddy-keepalive') connect(); });
chrome.runtime.onStartup.addListener(connect);
chrome.runtime.onInstalled.addListener(connect);
connect();
