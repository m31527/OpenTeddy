// OpenTeddy — the page side. Injected into a tab only when OpenTeddy reads
// or acts on it (background.js). Turns the page into text where every link,
// button and field has a number, clicks / types by number, and shows the
// "OpenTeddy is using this tab" bar and the allow / don't card.
//
// It runs in the extension's isolated world: the page's own scripts can't
// call these functions or reach inside the bar (closed shadow root), and
// only a real click on 「允許」 counts (event.isTrusted).
(() => {
  if (window.__openteddy) return;

  const SKIP = new Set(['SCRIPT', 'STYLE', 'NOSCRIPT', 'TEMPLATE', 'CANVAS', 'IFRAME', 'OBJECT', 'EMBED',
    'HEAD', 'META', 'LINK', 'VIDEO', 'AUDIO', 'MAP']);
  const BLOCK = new Set(['ADDRESS', 'ARTICLE', 'ASIDE', 'BLOCKQUOTE', 'DD', 'DETAILS', 'DIV', 'DL', 'DT',
    'FIELDSET', 'FIGCAPTION', 'FIGURE', 'FOOTER', 'FORM', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'HEADER',
    'HR', 'LI', 'MAIN', 'NAV', 'OL', 'P', 'PRE', 'SECTION', 'TABLE', 'TR', 'UL', 'DIALOG', 'LABEL']);
  const ROLES = new Set(['button', 'link', 'checkbox', 'radio', 'tab', 'menuitem', 'menuitemcheckbox',
    'menuitemradio', 'option', 'switch', 'combobox', 'textbox', 'searchbox', 'slider', 'spinbutton', 'treeitem']);
  const NATIVE = 'a[href],button,input:not([type=hidden]),select,textarea,[role=button],[role=link]';
  const SECRET = /(pass(word|wd)?|\bpwd\b|密碼|one.?time|\botp\b|驗證碼|cvv|cvc|\bcsc\b|card.?(no|num)|cc.?num|卡號|信用卡|安全碼|有效期|身分證|\bssn\b)/i;

  let refs = new Map();                 // n → WeakRef(element), from the latest read
  let hostEl = null;
  let root = null;
  let bar = null;
  let card = null;
  let barTimer = null;

  const squash = (s) => String(s || '').replace(/\s+/g, ' ').trim();
  const clip = (s, n) => {
    s = squash(s);
    return s.length > n ? `${s.slice(0, n - 1)}…` : s;
  };

  // ── Reading the page ──────────────────────────────────────────────────────

  function visible(el) {
    if (el.getAttribute('aria-hidden') === 'true') return false;
    if (typeof el.checkVisibility === 'function') {
      if (el.checkVisibility({ checkVisibilityCSS: true })) return true;
      return getComputedStyle(el).display === 'contents';
    }
    const cs = getComputedStyle(el);
    return cs.display !== 'none' && cs.visibility !== 'hidden';
  }

  function isInteractive(el) {
    const tag = el.tagName;
    const role = (el.getAttribute('role') || '').toLowerCase();
    if (tag === 'A') return el.hasAttribute('href') || ROLES.has(role);
    if (tag === 'BUTTON' || tag === 'SELECT' || tag === 'TEXTAREA' || tag === 'SUMMARY') return true;
    if (tag === 'INPUT') return (el.type || '').toLowerCase() !== 'hidden';
    if (ROLES.has(role)) return true;
    if (el.isContentEditable) return !(el.parentElement && el.parentElement.isContentEditable);
    if (el.hasAttribute('onclick')) return true;
    const ti = el.getAttribute('tabindex');
    return ti !== null && ti !== '-1' && el !== document.body;
  }

  // A link / button wrapping a whole card (picture, title, price, more
  // buttons): number it, and still read what's inside.
  function isContainer(el) {
    if (/^(INPUT|SELECT|TEXTAREA)$/.test(el.tagName) || el.isContentEditable) return false;
    if (el.querySelector(NATIVE)) return true;
    return squash(el.innerText).length > 100;
  }

  function kindOf(el) {
    const tag = el.tagName;
    const role = (el.getAttribute('role') || '').toLowerCase();
    if (tag === 'A') return ROLES.has(role) && role !== 'link' ? role : 'link';
    if (tag === 'SELECT') return 'select';
    if (tag === 'TEXTAREA') return 'textbox';
    if (tag === 'SUMMARY') return 'toggle';
    if (tag === 'INPUT') {
      const t = (el.type || 'text').toLowerCase();
      if (t === 'checkbox' || t === 'radio') return t;
      if (/^(submit|button|reset|image)$/.test(t)) return 'button';
      if (t === 'search') return 'searchbox';
      if (t === 'file') return 'file';
      if (t === 'range') return 'slider';
      return 'textbox';
    }
    if (tag === 'BUTTON') return ROLES.has(role) ? role : 'button';
    if (role) return role;
    if (el.isContentEditable) return 'textbox';
    return 'button';
  }

  function nameOf(el) {
    const aria = el.getAttribute('aria-label');
    if (squash(aria)) return aria;
    const by = el.getAttribute('aria-labelledby');
    if (by) {
      const t = by.split(/\s+/).map((id) => document.getElementById(id)).filter(Boolean)
        .map((e) => e.innerText).join(' ');
      if (squash(t)) return t;
    }
    if (el.labels && el.labels.length) {
      const t = Array.from(el.labels).map((l) => labelText(l, el)).join(' ');
      if (squash(t)) return t;
    }
    const tag = el.tagName;
    if (tag === 'INPUT' && /^(submit|button|reset)$/i.test(el.type) && el.value) return el.value;
    if (tag === 'INPUT' && el.type === 'image') return el.alt || el.value || '';
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') {
      return el.placeholder || el.title || el.getAttribute('aria-placeholder') || el.name || '';
    }
    if (squash(el.innerText)) return el.innerText;
    const img = el.querySelector('img[alt]');
    if (img && squash(img.alt)) return img.alt;
    if (el.title || el.getAttribute('data-tooltip')) return el.title || el.getAttribute('data-tooltip');
    if (tag === 'A') {
      try { return new URL(el.href).pathname; } catch { return ''; }
    }
    return '';
  }

  // A <label> wrapping its field: its own words, not the field's (a
  // dropdown's options would otherwise end up in the name).
  function labelText(label, control) {
    let s = '';
    const w = document.createTreeWalker(label, NodeFilter.SHOW_TEXT);
    for (let t = w.nextNode(); t; t = w.nextNode()) {
      if (!control.contains(t)) s += `${t.nodeValue} `;
    }
    return s;
  }

  function valueOf(el) {
    if (el.tagName === 'SELECT') {
      const o = el.selectedOptions && el.selectedOptions[0];
      return o ? o.text : '';
    }
    if (el.isContentEditable && !/^(INPUT|TEXTAREA)$/.test(el.tagName)) return el.innerText;
    return el.value || '';
  }

  function isSecret(el) {
    if ((el.type || '').toLowerCase() === 'password') return true;
    const ac = (el.getAttribute('autocomplete') || '').toLowerCase();
    if (/^(cc-|current-password|new-password|one-time-code)/.test(ac)) return true;
    return SECRET.test([el.name, el.id, nameOf(el), el.placeholder].join(' '));
  }

  function line(n, el, container) {
    const kind = kindOf(el);
    let s = `[${n}] ${kind} "${clip(nameOf(el), container ? 50 : 80)}"`;
    if (kind === 'textbox' || kind === 'searchbox' || kind === 'combobox') {
      s += isSecret(el) ? ' (password / card / code — the person types this)' : ` = "${clip(valueOf(el), 60)}"`;
    } else if (kind === 'select') {
      const opts = Array.from(el.options || []).map((o) => clip(o.text, 25)).filter(Boolean);
      s += ` = "${clip(valueOf(el), 40)}"`;
      if (opts.length) s += ` (options: ${opts.slice(0, 8).join(' | ')}${opts.length > 8 ? ' …' : ''})`;
    } else if (kind === 'checkbox' || kind === 'radio' || kind === 'switch') {
      s += (el.checked || el.getAttribute('aria-checked') === 'true') ? ' ✓' : ' ☐';
    }
    if (el.disabled || el.getAttribute('aria-disabled') === 'true') s += ' (disabled)';
    return s;
  }

  function snapshot(find, maxChars) {
    refs = new Map();
    let n = 0;
    let mute = 0;                       // inside a <label> whose words are already its field's name
    const lines = [];
    let buf = '';
    const MARKS = /^[•#|\s]*$/;         // a bullet / heading / cell mark with nothing after it
    const flush = () => {
      const t = squash(buf);
      if (t && !MARKS.test(t)) lines.push(t.length > 400 ? `${t.slice(0, 399)}…` : t);
      buf = '';
    };
    const walk = (parent) => {
      for (let c = parent.firstChild; c; c = c.nextSibling) visit(c);
    };
    const visit = (node) => {
      if (node.nodeType === Node.TEXT_NODE) {
        if (!mute) buf += node.nodeValue;
        return;
      }
      if (node.nodeType !== Node.ELEMENT_NODE) return;
      const el = node;
      const tag = el.tagName;
      if (el === hostEl || SKIP.has(tag) || el instanceof SVGElement || !visible(el)) return;
      if (tag === 'IMG') {
        const alt = squash(el.alt);
        if (alt.length > 1) buf += ` [image: ${clip(alt, 60)}] `;
        return;
      }
      if (tag === 'BR') { flush(); return; }
      if (isInteractive(el)) {
        // "• " / "## " just before it belongs on the element's own line.
        const lead = squash(buf);
        if (MARKS.test(lead)) buf = '';
        else flush();
        n += 1;
        refs.set(n, new WeakRef(el));
        const container = isContainer(el);
        lines.push(`${MARKS.test(lead) && lead ? `${lead} ` : ''}${line(n, el, container)}`);
        if (!container) return;
      }
      const block = BLOCK.has(tag);
      if (block) flush();
      if (/^H[1-6]$/.test(tag)) buf += `${'#'.repeat(Number(tag[1]))} `;
      else if (tag === 'LI') buf += '• ';
      else if (tag === 'TD' || tag === 'TH') buf += ' | ';
      const quiet = tag === 'LABEL' && el.control && el.contains(el.control);
      if (quiet) mute += 1;
      if (el.shadowRoot) walk(el.shadowRoot);
      walk(el);
      if (quiet) mute -= 1;
      if (block) flush();
    };
    if (document.body) visit(document.body);
    flush();

    let shown = lines;
    if (find) {
      const words = find.toLowerCase().split(/\s+/).filter(Boolean);
      shown = lines.filter((l) => words.some((w) => l.toLowerCase().includes(w)));
      if (!shown.length) shown = [`(nothing on this page matches "${find}")`];
    }
    let text = '';
    let i = 0;
    for (; i < shown.length; i += 1) {
      if (text.length + shown[i].length + 1 > maxChars) break;
      text += `${shown[i]}\n`;
    }
    if (i < shown.length) {
      text += `… (${shown.length - i} more lines not shown — browser_read with find="…" looks for something specific)`;
    }
    return { page: text || '(this page has no readable text)', elements: n };
  }

  // ── Acting ────────────────────────────────────────────────────────────────

  function deref(n) {
    const ref = refs.get(Number(n));
    const el = ref && ref.deref();
    if (!el || !el.isConnected) {
      throw new Error(`Element [${n}] isn't on the page any more — the page changed. Call browser_read again and use the new numbers.`);
    }
    return el;
  }

  function submits(el) {
    const t = (el.type || '').toLowerCase();
    if (el.tagName === 'BUTTON') return !!el.form && (t === 'submit' || !el.getAttribute('type'));
    if (el.tagName === 'INPUT') return !!el.form && (t === 'submit' || t === 'image');
    return false;
  }

  function isSearchForm(f) {
    if (f.getAttribute('role') === 'search' || f.closest('[role=search]') || f.querySelector('input[type=search]')) return true;
    return /search|query|\bq\b|find|搜尋/i.test(`${f.getAttribute('action') || ''} ${f.id} ${f.className} ${f.getAttribute('name') || ''}`);
  }

  function describe(p) {
    const el = deref(p.n);
    const form = el.form || el.closest('form');
    return {
      n: Number(p.n), kind: kindOf(el), name: clip(nameOf(el), 80), tag: el.tagName.toLowerCase(),
      type: (el.type || '').toLowerCase(), role: el.getAttribute('role') || '',
      autocomplete: el.getAttribute('autocomplete') || '', field: squash(`${el.name || ''} ${el.id || ''}`),
      secret: isSecret(el), submits: submits(el),
      form: form ? { search: isSearchForm(form), method: (form.method || 'get').toLowerCase(),
                     action: form.getAttribute('action') || '' } : null,
    };
  }

  function press(el) {
    el.scrollIntoView({ block: 'center', inline: 'center' });
    const r = el.getBoundingClientRect();
    const o = { bubbles: true, cancelable: true, composed: true, button: 0, buttons: 1,
                clientX: r.left + r.width / 2, clientY: r.top + r.height / 2 };
    el.dispatchEvent(new PointerEvent('pointerdown', { ...o, pointerType: 'mouse', isPrimary: true }));
    el.dispatchEvent(new MouseEvent('mousedown', o));
    if (typeof el.focus === 'function') el.focus({ preventScroll: true });
    el.dispatchEvent(new PointerEvent('pointerup', { ...o, buttons: 0, pointerType: 'mouse', isPrimary: true }));
    el.dispatchEvent(new MouseEvent('mouseup', { ...o, buttons: 0 }));
    el.click();
  }

  function click(p) {
    const el = deref(p.n);
    if (el.disabled || el.getAttribute('aria-disabled') === 'true') {
      throw new Error(`[${p.n}] is disabled — it can't be clicked right now.`);
    }
    const what = `${kindOf(el)} "${clip(nameOf(el), 60)}"`;
    busy();
    // A link that opens a new tab: the background opens it (a click from
    // here would be stopped as a pop-up).
    if (el.tagName === 'A' && el.target === '_blank' && /^https?:/.test(el.href)) return { clicked: what, open_url: el.href };
    setTimeout(() => press(el), 30);         // answer first: the click may unload this page
    return { clicked: what };
  }

  function enter(el) {
    const o = { key: 'Enter', code: 'Enter', keyCode: 13, which: 13, bubbles: true, cancelable: true, composed: true };
    const go = el.dispatchEvent(new KeyboardEvent('keydown', o));
    el.dispatchEvent(new KeyboardEvent('keypress', o));
    el.dispatchEvent(new KeyboardEvent('keyup', o));
    const form = el.form || el.closest('form');
    if (go && form && el.tagName !== 'TEXTAREA') {
      if (form.requestSubmit) form.requestSubmit();
      else form.submit();
    }
  }

  function type(p) {
    const el = deref(p.n);
    if (isSecret(el)) throw new Error('這是密碼、卡號或驗證碼欄位，OpenTeddy 不會填，請你自己輸入。');
    const kind = kindOf(el);
    const text = String(p.text == null ? '' : p.text);
    if (el.disabled || el.readOnly) throw new Error(`[${p.n}] can't be typed into right now (disabled or read-only).`);
    if (/^(checkbox|radio|button|link|file|toggle|tab|menuitem)$/.test(kind)) {
      throw new Error(`[${p.n}] is a ${kind}, not a text field — use browser_click for it.`);
    }
    busy();
    if (el.tagName === 'SELECT') {
      const want = squash(text).toLowerCase();
      const opts = Array.from(el.options);
      const o = opts.find((x) => squash(x.text).toLowerCase() === want || x.value.toLowerCase() === want)
        || opts.find((x) => squash(x.text).toLowerCase().includes(want));
      if (!o) throw new Error(`No option like "${text}". Options: ${opts.slice(0, 15).map((x) => squash(x.text)).join(' | ')}`);
      el.value = o.value;
      el.dispatchEvent(new Event('input', { bubbles: true }));
      el.dispatchEvent(new Event('change', { bubbles: true }));
      return { typed_into: `select "${clip(nameOf(el), 60)}"`, value: squash(o.text) };
    }
    el.scrollIntoView({ block: 'center' });
    el.focus();
    if (el.isContentEditable && !/^(INPUT|TEXTAREA)$/.test(el.tagName)) {
      const range = document.createRange();
      range.selectNodeContents(el);
      const sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
      if (!document.execCommand('insertText', false, text)) el.innerText = text;
    } else {
      // The prototype's setter, so frameworks that watch the element's own
      // value property (React) see a change and fire their onChange.
      const proto = el.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
      Object.getOwnPropertyDescriptor(proto, 'value').set.call(el, text);
      el.dispatchEvent(new InputEvent('input', { bubbles: true, composed: true, inputType: 'insertText', data: text }));
      el.dispatchEvent(new Event('change', { bubbles: true }));
    }
    const res = { typed_into: `${kind} "${clip(nameOf(el), 60)}"`, value: clip(valueOf(el), 80) };
    if (p.submit) setTimeout(() => enter(el), 30);
    return res;
  }

  // ── By position, for what the vision model found on a screenshot ──────────

  function viewport(p) {
    const down = Number(p.scroll_down) || 0;
    if (down) window.scrollBy(0, Math.round(window.innerHeight * 0.8 * down));
    if (hostEl) hostEl.style.display = 'none';
    return { width: window.innerWidth, height: window.innerHeight, scroll_y: Math.round(window.scrollY) };
  }

  function unhide() {
    if (hostEl) hostEl.style.display = '';
    return {};
  }

  function elementAt(x, y) {
    if (hostEl) hostEl.style.display = 'none';
    const el = document.elementFromPoint(Number(x), Number(y));
    if (hostEl) hostEl.style.display = '';
    if (!el) throw new Error(`Nothing at (${Math.round(x)}, ${Math.round(y)}) on the page.`);
    return el;
  }

  function describeAt(p) {
    const at = elementAt(p.x, p.y);
    const el = at.closest('a[href],button,input,select,textarea,summary,[role=button],[role=link],[onclick]') || at;
    const form = el.form || el.closest('form');
    return {
      kind: isInteractive(el) ? kindOf(el) : el.tagName.toLowerCase(), name: clip(nameOf(el), 80),
      tag: el.tagName.toLowerCase(), type: (el.type || '').toLowerCase(), submits: submits(el),
      form: form ? { search: isSearchForm(form) } : null,
    };
  }

  function ring(x, y) {
    stack();
    const r = node('div', 'ring');
    r.style.left = `${x - 14}px`;
    r.style.top = `${y - 14}px`;
    root.append(r);
    setTimeout(() => r.remove(), 900);
  }

  function clickAt(p) {
    const x = Number(p.x);
    const y = Number(p.y);
    const el = elementAt(x, y);
    const what = `${isInteractive(el) ? kindOf(el) : el.tagName.toLowerCase()} "${clip(nameOf(el), 60)}"`;
    busy();
    ring(x, y);
    // Events carry the position, so pages drawn on a canvas (maps, games,
    // charts) see the click where the vision model found the thing.
    setTimeout(() => {
      const o = { bubbles: true, cancelable: true, composed: true, button: 0, buttons: 1, clientX: x, clientY: y };
      el.dispatchEvent(new PointerEvent('pointerdown', { ...o, pointerType: 'mouse', isPrimary: true }));
      el.dispatchEvent(new MouseEvent('mousedown', o));
      if (typeof el.focus === 'function') el.focus({ preventScroll: true });
      el.dispatchEvent(new PointerEvent('pointerup', { ...o, buttons: 0, pointerType: 'mouse', isPrimary: true }));
      el.dispatchEvent(new MouseEvent('mouseup', { ...o, buttons: 0 }));
      el.dispatchEvent(new MouseEvent('click', { ...o, buttons: 0 }));
    }, 30);
    return { clicked: what };
  }

  function settle(max = 2500, quiet = 350) {
    return new Promise((resolve) => {
      let t = null;
      let cap = null;
      const obs = new MutationObserver(() => {
        clearTimeout(t);
        t = setTimeout(done, quiet);
      });
      function done() {
        obs.disconnect();
        clearTimeout(t);
        clearTimeout(cap);
        resolve({ settled: true });
      }
      obs.observe(document.documentElement, { subtree: true, childList: true, attributes: true, characterData: true });
      t = setTimeout(done, quiet);
      cap = setTimeout(done, max);
    });
  }

  // ── The bar and the question card ─────────────────────────────────────────

  const CSS = `
    .stack { display: flex; flex-direction: column; align-items: flex-end; gap: 8px;
      font: 13px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang TC", "Noto Sans TC", sans-serif; }
    .bar { display: flex; align-items: center; gap: 8px; padding: 6px 6px 6px 12px; border-radius: 999px;
      background: #1f2328; color: #fff; box-shadow: 0 4px 16px rgba(0,0,0,.25); }
    .dot { width: 8px; height: 8px; border-radius: 50%; background: #3fb950; animation: pulse 1.2s infinite; }
    @keyframes pulse { 50% { opacity: .35; } }
    button { font: inherit; border: 0; border-radius: 999px; padding: 4px 12px; cursor: pointer; }
    .bar button { background: rgba(255,255,255,.16); color: #fff; }
    .card { width: 300px; box-sizing: border-box; padding: 14px; border-radius: 14px; background: #fff; color: #1f2328;
      border: 1px solid #d0d7de; box-shadow: 0 10px 30px rgba(0,0,0,.25); }
    .title { font-weight: 600; margin-bottom: 6px; }
    .q { margin-bottom: 12px; word-break: break-word; }
    .row { display: flex; justify-content: flex-end; gap: 8px; }
    .no { background: #eaeef2; color: #1f2328; }
    .yes { background: #c2410c; color: #fff; }
    .ring { position: fixed; width: 28px; height: 28px; box-sizing: border-box; border-radius: 50%;
      border: 3px solid #c2410c; pointer-events: none; animation: ring .9s ease-out forwards; }
    @keyframes ring { to { transform: scale(1.8); opacity: 0; } }
  `;

  function node(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }

  function stack() {
    if (!hostEl || !hostEl.isConnected) {
      hostEl = document.createElement('div');
      hostEl.style.cssText = 'all: initial; position: fixed; top: 12px; right: 12px; z-index: 2147483647;';
      document.documentElement.appendChild(hostEl);
      root = hostEl.attachShadow({ mode: 'closed' });
      root.append(node('style', null, CSS), node('div', 'stack'));
      bar = null;
      card = null;
    }
    return root.querySelector('.stack');
  }

  function busy() {
    const s = stack();
    if (!bar) {
      bar = node('div', 'bar');
      const label = node('span', null, 'OpenTeddy 正在使用這個分頁');
      const stop = node('button', null, '停止');
      stop.type = 'button';
      stop.addEventListener('click', (e) => {
        if (!e.isTrusted) return;
        chrome.runtime.sendMessage({ type: 'pause', value: true });
        label.textContent = '已暫停：OpenTeddy 不會再操作這個瀏覽器';
        stop.remove();
      });
      bar.append(node('span', 'dot'), label, stop);
      s.prepend(bar);
    }
    clearTimeout(barTimer);
    barTimer = setTimeout(() => { if (bar) { bar.remove(); bar = null; } }, 8000);
  }

  function ask(id, text) {
    const s = stack();
    if (card) card.remove();
    card = node('div', 'card');
    card.dataset.id = id;
    const row = node('div', 'row');
    const no = node('button', 'no', '不要');
    const yes = node('button', 'yes', '允許');
    const answer = (allow) => (e) => {
      if (!e.isTrusted) return;                // only the person's own click
      chrome.runtime.sendMessage({ type: 'confirm', id, allow });
      if (card) { card.remove(); card = null; }
    };
    no.type = 'button';
    yes.type = 'button';
    no.addEventListener('click', answer(false));
    yes.addEventListener('click', answer(true));
    row.append(no, yes);
    card.append(node('div', 'title', '🐻 OpenTeddy 想先問你'), node('div', 'q', text), row);
    s.append(card);
    return { shown: true };
  }

  function unask(id) {
    if (card && card.dataset.id === id) { card.remove(); card = null; }
    return {};
  }

  async function run(op, p) {
    try {
      p = p || {};
      if (op === 'read') { busy(); return snapshot(String(p.find || ''), Number(p.max_chars) || 6000); }
      if (op === 'describe') return describe(p);
      if (op === 'click') return click(p);
      if (op === 'type') return type(p);
      if (op === 'settle') return await settle();
      if (op === 'viewport') return viewport(p);
      if (op === 'unhide') return unhide();
      if (op === 'describe_at') return describeAt(p);
      if (op === 'click_at') return clickAt(p);
      return { error: `unknown operation ${op}` };
    } catch (e) {
      return { error: String((e && e.message) || e) };
    }
  }

  window.__openteddy = { run, ask, unask };
})();
