// The toolbar popup: pair with an OpenTeddy, see whether it's connected,
// pause it, unpair. The work itself happens in background.js.

const $ = (id) => document.getElementById(id);
const STATES = {
  connected: '🟢 已連上',
  connecting: '🟡 連線中…',
  offline: '🔴 沒有連上',
  unpaired: '⚪ 還沒配對',
};

function host(url) {
  try { return new URL(url).host; } catch { return url; }
}

function showError(msg) {
  $('err').textContent = msg || '';
  $('err').hidden = !msg;
}

async function refresh() {
  const s = await chrome.runtime.sendMessage({ type: 'status' });
  if (!s || s.error) { showError(s && s.error); return; }
  let state = STATES[s.state] || STATES.offline;
  if (s.paired && s.paused) state = '⏸ 已暫停';
  $('state').textContent = s.paired && s.server ? `${state} · ${host(s.server)}` : state;
  $('pair').hidden = s.paired;
  $('paired').hidden = !s.paired;
  if (!s.paired && !$('server').value) $('server').value = s.server || '';
  $('pause').textContent = s.paused ? '繼續讓 OpenTeddy 操作' : '暫停 OpenTeddy 操作';
  $('pause').dataset.paused = s.paused ? '1' : '';
  $('detail').textContent = s.paired && s.state !== 'connected' ? s.detail : '';
  $('detail').hidden = !$('detail').textContent;
}

$('go').addEventListener('click', async () => {
  showError('');
  $('go').disabled = true;
  $('go').textContent = '連接中…';
  const r = await chrome.runtime.sendMessage({ type: 'pair', server: $('server').value, code: $('code').value });
  $('go').disabled = false;
  $('go').textContent = '連接';
  if (r && r.error) showError(r.error);
  else $('code').value = '';
  refresh();
});

$('code').addEventListener('keydown', (e) => { if (e.key === 'Enter') $('go').click(); });

$('pause').addEventListener('click', async () => {
  await chrome.runtime.sendMessage({ type: 'pause', value: !$('pause').dataset.paused });
  refresh();
});

$('unpair').addEventListener('click', async () => {
  if (!window.confirm('解除後 OpenTeddy 就不能再使用這個瀏覽器，要重新配對才行。確定嗎？')) return;
  await chrome.runtime.sendMessage({ type: 'unpair' });
  refresh();
});

chrome.storage.onChanged.addListener(() => refresh());
refresh();
