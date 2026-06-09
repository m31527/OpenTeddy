# Capturing a `storage_state.json` for OpenTeddy CDP

`storage_state.json` is Playwright's standard format for bottling up a
browser's cookies + localStorage so a different browser process can pick
up the same logged-in session. OpenTeddy's `chrome_attached_tool` looks
for one at `/var/lib/openteddy/storage_state.json` (and a couple of
fallback paths) and injects it on every CDP attach.

The capture has to happen **once on a machine with a display** — there's
no way around it, you need to physically click "Sign in" and enter your
password into a real browser window. After that, the resulting JSON can
be `scp`'d to every fleet node and reused indefinitely (until the site's
session token expires, typically 30-90 days for X / LinkedIn / etc.).

## On your laptop / dev machine (macOS, Linux, Windows)

```bash
# 1. Install Playwright (in a venv if you like — these are pure JS bindings)
pip install playwright
playwright install chromium

# 2. Open the recorder + capture state at exit
playwright codegen https://x.com/login --save-storage=storage_state.json

# 3. In the browser window that pops up:
#    - Log in to X normally (username / password / 2FA, whatever)
#    - Close the window when you see your timeline
#    → storage_state.json is now sitting next to your terminal

# 4. Sanity check — cookie count should be 20-40 for X
python3 -c 'import json; d = json.load(open("storage_state.json")); \
            print(f"{len(d[\"cookies\"])} cookies for", \
                  ", ".join(sorted({c["domain"] for c in d["cookies"]})))'
```

You should see something like:

```
38 cookies for .twitter.com, .x.com, x.com
```

## Push to each fleet node

```bash
# Replace dgx-01..dgx-N with your node hostnames
for host in dgx-01 dgx-02 dgx-03; do
  scp storage_state.json admin@${host}:/tmp/
  ssh admin@${host} '
    sudo mv /tmp/storage_state.json /var/lib/openteddy/storage_state.json
    sudo chown admin:admin /var/lib/openteddy/storage_state.json
    sudo chmod 600           /var/lib/openteddy/storage_state.json
    sudo systemctl restart openteddy-cdp.service
  '
done
```

The systemd restart is so Edge picks up the cookies on its next boot.
Alternatively, OpenTeddy's `chrome_attached_tool` will inject them at
attach time even without a restart — but a fresh boot is cleaner.

## How long does the captured state last?

| Site | Typical session lifetime |
|---|---|
| X / Twitter | 30 days (sliding — extended on each login activity) |
| LinkedIn | ~90 days |
| Google services | 14 days (then refreshes silently if you opt in) |
| Slack | ~30 days |

When `x_search` starts returning empty results across all fleet nodes
simultaneously, that's the signal: re-capture and re-deploy. Recommend
scripting that as a quarterly cron, OR triggering it from a fleet
heartbeat ("if 3 nodes report login_expired, alert ops").

## Security notes

- `storage_state.json` IS a credential. Anyone with the file can pretend
  to be you on X / LinkedIn / etc. Treat it like an SSH private key:
  - `chmod 600` after copying.
  - Don't commit to git.
  - Don't paste into chat / Telegram / etc.
- Each fleet node runs as the same user (`OPENTEDDY_USER` from the setup
  script). Use `sudo` access controls if you want only ops to be able to
  read the file.
- The CDP service listens on `127.0.0.1` only — never expose 9222 to the
  LAN. Anyone reachable on 9222 can read every cookie in the profile
  through DevTools. Treat it like the Docker daemon socket: localhost
  trust only.
