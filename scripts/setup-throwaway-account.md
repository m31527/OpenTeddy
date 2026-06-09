# Setting up a throwaway scraping account

The single highest-impact thing you can do to keep your main social
accounts safe from automated-scraping flags is to NOT scrape with them.
The browser running on the DGX is, from X / Threads / LinkedIn's
perspective, a brand new device that happens to be online 24/7 and
runs the exact same kind of search every day. Even with all of v1.1.6's
anti-bot hardening, that pattern is borderline. The clean solution is
a dedicated throwaway account whose only job is to run scraping
queries: if it gets flagged or shadow-banned, your main account
keeps full access.

This guide walks through setting up a clean throwaway account for X
specifically; the same principles apply to Threads, LinkedIn, Reddit,
Instagram, etc.

## What "clean throwaway" actually means

- **No real personal information.** No phone number that's also on
  your main account, no email that traces back to you, no birthday
  that matches yours.
- **A new email address.** Gmail / Outlook / ProtonMail / iCloud
  Hide-My-Email all work — anything that gives you a fresh inbox.
  Do NOT use a `+alias` on your real address (X correlates by
  reverse-mapping the alias).
- **Phone verification is the tricky bit.** X demands a phone number
  to complete signup. Options:
  - Use a second SIM (a cheap prepaid one is fine).
  - Burner-number services like TextNow / Google Voice work in most
    countries but X has begun rejecting them; YMMV.
  - Some operators report that signing up via the *web* (not the
    iOS / Android app) skips phone verification entirely for a
    while. Try web first.
- **Account stays low-profile.** Don't follow lots of accounts on
  the first day. Don't tweet. Don't like things programmatically.
  An account that does nothing but search is suspicious; an account
  that quietly searches once or twice a day for weeks before any
  other action is just lurkers — boring and ignored.

## Setting it up

### 1. Sign up

Do this **in a regular browser on your own computer**, not on the
DGX. The signup process triggers more aggressive anti-bot checks
than logged-in usage; doing it from your normal browser with your
normal IP is the path that least looks like automation.

- Open a private / incognito window
- Sign up at `https://x.com/i/flow/signup`
- Use the new email + (if required) a number you can receive a code at
- Set a clearly different display name from your real account — don't
  give X any "obvious correlation" hint

### 2. Warm it up

Before pointing OpenTeddy at it, spend ~10 minutes acting human:

- Follow 5-10 random public accounts in different topics (news, sport,
  someone funny, a tech company). Diversity > volume.
- Like 3-5 posts.
- Scroll your timeline for a few minutes.
- Optionally: post one innocuous tweet ("hello world", a meme).

This baseline of normal behaviour makes the later "this account does
nothing but search" pattern far less suspicious — X's system sees an
account that LOOKS like it's used by a real person, and gradually-
declining engagement just looks like someone who created an account
and lost interest.

### 3. Push the login to the DGX

From the throwaway account in your regular browser:
1. Log out and log back in once, so X's "device" record marks this as
   a confirmed live login (not just a signup-flow session).
2. Now go through the standard OpenTeddy flow:
   ```bash
   # On the DGX (AnyDesk / monitor / ssh -X)
   bash ~/OpenTeddy/scripts/login-helper.sh
   # Brave window opens → log in to the THROWAWAY account → close
   ```

The throwaway's cookies land in `/var/lib/openteddy/edge-profile`.
OpenTeddy will from now on scrape via this account.

## What about your main account?

Leave it logged in only on your daily-driver devices (phone, laptop).
Never let `login-helper.sh` open a window where you'd type the main
account credentials. If you ever do by accident:

```bash
# Wipe the profile and start over
sudo systemctl stop openteddy-cdp.service
sudo rm -rf /var/lib/openteddy/edge-profile/*
sudo systemctl start openteddy-cdp.service
bash ~/OpenTeddy/scripts/login-helper.sh   # log in to throwaway again
```

The cookies for the main account are gone. Note this doesn't log you
out of your main account on other devices — those have their own
cookie stores.

## How long does a throwaway last?

| Platform | Typical throwaway lifespan when used with v1.1.6 limits |
|---|---|
| X / Twitter | 3-12 months. Tends to die from "suspicious activity" flag rather than outright ban. Warning signs: search results suddenly drop to zero, "rate limit" toast on every page. |
| Threads / Instagram | 6+ months. Meta is sophisticated but slow to act on read-only scraping. |
| LinkedIn | 2-6 weeks. LinkedIn is notoriously aggressive — they'll lock unverified accounts that do anything unusual. |
| Reddit | Indefinite — Reddit has a public read-only JSON API; you can skip the throwaway entirely and just hit `/r/<sub>/.json`. |

When a throwaway dies, just register another. Treat them as consumable.
That's the whole point — your real account never sits in the
scraper's chair.

## Reading the hardening status

`chrome_attach_check` reports the v1.1.6 hardening state — handy for
"am I about to get the throwaway flagged?" sanity checks:

```bash
cd ~/OpenTeddy && .venv/bin/python -c "
import asyncio, json, sys; sys.path.insert(0, '.')
from tools.chrome_attached_tool import chrome_attach_check
r = asyncio.run(chrome_attach_check())
print(json.dumps(r['result']['hardening'], indent=2))
"
```

Output looks like:

```json
{
  "stealth_init_script": true,
  "sleep_window": {
    "start_hour": 2,
    "end_hour": 6,
    "currently_in": false,
    "disabled": false
  },
  "rate_limits": {
    "x_search": { "used_in_last_hour": 3, "cap_per_hour": 20, "remaining": 17 },
    "chrome_attached_browse": { "used_in_last_hour": 0, "cap_per_hour": 30, "remaining": 30 }
  }
}
```

If `remaining` drops to 0, schedule a longer interval. If you genuinely
need more headroom for one-off burst work:

```bash
OPENTEDDY_RATE_LIMIT_X_SEARCH=50 OPENTEDDY_NO_SLEEP=1 \
  .venv/bin/uvicorn main:app
```

— but every notch up is more lockout risk on the throwaway.
