# Amazon Shopping Integration — Setup Guide

**Branch:** `dev/seb-amazon-shopping`
**Author:** @sebastiandavidlee
**Status:** end-to-end verified — voice command → item in real Amazon cart, zero clicks during demo

## What this adds

Two new skills on top of the existing `app/` pipeline:

| Skill | Trigger | What it does |
|---|---|---|
| `amazon_search` | "find me apples on amazon", "search amazon for X", "buy paper towels" | Calls SerpAPI's Amazon engine, picks the top organic result, speaks back title + price + rating + a follow-up question |
| `amazon_add_to_cart` | "yes add it", "add to cart", "yeah go ahead" | Reads cookies from your logged-in browser, launches Playwright Chromium, auto-submits the Amazon "Add to Cart" form on your account |

Two-turn flow with explicit confirmation. The planner has an **anti-chain constraint** so `amazon_add_to_cart` can never run in the same turn as `amazon_search` — the user always confirms.

The item lands in **your** Amazon cart (whoever's logged into amazon.com on this machine), not anyone else's. Each teammate's machine reads their own browser cookies.

## What this does NOT change

- Existing voice skills (`go_home`, `read_notes`, `general_conversation`, `pick_blue_marker`, etc.) — all still work as before
- Existing API keys (`OPENAI_API_KEY`, `ELEVENLABS_API_KEY`) — same as main
- Robot/arm/camera/mic — untouched
- Main branch — untouched until we merge

## Prerequisites

You should already have:
- The main branch demo working on your machine (mic, TTS, orchestrator end-to-end)
- A Python env with the existing `alfred_orchestrator/requirements.txt` installed
- `OPENAI_API_KEY` and `ELEVENLABS_API_KEY` in `alfred_orchestrator/.env`

If main wasn't working for you yet, fix that first. This guide is the **delta** only.

## TL;DR — single block to copy-paste

```bash
# 1. Switch to the branch
git fetch origin
git checkout dev/seb-amazon-shopping

# 2. Activate your existing Python env (venv/conda/whatever you already use)
#    Example: source .venv/bin/activate   OR   conda activate alfred

# 3. Install the new Python deps
pip install -r alfred_orchestrator/requirements.txt

# 4. Install the Chromium browser binary Playwright uses
playwright install chromium

# 5. Linux only — install system libs Chromium needs (may prompt for sudo)
playwright install-deps chromium

# 6. Add the SerpAPI key to your .env (sign up at https://serpapi.com/manage-api-key, free 100/mo)
echo "SERPAPI_KEY=<paste-your-key-here>" >> alfred_orchestrator/.env

# 7. Make sure Firefox or Chrome on this machine is logged into amazon.com
#    (no command — just open the browser and confirm you're signed in)

# 8. Done. Run your normal orchestrator entry point.
```

## Step-by-step explanation

### Step 1 — Switch to the branch

```bash
git fetch origin
git checkout dev/seb-amazon-shopping
```

### Step 2 — Activate your Python env

Use whatever env you normally use to run the orchestrator. There's no enforced standard in the repo — venv, conda, or system Python all work.

### Step 3 — Install new Python deps

```bash
pip install -r alfred_orchestrator/requirements.txt
```

Adds 3 packages on top of what was already there:
- `playwright>=1.40` — drives the Chromium browser for auto cart-add
- `browser-cookie3>=0.19` — reads your existing browser's amazon.com cookies
- `scipy` — was already a transitive dep through `app/skills/listen.py` but wasn't in requirements before; now explicit

### Step 4 — Download the Chromium browser binary

```bash
playwright install chromium
```

This downloads ~150 MB into `~/.cache/ms-playwright/`. Playwright uses its own bundled Chromium — does NOT touch your existing Firefox/Chrome.

### Step 5 — Linux only: install Chromium's system libs

```bash
playwright install-deps chromium
```

Installs system libraries Chromium needs (`libnss3`, `libatk-bridge2.0-0`, etc.). Asks for sudo. **macOS users skip this step.**

### Step 6 — Add the SerpAPI key

Sign up free at https://serpapi.com/manage-api-key — 100 searches/month free, no credit card needed. Then add to your `.env`:

```bash
# alfred_orchestrator/.env (this file is gitignored — safe to put secrets in)
SERPAPI_KEY=<your-serpapi-key>
```

### Step 7 — Confirm you're logged into amazon.com

Open Firefox (preferred — works with Snap or native) or Chrome and confirm you're signed into your Amazon account. **Cart-add lands items in this same account.** Cookies are read at runtime from whichever browser you've authenticated with most recently.

### Step 8 — Run the orchestrator the way you normally do

Whatever your normal entry point is (e.g. `python alfred_orchestrator/examples/interactive_voice_loop.py`), no command change needed.

## How to test it works

After running the orchestrator and connecting your phone to the bridge URL:

| Voice prompt | Expected behavior |
|---|---|
| "find me a USB-C cable on amazon" | Alfred speaks: "Found Anker Cable, 9 99 dollars, 4.7 stars. ... Want me to add it?" |
| "yes add it" | Headed Chromium pops on your screen, navigates to Amazon, auto-clicks Add To Cart, closes. Item appears in your Amazon cart on phone/web. |
| "go home" | Existing skill — should still work, robot motion as before |
| "read my notes" | Existing skill — should still work |
| "how are you" | Existing skill — `general_conversation`, still works |

## Optional — use your own Amazon Associates tag

The cart-add URL requires `AssociateTag=...` to render the confirmation page. Without one, Amazon silently drops the request. The branch defaults to `sebilee2026-20` (Sebastian's tag) — works for everyone out of the box, no setup needed.

If you want your own tag (10 min):

1. Go to https://affiliate-program.amazon.com → Sign Up
2. Use your existing Amazon login
3. List a website (your GitHub profile works)
4. Pick a Store ID (must end in `-20`, e.g. `yourname-20`)
5. Phone-verify, accept agreement
6. Skip the tax/payment info — only needed if you actually want to RECEIVE commissions
7. Tag is live immediately. Add to your `.env`:
   ```
   AMAZON_ASSOCIATES_TAG=yourname-20
   ```

Note: the tag works in URLs from day 1 of your trial. The 180-day "make 3 sales" requirement is only for keeping the account; it doesn't affect URL functionality during the trial window.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `NoBrowserCookiesError: No browser cookie database found` | Not logged into amazon.com in any browser | Open Firefox/Chrome, log in, retry |
| `NoBrowserCookiesError: No 'amazon.com' cookies found` | Cookies exist but no amazon.com domain | Log in to amazon.com specifically |
| Skill returns `serpapi_auth_401` | Missing or invalid `SERPAPI_KEY` in `.env` | Get a key from https://serpapi.com, add to `.env` |
| Skill returns `serpapi_rate_429` | Hit SerpAPI's rate limit | Wait a minute, or upgrade the SerpAPI plan |
| Browser opens but cart stays empty | Looking at a stale cart view | Hard-refresh: `Ctrl+Shift+R` on the cart page |
| `playwright` import error | Forgot step 3 | `pip install -r alfred_orchestrator/requirements.txt` |
| Chromium fails to start on Linux | Missing system libs | `playwright install-deps chromium` |
| Chromium starts but the page never loads | Network issue or Amazon detected the session | Check your internet; refresh the cookie session by visiting amazon.com in your normal browser |

## How to verify the install worked (no voice needed)

```bash
cd alfred_orchestrator
python -c "
from app.config import Settings
from app.skills.amazon import AmazonSearchSkill, AmazonAddToCartSkill
from app.skills._browser_cookies import read_amazon_cookies

settings = Settings.load()
print(f'SERPAPI_KEY set: {bool(settings.serpapi_key)}')
print(f'Associates tag: {settings.amazon_associates_tag}')

cookies = read_amazon_cookies()
print(f'Amazon cookies found: {len(cookies)}')

# Live SerpAPI search (uses 1 of your monthly quota)
r = AmazonSearchSkill(settings).run(query='AAA batteries')
print(f'Search status: {r.status}')
if r.status == 'success':
    print(f'  Top result: {r.output[\"title_short\"]} ({r.output[\"price_text\"]})')
"
```

Expected output:
```
SERPAPI_KEY set: True
Associates tag: sebilee2026-20
Amazon cookies found: 15+
Search status: success
  Top result: <some Duracell or similar batteries> ($5-20)
```

If all four lines show non-empty values and "success", the install is good.

## Where the new code lives

| File | Purpose |
|---|---|
| `alfred_orchestrator/app/skills/amazon.py` | Both skill classes + helpers (truncate_title, speak_price, build_feature_summary) |
| `alfred_orchestrator/app/skills/_browser_cookies.py` | Reads amazon.com cookies from your default browser |
| `alfred_orchestrator/app/config.py` | Added `serpapi_key` and `amazon_associates_tag` to `Settings` |
| `alfred_orchestrator/app/pipeline.py` | Two new `router.register()` lines in `_build_router` |
| `alfred_orchestrator/configs/skills.yaml` | Two new skill entries with `examples:` |
| `alfred_orchestrator/configs/prompts.yaml` | Anti-chain constraint + few-shot in `skill_planner` system prompt |
| `alfred_orchestrator/tests/test_amazon_skill.py` | 29 unit tests, all mocked, offline, <0.5s total |

## Running the tests

```bash
cd alfred_orchestrator
python -m pytest tests/test_amazon_skill.py -v
```

Expected: 29 passed.

If you hit `ModuleNotFoundError: No module named 'lark'` (ROS leaking `launch_testing` into pytest), prefix with `env -u PYTHONPATH`:

```bash
env -u PYTHONPATH python -m pytest tests/test_amazon_skill.py -v
```
