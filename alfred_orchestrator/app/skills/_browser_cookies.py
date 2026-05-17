"""Read Amazon cookies from the user's default browser for Playwright reuse."""

from __future__ import annotations

from typing import List, Dict, Any

import browser_cookie3


class NoBrowserCookiesError(Exception):
    """Raised when no browser cookie DB is found or no matching cookies exist."""


_MAX_SAFE_EXPIRES = 2_147_483_647  # 2^31-1, year 2038 — well within Playwright's int range


def _safe_expires(value: Any) -> int:
    """Coerce a cookie expiry to Playwright's accepted range.

    Playwright only accepts -1 (session cookie) or a positive unix timestamp.
    Firefox occasionally stores 0, negative values, or absurd far-future
    floats that round-trip through int() into a number JS can't represent.
    Anything not in (0, 2^31-1] → -1.
    """
    try:
        v = int(value) if value else -1
    except (TypeError, ValueError):
        return -1
    if v <= 0:
        return -1
    if v > _MAX_SAFE_EXPIRES:
        return _MAX_SAFE_EXPIRES
    return v


def _cookie_to_playwright(cookie: Any) -> Dict[str, Any]:
    """Convert an http.cookiejar.Cookie into a Playwright-shaped dict."""
    return {
        "name": cookie.name,
        "value": cookie.value,
        "domain": cookie.domain,
        "path": cookie.path or "/",
        "expires": _safe_expires(cookie.expires),
        "httpOnly": bool(cookie.has_nonstandard_attr("HttpOnly")),
        "secure": bool(cookie.secure),
        "sameSite": "Lax",
    }


def read_amazon_cookies(domain: str = "amazon.com") -> List[Dict[str, Any]]:
    """Read cookies for `domain` from the user's default browser.

    Uses browser_cookie3 to auto-discover Firefox/Chrome/Edge/Safari profiles
    across Linux/macOS/Windows. Returns Playwright-shaped cookie dicts:
        {name, value, domain, path, expires, httpOnly, secure, sameSite}

    Raises:
      NoBrowserCookiesError: no browser cookie DB found OR no amazon cookies present.
    """
    jar = None
    last_error: Exception | None = None

    for loader in (
        lambda: browser_cookie3.firefox(domain_name=domain),
        lambda: browser_cookie3.chrome(domain_name=domain),
        lambda: browser_cookie3.load(domain_name=domain),
    ):
        try:
            jar = loader()
            if jar is not None and len(list(jar)) > 0:
                break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            jar = None
            continue

    if jar is None:
        raise NoBrowserCookiesError(
            f"No browser cookie database found for {domain!r}: {last_error}"
        )

    suffix = domain.lstrip(".")
    cookies: List[Dict[str, Any]] = []
    for c in jar:
        c_domain = (c.domain or "").lstrip(".")
        if not (c_domain == suffix or c_domain.endswith("." + suffix)):
            continue
        cookies.append(_cookie_to_playwright(c))

    if not cookies:
        raise NoBrowserCookiesError(
            f"No {domain!r} cookies found in any browser profile."
        )

    return cookies
