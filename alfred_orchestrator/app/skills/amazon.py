from __future__ import annotations

import re
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

from app.config import Settings
from app.memory.session_memory import TASK_HISTORY, SessionMemory
from app.orchestrator.schemas import SkillResult
from app.skills._browser_cookies import NoBrowserCookiesError, read_amazon_cookies
from app.skills.base import Skill, success


ASIN_REGEX = re.compile(r"^[A-Z0-9]{10}$")


FAILURE_LINES: Dict[str, str] = {
    "no_results": "I couldn't find anything on Amazon for that. Want to try different words?",
    "serpapi_auth_401": "My Amazon search key isn't working. Someone needs to check the SerpAPI credentials.",
    "serpapi_rate_429": "I'm being rate-limited on Amazon search. Give me a minute and try again.",
    "no_session_context": "I don't have a recent search to add. What would you like me to find on Amazon?",
    "invalid_asin": "Something's off with that product ID. Let me search again — what were you looking for?",
    "browser_launch_failed": "I couldn't open the browser. You can paste this link from the screen to add it manually.",
    "no_browser_login": "I can't see an Amazon login in your browser. Sign in to amazon.com and try again.",
}


def speak_price(p: Optional[float]) -> str:
    if p is None:
        return ""
    if p == int(p):
        return f"{int(p)} dollars"
    dollars = int(p)
    cents = round((p - dollars) * 100)
    return f"{dollars} {cents:02d}"


def truncate_title(title: str) -> str:
    if not title:
        return ""
    # Strip parenthesized and bracketed content first.
    cleaned = re.sub(r"\s*\([^)]*\)", "", title)
    cleaned = re.sub(r"\s*\[[^\]]*\]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Drop trailing comma-suffixes like ", Pack of 6", ", Black".
    while "," in cleaned:
        head, _, tail = cleaned.rpartition(",")
        head = head.strip()
        tail_stripped = tail.strip()
        # Only drop a trailing clause when the title is STILL too long AND the
        # tail looks like a size/count/color suffix (≤4 words). Otherwise we'd
        # drop real product description (e.g. "Amazon Grocery, Organic Gala
        # Apples, 2 Lb" would collapse to just "Amazon Grocery").
        if head and len(cleaned) > 40 and len(tail_stripped.split()) <= 4:
            cleaned = head
        else:
            break
    cleaned = cleaned.rstrip(",;: -")

    if len(cleaned) <= 40:
        return cleaned

    # Cut at last word boundary before char 40, no ellipsis.
    cutoff = cleaned[:40]
    last_space = cutoff.rfind(" ")
    if last_space > 0:
        return cutoff[:last_space].rstrip()
    return cutoff.rstrip()


def _parse_bought_last_month(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip().lower()
    if not text:
        return None
    # Common formats: "20K+ bought in past month", "1,000+ bought in past month"
    match = re.search(r"([\d,.]+)\s*([km]?)", text)
    if not match:
        return None
    raw_num, suffix = match.group(1), match.group(2)
    try:
        num = float(raw_num.replace(",", ""))
    except ValueError:
        return None
    if suffix == "k":
        num *= 1_000
    elif suffix == "m":
        num *= 1_000_000
    return int(num)


def _has_badge(badges: Any, target: str) -> bool:
    target_lower = target.lower()
    if isinstance(badges, str):
        return target_lower in badges.lower()
    if isinstance(badges, list):
        for b in badges:
            if isinstance(b, str) and target_lower in b.lower():
                return True
            if isinstance(b, dict):
                for value in b.values():
                    if isinstance(value, str) and target_lower in value.lower():
                        return True
    if isinstance(badges, dict):
        for value in badges.values():
            if isinstance(value, str) and target_lower in value.lower():
                return True
            if isinstance(value, list) and _has_badge(value, target):
                return True
    return False


def build_feature_summary(result: Dict[str, Any]) -> str:
    badges = result.get("badges") or result.get("badge") or []
    if _has_badge(badges, "Amazon's Choice") or result.get("amazons_choice"):
        return "It's an Amazon's Choice pick."
    if _has_badge(badges, "Best Seller") or result.get("best_seller"):
        return "It's a best seller."
    if _has_badge(badges, "Limited time deal") or result.get("limited_time_deal"):
        return "Limited-time deal right now."

    bought = _parse_bought_last_month(result.get("bought_last_month"))
    if bought is not None and bought >= 1000:
        k = bought // 1000
        return f"Over {k}K bought this past month."

    prime = bool(result.get("prime"))
    rating = result.get("rating")
    reviews = result.get("reviews")
    try:
        rating_val = float(rating) if rating is not None else None
    except (TypeError, ValueError):
        rating_val = None
    try:
        reviews_val = int(reviews) if reviews is not None else None
    except (TypeError, ValueError):
        reviews_val = None

    if prime and rating_val is not None and reviews_val is not None and rating_val >= 4.5 and reviews_val >= 500:
        return "Prime eligible with strong reviews."
    if prime:
        return "Ships with Prime."
    return ""


def _error_result(skill_name: str, error: str, answer_text: str) -> SkillResult:
    return SkillResult(
        skill_name=skill_name,
        status="error",
        error=error,
        output={"answer_text": answer_text},
    )


class AmazonSearchSkill(Skill):
    name = "amazon_search"

    def __init__(self, settings: Settings, task_history: SessionMemory = TASK_HISTORY):
        self.settings = settings
        self.task_history = task_history

    def run(self, **kwargs: Any) -> SkillResult:
        query = str(kwargs.get("query") or "").strip()
        max_price_raw = kwargs.get("max_price")
        try:
            max_price = float(max_price_raw) if max_price_raw not in (None, "") else None
        except (TypeError, ValueError):
            max_price = None

        if not query:
            return _error_result(
                self.name,
                "Empty query",
                FAILURE_LINES["no_results"],
            )

        if not self.settings.serpapi_key:
            return _error_result(
                self.name,
                "SERPAPI_KEY is not configured",
                FAILURE_LINES["serpapi_auth_401"],
            )

        params = {
            "engine": "amazon",
            "amazon_domain": "amazon.com",
            "k": query,
            "api_key": self.settings.serpapi_key,
        }
        print(f"[amazon_search] Query: {query}  max_price={max_price}")
        try:
            response = requests.get(
                "https://serpapi.com/search.json",
                params=params,
                timeout=15,
            )
        except requests.RequestException as exc:
            return _error_result(self.name, f"SerpAPI request failed: {exc}", FAILURE_LINES["no_results"])

        if response.status_code == 401:
            return _error_result(
                self.name,
                "SerpAPI returned 401",
                FAILURE_LINES["serpapi_auth_401"],
            )
        if response.status_code == 429:
            return _error_result(
                self.name,
                "SerpAPI returned 429",
                FAILURE_LINES["serpapi_rate_429"],
            )
        if response.status_code >= 400:
            return _error_result(
                self.name,
                f"SerpAPI HTTP {response.status_code}: {response.text[:200]}",
                FAILURE_LINES["no_results"],
            )

        try:
            payload = response.json()
        except ValueError as exc:
            return _error_result(self.name, f"SerpAPI returned non-JSON: {exc}", FAILURE_LINES["no_results"])

        organic_results: List[Dict[str, Any]] = payload.get("organic_results") or []
        non_sponsored = [r for r in organic_results if not r.get("sponsored")]
        if not non_sponsored:
            return _error_result(
                self.name,
                "No organic results returned",
                FAILURE_LINES["no_results"],
            )

        chosen: Optional[Dict[str, Any]] = None
        if max_price is not None:
            for r in non_sponsored:
                price = r.get("extracted_price")
                try:
                    price_val = float(price) if price is not None else None
                except (TypeError, ValueError):
                    price_val = None
                if price_val is not None and price_val <= max_price:
                    chosen = r
                    break
        else:
            chosen = non_sponsored[0]

        if chosen is None:
            return _error_result(
                self.name,
                "No organic results matched the max_price filter",
                FAILURE_LINES["no_results"],
            )

        asin = str(chosen.get("asin") or "").strip().upper()
        if not ASIN_REGEX.match(asin):
            return _error_result(
                self.name,
                f"Invalid ASIN returned: {asin!r}",
                FAILURE_LINES["no_results"],
            )

        title = str(chosen.get("title") or "").strip()
        title_short = truncate_title(title)
        price_text = str(chosen.get("price") or "").strip()
        price_extracted = chosen.get("extracted_price")
        try:
            price_val: Optional[float] = float(price_extracted) if price_extracted is not None else None
        except (TypeError, ValueError):
            price_val = None
        price_spoken = speak_price(price_val)

        rating_raw = chosen.get("rating")
        try:
            rating_val: Optional[float] = float(rating_raw) if rating_raw is not None else None
        except (TypeError, ValueError):
            rating_val = None

        reviews_raw = chosen.get("reviews")
        try:
            reviews_val: Optional[int] = int(reviews_raw) if reviews_raw is not None else None
        except (TypeError, ValueError):
            reviews_val = None

        thumbnail_url = chosen.get("thumbnail") or chosen.get("thumbnail_url")
        product_url = chosen.get("link_clean") or chosen.get("link")

        feature_summary = build_feature_summary(chosen)

        # Build answer_text
        rating_clause = f", {rating_val} stars" if rating_val is not None else ""
        feature_clause = f" {feature_summary}" if feature_summary else ""
        answer_text = (
            f"Found {title_short} for {price_spoken} dollars{rating_clause}.{feature_clause} "
            f"Want me to add it?"
        ).replace("  ", " ").strip()

        # Record to TASK_HISTORY so amazon_add_to_cart can find the ASIN later.
        self.task_history.add(
            {
                "kind": "amazon_search_result",
                "asin": asin,
                "title": title,
                "price_text": price_text,
                "url": product_url,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        output = {
            "asin": asin,
            "title": title,
            "title_short": title_short,
            "price_text": price_text,
            "price": price_val,
            "price_spoken": price_spoken,
            "rating": rating_val,
            "reviews": reviews_val,
            "thumbnail_url": thumbnail_url,
            "product_url": product_url,
            "feature_summary": feature_summary,
            "answer_text": answer_text,
        }
        print(f"[amazon_search] Selected ASIN: {asin}  title: {title_short!r}  price: {price_text}")
        return success(self.name, output)


class AmazonAddToCartSkill(Skill):
    name = "amazon_add_to_cart"

    def __init__(
        self,
        settings: Settings,
        open_browser: bool = True,
        task_history: SessionMemory = TASK_HISTORY,
        headless: bool = False,
    ):
        self.settings = settings
        self.open_browser = open_browser
        self.task_history = task_history
        self.headless = headless

    def _playwright_worker(self, cart_url: str) -> None:
        """Background worker: wait, inject cookies, click confirm. Never raises."""
        try:
            time.sleep(2.5)

            try:
                cookies = read_amazon_cookies()
            except NoBrowserCookiesError as exc:
                print(
                    "[amazon_add_to_cart] no Amazon login found in any browser — "
                    f"sign in to amazon.com first ({exc})",
                    file=sys.stderr,
                )
                return

            if not cookies:
                print(
                    "[amazon_add_to_cart] no Amazon login found in any browser — "
                    "sign in to amazon.com first",
                    file=sys.stderr,
                )
                return

            from playwright.sync_api import sync_playwright  # local import: heavy

            with sync_playwright() as pw:
                browser = pw.chromium.launch(headless=self.headless)
                # Firefox UA + locale to match the session cookies' origin browser.
                # Amazon's session-token isn't UA-locked for cart actions but
                # mismatch occasionally triggers a soft "verify" interstitial.
                context = browser.new_context(
                    user_agent=(
                        "Mozilla/5.0 (X11; Linux x86_64; rv:148.0) "
                        "Gecko/20100101 Firefox/148.0"
                    ),
                    locale="en-US",
                )
                context.add_cookies(cookies)
                page = context.new_page()
                page.goto(cart_url, wait_until="domcontentloaded", timeout=15000)
                # Confirmation form's submit input — Amazon renamed away from
                # `name="submit.add-to-cart"` in 2025. The form id is stable.
                page.wait_for_selector(
                    "form#activeCartViewForm", timeout=10000
                )
                try:
                    page.locator(
                        'form#activeCartViewForm input.a-button-input[type="submit"]'
                    ).first.click(timeout=5000)
                except Exception:
                    # Fallback: submit the form directly via JS. Bulletproof
                    # against UI churn — as long as the form id holds.
                    page.evaluate(
                        "document.getElementById('activeCartViewForm').submit()"
                    )
                try:
                    page.wait_for_url("**/cart/**", timeout=10000)
                except Exception:
                    try:
                        page.wait_for_selector("text=/Added to/i", timeout=5000)
                    except Exception:
                        pass
                time.sleep(0.5 if self.headless else 3.0)
                browser.close()
        except Exception as exc:  # noqa: BLE001 - thread must never raise
            print(
                f"[amazon_add_to_cart] Playwright worker error: {exc}",
                file=sys.stderr,
            )

    def run(self, **kwargs: Any) -> SkillResult:
        quantity_raw = kwargs.get("quantity", 1)
        try:
            quantity = int(quantity_raw) if quantity_raw is not None else 1
        except (TypeError, ValueError):
            quantity = 1

        # Find latest amazon_search_result in last 10 entries.
        all_entries = self.task_history.all()
        recent = all_entries[-10:]
        latest: Optional[Dict[str, Any]] = None
        for entry in reversed(recent):
            if isinstance(entry, dict) and entry.get("kind") == "amazon_search_result":
                latest = entry
                break

        if latest is None:
            return _error_result(
                self.name,
                "No recent amazon_search_result in task history",
                FAILURE_LINES["no_session_context"],
            )

        asin = str(latest.get("asin") or "").strip().upper()
        if not ASIN_REGEX.match(asin):
            return _error_result(
                self.name,
                f"Invalid ASIN in task history: {asin!r}",
                FAILURE_LINES["invalid_asin"],
            )

        if quantity < 1 or quantity > 99:
            return _error_result(
                self.name,
                f"Quantity {quantity} is outside [1, 99]",
                FAILURE_LINES["invalid_asin"],
            )

        tag = self.settings.amazon_associates_tag or "wired-20"
        cart_url = (
            f"https://www.amazon.com/gp/aws/cart/add.html?"
            f"ASIN.1={asin}&Quantity.1={quantity}&AssociateTag={tag}"
        )

        cart_add_scheduled = False
        if self.open_browser:
            try:
                worker = threading.Thread(
                    target=self._playwright_worker,
                    args=(cart_url,),
                    daemon=True,
                )
                worker.start()
                cart_add_scheduled = True
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[amazon_add_to_cart] Failed to start worker thread: {exc}")
                return _error_result(
                    self.name,
                    f"Browser launch failed: {exc}",
                    FAILURE_LINES["browser_launch_failed"],
                )

        title = str(latest.get("title") or "")
        answer_text = "Adding to your cart now."

        output = {
            "cart_url": cart_url,
            "asin": asin,
            "quantity": quantity,
            "cart_add_scheduled": cart_add_scheduled,
            "title": title,
            "answer_text": answer_text,
        }
        print(
            f"[amazon_add_to_cart] cart_url: {cart_url}  "
            f"cart_add_scheduled: {cart_add_scheduled}"
        )
        return success(self.name, output)
