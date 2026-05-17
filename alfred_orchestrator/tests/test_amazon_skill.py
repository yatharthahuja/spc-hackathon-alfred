from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from app.config import Settings
from app.memory.session_memory import SessionMemory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_settings() -> Settings:
    """Settings with a populated SerpAPI key."""
    return replace(Settings.load(), serpapi_key="test-key")


@pytest.fixture
def mock_settings_no_key() -> Settings:
    """Settings with an empty SerpAPI key."""
    return replace(Settings.load(), serpapi_key="")


@pytest.fixture
def mock_settings_with_tag() -> Settings:
    """Settings with a populated SerpAPI key and the default associates tag."""
    return replace(
        Settings.load(),
        serpapi_key="test-key",
        amazon_associates_tag="sebilee2026-20",
    )


@pytest.fixture
def sample_serpapi_response() -> Dict[str, Any]:
    return {
        "organic_results": [
            {
                "position": 1,
                "asin": "B0XXSPONSORED",
                "sponsored": True,
                "title": "[SPONSORED] Some Sponsored Apple Snack Pack",
                "price": "$9.99",
                "extracted_price": 9.99,
                "rating": 4.0,
                "reviews": 100,
                "prime": True,
                "link": "https://www.amazon.com/sponsored",
                "link_clean": "https://www.amazon.com/dp/B0XXSPONSORED/",
                "thumbnail": "https://example.com/sponsored.jpg",
                "badges": [],
            },
            {
                "position": 2,
                "asin": "B07PXGQC1Q",
                "sponsored": False,
                "title": "Honeycrisp Apples, Organic, 3 lb Bag (Pack of 1)",
                "price": "$4.99",
                "extracted_price": 4.99,
                "rating": 4.4,
                "reviews": 24024,
                "prime": True,
                "link": "https://www.amazon.com/dp/B07PXGQC1Q?th=1",
                "link_clean": "https://www.amazon.com/dp/B07PXGQC1Q/",
                "thumbnail": "https://m.media-amazon.com/images/I/71-qa6qkBuL._AC_UL320_.jpg",
                "badges": ["Amazon's Choice"],
                "bought_last_month": "20K+ bought in past month",
            },
            {
                "position": 3,
                "asin": "B0EXPENSIVE",
                "sponsored": False,
                "title": "Premium Heirloom Apples Gift Box",
                "price": "$49.99",
                "extracted_price": 49.99,
                "rating": 4.7,
                "reviews": 421,
                "prime": False,
                "link": "https://www.amazon.com/dp/B0EXPENSIVE",
                "link_clean": "https://www.amazon.com/dp/B0EXPENSIVE/",
                "thumbnail": "https://example.com/premium.jpg",
                "badges": ["Best Seller"],
            },
        ],
        "search_information": {"organic_results_state": "Results for exact spelling"},
    }


def _ok_response(payload: Dict[str, Any]) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


def _err_response(status_code: int) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = {}
    # Make raise_for_status raise an HTTPError so production code that calls
    # it (if any) still goes through the status_code branch.
    import requests

    err = requests.HTTPError(f"{status_code} error")
    err.response = resp
    resp.raise_for_status.side_effect = err
    return resp


@pytest.fixture
def fresh_history(monkeypatch) -> SessionMemory:
    """Provide a fresh SessionMemory for the test.

    Some skills capture the module-level TASK_HISTORY as a default argument at
    class-definition time, so monkeypatching the module attribute is not enough.
    Tests that construct skills should also pass ``task_history=fresh_history``
    to be safe — the production code accepts that kwarg.
    """
    memory = SessionMemory()
    # Patch the module reference as well, in case any code path resolves it
    # dynamically rather than via the captured default.
    monkeypatch.setattr("app.skills.amazon.TASK_HISTORY", memory, raising=False)
    return memory


# ---------------------------------------------------------------------------
# AmazonSearchSkill tests
# ---------------------------------------------------------------------------


class TestAmazonSearchSkill:
    def test_returns_top_organic_skipping_sponsored(
        self, mock_settings, sample_serpapi_response, fresh_history
    ):
        from app.skills.amazon import AmazonSearchSkill

        skill = AmazonSearchSkill(mock_settings, task_history=fresh_history)
        with patch(
            "app.skills.amazon.requests.get",
            return_value=_ok_response(sample_serpapi_response),
        ):
            result = skill.run(query="organic apples")

        assert result.status == "success"
        assert result.output["asin"] == "B07PXGQC1Q"
        assert result.output["title"].startswith("Honeycrisp Apples")

    def test_respects_max_price(
        self, mock_settings, sample_serpapi_response, fresh_history
    ):
        from app.skills.amazon import AmazonSearchSkill

        skill = AmazonSearchSkill(mock_settings, task_history=fresh_history)

        # max_price 10.0 — should pick B07PXGQC1Q ($4.99) and skip the $49.99 item.
        with patch(
            "app.skills.amazon.requests.get",
            return_value=_ok_response(sample_serpapi_response),
        ):
            result = skill.run(query="apples", max_price=10.0)
        assert result.status == "success"
        assert result.output["asin"] == "B07PXGQC1Q"

        # max_price 3.0 — nothing matches: failure with no_results phrasing.
        with patch(
            "app.skills.amazon.requests.get",
            return_value=_ok_response(sample_serpapi_response),
        ):
            result = skill.run(query="apples", max_price=3.0)
        assert result.status == "error"
        answer = result.output.get("answer_text", "").lower()
        assert (
            "couldn't find" in answer
            or "could not find" in answer
            or "no results" in answer
            or "didn't find" in answer
            or "did not find" in answer
        )

    def test_no_organic_results_returns_no_results_failure(
        self, mock_settings, fresh_history
    ):
        from app.skills.amazon import AmazonSearchSkill

        skill = AmazonSearchSkill(mock_settings, task_history=fresh_history)
        with patch(
            "app.skills.amazon.requests.get",
            return_value=_ok_response({"organic_results": []}),
        ):
            result = skill.run(query="zxcvbnm nonsense")

        assert result.status == "error"
        answer = result.output.get("answer_text", "").lower()
        assert (
            "couldn't find" in answer
            or "could not find" in answer
            or "no results" in answer
            or "didn't find" in answer
            or "did not find" in answer
        )

    def test_serpapi_401_returns_auth_failure(self, mock_settings, fresh_history):
        from app.skills.amazon import AmazonSearchSkill

        skill = AmazonSearchSkill(mock_settings, task_history=fresh_history)
        with patch(
            "app.skills.amazon.requests.get",
            return_value=_err_response(401),
        ):
            result = skill.run(query="apples")

        assert result.status == "error"
        answer = result.output.get("answer_text", "").lower()
        # Should be about authentication/key configuration, not generic.
        assert (
            "key" in answer
            or "auth" in answer
            or "credential" in answer
            or "not configured" in answer
            or "configure" in answer
        )

    def test_serpapi_429_returns_rate_limit_failure(self, mock_settings, fresh_history):
        from app.skills.amazon import AmazonSearchSkill

        skill = AmazonSearchSkill(mock_settings, task_history=fresh_history)
        with patch(
            "app.skills.amazon.requests.get",
            return_value=_err_response(429),
        ):
            result = skill.run(query="apples")

        assert result.status == "error"
        answer = result.output.get("answer_text", "").lower()
        assert (
            "rate" in answer
            or "limit" in answer
            or "too many" in answer
            or "throttl" in answer
            or "try again" in answer
        )

    def test_missing_api_key_returns_failure_without_http_call(
        self, mock_settings_no_key, fresh_history
    ):
        from app.skills.amazon import AmazonSearchSkill

        skill = AmazonSearchSkill(mock_settings_no_key, task_history=fresh_history)
        with patch("app.skills.amazon.requests.get") as mock_get:
            result = skill.run(query="apples")

        assert result.status == "error"
        assert mock_get.called is False
        answer = result.output.get("answer_text", "").lower()
        assert (
            "key" in answer
            or "auth" in answer
            or "credential" in answer
            or "not configured" in answer
            or "configure" in answer
        )

    def test_writes_to_task_history_on_success(
        self, mock_settings, sample_serpapi_response, fresh_history
    ):
        from app.skills.amazon import AmazonSearchSkill

        skill = AmazonSearchSkill(mock_settings, task_history=fresh_history)
        with patch(
            "app.skills.amazon.requests.get",
            return_value=_ok_response(sample_serpapi_response),
        ):
            result = skill.run(query="organic apples")

        assert result.status == "success"
        entries = fresh_history.all()
        amazon_entries = [e for e in entries if e.get("kind") == "amazon_search_result"]
        assert len(amazon_entries) == 1
        entry = amazon_entries[0]
        assert entry["asin"] == "B07PXGQC1Q"
        assert "title" in entry
        assert "price_text" in entry
        assert "url" in entry
        assert "timestamp" in entry

    def test_does_not_write_task_history_on_failure(
        self, mock_settings, fresh_history
    ):
        from app.skills.amazon import AmazonSearchSkill

        skill = AmazonSearchSkill(mock_settings, task_history=fresh_history)
        with patch(
            "app.skills.amazon.requests.get",
            return_value=_err_response(401),
        ):
            result = skill.run(query="apples")

        assert result.status == "error"
        amazon_entries = [
            e for e in fresh_history.all() if e.get("kind") == "amazon_search_result"
        ]
        assert amazon_entries == []

    def test_title_truncation(self, mock_settings, fresh_history):
        from app.skills.amazon import AmazonSearchSkill

        long_title = (
            "Amazon Basics Super Long Title (Pack of 6) [Black, 2-Pack], "
            "Premium Edition for Home Use"
        )
        payload = {
            "organic_results": [
                {
                    "position": 1,
                    "asin": "B0LONGTITL",
                    "sponsored": False,
                    "title": long_title,
                    "price": "$19.99",
                    "extracted_price": 19.99,
                    "rating": 4.5,
                    "reviews": 1234,
                    "prime": True,
                    "link": "https://www.amazon.com/dp/B0LONGTITL?th=1",
                    "link_clean": "https://www.amazon.com/dp/B0LONGTITL/",
                    "thumbnail": "https://example.com/long.jpg",
                    "badges": [],
                }
            ],
            "search_information": {"organic_results_state": "Results for exact spelling"},
        }

        skill = AmazonSearchSkill(mock_settings, task_history=fresh_history)
        with patch(
            "app.skills.amazon.requests.get",
            return_value=_ok_response(payload),
        ):
            result = skill.run(query="amazon basics")

        assert result.status == "success"
        title_short = result.output["title_short"]
        assert len(title_short) <= 40
        assert "(Pack of 6)" not in title_short
        assert not title_short.endswith("...")

    def test_feature_summary_amazons_choice(
        self, mock_settings, sample_serpapi_response, fresh_history
    ):
        from app.skills.amazon import AmazonSearchSkill

        skill = AmazonSearchSkill(mock_settings, task_history=fresh_history)
        with patch(
            "app.skills.amazon.requests.get",
            return_value=_ok_response(sample_serpapi_response),
        ):
            result = skill.run(query="apples")

        assert result.status == "success"
        assert result.output["feature_summary"] == "It's an Amazon's Choice pick."

    def test_feature_summary_bought_last_month(self, mock_settings, fresh_history):
        from app.skills.amazon import AmazonSearchSkill

        payload = {
            "organic_results": [
                {
                    "position": 1,
                    "asin": "B0BOUGHT01",
                    "sponsored": False,
                    "title": "Plain Item",
                    "price": "$12.00",
                    "extracted_price": 12.00,
                    "rating": 4.2,
                    "reviews": 800,
                    "prime": False,
                    "link": "https://www.amazon.com/dp/B0BOUGHT01",
                    "link_clean": "https://www.amazon.com/dp/B0BOUGHT01/",
                    "thumbnail": "https://example.com/x.jpg",
                    "badges": [],
                    "bought_last_month": "5K+ bought in past month",
                }
            ],
            "search_information": {"organic_results_state": "Results for exact spelling"},
        }

        skill = AmazonSearchSkill(mock_settings, task_history=fresh_history)
        with patch(
            "app.skills.amazon.requests.get",
            return_value=_ok_response(payload),
        ):
            result = skill.run(query="plain item")

        assert result.status == "success"
        assert result.output["feature_summary"] == "Over 5K bought this past month."

    def test_feature_summary_empty_when_no_signals(
        self, mock_settings, fresh_history
    ):
        from app.skills.amazon import AmazonSearchSkill

        payload = {
            "organic_results": [
                {
                    "position": 1,
                    "asin": "B0NOSIGNAL",
                    "sponsored": False,
                    "title": "Mundane Thing",
                    "price": "$8.00",
                    "extracted_price": 8.00,
                    "rating": 4.0,
                    "reviews": 10,
                    "prime": False,
                    "link": "https://www.amazon.com/dp/B0NOSIGNAL",
                    "link_clean": "https://www.amazon.com/dp/B0NOSIGNAL/",
                    "thumbnail": "https://example.com/m.jpg",
                    "badges": [],
                }
            ],
            "search_information": {"organic_results_state": "Results for exact spelling"},
        }

        skill = AmazonSearchSkill(mock_settings, task_history=fresh_history)
        with patch(
            "app.skills.amazon.requests.get",
            return_value=_ok_response(payload),
        ):
            result = skill.run(query="mundane thing")

        assert result.status == "success"
        assert result.output["feature_summary"] == ""

    def test_price_spoken_format(self, mock_settings, sample_serpapi_response, fresh_history):
        from app.skills.amazon import AmazonSearchSkill

        skill = AmazonSearchSkill(mock_settings, task_history=fresh_history)

        # $4.99 → "4 99"
        with patch(
            "app.skills.amazon.requests.get",
            return_value=_ok_response(sample_serpapi_response),
        ):
            result = skill.run(query="apples")
        assert result.status == "success"
        assert result.output["price_spoken"] == "4 99"

        # $19.00 → "19 dollars"
        whole_dollar = {
            "organic_results": [
                {
                    "position": 1,
                    "asin": "B019DOLLAR",
                    "sponsored": False,
                    "title": "Whole Dollar Item",
                    "price": "$19.00",
                    "extracted_price": 19.00,
                    "rating": 4.3,
                    "reviews": 50,
                    "prime": True,
                    "link": "https://www.amazon.com/dp/B019DOLLAR",
                    "link_clean": "https://www.amazon.com/dp/B019DOLLAR/",
                    "thumbnail": "https://example.com/w.jpg",
                    "badges": [],
                }
            ],
            "search_information": {"organic_results_state": "Results for exact spelling"},
        }
        with patch(
            "app.skills.amazon.requests.get",
            return_value=_ok_response(whole_dollar),
        ):
            result = skill.run(query="whole dollar")
        assert result.status == "success"
        assert result.output["price_spoken"] == "19 dollars"

    def test_answer_text_contains_title_and_price_and_question(
        self, mock_settings, sample_serpapi_response, fresh_history
    ):
        from app.skills.amazon import AmazonSearchSkill

        skill = AmazonSearchSkill(mock_settings, task_history=fresh_history)
        with patch(
            "app.skills.amazon.requests.get",
            return_value=_ok_response(sample_serpapi_response),
        ):
            result = skill.run(query="apples")

        assert result.status == "success"
        answer = result.output["answer_text"]
        title_short = result.output["title_short"]
        price_spoken = result.output["price_spoken"]
        # The truncated title should appear (or its prefix) in the spoken answer.
        # We check the first three significant characters of the short title to keep this
        # robust to small punctuation differences.
        assert title_short.split(",")[0].split("(")[0].strip()[:8].lower() in answer.lower()
        assert price_spoken in answer
        assert answer.rstrip().endswith("?")


# ---------------------------------------------------------------------------
# AmazonAddToCartSkill tests
# ---------------------------------------------------------------------------


def _seed_amazon_search_result(
    memory: SessionMemory,
    *,
    asin: str = "B07PXGQC1Q",
    title: str = "Honeycrisp Apples, Organic",
    price_text: str = "$4.99",
    url: str = "https://www.amazon.com/dp/B07PXGQC1Q/",
    timestamp: str = "2026-05-17T00:00:00Z",
) -> None:
    memory.add(
        {
            "kind": "amazon_search_result",
            "asin": asin,
            "title": title,
            "price_text": price_text,
            "url": url,
            "timestamp": timestamp,
        }
    )


class TestAmazonAddToCartSkill:
    def test_pulls_asin_from_task_history(self, mock_settings_with_tag, fresh_history):
        from app.skills.amazon import AmazonAddToCartSkill

        _seed_amazon_search_result(fresh_history)

        skill = AmazonAddToCartSkill(mock_settings_with_tag, task_history=fresh_history)
        with patch("app.skills.amazon.threading.Thread") as mock_thread:
            mock_thread.return_value = MagicMock()
            result = skill.run(quantity=1)

        assert result.status == "success"
        assert result.output["asin"] == "B07PXGQC1Q"
        assert result.output["cart_url"] == (
            "https://www.amazon.com/gp/aws/cart/add.html?"
            "ASIN.1=B07PXGQC1Q&Quantity.1=1&AssociateTag=sebilee2026-20"
        )

    def test_no_recent_search_returns_no_session_context_failure(
        self, mock_settings_with_tag, fresh_history
    ):
        from app.skills.amazon import AmazonAddToCartSkill

        skill = AmazonAddToCartSkill(mock_settings_with_tag, task_history=fresh_history)
        with patch("app.skills.amazon.threading.Thread") as mock_thread:
            result = skill.run(quantity=1)

        assert result.status == "error"
        assert mock_thread.called is False
        answer = result.output.get("answer_text", "").lower()
        # no_session_context — should mention nothing recent / no search yet.
        assert (
            "search" in answer
            or "haven't" in answer
            or "have not" in answer
            or "don't have" in answer
            or "do not have" in answer
            or "no recent" in answer
            or "nothing" in answer
            or "first" in answer
        )

    def test_finds_amazon_search_result_amid_other_history_entries(
        self, mock_settings_with_tag, fresh_history
    ):
        from app.skills.amazon import AmazonAddToCartSkill

        # 4 non-amazon entries, then amazon_search_result, then 3 more non-amazon
        # (8 total; amazon entry is at position 5, well within the last-10 window).
        fresh_history.add({"kind": "go_home", "answer_text": "home"})
        fresh_history.add({"kind": "describe_desk", "answer_text": "I see a marker"})
        fresh_history.add({"kind": "speak", "answer_text": "hi"})
        fresh_history.add({"kind": "listen", "answer_text": "ok"})
        _seed_amazon_search_result(fresh_history)
        fresh_history.add({"kind": "speak", "answer_text": "got it"})
        fresh_history.add({"kind": "listen", "answer_text": "yes"})
        fresh_history.add({"kind": "go_home", "answer_text": "home again"})

        skill = AmazonAddToCartSkill(mock_settings_with_tag, task_history=fresh_history)
        with patch("app.skills.amazon.threading.Thread") as mock_thread:
            mock_thread.return_value = MagicMock()
            result = skill.run(quantity=1)

        assert result.status == "success"
        assert result.output["asin"] == "B07PXGQC1Q"

    def test_walks_only_last_10_entries(self, mock_settings_with_tag, fresh_history):
        from app.skills.amazon import AmazonAddToCartSkill

        # Position 1 (oldest): amazon_search_result, then 12 noise entries after
        # so the amazon entry sits well outside the last-10 window.
        _seed_amazon_search_result(fresh_history)
        for i in range(12):
            fresh_history.add({"kind": "noise", "answer_text": f"noise {i}"})

        skill = AmazonAddToCartSkill(mock_settings_with_tag, task_history=fresh_history)
        with patch("app.skills.amazon.threading.Thread") as mock_thread:
            result = skill.run(quantity=1)

        assert result.status == "error"
        assert mock_thread.called is False

    def test_invalid_asin_in_memory_returns_failure(
        self, mock_settings_with_tag, fresh_history
    ):
        from app.skills.amazon import AmazonAddToCartSkill

        _seed_amazon_search_result(fresh_history, asin="not-an-asin")

        skill = AmazonAddToCartSkill(mock_settings_with_tag, task_history=fresh_history)
        with patch("app.skills.amazon.threading.Thread") as mock_thread:
            result = skill.run(quantity=1)

        assert result.status == "error"
        assert mock_thread.called is False
        answer = result.output.get("answer_text", "").lower()
        # invalid_asin fallback should mention the product / id problem,
        # or invite a re-search.
        assert (
            "asin" in answer
            or "invalid" in answer
            or "product id" in answer
            or "off with" in answer
            or "search again" in answer
            or "couldn't" in answer
            or "could not" in answer
        )

    def test_quantity_validation(self, mock_settings_with_tag, fresh_history):
        from app.skills.amazon import AmazonAddToCartSkill

        _seed_amazon_search_result(fresh_history)

        skill = AmazonAddToCartSkill(mock_settings_with_tag, task_history=fresh_history)

        # Invalid quantities should fail and NOT spawn a Thread.
        for bad_qty in (0, -1, 100):
            with patch("app.skills.amazon.threading.Thread") as mock_thread:
                result = skill.run(quantity=bad_qty)
            assert result.status == "error", f"quantity={bad_qty} should be invalid"
            assert mock_thread.called is False, (
                f"quantity={bad_qty} should not spawn a Thread"
            )

        # quantity=2 should succeed and produce a URL with Quantity.1=2.
        with patch("app.skills.amazon.threading.Thread") as mock_thread:
            mock_thread.return_value = MagicMock()
            result = skill.run(quantity=2)
        assert result.status == "success"
        assert "Quantity.1=2" in result.output["cart_url"]

    def test_url_format_exact(self, mock_settings_with_tag, fresh_history):
        from app.skills.amazon import AmazonAddToCartSkill

        _seed_amazon_search_result(fresh_history)

        skill = AmazonAddToCartSkill(mock_settings_with_tag, task_history=fresh_history)
        with patch("app.skills.amazon.threading.Thread") as mock_thread:
            mock_thread.return_value = MagicMock()
            result = skill.run(quantity=3)

        assert result.status == "success"
        assert result.output["cart_url"] == (
            "https://www.amazon.com/gp/aws/cart/add.html?"
            "ASIN.1=B07PXGQC1Q&Quantity.1=3&AssociateTag=sebilee2026-20"
        )

    def test_url_uses_custom_associates_tag(self, fresh_history):
        from app.skills.amazon import AmazonAddToCartSkill

        custom_settings = replace(
            Settings.load(),
            serpapi_key="test-key",
            amazon_associates_tag="myown-20",
        )
        _seed_amazon_search_result(fresh_history)

        skill = AmazonAddToCartSkill(custom_settings, task_history=fresh_history)
        with patch("app.skills.amazon.threading.Thread") as mock_thread:
            mock_thread.return_value = MagicMock()
            result = skill.run(quantity=1)

        assert result.status == "success"
        assert "AssociateTag=myown-20" in result.output["cart_url"]

    def test_url_falls_back_to_default_tag_when_empty(self, fresh_history):
        from app.skills.amazon import AmazonAddToCartSkill

        empty_tag_settings = replace(
            Settings.load(),
            serpapi_key="test-key",
            amazon_associates_tag="",
        )
        _seed_amazon_search_result(fresh_history)

        skill = AmazonAddToCartSkill(empty_tag_settings, task_history=fresh_history)
        with patch("app.skills.amazon.threading.Thread") as mock_thread:
            mock_thread.return_value = MagicMock()
            result = skill.run(quantity=1)

        assert result.status == "success"
        cart_url = result.output["cart_url"]
        # Spec: fall back to "wired-20" when tag is empty.
        assert "AssociateTag=" in cart_url
        # Not the empty string — must be a real non-empty tag value.
        assert "AssociateTag=&" not in cart_url
        assert not cart_url.endswith("AssociateTag=")
        # Per the contract, the empty-tag fallback is "wired-20".
        assert "AssociateTag=wired-20" in cart_url

    def test_thread_spawned_with_daemon_true(
        self, mock_settings_with_tag, fresh_history
    ):
        from app.skills.amazon import AmazonAddToCartSkill

        _seed_amazon_search_result(fresh_history)

        skill = AmazonAddToCartSkill(mock_settings_with_tag, task_history=fresh_history)
        with patch("app.skills.amazon.threading.Thread") as mock_thread:
            thread_instance = MagicMock()
            mock_thread.return_value = thread_instance
            result = skill.run(quantity=1)

        assert result.status == "success"
        assert mock_thread.called, "threading.Thread should have been constructed"

        # The Thread should be marked as a daemon — either via daemon=True kwarg
        # at construction time or via thread.daemon = True before start().
        _args, kwargs = mock_thread.call_args
        daemon_kwarg = kwargs.get("daemon", None)
        daemon_attr_assignments = [
            call for call in thread_instance.mock_calls
            # `t.daemon = True` shows up as a setattr; MagicMock doesn't record
            # plain attribute assignments, so we also inspect `.daemon`.
        ]
        daemon_attr = getattr(thread_instance, "daemon", None)
        # Accept either: kwarg daemon=True, or post-construction .daemon=True
        # (MagicMock auto-records attribute writes via __setattr__ on the mock —
        # the assigned value is reflected on the attribute).
        assert daemon_kwarg is True or daemon_attr is True, (
            f"Thread should be a daemon (kwarg={daemon_kwarg!r}, "
            f"attr={daemon_attr!r}, calls={daemon_attr_assignments!r})"
        )

        # .start() was called.
        assert thread_instance.start.called, "Thread.start() should be called"
        # .join() was NOT called — we never block on the worker.
        assert thread_instance.join.called is False, (
            "Thread.join() should NOT be called — the skill must not block"
        )

    def test_answer_text_says_adding_now(
        self, mock_settings_with_tag, fresh_history
    ):
        from app.skills.amazon import AmazonAddToCartSkill

        _seed_amazon_search_result(fresh_history)

        skill = AmazonAddToCartSkill(mock_settings_with_tag, task_history=fresh_history)
        with patch("app.skills.amazon.threading.Thread") as mock_thread:
            mock_thread.return_value = MagicMock()
            result = skill.run(quantity=1)

        assert result.status == "success"
        answer = result.output["answer_text"].lower()
        # New phrasing — should describe an in-progress add to cart.
        assert ("adding" in answer) or ("cart" in answer), (
            f"answer_text should mention adding/cart, got: {answer!r}"
        )
        # Should NOT use the old browser-opening phrasing.
        for stale in ("opening", "browser", "tap confirm", "hit confirm"):
            assert stale not in answer, (
                f"answer_text should not use stale phrasing {stale!r}, got: {answer!r}"
            )

    def test_cart_add_scheduled_flag_present(
        self, mock_settings_with_tag, fresh_history
    ):
        from app.skills.amazon import AmazonAddToCartSkill

        # Success path
        _seed_amazon_search_result(fresh_history)
        skill = AmazonAddToCartSkill(mock_settings_with_tag, task_history=fresh_history)
        with patch("app.skills.amazon.threading.Thread") as mock_thread:
            mock_thread.return_value = MagicMock()
            ok_result = skill.run(quantity=1)
        assert ok_result.status == "success"
        assert ok_result.output["cart_add_scheduled"] is True

        # Failure path: invalid_asin
        bad_history = SessionMemory()
        _seed_amazon_search_result(bad_history, asin="not-an-asin")
        skill_bad = AmazonAddToCartSkill(mock_settings_with_tag, task_history=bad_history)
        with patch("app.skills.amazon.threading.Thread"):
            bad_result = skill_bad.run(quantity=1)
        assert bad_result.status == "error"
        assert bad_result.output.get("cart_add_scheduled", False) is False

        # Failure path: no_session_context
        empty_history = SessionMemory()
        skill_empty = AmazonAddToCartSkill(
            mock_settings_with_tag, task_history=empty_history
        )
        with patch("app.skills.amazon.threading.Thread"):
            empty_result = skill_empty.run(quantity=1)
        assert empty_result.status == "error"
        assert empty_result.output.get("cart_add_scheduled", False) is False


# ---------------------------------------------------------------------------
# Optional: worker-function tests (only run if the Coder exposed an
# extractable worker function — they are skipped automatically otherwise).
# ---------------------------------------------------------------------------


def _resolve_worker_invocation(settings_factory):
    """Find the worker callable on app.skills.amazon, plus the right way to call it.

    Returns a ``call(cart_url)`` thunk that invokes the worker without args
    we don't have, or None if no worker shape we recognize is exported.

    Recognized shapes (in priority order):
      1. Module-level ``_run_playwright_cart_add(cart_url, headless=False)``
      2. Module-level ``_playwright_cart_add_worker(cart_url, headless=False)``
      3. Bound method ``AmazonAddToCartSkill._playwright_worker(self, cart_url)``
         — instantiated with a stub settings and `headless=True` for speed.
    """
    import importlib
    import inspect

    amazon = importlib.import_module("app.skills.amazon")

    # 1 & 2: module-level helpers
    for name in (
        "_run_playwright_cart_add",
        "_playwright_cart_add_worker",
        "_cart_add_worker",
        "run_playwright_cart_add",
    ):
        worker = getattr(amazon, name, None)
        if callable(worker) and not inspect.isclass(worker):
            sig = inspect.signature(worker)
            params = list(sig.parameters.values())
            if any(p.name == "headless" for p in params):
                return lambda cart_url, w=worker: w(cart_url, True)
            return lambda cart_url, w=worker: w(cart_url)

    # 3: bound method on AmazonAddToCartSkill instance
    cls = getattr(amazon, "AmazonAddToCartSkill", None)
    if cls is not None:
        method_name = None
        for name in (
            "_playwright_worker",
            "_run_playwright_cart_add",
            "_cart_add_worker",
        ):
            if hasattr(cls, name):
                method_name = name
                break
        if method_name is not None:
            def call(cart_url):
                skill = cls(settings_factory(), headless=True, task_history=SessionMemory())
                return getattr(skill, method_name)(cart_url)

            return call

    return None


def _make_test_settings() -> Settings:
    return replace(
        Settings.load(),
        serpapi_key="test-key",
        amazon_associates_tag="sebilee2026-20",
    )


def _build_playwright_chain():
    """Construct a MagicMock chain that mimics ``sync_playwright().__enter__()``.

    Returns (pw_cm, page, context, locator) so callers can assert on each part.
    """
    page = MagicMock()
    locator = MagicMock()
    page.locator.return_value = locator
    locator.first = MagicMock()

    context = MagicMock()
    context.new_page.return_value = page

    browser = MagicMock()
    browser.new_context.return_value = context

    chromium = MagicMock()
    chromium.launch.return_value = browser

    pw_obj = MagicMock()
    pw_obj.chromium = chromium

    pw_cm = MagicMock()
    pw_cm.__enter__.return_value = pw_obj
    pw_cm.__exit__.return_value = False
    return pw_cm, page, context, locator


# Where the worker actually resolves sync_playwright. The production code
# does ``from playwright.sync_api import sync_playwright`` *inside* the worker
# (lazy import to avoid paying the cost at module load), so the patch target
# must be the source module, not ``app.skills.amazon``.
SYNC_PLAYWRIGHT_PATCH = "playwright.sync_api.sync_playwright"


class TestAmazonAddToCartWorker:
    def test_worker_injects_cookies_and_clicks_button(self):
        call_worker = _resolve_worker_invocation(_make_test_settings)
        if call_worker is None:
            pytest.skip("No extractable worker callable found on app.skills.amazon")

        cookies = [
            {
                "name": "session-token",
                "value": "x",
                "domain": ".amazon.com",
                "path": "/",
                "expires": -1,
                "httpOnly": True,
                "secure": True,
                "sameSite": "Lax",
            }
        ]

        pw_cm, page, context, locator = _build_playwright_chain()
        cart_url = (
            "https://www.amazon.com/gp/aws/cart/add.html?"
            "ASIN.1=B07PXGQC1Q&Quantity.1=1&AssociateTag=sebilee2026-20"
        )

        with patch(
            "app.skills.amazon.read_amazon_cookies", return_value=cookies
        ) as mock_read, patch(SYNC_PLAYWRIGHT_PATCH, return_value=pw_cm), patch(
            "app.skills.amazon.time.sleep"
        ):
            call_worker(cart_url)

        assert mock_read.called, "worker should read browser cookies"
        context.add_cookies.assert_called_once_with(cookies)

        # page.goto called with the cart URL (positional or kwarg).
        goto_calls = page.goto.call_args_list
        assert goto_calls, "page.goto should be called"
        goto_args, goto_kwargs = goto_calls[0]
        assert cart_url in goto_args or goto_kwargs.get("url") == cart_url

        # Worker must select the cart-add submit input inside the
        # ``activeCartViewForm`` confirmation form (Amazon renamed away from
        # the old ``submit.add-to-cart`` name attribute in 2025; the form id
        # is what's stable).
        locator_call_args = [c.args for c in page.locator.call_args_list]
        flat = [a[0] for a in locator_call_args if a]
        assert any(
            ("activeCartViewForm" in s and "submit" in s) or "submit.add-to-cart" in s
            for s in flat
        ), f"Expected a locator call targeting the cart-add submit input, got: {flat!r}"
        assert locator.first.click.called, "Add-to-cart click should fire"

    def test_worker_swallows_no_browser_cookies_error(self):
        call_worker = _resolve_worker_invocation(_make_test_settings)
        if call_worker is None:
            pytest.skip("No extractable worker callable found on app.skills.amazon")

        from app.skills._browser_cookies import NoBrowserCookiesError

        cart_url = (
            "https://www.amazon.com/gp/aws/cart/add.html?"
            "ASIN.1=B07PXGQC1Q&Quantity.1=1&AssociateTag=sebilee2026-20"
        )

        with patch(
            "app.skills.amazon.read_amazon_cookies",
            side_effect=NoBrowserCookiesError("nope"),
        ), patch(SYNC_PLAYWRIGHT_PATCH) as mock_sp, patch(
            "app.skills.amazon.time.sleep"
        ):
            # Should NOT raise.
            call_worker(cart_url)

        # No Playwright launch should have occurred — the cookie failure
        # short-circuits before browser launch.
        assert mock_sp.called is False, (
            "sync_playwright must not be entered when cookies are missing"
        )

    def test_worker_swallows_playwright_errors(self):
        call_worker = _resolve_worker_invocation(_make_test_settings)
        if call_worker is None:
            pytest.skip("No extractable worker callable found on app.skills.amazon")

        cookies = [
            {
                "name": "session-token",
                "value": "x",
                "domain": ".amazon.com",
                "path": "/",
                "expires": -1,
                "httpOnly": True,
                "secure": True,
                "sameSite": "Lax",
            }
        ]

        pw_cm = MagicMock()
        pw_cm.__enter__.side_effect = RuntimeError("playwright boom")

        cart_url = (
            "https://www.amazon.com/gp/aws/cart/add.html?"
            "ASIN.1=B07PXGQC1Q&Quantity.1=1&AssociateTag=sebilee2026-20"
        )

        with patch(
            "app.skills.amazon.read_amazon_cookies", return_value=cookies
        ), patch(SYNC_PLAYWRIGHT_PATCH, return_value=pw_cm), patch(
            "app.skills.amazon.time.sleep"
        ):
            # Should NOT raise — the worker is expected to log and exit.
            call_worker(cart_url)
