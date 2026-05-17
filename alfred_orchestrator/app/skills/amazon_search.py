from __future__ import annotations

import re
from typing import Any, Optional

from app.config import Settings
from app.integrations.amazon.base import AmazonSearchClient
from app.integrations.amazon.factory import build_amazon_client
from app.integrations.amazon.mock_client import MockAmazonSearchClient
from app.orchestrator.schemas import SkillResult
from app.skills.base import Skill, failure, success


class AmazonSearchSkill(Skill):
    name = "amazon_search"

    def __init__(
        self,
        settings: Settings,
        client: Optional[AmazonSearchClient] = None,
    ):
        self.settings = settings
        self._client = client

    @property
    def client(self) -> AmazonSearchClient:
        if self._client is None:
            self._client = build_amazon_client(self.settings)
        return self._client

    def run(self, **kwargs: Any) -> SkillResult:
        try:
            user_text = str(kwargs.get("user_text") or "").strip()
            query = str(kwargs.get("query") or "").strip() or extract_product_query(user_text)
            if not query:
                raise ValueError("A product search query is required")

            print(f"[amazon_search] Searching Amazon for: {query}")
            product = self.client.search_first(query)

            price_part = f" — {product.price_display}" if product.price_display else ""
            answer_text = (
                f"The top Amazon result for \"{query}\" is: {product.title}{price_part}. "
                f"Link: {product.detail_url}"
            )
            print("[amazon_search] Top result:")
            print(f"  Title: {product.title}")
            print(f"  ASIN:  {product.asin}")
            if product.price_display:
                print(f"  Price: {product.price_display}")
            print(f"  URL:   {product.detail_url}")

            return success(
                self.name,
                {
                    "query": query,
                    "asin": product.asin,
                    "title": product.title,
                    "product_url": product.detail_url,
                    "price_display": product.price_display,
                    "answer_text": answer_text,
                    "used_mock": isinstance(self.client, MockAmazonSearchClient),
                },
            )
        except Exception as exc:
            return failure(self.name, exc)


def extract_product_query(user_text: str) -> str:
    normalized = " ".join(user_text.lower().split())
    if not normalized:
        return ""

    patterns = (
        r"(?:order|buy|get|find|search(?:\s+for)?|purchase)\s+(?:some\s+)?(.+?)\s+(?:on|from)\s+amazon",
        r"amazon\s+(?:order|search)\s+(?:for\s+)?(.+)",
        r"(?:what(?:'s| is)?|show me)\s+(?:the\s+)?(?:top|best)\s+(.+?)\s+on\s+amazon",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            query = match.group(1).strip(" .,!?:;\"'")
            query = re.sub(r"\b(please|yeah|sure|thanks|can you|could you)\b", "", query).strip()
            if query:
                return query

    if "amazon" in normalized:
        cleaned = re.sub(
            r"\b(can you|could you|please|yeah|sure|on amazon|from amazon|amazon|order|buy|find|search|get)\b",
            "",
            normalized,
        ).strip()
        if cleaned:
            return cleaned
    return ""
