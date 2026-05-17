from __future__ import annotations

from app.integrations.amazon.base import AmazonSearchClient
from app.integrations.amazon.models import AmazonProduct


class MockAmazonSearchClient(AmazonSearchClient):
    def search_first(self, query: str) -> AmazonProduct:
        label = query.strip() or "item"
        asin = "B0MOCK001"
        return AmazonProduct(
            asin=asin,
            title=f"{label.title()} — Mock Amazon Top Result",
            detail_url=f"https://www.amazon.com/dp/{asin}",
            price_display="$9.99",
        )
