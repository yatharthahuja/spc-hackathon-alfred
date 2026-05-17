from __future__ import annotations

from typing import Any, Optional

from app.integrations.amazon.models import AmazonProduct


def parse_first_search_item(payload: dict[str, Any]) -> Optional[AmazonProduct]:
    """Parse top item from PA-API (PascalCase) or Creators API (camelCase) search responses."""
    search_result = (
        payload.get("searchResult")
        or payload.get("SearchResult")
        or {}
    )
    items = search_result.get("items") or search_result.get("Items") or []
    if not items:
        return None

    item = items[0]
    asin = str(item.get("asin") or item.get("ASIN") or "").strip()
    if not asin:
        return None

    title = (
        _nested_text(item, ["itemInfo", "title", "displayValue"])
        or _nested_text(item, ["ItemInfo", "Title", "DisplayValue"])
        or f"Amazon item {asin}"
    )
    detail_url = str(
        item.get("detailPageURL")
        or item.get("DetailPageURL")
        or f"https://www.amazon.com/dp/{asin}"
    )
    price_display = (
        _nested_text(item, ["offersV2", "listings", 0, "price", "displayAmount"])
        or _nested_text(item, ["Offers", "Listings", 0, "Price", "DisplayAmount"])
        or _nested_text(item, ["offers", "listings", 0, "price", "displayAmount"])
    )
    image_url = (
        _nested_text(item, ["images", "primary", "medium", "url"])
        or _nested_text(item, ["Images", "Primary", "Medium", "URL"])
    )

    return AmazonProduct(
        asin=asin,
        title=title,
        detail_url=detail_url,
        price_display=price_display,
        image_url=image_url,
    )


def _nested_text(data: Any, path: list[Any]) -> str:
    current = data
    for key in path:
        if isinstance(key, int):
            if not isinstance(current, list) or len(current) <= key:
                return ""
            current = current[key]
        elif isinstance(current, dict):
            current = current.get(key)
        else:
            return ""
    return str(current or "").strip()
