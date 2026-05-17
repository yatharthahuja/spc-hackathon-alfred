from __future__ import annotations

from app.integrations.amazon.creators_auth import CreatorsOAuthConfig
from app.integrations.amazon.parse import parse_first_search_item


def test_creators_token_endpoint_for_us():
    config = CreatorsOAuthConfig("id", "secret", "2.1")
    assert "us-east-1" in config.token_endpoint()
    assert config.scope() == "creatorsapi/default"


def test_parse_creators_search_response():
    payload = {
        "searchResult": {
            "items": [
                {
                    "asin": "B012345678",
                    "detailPageURL": "https://www.amazon.com/dp/B012345678",
                    "itemInfo": {"title": {"displayValue": "Organic Apples"}},
                    "offersV2": {
                        "listings": [{"price": {"displayAmount": "$4.99"}}],
                    },
                }
            ]
        }
    }
    product = parse_first_search_item(payload)
    assert product is not None
    assert product.asin == "B012345678"
    assert product.title == "Organic Apples"
    assert product.price_display == "$4.99"
