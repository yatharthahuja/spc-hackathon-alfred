from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.integrations.amazon.base import AmazonSearchClient
from app.integrations.amazon.creators_auth import CreatorsOAuthConfig, CreatorsTokenProvider
from app.integrations.amazon.models import AmazonProduct
from app.integrations.amazon.parse import parse_first_search_item

CREATORS_API_BASE = "https://creatorsapi.amazon"
SEARCH_ITEMS_PATH = "/catalog/v1/searchItems"


class CreatorsAmazonSearchClient(AmazonSearchClient):
    """
    Amazon Creators API SearchItems (replaces legacy PA-API 5).
    Docs: https://affiliate-program.amazon.com/creatorsapi/docs/en-us/api-reference/operations/search-items
    """

    def __init__(
        self,
        *,
        credential_id: str,
        credential_secret: str,
        credential_version: str,
        partner_tag: str,
        marketplace: str = "www.amazon.com",
        search_index: str = "All",
        auth_endpoint: str = "",
        timeout: int = 30,
    ):
        self.partner_tag = partner_tag
        self.marketplace = marketplace
        self.search_index = search_index
        self.timeout = timeout
        self._token_provider = CreatorsTokenProvider(
            CreatorsOAuthConfig(
                credential_id=credential_id,
                credential_secret=credential_secret,
                version=credential_version,
                auth_endpoint=auth_endpoint,
            )
        )

    def search_first(self, query: str) -> AmazonProduct:
        keywords = query.strip()
        if not keywords:
            raise ValueError("Search query is required")

        payload = {
            "partnerTag": self.partner_tag,
            "keywords": keywords,
            "searchIndex": self.search_index,
            "itemCount": 1,
            "resources": [
                "itemInfo.title",
                "offersV2.listings.price",
                "images.primary.medium",
            ],
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": self._token_provider.get_authorization_header(),
            "x-marketplace": self.marketplace,
        }
        request = Request(
            f"{CREATORS_API_BASE}{SEARCH_ITEMS_PATH}",
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Creators API SearchItems HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Creators API SearchItems request failed: {exc}") from exc

        product = parse_first_search_item(result)
        if product is None:
            raise RuntimeError(f"No Amazon products found for: {keywords}")
        return product
