from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.integrations.amazon.base import AmazonSearchClient
from app.integrations.amazon.models import AmazonProduct
from app.integrations.amazon.parse import parse_first_search_item
from app.integrations.amazon.paapi_sign import sign_paapi_request


class PaapiAmazonSearchClient(AmazonSearchClient):
    def __init__(
        self,
        *,
        access_key: str,
        secret_key: str,
        partner_tag: str,
        host: str = "webservices.amazon.com",
        region: str = "us-east-1",
        marketplace: str = "www.amazon.com",
        search_index: str = "All",
        timeout: int = 30,
    ):
        self.access_key = access_key
        self.secret_key = secret_key
        self.partner_tag = partner_tag
        self.marketplace = marketplace
        self.search_index = search_index
        self.region = region
        self.timeout = timeout
        self._search_url = f"https://{host}/paapi5/searchitems"

    def search_first(self, query: str) -> AmazonProduct:
        keywords = query.strip()
        if not keywords:
            raise ValueError("Search query is required")

        payload = json.dumps(
            {
                "Keywords": keywords,
                "PartnerTag": self.partner_tag,
                "PartnerType": "Associates",
                "Marketplace": self.marketplace,
                "SearchIndex": self.search_index,
                "Resources": [
                    "ItemInfo.Title",
                    "Offers.Listings.Price",
                    "DetailPageURL",
                ],
                "ItemCount": 1,
            }
        )
        signed = sign_paapi_request(
            method="POST",
            url=self._search_url,
            payload=payload,
            access_key=self.access_key,
            secret_key=self.secret_key,
            region=self.region,
        )
        request = Request(
            self._search_url,
            data=payload.encode("utf-8"),
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Content-Encoding": "amz-1.0",
                "X-Amz-Target": "com.amazon.paapi5.v1.ProductAdvertisingAPIv1.SearchItems",
                **signed,
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Amazon PA-API HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Amazon PA-API request failed: {exc}") from exc

        product = parse_first_search_item(body)
        if product is None:
            raise RuntimeError(f"No Amazon products found for: {keywords}")
        return product
