from __future__ import annotations

from app.config import Settings
from app.integrations.amazon.base import AmazonSearchClient
from app.integrations.amazon.mock_client import MockAmazonSearchClient


def build_amazon_client(settings: Settings) -> AmazonSearchClient:
    if settings.amazon_use_mock:
        print("[amazon] Using mock Amazon search client (AMAZON_USE_MOCK=true)")
        return MockAmazonSearchClient()

    if not settings.amazon_paapi_ready:
        raise RuntimeError(
            "Live Amazon search requires AMAZON_ACCESS_KEY, AMAZON_SECRET_KEY, and "
            "AMAZON_PARTNER_TAG in alfred_orchestrator/.env (from Associates PA-API)."
        )

    from app.integrations.amazon.paapi_client import PaapiAmazonSearchClient

    print("[amazon] Using live PA-API Amazon search client")
    return PaapiAmazonSearchClient(
        access_key=settings.amazon_access_key,
        secret_key=settings.amazon_secret_key,
        partner_tag=settings.amazon_partner_tag,
        host=settings.amazon_host,
        region=settings.amazon_region,
        marketplace=settings.amazon_marketplace,
        search_index=settings.amazon_search_index,
    )
