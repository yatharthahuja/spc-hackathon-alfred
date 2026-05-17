from __future__ import annotations

from app.config import Settings
from app.integrations.amazon.base import AmazonSearchClient
from app.integrations.amazon.mock_client import MockAmazonSearchClient


def build_amazon_client(settings: Settings) -> AmazonSearchClient:
    if settings.amazon_use_mock:
        print("[amazon] Using mock Amazon search client (AMAZON_USE_MOCK=true)")
        return MockAmazonSearchClient()

    api_mode = settings.amazon_api.lower()
    if api_mode == "paapi":
        if not settings.amazon_paapi_ready:
            raise RuntimeError(
                "AMAZON_API=paapi requires AMAZON_ACCESS_KEY, AMAZON_SECRET_KEY, and "
                "AMAZON_PARTNER_TAG in alfred_orchestrator/.env."
            )
        from app.integrations.amazon.paapi_client import PaapiAmazonSearchClient

        print("[amazon] Using legacy PA-API 5 search client")
        return PaapiAmazonSearchClient(
            access_key=settings.amazon_access_key,
            secret_key=settings.amazon_secret_key,
            partner_tag=settings.amazon_partner_tag,
            host=settings.amazon_host,
            region=settings.amazon_region,
            marketplace=settings.amazon_marketplace,
            search_index=settings.amazon_search_index,
        )

    if not settings.amazon_creators_ready:
        raise RuntimeError(
            "Live Amazon Creators API search requires AMAZON_CREDENTIAL_ID, "
            "AMAZON_CREDENTIAL_SECRET, AMAZON_PARTNER_TAG, and AMAZON_CREDENTIAL_VERSION "
            "(from Associates → Tools → Creators API). "
            "Legacy PA-API keys can be used via AMAZON_API=paapi."
        )

    from app.integrations.amazon.creators_client import CreatorsAmazonSearchClient

    print("[amazon] Using Amazon Creators API search client")
    return CreatorsAmazonSearchClient(
        credential_id=settings.amazon_credential_id,
        credential_secret=settings.amazon_credential_secret,
        credential_version=settings.amazon_credential_version,
        partner_tag=settings.amazon_partner_tag,
        marketplace=settings.amazon_marketplace,
        search_index=settings.amazon_search_index,
        auth_endpoint=settings.amazon_auth_endpoint,
    )
