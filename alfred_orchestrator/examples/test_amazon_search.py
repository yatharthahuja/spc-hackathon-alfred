#!/usr/bin/env python3
"""Quick live Amazon PA-API search test. Run from alfred_orchestrator/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Settings
from app.integrations.amazon.factory import build_amazon_client


def main() -> int:
    parser = argparse.ArgumentParser(description="Test Amazon product search (PA-API or mock).")
    parser.add_argument("query", nargs="?", default="organic apples", help="Search keywords")
    args = parser.parse_args()

    settings = Settings.load()
    print(f"AMAZON_USE_MOCK={settings.amazon_use_mock}")
    print(f"PA-API ready={settings.amazon_paapi_ready}")
    print(f"Host={settings.amazon_host}  Region={settings.amazon_region}")
    print(f"SearchIndex={settings.amazon_search_index}")
    print(f"Query: {args.query}\n")

    client = build_amazon_client(settings)
    product = client.search_first(args.query)

    print("Top result:")
    print(f"  Title: {product.title}")
    print(f"  ASIN:  {product.asin}")
    if product.price_display:
        print(f"  Price: {product.price_display}")
    print(f"  URL:   {product.detail_url}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
