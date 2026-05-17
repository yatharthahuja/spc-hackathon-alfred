from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AmazonProduct:
    asin: str
    title: str
    detail_url: str
    image_url: str = ""
    price_display: str = ""
