from __future__ import annotations

import hashlib
import hmac
from datetime import datetime, timezone
from typing import Dict
from urllib.parse import urlparse


def sign_paapi_request(
    *,
    method: str,
    url: str,
    payload: str,
    access_key: str,
    secret_key: str,
    region: str,
    service: str = "ProductAdvertisingAPI",
) -> Dict[str, str]:
    parsed = urlparse(url)
    host = parsed.netloc
    amz_date = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    date_stamp = amz_date[:8]
    canonical_uri = parsed.path or "/"
    payload_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    canonical_headers = (
        f"content-encoding:amz-1.0\n"
        f"content-type:application/json; charset=utf-8\n"
        f"host:{host}\n"
        f"x-amz-date:{amz_date}\n"
        f"x-amz-target:com.amazon.paapi5.v1.ProductAdvertisingAPIv1.SearchItems\n"
    )
    signed_headers = "content-encoding;content-type;host;x-amz-date;x-amz-target"
    canonical_request = "\n".join(
        [method, canonical_uri, "", canonical_headers, signed_headers, payload_hash]
    )
    credential_scope = f"{date_stamp}/{region}/{service}/aws4_request"
    string_to_sign = "\n".join(
        [
            "AWS4-HMAC-SHA256",
            amz_date,
            credential_scope,
            hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
        ]
    )
    signing_key = _derive_signing_key(secret_key, date_stamp, region, service)
    signature = hmac.new(signing_key, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
    return {
        "Authorization": (
            "AWS4-HMAC-SHA256 "
            f"Credential={access_key}/{credential_scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"
        ),
        "X-Amz-Date": amz_date,
    }


def _derive_signing_key(secret_key: str, date_stamp: str, region: str, service: str) -> bytes:
    key = ("AWS4" + secret_key).encode("utf-8")
    for message in (date_stamp, region, service, "aws4_request"):
        key = hmac.new(key, message.encode("utf-8"), hashlib.sha256).digest()
    return key
