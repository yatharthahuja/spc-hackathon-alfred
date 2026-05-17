from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Optional


@dataclass
class CreatorsOAuthConfig:
    credential_id: str
    credential_secret: str
    version: str
    auth_endpoint: str = ""

    def is_lwa(self) -> bool:
        return self.version.startswith("3.")

    def token_endpoint(self) -> str:
        if self.auth_endpoint.strip():
            return self.auth_endpoint.strip()
        if self.version == "2.1":
            return "https://creatorsapi.auth.us-east-1.amazoncognito.com/oauth2/token"
        if self.version == "2.2":
            return "https://creatorsapi.auth.eu-south-2.amazoncognito.com/oauth2/token"
        if self.version == "2.3":
            return "https://creatorsapi.auth.us-west-2.amazoncognito.com/oauth2/token"
        if self.version == "3.1":
            return "https://api.amazon.com/auth/o2/token"
        if self.version == "3.2":
            return "https://api.amazon.co.uk/auth/o2/token"
        if self.version == "3.3":
            return "https://api.amazon.co.jp/auth/o2/token"
        raise ValueError(
            f"Unsupported AMAZON_CREDENTIAL_VERSION={self.version!r}. "
            "Use 2.1, 2.2, 2.3, 3.1, 3.2, or 3.3 (see Associates Creators API docs)."
        )

    def scope(self) -> str:
        return "creatorsapi::default" if self.is_lwa() else "creatorsapi/default"


class CreatorsTokenProvider:
    """OAuth2 client-credentials token cache for Amazon Creators API."""

    def __init__(self, config: CreatorsOAuthConfig):
        self.config = config
        self._access_token: Optional[str] = None
        self._expires_at: float = 0.0

    def get_authorization_header(self) -> str:
        token = self._get_access_token()
        if self.config.is_lwa():
            return f"Bearer {token}"
        return f"Bearer {token}, Version {self.config.version}"

    def _get_access_token(self) -> str:
        if self._access_token and time.time() < self._expires_at:
            return self._access_token
        return self._refresh_access_token()

    def _refresh_access_token(self) -> str:
        endpoint = self.config.token_endpoint()
        if self.config.is_lwa():
            body = json.dumps(
                {
                    "grant_type": "client_credentials",
                    "client_id": self.config.credential_id,
                    "client_secret": self.config.credential_secret,
                    "scope": self.config.scope(),
                }
            ).encode("utf-8")
            headers = {"Content-Type": "application/json"}
        else:
            body = urllib.parse.urlencode(
                {
                    "grant_type": "client_credentials",
                    "client_id": self.config.credential_id,
                    "client_secret": self.config.credential_secret,
                    "scope": self.config.scope(),
                }
            ).encode("utf-8")
            basic = base64.b64encode(
                f"{self.config.credential_id}:{self.config.credential_secret}".encode("utf-8")
            ).decode("ascii")
            headers = {
                "Content-Type": "application/x-www-form-urlencoded",
                "Authorization": f"Basic {basic}",
            }

        request = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Creators API OAuth failed HTTP {exc.code}: {detail}") from exc

        token = str(payload.get("access_token") or "").strip()
        if not token:
            raise RuntimeError(f"Creators API OAuth response missing access_token: {payload}")

        expires_in = int(payload.get("expires_in") or 3600)
        self._access_token = token
        self._expires_at = time.time() + max(60, expires_in - 30)
        return token
