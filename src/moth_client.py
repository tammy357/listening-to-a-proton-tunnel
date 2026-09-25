"""HTTP client for the Moth Quantum API.

Base URL and routes are taken from the OpenAPI document at
https://api.mothquantum.com/openapi.json (moth-api v0.41.0).

Authentication is ``Authorization: Bearer`` with the key in the
``MOTH_API_KEY`` environment variable. The key is never stored in source.

This client does not use atlascloud.ai or any Atlas Cloud SDK.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import requests

BASE_URL = "https://api.mothquantum.com/api/v1"
API_KEY_ENV = "MOTH_API_KEY"
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})


class MothError(RuntimeError):
    """The Moth API rejected a call or a job ended without a result."""


class MothClient:
    """Submit engines, poll status, and fetch results.

    Args:
        api_key: Explicit key. When omitted, ``MOTH_API_KEY`` is read from
            the environment.
        base_url: API root. Defaults to ``https://api.mothquantum.com/api/v1``.
    """

    def __init__(self, api_key: str | None = None, base_url: str = BASE_URL) -> None:
        if api_key is None:
            api_key = os.environ.get(API_KEY_ENV, "").strip() or None
        if not api_key:
            raise MothError(f"{API_KEY_ENV} is not set.")
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")

    def submit(
        self,
        engine_id: str,
        params: dict,
        input_files: dict[str, str] | None = None,
    ) -> str:
        """POST ``/engines/{engine_id}/process`` and return ``job_id``."""

        body: dict[str, Any] = {"params": params}
        if input_files:
            body["input_files"] = input_files
        payload = self._request(
            "POST",
            f"/engines/{engine_id}/process",
            json_body=body,
            expected=(200, 201, 202),
        )
        job_id = payload.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise MothError(f"{engine_id} did not return a job_id.")
        return job_id

    def wait(self, job_id: str, interval_s: float = 2.0, timeout_s: float = 900.0) -> dict:
        """Poll ``/jobs/{id}/status`` every 2 seconds until a terminal status."""

        deadline = time.monotonic() + timeout_s
        while True:
            status = self.status(job_id)
            state = str(status.get("status", ""))
            if state in TERMINAL_STATUSES:
                return status
            if time.monotonic() >= deadline:
                raise MothError(f"Job {job_id} still {state or 'unknown'} after {timeout_s:.0f}s.")
            time.sleep(interval_s)

    def status(self, job_id: str) -> dict:
        """GET ``/jobs/{id}/status``."""

        return self._request("GET", f"/jobs/{job_id}/status", expected=(200,))

    def result(self, job_id: str) -> dict:
        """GET ``/jobs/{id}/result``."""

        return self._request("GET", f"/jobs/{job_id}/result", expected=(200,))

    def upload_asset(self, path: Path, content_type: str) -> str:
        """Create a pending asset, PUT the bytes, and complete the upload.

        Returns the ``asset_id`` for an ``input_files`` slot.
        """

        data = Path(path).read_bytes()
        created = self._request(
            "POST",
            "/assets",
            json_body={
                "filename": Path(path).name,
                "content_type": content_type,
                "size_bytes": len(data),
            },
            expected=(200, 201),
        )
        upload = created.get("upload") or {}
        url = upload.get("url")
        method = str(upload.get("method") or "PUT")
        headers = dict(upload.get("headers") or {})
        if not url:
            raise MothError("Asset creation did not return an upload URL.")
        put = requests.request(method, url, data=data, headers=headers, timeout=120)
        if put.status_code not in (200, 201, 204):
            raise MothError(f"Asset upload failed with HTTP {put.status_code}.")
        asset_id = created.get("asset_id")
        if not isinstance(asset_id, str) or not asset_id:
            raise MothError("Asset creation did not return an asset_id.")
        self._request("POST", f"/assets/{asset_id}/complete", expected=(200, 201))
        return asset_id

    def _request(
        self,
        method: str,
        path: str,
        json_body: dict | None = None,
        expected: tuple[int, ...] = (200,),
    ) -> dict:
        url = f"{self.base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/json",
        }
        if json_body is not None:
            headers["Content-Type"] = "application/json"
        delay = 2.0
        last_error = "request failed"
        for _attempt in range(6):
            response = requests.request(
                method,
                url,
                headers=headers,
                json=json_body,
                timeout=60,
            )
            if response.status_code in (429, 503) and _attempt < 5:
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
                last_error = f"HTTP {response.status_code}"
                continue
            if response.status_code not in expected:
                detail = response.text[:500]
                raise MothError(f"{method} {path} returned HTTP {response.status_code}: {detail}")
            if not response.content:
                return {}
            return response.json()
        raise MothError(f"{method} {path} kept failing: {last_error}")


def _smoke() -> None:
    client = MothClient()
    job_id = client.submit("coin-toss-v1", {"shots": 10})
    print(f"job_id={job_id}")
    status = client.wait(job_id)
    print(f"status={status.get('status')}")
    if status.get("status") != "completed":
        raise SystemExit(f"coin-toss-v1 ended {status.get('status')}: {status.get('error')}")
    print(client.result(job_id))


if __name__ == "__main__":
    _smoke()
