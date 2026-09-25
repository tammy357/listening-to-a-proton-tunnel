"""Stub client for the Moth Quantum Atlas platform.

The HTTP routes live in the platform docs. Until those calls are written,
every method raises ``NotImplementedError`` and the dashboard stays on
"Atlas: not connected".

The API key is read from the ``ATLAS_API_KEY`` environment variable. It is
never stored in source.
"""

from __future__ import annotations

import os

ATLAS_API_KEY_ENV = "ATLAS_API_KEY"
ATLAS_PENDING = (
    "Atlas integration pending: fill in from platform.mothquantum.com docs"
)


class AtlasClient:
    """Placeholder for the Atlas job API.

    Args:
        api_key: Explicit key. When omitted, ``ATLAS_API_KEY`` is read from
            the environment. An empty environment leaves the key as ``None``.
    """

    def __init__(self, api_key: str | None = None) -> None:
        if api_key is None:
            raw = os.environ.get(ATLAS_API_KEY_ENV, "")
            api_key = raw.strip() or None
        self.api_key = api_key

    def submit_job(self, payload: dict) -> dict:
        """Submit a circuit job. Not implemented."""

        # TODO: POST the payload to the Atlas jobs route.
        raise NotImplementedError(ATLAS_PENDING)

    def get_job(self, job_id: str) -> dict:
        """Fetch one job record. Not implemented."""

        # TODO: GET the job identified by job_id.
        raise NotImplementedError(ATLAS_PENDING)

    def list_devices(self) -> list:
        """List backends the platform will run. Not implemented."""

        # TODO: GET the device list from the platform docs.
        raise NotImplementedError(ATLAS_PENDING)
