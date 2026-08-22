from __future__ import annotations

"""OpenSearch connection helpers."""
import time
from urllib.parse import urlparse

from opensearchpy import OpenSearch

from .config import OPENSEARCH_URL


def get_client(url: str = OPENSEARCH_URL) -> OpenSearch:
    """Client for the local demo cluster (no auth — security plugin is off)."""
    parsed = urlparse(url)
    return OpenSearch(
        hosts=[{"host": parsed.hostname, "port": parsed.port or 9200}],
        use_ssl=parsed.scheme == "https",
        verify_certs=False,
        ssl_show_warn=False,
        timeout=60,
        max_retries=3,
        retry_on_timeout=True,
    )


def wait_for_cluster(client: OpenSearch, timeout: int = 60) -> dict:
    """Block until the cluster answers, so notebooks fail loudly, not weirdly."""
    deadline = time.time() + timeout
    last_err = None
    while time.time() < deadline:
        try:
            return client.cluster.health(wait_for_status="yellow", request_timeout=5)
        except Exception as exc:          # connection refused while the JVM boots
            last_err = exc
            time.sleep(2)
    raise RuntimeError(
        f"OpenSearch did not become ready within {timeout}s. Is "
        f"`docker compose -f docker/docker-compose.yml up -d` running? "
        f"Last error: {last_err}"
    )
