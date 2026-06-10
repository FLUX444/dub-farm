from __future__ import annotations

import logging
import os
from urllib.parse import urlparse

from dub_farm.config import AppConfig, ProxyConfig

logger = logging.getLogger(__name__)


def proxy_url(config: ProxyConfig) -> str:
    scheme = config.scheme or "http"
    return f"{scheme}://{config.host}:{config.port}"


def should_bypass_proxy(url: str, config: ProxyConfig) -> bool:
    """Skip proxy for local services (Docker vLLM, Ollama, etc.)."""
    host = (urlparse(url).hostname or "").lower()
    if not host:
        return True
    bypass_hosts = {item.lower() for item in config.no_proxy}
    return host in bypass_hosts


def httpx_proxy_for(config: AppConfig, url: str) -> str | None:
    if not config.proxy.enabled:
        return None
    if should_bypass_proxy(url, config.proxy):
        return None
    return proxy_url(config.proxy)


def apply_proxy_env(config: AppConfig) -> None:
    """Apply proxy env vars for Hugging Face / transformers / urllib downloads."""
    if not config.proxy.enabled:
        return

    url = proxy_url(config.proxy)
    no_proxy = ",".join(config.proxy.no_proxy)
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ[key] = url
    os.environ["NO_PROXY"] = no_proxy
    os.environ["no_proxy"] = no_proxy
    logger.info("Proxy enabled: %s (no_proxy: %s)", url, no_proxy)
