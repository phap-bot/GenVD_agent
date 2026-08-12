from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from urllib.request import Request, urlopen

logger = logging.getLogger("auto_dubbing.ninerouter")

DEFAULT_9ROUTER_BASE_URL = "http://localhost:20128/v1"
_ninerouter_process: subprocess.Popen | None = None


def get_9router_base_url() -> str:
    return (
        os.environ.get("AUTODUB_9ROUTER_BASE_URL")
        or os.environ.get("AUTODUB_STT_BASE_URL")
        or DEFAULT_9ROUTER_BASE_URL
    ).rstrip("/")


def is_9router_running(base_url: str | None = None, timeout: float = 1.5) -> bool:
    target_base = (base_url or get_9router_base_url()).rstrip("/")
    target_url = f"{target_base}/models"
    try:
        request = Request(target_url, headers={"User-Agent": "auto-dubbing-healthcheck"}, method="GET")
        with urlopen(request, timeout=timeout) as response:
            return response.status in (200, 401, 403)
    except Exception:
        return False


def ensure_9router_running(base_url: str | None = None, wait_timeout: float = 12.0) -> bool:
    global _ninerouter_process

    target_base = (base_url or get_9router_base_url()).rstrip("/")
    if is_9router_running(target_base, timeout=1.5):
        return True

    auto_start = os.environ.get("AUTODUB_AUTO_START_9ROUTER", "true").strip().lower()
    if auto_start in ("false", "0", "off", "no"):
        logger.info("ninerouter.autostart.disabled_by_env")
        return False

    cmd = _find_9router_command(target_base)
    if not cmd:
        logger.warning(
            "ninerouter.autostart.failed cause=binary_not_found hint='Install 9router via `npm install -g 9router`'"
        )
        return False

    logger.info("ninerouter.autostart.launching cmd=%s base_url=%s", cmd, target_base)
    try:
        creation_flags = 0
        if os.name == "nt":
            creation_flags = subprocess.CREATE_NO_WINDOW | getattr(subprocess, "DETACHED_PROCESS", 0x00000008)

        _ninerouter_process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creation_flags,
        )
    except Exception as exc:
        logger.error("ninerouter.autostart.launch_error error=%s", exc)
        return False

    deadline = time.monotonic() + wait_timeout
    while time.monotonic() < deadline:
        if is_9router_running(target_base, timeout=1.0):
            logger.info("ninerouter.autostart.ready base_url=%s", target_base)
            return True
        time.sleep(0.5)

    logger.warning("ninerouter.autostart.timeout base_url=%s wait_timeout=%.1f", target_base, wait_timeout)
    return False


def _find_9router_command(base_url: str) -> list[str] | None:
    port = "20128"
    if ":" in base_url:
        try:
            port = base_url.split(":")[-1].split("/")[0]
        except Exception:
            port = "20128"

    binary_path = shutil.which("9router")
    if binary_path:
        return [binary_path, "-n", "-p", port]

    npx_path = shutil.which("npx")
    if npx_path:
        return [npx_path, "-y", "9router", "-n", "-p", port]

    return None
