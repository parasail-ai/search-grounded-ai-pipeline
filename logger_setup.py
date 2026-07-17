"""
logger_setup.py — Centralised logging configuration.

Usage:
    from logger_setup import get_logger
    logger = get_logger(__name__)

Log levels:
    DEBUG   — detailed trace: every SSE event sent, token counts, search params
    INFO    — normal operations: request received, search complete, answer sent
    WARNING — recoverable issues: 429 retry, slow response, truncated results
    ERROR   — failures: stream error, agent crash, bad request
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "server.log"
LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def configure(level: str = "INFO") -> None:
    global _configured
    if _configured:
        return
    _configured = True

    numeric = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)          # root accepts everything; handlers filter

    # ── Rotating file handler (DEBUG and above) ────────────────────────────
    fh = RotatingFileHandler(
        LOG_FILE,
        maxBytes=10 * 1024 * 1024,        # 10 MB per file
        backupCount=5,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    root.addHandler(fh)

    # ── Console handler (INFO and above by default, respects `level`) ──────
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(numeric)
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S"))
    root.addHandler(ch)

    # Silence noisy stdlib loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Call configure() first (server.py does this at startup)."""
    return logging.getLogger(name)
