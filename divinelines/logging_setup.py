"""Structured logging.

One configuration point for the whole platform.  Human-readable by default,
JSON lines when ``DL_LOG_JSON=1`` so logs can be shipped/parsed.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from .config import settings

_CONFIGURED = False

_RESERVED = set(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
) | {"message", "asctime", "taskName"}


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                try:
                    json.dumps(value)
                    payload[key] = value
                except (TypeError, ValueError):
                    payload[key] = repr(value)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


class _TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k not in _RESERVED and not k.startswith("_")
        }
        if extras:
            rendered = " ".join(f"{k}={v}" for k, v in extras.items())
            return f"{base} | {rendered}"
        return base


def configure_logging(level: str | None = None, *, force: bool = False) -> None:
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    root = logging.getLogger("divinelines")
    root.handlers.clear()
    root.setLevel(level or settings.log_level)
    root.propagate = False

    stream = logging.StreamHandler(sys.stderr)
    if settings.log_json:
        stream.setFormatter(_JsonFormatter())
    else:
        stream.setFormatter(
            _TextFormatter("%(asctime)s %(levelname)-7s %(name)-28s %(message)s", "%H:%M:%S")
        )
    root.addHandler(stream)

    settings.paths.ensure()
    file_handler = logging.FileHandler(
        settings.paths.logs_dir / "divinelines.log", encoding="utf-8"
    )
    file_handler.setFormatter(_JsonFormatter())
    file_handler.setLevel(logging.DEBUG)
    root.addHandler(file_handler)

    _CONFIGURED = True


class _SafeAdapter(logging.LoggerAdapter):
    """Renames structured fields that would collide with ``LogRecord`` slots.

    ``logging`` raises when ``extra`` contains a reserved key such as
    ``message`` or ``name``.  A pipeline should never die because of a log
    line, so colliding keys are prefixed instead of exploding.
    """

    def process(self, msg: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        extra = kwargs.get("extra")
        if extra:
            kwargs["extra"] = {
                (f"field_{k}" if k in _RESERVED else k): v for k, v in extra.items()
            }
        return msg, kwargs


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    if not name.startswith("divinelines"):
        name = f"divinelines.{name}"
    return _SafeAdapter(logging.getLogger(name), {})
