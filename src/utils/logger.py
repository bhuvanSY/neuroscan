"""Structured logging utility for NeuroScan AI."""

import os
import sys
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict


class StructuredJsonFormatter(logging.Formatter):
    """Custom formatter emitting logs in JSON format for automated monitoring."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "extra_data") and isinstance(record.extra_data, dict):
            log_obj.update(record.extra_data)
        return json.dumps(log_obj)


def get_logger(name: str = "neuroscan") -> logging.Logger:
    """Get or configure structured logger.
    
    Args:
        name: Name of logger (usually module __name__)
        
    Returns:
        Configured logging.Logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        log_level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
        log_level = getattr(logging, log_level_name, logging.INFO)
        logger.setLevel(log_level)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)

        # Standard clean human-readable formatter for terminal
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger
