# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Iris logging module - provides logging functionality.
"""

import logging
import os
import sys

# Logging constants (compatible with Python logging levels)
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR


class IrisFormatter(logging.Formatter):
    """Custom formatter that includes timestamp, level, rank, and module information."""

    def __init__(self):
        super().__init__()

    def format(self, record):
        rank = getattr(record, "iris_rank", "?")
        num_ranks = getattr(record, "iris_num_ranks", "?")
        ts = self.formatTime(record, "%H:%M:%S")
        level = record.levelname
        # Only show [module] for internal iris logs (set by _log_rank),
        # not for user-facing ctx.info()/ctx.debug() etc.
        iris_internal = getattr(record, "iris_internal", False)
        if iris_internal:
            module = record.module
            return f"{ts} {level:<5s} [Iris] [{rank}/{num_ranks}] [{module}] {record.getMessage()}"
        return f"{ts} {level:<5s} [Iris] [{rank}/{num_ranks}] {record.getMessage()}"


# Logger instance that can be accessed as iris.logger
logger = logging.getLogger("iris")

# Set up iris logger
logger.setLevel(logging.INFO)  # Default level

# Override from environment
_env_level = os.environ.get("IRIS_LOG_LEVEL", "").upper()
if _env_level in ("DEBUG", "INFO", "WARNING", "ERROR"):
    logger.setLevel(getattr(logging, _env_level))

# Add a console handler if none exists
if not logger.handlers:
    _console_handler = logging.StreamHandler()
    _formatter = IrisFormatter()
    _console_handler.setFormatter(_formatter)
    logger.addHandler(_console_handler)


def _log_rank(level, msg, *args, rank=None, num_ranks=None):
    """Log with optional rank injection. Captures caller's module automatically."""
    if logger.isEnabledFor(level):
        # Capture caller's file/line so the formatter can show [module]
        frame = sys._getframe(1)
        record = logging.LogRecord(
            name=logger.name,
            level=level,
            pathname=frame.f_code.co_filename,
            lineno=frame.f_lineno,
            msg=msg,
            args=args,
            exc_info=None,
        )
        record.iris_internal = True
        if rank is not None:
            record.iris_rank = rank
        if num_ranks is not None:
            record.iris_num_ranks = num_ranks
        logger.handle(record)


def set_logger_level(level):
    """
    Set the logging level for the iris logger.

    Args:
        level: Logging level (iris.DEBUG, iris.INFO, iris.WARNING, iris.ERROR)

    Example:
        >>> ctx = iris.iris()
        >>> iris.set_logger_level(iris.DEBUG)
        >>> ctx.debug("This will now be visible")  # [Iris] [0/1] This will now be visible
    """
    logger.setLevel(level)
