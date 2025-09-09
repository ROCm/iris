# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Iris logging module - provides logging functionality.
"""

import logging

# Logging constants (compatible with Python logging levels)
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR


class IrisFormatter(logging.Formatter):
    """Custom formatter that automatically includes rank information when available."""

    def __init__(self):
        super().__init__()

    def format(self, record):
        # Check if rank information is available in the record
        if hasattr(record, "iris_rank") and hasattr(record, "iris_num_ranks"):
            prefix = f"[Iris] [{record.iris_rank}/{record.iris_num_ranks}]"
        else:
            prefix = "[Iris]"

        # Format the message with the appropriate prefix
        return f"{prefix} {record.getMessage()}"


# Logger instance that can be accessed as iris.logger
logger = logging.getLogger("iris")

# Set up iris logger
logger.setLevel(logging.INFO)  # Default level

# Add a console handler if none exists
if not logger.handlers:
    _console_handler = logging.StreamHandler()
    _formatter = IrisFormatter()
    _console_handler.setFormatter(_formatter)
    logger.addHandler(_console_handler)


def set_logger_level(level):
    """
    Set the logging level for the iris logger.

    Args:
        level: Logging level (iris.DEBUG, iris.INFO, iris.WARNING, iris.ERROR)
    """
    logger.setLevel(level)
