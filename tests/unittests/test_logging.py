# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import logging
import pytest
import iris


def test_logging_constants():
    """Test that logging constants are properly defined."""
    # Verify constants match Python logging levels
    assert iris.DEBUG == logging.DEBUG
    assert iris.INFO == logging.INFO
    assert iris.WARNING == logging.WARNING
    assert iris.ERROR == logging.ERROR


def test_set_logger_level():
    """Test the set_logger_level function."""
    # Test setting different levels
    iris.set_logger_level(iris.DEBUG)
    assert iris.logger.level == logging.DEBUG

    iris.set_logger_level(iris.INFO)
    assert iris.logger.level == logging.INFO


def test_logger_setup():
    """Test that the iris logger is properly configured."""
    # Verify logger name
    assert iris.logger.name == "iris"

    # Verify default level
    assert iris.logger.level == logging.INFO

    # Verify handler exists
    assert len(iris.logger.handlers) > 0

    # Verify handler is a StreamHandler
    assert isinstance(iris.logger.handlers[0], logging.StreamHandler)


def test_iris_debug_logging():
    """Test that Iris debug logging convenience methods work correctly."""
    import logging

    # Test the _log_with_rank method logic by simulating it
    def _log_with_rank(level, message, rank=0, num_ranks=1):
        """Simulate the _log_with_rank method."""
        record = logging.LogRecord(
            name=iris.logger.name, level=level, pathname="", lineno=0, msg=message, args=(), exc_info=None
        )
        # Inject rank information into the record
        record.iris_rank = rank
        record.iris_num_ranks = num_ranks
        iris.logger.handle(record)

    # Capture log output
    import io

    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    from iris.logging import IrisFormatter

    handler.setFormatter(IrisFormatter())

    # Remove existing handlers and add our capture handler
    original_handlers = iris.logger.handlers[:]
    iris.logger.handlers.clear()
    iris.logger.addHandler(handler)
    iris.logger.setLevel(logging.DEBUG)

    try:
        # Test the rank-aware logging
        _log_with_rank(logging.DEBUG, "allocate: num_elements = 100, dtype = None", rank=0, num_ranks=1)

        output = log_capture.getvalue()
        assert "[Iris] [0/1] allocate: num_elements = 100, dtype = None" in output

    finally:
        # Restore original handlers
        iris.logger.handlers.clear()
        for handler in original_handlers:
            iris.logger.addHandler(handler)


def test_logger_api_usage():
    """Test direct logger API usage."""
    # Capture log output
    import io
    import logging

    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    from iris.logging import IrisFormatter

    handler.setFormatter(IrisFormatter())

    # Remove existing handlers and add our capture handler
    iris.logger.handlers.clear()
    iris.logger.addHandler(handler)

    # Test logging at different levels
    iris.set_logger_level(iris.INFO)
    iris.logger.info("Test info message")
    iris.logger.debug("Test debug message (should be hidden)")

    iris.set_logger_level(iris.DEBUG)
    iris.logger.debug("Test debug message (should be visible)")

    output = log_capture.getvalue()
    assert "[Iris] Test info message" in output
    assert "[Iris] Test debug message (should be visible)" in output
    # The hidden debug message should not appear
    lines = output.split("\n")
    hidden_debug_count = sum(1 for line in lines if "should be hidden" in line)
    assert hidden_debug_count == 0


def test_iris_formatter():
    """Test the IrisFormatter behavior."""
    from iris.logging import IrisFormatter
    import logging

    formatter = IrisFormatter()

    # Test record without rank information
    record_no_rank = logging.LogRecord(
        name="iris", level=logging.INFO, pathname="", lineno=0, msg="Test message without rank", args=(), exc_info=None
    )

    formatted_no_rank = formatter.format(record_no_rank)
    assert formatted_no_rank == "[Iris] Test message without rank"

    # Test record with rank information
    record_with_rank = logging.LogRecord(
        name="iris", level=logging.INFO, pathname="", lineno=0, msg="Test message with rank", args=(), exc_info=None
    )
    record_with_rank.iris_rank = 2
    record_with_rank.iris_num_ranks = 4

    formatted_with_rank = formatter.format(record_with_rank)
    assert formatted_with_rank == "[Iris] [2/4] Test message with rank"


def test_api_import():
    """Test that the new API can be imported from the main iris module."""
    # This test verifies the __init__.py exports work correctly
    try:
        # If we get here, the imports worked
        assert iris.set_logger_level is not None
        assert iris.logger is not None
        assert iris.logger.name == "iris"
        assert iris.DEBUG == logging.DEBUG
        assert iris.INFO == logging.INFO
        assert iris.WARNING == logging.WARNING
        assert iris.ERROR == logging.ERROR
    except ImportError as e:
        # If iris module can't be imported due to dependencies, skip this test
        pytest.skip(f"Skipping API import test due to dependency issues: {e}")
