import logging
import pytest
import sys
import os

# Add the iris package to the path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# Test logging functionality without requiring full iris environment
def test_logging_constants():
    """Test that logging constants are properly defined."""
    # Import iris.logging module directly to avoid MPI dependencies
    from iris.logging import DEBUG, INFO, WARNING, ERROR

    # Verify constants match Python logging levels
    assert DEBUG == logging.DEBUG
    assert INFO == logging.INFO
    assert WARNING == logging.WARNING
    assert ERROR == logging.ERROR


def test_set_logger_level():
    """Test the set_logger_level function."""
    from iris.logging import set_logger_level, logger, DEBUG, INFO

    # Test setting different levels
    set_logger_level(DEBUG)
    assert logger.level == logging.DEBUG

    set_logger_level(INFO)
    assert logger.level == logging.INFO


def test_logger_setup():
    """Test that the iris logger is properly configured."""
    from iris.logging import logger

    # Verify logger name
    assert logger.name == "iris"

    # Verify default level
    assert logger.level == logging.INFO

    # Verify handler exists
    assert len(logger.handlers) > 0

    # Verify handler is a StreamHandler
    assert isinstance(logger.handlers[0], logging.StreamHandler)


def test_iris_debug_logging():
    """Test that Iris debug logging convenience methods work correctly."""
    from iris.logging import logger
    import logging

    # Test the _log_with_rank method logic by simulating it
    def _log_with_rank(level, message, rank=0, num_ranks=1):
        """Simulate the _log_with_rank method."""
        record = logging.LogRecord(
            name=logger.name, level=level, pathname="", lineno=0, msg=message, args=(), exc_info=None
        )
        # Inject rank information into the record
        record.iris_rank = rank
        record.iris_num_ranks = num_ranks
        logger.handle(record)

    # Capture log output
    import io

    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    from iris.logging import IrisFormatter

    handler.setFormatter(IrisFormatter())

    # Remove existing handlers and add our capture handler
    original_handlers = logger.handlers[:]
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    try:
        # Test the rank-aware logging
        _log_with_rank(logging.DEBUG, "allocate: num_elements = 100, dtype = None", rank=0, num_ranks=1)

        output = log_capture.getvalue()
        assert "[Iris] [0/1] allocate: num_elements = 100, dtype = None" in output

    finally:
        # Restore original handlers
        logger.handlers.clear()
        for handler in original_handlers:
            logger.addHandler(handler)


def test_logger_api_usage():
    """Test direct logger API usage."""
    from iris.logging import logger, set_logger_level, DEBUG, INFO, IrisFormatter

    # Capture log output
    import io
    import logging

    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setFormatter(IrisFormatter())

    # Remove existing handlers and add our capture handler
    logger.handlers.clear()
    logger.addHandler(handler)

    # Test logging at different levels
    set_logger_level(INFO)
    logger.info("Test info message")
    logger.debug("Test debug message (should be hidden)")

    set_logger_level(DEBUG)
    logger.debug("Test debug message (should be visible)")

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
        from iris import set_logger_level, logger, DEBUG, INFO, WARNING, ERROR

        # If we get here, the imports worked
        assert set_logger_level is not None
        assert logger is not None
        assert logger.name == "iris"
        assert DEBUG == logging.DEBUG
        assert INFO == logging.INFO
        assert WARNING == logging.WARNING
        assert ERROR == logging.ERROR
    except ImportError as e:
        # If iris module can't be imported due to dependencies, skip this test
        pytest.skip(f"Skipping API import test due to dependency issues: {e}")


if __name__ == "__main__":
    # Run basic tests
    test_logging_constants()
    test_set_logger_level()
    test_logger_setup()
    test_logger_api_usage()
    test_iris_formatter()
    print("All basic logging tests passed!")
