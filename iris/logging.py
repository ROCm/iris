"""Compatibility shim — canonical code lives in iris.host.logging.logging"""
from iris.host.logging.logging import *  # noqa: F401,F403
from iris.host.logging.logging import set_logger_level, logger, DEBUG, INFO, WARNING, ERROR  # explicit
