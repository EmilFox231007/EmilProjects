"""
The configuration file for logging in this project.
To change logs level for debug, change the level parameter in the setup_logging function to logging.DEBUG
"""

import logging
import os
import sys


def setup_logging(level=logging.INFO, log_file="logs/app.log"):
    """
    Set up logging to both console and a file with a unified format.

    Parameters
    ----------
    level : int, optional
        Logging level (e.g., logging.INFO, logging.DEBUG). Default is logging.INFO.
    log_file : str, optional
        Path to the log file where logs will be saved. Default is "logs/app.log".

    Notes
    -----
    - Existing handlers are cleared before setting new ones.
    - Logs will be printed to both the console and the specified file.
    - The log format includes timestamps, log level, logger name, and message.
    - Creates the log directory if it does not exist.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    # File handler
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(formatter)

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)


def configure_logging(enabled=True, level=logging.INFO):
    """
    Configure basic logging settings or disable logging entirely.

    Parameters
    ----------
    enabled : bool, optional
        If False, disables all logging below CRITICAL. Default is True.
    level : int, optional
        Logging level to set if logging is enabled. Default is logging.INFO.

    Notes
    -----
    This function is a simpler alternative to `setup_logging` and does not
    set up custom handlers or formatting.
    """
    if not enabled:
        logging.disable(logging.CRITICAL)
    else:
        logging.basicConfig(level=level)
