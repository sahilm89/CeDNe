"""
Logging configuration for CeDNe.

Use `from cedne.core.logger import logger` to log.
"""

import logging

logger = logging.getLogger("cedne")
logger.addHandler(logging.NullHandler())


def configure_logging(
    log_file: str | None = None, console: bool = True, level: int = logging.INFO
):
    """Configure CeDNe logging for scripts and notebooks.

    Library imports stay quiet by default; applications can opt into console
    and/or file logging without CeDNe creating files as an import side effect.
    """
    logger.setLevel(level)
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")

    for handler in list(logger.handlers):
        if not isinstance(handler, logging.NullHandler):
            logger.removeHandler(handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
