"""
Logging configuration for CeDNe.

Use `from cedne.core.logger import logger` to log.
"""

import logging

logger = logging.getLogger("cedne")
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

file_handler = logging.FileHandler("CeDNe.log", mode='w')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)