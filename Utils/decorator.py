"""
@author: F. B. Pérez Maurera
"""

import logging
import time

from typing import Any, Tuple, Dict

logger = logging.getLogger("contentwise-impressions")


def timeit(method):
    """
    Decorator that measures execution time of a method. This is a modified version from the one published at:
    https://medium.com/pythonhive/python-decorator-to-measure-the-execution-time-of-methods-fa04cb6bb36d
    """

    def timed(*args: Tuple[Any, ...], **kwargs: Dict[str, Any]) -> None:
        ts = time.time()
        result = method(*args, **kwargs)
        te = time.time()
        logger.info(f"{method.__name__}|Execution time: {te - ts:.2f}s")
        return result

    return timed
