from functools import partial
import numpy
from threadpoolctl import threadpool_limits, threadpool_info

from ..logging import get_logger
logger = get_logger(__name__)

try:
    NP_USER_API = threadpool_info()[0]["user_api"]
except:
    NP_USER_API = "blas"


def get_threading_context(n_threads):
    logger.debug(f"Building threading context with {n_threads} threads for the {NP_USER_API} library")
    thread_context = partial(
        threadpool_limits, limits=n_threads, user_api=NP_USER_API)

    return thread_context
