from functools import partial
import numpy
from threadpoolctl import threadpool_limits, threadpool_info

try:
    NP_USER_API = threadpool_info()[0]["user_api"]
except:
    NP_USER_API = "blas"


single_threaded_context = partial(
    threadpool_limits, limits=1, user_api=NP_USER_API)
