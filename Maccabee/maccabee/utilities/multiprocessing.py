from multiprocessing import Process, Manager, cpu_count
from functools import partial
from ..logging import get_logger
logger = get_logger(__name__)


def parallel_map_wrapper_func(func, semaphore, results_list, args_index, args):
    semaphore.acquire()
    logger.info(f"Executing parallel map wrapper with index={args_index}")
    results_list[args_index] = func(*args)
    semaphore.release()

def robust_parallel_map(func, args_list, n_jobs=-1, retry_failed_jobs=True):
    """Quick and dirty parallel map capable of handling failure/sudden death of the worker processes.

    TODO finish this docstring.

    Args:
        func (type): Description of parameter `func`.
        args_list (type): Description of parameter `args_list`.
        n_jobs (type): Description of parameter `n_jobs`. Defaults to -1.
        retry_failed_jobs (type): xxx.

    Returns:
        type: Description of returned object.

    Raises:
        ExceptionName: Why the exception is raised.

    Examples
        Examples should be written in doctest format, and
        should illustrate how to use the function/class.
        >>>

    """
    if n_jobs == -1:
        n_jobs = cpu_count()

    logger.debug(f"Running robust parallel map with n_jobs={n_jobs}.")

    manager = Manager()
    n_arg_entries = len(args_list)

    results_list = manager.list([None]*n_arg_entries)
    semaphore = manager.Semaphore(n_jobs) # used to control parallelism

    wrapper_func = partial(parallel_map_wrapper_func,
        func,
        semaphore,
        results_list)

    processes = [] # store process instances for later control.
    for args_index, args in enumerate(args_list):
        logger.debug(f"Spawning process for arg index={args_index}.")
        p = Process(
            target=wrapper_func,
            args=(args_index, args))
        p.start()
        processes.append(p)

    logger.debug(f"Waiting for spawned processes to finish.")
    for process in processes:
        process.join()
        process.terminate()

    logger.debug(f"All processes terminated.")

    # Failure Recovery
    if retry_failed_jobs:
        for i, (res, args) in enumerate(zip(results_list, args_list)):
            if res is None:
                logger.error("Recovering from failed execution")
                results_list[i] = func(*args)

    # convert to standard list from shared mem
    results_list = list(results_list)

    return results_list
