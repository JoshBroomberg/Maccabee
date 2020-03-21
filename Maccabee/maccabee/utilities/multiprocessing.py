from multiprocessing import Process, Manager, cpu_count
from functools import partial
import time

from ..logging import get_logger
logger = get_logger(__name__)


class MultiprocessingNullResult():
    pass

class MultiprocessingExceptionResult():
    def __init__(self, index, base_exception):
        self.index = index
        self.base_exception = base_exception

def eval_wrapped_function(func, args):
    """Evaluates the function given as `func` with arguments given in `args.`
    Introspection of `args` is used to determine if its contents should be
    splatted into the function.

    Args:
        func (function): The function to evalaute
        args (object): An iterable set of arguments or a non-iterable argument to be supplied to `func` during evaluation.

    Returns:
        object: the value produced by the evaluation of `func`.
    """
    if hasattr(args, "__len__"):
        result = func(*args)
    else:
        result = func(args)
    return result

def worker_func(target_func,
                args_list, results_list,
                pending_index_queue,
                complete_index_list,
                exception_list,
                current_worker_index_map, p_uid):
    """This function is run in a subprocess to create an asynchronous worker. The worker executes the function in `target_func` for the arguments in `args_list` by pulling pending indeces from the `pending_index_queue`. Results are written to the `results_list`. The remaining arguments are used for error handling and robustness as described below.

    Args:
        target_func (function): The function executed by the worker.
        args_list (list): A shared list of arguments against which the `target_func` is to be evaluated.
        results_list (list): A shared list of results of the function evaluation. The length will match the `args_list` with each result corresponding to a set of arguments. The default list entry value is
        pending_index_queue (:class:`multiprocessing.Queue`): A shared queue containing the argument indeces pending evaluation.
        complete_index_list (list): A shared list of indeces for which execution has been completed.
        exception_list (list): A shared list of `MultiprocessingExceptionResult` objects corresponding to evaluations that generated exceptions.
        current_worker_index_map (dict): A shared dict mapping worker `p_uid` values to current argument indeces being worked on.
        p_uid (int): a unique identifier for this worker process.
    """

    # Run until stopped.
    while True:
        # Get the next argument index to execute.
        current_index = pending_index_queue.get()
        logger.debug("Worker: %s  working on index: %s", p_uid, current_index)

        # Fetch the corresponding arguments from the shared storage.
        current_args = args_list[current_index]

        # Record the index being worked on for failure recovery.
        current_worker_index_map[p_uid] = current_index

        # Execute function.
        try:
            results_list[current_index] = eval_wrapped_function(target_func, current_args)
            logger.debug("Worker: %s produced result: %s",
                         p_uid, results_list[current_index])
        except Exception as e:
            # Catch exceptions and append to exception list for later processing.
            logger.exception("Worker caught exception in target function execution.")
            exception_list.append(MultiprocessingExceptionResult(
                index=current_index, base_exception=e))

        # Mark index as completed by adding it to the completed list.
        complete_index_list.append(current_index)

        # Unstore current work index to prevent accidently retry.
        del current_worker_index_map[p_uid]

def start_worker_proc(worker_func, p_uid, worker_store):
    """
    This convenience function starts a background worker subprocess using the `worker_func` function defined above. The worker is given the supplied `p_uid` unique identified and the process instance is stored in the supplied `worker_store`.
    """
    proc = Process(
            target=worker_func,
            args=(p_uid,))
    proc.start()
    worker_store[proc] = p_uid

def robust_parallel_map(target_func, args_list, n_jobs=-1, raise_exceptions=True):
    """This is the main function in this module. It evaluates the supplied `target_func` for all of the arguments in `args_list` using `n_jobs` parallel processes. This function is responsible for managing the pool of worker supprocesses and handling failures. This function is robust because it designed to recover from the sudden death worker processes. It does this by maintaining the state required to retry jobs from failed workers and continuously monitoring worker health.

    Args:
        target_func (function): The function to be mapped over the list of arguments.
        args_list (list): The list of arguments over which the `target_func` is evaluated.
        n_jobs (int): The number of worker subprocesses to use.
        raise_exceptions (bool): Whether or not to raise exceptions. If false, all evaluations that produce exceptions will have a `MultiprocessingNullResult` in the results list. Defaults to True.

    Returns:
        list: a list where each entry contains the value of the function evaluated at the corresponding argument in the supplied `args_list`.

    Examples
        >>> def add_one(i):
        >>>   return i + 1
        >>> robust_parallel_map(add_one, [1, 2, 3])
        [2, 3, 4]

        >>> robust_parallel_map(add_one, [1, 2, "3"], raise_exceptions=False)
        [2, 3, <MultiprocessingNullResult at 0x7f78af649278>]

        >>> robust_parallel_map(add_one, [1, 2, "3"], raise_exceptions=True)
        TypeError: must be str, not int
    """
    if n_jobs == -1:
        n_jobs = cpu_count()

    n_arg_entries = len(args_list)

    # The manager is responsible for maintain all state shared between workers.
    manager = Manager()

    ## Shared State ##
    args_list = manager.list(list(args_list)) # the arguments for eval
    results_list = manager.list([MultiprocessingNullResult()]*n_arg_entries) # results
    complete_index_list = manager.list() # all completed arg indeces
    exception_list = manager.list() # all exceptions (of type MultiprocessingExceptionResult)

    # Queue of pending argument indeces
    pending_index_queue = manager.Queue(n_arg_entries)

    # Dictionary mapping process p_uids to current argument index being executed.
    # Used to retry evaluation on worker death.
    current_worker_index_map = manager.dict()

    # Enqueue jobs
    for arg_index in range(n_arg_entries):
        pending_index_queue.put_nowait(arg_index)

    # Prep worker func
    bound_worker_func = partial(worker_func, target_func,
                                args_list, results_list,
                                pending_index_queue,
                                complete_index_list,
                                exception_list,
                                current_worker_index_map)

    # Build worker pool.
    worker_processes = {}
    for p_uid in range(n_jobs):
        start_worker_proc(bound_worker_func, p_uid, worker_processes)

    # While pending tasks, monitor worker health.
    while len(complete_index_list) < n_arg_entries:
        dead_procs = []
        for proc, p_uid in worker_processes.items():

            # Detect worker death (never occurs intentionally)
            if not proc.is_alive():
                logger.warning("Worker with p_uid %s died with exitcode %s", p_uid, proc.exitcode)
                dead_procs.append(proc)

                # Check for failed job to requeue
                if p_uid in current_worker_index_map:
                    logger.warning("Worker with p_uid %s had work on failure...", p_uid)
                    potentially_failed_index = current_worker_index_map[p_uid]

                    # If no result for the argument that was being worked on, requeue.
                    if isinstance(results_list[potentially_failed_index], MultiprocessingNullResult):
                        logger.warning(f"Requeueing arg index {potentially_failed_index} to recover from worker failure.")
                        pending_index_queue.put_nowait(potentially_failed_index)
                        del current_worker_index_map[p_uid]
                    else:
                        # This should never happen. It would imply the worker died after
                        # evaluation of th function but before updating the current
                        # work mapping.
                        logger.error("Working proc died without updating current work...")

        # Start new workers to replace all dead workers.
        for proc in dead_procs:
            p_uid = worker_processes[proc] # recycle dead worker p_uid
            start_worker_proc(bound_worker_func, p_uid, worker_processes)

            proc.close() # free resources
            del worker_processes[proc] # drop from the worker store

            logger.warning(f"Started new worker. Worker count: %s", len(worker_processes))

        # Wait a second before rechecking pool state
        time.sleep(1)

    # Work complete, terminate all workers.
    for proc in worker_processes:
        proc.terminate()

    # Wait for termination.
    time.sleep(0.25)

    # Free resources.
    for proc in worker_processes:
        proc.close()

    # If exceptions occured, raise them if set to do so.
    if len(exception_list) > 0 and raise_exceptions:
        for exception_result in exception_list:
            raise(exception_result.base_exception)

    results_list = list(results_list)

    return results_list
