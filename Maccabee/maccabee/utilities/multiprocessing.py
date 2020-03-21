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
    # Assume that multiple arguments are stored in
    # a structure with a length. EG a tuple, list etc.
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

    while True:
        current_index = pending_index_queue.get()
        logger.debug("Worker: %s  working on index: %s", p_uid, current_index)
        current_args = args_list[current_index]

        # Store current work index
        current_worker_index_map[p_uid] = current_index

        # Execute
        try:
            results_list[current_index] = eval_wrapped_function(target_func, current_args)
            logger.debug("Worker: %s produced result: %s",
                         p_uid, results_list[current_index])
        except Exception as e:
            logger.exception("Worker caught exception in target function execution.")
            exception_list.append(MultiprocessingExceptionResult(
                index=current_index, base_exception=e))

        complete_index_list.append(current_index)

        # Unstore current work index
        del current_worker_index_map[p_uid]

def start_worker_proc(worker_func, p_uid, worker_store):
    proc = Process(
            target=worker_func,
            args=(p_uid,))
    proc.start()
    worker_store[proc] = p_uid

def robust_parallel_map(target_func, args_list, n_jobs=-1, raise_exceptions=True):
    if n_jobs == -1:
        n_jobs = cpu_count()

    n_arg_entries = len(args_list)

    manager = Manager()

    args_list = manager.list(list(args_list))
    results_list = manager.list([MultiprocessingNullResult()]*n_arg_entries)
    complete_index_list = manager.list()
    exception_list = manager.list()

    pending_index_queue = manager.Queue(n_arg_entries)
    current_worker_index_map = manager.dict()

    # enqueue jobs
    for arg_index in range(n_arg_entries):
        pending_index_queue.put_nowait(arg_index)

    # Prep worker func
    bound_worker_func = partial(worker_func, target_func,
                                args_list, results_list,
                                pending_index_queue,
                                complete_index_list,
                                exception_list,
                                current_worker_index_map)

    worker_processes = {}
    for p_uid in range(n_jobs):
        start_worker_proc(bound_worker_func, p_uid, worker_processes)

    while len(complete_index_list) < n_arg_entries:
        dead_procs = []
        for proc, p_uid in worker_processes.items():
            if not proc.is_alive():
                logger.warning("Worker with p_uid %s died with exitcode %s", p_uid, proc.exitcode)
                dead_procs.append(proc)

                # Check for failed job to requeue
                if p_uid in current_worker_index_map:
                    logger.warning("Worker with p_uid %s had work on failure...", p_uid)
                    potentially_failed_index = current_worker_index_map[p_uid]

                    if isinstance(results_list[potentially_failed_index], MultiprocessingNullResult):
                        logger.warning(f"Requeueing arg index {potentially_failed_index} to recover from worker failure.")
                        pending_index_queue.put_nowait(potentially_failed_index)
                        del current_worker_index_map[p_uid]
                    else:
                        logger.error("Working proc died without updating current work...")

        for proc in dead_procs:
            p_uid = worker_processes[proc]
            start_worker_proc(bound_worker_func, p_uid, worker_processes)

            proc.close()
            del worker_processes[proc]

            logger.warning(f"Started new worker. Worker count: %s", len(worker_processes))

        time.sleep(1)

    for proc in worker_processes:
        proc.terminate()

    time.sleep(0.25)

    for proc in worker_processes:
        proc.close()

    if len(exception_list) > 0 and raise_exceptions:
        for exception_result in exception_list:
            raise(exception_result.base_exception)

    results_list = list(results_list)

    return results_list
