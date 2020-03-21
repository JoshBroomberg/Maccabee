"""This submodule consists of a series of three independent but functionally 'nested' benchmarking functions. Each function can be used on its own but is also used by the functions higher up the nesting hierarchy.

* :func:`~maccabee.benchmarking.benchmarking.benchmark_model_using_concrete_dgp` is the basic benchmarking function. It takes a single :class:`~maccabee.data_generation.data_generating_process.DataGeneratingProcess` instance and evaluates a estimator using data sets sampled from the :term:`DGP` represented by the instance. The collected metrics are aggregated at two levels: the metric values for multiple samples are averaged and the resultant average metric values are then averaged across sampling runs. This two-level aggregation allows for the calculation of a standard deviation for the :term:`performance metrics <performance metric>` that are defined over multiple sample estimand values. For example - the absolute bias is found by averaging the signed error in the estimate over many trials. The standard deviation of the bias estimate thus requires multiple sampling runs each producing a single bias measure.

|

* :func:`~maccabee.benchmarking.benchmarking.benchmark_model_using_sampled_dgp` is the next function up the benchmarking hierarchy. It takes :term:`DGP` sampling parameters in the form of a :class:`~maccabee.parameters.parameter_store.ParameterStore` instance and a :class:`~maccabee.data_sources.data_sources.DataSource` instance. It then samples DGPs based on the sampling parameters and uses :func:`~maccabee.benchmarking.benchmarkingbenchmark_model_using_concrete_dgp` to collect metrics for each sampled DGP. This introduces a third aggregation level, with metrics (and associated standard deviations) being averaged over a number of sampled DGPs.

|

* :func:`~maccabee.benchmarking.benchmarking.benchmark_model_using_sampled_dgp_grid` is the next and final function up the benchmarking hierarchy. It takes a grid of sampling parameters corresponding to different levels of one of more data axes and then samples DGPs from each combination of sampling parameters in the grid using :func:`~maccabee.benchmarking.benchmarking.benchmark_model_using_sampled_dgp`. There is no additional aggregation as the metrics for each parameter combination are reported individually.
"""

from sklearn.model_selection import ParameterGrid
from collections import defaultdict
import numpy as np
from multiprocessing import Pool, Process, Manager, cpu_count
from functools import partial

from ..parameters import build_parameters_from_axis_levels
from ..data_generation import DataGeneratingProcessSampler, SampledDataGeneratingProcess
from ..data_analysis import calculate_data_axis_metrics
from ..modeling.performance_metrics import AVG_EFFECT_METRICS, INDIVIDUAL_EFFECT_METRICS
from ..exceptions import UnknownEstimandException, UnknownEstimandAggregationException
from ..constants import Constants

from ..utilities.threading import get_threading_context
from ..utilities.multiprocessing import robust_parallel_map

from ..logging import get_logger
logger = get_logger(__name__)

METRIC_ROUNDING = 3

def _aggregate_metric_results(metric_results, std=True):
    """Helper method used to calculate aggregate metric statistics (mean and standard deviation) for multiple supplied metrics, each with an arbitrary number of individual results.

    Args:
        metric_results (dict): A dictionary with metric names as keys and individual metric result lists as values.
        std (bool): Boolean indicating whether to calculate standard deviations. Defaults to True.

    Returns:
        dict: A dictionary with the metric names as keys and mean metric value as values. Standard deviations are included as keys made up of the metric name with an "(std)" appended.
    """
    aggregated_results = {}
    for metric, results_list in metric_results.items():
        aggregated_results[metric] = np.round(
            np.mean(results_list), METRIC_ROUNDING)

        if std:
            aggregated_results[metric + " (std)"] = np.round(
                np.std(results_list), METRIC_ROUNDING)

    return aggregated_results

def _gen_data_and_apply_model(dgp, model_class, estimand, index):
    """Helper method used execute the set of operations required to benchmark a single DGP. This set is as follows:

    * Sample a data set
    * Apply a causal model to the data to estimate a desired causal estimand
    * Calculate the true value of the estimand from the data

    Args:
        dgp (:class:`~maccabee.data_generation.data_generating_process.DataGeneratingProcess`): a DGP instance.
        model_class (:class:`~maccabee.modeling.models.CausalModel`): a class definition that inherits from :class:`~maccabee.modeling.models.CausalModel`, implementing a causal estimator.
        estimand (str): the string name of a causal estimand.
        index (int): An index associated with the DGP that is returned with the results of this function. This is for the convenience of calling functions that may execute this method in parallel.

    Returns:
        tuple: a tuple with the index as the first entry, the estimated and true causal effects as a tuple in the second entry and the generated data (from the DGP) associated with the causal effects as the third entry.
    """

    # TODO-FUTURE: consider implementing a mechanism to control this seeding.
    # This is complex given the parallelism requires each worker to have a
    # different source of entropy.
    np.random.seed() # randomly seed the data sampling process.

    logger.info(f"Generating data set {index+1}")
    dataset = dgp.generate_dataset()

    logger.debug(f"Fitting causal model to data set {index+1}")
    # Fit model
    model = model_class(dataset)
    model.fit()

    logger.debug(f"Collecting model estimand for data set {index+1}")
    # Collect estimand result
    estimate_val = model.estimate(estimand=estimand)

    logger.debug(f"Collecting ground truth effect from data set {index+1}")
    true_val = dataset.ground_truth(estimand=estimand)

    return index, (estimate_val, true_val), dataset

def _sample_dgp(dgp_sampler, index):
    """

    TODO fix these docs

    Helper method to sample a :class:`~maccabee.data_generation.data_generating_process.DataGeneratingProcess` instance from a :class:`~maccabee.data_generation.data_generating_process_sampler.DataGeneratingProcessSampler` instance. Optionally storing this in a supplied list-like data structure.

    Args:
        dgp_sampler (:class:`~maccabee.data_generation.data_generating_process_sampler.DataGeneratingProcessSampler`): a :class:`~maccabee.data_generation.data_generating_process_sampler.DataGeneratingProcessSampler` instance from which to sample DGPs.
        index (int): index at which to store the sampled DGP in the supplied data structure.
        dgps (list): a list or integer-indexable data structure in which to store the sampled DGP.

    Returns:
        tuple: a tuple with the index as the first entry and the sampled DGP as the second entry.
    """
    # TODO-FUTURE: consider implementing a mechanism to control this seeding.
    # This is complex given the parallelism requires each worker to have a
    # different source of entropy.
    np.random.seed() # seed the dgp sampling process
    logger.info(f"Sampling DGP {index+1}")
    sampled_dgp = dgp_sampler.sample_dgp()
    return sampled_dgp

def _get_performance_metric_data_structures(num_samples_from_dgp, n_observations, estimand):
    """Helper method to generate data structures to store performance metric data.

    Args:
        num_samples_from_dgp (int): the number of data sets that will be sampled from the DGP.
        n_observations (int): the number of observations in each data set.
        estimand (str): the name of the estimand being used for benchmarking. This influences the dimensionality of the data to be stored for metric calculation.

    Returns:
        tuple: a tuple describing the dimensions of the data structure requiring to store the performance metrics at the lowest level of aggregation.
    """
    if estimand not in Constants.Model.ALL_ESTIMANDS:
        raise UnknownEstimandException()

    if estimand == Constants.Model.ITE_ESTIMAND:
        data_store_shape = (num_samples_from_dgp, 2, n_observations)
    else:
        data_store_shape = (num_samples_from_dgp, 2)

    return data_store_shape

def _get_performance_metric_functions(estimand):
    """Helper method that returns the performance metric functions for a specified estimand.

    Args:
        estimand (str): the name of the estimand being used for benchmarking.

    Returns:
        dict: A dictionary with performance metric names as keys and functions as values.
    """
    if estimand not in Constants.Model.ALL_ESTIMANDS:
        raise UnknownEstimandException()

    if estimand in Constants.Model.INDIVIDUAL_ESTIMANDS:
        perf_metric_names_and_funcs = INDIVIDUAL_EFFECT_METRICS
    elif estimand in Constants.Model.AVERAGE_ESTIMANDS:
        perf_metric_names_and_funcs = AVG_EFFECT_METRICS
    else:
        raise UnknownEstimandAggregationException()

    return perf_metric_names_and_funcs

def benchmark_model_using_concrete_dgp(
    dgp,
    model_class, estimand,
    num_sampling_runs_per_dgp, num_samples_from_dgp,
    data_analysis_mode=False,
    data_metrics_spec=None,
    n_jobs=1):
    """Sample data sets from the given DGP instance and calculate performance and (optionally) data metrics.

    Args:
        dgp (:class:`~maccabee.data_generation.data_generating_process.DataGeneratingProcess`): A DGP instance produced by a sampling procedure or through a concrete definition.
        model_class (:class:`~maccabee.modeling.models.CausalModel`): A model instance defined by subclassing the base :class:`~maccabee.modeling.models.CausalModel` or using one of the included model types.
        estimand (string): A string describing the estimand. The class :class:`maccabee.constants.Constants.Model` contains constants which can be used to specify the allowed estimands.
        num_sampling_runs_per_dgp (int): The number of sampling runs to perform. Each run is comprised of `num_samples_from_dgp` data set samples which are passed to the metric functions.
        num_samples_from_dgp (int): The number of data sets sampled from the DGP per sampling run.
        data_analysis_mode (bool): If ``True``, data metrics are calculated according to the supplied `data_metrics_spec`. This can be slow and may be unecessary. Defaults to True.
        data_metrics_spec (type): A dictionary which specifies which :term:`data metrics <data metric>` to calculate and record. The keys are axis names and the values are lists of string metric names. All axis names and the metrics for each axis are available in the dictionary :obj:`maccabee.data_analysis.data_metrics.AXES_AND_METRIC_NAMES`. If None, all data metrics are calculated. Defaults to None.
        n_jobs (int): The number of processes on which to run the benchmark. Defaults to 1.

    Returns:
        tuple: A tuple with four entries. The first entry is a dictionary of aggregated performance metrics mapping names to numerical results aggregated across runs. The second entry is a dictionary of raw performance metrics mapping metric names to lists of numerical metric values from each run (averaged only across the samples in the run). This is useful for understanding the metric value distribution. The third and fourth entries are analogous dictionaries which contain the data metrics. They are empty dicts if `data_analysis_mode` is ``False``.

    Raises:
        UnknownEstimandException: If an unknown estimand is supplied.
    """

    # Set DGP data analysis mode
    dgp.set_data_analysis_mode(data_analysis_mode)

    # Build a runner function which samples a data set from the dgp,
    # builds the model and finds the estimand value.
    run_model_on_dgp = partial(_gen_data_and_apply_model, dgp, model_class, estimand)

    # Build a runner function which calculates the data metrics
    # for a dataset given the data_metrics_spec
    collect_dataset_metrics = partial(
        calculate_data_axis_metrics,
        observation_spec=data_metrics_spec,
        flatten_result=True)

    sample_indeces = range(num_samples_from_dgp)

    # Results data structures which store the metric values
    # for each sampling run, aggregated across the samples in the run.
    performance_metric_run_results = defaultdict(list)
    data_metric_run_results = defaultdict(list)

    # The within-run performance metric functions and data structures are different
    # for each estimand. These helper functions provide the state required
    # to adjust for each estimand.
    perf_metric_data_store_shape = _get_performance_metric_data_structures(
        num_samples_from_dgp, dgp.n_observations, estimand)
    perf_metric_names_and_funcs = _get_performance_metric_functions(estimand)

    if n_jobs == -1:
        n_jobs = cpu_count()

    # Only use up to the max parallelism allowed by num_samples_from_dgp.
    n_jobs = min(n_jobs, num_samples_from_dgp)

    if n_jobs >= 1:
        logger.info(f"Running concrete DGP benchmark using a multiprocessing pool with {n_jobs} workers")
        # Build a multiprocessing pool based on the desired parallelism.
        # I use the built-in pool here over the custom, robust implementation
        # used below in order to make use of the advanced chunksize handling.
        pool = Pool(processes=n_jobs) # , maxtasksperchild=1 todo remove
        map_func = partial(pool.map, chunksize=max(1, int(num_samples_from_dgp/n_jobs)))
    elif n_jobs == 0:
        logger.info("Running concrete DGP benchmark using a single process.")
        map_func = map
    else:
        raise ValueError("Invalid n_jobs value - should be integer from -1 to n")


    # Sampling occurs in a single threaded context so that, when split
    # across cores, the numpy functions in each process don't use
    # more than the resources allocated to the process.
    thread_context = get_threading_context(1)
    with thread_context():

        # Synchronous loop over the sampling runs.
        for run_index in range(num_sampling_runs_per_dgp):
            logger.debug(f"Starting sampling run {run_index+1}")
            # Data structures to store the datasets, sampled estimand values and
            # data metrics for each sample in this sampling run.

            datasets = np.empty(num_samples_from_dgp, dtype="O")
            estimand_sample_results = np.empty(perf_metric_data_store_shape)

            # Data metric collection is not on the key performance pathway.
            # Use dynamic data structure for convenience and readability.
            data_metrics_sample_results = defaultdict(list)

            # Begin sampling.

            # Use the runner function and multiprocessing pool to draw
            # and process data samples into estimand samples.

            logger.debug(f"Starting sampling for run {run_index+1}.")
            for sample_index, effect_estimate_and_truth, dataset in map_func(
                run_model_on_dgp, sample_indeces):

                # Store estimand and data set samples.
                estimand_sample_results[sample_index, :] = effect_estimate_and_truth
                datasets[sample_index] = dataset
            logger.debug(f"Done sampling for run {run_index+1}.")

            # If in data analysis mode, use the pool to run data metric
            # calculation.
            if data_analysis_mode:
                logger.debug(f"Starting data analysis for run {run_index+1}.")
                # Loop over generated datasets in order.
                for data_metric_results in map_func(collect_dataset_metrics, datasets):
                    # Record all metrics at the sampled data set level
                    # for later aggregation.
                    for axis_metric_name, axis_metric_val in data_metric_results.items():
                        data_metrics_sample_results[axis_metric_name].append(
                            axis_metric_val)
                logger.debug(f"Done data analysis for run {run_index+1}.")

            # At the end of the sampling for this sampling run, process sample
            # estimand results into metric estimates.
            estimate_vals = estimand_sample_results[:, 0]
            true_vals = estimand_sample_results[:, 1]

            logger.debug(f"Performing DGP aggregate perf metric collection.")
            for metric_name, metric_func in perf_metric_names_and_funcs.items():
                performance_metric_run_results[metric_name].append(metric_func(
                    estimate_vals, true_vals))

            # Aggregate the data metrics by averaging across samples so that
            # there is a single real value per sampling run as with the perf
            # metrics.
            if data_analysis_mode:
                logger.debug(f"Performing DGP aggregate data metric collection.")
                for axis_metric_name, vals in data_metrics_sample_results.items():
                    data_metric_run_results[axis_metric_name].append(
                        np.mean(vals))

    if n_jobs >= 1:
        pool.close()
        pool.join()

    # Aggregate perf and data metrics across sampling runs.
    performance_metric_aggregated_results = _aggregate_metric_results(performance_metric_run_results)
    data_metric_aggregated_results = _aggregate_metric_results(data_metric_run_results, std=False)

    return (performance_metric_aggregated_results,
        performance_metric_run_results,
        data_metric_aggregated_results,
        data_metric_run_results)

def benchmark_model_using_sampled_dgp(
    dgp_sampling_params, data_source,
    model_class, estimand,
    num_dgp_samples,
    num_samples_from_dgp,
    num_sampling_runs_per_dgp=1,
    data_analysis_mode=False,
    data_metrics_spec=None,
    data_metric_intervals=False,
    dgp_class=SampledDataGeneratingProcess,
    dgp_kwargs={},
    n_jobs=1,
    compile_functions=False):
    """Short summary.

    Args:
        dgp_sampling_params (:class:`~maccabee.parameters.parameter_store.ParameterStore`): A :class:`~maccabee.parameters.parameter_store.ParameterStore` instance which contains the DGP sampling parameters which will be used when sampling DGPs.
        data_source (:class:`~maccabee.data_sources.data_sources.DataSource`): a :class:`~maccabee.data_sources.data_sources.DataSource` instance which will be used as the source of covariates for sampled DGPs.
        model_class (:class:`~maccabee.modeling.models.CausalModel`): A model instance defined by subclassing the base :class:`~maccabee.modeling.models.CausalModel` or using one of the included model types.
        estimand (string): A string describing the estimand. The class :class:`maccabee.constants.Constants.Model` contains constants which can be used to specify the allowed estimands.
        num_dgp_samples (int): The number of DGPs to sample. Each sampled DGP is benchmarked using :func:`~maccabee.benchmarking.benchmarking.benchmark_model_using_concrete_dgp`.
        num_samples_from_dgp (int): See :func:`~maccabee.benchmarking.benchmarking.benchmark_model_using_concrete_dgp`.
        num_sampling_runs_per_dgp (int): See :func:`~maccabee.benchmarking.benchmarking.benchmark_model_using_concrete_dgp`. Defaults to 1.
        data_analysis_mode (bool): See :func:`~maccabee.benchmarking.benchmarking.benchmark_model_using_concrete_dgp`. Defaults to False.
        data_metrics_spec (dict):  See :func:`~maccabee.benchmarking.benchmarking.benchmark_model_using_concrete_dgp`. Defaults to None.
        dgp_class (:class:`maccabee.data_generation.data_generating_process.SampledDataGeneratingProcess`): The DGP class to instantiate after function sampling. This must be a subclass of :class:`maccabee.data_generation.data_generating_process.SampledDataGeneratingProcess`. This can be used to tweak aspects of the default sampled DGP. Defaults to SampledDataGeneratingProcess.
        dgp_kwargs (dict): A dictionary of keyword arguments to pass to the sampled DGPs at instantion time. Defaults to {}.
        n_jobs (int): See :func:`~maccabee.benchmarking.benchmarking.benchmark_model_using_concrete_dgp`. Defaults to 1.
        compile_functions (bool): A boolean indicating whether sampling DGP functions should be compiled prior to execution. Defaults to ``False``.

    Returns:
        tuple: A tuple with four entries. See :func:`~maccabee.benchmarking.benchmarking.benchmark_model_using_concrete_dgp` for a description of the entries but note that, in this func, the aggregate metric values are averaged across dgp samples and sampling runs and the raw metric values correspond to averages over sampling runs for each sampled DGP. This means each entry in the raw metrics list corresponds to the aggregated result of the :func:`~maccabee.benchmarking.benchmarking.benchmark_model_using_concrete_dgp` function.

    Raises:
        UnknownEstimandException: If an unknown estimand is supplied.
    """

    perf_metric_names_and_funcs = _get_performance_metric_functions(estimand)

    dgp_kwargs["compile_functions"] = compile_functions

    dgp_sampler = DataGeneratingProcessSampler(
        dgp_class=dgp_class,
        parameters=dgp_sampling_params,
        data_source=data_source,
        dgp_kwargs=dgp_kwargs)

    dgp_sampler = partial(_sample_dgp, dgp_sampler)

    # Build a multiprocessing pool which exploits the parallelism in the DGP
    # sampling. Use it to sample all DGPs.

    # TODO-MULTIPROC: refactor this multiprocessing approach.
    # Attempt to move to the compile expression itself
    if n_jobs == -1:
        n_jobs = cpu_count()

    logger.info(f"Sampling DGPs using {n_jobs} processes")

    dgps = robust_parallel_map(
        dgp_sampler,
        range(num_dgp_samples),
        n_jobs=n_jobs)

    # Data structures for storing the metric results for each sampled DGP.
    performance_metric_dgp_results = defaultdict(list)
    performance_metric_raw_run_results = defaultdict(list)
    data_metric_dgp_results = defaultdict(list)

    # Build benchmark executable. Note that parallelism is turned off
    # in this executable. Parallelism is at the DGP level.
    benchmark_dgp = partial(
        benchmark_model_using_concrete_dgp,
        model_class=model_class,
        estimand=estimand,
        num_sampling_runs_per_dgp=num_sampling_runs_per_dgp,
        num_samples_from_dgp=num_samples_from_dgp,
        data_analysis_mode=data_analysis_mode,
        data_metrics_spec=data_metrics_spec,
        n_jobs=0)

    n_benchmark_workers = min(n_jobs, num_dgp_samples)

    logger.info(f"Starting benchmarking with sampled DGPs using {n_benchmark_workers} workers.")
    results_data = robust_parallel_map(
        benchmark_dgp,
        dgps,
        n_jobs=n_jobs)

    # TODO remove
    # with Pool(processes=n_benchmark_workers, maxtasksperchild=1) as pool:
    # for res_data in pool.imap_unordered(benchmark_dgp, dgps):
    for i, res_data in enumerate(results_data):
        logger.debug(f"Done data and metric sampling for DGP {i+1}/{num_dgp_samples}")

        performance_metric_data, performance_raw_data, data_metric_data, _ = res_data

        # Extract and store the aggregated perf metric results (across
        # all the sampling runs). This loop excludes the standard deviation
        # from being collected at this stage. It is calculated over the
        # sampled dgp results.
        for metric_name in perf_metric_names_and_funcs:
            performance_metric_dgp_results[metric_name].append(
                performance_metric_data[metric_name])
            performance_metric_raw_run_results[metric_name].append(
                performance_raw_data[metric_name])

        logger.debug(f"Done aggregate perf metric collection for DGP {i+1}/{num_dgp_samples}")

        # As above, but for the data metrics which don't have a standard dev.
        if data_analysis_mode:
            for axis_metric_name, val in data_metric_data.items():
                data_metric_dgp_results[axis_metric_name].append(val)
            logger.debug(f"Done aggregate data metric collection for DGP {i+1}/{num_dgp_samples}")

    logger.info("Done benchmarking with sampled DGPs.")

    return (_aggregate_metric_results(performance_metric_dgp_results),
        performance_metric_dgp_results, performance_metric_raw_run_results,
        _aggregate_metric_results(data_metric_dgp_results, std=data_metric_intervals),
        data_metric_dgp_results, dgps)


def benchmark_model_using_sampled_dgp_grid(
    dgp_param_grid, data_source,
    model_class, estimand,
    num_dgp_samples,
    num_samples_from_dgp,
    num_sampling_runs_per_dgp=1,
    data_analysis_mode=False,
    data_metrics_spec=None,
    data_metric_intervals=True,
    param_overrides={},
    dgp_class=SampledDataGeneratingProcess,
    dgp_kwargs={},
    n_jobs=1,
    compile_functions=False):
    """This function is a thin wrapper around the :func:`~maccabee.benchmarking.benchmarking.benchmark_model_using_sampled_dgp` function. It is used to run the sampeld DGP benchmark across many different sampling parameter value combinations. The signature is the same as the wrapped function with `dgp_sampling_params` replaced by `dgp_param_grid` and the new `param_overrides` option. For all other arguments, see :func:`~maccabee.benchmarking.benchmarking.benchmark_model_using_sampled_dgp`.

    Args:
        dgp_param_grid (dict): A dictionary mapping :term:`data axis <distributional problem space axis>` names to a list of data axis levels. Axis names are available as constants in :class:`maccabee.constants.Constants.AxisNames` and axis levels available as constants in :class:`maccabee.constants.Constants.AxisLevels`. The :func:`~maccabee.benchmarking.benchmarking.benchmark_model_using_sampled_dgp` function is called for each combination of axis level values - the cartesian product of the lists in the dictionary.
        param_overrides (dict): A dictionary mapping parameter names to values of those parameters. The values in this dict override the values in the grid and any default parameter values. For all available parameter names and allowed values, see the :download:`parameter_schema.yml </../../maccabee/parameters/parameter_schema.yml>` file.

    Returns:
        :class:`~pandas.DataFrame`: A :class:`~pandas.DataFrame` containing one row per axis level combination and a column for each axis and each performance and data metric (as well as their standard deviations).
    """

    metric_param_results = defaultdict(list)


    # Iterate over all DGP sampler parameter configurations
    for param_spec in ParameterGrid(dgp_param_grid):
        # Construct the DGP sampler for these params.
        dgp_params = build_parameters_from_axis_levels(param_spec)

        # Apply overrides.
        for param_name in param_overrides:
            dgp_params.set_parameter(param_name, param_overrides[param_name])

        # Run sampling benchmark.
        logger.info(f"Running benchmarking with params {param_spec} and {n_jobs} workers.")
        param_performance_metric_data, _, _, param_data_metric_data, _, dgps = \
            benchmark_model_using_sampled_dgp(
                dgp_sampling_params=dgp_params,
                data_source=data_source,
                model_class=model_class,
                estimand=estimand,
                num_dgp_samples=num_dgp_samples,
                num_sampling_runs_per_dgp=num_sampling_runs_per_dgp,
                num_samples_from_dgp=num_samples_from_dgp,
                data_analysis_mode=data_analysis_mode,
                data_metrics_spec=data_metrics_spec,
                data_metric_intervals=data_metric_intervals,
                dgp_class=dgp_class,
                dgp_kwargs=dgp_kwargs,
                n_jobs=n_jobs,
                compile_functions=compile_functions)


        # Store the params for this run in the results dict
        for param_name, param_value in param_spec.items():
            metric_param_results[f"param_{param_name.lower()}"].append(param_value)

        # Calculate and store the requested metric values.
        for metric_name, metric_result in param_performance_metric_data.items():
            metric_param_results[metric_name].append(metric_result)

        if data_analysis_mode:
            for metric_name, metric_result in param_data_metric_data.items():
                metric_param_results[metric_name].append(metric_result)

    return metric_param_results
