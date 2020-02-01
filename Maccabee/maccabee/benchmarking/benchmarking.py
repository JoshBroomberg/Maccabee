"""This module contains the high-levels functions used to run benchmarks."""

from sklearn.model_selection import ParameterGrid
from collections import defaultdict
import numpy as np
from multiprocessing import Pool
from functools import partial

from ..parameters import build_parameters_from_axis_levels
from ..data_generation import DataGeneratingProcessSampler, SampledDataGeneratingProcess
from ..data_analysis import calculate_data_axis_metrics
from ..utilities.threading import single_threaded_context
from ..modeling.model_metrics import ATE_ACCURACY_METRICS, ITE_ACCURACY_METRICS
from ..exceptions import UnknownEstimandException
from ..constants import Constants

METRIC_ROUNDING = 3

def _aggregate_metric_results(metric_results, std=True):
    aggregated_results = {}
    for metric, results_list in metric_results.items():
        aggregated_results[metric] = np.round(
            np.mean(results_list), METRIC_ROUNDING)

        if std:
            aggregated_results[metric + " (std)"] = np.round(
                np.std(results_list), METRIC_ROUNDING)

    return aggregated_results

def _gen_data_and_apply_model(dgp, model_class, estimand, index):
    np.random.seed()
    dataset = dgp.generate_dataset()

    # Fit model
    model = model_class(dataset)
    model.fit()

    # Collect estimand result
    estimate_val = model.estimate(estimand=estimand)
    true_val = dataset.ground_truth(estimand=estimand)

    return index, (estimate_val, true_val), dataset

def _sample_dgp(dgp_sampler, index):
    np.random.seed()
    return index, dgp_sampler.sample_dgp()

def _get_performance_metric_data_structures(num_samples_from_dgp, n_observations, estimand):
    if estimand == Constants.Model.ATE_ESTIMAND:
        data_store_shape = (num_samples_from_dgp, 2)
    elif estimand == Constants.Model.ITE_ESTIMAND:
        data_store_shape = (num_samples_from_dgp, 2, n_observations)
    else:
        raise UnknownEstimandException()

    return data_store_shape

def _get_performance_metric_functions(estimand):
    if estimand == Constants.Model.ATE_ESTIMAND:
        perf_metric_names_and_funcs = ATE_ACCURACY_METRICS
    elif estimand == Constants.Model.ITE_ESTIMAND:
        perf_metric_names_and_funcs = ITE_ACCURACY_METRICS
    else:
        raise UnknownEstimandException()

    return perf_metric_names_and_funcs

def benchmark_model_using_concrete_dgp(
    dgp,
    model_class, estimand,
    num_runs, num_samples_from_dgp,
    data_analysis_mode=False,
    data_metrics_spec=None,
    n_jobs=1):

    # Set DGP data analysis mode
    if dgp.get_data_analysis_mode() != data_analysis_mode:
        print("WARNING: DGP analysis mode contradicts parameter data_analysis_mode.")
        print(f"WARNING: Setting DGP data_analysis_mode to {data_analysis_mode}")

    dgp.set_data_analysis_mode(data_analysis_mode)

    run_model_on_dgp = partial(_gen_data_and_apply_model, dgp, model_class, estimand)
    sample_indeces = range(num_samples_from_dgp)

    performance_metric_run_results = defaultdict(list)
    data_metric_run_results = defaultdict(list)

    data_store_shape = _get_performance_metric_data_structures(
        num_samples_from_dgp, dgp.n_observations, estimand)
    perf_metric_names_and_funcs = _get_performance_metric_functions(estimand)

    for run_index in range(num_runs):
        estimand_sample_results = np.empty(data_store_shape)
        data_analysis_sample_results = defaultdict(list)
        datasets = np.empty(num_samples_from_dgp, dtype="O")

        with single_threaded_context():
            with Pool(processes=min(n_jobs, num_samples_from_dgp)) as pool:
                for sample_index, effect_estimate_and_truth, dataset in pool.imap_unordered(
                    run_model_on_dgp, sample_indeces):

                    estimand_sample_results[sample_index, :] = effect_estimate_and_truth
                    datasets[sample_index] = dataset

                if data_analysis_mode:
                    collect_dataset_metrics = partial(
                        calculate_data_axis_metrics,
                        observation_spec=data_metrics_spec)

                    # Preserve order to associate results with datasets.
                    for _, data_metric_results in pool.map(
                        collect_dataset_metrics, datasets):

                        for axis_metric_name, axis_metric_val in data_metric_results.items():
                            data_analysis_sample_results[axis_metric_name].append(
                                axis_metric_val)

        # Process sample results into metric estimates
        estimate_vals = estimand_sample_results[:, 0]
        true_vals = estimand_sample_results[:, 1]

        for metric_name, metric_func in perf_metric_names_and_funcs.items():
            performance_metric_run_results[metric_name].append(metric_func(
                estimate_vals, true_vals))

        if data_analysis_mode:
            for axis_metric_name, vals in data_analysis_sample_results.items():
                data_metric_run_results[axis_metric_name].append(
                    np.mean(vals))

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
    dgp_class=SampledDataGeneratingProcess,
    dgp_kwargs={},
    n_jobs=1):

    perf_metric_names_and_funcs = _get_performance_metric_functions(estimand)

    dgp_sampler = DataGeneratingProcessSampler(
        dgp_class=dgp_class,
        parameters=dgp_sampling_params,
        data_source=data_source,
        dgp_kwargs=dgp_kwargs)

    dgp_sampler = partial(_sample_dgp, dgp_sampler)
    with Pool(processes=min(n_jobs, num_dgp_samples)) as pool:
        dgps = pool.map(dgp_sampler, range(num_dgp_samples))

    performance_metric_dgp_results = defaultdict(list)
    data_metric_dgp_results = defaultdict(list)
    for _, dgp in dgps:
        performance_metric_data, _, data_metric_data, _ = \
            benchmark_model_using_concrete_dgp(
                dgp, model_class, estimand,
                num_runs=num_sampling_runs_per_dgp,
                num_samples_from_dgp=num_samples_from_dgp,
                data_analysis_mode=data_analysis_mode,
                data_metrics_spec=data_metrics_spec,
                n_jobs=n_jobs)

        for metric_name in perf_metric_names_and_funcs:
            performance_metric_dgp_results[metric_name].append(
                performance_metric_data[metric_name])

        if data_analysis_mode:
            for axis_metric_name, val in data_metric_data.items():
                data_metric_dgp_results[axis_metric_name].append(val)

    return (_aggregate_metric_results(performance_metric_dgp_results),
        performance_metric_dgp_results,
        _aggregate_metric_results(data_metric_dgp_results, std=False),
        data_metric_dgp_results)


def benchmark_model_using_sampled_dgp_grid(
    dgp_param_grid, data_source,
    model_class, estimand,
    num_dgp_samples, num_samples_from_dgp,
    data_analysis_mode=False,
    data_metrics_spec=None,
    param_overrides={},
    dgp_class=SampledDataGeneratingProcess,
    dgp_kwargs={}):

    metric_param_results = defaultdict(list)

    # Iterate over all DGP sampler parameter configurations
    for param_spec in ParameterGrid(dgp_param_grid):

        # Construct the DGP sampler for these params.
        dgp_params = build_parameters_from_axis_levels(param_spec)

        for param_name in param_overrides:
            dgp_params.set_parameter(param_name, param_overrides[param_name])

        param_performance_metric_data, _, param_data_metric_data, _ = \
            benchmark_model_using_sampled_dgp(
                param_spec, data_source,
                model_class, estimand,
                num_dgp_samples, num_samples_from_dgp,
                data_analysis_mode=data_analysis_mode,
                data_metrics_spec=data_metrics_spec,
                dgp_class=dgp_class,
                dgp_kwargs=dgp_kwargs)

        # Store the params for this run in the results dict
        for param_name, param_value in param_spec.items():
            results[f"param_{param_name.lower()}"].append(param_value)

        # Calculate and store the requested metric values.
        for metric_name, metric_result in param_performance_metric_data.items():
            metric_param_results[metric_name].append(metric_result)

        if data_analysis_mode:
            for metric_name, metric_result in param_data_metric_data.items():
                metric_param_results[metric_name].append(metric_result)

    return metric_param_results
