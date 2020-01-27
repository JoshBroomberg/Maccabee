from sklearn.model_selection import ParameterGrid
from collections import defaultdict
from ..parameters import build_parameters_from_axis_levels
from ..data_generation import DataGeneratingProcessSampler, SampledDataGeneratingProcess
import numpy as np


def _absolute_mean_error(estimate_vals, true_vals):
    non_zeros = np.logical_not(np.isclose(true_vals, 0))
    return 100*np.abs(
        np.mean(
            (estimate_vals[non_zeros] - true_vals[non_zeros])/true_vals[non_zeros]))

ACCURACY_METRICS = {
    "absolute mean bias %": _absolute_mean_error,
    "root mean squared error": lambda estimate_vals, true_vals: np.sqrt(
        np.mean((estimate_vals - true_vals)**2))
}

METRIC_ROUNDING = 3

def _apply_model_to_dataset(model_class, estimand, dataset):
    model = model_class(dataset)
    model.fit()
    estimate_val = model.estimate(estimand=estimand)
    true_val = dataset.ground_truth(estimand=estimand)

    return estimate_val, true_val

def _aggregate_metric_results(metric_results):
    aggregated_results = {}
    for metric, results_list in metric_results.items():
        aggregated_results[metric] = np.round(
            np.mean(results_list), METRIC_ROUNDING)
        aggregated_results[metric + " (std)"] = np.round(
            np.std(results_list), METRIC_ROUNDING)

    return aggregated_results

def benchmark_model_using_concrete_dgp(
    dgp,
    model_class, estimand,
    num_runs, num_samples_from_dgp):

    metric_run_results = defaultdict(list)
    for run_index in range(num_runs):

        estimand_sample_results = []
        for sample_index in range(num_samples_from_dgp):
            dataset = dgp.generate_dataset()

            # Fit model and generate estimates + ground truth.
            effect_data = _apply_model_to_dataset(
                model_class, estimand, dataset)

            estimand_sample_results.append(effect_data)

        # Process sample results into metric estimates
        sample_effect_data = np.array(estimand_sample_results)
        estimate_vals = sample_effect_data[:, 0]
        true_vals = sample_effect_data[:, 1]

        for metric_name, metric_func in ACCURACY_METRICS.items():
            metric_run_results[metric_name].append(metric_func(
                estimate_vals, true_vals))

    metric_aggregated_results = _aggregate_metric_results(metric_run_results)

    return metric_aggregated_results, metric_run_results

def benchmark_model_using_sampled_dgp(
    dgp_sampling_params, data_source,
    model_class, estimand,
    num_dgp_samples, num_samples_from_dgp,
    dgp_class=SampledDataGeneratingProcess,
    dgp_kwargs={}):

    dgp_sampler = DataGeneratingProcessSampler(
        dgp_class=dgp_class,
        parameters=dgp_sampling_params,
        data_source=data_source,
        dgp_kwargs=dgp_kwargs)

    metric_dgp_results = defaultdict(list)
    for dgp_index in range(num_dgp_samples):

        # Sample DGPs
        dgp = dgp_sampler.sample_dgp()

        dgp_metric_data, _ = benchmark_model_using_concrete_dgp(
            dgp, model_class, estimand,
            num_runs=1,
            num_samples_from_dgp=num_samples_from_dgp)

        for metric_name in ACCURACY_METRICS:
            metric_dgp_results[metric_name].append(dgp_metric_data[metric_name])

    return _aggregate_metric_results(metric_dgp_results), metric_dgp_results

def benchmark_model_using_sampled_dgp_grid(
    dgp_param_grid, data_source,
    model_class, estimand,
    num_dgp_samples, num_samples_from_dgp,
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

        param_metric_data, _ = benchmark_model_using_sampled_dgp(
            param_spec, data_source,
            model_class, estimand,
            num_dgp_samples, num_samples_from_dgp,
            dgp_class=dgp_class,
            dgp_kwargs=dgp_kwargs)

        # Store the params for this run in the results dict
        for param_name, param_value in param_spec.items():
            results[f"param_{param_name.lower()}"].append(param_value)

        # Calculate and store the requested metric values.
        for metric_name, metric_result in param_metric_data.items():
            metric_param_results[metric_name].append(metric_result)

    return metric_param_results
