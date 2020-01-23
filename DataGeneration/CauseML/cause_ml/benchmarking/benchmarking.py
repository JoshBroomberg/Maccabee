from sklearn.model_selection import ParameterGrid
import ray
from collections import defaultdict
from cause_ml.parameters import build_parameters_from_axis_levels
from cause_ml.data_generation import DataGeneratingProcessSampler
import numpy as np

ACCURACY_METRICS = {
    "absolute mean bias": lambda estimate_vals, true_vals: np.abs(
        np.mean(estimate_vals - true_vals)),
    "root mean squared error": lambda estimate_vals, true_vals: np.sqrt(
        np.mean((estimate_vals - true_vals)**2))
}

def _sample_dgp(dgp_sampler):
    return dgp_sampler.sample_dgp()

_sample_dgp_remote = ray.remote(_sample_dgp).remote

def _sample_data(dgp):
    # Sample data
    dataset = dgp.generate_dataset()
    return dataset

_sample_data_remote = ray.remote(_sample_data).remote

def _fit_and_apply_model(model_class, estimand, dataset):
    model = model_class(dataset)
    model.fit()
    estimate_val = model.estimate(estimand=estimand)
    true_val = dataset.ground_truth(estimand=estimand)

    return estimate_val, true_val

_fit_and_apply_model_remote = ray.remote(_fit_and_apply_model).remote

def _process_effect_data(sample_effect_data, ray_enabled):
    # Process potentially async results.
    if ray_enabled:
        sample_effect_data = ray.get(sample_effect_data)

    # Extract estimane and ground truth
    sample_effect_data = np.array(sample_effect_data)
    estimate_vals = sample_effect_data[:, 0]
    true_vals = sample_effect_data[:, 1]

    return estimate_vals, true_vals

def run_concrete_dgp_benchmark(
    dgp, model_class, estimand, num_samples_from_dgp,
    enable_ray_multiprocessing=False,
    process_results=True):

    # Configure multiprocessing
    if enable_ray_multiprocessing:
        if not ray.is_initialized():
            ray.init()

        sample_data = _sample_data_remote
        fit_and_apply_model = _fit_and_apply_model_remote
    else:
        sample_data = _sample_data
        fit_and_apply_model = _fit_and_apply_model

    sample_effect_data = []
    for _ in range(num_samples_from_dgp):
        dataset = sample_data(dgp)

        # Fit model and use to generate estimates.
        effect_data = fit_and_apply_model(
            model_class, estimand, dataset)
        sample_effect_data.append(effect_data)

    if process_results:
        # Extract estimates and ground truth results.
        estimate_vals, true_vals = _process_effect_data(
            sample_effect_data, ray_enabled=enable_ray_multiprocessing)

        results = {}
        for metric_name, metric_func in ACCURACY_METRICS.items():
            results[metric_name] = metric_func(estimate_vals, true_vals)

        return results

    return sample_effect_data

def run_sampled_dgp_benchmark(
    model_class, estimand,
    data_source, param_grid,
    num_dgp_samples=1,
    num_data_samples_per_dgp=1,
    dgp_kwargs={},
    enable_ray_multiprocessing=False):

    # Configure multiprocessing
    if enable_ray_multiprocessing:
        if not ray.is_initialized():
            ray.init()

        sample_dgp = _sample_dgp_remote
    else:
        sample_dgp = _sample_dgp

    results = defaultdict(list)

    # Iterate over all DGP sampler parameter configurations
    for param_spec in ParameterGrid(param_grid):

        # Construct the DGP sampler for these params.
        dgp_params = build_parameters_from_axis_levels(param_spec)
        dgp_sampler = DataGeneratingProcessSampler(
            parameters=dgp_params,
            data_source=data_source,
            dgp_kwargs=dgp_kwargs)

        # Sample DGPs
        sample_effect_data = []
        for _ in range(num_dgp_samples):

            # Sample data from the DGP
            dgp = sample_dgp(dgp_sampler)
            dgp_sample_effect_data = run_concrete_dgp_benchmark(
                dgp, model_class, estimand,
                num_samples_from_dgp=num_data_samples_per_dgp,
                enable_ray_multiprocessing=enable_ray_multiprocessing,
                process_results=False)

            sample_effect_data.extend(dgp_sample_effect_data)

        # Store the params for this run in the results dict
        for param_name, param_value in param_spec.items():
            results[f"param_{param_name.lower()}"].append(param_value)

        # Extract estimates and ground truth results.
        estimate_vals, true_vals = _process_effect_data(sample_effect_data,
            ray_enabled=enable_ray_multiprocessing)

        # Calculate and store the requested metric values.
        for metric_name, metric_func in ACCURACY_METRICS.items():
            results[metric_name].append(
                metric_func(estimate_vals, true_vals))

    return results
