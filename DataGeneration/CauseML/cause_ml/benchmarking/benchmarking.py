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

def _sample_data(dgp):
    # Sample data
    dataset = dgp.generate_data()
    return dataset

def _fit_and_apply_model(model_class, estimand, dataset):
    model = model_class(dataset)
    model.fit()
    estimate_val = model.estimate(estimand=estimand)
    true_val = dataset.ground_truth(estimand=estimand)

    return estimate_val, true_val

def run_benchmark(model_class, estimand,
                   data_source, param_grid,
                   num_dgp_samples=1,
                   num_data_samples_per_dgp=1,
                   enable_ray_multiprocessing=False,
                   metrics=ACCURACY_METRICS):

    # Configure multiprocessing
    if enable_ray_multiprocessing:
        if not ray.is_initialized():
            ray.init()

        sample_dgp = ray.remote(_sample_dgp).remote
        sample_data = ray.remote(_sample_data).remote
        fit_and_apply_model = ray.remote(_fit_and_apply_model).remote
    else:
        sample_dgp = _sample_dgp
        sample_data = _sample_data
        fit_and_apply_model = fit_and_apply_model

    results = defaultdict(list)

    # Iterate over all DGP sampler parameter configurations
    for param_spec in ParameterGrid(param_grid):

        # Construct the DGP sampler for these params.
        dgp_params = build_parameters_from_axis_levels(param_spec)
        dgp_sampler = DataGeneratingProcessSampler(
            parameters=dgp_params, data_source=data_source)

        # Sample DGPs
        async_sample_effect_data = []
        for _ in range(num_dgp_samples):

            # Sample data from the DGP
            dgp = sample_dgp(dgp_sampler)
            for _ in range(num_data_samples_per_dgp):
                dataset = sample_data(dgp)

                # Fit model and use to generate estimates.
                effect_data = fit_and_apply_model(
                    model_class, estimand, dataset)
                async_sample_effect_data.append(effect_data)

        # Process potentially async results.
        if enable_ray_multiprocessing:
            sample_effect_data = ray.get(async_sample_effect_data)
        else:
            sample_effect_data = async_sample_effect_data

        # Extract estimates and ground truth results.
        sample_effect_data = np.array(sample_effect_data)
        estimate_vals = sample_effect_data[:, 0]
        true_vals = sample_effect_data[:, 1]

        # Store the params for this run in the results dict
        for param_name, param_value in param_spec.items():
            results[f"param_{param_name.lower()}"].append(param_value)

        # Calculate and store the requested metric values.
        for metric_name, metric_func in metrics.items():
            results[metric_name].append(
                metric_func(estimate_vals, true_vals))

    return results
