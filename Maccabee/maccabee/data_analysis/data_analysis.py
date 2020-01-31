"""This module contains the high-level functions used to analyze observational datasets to determine the distributional setting that they represent."""

import numpy as np
import matplotlib.pyplot as plt

from .data_metrics import AXES_AND_METRICS, AXIS_METRIC_FUNCTIONS, AXIS_METRIC_FUNC_INPUT_ACCESSORS


def _get_metric_func_arg_value(arg_val_name, dataset):
    if arg_val_name in AXIS_METRIC_FUNC_INPUT_ACCESSORS:
        return AXIS_METRIC_FUNC_INPUT_ACCESSORS[arg_val_name](dataset)
    else:
        return arg_val_name

def calculate_data_axis_metrics(dataset, observation_spec=None):
    '''
    Given a set of generated data, calculate all metrics
    in the observation spec.
    '''

    axis_metric_results = {}
    axis_metric_results_flat = {}

    for axis, metrics in AXES_AND_METRICS.items():
        if (observation_spec is not None) and (axis not in observation_spec):
            continue # this axis not in observation specs.

        axis_metric_results[axis] = {}

        for metric in metrics:
            if (observation_spec is not None) and (metric["name"] not in observation_spec[axis]):
                 continue # this metric not in observation specs.

            func_name = metric["function"]
            func = AXIS_METRIC_FUNCTIONS[func_name]

            # Assemble the argument values by fetching the relevant portions
            # of the data
            kwargs = dict([(arg_name, _get_metric_func_arg_value(arg_val_name, dataset))
                for arg_name, arg_val_name in metric["args"].items()])

            # Call the metric function with inputs as kwargs.
            res = func(**kwargs)

            # Store results.
            metric_name = metric["name"]
            axis_metric_results[axis][metric_name] = res
            axis_metric_results_flat[f"{axis} {metric_name}"] = res

    return axis_metric_results, axis_metric_results_flat

def plot_axis_metric_analysis(results, levels, max_measure_count=1, mean=False, min_max=False):
    '''
    Plot the results of the analyze_metric_measures_across_levels
    function. Show the median, mean and 1st and 3rd quartiles for each
    metric measure at each level.
    '''

    level_colors = ["g", "b", "y"] # low medium high will be green blue yellow
    mean_color = "r" # mean will be red
    max_min_color = "k" # min max will be black
    for metric, measures in results.items():
        plt.figure(figsize=(10, 4))
        plt.suptitle(f"{metric}")
        plt.tight_layout()

        for measure_num, (measure_name, measure_values) in enumerate(measures.items()):
            plt.subplot(1, max_measure_count, 1 + measure_num)
            plt.title(f"{measure_name}")
            for level_num, level in enumerate(levels):
                level_values = measure_values[level]

                # Find quartile values in data
                quartiles = np.percentile(
                    level_values,
                    [25, 50, 75],
                    interpolation = 'midpoint')

                # Prepare plotting data
                x = level_num+1
                y_median = quartiles[1]

                err = np.array([
                    [y_median - quartiles[0]],
                    [quartiles[2] - y_median]
                ])

                # Plot
                color = level_colors[level_num]
                plt.xlim((0, max_measure_count+1))
                plt.scatter(x, y_median, label=level, color=color)
                plt.errorbar(x, y_median, err, color=color)

                if mean:
                    y_mean = np.mean(level_values)
                    plt.scatter(x, y_mean, color=mean_color)

                if min_max:
                    y_max = np.max(level_values)
                    y_min = np.min(level_values)
                    plt.scatter(x, y_max, color=max_min_color)
                    plt.scatter(x, y_min, color=max_min_color)

            plt.legend()

        plt.show()
