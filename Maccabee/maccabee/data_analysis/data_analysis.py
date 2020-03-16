"""This submodule contains the functions responsible for calculating the metrics used to quantify the position of a data set on each of the :term:`axes <distributional problem space axis>` of the :term:`distributional problem space`. As described in the sibling :mod:`~maccabee.data_analysis.data_metrics` submodule, each axis may have multiple metrics. The primary function in this module takes a :class:`maccabee.data_generation.generated_data_set.GeneratedDataSet` instance and an `observation_spec` which selects which axes and associated metrics to calculate. It then executes the calculation using the dictionary-based metric definitions from :mod:`~maccabee.data_analysis.data_metrics`.
"""

import numpy as np
import matplotlib.pyplot as plt

from .data_metrics import AXES_AND_METRICS, AXIS_METRIC_FUNCTIONS


def calculate_data_axis_metrics(dataset, observation_spec=None, flatten_result=False):
    """This function takes a :class:`maccabee.data_generation.generated_data_set.GeneratedDataSet` instance and calculates the data metrics specified in `observation_spec`. It is primarily during the benchmarking process but can be used as a stand alone method for custom workflows.

    Args:
        dataset (:class:`~maccabee.data_generation.generated_data_set.GeneratedDataSet`): A :class:`~maccabee.data_generation.generated_data_set.GeneratedDataSet` instance generated from a :class:`~maccabee.data_generation.data_generating_process.DataGeneratingProcess`.

        observation_spec (dict): A dictionary which specifies which :term:`data metrics <data metric>` to calculate and record. The keys are axis names and the values are lists of string metric names. All axis names and the metrics for each axis are available in the dictionary :obj:`maccabee.data_analysis.data_metrics.AXES_AND_METRIC_NAMES`. If None, all data metrics are calculated. Defaults to None.

        flatten_result (bool): indicates whether the results should be flattened into a single dictionary by concatenating the axis and metric names into a single key rather than returning nested dictionaries with axis and metric names as the keys at the first and second level of nesting.

    Returns:
        dict: A dictionary of axis names and associated metric results. If flatten_result is ``False``, then this is a dictionary in which axis names are mapped to dictionaries with metric name keys and real valued values. If flatten_result is ``True``, then this is a dictionary in which the keys are the concatenation of axis and metric names and the values are the corresponding real values.

    Raises:
        UnknownDGPVariableException: if a selected metric function specifies an unknown DGP variable as an arg to its calculation function.
    """

    axis_metric_results = {}

    for axis, metrics in AXES_AND_METRICS.items():
        if (observation_spec is not None) and (axis not in observation_spec):
            continue # this axis not in observation specs.

        if not flatten_result:
            axis_metric_results[axis] = {}

        for metric in metrics:
            if (observation_spec is not None) and (metric["name"] not in observation_spec[axis]):
                 continue # this metric not in observation specs.

            func = metric["function"]
            if not callable(func):
                func = AXIS_METRIC_FUNCTIONS[func]

            # Assemble the argument values by fetching the relevant portions
            # of the data

            dgp_var_kwargs = dict([
                (arg_name, dataset.get_dgp_variable(dgp_var_name))
                for arg_name, dgp_var_name in metric["args"].items()
            ])

            constant_kwargs = metric.get("constant_args", {})

            # Call the metric function with inputs as kwargs.
            res = func(**dgp_var_kwargs, **constant_kwargs)

            # Store results.
            metric_name = metric["name"]

            if flatten_result:
                axis_metric_results[f"{axis} {metric_name}"] = res
            else:
                axis_metric_results[axis][metric_name] = res

    return axis_metric_results
