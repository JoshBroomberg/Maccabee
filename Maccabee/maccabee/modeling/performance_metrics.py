"""This submodule contains the definitions for the :term:`performance metrics <performance metric>` used to quantify the performance of causal estimators implemented using the model classes from :mod:`~maccabee.modeling.models`.

The metrics defined in this submodule are divided into those which quantify the performance of average effect estimators and those which quantify the performance of individual effect estimators. These two classes of metrics are operationally very similar. They define some measure of the estimator quality based on a comparison of the estimate value and the ground truth. And, crucially, they are both defined over samples of estimate values rather than a single estimate-ground-truth pair (although some are well defined in this individual limit). The difference between them is that average effect metrics operate over a pair of real numbers for the estimated/true effect per data set while individual effect metrics operate over a pair of real-valued estimate/truth vectors.

The submodule exposes these two sets of metrics as dictionaries - one for average effect metrics and one for individual effect metrics. These dictionaries map metric names to callables that calculate the metrics given the inputs outlined above. These dictionaries are used by the :mod:`~maccabee.benchmarking` module to calculate performance metrics for models when applied to :class:`~maccabee.data_generation.generated_data_set.GeneratedDataSet` instances produced by concrete/sampled :class:`DataGeneratingProcesses <maccabee.data_generation.data_generating_process.DataGeneratingProcess>`. The dictionaries, and the metrics contained by each, are outlined below.

In the documentation below, assume a sample of :math:`N` data sets each of which contain :math:`M` observations. The estimated effects will be referred to as :math:`\\hat{\\tau}` and the ground truth effects as :math:`\\tau`. For average effects, of which there is one estimate/ground-truth pair per data set, the :math:`i` th pair of values will be referred to as :math:`\\hat{\\tau_i}`/:math:`\\tau_i` respectively. For individual effects, of which there are :math:`M` per data set, the :math:`j` th observation in the :math:`i` th data set will be referred to as :math:`\\hat{\\tau_{ij}}`/:math:`\\tau_{ij}`.
"""

import numpy as np
from ..exceptions import UnknownEstimandAggregationException
from ..constants import Constants

def root_mean_sqaured_error(avg_effect_estimate_vals, avg_effect_true_vals):
    """root_mean_sqaured_error(...)

    The Root Mean Squared Error (RMSE):

    .. math::

         \\sqrt{\\frac{1}{N} \\times \\sum_{i=1}^N \\left(\\hat{\\tau_i} - \\tau_i \\right)^2}

    The RMSE is a well-known measure which captures the sum of estimator bias and variance. A zero RMSE implies both zero variance and bias. But a non-zero RMSE does not indicate wether bias or variance is the source of the error. This metric is, therefore, paired with the Absolute Mean Bias Percentage, which measures the bias. If bias is low but the RMSE is high, then the source of this high RMSE is isolated to estimator variance.

    Args:
        avg_effect_estimate_vals (:class:`numpy.ndarray`): an array of sampled average effect estimates.
        avg_effect_true_vals (:class:`numpy.ndarray`): an array of sampled average effect ground truths.

    Returns:
        float: The Root Mean Squared Error.
    """
    return np.sqrt(
        np.mean((avg_effect_estimate_vals - avg_effect_true_vals)**2))

def mean_absolute_bias_percentage(avg_effect_estimate_vals, avg_effect_true_vals):
    """mean_absolute_bias_percentage(...)

    The Mean Absolute Bias Percentage (MABP):

    .. math::

        100 \\times \\frac{1}{N} \\times \\sum_{i=1}^N \\left( \\left| \\frac{\\hat{\\tau_i} - \\tau_i}{\\tau_i} \\right| \\right)

    This metric quantifies the average bias in the estimator. It measures the mean absolute value of this bias in the estimate as a percentage of the ground-truth effects. Note: an unbiased estimator may still have a large MAB if each estimate
    tends to be far from the ground truth.

    Args:
        avg_effect_estimate_vals (:class:`numpy.ndarray`): an array of sampled average effect estimates.
        avg_effect_true_vals (:class:`numpy.ndarray`): an array of sampled average effect ground truths.

    Returns:
        float: The Absolute Mean Error Percentage.
    """
    non_zeros = np.logical_not(np.isclose(avg_effect_true_vals, 0))
    return 100*np.mean(
        np.abs((avg_effect_estimate_vals[non_zeros] - avg_effect_true_vals[non_zeros]) /
            avg_effect_true_vals[non_zeros]))

def absolute_mean_bias_percentage(avg_effect_estimate_vals, avg_effect_true_vals):
    """absolute_mean_bias_percentage(...)

    The Absolute Mean Bias Percentage (AMBP):

    .. math::

        100 \\times \\left| \\frac{1}{N} \\times \\sum_{i=1}^N \\frac{\\hat{\\tau_i} - \\tau_i}{\\tau_i} \\right|

    This metric quantifies the degree of systematic bias in the estimator. An unbiased estimator will not favor estimates either above or below the ground truth. If this is true, when many estimates are averaged, their mean will be zero. A non-zero mean indicates bias exists. Taking the absolute value of this bias and scaling as a percentage of the ground-truth effects reflects the equal severity of positive/negative bias and allows the bias to be averaged/compared for different magnitudes of effect.

    Args:
        avg_effect_estimate_vals (:class:`numpy.ndarray`): an array of sampled average effect estimates.
        avg_effect_true_vals (:class:`numpy.ndarray`): an array of sampled average effect ground truths.

    Returns:
        float: The Absolute Mean Error Percentage.
    """
    non_zeros = np.logical_not(np.isclose(avg_effect_true_vals, 0))
    return 100*np.abs(
        np.mean((avg_effect_estimate_vals[non_zeros] - avg_effect_true_vals[non_zeros]) /
            avg_effect_true_vals[non_zeros]))

#: The dictionary containing the average effect metrics.
#: It contains the following metrics:
#:
#: * :func:`RMSE <maccabee.modeling.performance_metrics.root_mean_sqaured_error>` - the Root Mean Squared Error.
#: * :func:`AMBP <maccabee.modeling.performance_metrics.absolute_mean_bias_percentage>`  - Absolute Mean Bias Percentage.
#: * :func:`MABP <maccabee.modeling.performance_metrics.mean_absolute_bias_percentage>`  - Mean Absolute Bias Percentage.
AVG_EFFECT_METRICS = {
    "RMSE": root_mean_sqaured_error,
    "AMBP": absolute_mean_bias_percentage,
    "MABP": mean_absolute_bias_percentage
}

def precision_in_estimating_heterogenous_treat_effects(
    indv_effect_estimate_vals, indv_effect_true_vals):
    """precision_in_estimating_heterogenous_treat_effects(...)

    The Precision in Estimation of Heterogenous (Treatment) Effects (PEHE):

    .. math::

        \\frac{1}{N} \\times \\sum_{i=1}^N \\left( \\sqrt{\\frac{1}{M} \\times \\sum_{j=1}^M \\left(\\hat{\\tau_{ij}} - \\tau_{ij} \\right)^2} \\right)

    This metric, first introduced by Hill (2011) [#hill]_, measures the quality of the estimation of the individual treatment effects in a data set. It measures the RMSE of the individual effect estimate within data sets and then averages this value across data sets. As such, it measures both the bias and variance of an individual effect estimator in producing individual effect estimates.

    Args:
        indv_effect_estimate_vals (:class:`numpy.ndarray`): a 2D array of sampled average effect estimates. Each row representing the individual effect estimate data for a sampled data set.
        indv_effect_true_vals (:class:`numpy.ndarray`): a 2D array of sampled average effect ground truths. Each row representing the individual effect ground truth data for a sampled data set.

    Returns:
        float: Precision in Estimation of Heterogenous (Treatment) Effects

    """
    return np.mean(
        np.sqrt(
            np.mean((indv_effect_estimate_vals - indv_effect_true_vals)**2, axis=1)))

#: The dictionary containing the individual effect metrics.
#: It contains the following metrics:
#:
#: * :func:`PEHE <maccabee.modeling.performance_metrics.precision_in_estimating_heterogenous_treat_effects>` - The Precision in Estimation of Heterogenous (Treatment) Effects
INDIVIDUAL_EFFECT_METRICS = {
    "PEHE": precision_in_estimating_heterogenous_treat_effects
}

def add_performance_metric(aggregation_level, metric_name, metric_callable):
    """Adds a performance metric for individual or average estimands to the benchmarks.

    Args:
        aggregation_level (str): The estimand aggregation level to which the metric applies. One of either: ``Constants.Model.INDIVIDUAL_ESTIMANDS`` or ``Constants.Model.AVERAGE_ESTIMANDS``.
        metric_name (str): The name of the metric. Must be unique for the aggregation level.
        metric_callable (function): A callable which accepts two arguments - one for the estimand estimate values and one for the ground truth values. If at the average effect aggregation level, each argument is a 1D :class:`numpy.ndarray` with each entry corresponding to an estimand value in a different dataset of a sampling run. If at the individial effect aggregation level, each argument is a 2D :class:`numpy.ndarray` with each row corresponding to individual effect estimand values for a single dataset (one column per individual).
    """

    if aggregation_level == Constants.Model.INDIVIDUAL_ESTIMANDS:
        metric_dict = INDIVIDUAL_EFFECT_METRICS
    elif aggregation_level == Constants.Model.AVERAGE_ESTIMANDS:
        metric_dict = AVG_EFFECT_METRICS
    else:
        raise UnknownEstimandAggregationException()

    if metric_name in metric_dict:
        raise ValueError(f"Metric {metric_name} already exists for aggregation level.")

    metric_dict[metric_name] = metric_callable
