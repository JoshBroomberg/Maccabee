"""This submodule contains the definitions for the metrics used to quantify the position of a data set on each of the :term:`axes <distributional problem space axis>` of the :term:`distributional problem space`. Each axis has one or more metrics, each of which operates on (potentially overlapping) components of the data set to output a single, real metric value that measures the data's position on the associated axis.

The module uses a dictionary-based data structure to define data metrics. IE, rather than concretely defining metrics for each axis as python functions which extract the relevant data from a :class:`~maccabee.data_generation.generated_data_set.GeneratedDataSet` instance and perform some calculation, metrics are defined using dictionaries specify which data and functions to (re)use. This allows different concrete metrics to share the same data and calculation functions without code repetition. It also allows package users to inject new metrics at run time by simply modifying the dictionaries outlined below.

The module actually uses, and exposes, three dictionaries which work together to define the data metrics. The main dictionary is the :data:`~maccabee.data_analysis.data_metrics.AXES_AND_METRICS` dictionary. The other two dictionaries support the specification and use of the main dictionary. :data:`~maccabee.data_analysis.data_metrics.AXES_AND_METRIC_NAMES` summarizes the content of :data:`~maccabee.data_analysis.data_metrics.AXES_AND_METRICS` by mapping the axis names to a list of unique metric names. This can be used for convenient selection of the metrics to record when running a benchmark (see the :mod:`~maccabee.benchmarking` module for more). The :data:`~maccabee.data_analysis.data_metrics.AXIS_METRIC_FUNCTIONS` maps function name constants from :class:`maccabee.constants.Constants.DataMetricFunctions` to generic calculation functions/callables defined in this module.

The main :data:`~maccabee.data_analysis.data_metrics.AXES_AND_METRICS` dictionary defines the data metrics by mapping each axis name from :class:`maccabee.constants.Constants.AxisNames` to a list of *metric definition dictionaries*. Each metric definition dictionary has three components:

* The unique name of the metric. This is the name which appears in the :data:`~maccabee.data_analysis.data_metrics.AXES_AND_METRIC_NAMES` dictionary and is used when specifying which metrics to collect during a benchmark.

* The generic metric function used to calculate the metric. This is specified as a function name constant from :class:`maccabee.constants.Constants.DataMetricFunctions`. As mentioned above, the dictionary :data:`~maccabee.data_analysis.data_metrics.AXIS_METRIC_FUNCTIONS` maps these names to callables defined in this module. This structure is used because the functions are typically repeated across many different concrete metrics. So it is efficient to define generic functions once and reuse the same callable repeatedly. For example, there is a linear regression :math:`R^2` function which regresses the vector arg :math:`y` against the matrix arg :math:`X`.

* The arguments to the generic metric function. These concretize what the metric measures by applying the generic function to specific data. For example, by passing the original covariates and observed outcome as the arguments :math:`X` and :math:`y` of the linear regression function, one can construct a metric for the linearity of the outcome. The arguments are specified by a dictionary which maps the generic functions (generic) argument names to DGP data variable names from :class:`maccabee.constants.Constants.DGPVariables`. These constant names are then used to access the corresponding data from :class:`~maccabee.data_generation.generated_data_set.GeneratedDataSet` instances.


"""

import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from scipy.spatial.distance import cdist
import ot
import pandas as pd
from collections import defaultdict

from ..constants import Constants
from ..parameters import build_parameters_from_axis_levels
from ..data_generation import DataGeneratingProcessSampler

from ..logging import get_logger
logger = get_logger(__name__)

AxisNames = Constants.AxisNames
DGPVariables = Constants.DGPVariables
DataMetricFunctions = Constants.DataMetricFunctions


#: The dictionary mapping axis names to a list of metric definition dictionaries.
#: Each metric definition dictionary has a name, calculation function, and argument
#: mapping that specifies which DGP variables to supply as each function arg.
AXES_AND_METRICS = {
    AxisNames.OUTCOME_NONLINEARITY: [
        {
            "function": DataMetricFunctions.LINEAR_R2,
            "args": {
                "X": DGPVariables.COVARIATES_NAME,
                "y": DGPVariables.OBSERVED_OUTCOME_NAME
            },
            "name": "Lin r2(X_obs, Y)"
        },
        {
            "function": DataMetricFunctions.LINEAR_R2,
            "args": {
                "X": DGPVariables.TRANSFORMED_COVARIATES_NAME,
                "y": DGPVariables.OBSERVED_OUTCOME_NAME
            },
            "name": "Lin r2(X_true, Y)"
        },
        {
            "function": DataMetricFunctions.LINEAR_R2,
            "args": {
                "X": DGPVariables.COVARIATES_NAME,
                "y": DGPVariables.POTENTIAL_OUTCOME_WITH_TREATMENT_NAME
            },
            "name": "Lin r2(X_obs, Y1)"
        },
        {
            "function": DataMetricFunctions.LINEAR_R2,
            "args": {
                "X": DGPVariables.COVARIATES_NAME,
                "y": DGPVariables.POTENTIAL_OUTCOME_WITHOUT_TREATMENT_NAME
            },
            "name": "Lin r2(X_obs, Y0)"
        },
        {
            "function": DataMetricFunctions.LINEAR_R2,
            "args": {
                "X": DGPVariables.TRANSFORMED_COVARIATES_NAME,
                "y": DGPVariables.POTENTIAL_OUTCOME_WITH_TREATMENT_NAME
            },
            "name": "Lin r2(X_true, Y1)"
        },
        {
            "function": DataMetricFunctions.LINEAR_R2,
            "args": {
                "X": DGPVariables.TRANSFORMED_COVARIATES_NAME,
                "y": DGPVariables.POTENTIAL_OUTCOME_WITHOUT_TREATMENT_NAME
            },
            "name": "Lin r2(X_true, Y0)"
        },
        {
            "function": DataMetricFunctions.LINEAR_R2,
            "args": {
                "X": DGPVariables.COVARIATES_NAME,
                "y": DGPVariables.TREATMENT_EFFECT_NAME
            },
            "name": "Lin r2(X_obs, TE)"
        },
        {
            "function": DataMetricFunctions.LINEAR_R2,
            "args": {
                "X": DGPVariables.TRANSFORMED_COVARIATES_NAME,
                "y": DGPVariables.TREATMENT_EFFECT_NAME
            },
            "name": "Lin r2(X_true, TE)"
        }
    ],

    AxisNames.TREATMENT_NONLINEARITY: [
        {
            "function": DataMetricFunctions.LOGISTIC_R2,
            "args": {
                "X": DGPVariables.COVARIATES_NAME,
                "y": DGPVariables.TREATMENT_ASSIGNMENT_NAME
            },
            "name": "Log r2(X_obs, T)"
        },
        {
            "function": DataMetricFunctions.LINEAR_R2,
            "args": {
                "X": DGPVariables.COVARIATES_NAME,
                "y": DGPVariables.PROPENSITY_LOGIT_NAME
            },
            "name": "Lin r2(X_obs, Treat Logit)"
        },
        {
            "function": DataMetricFunctions.LINEAR_R2,
            "args": {
                "X": DGPVariables.TRANSFORMED_COVARIATES_NAME,
                "y": DGPVariables.PROPENSITY_LOGIT_NAME
            },
            "name": "Lin r2(X_true, Treat Logit)"
        }
    ],

    AxisNames.PERCENT_TREATED: [
        {
            "function": DataMetricFunctions.PERCENT,
            "args": {
                "x": DGPVariables.TREATMENT_ASSIGNMENT_NAME
            },
            "constant_args": {
                "value": 1
            },
            "name": "Percent(T==1)"
        }
    ],

    AxisNames.OVERLAP: [
        {
            "function": DataMetricFunctions.NN_CF_MAHALA_DIST,
            "args": {
                "covariates": DGPVariables.COVARIATES_NAME,
                "treatment_status": DGPVariables.TREATMENT_ASSIGNMENT_NAME
            },
            "name": "NN dist X_obs: T=1<->T=0"
        },
        {
            "function": DataMetricFunctions.NN_CF_MAHALA_DIST,
            "args": {
                "covariates": DGPVariables.TRANSFORMED_COVARIATES_NAME,
                "treatment_status": DGPVariables.TREATMENT_ASSIGNMENT_NAME
            },
            "name": "NN dist X_true: T=1<->T=0"
        }
    ],

    AxisNames.BALANCE: [
        {
            "function": DataMetricFunctions.L2_MEAN_DIST,
            "args": {
                "covariates": DGPVariables.TRANSFORMED_COVARIATES_NAME,
                "treatment_status": DGPVariables.TREATMENT_ASSIGNMENT_NAME
            },
            "name": "Mean dist X_true: T=1<->T=0"
        },
        {
            "function": DataMetricFunctions.WASS_DIST,
            "args": {
                "covariates": DGPVariables.TRANSFORMED_COVARIATES_NAME,
                "treatment_status": DGPVariables.TREATMENT_ASSIGNMENT_NAME
            },
            "name": "Wass dist X_true: T=1<->T=0"
        },
        {
            "function": DataMetricFunctions.WASS_DIST,
            "args": {
                "covariates": DGPVariables.COVARIATES_NAME,
                "treatment_status": DGPVariables.TREATMENT_ASSIGNMENT_NAME
            },
            "name": "Wass dist X_obs: T=1<->T=0"
        },
        {
            "function": DataMetricFunctions.NAIVE_TE,
            "args": {
                "TE": DGPVariables.TREATMENT_EFFECT_NAME,
                "observed_outcome": DGPVariables.OBSERVED_OUTCOME_NAME,
                "treatment_status": DGPVariables.TREATMENT_ASSIGNMENT_NAME
            },
            "name": "Naive TE"
        }
    ],

    AxisNames.ALIGNMENT: [
        {
            "function": DataMetricFunctions.LINEAR_R2,
            "args": {
                "X": DGPVariables.OBSERVED_OUTCOME_NAME,
                "y": DGPVariables.PROPENSITY_LOGIT_NAME
            },
            "name": "Lin r2(Y, Treat Logit)"
        },
        {
            "function": DataMetricFunctions.LINEAR_R2,
            "args": {
                "X": DGPVariables.POTENTIAL_OUTCOME_WITHOUT_TREATMENT_NAME,
                "y": DGPVariables.PROPENSITY_LOGIT_NAME
            },
            "name": "Lin r2(Y0, Treat Logit)"
        },
        {
            "function": DataMetricFunctions.LOGISTIC_R2,
            "args": {
                "X": DGPVariables.POTENTIAL_OUTCOME_WITHOUT_TREATMENT_NAME,
                "y": DGPVariables.TREATMENT_ASSIGNMENT_NAME
            },
            "name": "Log r2(Y0, T)"
        },
        {
            "function": DataMetricFunctions.LOGISTIC_R2,
            "args": {
                "X": DGPVariables.OBSERVED_OUTCOME_NAME,
                "y": DGPVariables.TREATMENT_ASSIGNMENT_NAME
            },
            "name": "Log r2(Y, T)"
        }
    ],

    AxisNames.TE_HETEROGENEITY: [
        {
            "function": DataMetricFunctions.STD_RATIO,
            "args": {
                "x1": DGPVariables.TREATMENT_EFFECT_NAME,
                "x2": DGPVariables.OBSERVED_OUTCOME_NAME
            },
            "name": "std(TE)/std(Y)"
        }
    ]
}

#: The dictionary mapping axis names to a list of available
#: metric names.
AXES_AND_METRIC_NAMES = dict(
    (axis, [metric["name"] for metric in metrics])
    for axis, metrics in AXES_AND_METRICS.items())

CUSTOM_METRICS = defaultdict(list)

def add_data_metric(axis_name, metric_dict):
    """Add a data metric specified by the components of `metric_dict` to the metrics for the axis in `axis_name`.

    Args:
        axis_name (str): The name of an axis from :class:`~maccabee.constants.Constants.AxisNames`.
        metric_dict (dict): A dict, as described above, which contains keys for the name, args and function that is used to calculate the metric.
    """

    req_fields = ["function", "args", "name"]
    for field in req_fields:
        if field not in metric_dict:
            raise ValueError(f"Missing field {field} from metric_dict")

    metric_name = metric_dict["name"]
    if metric_name in AXES_AND_METRIC_NAMES[axis_name]:
        raise ValueError(f"Metric name {metric_name} already exists for {axis_name} and cannot be redefined.")

    AXES_AND_METRICS[axis_name].append(metric_dict)
    AXES_AND_METRIC_NAMES[axis_name].append(metric_name)
    CUSTOM_METRICS[axis_name].append(metric_name)

### Metric functions

# Below are the functions which are used
# to calculate the metric values. Each function
# is used in multiple metrics so they are named and
# parameterized generically.

# NOTE: these functions follow a functional programming
# error paradigm in which None is returned in place of errors.
# This allows for more convenient bulk calculation of metrics
# for many sampled data sets.

def _extract_treat_and_control_data(covariates, treatment_status):
    # Extract the treated and control observations from a set of
    # covariates given treatment statuses.

    X_treated = covariates[(treatment_status==1).to_numpy()]
    X_control = covariates[(treatment_status==0).to_numpy()]
    return X_treated, X_control

def _linear_regression_r2(X, y):
    if type(X) == pd.Series:
        X = X.to_numpy().reshape((-1, 1))

    lr = LinearRegression()
    lr.fit(X, y)
    return lr.score(X, y)

def _logistic_regression_r2(X, y):
    if type(X) == pd.Series:
        X = X.to_numpy().reshape((-1, 1))

    lr = LogisticRegression(solver='lbfgs')
    lr.fit(X, y)
    return lr.score(X, y)

def _percent(x, value):
    '''
    Percent of values in x which have the value in value.
    '''
    x = np.array(x)
    return 100*np.sum((x == value).astype(int))/len(x)

def _l2_distance_between_means(covariates, treatment_status):
    '''
    L2 norm of the distance between the means of the covariates
    in the treat and control groups.
    '''
    X_treated, X_control = _extract_treat_and_control_data(
        covariates, treatment_status)

    X_treated_mean = np.mean(X_treated, axis=0)
    X_control_mean = np.mean(X_control, axis=0)

    return np.linalg.norm(X_treated_mean - X_control_mean)

def _mean_mahalanobis_between_nearest_counterfactual(covariates, treatment_status):
    '''
    Mahalanobis distance between the nearest neighbor of each treated unit
    which is in the control group.
    '''

    X_treated, X_control = _extract_treat_and_control_data(
        covariates, treatment_status)

    try:
        # Under degenerate conditions, this will through a singular
        # matrix exception. In this case, a None is returned.
        distance_matrix = cdist(X_treated, X_control, "mahalanobis")
        np.nan_to_num(distance_matrix, copy=False, nan=np.inf)
        return np.mean(np.min(distance_matrix, axis=1))
    except:
        logger.exception("Ill-conditioned Mahalanobis distance calculation")
        return None

def _standard_deviation_ratio(x1, x2):
    '''
    Ratio of the std of x1 and x2
    '''
    return np.std(x1)/np.std(x2)

def _wasserstein(covariates, treatment_status):
    '''
    Wasserstein distance between the covariates in the treat and control groups.
    '''

    X_treated, X_control = _extract_treat_and_control_data(
        covariates, treatment_status)

    num_treated, num_control = len(X_treated), len(X_control)

    a = np.ones(num_treated)/num_treated
    b = np.ones(num_control)/num_control

    M = ot.dist(X_treated, X_control)
    M /= M.max()

    # Remove all zero-cost entries to produce
    # a properly conditioned problem.
    a, b, M = ot.utils.clean_zeros(a, b, M)

    lambd = 1e-2

    wass_dist = ot.sinkhorn2(a, b, M, lambd, maxiter=3000)[0]
    if wass_dist < 1e-3 and \
        _l2_distance_between_means(covariates, treatment_status) > 1e-4:
        logger.error(f"Detected failure in Wasserstein distance with W={wass_dist}...")
        return None
    else:
        return wass_dist

def _naive_TE_estimate_error(TE, observed_outcome, treatment_status):
    '''
    Absolute difference between the true treatment effect and the
    naive estimate based on mean outcome in each group.
    '''
    Y_t, Y_c = _extract_treat_and_control_data(observed_outcome, treatment_status)
    ATE_true = np.mean(TE)
    ATE_est = np.mean(Y_t) - np.mean(Y_c)
    return np.abs(ATE_true - ATE_est)

#: The dictionary mapping constant metric function names to
#: function callables from this module.
AXIS_METRIC_FUNCTIONS = {
    DataMetricFunctions.LINEAR_R2: _linear_regression_r2,
    DataMetricFunctions.LOGISTIC_R2: _logistic_regression_r2,
    DataMetricFunctions.PERCENT: _percent,
    DataMetricFunctions.L2_MEAN_DIST: _l2_distance_between_means,
    DataMetricFunctions.NN_CF_MAHALA_DIST: _mean_mahalanobis_between_nearest_counterfactual,
    DataMetricFunctions.STD_RATIO: _standard_deviation_ratio,
    DataMetricFunctions.WASS_DIST: _wasserstein,
    DataMetricFunctions.NAIVE_TE: _naive_TE_estimate_error
}
