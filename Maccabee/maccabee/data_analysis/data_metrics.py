"""This module contains the low-level components used to analyze observational datasets to determine the distributional setting that they represent."""

import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from scipy.spatial.distance import cdist
import ot
import pandas as pd
from collections import defaultdict

from ..constants import Constants
from .utils import extract_treat_and_control_data
from ..parameters import build_parameters_from_axis_levels
from ..data_generation import DataGeneratingProcessSampler

AxisNames = Constants.AxisNames
DataMetricFunctions = Constants.DataMetricFunctions
DGPVariables = Constants.DGPVariables


#: This dict specifies the measured axes and the metrics for each axis.
#: Each axis has a list of metrics which are defined
#: by a metric calculation function and the arguments to be supplied to it
#: as well as a unique name.
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
                "x": DGPVariables.TREATMENT_ASSIGNMENT_NAME,
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

#: A dictionary of all available data axes and the associated metrics
#: which measure the position of a dataset along each axis.
AXES_AND_METRIC_NAMES = dict(
    (axis, [metric["name"] for metric in metrics])
    for axis, metrics in AXES_AND_METRICS.items())


### Metric functions

# Below are the functions which are used
# to calculate the metric values. Each function
# is used in multiple metrics so they are named and
# parameterized generically.

def linear_regression_r2(X, y):
    if type(X) == pd.Series:
        X = X.to_numpy().reshape((-1, 1))

    lr = LinearRegression()
    lr.fit(X, y)
    return lr.score(X, y)

def logistic_regression_r2(X, y):
    if type(X) == pd.Series:
        X = X.to_numpy().reshape((-1, 1))

    lr = LogisticRegression(solver='lbfgs')
    lr.fit(X, y)
    return lr.score(X, y)

def percent(x, value):
    '''
    Percent of values in x which have the value in value.
    '''
    x = np.array(x)
    return 100*np.sum((x == value).astype(int))/len(x)

def l2_distance_between_means(covariates, treatment_status):
    '''
    L2 norm of the distance between the means of the covariates
    in the treat and control groups.
    '''
    X_treated, X_control = extract_treat_and_control_data(
        covariates, treatment_status)

    X_treated_mean = np.mean(X_treated, axis=0)
    X_control_mean = np.mean(X_control, axis=0)

    return np.linalg.norm(X_treated_mean - X_control_mean)

def mean_mahalanobis_between_nearest_counterfactual(covariates, treatment_status):
    '''
    Mahalanobis distance between the nearest neighbor of each treated unit
    which is in the control group.
    '''

    X_treated, X_control = extract_treat_and_control_data(
        covariates, treatment_status)

    # TODO: fix singular matrix issue.
    try:
        distance_matrix = cdist(X_treated, X_control, "mahalanobis")
        np.nan_to_num(distance_matrix, copy=False, nan=np.inf)
        return np.mean(np.min(distance_matrix, axis=1))
    except:
        return np.NaN

def standard_deviation_ratio(x1, x2):
    '''
    Ratio of the std of x1 and x2
    '''
    return np.std(x1)/np.std(x2)

def wasserstein(covariates, treatment_status):
    '''
    Wasserstein distance between the covariates in the treat and control groups.
    '''

    X_treated, X_control = extract_treat_and_control_data(
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
        l2_distance_between_means(covariates, treatment_status) > 1e-4:
        print(f"Detected failure with W={wass_dist}...")
        return None
    else:
        return wass_dist

def naive_TE_estimate_error(TE, observed_outcome, treatment_status):
    '''
    Absolute difference between the true treatment effect and the
    naive estimate based on mean outcome in each group.
    '''
    Y_t, Y_c = extract_treat_and_control_data(observed_outcome, treatment_status)
    ATE_true = np.mean(TE)
    ATE_est = np.mean(Y_t) - np.mean(Y_c)
    return np.abs(ATE_true - ATE_est)

# Map function names to function callables.
AXIS_METRIC_FUNCTIONS = {
    DataMetricFunctions.LINEAR_R2: linear_regression_r2,
    DataMetricFunctions.LOGISTIC_R2: logistic_regression_r2,
    DataMetricFunctions.PERCENT: percent,
    DataMetricFunctions.L2_MEAN_DIST: l2_distance_between_means,
    DataMetricFunctions.NN_CF_MAHALA_DIST: mean_mahalanobis_between_nearest_counterfactual,
    DataMetricFunctions.STD_RATIO: standard_deviation_ratio,
    DataMetricFunctions.WASS_DIST: wasserstein,
    DataMetricFunctions.NAIVE_TE: naive_TE_estimate_error
}

# METRIC INPUT ACCESSORS

# The dictionary below defines accessor functions which extract each
# metric input from the GeneratedDataSet object.

AXIS_METRIC_FUNC_INPUT_ACCESSORS = {
    # Covariate Data
    DGPVariables.COVARIATES_NAME: lambda ds: ds.observed_covariate_data,
    DGPVariables.TRANSFORMED_COVARIATES_NAME: lambda ds: ds.transformed_covariate_data,

    # Observed Variables
    DGPVariables.OBSERVED_OUTCOME_NAME: lambda ds: ds.Y,
    DGPVariables.TREATMENT_ASSIGNMENT_NAME: lambda ds: ds.T,

    # Oracle
    DGPVariables.POTENTIAL_OUTCOME_WITHOUT_TREATMENT_NAME: lambda ds: ds.Y0,
    DGPVariables.POTENTIAL_OUTCOME_WITH_TREATMENT_NAME: lambda ds: ds.Y1,
    DGPVariables.TREATMENT_EFFECT_NAME: lambda ds: ds.TE,
    DGPVariables.PROPENSITY_LOGIT_NAME: lambda ds: ds.oracle_outcome_data[DGPVariables.PROPENSITY_LOGIT_NAME],
    DGPVariables.PROPENSITY_SCORE_NAME: lambda ds: ds.oracle_outcome_data[DGPVariables.PROPENSITY_SCORE_NAME],
}
