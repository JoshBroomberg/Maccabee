import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from scipy.spatial.distance import cdist
import ot
from .constants import Constants
import pandas as pd
from .utilities import extract_treat_and_control_data

### Metric functions
# These are the functions which are used
# to calculate the measure values. Each function
# is used in multiple measures so they are named and
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


MetricFunctions = Constants.MetricFunctions
MetricData = Constants.MetricData
MetricNames = Constants.MetricNames

### NOTE: the code below is complex because it is designed
# to allow new measures to be created very easily. Feel free
# to skip down to the measure definitions.

# Map the function names to function callables.
metric_functions = {
    MetricFunctions.LINEAR_R2: linear_regression_r2,
    MetricFunctions.LOGISTIC_R2: logistic_regression_r2,
    MetricFunctions.PERCENT: percent,
    MetricFunctions.L2_MEAN_DIST: l2_distance_between_means,
    MetricFunctions.NN_CF_MAHALA_DIST: mean_mahalanobis_between_nearest_counterfactual,
    MetricFunctions.STD_RATIO: standard_deviation_ratio,
    MetricFunctions.WASS_DIST: wasserstein,
    MetricFunctions.NAIVE_TE: naive_TE_estimate_error
}

# TODO: remove names and simplify access.
# Specify ways to access each input variable.
data_accessors = {
    # Covariate Data
    MetricData.OBSERVED_COVARIATE_DATA: {
        "accessor": lambda data: data[MetricData.OBSERVED_COVARIATE_DATA],
        "name": "X_obs"
    },
    MetricData.ORACLE_COVARIATE_DATA: {
        "accessor": lambda data: data[MetricData.ORACLE_COVARIATE_DATA],
        "name": "X_true"
    },

    # Model Variables
    Constants.OBSERVED_OUTCOME_VAR_NAME: {
        "accessor": lambda data: data[
            MetricData.OBSERVED_OUTCOME_DATA][
            Constants.OBSERVED_OUTCOME_VAR_NAME],
        "name": Constants.OBSERVED_OUTCOME_VAR_NAME
    },

    Constants.POTENTIAL_OUTCOME_WITHOUT_TREATMENT_VAR_NAME: {
        "accessor": lambda data: data[
            MetricData.ORACLE_OUTCOME_DATA][
            Constants.POTENTIAL_OUTCOME_WITHOUT_TREATMENT_VAR_NAME],
        "name": Constants.POTENTIAL_OUTCOME_WITHOUT_TREATMENT_VAR_NAME
    },

    Constants.POTENTIAL_OUTCOME_WITH_TREATMENT_VAR_NAME: {
        "accessor": lambda data: data[
            MetricData.ORACLE_OUTCOME_DATA][
            Constants.POTENTIAL_OUTCOME_WITH_TREATMENT_VAR_NAME],
        "name": Constants.POTENTIAL_OUTCOME_WITH_TREATMENT_VAR_NAME
    },

    Constants.TREATMENT_EFFECT_VAR_NAME: {
        "accessor": lambda data: data[
            MetricData.ORACLE_OUTCOME_DATA][
            Constants.TREATMENT_EFFECT_VAR_NAME],
        "name": "TE"
    },

    Constants.TREATMENT_ASSIGNMENT_LOGIT_VAR_NAME: {
        "accessor": lambda data: data[
            MetricData.ORACLE_OUTCOME_DATA][
            Constants.TREATMENT_ASSIGNMENT_LOGIT_VAR_NAME],
        "name": "Treat Logit"
    },

    Constants.PROPENSITY_SCORE_VAR_NAME: {
        "accessor": lambda data: data[
            MetricData.ORACLE_OUTCOME_DATA][
            Constants.PROPENSITY_SCORE_VAR_NAME],
        "name": "P-score"
    },

    Constants.TREATMENT_ASSIGNMENT_VAR_NAME: {
        "accessor": lambda data: data[
            MetricData.OBSERVED_OUTCOME_DATA][
            Constants.TREATMENT_ASSIGNMENT_VAR_NAME],
        "name": "T"
    }
}

# TODO: add metrics for new params.
# Specify the metrics and the measures for each metric.
# Each metric has a list of measures which are defined
# by a calculation function and the arguments to be supplied
# to it.
metrics = {
    MetricNames.OUTCOME_NONLINEARITY: [
        {
            "function": MetricFunctions.LINEAR_R2,
            "args": {
                "X": MetricData.OBSERVED_COVARIATE_DATA,
                "y": Constants.OBSERVED_OUTCOME_VAR_NAME
            },
            "name": "Lin r2(X_obs, Y)"
        },
        {
            "function": MetricFunctions.LINEAR_R2,
            "args": {
                "X": MetricData.ORACLE_COVARIATE_DATA,
                "y": Constants.OBSERVED_OUTCOME_VAR_NAME
            },
            "name": "Lin r2(X_true, Y)"
        },
        {
            "function": MetricFunctions.LINEAR_R2,
            "args": {
                "X": MetricData.OBSERVED_COVARIATE_DATA,
                "y": Constants.POTENTIAL_OUTCOME_WITH_TREATMENT_VAR_NAME
            },
            "name": "Lin r2(X_obs, Y1)"
        },
        {
            "function": MetricFunctions.LINEAR_R2,
            "args": {
                "X": MetricData.OBSERVED_COVARIATE_DATA,
                "y": Constants.POTENTIAL_OUTCOME_WITHOUT_TREATMENT_VAR_NAME
            },
            "name": "Lin r2(X_obs, Y0)"
        },
        {
            "function": MetricFunctions.LINEAR_R2,
            "args": {
                "X": MetricData.ORACLE_COVARIATE_DATA,
                "y": Constants.POTENTIAL_OUTCOME_WITH_TREATMENT_VAR_NAME
            },
            "name": "Lin r2(X_true, Y1)"
        },
        {
            "function": MetricFunctions.LINEAR_R2,
            "args": {
                "X": MetricData.ORACLE_COVARIATE_DATA,
                "y": Constants.POTENTIAL_OUTCOME_WITHOUT_TREATMENT_VAR_NAME
            },
            "name": "Lin r2(X_true, Y0)"
        },
        {
            "function": MetricFunctions.LINEAR_R2,
            "args": {
                "X": MetricData.OBSERVED_COVARIATE_DATA,
                "y": Constants.TREATMENT_EFFECT_VAR_NAME
            },
            "name": "Lin r2(X_obs, TE)"
        },
        {
            "function": MetricFunctions.LINEAR_R2,
            "args": {
                "X": MetricData.ORACLE_COVARIATE_DATA,
                "y": Constants.TREATMENT_EFFECT_VAR_NAME
            },
            "name": "Lin r2(X_true, TE)"
        }
    ],

    MetricNames.TREATMENT_NONLINEARITY: [
        {
            "function": MetricFunctions.LOGISTIC_R2,
            "args": {
                "X": MetricData.OBSERVED_COVARIATE_DATA,
                "y": Constants.TREATMENT_ASSIGNMENT_VAR_NAME
            },
            "name": "Log r2(X_obs, T)"
        },
        {
            "function": MetricFunctions.LINEAR_R2,
            "args": {
                "X": MetricData.OBSERVED_COVARIATE_DATA,
                "y": Constants.TREATMENT_ASSIGNMENT_LOGIT_VAR_NAME
            },
            "name": "Lin r2(X_obs, Treat Logit)"
        }
    ],

    MetricNames.PERCENT_TREATED: [
        {
            "function": MetricFunctions.PERCENT,
            "args": {
                "x": Constants.TREATMENT_ASSIGNMENT_VAR_NAME,
                "value": 1
            },
            "name": "Percent(T==1)"
        }
    ],

    MetricNames.OVERLAP: [
        {
            "function": MetricFunctions.NN_CF_MAHALA_DIST,
            "args": {
                "covariates": MetricData.OBSERVED_COVARIATE_DATA,
                "treatment_status": Constants.TREATMENT_ASSIGNMENT_VAR_NAME
            },
            "name": "NN dist X_obs: T=1<->T=0"
        },
        {
            "function": MetricFunctions.NN_CF_MAHALA_DIST,
            "args": {
                "covariates": MetricData.ORACLE_COVARIATE_DATA,
                "treatment_status": Constants.TREATMENT_ASSIGNMENT_VAR_NAME
            },
            "name": "NN dist X_true: T=1<->T=0"
        }
    ],

    MetricNames.BALANCE: [
        {
            "function": MetricFunctions.L2_MEAN_DIST,
            "args": {
                "covariates": MetricData.ORACLE_COVARIATE_DATA,
                "treatment_status": Constants.TREATMENT_ASSIGNMENT_VAR_NAME
            },
            "name": "Mean dist X_true: T=1<->T=0"
        },
        {
            "function": MetricFunctions.WASS_DIST,
            "args": {
                "covariates": MetricData.ORACLE_COVARIATE_DATA,
                "treatment_status": Constants.TREATMENT_ASSIGNMENT_VAR_NAME
            },
            "name": "Wass dist X_true: T=1<->T=0"
        },
        {
            "function": MetricFunctions.WASS_DIST,
            "args": {
                "covariates": MetricData.OBSERVED_COVARIATE_DATA,
                "treatment_status": Constants.TREATMENT_ASSIGNMENT_VAR_NAME
            },
            "name": "Wass dist X_obs: T=1<->T=0"
        },
        {
            "function": MetricFunctions.NAIVE_TE,
            "args": {
                "TE": Constants.TREATMENT_EFFECT_VAR_NAME,
                "observed_outcome": Constants.OBSERVED_OUTCOME_VAR_NAME,
                "treatment_status": Constants.TREATMENT_ASSIGNMENT_VAR_NAME
            },
            "name": "Naive TE"
        }
    ],

    MetricNames.ALIGNMENT: [
        {
            "function": MetricFunctions.LINEAR_R2,
            "args": {
                "X": Constants.OBSERVED_OUTCOME_VAR_NAME,
                "y": Constants.TREATMENT_ASSIGNMENT_LOGIT_VAR_NAME
            },
            "name": "Lin r2(Y, Treat Logit)"
        },
        {
            "function": MetricFunctions.LINEAR_R2,
            "args": {
                "X": Constants.POTENTIAL_OUTCOME_WITHOUT_TREATMENT_VAR_NAME,
                "y": Constants.TREATMENT_ASSIGNMENT_LOGIT_VAR_NAME
            },
            "name": "Lin r2(Y0, Treat Logit)"
        }
    ],

    MetricNames.TE_HETEROGENEITY: [
        {
            "function": MetricFunctions.STD_RATIO,
            "args": {
                "x1": Constants.TREATMENT_EFFECT_VAR_NAME,
                "x2": Constants.OBSERVED_OUTCOME_VAR_NAME
            },
            "name": "std(TE)/std(Y)"
        }
    ]
}

def _get_arg_value(arg_val_name, data):
    if arg_val_name in data_accessors:
        return data_accessors[arg_val_name]["accessor"](data)
    else:
        return arg_val_name

# TODO: accept input data in a more reasonable format.
def calculate_data_metrics(
    observed_covariate_data,
    observed_outcome_data,
    oracle_covariate_data,
    oracle_outcome_data,
    observation_spec = None):
    '''
    Given a set of generated data, calculate all metrics
    in the observation spec.
    '''

    data = {
        MetricData.OBSERVED_COVARIATE_DATA: observed_covariate_data,
        MetricData.OBSERVED_OUTCOME_DATA: observed_outcome_data,
        MetricData.ORACLE_COVARIATE_DATA: oracle_covariate_data,
        MetricData.ORACLE_OUTCOME_DATA: oracle_outcome_data
    }

    metric_results = {}
    for metric, metric_measures in metrics.items():
        if (observation_spec is not None) and (metric not in observation_spec):
            continue # not in observation specs.

        metric_results[metric] = {}

        for measure in metric_measures:
            if measure["name"] not in observation_spec[metric]:
                 continue # not in observation specs.

            func_name = measure["function"]
            func = metric_functions[func_name]

            # Assemble the argument values by fetching the relevant portions
            # of the data
            args = dict([(arg_name, _get_arg_value(arg_val_name, data))
                for arg_name, arg_val_name in measure["args"].items()])

            # Call the metric function with the args.
            res = func(**args)

            # Store results.
            measure_name = measure["name"]
            metric_results[metric][f"{measure_name}"] = res

    return metric_results
