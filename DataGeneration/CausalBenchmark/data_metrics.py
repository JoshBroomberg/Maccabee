import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from scipy.spatial.distance import cdist
import ot
from .constants import Constants
import pandas as pd
# pip install POT

def _extract_treat_and_control_covariates(covariates, treatment_status):
    X_treated = covariates[(treatment_status==1).to_numpy()]
    X_control = covariates[(treatment_status==0).to_numpy()]
    return X_treated, X_control

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
    x = np.array(x)
    return 100*np.sum((x == value).astype(int))/len(x)

def l2_distance_between_means(covariates, treatment_status):
    X_treated, X_control = _extract_treat_and_control_covariates(
        covariates, treatment_status)

    X_treated_mean = np.mean(X_treated, axis=0)
    X_control_mean = np.mean(X_control, axis=0)

    return np.linalg.norm(X_treated_mean - X_control_mean)

def mean_mahalanobis_between_nearest_counterfactual(covariates, treatment_status):
    X_treated, X_control = _extract_treat_and_control_covariates(
        covariates, treatment_status)

    distance_matrix = cdist(X_treated, X_control, "mahalanobis")
    np.nan_to_num(distance_matrix, copy=False, nan=np.inf)

    return np.mean(np.min(distance_matrix, axis=1))

def standard_deviation_ratio(x1, x2):
    return np.std(x1)/np.std(x2)

def wasserstein(covariates, treatment_status):
    X_treated, X_control = _extract_treat_and_control_covariates(
        covariates, treatment_status)

    num_treated, num_control = len(X_treated), len(X_control)

    a = np.ones(num_treated)/num_treated
    b = np.ones(num_control)/num_control

    M = ot.dist(X_treated, X_control)
    M /= M.max()
    lambd = 1e-3

    return ot.sinkhorn2(a, b, M, lambd)[0]

LINEAR_R2 = "Linear r2"
LOGISTIC_R2 = "Logistic r2"
PERCENT = "Percent"
L2_MEAN_DIST = "L2 dist between means"
NN_CF_MAHALA_DIST = "Mahalanobis distance to nearest counterfactual"
STD_RATIO = "Std Ratio"
WASS_DIST = "Wasserstein Distance"

metric_functions = {
    Constants.LINEAR_R2: linear_regression_r2,
    Constants.LOGISTIC_R2: logistic_regression_r2,
    Constants.PERCENT: percent,
    Constants.L2_MEAN_DIST: l2_distance_between_means,
    Constants.NN_CF_MAHALA_DIST: mean_mahalanobis_between_nearest_counterfactual,
    Constants.STD_RATIO: standard_deviation_ratio,
    Constants.WASS_DIST: wasserstein
}

data_accessors = {
    Constants.OBSERVED_COVARIATE_DATA: {
        "accessor": lambda data: data[Constants.OBSERVED_COVARIATE_DATA],
        "name": "X_obs"
    },
    Constants.ORACLE_COVARIATE_DATA: {
        "accessor": lambda data: data[Constants.ORACLE_COVARIATE_DATA],
        "name": "X_true"
    },

    Constants.OBSERVED_OUTCOME_VAR_NAME: {
        "accessor": lambda data: data[
            Constants.OBSERVED_OUTCOME_DATA][
            Constants.OBSERVED_OUTCOME_VAR_NAME],
        "name": Constants.OBSERVED_OUTCOME_VAR_NAME
    },

    Constants.POTENTIAL_OUTCOME_WITHOUT_TREATMENT_VAR_NAME: {
        "accessor": lambda data: data[
            Constants.ORACLE_OUTCOME_DATA][
            Constants.POTENTIAL_OUTCOME_WITHOUT_TREATMENT_VAR_NAME],
        "name": Constants.POTENTIAL_OUTCOME_WITHOUT_TREATMENT_VAR_NAME
    },

    Constants.POTENTIAL_OUTCOME_WITH_TREATMENT_VAR_NAME: {
        "accessor": lambda data: data[
            Constants.ORACLE_OUTCOME_DATA][
            Constants.POTENTIAL_OUTCOME_WITH_TREATMENT_VAR_NAME],
        "name": Constants.POTENTIAL_OUTCOME_WITH_TREATMENT_VAR_NAME
    },

    Constants.TREATMENT_EFFECT_VAR_NAME: {
        "accessor": lambda data: data[
            Constants.ORACLE_OUTCOME_DATA][
            Constants.TREATMENT_EFFECT_VAR_NAME],
        "name": "TE"
    },

    Constants.TREATMENT_ASSIGNMENT_LOGIT_VAR_NAME: {
        "accessor": lambda data: data[
            Constants.ORACLE_OUTCOME_DATA][
            Constants.TREATMENT_ASSIGNMENT_LOGIT_VAR_NAME],
        "name": "Treat Logit"
    },

    Constants.PROPENSITY_SCORE_VAR_NAME: {
        "accessor": lambda data: data[
            Constants.ORACLE_OUTCOME_DATA][
            Constants.PROPENSITY_SCORE_VAR_NAME],
        "name": "P-score"
    },

    Constants.TREATMENT_ASSIGNMENT_VAR_NAME: {
        "accessor": lambda data: data[
            Constants.OBSERVED_OUTCOME_DATA][
            Constants.TREATMENT_ASSIGNMENT_VAR_NAME],
        "name": "T"
    }
}

metrics = {
    "outcome non-linearity": [
        {
            "function": Constants.LINEAR_R2,
            "args": {
                "X": Constants.OBSERVED_COVARIATE_DATA,
                "y": Constants.OBSERVED_OUTCOME_VAR_NAME
            }
        },
        {
            "function": Constants.LINEAR_R2,
            "args": {
                "X": Constants.ORACLE_COVARIATE_DATA,
                "y": Constants.OBSERVED_OUTCOME_VAR_NAME
            }
        },
        {
            "function": Constants.LINEAR_R2,
            "args": {
                "X": Constants.OBSERVED_COVARIATE_DATA,
                "y": Constants.POTENTIAL_OUTCOME_WITH_TREATMENT_VAR_NAME
            }
        },
        {
            "function": Constants.LINEAR_R2,
            "args": {
                "X": Constants.OBSERVED_COVARIATE_DATA,
                "y": Constants.POTENTIAL_OUTCOME_WITHOUT_TREATMENT_VAR_NAME
            }
        },
        {
            "function": Constants.LINEAR_R2,
            "args": {
                "X": Constants.ORACLE_COVARIATE_DATA,
                "y": Constants.POTENTIAL_OUTCOME_WITH_TREATMENT_VAR_NAME
            }
        },
        {
            "function": Constants.LINEAR_R2,
            "args": {
                "X": Constants.ORACLE_COVARIATE_DATA,
                "y": Constants.POTENTIAL_OUTCOME_WITHOUT_TREATMENT_VAR_NAME
            }
        },
        {
            "function": Constants.LINEAR_R2,
            "args": {
                "X": Constants.OBSERVED_COVARIATE_DATA,
                "y": Constants.TREATMENT_EFFECT_VAR_NAME
            }
        },
        {
            "function": Constants.LINEAR_R2,
            "args": {
                "X": Constants.ORACLE_COVARIATE_DATA,
                "y": Constants.TREATMENT_EFFECT_VAR_NAME
            }
        }
    ],

    "treatment non-linearity": [
        {
            "function": Constants.LOGISTIC_R2,
            "args": {
                "X": Constants.OBSERVED_COVARIATE_DATA,
                "y": Constants.TREATMENT_ASSIGNMENT_VAR_NAME
            }
        },
        {
            "function": Constants.LINEAR_R2,
            "args": {
                "X": Constants.OBSERVED_COVARIATE_DATA,
                "y": Constants.TREATMENT_ASSIGNMENT_LOGIT_VAR_NAME
            }
        }
    ],

    "percent": [
        {
            "function": Constants.PERCENT,
            "args": {
                "x": Constants.TREATMENT_ASSIGNMENT_VAR_NAME,
                "value": 1
            }
        }
    ],

    "overlap": [
        {
            "function": Constants.NN_CF_MAHALA_DIST,
            "args": {
                "covariates": Constants.OBSERVED_COVARIATE_DATA,
                "treatment_status": Constants.TREATMENT_ASSIGNMENT_VAR_NAME
            }
        },
        {
            "function": Constants.NN_CF_MAHALA_DIST,
            "args": {
                "covariates": Constants.ORACLE_COVARIATE_DATA,
                "treatment_status": Constants.TREATMENT_ASSIGNMENT_VAR_NAME
            }
        }
    ],

    "balance": [
        {
            "function": Constants.L2_MEAN_DIST,
            "args": {
                "covariates": Constants.ORACLE_COVARIATE_DATA,
                "treatment_status": Constants.TREATMENT_ASSIGNMENT_VAR_NAME
            }
        },
        {
            "function": Constants.WASS_DIST,
            "args": {
                "covariates": Constants.ORACLE_COVARIATE_DATA,
                "treatment_status": Constants.TREATMENT_ASSIGNMENT_VAR_NAME
            }
        },
        {
            "function": Constants.WASS_DIST,
            "args": {
                "covariates": Constants.OBSERVED_COVARIATE_DATA,
                "treatment_status": Constants.TREATMENT_ASSIGNMENT_VAR_NAME
            }
        }
    ],

    "alignment": [
        {
            "function": Constants.LINEAR_R2,
            "args": {
                "X": Constants.OBSERVED_OUTCOME_VAR_NAME,
                "y": Constants.TREATMENT_ASSIGNMENT_LOGIT_VAR_NAME
            }
        }
    ],

    "Treatment Effect Heterogeneity": [
        {
            "function": Constants.STD_RATIO,
            "args": {
                "x1": Constants.TREATMENT_EFFECT_VAR_NAME,
                "x2": Constants.OBSERVED_OUTCOME_VAR_NAME
            }
        }
    ]
}

def _get_arg_value(arg_val_name, data):
    if arg_val_name in data_accessors:
        return data_accessors[arg_val_name]["accessor"](data)
    else:
        return arg_val_name

def calculate_data_metrics(
    observed_covariate_data,
    observed_outcome_data,
    oracle_covariate_data,
    oracle_outcome_data):

    data = {
        Constants.OBSERVED_COVARIATE_DATA: observed_covariate_data,
        Constants.OBSERVED_OUTCOME_DATA: observed_outcome_data,
        Constants.ORACLE_COVARIATE_DATA: oracle_covariate_data,
        Constants.ORACLE_OUTCOME_DATA: oracle_outcome_data
    }

    dimension_results = {}
    for dimension, dimension_metrics in metrics.items():
        dimension_results[dimension] = {}

        for metric in dimension_metrics:
            func_name = metric["function"]
            func = metric_functions[func_name]

            arg_val_names = ", ".join([
                (data_accessors[arg_val_name]["name"] if arg_val_name in data_accessors else str(arg_val_name))
                for arg_val_name in metric["args"].values()])

            args = dict([(arg_name, _get_arg_value(arg_val_name, data))
                for arg_name, arg_val_name in metric["args"].items()])

            res = func(**args)

            dimension_results[dimension][f"{func_name}({arg_val_names})"] = res


    return dimension_results
