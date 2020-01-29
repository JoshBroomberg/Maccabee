import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from scipy.spatial.distance import cdist
import ot
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from .constants import Constants
from .utilities import extract_treat_and_control_data
from .parameters import build_parameters_from_axis_levels
from .data_generation import DataGeneratingProcessSampler

### Metric functions

# These are the functions which are used
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


AnalysisMetricFunctions = Constants.AnalysisMetricFunctions
AnalysisMetricData = Constants.AnalysisMetricData
AxisNames = Constants.AxisNames

### NOTE: the code below is complex because it is designed
# to allow new metrics to be created very easily. Feel free
# to skip down to the metric definitions.

# Map the function names to function callables.
metric_functions = {
    AnalysisMetricFunctions.LINEAR_R2: linear_regression_r2,
    AnalysisMetricFunctions.LOGISTIC_R2: logistic_regression_r2,
    AnalysisMetricFunctions.PERCENT: percent,
    AnalysisMetricFunctions.L2_MEAN_DIST: l2_distance_between_means,
    AnalysisMetricFunctions.NN_CF_MAHALA_DIST: mean_mahalanobis_between_nearest_counterfactual,
    AnalysisMetricFunctions.STD_RATIO: standard_deviation_ratio,
    AnalysisMetricFunctions.WASS_DIST: wasserstein,
    AnalysisMetricFunctions.NAIVE_TE: naive_TE_estimate_error
}

# TODO: remove names and simplify access.
# Specify ways to access each input variable.
data_accessors = {
    # Covariate Data
    AnalysisMetricData.OBSERVED_COVARIATE_DATA: {
        "accessor": lambda data: data[AnalysisMetricData.OBSERVED_COVARIATE_DATA],
        "name": "X_obs"
    },
    AnalysisMetricData.TRANSFORMED_COVARIATE_DATA: {
        "accessor": lambda data: data[AnalysisMetricData.TRANSFORMED_COVARIATE_DATA],
        "name": "X_true"
    },

    # Model Variables
    Constants.OBSERVED_OUTCOME_NAME: {
        "accessor": lambda data: data[
            AnalysisMetricData.OBSERVED_OUTCOME_DATA][
            Constants.OBSERVED_OUTCOME_NAME],
        "name": Constants.OBSERVED_OUTCOME_NAME
    },

    Constants.POTENTIAL_OUTCOME_WITHOUT_TREATMENT_NAME: {
        "accessor": lambda data: data[
            AnalysisMetricData.ORACLE_OUTCOME_DATA][
            Constants.POTENTIAL_OUTCOME_WITHOUT_TREATMENT_NAME],
        "name": Constants.POTENTIAL_OUTCOME_WITHOUT_TREATMENT_NAME
    },

    Constants.POTENTIAL_OUTCOME_WITH_TREATMENT_NAME: {
        "accessor": lambda data: data[
            AnalysisMetricData.ORACLE_OUTCOME_DATA][
            Constants.POTENTIAL_OUTCOME_WITH_TREATMENT_NAME],
        "name": Constants.POTENTIAL_OUTCOME_WITH_TREATMENT_NAME
    },

    Constants.TREATMENT_EFFECT_NAME: {
        "accessor": lambda data: data[
            AnalysisMetricData.ORACLE_OUTCOME_DATA][
            Constants.TREATMENT_EFFECT_NAME],
        "name": "TE"
    },

    Constants.PROPENSITY_LOGIT_NAME: {
        "accessor": lambda data: data[
            AnalysisMetricData.ORACLE_OUTCOME_DATA][
            Constants.PROPENSITY_LOGIT_NAME],
        "name": "Treat Logit"
    },

    Constants.PROPENSITY_SCORE_NAME: {
        "accessor": lambda data: data[
            AnalysisMetricData.ORACLE_OUTCOME_DATA][
            Constants.PROPENSITY_SCORE_NAME],
        "name": "P-score"
    },

    Constants.TREATMENT_ASSIGNMENT_NAME: {
        "accessor": lambda data: data[
            AnalysisMetricData.OBSERVED_OUTCOME_DATA][
            Constants.TREATMENT_ASSIGNMENT_NAME],
        "name": "T"
    }
}

# TODO: add axes for new params.

# This dict specifies the measured axes and the metrics for each axis.
# Each axis has a list of metrics which are defined
# by a metric calculation function and the arguments to be supplied to it
# as well as a unique name.
axes = {
    AxisNames.OUTCOME_NONLINEARITY: [
        {
            "function": AnalysisMetricFunctions.LINEAR_R2,
            "args": {
                "X": AnalysisMetricData.OBSERVED_COVARIATE_DATA,
                "y": Constants.OBSERVED_OUTCOME_NAME
            },
            "name": "Lin r2(X_obs, Y)"
        },
        {
            "function": AnalysisMetricFunctions.LINEAR_R2,
            "args": {
                "X": AnalysisMetricData.TRANSFORMED_COVARIATE_DATA,
                "y": Constants.OBSERVED_OUTCOME_NAME
            },
            "name": "Lin r2(X_true, Y)"
        },
        {
            "function": AnalysisMetricFunctions.LINEAR_R2,
            "args": {
                "X": AnalysisMetricData.OBSERVED_COVARIATE_DATA,
                "y": Constants.POTENTIAL_OUTCOME_WITH_TREATMENT_NAME
            },
            "name": "Lin r2(X_obs, Y1)"
        },
        {
            "function": AnalysisMetricFunctions.LINEAR_R2,
            "args": {
                "X": AnalysisMetricData.OBSERVED_COVARIATE_DATA,
                "y": Constants.POTENTIAL_OUTCOME_WITHOUT_TREATMENT_NAME
            },
            "name": "Lin r2(X_obs, Y0)"
        },
        {
            "function": AnalysisMetricFunctions.LINEAR_R2,
            "args": {
                "X": AnalysisMetricData.TRANSFORMED_COVARIATE_DATA,
                "y": Constants.POTENTIAL_OUTCOME_WITH_TREATMENT_NAME
            },
            "name": "Lin r2(X_true, Y1)"
        },
        {
            "function": AnalysisMetricFunctions.LINEAR_R2,
            "args": {
                "X": AnalysisMetricData.TRANSFORMED_COVARIATE_DATA,
                "y": Constants.POTENTIAL_OUTCOME_WITHOUT_TREATMENT_NAME
            },
            "name": "Lin r2(X_true, Y0)"
        },
        {
            "function": AnalysisMetricFunctions.LINEAR_R2,
            "args": {
                "X": AnalysisMetricData.OBSERVED_COVARIATE_DATA,
                "y": Constants.TREATMENT_EFFECT_NAME
            },
            "name": "Lin r2(X_obs, TE)"
        },
        {
            "function": AnalysisMetricFunctions.LINEAR_R2,
            "args": {
                "X": AnalysisMetricData.TRANSFORMED_COVARIATE_DATA,
                "y": Constants.TREATMENT_EFFECT_NAME
            },
            "name": "Lin r2(X_true, TE)"
        }
    ],

    AxisNames.TREATMENT_NONLINEARITY: [
        {
            "function": AnalysisMetricFunctions.LOGISTIC_R2,
            "args": {
                "X": AnalysisMetricData.OBSERVED_COVARIATE_DATA,
                "y": Constants.TREATMENT_ASSIGNMENT_NAME
            },
            "name": "Log r2(X_obs, T)"
        },
        {
            "function": AnalysisMetricFunctions.LINEAR_R2,
            "args": {
                "X": AnalysisMetricData.OBSERVED_COVARIATE_DATA,
                "y": Constants.PROPENSITY_LOGIT_NAME
            },
            "name": "Lin r2(X_obs, Treat Logit)"
        }
    ],

    AxisNames.PERCENT_TREATED: [
        {
            "function": AnalysisMetricFunctions.PERCENT,
            "args": {
                "x": Constants.TREATMENT_ASSIGNMENT_NAME,
                "value": 1
            },
            "name": "Percent(T==1)"
        }
    ],

    AxisNames.OVERLAP: [
        {
            "function": AnalysisMetricFunctions.NN_CF_MAHALA_DIST,
            "args": {
                "covariates": AnalysisMetricData.OBSERVED_COVARIATE_DATA,
                "treatment_status": Constants.TREATMENT_ASSIGNMENT_NAME
            },
            "name": "NN dist X_obs: T=1<->T=0"
        },
        {
            "function": AnalysisMetricFunctions.NN_CF_MAHALA_DIST,
            "args": {
                "covariates": AnalysisMetricData.TRANSFORMED_COVARIATE_DATA,
                "treatment_status": Constants.TREATMENT_ASSIGNMENT_NAME
            },
            "name": "NN dist X_true: T=1<->T=0"
        }
    ],

    AxisNames.BALANCE: [
        {
            "function": AnalysisMetricFunctions.L2_MEAN_DIST,
            "args": {
                "covariates": AnalysisMetricData.TRANSFORMED_COVARIATE_DATA,
                "treatment_status": Constants.TREATMENT_ASSIGNMENT_NAME
            },
            "name": "Mean dist X_true: T=1<->T=0"
        },
        {
            "function": AnalysisMetricFunctions.WASS_DIST,
            "args": {
                "covariates": AnalysisMetricData.TRANSFORMED_COVARIATE_DATA,
                "treatment_status": Constants.TREATMENT_ASSIGNMENT_NAME
            },
            "name": "Wass dist X_true: T=1<->T=0"
        },
        {
            "function": AnalysisMetricFunctions.WASS_DIST,
            "args": {
                "covariates": AnalysisMetricData.OBSERVED_COVARIATE_DATA,
                "treatment_status": Constants.TREATMENT_ASSIGNMENT_NAME
            },
            "name": "Wass dist X_obs: T=1<->T=0"
        },
        {
            "function": AnalysisMetricFunctions.NAIVE_TE,
            "args": {
                "TE": Constants.TREATMENT_EFFECT_NAME,
                "observed_outcome": Constants.OBSERVED_OUTCOME_NAME,
                "treatment_status": Constants.TREATMENT_ASSIGNMENT_NAME
            },
            "name": "Naive TE"
        }
    ],

    AxisNames.ALIGNMENT: [
        {
            "function": AnalysisMetricFunctions.LINEAR_R2,
            "args": {
                "X": Constants.OBSERVED_OUTCOME_NAME,
                "y": Constants.PROPENSITY_LOGIT_NAME
            },
            "name": "Lin r2(Y, Treat Logit)"
        },
        {
            "function": AnalysisMetricFunctions.LINEAR_R2,
            "args": {
                "X": Constants.POTENTIAL_OUTCOME_WITHOUT_TREATMENT_NAME,
                "y": Constants.PROPENSITY_LOGIT_NAME
            },
            "name": "Lin r2(Y0, Treat Logit)"
        }
    ],

    AxisNames.TE_HETEROGENEITY: [
        {
            "function": AnalysisMetricFunctions.STD_RATIO,
            "args": {
                "x1": Constants.TREATMENT_EFFECT_NAME,
                "x2": Constants.OBSERVED_OUTCOME_NAME
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

def analyze_axis_metrics_across_levels(
    axes_and_metrics, levels, data_source_generator,
    n_trials=20):
    '''
    Collect values for the given metrics and measures
    across all possible settings for each metric. Uses
    the gather_metric_measures_for_given_params to generate
    the metric measure values.
    '''

    results = defaultdict(lambda: defaultdict(dict))

    # Run for each given metric and set of measures
    for axis, metrics in axes_and_metrics.items():
        print(f"\nRunning for {axis}. Level: ", end=" ")

        # Construct observation list of measures.
        observation_spec = { axis: metrics }

        # Run trials at all levels of metric.
        for level in levels:
            print(level, end=" ")
            dgp_params = build_parameters_from_axis_levels({
                axis: level
            })

            res = gather_axis_metrics_for_given_params(
                    dgp_params, observation_spec, data_source_generator,
                    n_trials=n_trials)

            for metric, values in res[axis].items():
                results[axis][metric][level] = values

    return results

def gather_axis_metrics_for_given_params(
        dgp_params, observation_spec,
        data_source_generator,
        n_trials=10, verbose=False):

    '''
    Create n_trials datasets by sampling data generating processes according
    to the given dgp_params and data source parameters (n_covars, n_obs).

    Collect the metrics given in observation spec.
    '''
    results = defaultdict(lambda: defaultdict(list))
    for i in range(n_trials):
        if verbose:
            print("Trials run:", i+1)
        results

        # Generate data
        data_source = data_source_generator()

        # Sample DGP
        dgp_sampler = DataGeneratingProcessSampler(
            parameters=dgp_params, data_source=data_source)
        dgp = dgp_sampler.sample_dgp()
        data_set = dgp.generate_dataset()

        # Calculate metrics
        metrics = calculate_data_axis_metrics(data_set, observation_spec)

        # Build results
        for metric, measures in observation_spec.items():
            for measure in measures:
                res = metrics[metric][measure]
                if res is not None:
                    results[metric][measure].append(res)

    if verbose:
        for metric, measures in results.items():
            for measure, result_data in measures.items():
                print(f"{metric} {measure}:")
                print("min", round(np.min(result_data), 3), end=" ")
                print("mean:", round(np.mean(result_data), 3), end=" ")
                print("max", round(np.max(result_data), 3))
                print("-------------\n\n")

    return results

def calculate_data_axis_metrics(
    data_set,
    observation_spec = None):
    '''
    Given a set of generated data, calculate all metrics
    in the observation spec.
    '''

    data = {
        AnalysisMetricData.OBSERVED_COVARIATE_DATA: data_set.observed_covariate_data,
        AnalysisMetricData.OBSERVED_OUTCOME_DATA: data_set.observed_outcome_data,
        AnalysisMetricData.TRANSFORMED_COVARIATE_DATA: data_set.transformed_covariate_data,
        AnalysisMetricData.ORACLE_OUTCOME_DATA: data_set.oracle_outcome_data
    }

    axis_metric_results = {}
    for axis, axis_metrics in axes.items():
        if (observation_spec is not None) and (axis not in observation_spec):
            continue # this axis not in observation specs.

        axis_metric_results[axis] = {}

        for metric in axis_metrics:
            if (observation_spec is not None) and (metric["name"] not in observation_spec[axis]):
                 continue # this metric not in observation specs.

            func_name = metric["function"]
            func = metric_functions[func_name]

            # Assemble the argument values by fetching the relevant portions
            # of the data
            args = dict([(arg_name, _get_arg_value(arg_val_name, data))
                for arg_name, arg_val_name in metric["args"].items()])

            # Call the metric function with the args.
            res = func(**args)

            # Store results.
            metric_name = metric["name"]
            axis_metric_results[axis][f"{metric_name}"] = res

    return axis_metric_results

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
