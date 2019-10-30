import numpy as np
import sympy as sp
from .constants import Constants
import pandas as pd

def extract_treat_and_control_data(covariates, treatment_status):
    X_treated = covariates[(treatment_status==1).to_numpy()]
    X_control = covariates[(treatment_status==0).to_numpy()]
    return X_treated, X_control

def generate_random_covariates(n_covars = 20, n_observations = 1000):

    # Generate random covariates and name sequentially
    covar_data = np.random.normal(loc=0, scale=5, size=(n_observations, n_covars))
    covar_names = np.array([f"X{i}" for i in range(n_covars)])

    # Build DF
    return pd.DataFrame(
            data=covar_data,
            columns=covar_names,
            index=np.arange(n_observations))

def normalize_covariate_data(covariate_data):
    X_min = np.min(covariate_data, axis=0)
    X_max = np.max(covariate_data, axis=0)
    normalized_data = (covariate_data - X_min)/(X_max - X_min)
    scaled_data = 2*normalized_data - 1
    return scaled_data


def select_given_probability_distribution(full_list, selection_probabilities):
    full_list = np.array(full_list)

    flat = len(full_list.shape) == 1
    if flat:
        full_list = full_list.reshape((-1, 1))

    selections = np.random.uniform(size=full_list.shape[0]) < selection_probabilities
    selected = full_list[selections, :]
    if flat:
        selected = selected.flatten()
    return selected, selections

def evaluate_expression(expression, data):
    if hasattr(expression, 'free_symbols'):
        free_symbols = list(expression.free_symbols)
        func = np.vectorize(sp.lambdify(
                free_symbols,
                expression,
                "numpy",
                dummify=False))

        covar_data = [
            data[[str(symbol)]].to_numpy()
            for symbol in free_symbols
        ]

        res = func(*covar_data)
        if hasattr(res, 'flatten'):
            res = res.flatten()
    else:
        res = expression

    return res

def initialize_expression_constants(parameters, expressions):
    initialized_expressions = []

    for expression in expressions:
        constants_to_initialize = \
            Constants.SUBFUNCTION_CONSTANT_SYMBOLS.intersection(expression.free_symbols)

        initialized_expressions.append(expression.subs(
            zip(constants_to_initialize,
                parameters.sample_subfunction_constants(
                    size=len(constants_to_initialize)))))

    return initialized_expressions
