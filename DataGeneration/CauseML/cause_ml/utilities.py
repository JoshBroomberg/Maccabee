import numpy as np
import sympy as sp
from .constants import Constants
import pandas as pd
from functools import partial


def random_covar_matrix(dimension, correlation_deg = 0.5):
    """
    Generate random covariance matrix by approximating the random
    vine method: https://stats.stackexchange.com/questions/2746/
    how-to-efficiently-generate-random-positive-semidefinite-correlation-matrices
    """
    if not (0 <= correlation_deg <= 1):
        raise Exception("Invalid correlation_deg. Must be in [0, 1]")

    # Small K means a larger degree of correlation.
    k = min(max(1, int(dimension*(1-correlation_deg))), dimension-1)
    W = np.random.normal(loc=0, scale=1, size=(dimension, k))
    S = W@W.T + np.diag(np.random.random(dimension))
    S = np.diag(1./np.sqrt(np.diag(S))) @ S @ np.diag(1./np.sqrt(np.diag(S)))

    return S

def extract_treat_and_control_data(covariates, treatment_status):
    X_treated = covariates[(treatment_status==1).to_numpy()]
    X_control = covariates[(treatment_status==0).to_numpy()]
    return X_treated, X_control

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
    try:
        free_symbols = list(expression.free_symbols)

        expr_func = sp.lambdify(
                free_symbols,
                expression,
                modules=[
                    {
                        "amax": lambda x: np.maximum(*x),
                        "amin": lambda x: np.minimum(*x)
                    },
                    "numpy"
                ],
                dummify=True)

        column_data = [data[str(sym)] for sym in free_symbols]
        res = expr_func(*column_data)

        return res
    except AttributeError:
        # No free symbols, return expression itself.
        return expression

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
