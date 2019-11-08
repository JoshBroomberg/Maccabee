import numpy as np
import sympy as sp
from .constants import Constants
import pandas as pd

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
    if hasattr(expression, 'free_symbols'):
        free_symbols = list(expression.free_symbols)
        core_func = sp.lambdify(
                free_symbols,
                expression,
                "numpy",
                dummify=False)

        free_symbol_names = [ str(sym) for sym in free_symbols ]
        wrapper_func = lambda arg_arr: core_func(*arg_arr)
        res = np.apply_along_axis(wrapper_func, 1, data[free_symbol_names].to_numpy())
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
