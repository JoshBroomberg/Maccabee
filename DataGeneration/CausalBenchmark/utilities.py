import numpy as np
import sympy as sp
from .constants import Constants
from .parameters import Parameters

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
    free_symbols = list(expression.free_symbols)
    func = np.vectorize(sp.lambdify(
            free_symbols,
            expression,
            "numpy",
            dummify=False))

    covar_data = [data[[str(symbol)]].to_numpy() for symbol in free_symbols]
    return func(*covar_data).flatten()

@np.vectorize
def initialize_expression_constants(expression):
    constants_to_initialize = \
        Constants.SUBFUNCTION_CONSTANT_SYMBOLS.intersection(expression.free_symbols)

    return expression.subs(
        zip(constants_to_initialize,
            Parameters.sample_subfunction_constants(size=len(constants_to_initialize))))
