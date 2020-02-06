"""
This submodule contains utility functions that are used during both DGP sampling and the sampling of data from DGPs. The functions in this module may be useful for users writing their own concrete DGPs.
"""
import numpy as np
import sympy as sp
import pandas as pd
from functools import partial

from ..constants import Constants

def select_objects_given_probability(objects_to_sample, selection_probability):
    """Samples objects from `objects_to_sample` based on `selection_probability`.

    Args:
        objects_to_sample (list): List of objects to sample.
        selection_probability (list or float): The probability with which to sample objects from `objects_to_sample`. If list, this should be the same length as the list in `objects_to_sample` and will be the per-object selection probability. If float, then this is the uniform selection probability for all objects. The value/s should be between 0 and 1.

    Returns:
        tuple: The first entry is a :class:`numpy.ndarray` of the selected objects. The second entry is a :class:`numpy.ndarray` the same length as `objects_to_sample` with a ``1`` in the place of selected items and a ``0`` in the place of non-selected items.

    Examples
        >>> select_objects_given_probability(["a", "b", "c"], [0.5, 0.1, 0.001])
        (["a"], [1, 0, 0])
    """
    objects_to_sample = np.array(objects_to_sample)

    flat = len(objects_to_sample.shape) == 1
    if flat:
        objects_to_sample = objects_to_sample.reshape((-1, 1))

    selections = np.random.uniform(size=objects_to_sample.shape[0]) < selection_probability
    selected = objects_to_sample[selections, :]
    if flat:
        selected = selected.flatten()
    return selected, selections

def evaluate_expression(expression, data):
    """Evaluates the Sympy expression in `expression` using the :class:`pandas.DataFrame` in `data` to fill in the value of all the variables in the expression. The expression is evaluated once for each row of the DataFrame.

    Args:
        expression (Sympy Expression): A Sympy expression with variables that are a subset of the variables in columns data.
        data (:class:`~pandas.DataFrame`): A DataFrame containing observations of the variables in the expression. The names of the columns must match the names of the symbols in the expression.

    Returns:
        :class:`~numpy.ndarray`: An array of expression values corresponding to the rows of the `data`.
    """
    free_symbols = getattr(expression, "free_symbols", None)
    if free_symbols is not None:
        free_symbols = list(free_symbols)

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
    else:
        # No free symbols, return expression itself.
        return expression

def initialize_expression_constants(
    constants_sampling_distro, expressions,
    constant_symbols=Constants.DGPSampling.SUBFUNCTION_CONSTANT_SYMBOLS):
    """Initialize the constants in the expressions in `expressions` by sampling from `constants_sampling_distro`.

    Args:
        constants_sampling_distro (function): A function which produces `n` samples from some distribution over real values when called using a size keyword argument as in ``constants_sampling_distro(size=n)``.
        expressions (list): A list of Sympy expressions in which the constant symbols from `constant_symbols` appears. These are initialized to the values sampled from `constants_sampling_distro`.
        constant_symbols (list): A list of Sympy symbols which are constants to be initialized. Defaults to ``{sympy.abc.a, sympy.abc.c}``.

    Returns:
        list: A list of the sympy expressions from `expressions` with the constant symbols from `constant_symbols` randomly intialized.

    Examples
        >>> from sympy.abc import a, x
        >>> import numpy as np
        >>> initialize_expression_constants(np.random.normal, [a*x], [a])
        0.1*x
        >>> initialize_expression_constants(np.random.normal, [a*x], [a])
        -0.3*x
    """

    initialized_expressions = []

    for expression in expressions:
        # Find the free symbols which are in the constant symbols arg.
        constants_to_initialize = \
            constant_symbols.intersection(expression.free_symbols)

        initialized_expressions.append(
            # Init expression
            expression.subs(
                # enumerable of (symbol, val) tuples
                zip(constants_to_initialize,
                    constants_sampling_distro(size=len(constants_to_initialize))
                    )
                )
            )

    return initialized_expressions
