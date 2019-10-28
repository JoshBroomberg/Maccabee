from sympy.abc import a, c, x, y, z
import sympy as sp

# OPERATIONAL CONSTANTS
class Constants:
    LINEAR = "LINEAR"
    POLY_QUADRATIC = "POLY_QUAD"
    POLY_CUBIC = "POLY_CUBIC"
    STEP_CONSTANT = "STEP_JUMP"
    STEP_VARIABLE = "STEP_KINK"
    INTERACTION_TWO_WAY = "INTERACTION_TWO_WAY"
    INTERACTION_THREE_WAY = "INTERACTION_THREE_WAY"

    COVARIATE_SYMBOLS_KEY = "covariates"
    EXPRESSION_KEY = "expr"

    SUBFUNCTION_FORMS = {
        LINEAR: {
                COVARIATE_SYMBOLS_KEY: [x],
                EXPRESSION_KEY: c*x
        },
        POLY_QUADRATIC: {
                COVARIATE_SYMBOLS_KEY: [x],
                EXPRESSION_KEY: c*(x**2)
        },
        POLY_CUBIC: {
                COVARIATE_SYMBOLS_KEY: [x],
                EXPRESSION_KEY: c*(x**3)
        },
        STEP_CONSTANT: {
                COVARIATE_SYMBOLS_KEY: [x],
                EXPRESSION_KEY: sp.Piecewise((0, x < a), (c, True))
        },
        STEP_VARIABLE: {
                COVARIATE_SYMBOLS_KEY: [x],
                EXPRESSION_KEY: sp.Piecewise((0, x < a), (c*x, True))
        },
        INTERACTION_TWO_WAY: {
                COVARIATE_SYMBOLS_KEY: [x, y],
                EXPRESSION_KEY: c*x*y
        },
        INTERACTION_THREE_WAY: {
                COVARIATE_SYMBOLS_KEY: [x, y, z],
                EXPRESSION_KEY: c*x*y*z
        },
    }

    SUBFUNCTION_CONSTANT_SYMBOLS = {a, c}

    TREATMENT_EFFECT_SYMBOL = sp.symbols("T")
    OUTCOME_NOISE_SYMBOL = sp.symbols("NOISE")
