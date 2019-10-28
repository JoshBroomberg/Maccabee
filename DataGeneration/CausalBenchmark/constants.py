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

    SUBFUNCTION_CONSTANT_SYMBOLS = {a, c}
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

    # How much of the covariate data to use when normalizing functions.
    NORMALIZATION_DATA_SAMPLE_FRACTION = 1

    TREATMENT_ASSIGNMENT_LOGIT_VAR_NAME = "logit(P(T|X))"
    PROPENSITY_SCORE_VAR_NAME = "P(T|X)"
    TREATMENT_ASSIGNMENT_VAR_NAME = "T"
    TREATMENT_ASSIGNMENT_SYMBOL = sp.symbols(TREATMENT_ASSIGNMENT_VAR_NAME)

    OBSERVED_OUTCOME_VAR_NAME = "Y"
    POTENTIAL_OUTCOME_WITHOUT_TREATMENT_VAR_NAME = "Y0"
    POTENTIAL_OUTCOME_WITH_TREATMENT_VAR_NAME = "Y1"
    TREATMENT_EFFECT_VAR_NAME = "TE"

    OUTCOME_NOISE_VAR_NAME = "NOISE(Y)"
    OUTCOME_NOISE_SYMBOL = sp.symbols(OUTCOME_NOISE_VAR_NAME)
