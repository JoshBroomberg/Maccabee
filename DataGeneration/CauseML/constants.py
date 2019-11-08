from sympy.abc import a, c, x, y, z
import sympy as sp
from pkg_resources import resource_filename
import yaml

### Define constants ###

# OPERATIONAL CONSTANTS
class Constants:
    '''
    This class defines constants which are used throughout
    the package.
    '''

    class Params:
        _SCHEMA_PATH = resource_filename(
            'CauseML', 'parameters/parameter_schema.yml')

        with open(_SCHEMA_PATH, "r") as schema_file:
            SCHEMA = yaml.safe_load(schema_file)["SCHEMA"]

        DEFAULT_SPEC_PATH = resource_filename(
            'CauseML', 'parameters/default_parameter_specification.yml')
        METRIC_LEVEL_SPEC_PATH = resource_filename(
            'CauseML', 'parameters/metric_level_parameter_specifications.yml')

        class ParamInfo:
            DESCRIPTION_KEY = "description"

            TYPE_KEY = "type"
            TYPE_NUMBER = "number"
            TYPE_BOOL = "bool"
            TYPE_DICTIONARY = "dictionary"
            TYPE_CALCULATED = "calculated"

            # Number type keys
            MIN_KEY = "min"
            MAX_KEY = "max"

            # Dict type keys
            DICT_KEYS_KEY = "required_keys"

            # Calc type keys
            EXPRESSION_KEY = "expr"

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

    # The various transforms which can be applied to covariates
    # and combinations of covariates.
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

    TRANSFORMED_COVARIATE_PREFIX = "TRANSFORMED_X"

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

    ### Metric constants ###

    # Data inputs
    class MetricData:
        OBSERVED_COVARIATE_DATA = "OBSERVED_COVARIATES"
        OBSERVED_OUTCOME_DATA = "OBSERVED_OUTCOMES"
        ORACLE_COVARIATE_DATA = "ORACLE_COVARIATES"
        ORACLE_OUTCOME_DATA = "ORACLE_OUTCOMES"

    # Functions
    class MetricFunctions:
        LINEAR_R2 = "Lin r2"
        LOGISTIC_R2 = "Log r2"
        PERCENT = "Percent"
        L2_MEAN_DIST = "mean dist"
        NN_CF_MAHALA_DIST = "NN c-factual dist"
        STD_RATIO = "Normed std"
        WASS_DIST = "Wass dist"
        NAIVE_TE = "Naive TE"

    class MetricNames:
        OUTCOME_NONLINEARITY = "OUTCOME_NONLINEARITY"
        TREATMENT_NONLINEARITY = "TREATMENT_NONLINEARITY"
        PERCENT_TREATED = "PERCENT_TREATED"
        OVERLAP = "OVERLAP"
        BALANCE = "BALANCE"
        ALIGNMENT = "ALIGNMENT"
        TE_HETEROGENEITY = "TE_HETEROGENEITY"

    class MetricLevels:
        LOW = "LOW"
        MEDIUM = "MEDIUM"
        HIGH = "HIGH"
