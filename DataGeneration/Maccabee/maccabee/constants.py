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
            'maccabee', 'parameters/parameter_schema.yml')

        with open(_SCHEMA_PATH, "r") as schema_file:
            SCHEMA = yaml.safe_load(schema_file)["SCHEMA"]

        DEFAULT_SPEC_PATH = resource_filename(
            'maccabee', 'parameters/default_parameter_specification.yml')
        METRIC_LEVEL_SPEC_PATH = resource_filename(
            'maccabee', 'parameters/metric_level_parameter_specifications.yml')

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

    # TODO: Move all of the below into a namespace equivalent to
    # other constants.

    LINEAR = "LINEAR"
    POLY_QUADRATIC = "POLY_QUAD"
    POLY_CUBIC = "POLY_CUBIC"
    STEP_CONSTANT = "STEP_JUMP"
    STEP_VARIABLE = "STEP_KINK"
    INTERACTION_TWO_WAY = "INTERACTION_TWO_WAY"
    INTERACTION_THREE_WAY = "INTERACTION_THREE_WAY"

    COVARIATE_SYMBOLS_KEY = "covariates"
    EXPRESSION_KEY = "expr"
    DISCRETE_ALLOWED_KEY = "disc"

    SUBFUNCTION_CONSTANT_SYMBOLS = {a, c}

    # The various transforms which can be applied to covariates
    # and combinations of covariates.
    SUBFUNCTION_FORMS = {
        LINEAR: {
                COVARIATE_SYMBOLS_KEY: [x],
                EXPRESSION_KEY: c*x,
                DISCRETE_ALLOWED_KEY: True
        },
        POLY_QUADRATIC: {
                COVARIATE_SYMBOLS_KEY: [x],
                EXPRESSION_KEY: c*(x**2),
                DISCRETE_ALLOWED_KEY: False
        },
        POLY_CUBIC: {
                COVARIATE_SYMBOLS_KEY: [x],
                EXPRESSION_KEY: c*(x**3),
                DISCRETE_ALLOWED_KEY: False
        },
        STEP_CONSTANT: {
                COVARIATE_SYMBOLS_KEY: [x],
                EXPRESSION_KEY: sp.Piecewise((0, x < a), (c, True)),
                DISCRETE_ALLOWED_KEY: False
        },
        STEP_VARIABLE: {
                COVARIATE_SYMBOLS_KEY: [x],
                EXPRESSION_KEY: sp.Piecewise((0, x < a), (c*x, True)),
                DISCRETE_ALLOWED_KEY: False
        },
        INTERACTION_TWO_WAY: {
                COVARIATE_SYMBOLS_KEY: [x, y],
                EXPRESSION_KEY: c*x*y,
                DISCRETE_ALLOWED_KEY: True
        },
        INTERACTION_THREE_WAY: {
                COVARIATE_SYMBOLS_KEY: [x, y, z],
                EXPRESSION_KEY: c*x*y*z,
                DISCRETE_ALLOWED_KEY: True
        },
    }

    # All X times the number of original covariates.
    MAX_RATIO_TRANSFORMED_TO_ORIGINAL_TERMS = 5

    # How much of the covariate data to use when normalizing functions.
    NORMALIZATION_DATA_SAMPLE_FRACTION = 1

    COVARIATES_NAME = "X"
    TRANSFORMED_COVARIATES_NAME = "TRANSFORMED_X"

    PROPENSITY_LOGIT_NAME = "logit(P(T|X))"
    PROPENSITY_SCORE_NAME = "P(T|X)"
    TREATMENT_ASSIGNMENT_NAME = "T"
    TREATMENT_ASSIGNMENT_SYMBOL = sp.symbols(TREATMENT_ASSIGNMENT_NAME)

    OBSERVED_OUTCOME_NAME = "Y"
    POTENTIAL_OUTCOME_WITHOUT_TREATMENT_NAME = "Y0"
    POTENTIAL_OUTCOME_WITH_TREATMENT_NAME = "Y1"
    TREATMENT_EFFECT_NAME = "TE"

    OUTCOME_NOISE_NAME = "NOISE(Y)"
    OUTCOME_NOISE_SYMBOL = sp.symbols(OUTCOME_NOISE_NAME)

    ### Metric constants ###

    # Data inputs
    class AnalysisMetricData:
        OBSERVED_COVARIATE_DATA = "OBSERVED_COVARIATES"
        OBSERVED_OUTCOME_DATA = "OBSERVED_OUTCOMES"
        TRANSFORMED_COVARIATE_DATA = "TRANSFORMED_COVARIATES"
        ORACLE_OUTCOME_DATA = "ORACLE_OUTCOMES"

    # Functions
    class AnalysisMetricFunctions:
        LINEAR_R2 = "Lin r2"
        LOGISTIC_R2 = "Log r2"
        PERCENT = "Percent"
        L2_MEAN_DIST = "mean dist"
        NN_CF_MAHALA_DIST = "NN c-factual dist"
        STD_RATIO = "Normed std"
        WASS_DIST = "Wass dist"
        NAIVE_TE = "Naive TE"

    class AxisNames:
        OUTCOME_NONLINEARITY = "OUTCOME_NONLINEARITY"
        TREATMENT_NONLINEARITY = "TREATMENT_NONLINEARITY"
        PERCENT_TREATED = "PERCENT_TREATED"
        OVERLAP = "OVERLAP"
        BALANCE = "BALANCE"
        ALIGNMENT = "ALIGNMENT"
        TE_HETEROGENEITY = "TE_HETEROGENEITY"

    class AxisLevels:
        LOW = "LOW"
        MEDIUM = "MEDIUM"
        HIGH = "HIGH"

    class Data:
        get_dataset_path = lambda file_name: resource_filename(
            'maccabee', f"data/{file_name}.csv")

        LALONDE_PATH = get_dataset_path("lalonde")
        LALONDE_DISCRETE_COVARS = ["black","hispanic","married","nodegree"]

        CPP_PATH = get_dataset_path("cpp")
        CPP_DISCRETE_COVARS = ['x_17','x_22','x_38','x_51','x_54','x_2_A','x_2_B',
            'x_2_C','x_2_D','x_2_E','x_2_F','x_21_A','x_21_B','x_21_C','x_21_D',
            'x_21_E','x_21_F','x_21_G','x_21_H','x_21_I','x_21_J','x_21_K',
            'x_21_L','x_21_M','x_21_N','x_21_O','x_21_P','x_24_A','x_24_B',
            'x_24_C','x_24_D', 'x_24_E']

    class Model:
        ITE_ESTIMAND = "ITE"
        ATE_ESTIMAND = "ATE"
