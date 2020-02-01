"""This module contains constants which are used throughout the package both in the internal operation code and to simplify and standardize user interaction with the external API.

All constants are stored as attributes of the :class:`~maccabee.constants.Constants` class. So, all usage of constants will involve importing this class using the line ``from maccabee.constants import Constants``. For convenience and clarity, constants are nested into ``ConstantGroup`` classes which are the only attributes of the :class:`~maccabee.constants.Constants` class and which store actual parameters as their attributes. So, for example, the axis names for the axes of the distributional setting space can be accessed as the attributes of the class ``Constants.AxisNames``.

All of the ``ConstantGroup`` classes can be introspected to view the constants stored in the group. This is done by calling the ``.all()`` method. By default this will (pretty) print a dictionary of the constant names and values and return the dictionary. Call the function with ``print=False`` to avoid printing the dictionary

>>> from maccabee.constants import Constants
>>> Constants.AxisNames.all(print=False)
    { 'ALIGNMENT': 'ALIGNMENT',
      'BALANCE': 'BALANCE',
      'OUTCOME_NONLINEARITY': 'OUTCOME_NONLINEARITY',
      'OVERLAP': 'OVERLAP',
      'PERCENT_TREATED': 'PERCENT_TREATED',
      'TE_HETEROGENEITY': 'TE_HETEROGENEITY',
      'TREATMENT_NONLINEARITY': 'TREATMENT_NONLINEARITY'}
"""

from sympy.abc import a, c, x, y, z
import sympy as sp
from pkg_resources import resource_filename
import yaml
import pprint

class ConstantGroup:
    pp = pprint.PrettyPrinter(indent=2, width=40)

    @classmethod
    def all(cls, print=True):
        constants = {
            k: v
            for k, v in cls.__dict__.items()
            if (not k.startswith("_")) and (True)
        }

        if print:
            cls.pp.pprint(constants)

        return constants

### Define constants ###

# OPERATIONAL CONSTANTS
class Constants:
    """
    This Constants class contains the operational and interaction constants which are used throughout the package. Its attributes are subclasses of the ``ConstantGroup`` class each of which groups together a set of related constants. All of the ``ConstantGroup`` classes are below. Constants which are predominantly for internal use are marked [INTERNAL] and do not have explanations here. See the source code in :download:`constants.py </../../maccabee/constants.py>` for in-line comments.
    """

    class ParamFilesAndPaths:
        """[INTERNAL] Constants related to the location and parsed content of the YAML parameter specification files which control the parameter schema, default parameter values and metric-level parameter values. See the docs for the :mod:`maccabee.parameters` module for more detail.
        """

        _SCHEMA_PATH = resource_filename(
            'maccabee', 'parameters/parameter_schema.yml')

        with open(_SCHEMA_PATH, "r") as schema_file:
            SCHEMA = yaml.safe_load(schema_file)["SCHEMA"]

        DEFAULT_SPEC_PATH = resource_filename(
            'maccabee', 'parameters/default_parameter_specification.yml')
        METRIC_LEVEL_SPEC_PATH = resource_filename(
            'maccabee', 'parameters/metric_level_parameter_specifications.yml')

    class ParamSchemaKeysAndVals(ConstantGroup):
        """[INTERNAL] Constants related to the keys and values of the parameter specification files mentioned under :class:`~maccabee.constants.Constants.ParamFilesAndPaths`."""

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

    ### DGP Sampling constants ###
    class DGPSampling(ConstantGroup):
        """[INTERNAL] Constants related to the sampling of the subfunctions which make up the sampled DGP treatment and outcome functions.
        """

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

    ### DGP Component constants ###

    class DGPVariables(ConstantGroup):
        """Constants related to the naming of the variables over which DGPs are defined. These will be of interest for users when handing the data properties of :class:`~maccabee.data_generation.GeneratedDataSet` instances as the columns of the :class:`~pandas.DataFrame` instances returned by these properties will be named according to these constants.
        """

        COVARIATES_NAME = "X"
        TRANSFORMED_COVARIATES_NAME = "TRANSFORMED_X"

        PROPENSITY_LOGIT_NAME = "logit(P(T|X))"
        PROPENSITY_SCORE_NAME = "P(T|X)"
        TREATMENT_ASSIGNMENT_NAME = "T"

        OBSERVED_OUTCOME_NAME = "Y"
        POTENTIAL_OUTCOME_WITHOUT_TREATMENT_NAME = "Y0"
        POTENTIAL_OUTCOME_WITH_TREATMENT_NAME = "Y1"
        TREATMENT_EFFECT_NAME = "TE"

        OUTCOME_NOISE_NAME = "NOISE(Y)"

        TREATMENT_ASSIGNMENT_SYMBOL = sp.symbols(TREATMENT_ASSIGNMENT_NAME)
        OUTCOME_NOISE_SYMBOL = sp.symbols(OUTCOME_NOISE_NAME)

    ### Data Metric constants ###

    class AxisNames(ConstantGroup):
        """Constants for the names of the :term:`axes <distributional problem space axis>` of the :term:`distributional problem space`. Maccabee allows for the sampling of DGPs (and associated data samples) at different 'levels' (locations) along these axes. See the :doc:`Design doc </design>` for more on Maccabee's theoretical approach.
        """

        #: The outcome nonlinearity axis - controls the degree of nonlinearity
        #: in the outcome mechanism.
        OUTCOME_NONLINEARITY = "OUTCOME_NONLINEARITY"

        #: The treatment nonlinearity axis - controls the degree of nonlinearity
        #: in the treatment assignment mechanism.
        TREATMENT_NONLINEARITY = "TREATMENT_NONLINEARITY"

        #: The percent treated axis - controls the percent of units that are exposed to treatment.
        PERCENT_TREATED = "PERCENT_TREATED"

        #: The overlap axis - controls to covariate distribution overlap in
        #: the treated and control groups.
        #: WARNING: not currently supported in sampling.
        OVERLAP = "OVERLAP"

        #: The balance axis - controls the degree of similarity between the
        #: covariate distribution in the treated and control group.
        BALANCE = "BALANCE"

        #: The alignment axis - controls the degree of overlap of appearance of covariates
        #: in the treatment and outcome mechanisms. Effectively controlling
        #: the number of confounders and ratio of confounders to non-confounders.
        ALIGNMENT = "ALIGNMENT"

        #: The treatment effect heterogeneity axis - controls the degree to which
        #: the treatment effect varies per unit.
        TE_HETEROGENEITY = "TE_HETEROGENEITY"

    class AxisLevels(ConstantGroup):
        """Constants related to the 'levels' of each axis at which Maccabee can sample DGPs. These levels represent the existing presets but do not preclude sampling at any other point by manually specifying sampling parameters. See :class:`~maccabee.constants.Constants.AxisNames` for links to further explanation.
        """

        #: The constant for a 'low' level on a specific axis.
        LOW = "LOW"

        #: The constant for a 'medium' level on a specific axis.
        MEDIUM = "MEDIUM"

        #: The constant for a 'high' level on a specific axis.
        HIGH = "HIGH"

        #: The constant for conveniently loading all of the level constants.
        LEVELS = (LOW, MEDIUM, HIGH)

    class DataMetricFunctions(ConstantGroup):
        """[INTERNAL] Constants related to the functions used to calculate the metrics which quantify the location of data in the distributional problem space."""
        LINEAR_R2 = "Lin r2"
        LOGISTIC_R2 = "Log r2"
        PERCENT = "Percent"
        L2_MEAN_DIST = "mean dist"
        NN_CF_MAHALA_DIST = "NN c-factual dist"
        STD_RATIO = "Normed std"
        WASS_DIST = "Wass dist"
        NAIVE_TE = "Naive TE"

    ### External Data constants ###

    class ExternalCovariateData(ConstantGroup):
        """[INTERNAL] Constants related to external covariate data. See the :doc:`doc </usage/empirical-datasets>` on the empirical data sets built into Maccabee for more detail.
        """

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

    ### Modeling constants ###

    class Model(ConstantGroup):
        """Constants related to models and estimands."""
        ITE_ESTIMAND = "ITE"
        ATE_ESTIMAND = "ATE"
