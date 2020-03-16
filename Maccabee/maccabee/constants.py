"""This module contains constants which are used throughout the package. The constants defined here serve two purposes: First, to simplify and standardize user interaction with the external APIs by providing users a convenient way to refer to common concepts/values. Second, to centralize the configuaration constants which control the internal operation of the package, giving more advanced users the ability to control/modify operation by making changes in one location.

All Maccabee constants are stored as attributes of the :class:`maccabee.constants.Constants` class. Within this class, constants are nested into subclasses of the ``ConstantGroup`` class (IE, the only attributes of the :class:`~maccabee.constants.Constants` class are ``ConstantGroup`` subclasses). These subclasses store actual parameters as their attributes. So, for example, the axis names for the axes of the :term:`distributional problem space` can be accessed as the attributes of the class ``Constants.AxisNames``.

All of the ``ConstantGroup`` classes can be introspected to view the constant names/values stored in the group. This is done by calling the ``.all()`` method which will return a name/value dictionary. If the ``print=True`` option is supplied this will (pretty) print the dictionary.

So, to access the axis name constants mentioned above, one first imports the constants class and then uses the ``AxisNames`` attribute. Because this attribute is a subclass of the ``ConstantGroup`` class, the ``all()`` method can be used to introspect the values.

>>> from maccabee.constants import Constants
>>> Constants.AxisNames.all()
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
    def all(cls, print=False):
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
    As discussed above, this class contains the constants used throughout the package. All of the ``ConstantGroup`` attribute classes are listed below. Constants which are predominantly for internal use are marked [INTERNAL] and do not have explanations here. Advanced users interested in modifying the value of these constants should see the source code and inline comes in :download:`constants.py </../../maccabee/constants.py>`.
    """

    class ParamFilesAndPaths:
        """[INTERNAL] Constants related to the location and parsed content of the YAML parameter specification files which control the parameter schema, default parameter values and metric-level parameter values. See the docs for the :mod:`maccabee.parameters` module for more detail.
        """

        # The path of the parameter schema.
        _SCHEMA_PATH = resource_filename(
            'maccabee', 'parameters/parameter_schema.yml')

        # The in-memory cache of the parameter schema, used to construct
        # parameter store objects.
        with open(_SCHEMA_PATH, "r") as schema_file:
            SCHEMA = yaml.safe_load(schema_file)["SCHEMA"]

        # The path for the specification of default parameter values.
        DEFAULT_SPEC_PATH = resource_filename(
            'maccabee', 'parameters/default_parameter_specification.yml')

        # The path for the specification of the parameter values for each
        # axis level.
        AXIS_LEVEL_SPEC_PATH = resource_filename(
            'maccabee', 'parameters/metric_level_parameter_specifications.yml')

    class ParamSchemaKeysAndVals(ConstantGroup):
        """[INTERNAL] Constants related to the keys and values of the parameter specification files mentioned under :class:`~maccabee.constants.Constants.ParamFilesAndPaths`."""

        # The constants below define the keys and allowed values
        # which are used in the specification of the parameter schema.
        DESCRIPTION_KEY = "description"

        # Type key and values
        TYPE_KEY = "type" # The key itself
        TYPE_NUMBER = "number" # val
        TYPE_BOOL = "bool" # val
        TYPE_DICTIONARY = "dictionary" # val
        TYPE_CALCULATED = "calculated" # val

        # Number type keys. The value is a real number.
        MIN_KEY = "min"
        MAX_KEY = "max"

        # Dict type key. The value is a dict.
        DICT_KEYS_KEY = "required_keys"

        # Calc type key. The value is a python expression.
        EXPRESSION_KEY = "expr"

    ### DGP Sampling constants ###
    class DGPSampling(ConstantGroup):
        """[INTERNAL] Constants related to the sampling of the subfunctions which make up the sampled DGP treatment and outcome functions.
        """

        ## FEATURE FLAGS: The constants below are used to
        # turn features of the sampling process on/off. They
        # are used primarily in development but may be useful
        # for advanced users.

        # Normalization: whether to apply approximate normalization schemes to the sampled
        # treatment and outcome functions such that the outcome and propensity
        # values have zero mean with an approx std of 1.
        NORMALIZE_SAMPLED_TREATMENT_FUNCTION = True
        NORMALIZE_SAMPLED_OUTCOME_FUNCTION = False
        CENTER_SAMPLED_OUTCOME_FUNCTION = False

        # Alignment: whether to apply the alignment adjustment.
        ADJUST_ALIGNMENT = True

        # Sampling Strategy: force sampling of items based on per item probability rather than fixed choice of the expected number of selected items.
        FORCE_PER_ITEM_SAMPLING = False

        ## Sampling Configuration: the config which controls the
        # operation of the sampling process.

        # How much of the covariate data to use when normalizing
        # the sampled functions. For large data sets, a smaller proportion
        # is necessary.
        NORMALIZATION_DATA_SAMPLE_FRACTION = 0.75

        # The subfunctions which are sampled to construct the sampled
        # treatent and outcome functions.
        LINEAR = "LINEAR"
        POLY_QUADRATIC = "POLY_QUAD"
        POLY_CUBIC = "POLY_CUBIC"
        STEP_CONSTANT = "STEP_JUMP"
        STEP_VARIABLE = "STEP_KINK"
        INTERACTION_TWO_WAY = "INTERACTION_TWO_WAY"
        INTERACTION_THREE_WAY = "INTERACTION_THREE_WAY"

        # Subfunction forms
        # To perform sampling, it is necessary to know the mathematical expression
        # of the subfunctions, the symbols and constants in the expression, and the
        # types of covariates allowed. This information is in the subfunction forms
        # dictionary below.

        # The keys which are used in the subfunction forms dictionary.
        COVARIATE_SYMBOLS_KEY = "covariates"
        EXPRESSION_KEY = "expr"
        DISCRETE_ALLOWED_KEY = "disc"
        SUBFUNCTION_CONSTANT_SYMBOLS = {a, c}

        # The subfunctions forms dictionary itself.
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

        # The maximum number of selected subfunctions (each producing a
        # transformed covariate) is defined by a multiplier on the
        # the number of original covariates.
        MAX_MULTIPLE_TRANSFORMED_TO_ORIGINAL_TERMS = 5

    ### DGP Component constants ###

    class DGPVariables(ConstantGroup):
        """Constants related to the naming of the variables over which DGPs are defined. These are used when specifying concrete DGPs using the :class:`~maccabee.data_generation.data_generating_process.ConcreteDataGeneratingProcess` class and when interacting with the data in the :class:`~maccabee.data_generation.generated_data_set.GeneratedDataSet` instances produced when sampling from any :class:`~maccabee.data_generation.data_generating_process.DataGeneratingProcess` instance.
        """

        #: The collective name for the observed covariates for each individual
        #: observation in the data set.
        COVARIATES_NAME = "X"

        #: Transformed covariates are produced by applying all of the sampled
        #: subfunctions to the original covariates. Because the treatment and outcome
        #: functions are made up of some combination of these subfunctions, the values
        #: produced by each subfunction can be thought of as the true covariates
        #: of the treatment and outcome functions.
        TRANSFORMED_COVARIATES_NAME = "X_transformed"

        #: The logit of the true probability of/propensity for treatment.
        PROPENSITY_LOGIT_NAME = "logit_p_score"

        #: The true probability of/propensity for treatment.
        PROPENSITY_SCORE_NAME = "p_score"

        #: The treatment assignment/status
        TREATMENT_ASSIGNMENT_NAME = "T"

        #: The observed outcome.
        OBSERVED_OUTCOME_NAME = "Y"

        #: The potential outcome without treatment, understood in terms of the
        #: Rubin-Neyman causal model.
        POTENTIAL_OUTCOME_WITHOUT_TREATMENT_NAME = "Y0"

        #: The potential outcome with treatment, understood in terms of the
        #: Rubin-Neyman causal model.
        POTENTIAL_OUTCOME_WITH_TREATMENT_NAME = "Y1"

        #: The true treatment effect, the different between the potential
        #: outcome with and without treatment.
        TREATMENT_EFFECT_NAME = "treatment_effect"

        #: The noise in the observation of the units potential outcomes.
        OUTCOME_NOISE_NAME = "Y_noise"

        # Symbols corresponding to the components above which appear explicitly
        # in the sampled DGP.
        _TREATMENT_ASSIGNMENT_SYMBOL = sp.symbols(TREATMENT_ASSIGNMENT_NAME)
        _OUTCOME_NOISE_SYMBOL = sp.symbols(OUTCOME_NOISE_NAME)

        # Variable groups used for internal data structures
        # DGP_VARIABLE_DF_GROUPS = {
        #     "OBSERVABLE_COVARIATES": COVARIATES_NAME,
        #     "ORACLE_COVARIATES": TRANSFORMED_COVARIATES_NAME,
        #     "OBSERVABLE_OUTCOME_DATA": [
        #         TREATMENT_ASSIGNMENT_NAME,
        #         OBSERVED_OUTCOME_NAME
        #     ],
        #     "ORACLE_OUTCOME_DATA": [
        #         PROPENSITY_SCORE_NAME,
        #         PROPENSITY_LOGIT_NAME,
        #         OUTCOME_NOISE_NAME,
        #         POTENTIAL_OUTCOME_WITHOUT_TREATMENT_NAME,
        #         POTENTIAL_OUTCOME_WITH_TREATMENT_NAME,
        #         TREATMENT_EFFECT_NAME,
        #     ]
        # }


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

        # The functions below are used in the calculation of one or more
        # data metrics.

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

        # Build the path to a data set given a data set name.
        get_dataset_path = lambda file_name: resource_filename(
            'maccabee', f"data/{file_name}.csv")

        # Constants related to the Lalonde data
        LALONDE_PATH = get_dataset_path("lalonde")
        LALONDE_DISCRETE_COVARS = ["black", "hispanic", "married", "nodegree"]

        # Constants related to the CPP data
        CPP_PATH = get_dataset_path("cpp")
        CPP_DISCRETE_COVARS = ['x_17','x_22','x_38','x_51','x_54','x_2_A','x_2_B',
            'x_2_C','x_2_D','x_2_E','x_2_F','x_21_A','x_21_B','x_21_C','x_21_D',
            'x_21_E','x_21_F','x_21_G','x_21_H','x_21_I','x_21_J','x_21_K',
            'x_21_L','x_21_M','x_21_N','x_21_O','x_21_P','x_24_A','x_24_B',
            'x_24_C','x_24_D', 'x_24_E']

    ### Modeling constants ###

    class Model(ConstantGroup):
        """Constants related to models and estimands."""

        #: The Individual Treatment Effect estimand.
        ITE_ESTIMAND = "ITE"

        #: The Average Treatment Effect estimand.
        ATE_ESTIMAND = "ATE"

        #: The Average Treatment Effect for the Treated estimand.
        ATT_ESTIMAND = "ATT"

        #: A list of all estimands supported by the package.
        ALL_ESTIMANDS = [
            ITE_ESTIMAND,
            ATE_ESTIMAND,
            ATT_ESTIMAND
        ]

        #: A list of all estimands which target a sample-level average effect
        AVERAGE_ESTIMANDS = [
            ATE_ESTIMAND,
            ATT_ESTIMAND
        ]

        #: A list of all estimands which target an observation-level individual effect
        INDIVIDUAL_ESTIMANDS = [
            ITE_ESTIMAND
        ]
