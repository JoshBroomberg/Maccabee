import numpy as np
import sympy as sp
from sympy.abc import x
import yaml
from copy import deepcopy

DEFAULT_PARAMETER_PATH = "CausalBenchmark/default_params.yml"
ALLOWED_PARAMS = [
    # Probability that a covariate is a potential confounder
    # (affecting one or both of treatment and outcome)
    # Lower values reduce the number of covariates which are predicitive
    # of treatment/outcome. This makes modelling harder given need
    # for variable selection.
    # TODO: consider differential selection of different covariate types.
    "POTENTIAL_CONFOUNDER_SELECTION_PROBABILITY",

    # Probability that a covariate in the true space
    # will appear in both the outcome and treatment functions.
    # This is a soft target, with some room for variance.
    "ACTUAL_CONFOUNDER_ALIGNMENT",

    ### SHARED TREAT/OUTCOME FUNCTION SETTINGS ###

    # Probabilities which govern the probability with which
    # covariates appear in the treatment mechanism in different
    # forms.
    "TREAT_MECHANISM_COVARIATE_SELECTION_PROBABILITY",
    "OUTCOME_MECHANISM_COVARIATE_SELECTION_PROBABILITY",

    # DF for the T-distribution over subfunction constants.
    "SUBFUNCTION_CONSTANT_TAIL_THICKNESS",

    ### TREATMENT FUNCTION PARAMS ###

    # propensity score settings
    "MIN_PROPENSITY_SCORE",
    "MAX_PROPENSITY_SCORE",
    "TARGET_PROPENSITY_SCORE",

    ### OUTCOME FUNCTION PARAMS ###

    # DF for the T-distribution over outcome noise.
    "OUTCOME_NOISE_TAIL_THICKNESS",

    ### TREATMENT EFFECT PARAMS ###

    # Marginal probability that there is an interaction between the
    # base treatment effect and each subfunction in the outcome function.
    "TREATMENT_EFFECT_HETEROGENEITY",

    # DF for the T-distribution over treatment effects.
    "TREATMENT_EFFECT_TAIL_THICKNESS",

    # Marginal probability of observing any given row of the dataset.
    # Used to reduce the overall number of observations if desired.
    "OBSERVATION_PROBABILITY"
]

# TODO: refactor the way calculated params and distributions are handled.


class Parameters:
    instance = None

    def __new__(cls):
        if cls.instance is None:
            obj = object.__new__(cls)
            cls.instance = obj
            return obj
        else:
            return cls.instance

    def __init__(self):
        self.params_loaded = False

    def __getattr__(self, name):
        if not self.params_loaded:
            raise Exception("Parameters not loaded!")
        else:
            return super().__getattr__(self, name)

    def load(self, parameter_file_path=DEFAULT_PARAMETER_PATH):
        self.params_loaded = True

        params_file = open(parameter_file_path, "r")
        parameters = yaml.safe_load(params_file)

        for name, value in parameters.items():
            if name in ALLOWED_PARAMS:
                setattr(self, name, value)
            else:
                raise Exception("{} is not an allowed parameter".format(name))

        logistic_function = (sp.functions.exp(x)/(1+sp.functions.exp(x)))
        self.TARGET_MIN_LOGIT = sp.solve(
            logistic_function - self.MIN_PROPENSITY_SCORE, x)[0]
        self.TARGET_MAX_LOGIT = sp.solve(
            logistic_function - self.MAX_PROPENSITY_SCORE, x)[0]
        self.TARGET_MEAN_LOGIT = sp.solve(
            logistic_function - self.TARGET_PROPENSITY_SCORE, x)[0]

    def sample_subfunction_constants(self, size=1):
        return np.round(np.random.standard_t(
                             self.SUBFUNCTION_CONSTANT_TAIL_THICKNESS, size=size), 3)

    def sample_outcome_noise(self, size=1):
        return np.round(np.random.standard_t(
                            self.OUTCOME_NOISE_TAIL_THICKNESS, size=size), 3)

    def sample_treatment_effect(self, size=1):
        return np.round(np.random.standard_t(
                            self.TREATMENT_EFFECT_TAIL_THICKNESS, size=size), 3)
