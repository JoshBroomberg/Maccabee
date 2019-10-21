import numpy as np
import sympy as sp
from sympy.abc import x
import yaml

DEFAULT_CONFIG = "CausalBenchmark/default_params.yml"

class ParameterStore:

    def __init__(self):
        self.params_loaded = False

    def get(self, parameter_name):
        if not self.params_loaded:
            raise Exception("Parameters not initialized")
        else:
            pass

    def load(self, parameter_file_path):
        params_file = open(parameter_file_path, "r")
        parameters = yaml.safe_load(params_file)

        for parameter in parameters

        ### CONFOUNDER SETTINGS ##

        # Probability that a covariate is a potential confounder
        # (affecting one or both of treatment and outcome)
        # Lower values reduce the number of covariates which are predicitive
        # of treatment/outcome. This makes modelling harder given need
        # for variable selection.
        # TODO: consider differential selection of different covariate types.
        self.POTENTIAL_CONFOUNDER_SELECTION_PROBABILITY = \
            parameters["POTENTIAL_CONFOUNDER_SELECTION_PROBABILITY"]

        # Probability that a covariate in the true space
        # will appear in both the outcome and treatment functions.
        # This is a soft target, with some room for variance.
        self.ACTUAL_CONFOUNDER_ALIGNMENT = \
            parameters["ACTUAL_CONFOUNDER_ALIGNMENT"]

        ### SHARED TREAT/OUTCOME FUNCTION SETTINGS ###

        # Probabilities which govern the probability with which
        # covariates appear in the treatment mechanism in different
        # forms.
        self.TREAT_MECHANISM_COVARIATE_SELECTION_PROBABILITY = \
            parameters["TREAT_MECHANISM_COVARIATE_SELECTION_PROBABILITY"]
        self.OUTCOME_MECHANISM_COVARIATE_SELECTION_PROBABILITY = \
            parameters["OUTCOME_MECHANISM_COVARIATE_SELECTION_PROBABILITY"]

        # DF for the T-distribution over subfunction constants.
        self.SUBFUNCTION_CONSTANT_TAIL_THICKNESS = \
            parameters["SUBFUNCTION_CONSTANT_TAIL_THICKNESS"]

        ### TREATMENT FUNCTION PARAMS ###

        # propensity score settings
        self.MIN_PROPENSITY_SCORE = parameters["MIN_PROPENSITY_SCORE"]
        self.MAX_PROPENSITY_SCORE = parameters["MAX_PROPENSITY_SCORE"]
        self.TARGET_PROPENSITY_SCORE = parameters["TARGET_PROPENSITY_SCORE"]

        logistic_function = (sp.functions.exp(x)/(1+sp.functions.exp(x)))
        self.TARGET_MIN_LOGIT = sp.solve(
            logistic_function - self.MIN_PROPENSITY_SCORE, x)[0]
        self.TARGET_MAX_LOGIT = sp.solve(
            logistic_function - self.MAX_PROPENSITY_SCORE, x)[0]
        self.TARGET_MEAN_LOGIT = sp.solve(
            logistic_function - self.TARGET_PROPENSITY_SCORE, x)[0]

        ### OUTCOME FUNCTION PARAMS ###

        # DF for the T-distribution over outcome noise.
        self.OUTCOME_NOISE_TAIL_THICKNESS = parameters["OUTCOME_NOISE_TAIL_THICKNESS"]

        ### TREATMENT EFFECT PARAMS ###

        # Marginal probability that there is an interaction between the
        # base treatment effect and each subfunction in the outcome function.
        self.TREATMENT_EFFECT_HETEROGENEITY = parameters["TREATMENT_EFFECT_HETEROGENEITY"]

        # DF for the T-distribution over treatment effects.
        self.TREATMENT_EFFECT_TAIL_THICKNESS = parameters["TREATMENT_EFFECT_TAIL_THICKNESS"]

        # Marginal probability of observing any given row of the dataset.
        # Used to reduce the overall number of observations if desired.
        self.OBSERVATION_PROBABILITY = parameters["OBSERVATION_PROBABILITY"]

        self.params_loaded = True

    def sample_subfunction_constants(self, size=1):
        return np.round(np.random.standard_t(
                             self.SUBFUNCTION_CONSTANT_TAIL_THICKNESS, size=size), 3)

    def sample_outcome_noise(self, size=1):
        return np.round(np.random.standard_t(
                            self.OUTCOME_NOISE_TAIL_THICKNESS, size=size), 3)

    def sample_treatment_effect(self, size=1):
        return np.round(np.random.standard_t(
                            self.TREATMENT_EFFECT_TAIL_THICKNESS, size=size), 3)
