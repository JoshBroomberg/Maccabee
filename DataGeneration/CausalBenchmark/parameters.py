import numpy as np
import sympy as sp
from sympy.abc import x
import yaml
from copy import deepcopy

PARAMETER_SPECIFICATION_PATH = "CausalBenchmark/param_specification.yml"
DEFAULT_PARAMETER_PATH = "CausalBenchmark/default_params.yml"

PARAMETER_SPEC = yaml.safe_load(open(PARAMETER_SPECIFICATION_PATH, "r"))

PARAM_DESC_KEY = "description"

PARAM_TYPE_KEY = "type"
PARAM_TYPE_NUMBER = "number"
PARAM_TYPE_DICTIONARY = "dictionary"

# Number type keys
PARAM_MIN_KEY = "min"
PARAM_MAX_KEY = "max"

# Dict type keys
PARAM_DICT_KEYS_KEY = "required_keys"

# TODO: refactor the way calculated params and distributions are handled.
class ParameterStore(object):

    def __init__(self):
        self.params_loaded = False

    def load(self, parameter_file_path):
        self.params_loaded = True

        params_file = open(parameter_file_path, "r")
        parameters = yaml.safe_load(params_file)

        for param_name, param_info in PARAMETER_SPEC.items():
            param_type = param_info[PARAM_TYPE_KEY]
            if param_name in parameters:
                param_value = parameters[param_name]
                # TODO: add checks for min/max/keys
                setattr(self, param_name, param_value)
            else:
                raise Exception("Missing parameter: {}".format(param_name))

        # TODO: provide a spec for calculated params
        logistic_function = (sp.functions.exp(x)/(1+sp.functions.exp(x)))
        self.TARGET_MIN_LOGIT = sp.solve(
            logistic_function - self.MIN_PROPENSITY_SCORE, x)[0]
        self.TARGET_MAX_LOGIT = sp.solve(
            logistic_function - self.MAX_PROPENSITY_SCORE, x)[0]
        self.TARGET_MEAN_LOGIT = sp.solve(
            logistic_function - self.TARGET_PROPENSITY_SCORE, x)[0]

    # TODO: provide a spec for sampling functions.
    def sample_subfunction_constants(self, size=1):
        return np.round(np.random.standard_t(
                             self.SUBFUNCTION_CONSTANT_TAIL_THICKNESS, size=size), 3)

    def sample_outcome_noise(self, size=1):
        return np.round(np.random.standard_t(
                            self.OUTCOME_NOISE_TAIL_THICKNESS, size=size), 3)

    def sample_treatment_effect(self, size=1):
        return np.round(np.random.standard_t(
                            self.TREATMENT_EFFECT_TAIL_THICKNESS, size=size), 3)

Parameters = ParameterStore()

def load_parameters(parameter_file_path=DEFAULT_PARAMETER_PATH):
    Parameters.load(parameter_file_path)
