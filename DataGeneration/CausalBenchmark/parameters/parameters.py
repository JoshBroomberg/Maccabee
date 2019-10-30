import numpy as np
import sympy as sp
from sympy.abc import x
import yaml

from pkg_resources import resource_filename

PARAMETER_SPECIFICATION_PATH = resource_filename(
    'CausalBenchmark', 'parameters/parameter_schema.yml')
DEFAULT_PARAMETER_PATH = resource_filename(
    'CausalBenchmark', 'parameters/default_parameter_specification.yml')
METRIC_LEVEL_PARAMETER_PATH = resource_filename(
    'CausalBenchmark', 'parameters/metric_level_parameter_specifications.yml')

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
class ParameterStore():

    def __init__(self, parameter_spec_path):
        with open(parameter_spec_path, "r") as params_file:
            raw_parameter_dict = yaml.safe_load(params_file)
        self.parsed_parameter_dict = {}

        for param_name, param_info in PARAMETER_SPEC.items():
            param_type = param_info[PARAM_TYPE_KEY]
            if param_name in raw_parameter_dict:
                param_value = raw_parameter_dict[param_name]
                self.set_parameter(param_name, param_value)
            else:
                raise Exception("Missing parameter: {}".format(param_name))

        self.find_calculated_params()

        # TODO: provide a spec for calculated params
    def find_calculated_params(self):
        logistic_function = (sp.functions.exp(x)/(1+sp.functions.exp(x)))
        self.TARGET_MIN_LOGIT = sp.solve(
            logistic_function - self.MIN_PROPENSITY_SCORE, x)[0]
        self.TARGET_MAX_LOGIT = sp.solve(
            logistic_function - self.MAX_PROPENSITY_SCORE, x)[0]
        self.TARGET_MEAN_LOGIT = sp.solve(
            logistic_function - self.TARGET_PROPENSITY_SCORE, x)[0]

    def set_parameter(self, param_name, param_value):
        # TODO: add checks for min/max/keys
        setattr(self, param_name, param_value)
        self.parsed_parameter_dict[param_name] = param_value

    def write(self):
        # TODO: dump parsed params to yaml spec
        # with open('data.yml', 'w') as outfile:
        #     yaml.dump(data, outfile, default_flow_style=False)
        pass

    # TODO: provide a spec for sampling functions.
    def sample_subfunction_constants(self, size=1):
        std = np.sqrt(self.SUBFUNCTION_CONSTANT_TAIL_THICKNESS/(self.SUBFUNCTION_CONSTANT_TAIL_THICKNESS-2))
        return np.round(np.random.standard_t(
                             self.SUBFUNCTION_CONSTANT_TAIL_THICKNESS, size=size)/std, 3)

    def sample_outcome_noise(self, size=1):
        std = 3*np.sqrt(self.OUTCOME_NOISE_TAIL_THICKNESS/(self.OUTCOME_NOISE_TAIL_THICKNESS-2))
        return np.round(np.random.standard_t(
                            self.OUTCOME_NOISE_TAIL_THICKNESS, size=size)/std, 3)

    def sample_treatment_effect(self, size=1):
        return np.round(np.random.standard_t(
                            self.TREATMENT_EFFECT_TAIL_THICKNESS, size=size), 3)

def build_parameters_from_specification(parameter_spec_path):
    return ParameterStore(parameter_spec_path=parameter_spec_path)

def build_parameters_from_metric_levels(metric_levels, save=False):
    params = ParameterStore(parameter_spec_path=DEFAULT_PARAMETER_PATH)

    with open(METRIC_LEVEL_PARAMETER_PATH, "r") as metric_level_file:
        metric_level_param_specs = yaml.safe_load(metric_level_file)

    for metric_name, metric_level in metric_levels.items():
        if metric_name in metric_level_param_specs:
            metric_level_specs = metric_level_param_specs[metric_name]
            if metric_level in metric_level_specs:
                for param_name, param_value in metric_level_specs[metric_level].items():
                    params.set_parameter(param_name, param_value)
            else:
                raise Exception(f"{metric_level} is not a valid level for {metric_name}")
        else:
            raise Exception(f"{metric_name} is not a valid metric")

    params.find_calculated_params()
    return params
