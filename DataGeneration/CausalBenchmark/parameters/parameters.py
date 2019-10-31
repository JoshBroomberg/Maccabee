import numpy as np
import sympy as sp
from sympy.abc import x
import yaml

from pkg_resources import resource_filename

### Define constants ###

PARAMETER_SCHEMA_PATH = resource_filename(
    'CausalBenchmark', 'parameters/parameter_schema.yml')
DEFAULT_PARAMETER_PATH = resource_filename(
    'CausalBenchmark', 'parameters/default_parameter_specification.yml')
METRIC_LEVEL_PARAMETER_PATH = resource_filename(
    'CausalBenchmark', 'parameters/metric_level_parameter_specifications.yml')

PARAMETER_SCHEMA = yaml.safe_load(open(PARAMETER_SCHEMA_PATH, "r"))

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
    '''
    This class stores all the parameters which are used to control the sampling
    of the Data Generating Process. It ensures that supplied parameters meet
    the schema at PARAMETER_SCHEMA_PATH and is responsible for managing
    calculated and sampling parameters (which are not concretely specified).
    '''

    def __init__(self, parameter_spec_path):
        with open(parameter_spec_path, "r") as params_file:
            raw_parameter_dict = yaml.safe_load(params_file)
        self.parsed_parameter_dict = {}

        # Read in the parameter values for each param in the
        # schema.
        for param_name, param_info in PARAMETER_SCHEMA.items():
            param_type = param_info[PARAM_TYPE_KEY]

            # If the parameter is in the specification file
            if param_name in raw_parameter_dict:
                param_value = raw_parameter_dict[param_name]
                self.set_parameter(param_name, param_value)

            # Parameter is missing.
            else:
                raise Exception("Missing parameter: {}".format(param_name))

        self.find_calculated_params()

    # TODO: provide a way to speciffy calculated params to avoid hard coding
    # them here.
    def find_calculated_params(self):
        '''
        Finds the value of the calculated params based on the fixed params.
        '''
        logistic_function = (sp.functions.exp(x)/(1+sp.functions.exp(x)))
        self.TARGET_MIN_LOGIT = sp.solve(
            logistic_function - self.MIN_PROPENSITY_SCORE, x)[0]
        self.TARGET_MAX_LOGIT = sp.solve(
            logistic_function - self.MAX_PROPENSITY_SCORE, x)[0]
        self.TARGET_MEAN_LOGIT = sp.solve(
            logistic_function - self.TARGET_PROPENSITY_SCORE, x)[0]

    # Makes a parameter value available on the ParamStore object
    # and stores the value in a dict for later write out.
    def set_parameter(self, param_name, param_value):
        # TODO: add checks for min/max/keys
        setattr(self, param_name, param_value)
        self.parsed_parameter_dict[param_name] = param_value

    def write(self):
        # TODO: dump parsed params to valid yaml spec
        # with open('data.yml', 'w') as outfile:
        #     yaml.dump(data, outfile, default_flow_style=False)
        pass

    # TODO: provide a wat to specify sampling functions
    # to avoid hard coding.
    def sample_subfunction_constants(self, size=1):
        std = 5*np.sqrt(self.SUBFUNCTION_CONSTANT_TAIL_THICKNESS/(self.SUBFUNCTION_CONSTANT_TAIL_THICKNESS-2))
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
    '''
    Build a parameter store from a give specification file.
    '''
    return ParameterStore(parameter_spec_path=parameter_spec_path)

def build_parameters_from_metric_levels(metric_levels, save=False):
    '''
    Build a parameter store from a set of metric levels. These
    are applied onto the default parameter spec.
    '''
    params = ParameterStore(parameter_spec_path=DEFAULT_PARAMETER_PATH)

    with open(METRIC_LEVEL_PARAMETER_PATH, "r") as metric_level_file:
        metric_level_param_specs = yaml.safe_load(metric_level_file)

    # Set the value of each metric to the correct values.
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
