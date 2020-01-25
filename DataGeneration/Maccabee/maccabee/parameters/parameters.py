import numpy as np
import sympy as sp
from sympy.abc import x
import yaml
from ..constants import Constants

PARAM_CONSTANTS = Constants.Params

# TODO: refactor the way calculated params and distributions are handled.
class ParameterStore():
    '''
    This class stores all the parameters which are used to control the sampling
    of the Data Generating Process. It ensures that supplied parameters meet
    the fixed schema and is responsible for managing
    calculated and sampling parameters (which are not concretely specified).
    '''

    def __init__(self, parameter_spec_path):
        with open(parameter_spec_path, "r") as params_file:
            raw_parameter_dict = yaml.safe_load(params_file)
        self.parsed_parameter_dict = {}
        self.calculated_parameters = {}

        # Read in the parameter values for each param in the
        # schema.
        for param_name, param_info in PARAM_CONSTANTS.SCHEMA.items():
            param_type = param_info[PARAM_CONSTANTS.ParamInfo.TYPE_KEY]

            # If param should be calculated
            if param_type == PARAM_CONSTANTS.ParamInfo.TYPE_CALCULATED:
                if param_name in raw_parameter_dict:
                    raise Exception(
                        "{} is calculated. It can't be supplied.".format(
                            param_name))

                param_value = self.get_calculated_param_value(param_info)
                self.calculated_parameters[param_name] = param_info

            # If the parameter is in the specification file
            elif param_name in raw_parameter_dict:
                param_value = raw_parameter_dict[param_name]
                if not self.validate_param_value(param_info, param_value):
                    raise Exception("Invalid value for {}: {}".format(
                        param_name, param_value))

            # Parameter is missing.
            else:
                raise Exception("Param spec is missing: {}".format(param_name))

            self.set_parameter(
                param_name, param_value,
                recalculate_calculated_params=False)

    def get_calculated_param_value(self, param_info):
        expr = param_info[PARAM_CONSTANTS.ParamInfo.EXPRESSION_KEY]
        return eval(expr, globals(), self.parsed_parameter_dict)

    def recalculate_calculated_params(self):
        for param_name, param_info in self.calculated_parameters.items():
            param_value = self.get_calculated_param_value(param_info)
            self.set_parameter(param_name, param_value,
                recalculate_calculated_params=False)

    def validate_param_value(self, param_info, param_value):
        param_type = param_info[PARAM_CONSTANTS.ParamInfo.TYPE_KEY]
        if param_type == PARAM_CONSTANTS.ParamInfo.TYPE_NUMBER:
            return param_info[PARAM_CONSTANTS.ParamInfo.MIN_KEY] <= param_value <= param_info[PARAM_CONSTANTS.ParamInfo.MAX_KEY]

        elif param_type == PARAM_CONSTANTS.ParamInfo.TYPE_DICTIONARY:
            required_keys = set(param_info[PARAM_CONSTANTS.ParamInfo.DICT_KEYS_KEY])
            supplied_keys = set(param_value.keys())
            return required_keys == supplied_keys

        elif param_type == PARAM_CONSTANTS.ParamInfo.TYPE_BOOL:
            return param_value in [True, False]

        else:
            # Unknown param type, cannot validate. Fail at medium volume.
            return False

    # Makes a parameter value available on the ParamStore object
    # and stores the value in a dict for later write out.
    def set_parameter(self, param_name, param_value, recalculate_calculated_params=True):
        setattr(self, param_name, param_value)
        self.parsed_parameter_dict[param_name] = param_value
        if recalculate_calculated_params:
            self.recalculate_calculated_params()

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

def build_parameters_from_axis_levels(metric_levels, save=False):
    '''
    Build a parameter store from a set of metric levels. These
    are applied onto the default parameter spec.
    '''

    params = ParameterStore(parameter_spec_path=PARAM_CONSTANTS.DEFAULT_SPEC_PATH)

    with open(PARAM_CONSTANTS.METRIC_LEVEL_SPEC_PATH, "r") as metric_level_file:
        metric_level_param_specs = yaml.safe_load(metric_level_file)

    # Set the value of each metric to the correct values.
    for metric_name, metric_level in metric_levels.items():

        if metric_name in metric_level_param_specs:
            metric_level_specs = metric_level_param_specs[metric_name]

            if metric_level in metric_level_specs:
                for param_name, param_value in metric_level_specs[metric_level].items():
                    params.set_parameter(
                        param_name, param_value, recalculate_calculated_params=False)
            else:
                raise Exception(f"{metric_level} is not a valid level for {metric_name}")
        else:
            raise Exception(f"{metric_name} is not a valid metric")


    params.recalculate_calculated_params()

    return params
