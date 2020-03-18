"""This submodule defines the :class:`~maccabee.parameters.parameter_store.ParameterStore` class. If you haven't already read the overview of sampling parameterization provided in the docs for the :mod:`maccabee.parameters` module you should read those docs before proceeding to read the content below.
"""

import numpy as np
import sympy as sp
from sympy.abc import x
import yaml
from ..constants import Constants
from ..exceptions import ParameterMissingFromSpecException, ParameterInvalidValueException, CalculatedParameterException
from .utils import _non_zero_uniform_sampler

from ..logging import get_logger
logger = get_logger(__name__)

ParamFileConstants = Constants.ParamFilesAndPaths
SchemaConstants = Constants.ParamSchemaKeysAndVals



class ParameterStore():
    """
    **The Design of the ParameterStore**

    In order to understand the :class:`~maccabee.parameters.parameter_store.ParameterStore` class, it is important to understand the motivation behind its design. The goal of the class is to provide a simple interface to specificy and access a complex set of parameters. As mentioned in the parent module docs, the DGP sampling parameters can be of very different types with different validity conditions and access workflows (simple data retrieval, once-off calculation, dynamic calculation). This means standard data structure based storage, for example in a dictionary, would be very difficult and require tight coupling between the consumption of the parameters (in the :class:`~maccabee.data_generation.data_generating_process_sampler.DataGeneratingProcessSampler`) and their specification format. It would also make the task of specifying the parameters laborious as large dictionaries are hard to organize and parse by visual inspection.

    In theory, using a vanilla parameter class would allow for all of this complexity to be encapsulated behind a simple interface. Verification of parameters, execution of arbitrary code etc are easy to implement using standard class design principles/mechanisms. Unfortunately, users are likely to experiment with many different parameterizations as they explore the :term:`distributional problem space` and using a vanilla class for parameter storage is not good for experimentation. In order to experiment with many different sets of parameter values using classes requires either:

    1. The maintenance of many separate classes with different methods/values. This strategy introduces a lot of boilerplate code to define and manage different classes. Even if inheritance is used, defining many new classes is cumbersome and produces parameter specifications which are hard to grok by visual inspection.

    2. Using a single class and changing the methods/attributes by manual run-time parameterization. This strategy makes it hard to switch between different parameterizations and risks loss of reproducibility if specific parameterizations are lost/forgotten.

    Further, it is possible and likely that the  parameters will be added by users of this package. This should, ideally, be facillitated without requiring code changes to the package itself. Vanilla classes would necessitate such changes.

    With the above context in mind, the :class:`~maccabee.parameters.parameter_store.ParameterStore` class pursues a hybrid approach. Parameter values are specified using structured :term:`YML` files referred to as :term:`parameter specification files <parameter specification file>`. These files are easy to read and allow for low-overhead, reliable storage, duplication, and editing. The :term:`parameter specification file` is read by the :class:`~maccabee.parameters.parameter_store.ParameterStore` class at instantiation time. Its content is interpretted based on a package-level :term:`parameter schema file` that specifies the set of expected parameters and their format/handling mechanism and validity conditions. This allows a single set of parameter handling code in the :class:`~maccabee.parameters.parameter_store.ParameterStore` class to be (re)used in storing and accessing an arbitrary set of parameters which are defined in the schema (arbitrary up to the pre-defined set of handling mechanisms which require code changes to modify).

    After instantiation, the parameters are available as **attributes** of the :class:`~maccabee.parameters.parameter_store.ParameterStore` instance as if they had been coded directly into the class definition (despite being stored in a specification file which stores their values for reproducibility and inspection).

    **Building ParameterStore Instances**

    While it is possible to build a :class:`~maccabee.parameters.parameter_store.ParameterStore` instance using the constructor, this is not recommended. There are three supported ways to build instances. The first two are useful if direct control of parameter values is useful/required. The third is designed to build instances using higher-level specification of the desired parameterization (as outlined in the documentation of the parent module).

    * The helper function :func:`~maccabee.parameters.parameter_store_builders.build_parameters_from_specification` can used to easily construct a :class:`~maccabee.parameters.parameter_store.ParameterStore` instance from a parameter specification file.

    * The helper function :func:`~maccabee.parameters.parameter_store_builders.build_default_parameters` builds an instance using the :download:`default_parameter_specification.yml </../../maccabee/parameters/default_parameter_specification.yml>` file

    * Finally, the helper function :func:`~maccabee.parameters.parameter_store_builders.build_parameters_from_axis_levels` builds an instance using a specification of location in the :term:`distributional problem space`. See its documentation for detail.

    **Using ParameterStore Instances**

    As mentioned above, after instantiation, the sampling parameters are available as attributes of the instance. They can therefore be *accessed* as standard attributes of an instance. However, when *setting* the value on an attribute, it is important to use the :meth:`~maccabee.parameters.parameter_store.ParameterStore.set_parameter`/:meth:`~maccabee.parameters.parameter_store.ParameterStore.set_parameters` methods so that calculated parameters are updated appropriately.

    **Instance Method Documentation**

    Args:
        parameter_spec_path (string): The path to a :term:`parameter specification file`.
    """

    def __init__(self, parameter_spec_path):
        logger.debug(f"Reading parameter spec from path {parameter_spec_path}")
        with open(parameter_spec_path, "r") as params_file:
            raw_parameter_dict = yaml.safe_load(params_file)

        self.parsed_parameter_dict = {}
        self.calculated_parameters = {}

        # Read in the parameter values for each param in the
        # schema.
        logger.debug("Build parameter store from schema")
        for param_name, param_info in ParamFileConstants.SCHEMA.items():
            param_type = param_info[SchemaConstants.TYPE_KEY]

            # If param should be calculated
            if param_type == SchemaConstants.TYPE_CALCULATED:
                logger.debug(f"Calculating value for parameter {param_name}")
                if param_name in raw_parameter_dict:
                    raise CalculatedParameterException(param_name)

                param_value = self._find_calculated_param_value(param_info)
                self.calculated_parameters[param_name] = param_info

            # If the parameter is in the specification file
            elif param_name in raw_parameter_dict:
                logger.debug(f"Validating supplied value for parameter {param_name}")
                param_value = raw_parameter_dict[param_name]
                if not self._validate_param_value(param_info, param_value):
                    raise ParameterInvalidValueException(
                        param_name, param_value)

            # Parameter is missing.
            else:
                raise ParameterMissingFromSpecException(param_name)

            self.set_parameter(
                param_name, param_value,
                recalculate_calculated_params=False)

    def set_parameter(self,
        param_name, param_value,
        recalculate_calculated_params=True):
        """Set the value of the param with the name `param_name` to the value `param_value.`

        Args:
            param_name (string): The name of the parameter to set.
            param_value (type): The value to set the parameter to.
            recalculate_calculated_params (bool): Indicates whether calculated params should be recalculated. Defaults to True.

        Examples
            >>> from maccabee.parameters import build_default_parameters
            >>> params = build_default_parameters()
            >>> default_target_propensity = params.TARGET_PROPENSITY_SCORE
            >>> params.set_parameter("TARGET_PROPENSITY_SCORE", 0.8)
            >>> default_target_propensity, params.TARGET_PROPENSITY_SCORE
            (0.5, 0.8)
        """

        # Make the parameter value available on the ParamStore object
        # as an attribute and store the value in a dict for
        # later write out or use in finding determining calculated params.
        setattr(self, param_name, param_value)
        self.parsed_parameter_dict[param_name] = param_value
        if recalculate_calculated_params:
            self._recalculate_calculated_params()

    def set_parameters(self, param_dict, recalculate_calculated_params=True):
        """Set the value of the params with the names of the keys in the `param_dict` dictionary to the corresponding value in the dictionary.

        Args:
            param_dict (dict): A dictionary mapping parameter names to parameter values.
            recalculate_calculated_params (bool): Indicates whether calculated params should be recalculated. Defaults to True.

        Examples
            >>> from maccabee.parameters import build_default_parameters
            >>> params = build_default_parameters()
            >>> default_target_propensity = params.TARGET_PROPENSITY_SCORE
            >>> params.set_parameters({"TARGET_PROPENSITY_SCORE": 0.8})
            >>> default_target_propensity, params.TARGET_PROPENSITY_SCORE
            (0.5, 0.8)
        """

        for param_name in param_dict:
            self.set_parameter(
                param_name, param_dict[param_name],
                recalculate_calculated_params=recalculate_calculated_params)

    def write(self):
        # TODO-FUTURE: enable parsed params to be dumped as yaml spec for later
        # reuse.
        pass

    ### PRIVATE HELPER FUNCTIONS ###
    def _find_calculated_param_value(self, param_info):
        # Evaluate the expression for the calculated param
        # supplied in param_info using the existing parameter
        # values in the parsed_parameter_dict attribute.
        expr = param_info[SchemaConstants.EXPRESSION_KEY]
        return eval(expr, globals(), self.parsed_parameter_dict)

    def _recalculate_calculated_params(self):
        # Recalculate all calculated param values
        # This is used following a change in param value.
        logger.debug("Recalculating calculated parameter values.")
        for param_name, param_info in self.calculated_parameters.items():
            param_value = self._find_calculated_param_value(param_info)
            self.set_parameter(param_name, param_value,
                recalculate_calculated_params=False)

    def _validate_param_value(self, param_info, param_value):
        # Validate the param value given in param_value
        # based on the validity conditions in param_info
        param_type = param_info[SchemaConstants.TYPE_KEY]
        if param_type == SchemaConstants.TYPE_NUMBER:
            return param_info[SchemaConstants.MIN_KEY] <= param_value <= param_info[SchemaConstants.MAX_KEY]

        elif param_type == SchemaConstants.TYPE_DICTIONARY:
            required_keys = set(param_info[SchemaConstants.DICT_KEYS_KEY])
            supplied_keys = set(param_value.keys())
            return required_keys == supplied_keys

        elif param_type == SchemaConstants.TYPE_BOOL:
            return param_value in [True, False]

        else:
            # Unknown param type, cannot validate. Fail at medium volume.
            return False

    ### TEMPORARY HARD CODED SOLUTION FOR DYNAMIC PARAMS ###

    # TODO-FUTURE: provide a way to specify sampling functions in param spec file to avoid this hard coding.

    # TODO-FUTURE: consider allowing parameterization of the sampling functions
    # below. The current thinking is that any customization will likely involve
    # replacing the whole function rather than only changing sampling parameters.

    def sample_subfunction_constants(self, size=1):
        return _non_zero_uniform_sampler(
            abs_low=0.25, abs_high=10, size=size)

    def sample_outcome_noise(self, size=1):
        return np.random.normal(size=size)

    def sample_treatment_effect(self, size=1):
        return _non_zero_uniform_sampler(
            abs_low=0.25, abs_high=10, size=size)
