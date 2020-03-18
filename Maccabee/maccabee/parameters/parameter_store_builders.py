import yaml
from ..constants import Constants
from .parameter_store import ParameterStore

ParamFileConstants = Constants.ParamFilesAndPaths


def build_default_parameters():
    """Return a :class:`~maccabee.parameters.parameter_store.ParameterStore` instance based on the :term:`default parameter specification file`.

    Returns:
        :class:`~maccabee.parameters.parameter_store.ParameterStore`: A :class:`~maccabee.parameters.parameter_store.ParameterStore` instance.

    Examples
        >>> from maccabee.parameters import build_default_parameters
        >>> build_default_parameters()
        <maccabee.parameters.parameter_store.ParameterStore at ...>

    """
    return ParameterStore(parameter_spec_path=ParamFileConstants.DEFAULT_SPEC_PATH)

def build_parameters_from_specification(parameter_spec_path):
    """Return a :class:`~maccabee.parameters.parameter_store.ParameterStore` instance based on the provided :term:`parameter specification file`.

    Args:
        parameter_spec_path (string): A string path to the parameter specification file.

    Returns:
        :class:`~maccabee.parameters.parameter_store.ParameterStore`: A :class:`~maccabee.parameters.parameter_store.ParameterStore` instance.

    Raises:
        ParameterSpecificationException: if there is a problem with the parameter specification. More detail will be provided by the exception subclass and message.

    Examples
        >>> from maccabee.parameters import build_parameters_from_specification
        >>> build_parameters_from_specification("./param_spec.yml")
        <maccabee.parameters.parameter_store.ParameterStore at ...>

    """
    return ParameterStore(parameter_spec_path=parameter_spec_path)

def build_parameters_from_axis_levels(metric_levels, save=False):
    """Return a :class:`~maccabee.parameters.parameter_store.ParameterStore` instance based on the provided :term:`parameter specification file`.

    This function uses the values in the :download:`metric_level_parameter_specifications.yml </../../maccabee/parameters/metric_level_parameter_specifications.yml>` file to change the parameter values in order to achieve the desired level/position on the given :term:`axes <distributional problem space axis>`.

    Args:
        parameter_spec_path (dict): A dictionary mapping :term:`data axis <distributional problem space axis>` names to a data axis level. Axis names are available as constants in :class:`maccabee.constants.Constants.AxisNames` and axis levels available as constants in :class:`maccabee.constants.Constants.AxisLevels`.

    Returns:
        :class:`~maccabee.parameters.parameter_store.ParameterStore`: A :class:`~maccabee.parameters.parameter_store.ParameterStore` instance.

    Examples
        >>> from maccabee.parameters import build_parameters_from_axis_levels
        >>> from maccabee.constants import Constants
        >>> # define the axis level for treatment nonlinearity.
        >>> # all others will remain at default.
        >>> axis_levels = { Constants.AxisNames.TREATMENT_NONLINEARITY: Constants.AxisLevels.HIGH }
        >>> build_parameters_from_axis_levels(axis_levels)
        <maccabee.parameters.parameter_store.ParameterStore at ...>

    """

    params = build_default_parameters()

    with open(ParamFileConstants.AXIS_LEVEL_SPEC_PATH, "r") as metric_level_file:
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
                raise ValueError(f"{metric_level} is not a valid level for {metric_name}")
        else:
            raise ValueError(f"{metric_name} is not a valid metric")


    params._recalculate_calculated_params()

    return params
