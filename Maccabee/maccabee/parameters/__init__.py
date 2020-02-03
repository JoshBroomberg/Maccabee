"""This module contains the classes and files which are used to specify the sampling parameters that control aspects of the DGP sampling process. IE, the parameters which control the operation of  :class:`~maccabee.data_generation.data_generating_process_sampler.DataGeneratingProcessSampler` class which produces :class:`~maccabee.data_generation.data_generating_process.SampledDataGeneratingProcess` instances.

The parameters which control the DGP sampling process are non-trivial both in number and kind. The parameters can take the form of:

* Single values with different valid values.
* Dictionaries with sets of required keys
* Calculated parameters which are derived from other parameters in a non-trivial way through some one-off calculation.
* Functional parameters (specified using python code) which allow for the injection of arbitrary analytical expressions at appropriate points in the DGP sampling process. For example, different sampling distributions can be specified using functional parameters that contain arbitrary sampling code.

The complexity of the parameterization necessitates an encapsulation layer so that the :class:`~maccabee.data_generation.data_generating_process_sampler.DataGeneratingProcessSampler` class can consume parameters without worrying about the detail of their specification. The :class:`~maccabee.parameters.parameter_store.ParameterStore` class serves this role, acting as a unified store for all of the parameters and providing a simple access interface.

Instances of the :class:`~maccabee.parameters.parameter_store.ParameterStore` class can be created in a variety of ways depending on user requirements. Typically, instances are created automatically by the :mod:`maccabee.benchmarking` functions based on the user's desired position for sampled DGPs in :term:`distributional problem space` in terms of levels/positions along the :term:`axes <distributional problem space axis>` of the space. This allows users to specify, at a high-level, the parameterization they want and leave the detailed parameter value specification to Maccabee. Under the hood, this approach uses the :func:`~maccabee.parameters.parameter_store_builders.build_parameters_from_axis_levels` function. See the docs for that function or the :mod:`~maccabee.benchmarking.benchmarking` docs for more on this approach.

Beyond specifying axis levels, there are also ways to specify parameter values more directly. The :func:`~maccabee.parameters.parameter_store_builders.build_default_parameters` function returns a :class:`~maccabee.parameters.parameter_store.ParameterStore` instance containing a set of default parameter values. This instance can then act as a starting point for small modifications using the :meth:`~maccabee.parameters.parameter_store.ParameterStore.set_parameter`/:meth:`~maccabee.parameters.parameter_store.ParameterStore.set_parameters` methods. Finally, the :func:`~maccabee.parameters.parameter_store_builders.build_parameters_from_specification` function can be used to generate a :class:`~maccabee.parameters.parameter_store.ParameterStore` instance from a :term:`parameter specification file`. This allows for granular control over parameter values. Users interested in setting custom parameter values should look at the :download:`parameter_schema.yml </../../maccabee/parameters/parameter_schema.yml>` file. This contains the names, validity conditions and descriptions of all the sampling parameters.

See the :class:`~maccabee.parameters.parameter_store.ParameterStore` docs for detail on the set of sampling parameters, parameter specification files, and the parameter schema file.

.. note::
  For convenience, the classes and functions in the :mod:`maccabee.parameters.parameter_store` submodule can be imported directly from this module.
"""

from .parameter_store import *
from .parameter_store_builders import *
