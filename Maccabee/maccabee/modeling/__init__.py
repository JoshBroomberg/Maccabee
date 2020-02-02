"""This module contains the code used to define and evaluate :term:`causal models <causal model>`. Causal models are responsible for estimating causal effects from observational data (a subset of the data available as part of :class:`~maccabee.data_generation.generated_data_set.GeneratedDataSet` instance).

The code in this model is split into two submodules. The :mod:`maccabee.modeling.models` submodule defines the base :class:`~maccabee.modeling.models.CausalModel` class which all concrete models inherit from. It also contains some derived example models. Second, the :mod:`maccabee.modeling.performance_metrics` submodule defines the metrics used to evaluate all causal models.
"""
