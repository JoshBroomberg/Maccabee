"""The benchmarking module contains the code responsible for running Monte Carlo trials and collecting metrics which measure estimator performance (performance metrics) and data distributional settings (data metrics). This module is responsible for *execution* of the experiments and metric functions. The metric functions are defined alongside the objects which they measure. See :mod:`maccabee.modeling.performance_metrics` for performance metrics and :mod:`maccabee.data_analysis.data_metrics` for data metrics

For now, this module is made up of one submodule. In future releases, it may be broken into smaller, more specialized submodules.

.. note::
  As a convenience, all functions from the single submodule of this module can be imported directly from the module itself.
"""

from .benchmarking import *
