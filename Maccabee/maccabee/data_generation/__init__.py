"""This module contains the classes and functions responsible for data generation. The :class:`~maccabee.data_generation.data_generating_process.DataGeneratingProcess` class is central to the data generation process: :class:`~maccabee.data_generation.data_generating_process.DataGeneratingProcess` instances are used to sample :class:`~maccabee.data_generation.data_generating_process.DataGeneratingProcess` instances (sampled :term:`DGPs <DGP>`). :class:`~maccabee.data_generation.data_generating_process.DataGeneratingProcess` instances - either sampled as above or concretely defined - are then used to sample :class:`~maccabee.data_generation.generated_data_set.GeneratedDataSet` instances (sampled data sets). Models are then benchmarked against these sampled data sets.

This module is comprised of three submodules which align with the three components of the data generation process as outlined above.

.. note::

  For convenience, all the classes and functions that are split across the three submodules below can be imported directly from the parent :mod:`maccabee.data_generation` module.
"""

from .data_generating_process import *
from .data_generating_process_sampler import *
from .generated_data_set import *
