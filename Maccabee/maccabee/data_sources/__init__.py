"""This module is comprised of the submodules listed below. The :mod:`~maccabee.data_sources.data_sources` module contains the :class:`~maccabee.data_sources.data_sources.DataSource` classes which define the internal data handling logic and external API. The :mod:`~maccabee.data_sources.data_source_builders` module contains utility functions which can be used to build :class:`~maccabee.data_sources.data_sources.DataSource` instances that correspond to commonly used covariate data sources.

.. note::

  For convenience, all the classes and functions that are split across the two submodules below can be imported directly from the parent :mod:`maccabee.data_sources` module.
"""

from .data_source_builders import *
from .data_sources import DataSource
