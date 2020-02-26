"""The data analysis module contains the code responsible for calculating :term:`data metrics <data metric>` - metrics which quantify the location of a data set in the :term:`distributional problem space`. More on the theory behind these metrics can be found in Chapter 3 of the :download:`theory paper </maccabee-theory-paper.pdf>`.

The module is not responsible for actually executing these calculations, that is handled by the :mod:`maccabee.benchmarking` module. Rather, this module is responsible for defining the actual metrics used to quantify the location of a data set on each :term:`distributional problem space axis` and providing wrapper functionality to calculate multiple metrics given a :class:`~maccabee.data_generation.generated_data_set.GeneratedDataSet` instance.

This module is split into two submodules. :mod:`~maccabee.data_analysis.data_metrics` contains the metric definitions and :mod:`~maccabee.data_analysis.data_analysis` contains the code which calculates these metrics given a :class:`maccabee.data_generation.generated_data_set.GeneratedDataSet` instance and code which plots calculated metric results.

.. note::

  For convenience, all the classes and functions that are split across the submodules below can be imported directly from the parent :mod:`maccabee.data_analysis` module.
"""

from .data_analysis import *
