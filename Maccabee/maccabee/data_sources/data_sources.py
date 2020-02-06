"""This module contains :class:`DataSource`-derived objects that standardize access to and management of different sources of covariate data and meta-data used by Maccabee DGPs."""

import numpy as np
from .utils import build_covar_data_frame


class DataSource():
    """An abstract class that defines the encapsulation logic used to store, process and access covariate data and meta-data. The external API provides clean access to the data required for the sampling and application of treatment/outcome functions. Concrete implementations are responsible for the different loading/sampling and normalization schemes required to handle different static/stochastic covariate data.

    * The primary (abstract) method is :meth:`_generate_covar_df` which returns an unnormalized :class:`DataFrame <pandas.DataFrame>` that contains the covariate observations and covariate names.
    * The concrete method :meth:`_normalize_covariate_data` provides a default normalization scheme for the data in the covariate :class:`DataFrame <pandas.DataFrame>`.
    * The methods :meth:`get_covar_df`, :meth:`get_covar_names` , and :meth:`get_discrete_covar_names` provide the external API which is used to access the covariate data/meta-data during DGP and data sampling.

    Args:
        covar_names (list): `covar_names` is a list of the string names of the covariates present in the :class:`DataFrame <pandas.DataFrame>` produced by :meth:`_generate_covar_df`.

        discrete_covar_names (list): `discrete_covar_names` is a list of the string names of the discrete covariates present in the :class:`DataFrame <pandas.DataFrame>` produced by :meth:`_generate_covar_df`.

        normalize (bool): `normalize` indicates whether the covariates in the :class:`DataFrame <pandas.DataFrame>` returned by :meth:`_generate_covar_df` should be normalized prior to use by applying the :meth:`_normalize_covariate_data` method. The default normalize scheme provided by this method assumes a normal distribution over the data in each continuous covariate and leaves discrete covariates as is. Defaults to `True`.

    Attributes:
        covar_names: a list of the string names of the covariates present in the covariate :class:`DataFrame <pandas.DataFrame>` produced by `_generate_covar_df`.
        discrete_covar_names: list of the string names of the discrete covariates present in the covariate :class:`DataFrame <pandas.DataFrame>` produced by :meth:`_generate_covar_df`.
        normalize: indicates whether the covariate :class:`DataFrame <pandas.DataFrame>` returned by :meth:`_generate_covar_df` will be normalized prior to use.
    """

    def __init__(self, covar_names, discrete_covar_names, normalize=True):
        self.covar_names = covar_names
        self.discrete_covar_names = discrete_covar_names
        self.normalize = normalize

    def _generate_covar_df(self):
        """Abstract method which, when implemented, returns a :class:`DataFrame <pandas.DataFrame>` that contains the covariate observations and covariate names as the column names. This may involve sampling a joint distribution over the covariates, reading a static set of covariates from disk/memory etc.

        Returns:
            A :class:`DataFrame <pandas.DataFrame>`: a :class:`DataFrame <pandas.DataFrame>` that contains the covariate observations and covariate names as the column names.

        Raises:
            NotImplementedError: This is an abstract function which is concretized in inheriting classes.
        """
        raise NotImplementedError

    def _normalize_covariate_data(self, covar_df):
        """This method normalizes the covariate data returned by :meth:`_generate_covar_df` in preparation for DGP sampling. If the :class:`DataSource` is to be used with sampled DGPs then all continuous covariates should be 0 mean and have an approximate standard deviation of 1. Symmetry in the covariate distributions is not required but will improve the ability of the sampling process to achieve the desired distributional setting. See the docs for the :class:`maccabee.data_generation.data_generating_process.DataGeneratingProcess` for more detail).

        Args:
            covar_df (:class:`DataFrame <pandas.DataFrame>`): The :class:`DataFrame <pandas.DataFrame>` which contains the unnormalized covariate observations.

        Returns:
            :class:`DataFrame <pandas.DataFrame>`: a :class:`DataFrame <pandas.DataFrame>` which contains the normalized covariate observations.
        """

        discrete_column_indeces = [
            covar_df.columns.get_loc(name)
            for name in self.discrete_covar_names
        ]

        included_filter = np.ones(covar_df.shape[1])
        included_filter[discrete_column_indeces] = 0
        excluded_filter = 1 - included_filter

        X_min = np.min(covar_df, axis=0)
        X_max = np.max(covar_df, axis=0)

        # Amount to shift columns. Excluded cols shift 0.
        column_shifts = (-1*X_min) * included_filter

        # Amount to scale columns. Excluded cols scale by 1.
        inverse_column_scales = ((X_max - X_min) * included_filter) + excluded_filter

        # Shift and scale to [0, 1]
        normalized_data = (covar_df + column_shifts)/inverse_column_scales

        # Shift and scale to [-1, 1]
        rescaled_data = (normalized_data * \
            ((2*included_filter) + excluded_filter)) - \
            (1*included_filter)

        return rescaled_data

    def get_covar_names(self):
        """Accessor method for :attr:`covar_names`.

        Returns:
            list: The list of string names of the covariates in this data source.

        Examples
            >>> data_source = DataSource(covar_names=["X1", "X2", "X3"], discrete_covar_names=["X1"])
            >>> data_source.get_covar_names()
            ["X1", "X2", "X3"]

        """
        return self.covar_names

    def get_discrete_covar_names(self):
        """Accessor method for :attr:`discrete_covar_names`.

        Returns:
            list: The list of string names of the discrete covariates in this data source.

        Examples
            >>> data_source = DataSource(covar_names=["X1", "X2", "X3"], discrete_covar_names=["X1"])
            >>> data_source.get_discrete_covar_names()
            ["X1"]
        """
        return self.discrete_covar_names

    def get_covar_df(self):
        """Main API method that is used by external classes to access the generated covariate data.

        Returns:
            :class:`DataFrame <pandas.DataFrame>`: The :class:`DataFrame <pandas.DataFrame>` containing normalized covariate observations and covariate names.
        """
        covar_df = self._generate_covar_df()

        if self.normalize:
            covar_df = self._normalize_covariate_data(covar_df)

        return covar_df


class StochasticDataSource(DataSource):
    """A concrete implementation of the abstract :class:`DataSource`, which can be used for sampling stochastic sources of covariate data by automatically using a supplied sampling function for each call to :meth:`get_covar_df`.

    Args:
        covar_data_generator (function): a function which samples some joint distribution over covariates and returns a 2D :class:`numpy.ndarray` of covariate data.

        covar_names (list): see :class:`DataSource`.

        discrete_covar_names (list): see :class:`DataSource`.

        normalize (bool): see :class:`DataSource`.

    """
    def __init__(self, covar_data_generator,
        covar_names, discrete_covar_names,
        normalize=True):

        super().__init__(covar_names, discrete_covar_names, normalize)
        self._covar_data_generator = covar_data_generator

    def _generate_covar_df(self):
        """Concretized implementation of :meth:`DataSource._generate_covar_df` which calls the `covar_data_generator` supplied at initialization, and returns a :class:`DataFrame <pandas.DataFrame>` containing the data returned by it and the :attr:`covar_names`.

        Returns:
            :class:`DataFrame <pandas.DataFrame>`: a :class:`DataFrame <pandas.DataFrame>` containing sampled covariate observations.
        """
        covar_data = self._covar_data_generator()
        covar_df = build_covar_data_frame(covar_data, self.covar_names)
        return covar_df

class StaticDataSource(DataSource):
    """A concrete implementation of the abstract :class:`DataSource`, which can be used for sampling static sources of covariate data (ones which do not change).

    Args:
        static_covar_data (:class:`numpy.ndarray`): a 2D :class:`numpy.ndarray` of covariate data.

        covar_names (list): see :class:`DataSource`.

        discrete_covar_names (list): see :class:`DataSource`.

        normalize (bool): see :class:`DataSource`.

    """

    def __init__(self, static_covar_data,
        covar_names, discrete_covar_names,
        normalize=True):

        super().__init__(covar_names, discrete_covar_names, normalize=False)

        covar_df = build_covar_data_frame(static_covar_data, covar_names)
        self.static_covar_df = covar_df

        if normalize:
            self.static_covar_df = self._normalize_covariate_data(
                self.static_covar_df)

    def _generate_covar_df(self):
        """Concretized implementation of :meth:`DataSource._generate_covar_df` which returns a :class:`DataFrame <pandas.DataFrame>` containing the data supplied in `static_covar_data` at initialization time.

        Returns:
            :class:`DataFrame <pandas.DataFrame>`: a :class:`DataFrame <pandas.DataFrame>` containing static covariate observations.
        """
        return self.static_covar_df
