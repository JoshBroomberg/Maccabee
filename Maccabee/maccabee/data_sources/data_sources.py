import numpy as np
from .utils import build_covar_data_frame


class DataSource():
    """Abstract class which encapsulates a source of covariate data and the
    methods/data required to sample and normalize this data prior to application
    of the treatment/outcome functions as well as meta-data used during
    function application: covar names, data type etc.

    The key abstract method is :meth:`_generate_covar_df` which returns a DataFrame
    containing the covariate observations.

    Args:
        covar_names (list): `covar_names` is a list of the string names of the covariates present in the DataFrame produced by `_generate_covar_df`.

        discrete_covar_names (list): `discrete_covar_names` is a list of the string names of the discrete covariates present in the DataFrame produced by `_generate_covar_df`.

        normalize (bool): `normalize` indicates whether the DataFrame returned by :meth:`_generate_covar_df` should be normalized prior to use by applying the :meth:`_normalize_covariate_data` method. The default normalize scheme provided by this method assumes a normal distribution over the data in each continuous covariate and leaves discrete covariates as is. Defaults to `True`.

    Attributes:
        covar_names: a list of the string names of the covariates present in the DataFrame produced by `_generate_covar_df`.
        discrete_covar_names: list of the string names of the discrete covariates present in the DataFrame produced by :meth:`_generate_covar_df`.
        normalize: indicates whether the DataFrame returned by :meth:`_generate_covar_df`` will be normalized prior to use.
    """

    def __init__(self, covar_names, discrete_covar_names, normalize=True):
        self.covar_names = covar_names
        self.discrete_covar_names = discrete_covar_names
        self.normalize = normalize

    def _generate_covar_df(self):
        """Short summary.

        Returns:
            type: Description of returned object.

        Raises:
            ExceptionName: Why the exception is raised.

        Examples
            Examples should be written in doctest format, and
            should illustrate how to use the function/class.
            >>>

        """
        raise NotImplementedError

    def _normalize_covariate_data(self, covariate_data):
        discrete_column_indeces = [
            covariate_data.columns.get_loc(name)
            for name in self.discrete_covar_names
        ]

        included_filter = np.ones(covariate_data.shape[1])
        included_filter[discrete_column_indeces] = 0
        excluded_filter = 1 - included_filter

        X_min = np.min(covariate_data, axis=0)
        X_max = np.max(covariate_data, axis=0)

        # Amount to shift columns. Excluded cols shift 0.
        column_shifts = (-1*X_min) * included_filter

        # Amount to scale columns. Excluded cols scale by 1.
        inverse_column_scales = ((X_max - X_min) * included_filter) + excluded_filter

        # Shift and scale to [0, 1]
        normalized_data = (covariate_data + column_shifts)/inverse_column_scales

        # Shift and scale to [-1, 1]
        rescaled_data = (normalized_data * \
            ((2*included_filter) + excluded_filter)) - \
            (1*included_filter)

        return rescaled_data

    def get_covar_names(self):
        return self.covar_names

    def get_discrete_covar_names(self):
        return self.discrete_covar_names

    def get_covar_df(self):
        covar_df = self._generate_covar_df()

        if self.normalize:
            covar_df = self._normalize_covariate_data(covar_df)

        return covar_df


class StochasticDataSource(DataSource):
    def __init__(self, covar_data_generator,
        covar_names, discrete_covar_names,
        normalize=True):

        super().__init__(covar_names, discrete_covar_names, normalize)
        self.covar_data_generator = covar_data_generator

    def _generate_covar_df(self):
        covar_data = self.covar_data_generator()
        covar_df = build_covar_data_frame(covar_data, self.covar_names)
        return covar_df

class StaticDataSource(DataSource):
    def __init__(self, covar_data,
        covar_names, discrete_covar_names,
        normalize=True):

        super().__init__(covar_names, discrete_covar_names, normalize=False)

        covar_df = build_covar_data_frame(covar_data, covar_names)
        self.static_covar_df = covar_df

        if normalize:
            self.static_covar_df = self._normalize_covariate_data(
                self.static_covar_df)

    def _generate_covar_df(self):
        return self.static_covar_df
