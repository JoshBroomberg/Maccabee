import numpy as np

class DataSource():
    """Short summary.

    Args:
        discrete_column_names (type): Description of parameter `discrete_column_names`.
        normalize (type): Description of parameter `normalize`. Defaults to True.

    Examples
        Examples should be written in doctest format, and
        should illustrate how to use the function/class.
        >>>

    Attributes:
        discrete_column_names
        normalize

    """

    def __init__(self, discrete_column_names, normalize=True):
        """Short summary.

        Args:
            discrete_column_names (type): Description of parameter `discrete_column_names`.
            normalize (type): Description of parameter `normalize`. Defaults to True.

        Returns:
            type: Description of returned object.

        Raises:
            ExceptionName: Why the exception is raised.

        Examples
            Examples should be written in doctest format, and
            should illustrate how to use the function/class.
            >>>

        """
        self.discrete_column_names = discrete_column_names
        self.normalize = normalize


    def _get_covariate_dataframe(self):
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
            for name in self.discrete_column_names
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

    def get_data(self):
        covar_df = self._get_covariate_dataframe()

        if self.normalize:
            covar_df = self._normalize_covariate_data(covar_df)

        return covar_df

class StochasticDataSource(DataSource):
    def __init__(self, covar_df_generator, discrete_column_names, normalize=True):
        super().__init__(discrete_column_names, normalize)
        self.covar_df_generator = covar_df_generator

    def _get_covariate_dataframe(self):
        return self.covar_df_generator()

class StaticDataSource(DataSource):
    def __init__(self, covar_df, discrete_column_names, normalize=True):
        super().__init__(discrete_column_names, normalize=False)

        self.static_covar_df = covar_df
        if normalize:
            self.static_covar_df = self._normalize_covariate_data(
                self.static_covar_df)

    def _get_covariate_dataframe(self):
        return self.static_covar_df
