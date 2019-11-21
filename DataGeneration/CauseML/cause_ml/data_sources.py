import numpy as np
import pandas as pd
from .constants import Constants
from .utilities import random_covar_matrix


class DataSource():

    def __init__(self, covariate_data, binary_column_names):
        self.original_covariate_data = covariate_data
        self.binary_column_names = binary_column_names

        self.binary_column_indeces = [
            self.original_covariate_data.columns.get_loc(name)
            for name in self.binary_column_names
        ]

        self.normalized_covariate_data = _normalize_covariate_data(
            self.original_covariate_data,
            exclude_columns=self.binary_column_indeces)

    def get_data(self):
        return self.normalized_covariate_data

    # TODO: Build in a mechanism to limit/control the size of the
    # returned dataset? See below
    # cpp_covars = cpp_covars[[f"x_{i}" for i in range(1, 20)]]
    # Constants.Data.CPP_DISCRETE_COVARS = [i for i in Constants.Data.CPP_DISCRETE_COVARS if i < 20]
    # cpp_covars = cpp_covars.head(200)

def load_cpp():
    cpp_covars = load_covars_from_path(Constants.Data.CPP_PATH)


    cpp_data_source = DataSource(
        covariate_data=cpp_covars,
        binary_column_names=Constants.Data.CPP_DISCRETE_COVARS)

    return cpp_data_source

def load_lalonde():
    lalonde_covars = load_covars_from_path(Constants.Data.LALONDE_PATH)

    lalone_data_source = DataSource(
        covariate_data=lalonde_covars,
        binary_column_names=Constants.Data.LALONDE_DISCRETE_COVARS)

    return lalone_data_source

def load_covars_from_path(covar_path):
    '''
    Fetches covariate data from the supplied path. This function expects
    a CSV file with column names in the first row.
    '''

    return pd.read_csv(covar_path)

def load_random_normal_covariates(
    n_covars = 20, n_observations = 1000,
    mean=0, std=1, partial_correlation_degree=0.0):
    '''
    Generate random normal covariate data-frame based on the
    supplied parameters.
    '''

    covar = random_covar_matrix(
        dimension=n_covars,
        correlation_deg=partial_correlation_degree)

    # Generate random covariates
    covar_data = np.random.multivariate_normal(
        mean=np.full((n_covars,), mean),
        cov=std*covar,
        size=n_observations)

    # Normalize the generated data (this nullifies the impact of the
    # mean and std params but they are supplied for potential later use)
    covar_data = _normalize_covariate_data(covar_data)

    # Name sequentially
    covar_names = np.array([f"X{i}" for i in range(n_covars)])

    # Build DF
    norm_covars = _build_covar_data_frame(
        covar_data, covar_names, np.arange(n_observations))

    norm_data_source = DataSource(
        covariate_data=norm_covars,
        binary_column_names=[])

    return norm_data_source

def _build_covar_data_frame(data, column_names, index):
    return pd.DataFrame(
            data=data,
            columns=column_names,
            index=index)

def _normalize_covariate_data(covariate_data, exclude_columns=[]):
    included_filter = np.ones(covariate_data.shape[1])
    included_filter[exclude_columns] = 0
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
