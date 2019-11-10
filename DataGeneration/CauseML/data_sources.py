import numpy as np
import pandas as pd
from .constants import Constants

def load_cpp():
    cpp_covars = load_covars_from_path(Constants.Data.CPP_PATH)

    # TODO: remove this hacked column limit...
    cpp_covars = cpp_covars[[f"x_{i}" for i in range(1, 20)]]
    Constants.Data.CPP_DISCRETE_COVARS = [i for i in Constants.Data.CPP_DISCRETE_COVARS if i < 20]
    cpp_covars = cpp_covars.head(200)
    normalized_cpp = _normalize_covariate_data(
        cpp_covars, exclude_columns=Constants.Data.CPP_DISCRETE_COVARS)

    return normalized_cpp

def load_lalonde():
    lalonde_covars = load_covars_from_path(Constants.Data.LALONDE_PATH)
    normalized_lalonde = _normalize_covariate_data(
        lalonde_covars, exclude_columns=Constants.Data.LALONDE_DISCRETE_COVARS)

    return normalized_lalonde

def load_covars_from_path(covar_path):
    '''
    Fetches covariate data from the supplied path. This function expects
    a CSV file with column names in the first row.
    '''

    return pd.read_csv(covar_path)

def load_random_normal_covariates(
    n_covars = 20, n_observations = 1000,
    mean=0, std=5):
    '''
    Generate random normal covariate data-frame based on the
    supplied parameters.
    '''

    # Generate random covariates and name sequentially
    covar_data = np.random.normal(
        loc=mean, scale=std, size=(n_observations, n_covars))

    covar_data = _normalize_covariate_data(covar_data)
    covar_names = np.array([f"X{i}" for i in range(n_covars)])

    # Build DF
    return _build_covar_data_frame(
        covar_data, covar_names, np.arange(n_observations))

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
    column_scales = ((X_max - X_min) * included_filter) + excluded_filter

    # Shift and scale to [0, 1]
    normalized_data = (covariate_data + column_shifts)/column_scales

    # Shift and scale to [-1, 1]
    rescaled_data = (normalized_data * \
        ((2*included_filter) + excluded_filter)) - \
        (1*included_filter)

    return rescaled_data
