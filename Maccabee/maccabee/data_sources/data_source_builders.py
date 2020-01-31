import numpy as np
import pandas as pd
from functools import partial

from ..constants import Constants

from .data_sources import StaticDataSource, StochasticDataSource
from .utils import random_covar_matrix, load_covars_from_csv_path, build_covar_data_frame


def build_csv_datasource(csv_path, discrete_covar_names):
    covar_data, covar_names = load_covars_from_csv_path(csv_path)

    return StaticDataSource(
        covariate_data=covar_data,
        covar_names=covar_names,
        discrete_covar_names=discrete_covar_names)

def build_cpp_datasource():
    return build_csv_datasource(
        Constants.Data.CPP_PATH, Constants.Data.CPP_DISCRETE_COVARS)

def build_lalonde_datasource():
    return build_csv_datasource(
        Constants.Data.LALONDE_PATH, Constants.Data.LALONDE_DISCRETE_COVARS)

def build_random_normal_datasource(
    n_covars = 20, n_observations = 1000,
    partial_correlation_degree=0.0):
    '''
    Generate random normal covariate data-frame based on the
    supplied parameters.
    '''

    # Name covars sequentially
    covar_names = np.array([f"X{i}" for i in range(n_covars)])

    gen_random_normal_data = partial(_gen_random_normal_data,
        n_covars, n_observations, partial_correlation_degree)

    return StochasticDataSource(
        covar_data_generator=gen_random_normal_data,
        covar_names=covar_names,
        discrete_covar_names=[])

def _gen_random_normal_data(n_covars, n_observations,
    correlation_deg):

    covar = random_covar_matrix(
        dimension=n_covars,
        correlation_deg=partial_correlation_degree)

    # Generate random covariates
    covar_data = np.random.multivariate_normal(
        mean=np.full((n_covars,), mean),
        cov=std*covar,
        size=n_observations)

    return covar_data
