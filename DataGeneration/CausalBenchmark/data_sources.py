import numpy as np
import pandas as pd

def generate_random_covariates(n_covars = 20, n_observations = 1000):

    # Generate random covariates and name sequentially
    covar_data = np.random.normal(loc=0, scale=5, size=(n_observations, n_covars))
    covar_names = np.array([f"X{i}" for i in range(n_covars)])

    # Build DF
    return pd.DataFrame(
            data=covar_data,
            columns=covar_names,
            index=np.arange(n_observations))
