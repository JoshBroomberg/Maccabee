from ..constants import Constants
import numpy as np
import pandas as pd


class DataSet():
    '''
    Philosophy: this is an external facing object, so we switch
    to easily usable names and a nice API.
    '''
    # TODO: write tooling to go to and from file to support static
    # benchmarking runs in future.

    # TODO: write tooling for convenient creation of DataSet objects
    # from standard data frames.

    def __init__(self,
        observed_covars, observed_covar_names,
        transformed_covars, transformed_covar_names,
        propensity_scores, propensity_logit, T,
        outcome_noise, Y0, TE, Y1, Y):

        self.observed_covars = observed_covars
        self.observed_covar_names = observed_covar_names
        self.transformed_covars = transformed_covars
        self.transformed_covar_names = transformed_covar_names

        self.propensity_scores = propensity_scores.reshape(-1, 1)
        self.propensity_logit = propensity_logit.reshape(-1, 1)
        self.T = T.reshape(-1, 1)
        self.outcome_noise = outcome_noise.reshape(-1, 1)
        self.Y0 = Y0.reshape(-1, 1)
        self.TE = TE.reshape(-1, 1)
        self.Y1 = Y1.reshape(-1, 1)
        self.Y = Y.reshape(-1, 1)

    @property
    def X(self):
        return self.observed_covars

    @property
    def observed_data(self):
        '''
        Assemble and return the observable data.
        '''
        data = np.hstack([
            self.observed_covars,
            self.T,
            self.Y])

        return pd.DataFrame(data,
            columns=np.hstack([self.observed_covar_names, [
                Constants.TREATMENT_ASSIGNMENT_NAME,
                Constants.OBSERVED_OUTCOME_NAME
            ]]))

    @property
    def oracle_data(self):
        '''
        Assemble and return the non-observable/oracle data.
        '''

        return pd.concat([
            self.transformed_covars,
            self.propensity_scores,
            self.propensity_logit,
            self.outcome_noise,
            self.outcome_noise,
            self.Y0,
            self.TE,
            self.Y1
        ], axis=1)

    # Estimand accessors

    @property
    def ITE(self):
        return self.TE

    @property
    def ATE(self):
        return np.mean(self.TE)

    def ground_truth(self, estimand):
        if estimand == Constants.Model.ATE_ESTIMAND:
            return self.ATE
        elif estimand == Constants.Model.ITE_ESTIMAND:
            return self.ITE
        else:
            raise Exception("Unrecognized estimand.")
