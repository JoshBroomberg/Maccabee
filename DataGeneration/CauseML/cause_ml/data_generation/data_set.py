from ..constants import Constants
import numpy as np

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
        observed_covars, transformed_covars,
        propensity_scores, propensity_logit, T,
        outcome_noise, Y0, TE, Y1, Y):

        self.observed_covars = observed_covars
        self.oracle_covars = oracle_covars
        self.propensity_scores = propensity_scores
        self.propensity_logit = propensity_logit
        self.T = T
        self.outcome_noise = outcome_noise
        self.Y0 = Y0
        self.TE = TE
        self.Y1 = Y1
        self.Y = Y

    @property
    def X(self):
        return self.observed_covars

    @property
    def observed_data(self):
        '''
        Assemble and return the observable data.
        '''

        observable_data = {
            Constants.COVARIATES_NAME: self.observed_covars,
            Constants.TREATMENT_ASSIGNMENT_NAME: self.T,
            Constants.OBSERVED_OUTCOME_NAME: self.Y,
        }

        return pd.DataFrame(observable_data)

    @property
    def oracle_data(self):
        '''
        Assemble and return the non-observable/oracle data.
        '''

        oracle_data = {
            Constants.TRANSFORMED_COVARIATES_NAME: self.transformed_covars,
            Constants.PROPENSITY_SCORE_NAME: self.propensity_scores,
            Constants.PROPENSITY_LOGIT_NAME: self.propensity_logit,
            Constants.OUTCOME_NOISE_NAME: self.outcome_noise,
            Constants.POTENTIAL_OUTCOME_WITHOUT_TREATMENT_NAME: self.Y0,
            Constants.TREATMENT_EFFECT_NAME: self.TE,
            Constants.POTENTIAL_OUTCOME_WITH_TREATMENT_NAME: self.Y1,
        }

        return pd.DataFrame(oracle_data)

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
