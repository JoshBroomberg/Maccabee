"""This module contains the Generated Data Set class which represents Generated Data."""

from ..constants import Constants
import numpy as np
import pandas as pd


class GeneratedDataSet():
    '''
    Philosophy: this is an external facing object, so we switch
    to easily usable names and a nice API.
    '''
    # TODO: write tooling to go to and from file to support static
    # benchmarking runs in future.

    # TODO: write tooling for convenient creation of GeneratedDataSet objects
    # from standard data frames.

    def __init__(self,
        observed_covariate_data,
        observed_outcome_data,
        oracle_outcome_data,
        transformed_covariate_data=None):

        self.observed_covariate_data = observed_covariate_data
        self.observed_outcome_data = observed_outcome_data
        self.oracle_outcome_data = oracle_outcome_data
        self.transformed_covariate_data = transformed_covariate_data

    # TODO: this external API maps to the various component names defined in constants. It could be implemented programtically to avoid the mapping in data_metrics.py.

    @property
    def X(self):
        return self.observed_covariate_data

    @property
    def T(self):
        return self.observed_outcome_data[Constants.TREATMENT_ASSIGNMENT_NAME]

    @property
    def Y(self):
        return self.observed_outcome_data[Constants.OBSERVED_OUTCOME_NAME]

    @property
    def Y0(self):
        return self.oracle_outcome_data[
            Constants.POTENTIAL_OUTCOME_WITHOUT_TREATMENT_NAME]

    @property
    def Y1(self):
        return self.oracle_outcome_data[
            Constants.TREATMENT_EFFECT_NAME]

    @property
    def TE(self):
        return self.oracle_outcome_data[
            Constants.POTENTIAL_OUTCOME_WITH_TREATMENT_NAME]

    @property
    def observed_data(self):
        '''
        Assemble and return the observable data.
        '''

        return self.observed_covariate_data.join(self.observed_outcome_data)

    # Estimand accessors

    @property
    def ITE(self):
        return self.Y1 - self.Y0

    @property
    def ATE(self):
        return np.mean(self.ITE)

    def ground_truth(self, estimand):
        if estimand == Constants.Model.ATE_ESTIMAND:
            return self.ATE
        elif estimand == Constants.Model.ITE_ESTIMAND:
            return self.ITE
        else:
            raise Exception("Unrecognized estimand.")
