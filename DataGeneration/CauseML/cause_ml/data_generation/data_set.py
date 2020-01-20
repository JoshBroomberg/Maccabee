from ..constants import Constants
import numpy as np

class DataSet():
    # TODO: write tooling to go to and from file to support static
    # benchmarking runs in future.

    def __init__(self,
        observed_covariate_data,
        oracle_covariate_data,
        observed_outcome_data,
        oracle_outcome_data):
        # Observed and oracle/transformed covariate data.
        self.observed_covariate_data = observed_covariate_data
        self.oracle_covariate_data = oracle_covariate_data

        # Generated data
        self.observed_outcome_data = observed_outcome_data
        self.oracle_outcome_data = oracle_outcome_data

    def ground_truth(self, estimand):
        if estimand == Constants.Model.ATE_ESTIMAND:
            return self.ATE
        elif estimand == Constants.Model.ITE_ESTIMAND:
            return self.ITE
        else:
            raise Exception("Unrecognized estimand.")

    @property
    def observed_data(self):
        '''
        Assemble and return the observable data.
        '''

        return self.observed_outcome_data \
            .join(self.observed_covariate_data)

    @property
    def oracle_data(self):
        '''
        Assemble and return the non-observable/oracle data.
        '''

        return self.oracle_outcome_data.join(self.oracle_covariate_data)

    @property
    def X(self):
        return self.observed_data.drop([
            Constants.TREATMENT_ASSIGNMENT_VAR_NAME,
            Constants.OBSERVED_OUTCOME_VAR_NAME], axis=1)

    @property
    def T(self):
        return self.observed_data[Constants.TREATMENT_ASSIGNMENT_VAR_NAME]

    @property
    def Y(self):
        return self.observed_data[Constants.OBSERVED_OUTCOME_VAR_NAME]

    @property
    def Y0(self):
        return self.oracle_data[Constants.POTENTIAL_OUTCOME_WITHOUT_TREATMENT_VAR_NAME]

    @property
    def Y1(self):
        return self.oracle_data[Constants.POTENTIAL_OUTCOME_WITH_TREATMENT_VAR_NAME]

    @property
    def ITE(self):
        return self.Y1 - self.Y0

    @property
    def ATE(self):
        return np.mean(self.ITE)
