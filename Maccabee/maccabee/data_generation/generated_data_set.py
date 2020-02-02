"""This module contains the Generated Data Set class which represents Generated Data."""

from ..constants import Constants
from ..exceptions import UnknownDGPVariableException, UnknownEstimandException
import numpy as np
import pandas as pd

DGPVariables = Constants.DGPVariables

class GeneratedDataSet():
    '''
    Philosophy: this is an external facing object, so we switch
    to easily usable names and a nice API.
    '''

    #: This dictionary maps the various dgp variable names from
    #: :class:`maccabee.constants.Constants.DGPVariables` to the
    #: properties of the :class:`maccabee.data_generation.GeneratedDataSet` #: instance to allow for convenient access via :meth:`maccabee.data_generation.GeneratedDataSet.get_dgp_variable`.
    DGP_VARIABLE_ACCESSORS = {
        # Covariate Data
        DGPVariables.COVARIATES_NAME: lambda ds: ds.observed_covariate_data,
        DGPVariables.TRANSFORMED_COVARIATES_NAME: lambda ds: ds.transformed_covariate_data,

        # Observed Variables
        DGPVariables.OBSERVED_OUTCOME_NAME: lambda ds: ds.Y,
        DGPVariables.TREATMENT_ASSIGNMENT_NAME: lambda ds: ds.T,

        # Oracle
        DGPVariables.POTENTIAL_OUTCOME_WITHOUT_TREATMENT_NAME: lambda ds: ds.Y0,
        DGPVariables.POTENTIAL_OUTCOME_WITH_TREATMENT_NAME: lambda ds: ds.Y1,
        DGPVariables.TREATMENT_EFFECT_NAME: lambda ds: ds.TE,
        DGPVariables.PROPENSITY_LOGIT_NAME: lambda ds: ds.oracle_outcome_data[DGPVariables.PROPENSITY_LOGIT_NAME],
        DGPVariables.PROPENSITY_SCORE_NAME: lambda ds: ds.oracle_outcome_data[DGPVariables.PROPENSITY_SCORE_NAME],
    }

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

    def get_dgp_variable(self, var_name):
        """Short summary.

        Args:
            var_name (type): Description of parameter `var_name`.

        Returns:
            type: Description of returned object.

        Raises:
            ExceptionName: Why the exception is raised.

        Examples
            Examples should be written in doctest format, and
            should illustrate how to use the function/class.
            >>>

        """
        if var_name in self.DGP_VARIABLE_ACCESSORS:
            return self.DGP_VARIABLE_ACCESSORS[var_name](self)
        else:
            raise UnknownDGPVariableException()

    @property
    def X(self):
        return self.observed_covariate_data

    @property
    def T(self):
        return self.observed_outcome_data[DGPVariables.TREATMENT_ASSIGNMENT_NAME]

    @property
    def Y(self):
        return self.observed_outcome_data[DGPVariables.OBSERVED_OUTCOME_NAME]

    @property
    def Y0(self):
        return self.oracle_outcome_data[
            DGPVariables.POTENTIAL_OUTCOME_WITHOUT_TREATMENT_NAME]

    @property
    def Y1(self):
        return self.oracle_outcome_data[
            DGPVariables.TREATMENT_EFFECT_NAME]

    @property
    def TE(self):
        return self.oracle_outcome_data[
            DGPVariables.POTENTIAL_OUTCOME_WITH_TREATMENT_NAME]

    @property
    def observed_data(self):
        return self.observed_covariate_data.join(self.observed_outcome_data)

    # Estimand accessors

    @property
    def ITE(self):
        return self.Y1 - self.Y0

    @property
    def ATE(self):
        return np.mean(self.ITE)

    @property
    def ATT(self):
        return np.mean(self.ITE[self.T == "1"])

    def ground_truth(self, estimand):
        if estimand not in Constants.Model.ALL_ESTIMANDS:
            raise UnknownEstimandException()

        if not hasattr(self, estimand):
            raise NotImplementedError

        return getattr(self, estimand)
