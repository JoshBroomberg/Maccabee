from maccabee.data_generation import DataGeneratingProcess, data_generating_method
from ..constants import Constants
from ..utilities import evaluate_expression
from ..modeling.models import CausalModel
import numpy as np
import sympy as sp
import pandas as pd

# RPY2 is used an interconnect between Python and R. It allows
# my to run R code from python which makes this experimentation
# process smoother.
import rpy2
from rpy2.robjects import IntVector, FloatVector, Formula
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
numpy2ri.activate()

stats = importr('stats') # standard regression package
matching = importr('Matching') # GenMatch package

MILD_NONLINEARITY = [2]
MODERATE_NONLINEARITY = [2, 4, 7]
MILD_NONADDITIVITY = [(1,3, 0.5), (2, 4, 0.7), (4,5, 0.5), (5,6, 0.5)]
MODERATE_NONADDITIVITY = [
        (1,3, 0.5),
        (2, 4, 0.7),
        (3,5, 0.5),
        (4,6, 0.7),
        (5,7, 0.5),
        (1,6, 0.5),
        (2,3, 0.7),
        (3,4, 0.5),
        (4,5, 0.5),
        (5,6, 0.5)]

GENMATCH_SPECS = {
    "A": ([], []),
    "B": (MILD_NONLINEARITY, []),
    "C": (MODERATE_NONLINEARITY, []),
    "D": ([], MILD_NONADDITIVITY),
    "E": (MILD_NONLINEARITY, MILD_NONADDITIVITY),
    "F": ([], MODERATE_NONADDITIVITY),
    "G": (MODERATE_NONLINEARITY, MODERATE_NONADDITIVITY)
}

class GenmatchDataGeneratingProcess(DataGeneratingProcess):
    def __init__(self,
                 quadratic_indeces, interactions_list,
                 n_observations, analysis_mode):

        super().__init__(n_observations, analysis_mode)

        # Var count
        self.n_vars = 11
        self.covar_names = [f"X_{i}" for i in range(self.n_vars)]
        self.covar_symbols = np.array(sp.symbols(self.covar_names))

        # Data types (default is standard normal)
        self.binary_indeces = [1, 3, 5, 6, 8, 9]

        self.assignment_weights = np.array(
            [0, 0.8, -0.25, 0.6, -0.4, -0.8, -0.5, 0.7, 0, 0, 0])

        self.outcome_weights = np.array(
            [-3.85, 0.3, -0.36, -0.73, -0.2, 0, 0, 0, 0.71, -0.19, 0.26])
        self.true_treat_effect = -0.4

        self.quad_terms_indeces = quadratic_indeces
        self.interactions_list = np.array(interactions_list)

        self._preprocess_functions()

    def _preprocess_functions(self):
        self.treatment_logit_terms = self.assignment_weights * self.covar_symbols


         # Add quad terms
        if len(self.quad_terms_indeces) > 0:
            self.treatment_logit_terms = np.append(
                self.treatment_logit_terms, (
                    self.assignment_weights[self.quad_terms_indeces] *
                    self.covar_symbols[self.quad_terms_indeces]**2))

        # Add interact terms
        if len(self.interactions_list) > 0:
            interact_1_indeces = self.interactions_list[:, 0].astype(int)
            interact_2_indeces = self.interactions_list[:, 1].astype(int)
            interact_weights = self.interactions_list[:, 2]

            self.treatment_logit_terms = np.append(
                self.treatment_logit_terms, (
                    self.assignment_weights[interact_1_indeces] *
                       self.covar_symbols[interact_1_indeces]*
                       self.covar_symbols[interact_2_indeces]*
                       interact_weights))

        self.treatment_logit_expression = np.sum(self.treatment_logit_terms)

        self.base_outcome_terms = self.outcome_weights * self.covar_symbols
        self.base_outcome_expression = np.sum(self.base_outcome_terms)

    @data_generating_method(Constants.COVARIATES_NAME, [])
    def _generate_observed_covars(self, input_vars):
        X = np.random.normal(loc=0.0, scale=1.0, size=(
            self.n_observations, self.n_vars - 1))

        # Add bias/intercept dummy column
        X = np.hstack([np.ones((self.n_observations, 1)), X])

        # Make binary columns binary.
        for var in self.binary_indeces:
            X[:, var-1] = (X[:, var -1] > 0).astype(int)

        return pd.DataFrame(X, columns=self.covar_names)

    @data_generating_method(
        Constants.TRANSFORMED_COVARIATES_NAME,
        [Constants.COVARIATES_NAME],
        analysis_mode_only=True)
    def _generate_transformed_covars(self, input_vars):
        # Generate the values of all the transformed covariates by running the
        # original covariate data through the transforms used in the outcome and
        # treatment functions.

        observed_covariate_data = input_vars[Constants.COVARIATES_NAME]

        all_transforms = list(set(self.base_outcome_terms).union(
            self.treatment_logit_terms))

        data = {}
        for index, transform in enumerate(all_transforms):
            data[f"{Constants.TRANSFORMED_COVARIATES_NAME}{index}"] = \
                evaluate_expression(transform, observed_covariate_data)

        return pd.DataFrame(data)

    @data_generating_method(Constants.PROPENSITY_SCORE_NAME, [Constants.COVARIATES_NAME])
    def _generate_true_propensity_scores(self, input_vars):
        observed_covariate_data = input_vars[Constants.COVARIATES_NAME]

        logits = evaluate_expression(
            self.treatment_logit_expression,
            observed_covariate_data)

        return 1/(1 + np.exp(-1*logits))

    @data_generating_method(
        Constants.POTENTIAL_OUTCOME_WITHOUT_TREATMENT_NAME,
        [Constants.COVARIATES_NAME])
    def _generate_outcomes_without_treatment(self, input_vars):
        observed_covariate_data = input_vars[Constants.COVARIATES_NAME]

        return evaluate_expression(
            self.base_outcome_expression,
            observed_covariate_data)

    @data_generating_method(
        Constants.TREATMENT_EFFECT_NAME,
        [Constants.COVARIATES_NAME])
    def _generate_treatment_effects(self, input_vars):
        observed_covariate_data = input_vars[Constants.COVARIATES_NAME]
        return self.true_treat_effect

class LogisticPropensityMatchingCausalModel(CausalModel):
    def __init__(self, dataset):
        self.dataset = dataset
        self.data = dataset.observed_data.drop("Y", axis=1)

    def fit(self):
        fmla = Formula('y ~ X')
        env = fmla.environment
        env['X'] = self.dataset.X.to_numpy()
        env['y'] = IntVector(self.dataset.T)

        # Run propensiy regression
        fitted_logistic = stats.glm(fmla, family="binomial")

        propensity_scores = fitted_logistic.rx2("fitted.values")

        # Run matching on prop scores
        self.match_out = matching.Match(
            Y=FloatVector(self.dataset.Y),
            Tr=IntVector(self.dataset.T),
            X=propensity_scores,
            replace=True)

    def estimate_ITE(self):
        raise NotImplementedError

    def estimate_ATE(self):
        return np.array(self.match_out.rx2("est").rx(1,1))[0]
