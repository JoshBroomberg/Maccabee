from ..data_generation import DataGeneratingProcess, data_generating_method
from ..data_sources.data_sources import StochasticDataSource

from ..constants import Constants
from ..utilities import evaluate_expression
from ..modeling.models import CausalModel

import numpy as np
import sympy as sp
import pandas as pd
from functools import partial

from sklearn.linear_model import LogisticRegression

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

GENMATCH_N_COVARS = 11
GENMATCH_BINARY_COVAR_INDECES = [1, 3, 5, 6, 8, 9]
GENMATCH_COVAR_NAMES = np.array([f"X{i}" for i in range(GENMATCH_N_COVARS)])

def _generate_genmatch_data(n_observations):
    covar_data = np.random.normal(loc=0.0, scale=1.0, size=(
            n_observations, GENMATCH_N_COVARS-1))

    # Add bias/intercept dummy column
    covar_data = np.hstack([np.ones((n_observations, 1)), covar_data])

    # Make binary columns binary.
    for var in GENMATCH_BINARY_COVAR_INDECES:
        covar_data[:, var-1] = (covar_data[:, var -1] > 0).astype(int)

    return covar_data

def build_genmatch_datasource(n_observations=1000):
    return StochasticDataSource(
        covar_data_generator=partial(_generate_genmatch_data, n_observations),
        covar_names=list(GENMATCH_COVAR_NAMES),
        discrete_covar_names=list(
            GENMATCH_COVAR_NAMES[GENMATCH_BINARY_COVAR_INDECES]) + ["X0"])

class GenmatchDataGeneratingProcess(DataGeneratingProcess):
    def __init__(self,
                 quadratic_indeces, interactions_list,
                 n_observations, data_analysis_mode):

        super().__init__(n_observations, data_analysis_mode)

        self.data_source = build_genmatch_datasource(n_observations)
        self.covar_symbols = np.array(sp.symbols(list(GENMATCH_COVAR_NAMES)))

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
        return self.data_source.get_covar_df()

    @data_generating_method(
        Constants.TRANSFORMED_COVARIATES_NAME,
        [Constants.COVARIATES_NAME],
        data_analysis_mode_only=True)
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
        logistic_model = LogisticRegression(solver='lbfgs', n_jobs=1)
        logistic_model.fit(
            self.dataset.X.to_numpy(), self.dataset.T.to_numpy())
        class_proba = logistic_model.predict_proba(
            self.dataset.X.to_numpy())
        propensity_scores = class_proba[:, logistic_model.classes_ == 1].flatten()

        # Run matching on prop scores
        self.match_out = matching.Match(
            Y=FloatVector(self.dataset.Y),
            Tr=IntVector(self.dataset.T),
            X=FloatVector(propensity_scores),
            replace=True,
            version="fast")

    def estimate_ITE(self):
        raise NotImplementedError

    def estimate_ATE(self):
        return np.array(self.match_out.rx2("est").rx(1,1))[0]
