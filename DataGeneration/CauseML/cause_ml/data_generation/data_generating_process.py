from ..constants import Constants
from .data_set import DataSet
from ..utilities import evaluate_expression
import pandas as pd
import numpy as np

class DataGeneratingProcess():
    def __init__(self,
        params,
        observed_covariate_data,
        outcome_covariate_transforms,
        treatment_covariate_transforms,
        treatment_assignment_function,
        treatment_effect_subfunction,
        base_outcome_subfunction,
        generate_oracle_covariate_data=True):

        self.params = params

        # DGP COMPONENTS

        # Sampled covariate transforms for the treat and outcome functions.
        self.outcome_covariate_transforms = outcome_covariate_transforms
        self.treatment_covariate_transforms = treatment_covariate_transforms

        # Treatment assignment function and subfunctions
        self.treatment_assignment_logit_function = None
        self.treatment_assignment_function = treatment_assignment_function

        # Outcome function and subfunctions
        self.treatment_effect_subfunction = treatment_effect_subfunction
        self.base_outcome_subfunction = base_outcome_subfunction
        self.outcome_function = None

        # DATA
        self.observed_covariate_data = observed_covariate_data
        self.n_observations = self.observed_covariate_data.shape[0]

        self._preprocess_component_functions(generate_oracle_covariate_data)

    def _preprocess_component_functions(self, generate_oracle_covariate_data):
        '''
        This function is run at initialization and performs calculations
        which do not need to be repeated with each data generation.
        '''

        # Generate treatment assignment propensity scores.
        self.propensity_scores = evaluate_expression(
            self.treatment_assignment_function,
            self.observed_covariate_data)

        self.logit_values = np.log(self.propensity_scores/(1-self.propensity_scores))

        # Generate base outcomes and treatment effects. This is more efficient
        # than using the complete outcome function which required re-evaluating
        # the base outcome.
        self.base_outcomes = evaluate_expression(
            self.base_outcome_subfunction,
            self.observed_covariate_data)

        self.treatment_effects = evaluate_expression(
            self.treatment_effect_subfunction,
            self.observed_covariate_data)

        # Covariate transforms
        if generate_oracle_covariate_data:
            self.oracle_covariate_data = self._generate_transformed_covariate_data()
        else:
            self.oracle_covariate_data = pd.DataFrame()

    def _generate_transformed_covariate_data(self):
        '''
        Generate the values of all the transformed covariates by running the
        original covariate data through the transforms used in the outcome and
        treatment functions.
        '''
        all_transforms = list(set(self.outcome_covariate_transforms).union(
            self.treatment_covariate_transforms))

        data = {}
        for index, transform in enumerate(all_transforms):
            data[f"{Constants.TRANSFORMED_COVARIATE_PREFIX}{index}"] = \
                evaluate_expression(transform, self.observed_covariate_data)

        return pd.DataFrame(data)

    def generate_data(self):
        """Perform data generation"""

        # Sample treatment assignment given pre-calculated propensity_scores
        T = (np.random.uniform(
            size=self.n_observations) < self.propensity_scores).astype(int)

        # Only perform balance adjustment if there is some heterogeneity
        # in the propensity scores.
        if not np.all(np.isclose(self.propensity_scores, self.propensity_scores[0])):
            # Balance adjustment
            control_p_scores = self.propensity_scores.where(T == 0)
            treat_p_scores = self.propensity_scores.where(T == 1)

            num_controls = control_p_scores.count()
            n_to_switch = int(num_controls*self.params.FORCED_IMBALANCE_ADJUSTMENT)

            control_switch_targets = control_p_scores.nlargest(n_to_switch).index.values
            treat_switch_targets = treat_p_scores.nsmallest(n_to_switch).index.values

            T[control_switch_targets] = 1
            T[treat_switch_targets] = 0

        # Sample and add noise column
        noise_samples = self.params.sample_outcome_noise(size=self.n_observations)

        Y0 = self.base_outcomes + noise_samples
        Y1 = Y0 + self.treatment_effects

        # Observed outcome
        Y = (T*Y1) + ((1-T)*Y0)

        # Build outcome data frames.

        # Data available for causal inference
        observed_outcome_data = pd.DataFrame()
        observed_outcome_data[Constants.TREATMENT_ASSIGNMENT_VAR_NAME] = T
        observed_outcome_data[Constants.OBSERVED_OUTCOME_VAR_NAME] = Y

        # Data not available for causal inference.
        oracle_outcome_data = pd.DataFrame()
        oracle_outcome_data[Constants.OUTCOME_NOISE_VAR_NAME] = noise_samples
        oracle_outcome_data[Constants.TREATMENT_ASSIGNMENT_LOGIT_VAR_NAME] = self.logit_values
        oracle_outcome_data[Constants.PROPENSITY_SCORE_VAR_NAME] = self.propensity_scores
        oracle_outcome_data[Constants.POTENTIAL_OUTCOME_WITHOUT_TREATMENT_VAR_NAME] = Y0
        oracle_outcome_data[Constants.POTENTIAL_OUTCOME_WITH_TREATMENT_VAR_NAME] = Y1
        oracle_outcome_data[Constants.TREATMENT_EFFECT_VAR_NAME] = self.treatment_effects

        dataset = DataSet(
            observed_covariate_data=self.observed_covariate_data,
            oracle_covariate_data=self.oracle_covariate_data,
            observed_outcome_data=observed_outcome_data,
            oracle_outcome_data=oracle_outcome_data)

        return dataset
