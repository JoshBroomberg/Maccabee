from sympy.abc import x
import sympy as sp
import numpy as np
import pandas as pd
from itertools import combinations
from .constants import Constants
from .utilities import select_given_probability_distribution, evaluate_expression, initialize_expression_constants

class DataGeneratingProcessWrapper():
    def __init__(self, parameters, source_covariate_data):
        self.params = parameters
        self.source_covariate_data = source_covariate_data

        self.covariate_symbols = np.array(sp.symbols(list(self.source_covariate_data.columns)))

        # Potential confounders
        self.potential_confounder_symbols = None

        # Observed covariate data.
        self.observed_covariate_data = None

        # Sampled covariate transforms for the treat and outcome functions.
        self.outcome_covariate_transforms = None
        self.treatment_covariate_transforms = None

        # Treatment assignment function and subfunctions
        self.treatment_assignment_logit_function = None
        self.treatment_assignment_function = None

        # Outcome function and subfunctions
        self.treatment_effect_subfunction = None
        self.outcome_function = None

        # Generated data
        self.observed_data = None
        self.oracle_data = None

    def sample_dgp(self):
        self.sample_observed_covariate_data()
        self.sample_potential_confounders()
        self.sample_treatment_and_outcome_covariate_transforms()
        self.sample_treatment_assignment_function()
        self.sample_outcome_function()

    def sample_observed_covariate_data(self):
        # Sample observations to reduce observation count if desired.
        self.observed_covariate_data = self.source_covariate_data.sample(
            frac=self.params.OBSERVATION_PROBABILITY)

        n_observed = self.observed_covariate_data.shape[0]
        # Sample and add noise column
        noise_samples = self.params.sample_outcome_noise(size=n_observed)
        self.observed_covariate_data[Constants.OUTCOME_NOISE_VAR_NAME] = noise_samples

    def sample_potential_confounders(self):
        self.potential_confounder_symbols, _ = select_given_probability_distribution(
            full_list=self.covariate_symbols,
            selection_probabilities=self.params.POTENTIAL_CONFOUNDER_SELECTION_PROBABILITY)

    def sample_covariate_transforms(self, covariate_symbols, transform_probabilities):
        """
        Sample a set of transforms which will be applied to the base covariates.
        The set of transforms is governed by the specified probabilities for
        each possible transform type. Each transform is parameterized by
        a set of constants.

        These transforms are used in the finalized functional form of the
        treat/outcome functions. Each of these functions uses
        combines a set of random transforms of the covariates (produced by this
        function). The subfunction form probabilities and combination of the
        transforms is different for the treat and outcome function.
        """
        #TODO: Post-process the output of this function to group terms
        # based on the same covariates and produce new multiplicative
        # combinations of different covariates.

        covariate_transforms = []
        for transform_name, transform_spec in Constants.SUBFUNCTION_FORMS.items():
            transform_expression = transform_spec[Constants.EXPRESSION_KEY]
            transform_covariate_symbols = transform_spec[Constants.COVARIATE_SYMBOLS_KEY]

            # All possible combinations of covariates for the given transform.
            covariate_combinations = np.array(
                list(combinations(
                    covariate_symbols,
                    len(transform_covariate_symbols))))

            selected_covar_combinations, _ = select_given_probability_distribution(
                full_list=covariate_combinations,
                selection_probabilities=transform_probabilities[transform_name])


            covariate_transforms.extend([transform_expression.subs(
                                    zip(transform_covariate_symbols, covar_comb))
                                    for covar_comb in selected_covar_combinations
                                ])

        return covariate_transforms

    def sample_treatment_and_outcome_covariate_transforms(self):
        """
        Sample covariate transforms for the treatment and outcome function
        and then modify the sampled transforms to generate desired alignment.
        """
        outcome_covariate_transforms = self.sample_covariate_transforms(
                self.potential_confounder_symbols,
                self.params.OUTCOME_MECHANISM_COVARIATE_SELECTION_PROBABILITY)

        treatment_covariate_transforms = self.sample_covariate_transforms(
                self.potential_confounder_symbols,
                self.params.TREAT_MECHANISM_COVARIATE_SELECTION_PROBABILITY)

        # Unique set of all covariate transforms
        all_transforms = np.array(
            list(set(treatment_covariate_transforms + outcome_covariate_transforms)))

        # Select overlapping covariates transforms (effective confounder space)
        # based on the alignment parameter.
        aligned_transforms, _ = select_given_probability_distribution(
                all_transforms,
                selection_probabilities=self.params.ACTUAL_CONFOUNDER_ALIGNMENT)
        aligned_transforms = set(aligned_transforms)

        # Union the true confounders into the original covariate selections.
        self.outcome_covariate_transforms = list(
            aligned_transforms.union(outcome_covariate_transforms))
        self.treatment_covariate_transforms = list(
            aligned_transforms.union(treatment_covariate_transforms))

    def sample_treatment_assignment_function(self):
        """
        Sample a treatment assignment function by combining the sampled covariate
        transforms, initializing the constants, and normalizing the function
        outputs to conform to constraints and targets on the propensity scores.
        """
        # TODO: consider recursively applying the covariate transformation
        # to produce "deep" functions. Probably overkill.

        # TODO: enable overlap adjustment

        # Randomly initialize transform constants
        # TODO: rescale each covar to map to the range -1, 1
        self.treatment_covariate_transforms = initialize_expression_constants(
            self.params,
            self.treatment_covariate_transforms)

        # Build base treatment logit function. Additive combination of the true covariates.
        # TODO: add non-linear activation function
        base_treatment_logit_expression = np.sum(self.treatment_covariate_transforms)

        # Sample data to evaluate distribution.
        sampled_data = self.observed_covariate_data.sample(
            frac=Constants.NORMALIZATION_DATA_SAMPLE_FRACTION)

        # Adjust logit
        logit_values = evaluate_expression(
            base_treatment_logit_expression, sampled_data)

        max_logit = np.max(logit_values)
        min_logit = np.min(logit_values)
        mean_logit = np.mean(logit_values)

        # Rescale to meet min/max constraints and target propensity.
        # First, construct function to rescale between 0 and 1
        normalized_logit_expr = (x - min_logit)/(max_logit - min_logit)

        # Second, construct function to rescale to target interval
        constrained_logit_expr = self.params.TARGET_MIN_LOGIT + \
            (x*(self.params.TARGET_MAX_LOGIT - self.params.TARGET_MIN_LOGIT))

        rescaling_expr = constrained_logit_expr.subs(x, normalized_logit_expr)
        rescaled_logit_mean = rescaling_expr.evalf(subs={x: mean_logit})

        # Third, construct function to apply offset for targeted propensity.
        # This requires the rescaled mean found above.
        target_propensity_adjustment = self.params.TARGET_MEAN_LOGIT - rescaled_logit_mean
        targeted_logit_expr = rescaling_expr + target_propensity_adjustment

        # Apply max/min truncation to account for adjustment shift.
        max_min_capped_targeted_logit = sp.functions.Max(
                sp.functions.Min(targeted_logit_expr, self.params.TARGET_MAX_LOGIT),
                self.params.TARGET_MIN_LOGIT)

        # Finally, construct the full function expression.
        self.treatment_assignment_logit_function = \
            max_min_capped_targeted_logit.subs(x, base_treatment_logit_expression)
        exponentiated_logit = sp.functions.exp(self.treatment_assignment_logit_function)

        self.treatment_assignment_function = exponentiated_logit/(1 + exponentiated_logit)


    def sample_treatment_effect_subfunction(self):
        """ Create treatment effect subfunction """

        base_treatment_effect = self.params.sample_treatment_effect()[0]

        # Sample outcome subfunctions to interact with Treatment effect.
        selected_interaction_terms, _ = select_given_probability_distribution(
                full_list=self.outcome_covariate_transforms,
                selection_probabilities=self.params.TREATMENT_EFFECT_HETEROGENEITY)

        initialized_interaction_terms = initialize_expression_constants(
            self.params,
            selected_interaction_terms)

        # Build multiplier
        treatment_effect_multiplier_expr = np.sum(initialized_interaction_terms)

        # Normalize multiplier size but not location. This keeps the size
        # of the effect bounded but doesn't center the effect for different units
        # at 0.
        sampled_data = self.observed_covariate_data.sample(
            frac=Constants.NORMALIZATION_DATA_SAMPLE_FRACTION)

        treatment_effect_multiplier_values = evaluate_expression(
            treatment_effect_multiplier_expr, sampled_data)

        multiplier_std = np.std(treatment_effect_multiplier_values)
        multiplier_mean = np.mean(treatment_effect_multiplier_values)

        normalized_treatment_effect_multiplier_expr = \
            (treatment_effect_multiplier_expr - multiplier_mean)/multiplier_std

        self.treatment_effect_subfunction = base_treatment_effect * \
            (1+normalized_treatment_effect_multiplier_expr)


    def sample_outcome_function(self):
        """ Create outcome function"""

        # TODO: consider recursively applying the covariate transformation
        # to produce "deep" functions. Probability overkill.

        # Randomly initialize subfunction constants.
        # TODO: rescale each covar to map to the range -1, 1
        self.outcome_covariate_transforms = initialize_expression_constants(
            self.params,
            self.outcome_covariate_transforms)

        # Build base outcome function. Additive combination of the true covariates.

        # TODO: add non-linear activation function and ensure proper normalization.
        # use the OUTCOME_MECHANISM_EXPONENTIATION param.
        base_outcome_expression = np.sum(self.outcome_covariate_transforms)

        # Sample data to evaluate distribution.
        sampled_data = self.observed_covariate_data.sample(
            frac=Constants.NORMALIZATION_DATA_SAMPLE_FRACTION)

        # Normalized outcome values to have approximate mean=0 and std=1.
        # This prevents situations where large outcome values drown out
        # the treatment effect or the treatment effect dominates small average outcomes.
        outcome_values = evaluate_expression(base_outcome_expression, sampled_data)
        outcome_mean = np.mean(outcome_values)
        outcome_std = np.std(outcome_values)

        # This is only an approximate normalization. It will shift the mean to zero
        # but the exact effect on std will depend on the distribution.
        normalized_outcome_expression = \
            (base_outcome_expression - outcome_mean)/outcome_std

        # Create the treatment effect subfunction.
        self.sample_treatment_effect_subfunction()

        self.outcome_function = normalized_outcome_expression + \
            Constants.TREATMENT_ASSIGNMENT_SYMBOL*self.treatment_effect_subfunction + \
            Constants.OUTCOME_NOISE_SYMBOL

    def generate_data(self):
        """Perform data generation"""
        n_observations = self.observed_covariate_data.shape[0]

        logit_values = evaluate_expression(
            self.treatment_assignment_logit_function,
            self.observed_covariate_data)
        propensity_scores = evaluate_expression(
            self.treatment_assignment_function, self.observed_covariate_data)

        T = (np.random.uniform(size=n_observations) < propensity_scores).astype(int)

        Y0 = evaluate_expression(
                self.outcome_function.subs(Constants.TREATMENT_ASSIGNMENT_SYMBOL, 0),
                self.observed_covariate_data)
        Y1 = evaluate_expression(
                self.outcome_function.subs(Constants.TREATMENT_ASSIGNMENT_SYMBOL, 1),
                self.observed_covariate_data)

        Y = (T*Y1) + ((1-T)*Y0)
        treatment_effect = Y1 - Y0

        # Data available for causal inference
        self.observed_data = self.observed_covariate_data.copy()
        self.observed_data[Constants.TREATMENT_ASSIGNMENT_VAR_NAME] = T
        self.observed_data[Constants.OBSERVED_OUTCOME_VAR_NAME] = Y

        # Data not available for causal inference.
        self.oracle_data = self.generate_transformed_covariate_data()
        self.oracle_data[Constants.TREATMENT_ASSIGNMENT_LOGIT_VAR_NAME] = logit_values
        self.oracle_data[Constants.PROPENSITY_SCORE_VAR_NAME] = propensity_scores
        self.oracle_data[Constants.POTENTIAL_OUTCOME_WITHOUT_TREATMENT_VAR_NAME] = Y0
        self.oracle_data[Constants.POTENTIAL_OUTCOME_WITH_TREATMENT_VAR_NAME] = Y1
        self.oracle_data[Constants.TREATMENT_EFFECT_VAR_NAME] = treatment_effect

        return self.observed_data, self.oracle_data

    def generate_transformed_covariate_data(self):
        all_transforms = list(set(self.outcome_covariate_transforms).union(self.treatment_covariate_transforms))

        data = {}
        for index, transform in enumerate(all_transforms):
            data[f"X'_{index}={transform}"] = evaluate_expression(
                transform, self.observed_covariate_data)

        return pd.DataFrame(data)
