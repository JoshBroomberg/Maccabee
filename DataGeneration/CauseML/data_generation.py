from sympy.abc import x
import sympy as sp
import numpy as np
import pandas as pd
from itertools import combinations
from .constants import Constants
from .utilities import select_given_probability_distribution, evaluate_expression, initialize_expression_constants

class DataGeneratingProcessWrapper():
    def __init__(self, parameters, data_source):


        self.params = parameters
        self.data_source = data_source
        self.source_covariate_data = data_source.get_data()

        self.covariate_symbols = np.array(sp.symbols(list(self.source_covariate_data.columns)))

        # Potential confounders
        self.potential_confounder_symbols = None

        # Observed and oracle/transformed covariate data.
        self.observed_covariate_data = None
        self.oracle_covariate_data = None

        # Sampled covariate transforms for the treat and outcome functions.
        self.outcome_covariate_transforms = None
        self.treatment_covariate_transforms = None

        # Treatment assignment function and subfunctions
        self.treatment_assignment_logit_function = None
        self.treatment_assignment_function = None

        # Outcome function and subfunctions
        self.treatment_effect_subfunction = None
        self.treatment_free_outcome_subfunction = None
        self.outcome_function = None

        # Generated data
        self.observed_outcome_data = None
        self.oracle_outcome_data = None

        # Operational constants.
        self.data_generated = False

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

        if len(self.potential_confounder_symbols) == 0:
            self.potential_confounder_symbols = [
                np.random.choice(self.covariate_symbols)]

    def sample_covariate_transforms(self, covariate_symbols,
        transform_probabilities, empty_allowed=False):
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
        #TODO NB: Post-process the output of this function to group terms
        # based on the same covariates and produce new multiplicative
        # combinations of different covariates.

        selected_covariate_transforms = []
        for transform_name, transform_spec in Constants.SUBFUNCTION_FORMS.items():
            transform_expression = transform_spec[Constants.EXPRESSION_KEY]
            transform_covariate_symbols = transform_spec[Constants.COVARIATE_SYMBOLS_KEY]
            transform_discrete_allowed = transform_spec[Constants.DISCRETE_ALLOWED_KEY]

            usable_covariate_symbols = covariate_symbols
            if not transform_discrete_allowed:
                usable_covariate_symbols = list(filter(
                    lambda sym: str(sym) not in self.data_source.binary_column_names,
                    covariate_symbols))

            # All possible combinations of covariates for the given transform.
            covariate_combinations = np.array(
                list(combinations(
                    usable_covariate_symbols,
                    len(transform_covariate_symbols))))

            selected_covar_combinations, _ = select_given_probability_distribution(
                full_list=covariate_combinations,
                selection_probabilities=transform_probabilities[transform_name])

            selected_covariate_transforms.extend([transform_expression.subs(
                                    zip(transform_covariate_symbols, covar_comb))
                                    for covar_comb in selected_covar_combinations
                                ])

        # Add at least one transform if empty_allowed is False
        # and no transforms slected above.
        # TODO: this is an ugly solution to a complex problem. Improve this.
        if len(selected_covariate_transforms) == 0 and not empty_allowed:

            transform_spec = Constants.SUBFUNCTION_FORMS[Constants.LINEAR]
            transform_expression = transform_spec[Constants.EXPRESSION_KEY]
            transform_covariate_symbols = transform_spec[Constants.COVARIATE_SYMBOLS_KEY]
            required_covars = covariate_symbols[:len(transform_covariate_symbols)]

            selected_covariate_transforms = [
                transform_expression.subs(zip(
                transform_covariate_symbols, required_covars))
            ]

        # The number of combinations grows factorially with the number of
        # covariates. Cap the number of covariate transforms at a multiple
        # of the number of base covariates.
        max_transform_count = \
            Constants.MAX_RATIO_TRANSFORMED_TO_ORIGINAL_TERMS*len(covariate_symbols)

        if len(selected_covariate_transforms) > max_transform_count:
            # Randomly sample selected transforms with expected number selected
            # equal to the max.
            selection_p = max_transform_count/len(selected_covariate_transforms)

            selected_covariate_transforms, _ = select_given_probability_distribution(
                full_list=selected_covariate_transforms,
                selection_probabilities=selection_p)

            selected_covariate_transforms = list(selected_covariate_transforms)

        return selected_covariate_transforms

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
        all_transforms = list(set(
            treatment_covariate_transforms + outcome_covariate_transforms))

        # Select overlapping covariates transforms (effective confounder space)
        # based on the alignment parameter.
        aligned_transforms, _ = select_given_probability_distribution(
                all_transforms,
                selection_probabilities=self.params.ACTUAL_CONFOUNDER_ALIGNMENT)

        treat_only_transforms = list(set(treatment_covariate_transforms).difference(aligned_transforms))
        outcome_only_transforms = list(set(outcome_covariate_transforms).difference(aligned_transforms))

        # Initialize the constants in all the transforms.
        aligned_transforms = initialize_expression_constants(
            self.params,
            aligned_transforms)

        treat_only_transforms = initialize_expression_constants(
            self.params,
            treat_only_transforms)

        outcome_only_transforms = initialize_expression_constants(
            self.params,
            outcome_only_transforms)

        # Union the true confounders into the original covariate selections.
        self.outcome_covariate_transforms = np.hstack([aligned_transforms, outcome_only_transforms])
        self.treatment_covariate_transforms = np.hstack([aligned_transforms, treat_only_transforms])


    def sample_treatment_assignment_function(self):
        """
        Sample a treatment assignment function by combining the sampled covariate
        transforms, initializing the constants, and normalizing the function
        outputs to conform to constraints and targets on the propensity scores.
        """
        # TODO: consider recursively applying the covariate transformation
        # to produce "deep" functions. Probably overkill.

        # TODO: enable overlap adjustment

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

        # Process interaction terms into treatment subfunction.
        if treatment_effect_multiplier_expr != 0:

            # Normalize multiplier size.
            # TODO: vaidate the approach to normalization used here
            # vs other function construction.
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

        # No interaction terms
        else:
            self.treatment_effect_subfunction = base_treatment_effect


    def sample_outcome_function(self):
        """ Create outcome function"""

        # TODO: consider recursively applying the covariate transformation
        # to produce "deep" functions. Probability overkill.

        # Build base outcome function. Additive combination of the true covariates.

        # TODO: add non-linear activation function and ensure proper normalization.
        # using or changing the OUTCOME_MECHANISM_EXPONENTIATION param.
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

        self.treatment_free_outcome_subfunction = normalized_outcome_expression + \
            Constants.OUTCOME_NOISE_SYMBOL

        self.outcome_function = self.treatment_free_outcome_subfunction + \
            (Constants.TREATMENT_ASSIGNMENT_SYMBOL*self.treatment_effect_subfunction)


    def generate_transformed_covariate_data(self):
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

        self.data_generated = True

        n_observations = self.observed_covariate_data.shape[0]

        # Generate treatment assignment data.
        logit_values = evaluate_expression(
            self.treatment_assignment_logit_function,
            self.observed_covariate_data)

        propensity_scores = np.exp(logit_values)/(1+np.exp(logit_values))

        T = (np.random.uniform(size=n_observations) < propensity_scores).astype(int)

        # Balance adjustment
        control_p_scores = propensity_scores.where(T == 0)
        treat_p_scores = propensity_scores.where(T == 1)

        num_controls = control_p_scores.count()
        n_to_switch = int(num_controls*self.params.FORCED_IMBALANCE_ADJUSTMENT)

        control_switch_targets = control_p_scores.nlargest(n_to_switch).index.values
        treat_switch_targets = treat_p_scores.nsmallest(n_to_switch).index.values

        T[control_switch_targets] = 1
        T[treat_switch_targets] = 0

        # Generate base outcomes, treatment effects and potential outcomes.
        base_outcomes = evaluate_expression(
            self.treatment_free_outcome_subfunction,
            self.observed_covariate_data)

        treatment_effects = evaluate_expression(
            self.treatment_effect_subfunction,
            self.observed_covariate_data)

        Y0 = base_outcomes
        Y1 = base_outcomes + treatment_effects

        # Observed outcome
        Y = (T*Y1) + ((1-T)*Y0)

        # Build data frames.

        # Data available for causal inference
        self.observed_outcome_data = pd.DataFrame()
        self.observed_outcome_data[Constants.TREATMENT_ASSIGNMENT_VAR_NAME] = T
        self.observed_outcome_data[Constants.OBSERVED_OUTCOME_VAR_NAME] = Y

        # Data not available for causal inference.
        self.oracle_covariate_data = self.generate_transformed_covariate_data()
        self.oracle_outcome_data = pd.DataFrame()
        self.oracle_outcome_data[Constants.TREATMENT_ASSIGNMENT_LOGIT_VAR_NAME] = logit_values
        self.oracle_outcome_data[Constants.PROPENSITY_SCORE_VAR_NAME] = propensity_scores
        self.oracle_outcome_data[Constants.POTENTIAL_OUTCOME_WITHOUT_TREATMENT_VAR_NAME] = Y0
        self.oracle_outcome_data[Constants.POTENTIAL_OUTCOME_WITH_TREATMENT_VAR_NAME] = Y1
        self.oracle_outcome_data[Constants.TREATMENT_EFFECT_VAR_NAME] = treatment_effects

        return (self.observed_covariate_data, self.observed_outcome_data,
            self.oracle_covariate_data, self.oracle_outcome_data)

    def get_observed_data(self):
        '''
        Assemble and return the observable data.
        '''
        if not self.data_generated:
            raise Exception("You must run generate_data first.")

        return self.observed_outcome_data.join(self.observed_covariate_data)

    def get_oracle_data(self):
        '''
        Assemble and return the non-observable/oracle data.
        '''
        if not self.data_generated:
            raise Exception("You must run generate_data first.")

        return self.oracle_outcome_data.join(self.oracle_covariate_data)
