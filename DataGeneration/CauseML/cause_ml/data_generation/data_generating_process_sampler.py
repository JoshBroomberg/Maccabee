from sympy.abc import x
import sympy as sp
import numpy as np
import pandas as pd
from itertools import combinations
from ..constants import Constants
from ..utilities import select_given_probability_distribution, evaluate_expression, initialize_expression_constants
from .data_generating_process import SampledDataGeneratingProcess

class DataGeneratingProcessSampler():
    def __init__(self, parameters, data_source,
        dgp_class=SampledDataGeneratingProcess, dgp_kwargs={}):

        self.dgp_class = dgp_class
        self.params = parameters
        self.data_source = data_source
        self.dgp_kwargs = dgp_kwargs

    def sample_dgp(self):
        source_covariate_data = self.data_source.get_data()
        covariate_symbols = np.array(sp.symbols(list(source_covariate_data.columns)))

        # Sample the source data to generate the observed covariate data.
        observed_covariate_data = self.sample_observed_covariate_data(
            source_covariate_data)

        # Select the observed variables which may appear in the assignment or
        # outcome functions. These are potential confounders.
        potential_confounder_symbols = self.sample_potential_confounders(
            covariate_symbols)

        # Sample the covariate transforms which make up the assignment and
        # outcome functions.
        outcome_covariate_transforms, treatment_covariate_transforms = \
            self.sample_treatment_and_outcome_covariate_transforms(
                potential_confounder_symbols)

        treatment_assignment_logit_func, treatment_assignment_function = \
            self.sample_treatment_assignment_function(
                treatment_covariate_transforms, observed_covariate_data)

        outcome_function, base_outcome_subfunc, treat_effect_subfunc = \
            self.sample_outcome_function(
                outcome_covariate_transforms, observed_covariate_data)

        dgp = self.dgp_class(
            params=self.params,
            observed_covariate_data=observed_covariate_data,
            outcome_covariate_transforms=outcome_covariate_transforms,
            treatment_covariate_transforms=treatment_covariate_transforms,
            treatment_assignment_logit_func=treatment_assignment_logit_func,
            treatment_assignment_function=treatment_assignment_function,
            treatment_effect_subfunction=treat_effect_subfunc,
            base_outcome_subfunction=base_outcome_subfunc,
            outcome_function=outcome_function,
            **self.dgp_kwargs)

        return dgp

    def sample_observed_covariate_data(self, source_covariate_data):
        # Sample observations.

        # For now, we allow simple uniform sampling. In future, we may
        # support more complex sampling procedures to allow for simulation
        # of observation censorship etc.

        # NOTE: in the ideal world, this would occur on data generation
        # (as part of the DGP run) rather than here. But normalization of the
        # DGP to target specific distribution properties requires a static
        # dataset.
        observed_covariate_data = source_covariate_data.sample(
            frac=self.params.OBSERVATION_PROBABILITY)

        return observed_covariate_data

    def sample_potential_confounders(self, covariate_symbols):
        potential_confounder_symbols, _ = select_given_probability_distribution(
            full_list=covariate_symbols,
            selection_probabilities=self.params.POTENTIAL_CONFOUNDER_SELECTION_PROBABILITY)

        if len(potential_confounder_symbols) == 0:
            potential_confounder_symbols = [np.random.choice(covariate_symbols)]

        return potential_confounder_symbols

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
        # and no transforms selected above.
        # TODO: this is an ugly solution to a complex problem. Improve this.
        if len(selected_covariate_transforms) == 0 and not empty_allowed:

            # TODO: eval and delete this if no longer desired.
            # transform_spec = Constants.SUBFUNCTION_FORMS[Constants.LINEAR]
            # transform_expression = transform_spec[Constants.EXPRESSION_KEY]
            # transform_covariate_symbols = transform_spec[Constants.COVARIATE_SYMBOLS_KEY]
            # required_covars = covariate_symbols[:len(transform_covariate_symbols)]
            #
            # selected_covariate_transforms = [
            #     transform_expression.subs(zip(
            #     transform_covariate_symbols, required_covars))
            # ]
            selected_covariate_transforms.append(list(Constants.SUBFUNCTION_CONSTANT_SYMBOLS)[0])

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

    def sample_treatment_and_outcome_covariate_transforms(self, potential_confounder_symbols):
        """
        Sample covariate transforms for the treatment and outcome function
        and then modify the sampled transforms to generate desired alignment.
        """
        outcome_covariate_transforms = self.sample_covariate_transforms(
                potential_confounder_symbols,
                self.params.OUTCOME_MECHANISM_COVARIATE_SELECTION_PROBABILITY)

        treatment_covariate_transforms = self.sample_covariate_transforms(
                potential_confounder_symbols,
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
        outcome_covariate_transforms = np.hstack(
            [aligned_transforms, outcome_only_transforms])
        treatment_covariate_transforms = np.hstack(
            [aligned_transforms, treat_only_transforms])

        return outcome_covariate_transforms, treatment_covariate_transforms


    def sample_treatment_assignment_function(self,
        treatment_covariate_transforms, observed_covariate_data):
        """
        Sample a treatment assignment function by combining the sampled covariate
        transforms, initializing the constants, and normalizing the function
        outputs to conform to constraints and targets on the propensity scores.
        """
        # TODO: consider recursively applying the covariate transformation
        # to produce "deep" functions. Probably overkill.
        # TODO: add non-linear activation function
        # TODO: enable overlap adjustment

        # Build base treatment logit function. Additive combination of the true covariates.
        base_treatment_logit_expression = np.sum(
            treatment_covariate_transforms)

        # Sample data to evaluate distribution.
        sampled_data = observed_covariate_data.sample(
            frac=Constants.NORMALIZATION_DATA_SAMPLE_FRACTION)

        # Adjust logit
        logit_values = evaluate_expression(
            base_treatment_logit_expression, sampled_data)

        max_logit = np.max(logit_values)
        min_logit = np.min(logit_values)
        mean_logit = np.mean(logit_values)

        # Rescale to meet min/max constraints and target propensity.
        # First, construct function to rescale between 0 and 1
        diff = (max_logit - min_logit)
        if np.min(diff) > 0:
            normalized_logit_expr = (x - min_logit)/diff
        else:
            prop_score_logit = self.params.TARGET_MEAN_LOGIT
            treatment_assignment_logit_function = prop_score_logit
            treatment_assignment_function = \
                np.exp(prop_score_logit)/(1+np.exp(prop_score_logit))

            return (treatment_assignment_logit_function,
                treatment_assignment_function)

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
        treatment_assignment_logit_function = \
            max_min_capped_targeted_logit.subs(x, base_treatment_logit_expression)

        exponentiated_logit = sp.functions.exp(treatment_assignment_logit_function)
        treatment_assignment_function = exponentiated_logit/(1 + exponentiated_logit)

        return (treatment_assignment_logit_function,
            treatment_assignment_function)

    def sample_treatment_effect_subfunction(self,
        outcome_covariate_transforms, observed_covariate_data):
        """ Create treatment effect subfunction """

        base_treatment_effect = self.params.sample_treatment_effect()[0]

        # Sample outcome subfunctions to interact with Treatment effect.
        selected_interaction_terms, _ = select_given_probability_distribution(
                full_list=outcome_covariate_transforms,
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
            sampled_data = observed_covariate_data.sample(
                frac=Constants.NORMALIZATION_DATA_SAMPLE_FRACTION)

            treatment_effect_multiplier_values = evaluate_expression(
                treatment_effect_multiplier_expr, sampled_data)

            multiplier_std = np.std(treatment_effect_multiplier_values)
            multiplier_mean = np.mean(treatment_effect_multiplier_values)

            normalized_treatment_effect_multiplier_expr = \
                (treatment_effect_multiplier_expr - multiplier_mean)/multiplier_std

            treatment_effect_subfunction = base_treatment_effect * \
                (1+normalized_treatment_effect_multiplier_expr)

            return treatment_effect_subfunction

        # No interaction terms. Return base treatment effect as the
        # treatment effect subfunction.
        else:
            return base_treatment_effect


    def sample_outcome_function(self,
        outcome_covariate_transforms, observed_covariate_data):
        """ Create outcome function"""

        # TODO: consider recursively applying the covariate transformation
        # to produce "deep" functions. Probability overkill.

        # Build base outcome function. Additive combination of the true covariates.

        # TODO: add non-linear activation function and ensure proper normalization.
        # using or changing the OUTCOME_MECHANISM_EXPONENTIATION param.
        base_outcome_expression = np.sum(outcome_covariate_transforms)

        # Sample data to evaluate distribution.
        sampled_data = observed_covariate_data.sample(
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
        treat_effect_subfunction = self.sample_treatment_effect_subfunction(
            outcome_covariate_transforms, observed_covariate_data)

        base_outcome_subfunction = normalized_outcome_expression
        outcome_function = base_outcome_subfunction + \
            Constants.OUTCOME_NOISE_SYMBOL + \
            (Constants.TREATMENT_ASSIGNMENT_SYMBOL*treat_effect_subfunction)

        return outcome_function, base_outcome_subfunction, treat_effect_subfunction
