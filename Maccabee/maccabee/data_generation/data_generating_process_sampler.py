"""This module contains the :class:`~maccabee.data_generation.data_generating_process.DataGeneratingProcess` class which is used to sample :term:`DGPs <DGP>` given sampling parameters which determine where in the :term:`distributional problem space` the :class:`~maccabee.data_generation.data_generating_process.DataGeneratingProcess` targets for sampling.
"""

from sympy.abc import x
import sympy as sp
import numpy as np
import pandas as pd
from itertools import combinations
from ..constants import Constants
from .utils import select_objects_given_probability, evaluate_expression, initialize_expression_constants
from .data_generating_process import SampledDataGeneratingProcess

SamplingConstants = Constants.DGPSampling
ComponentConstants = Constants.DGPVariables

class DataGeneratingProcessSampler():
    """DataGeneratingProcessSampler(...)

    The :class:`~maccabee.data_generation.data_generating_process_sampler.DataGeneratingProcessSampler` class takes a set of sampling parameters and a data source (a :class:`~maccabee.data_sources.data_sources.DataSource` instance) that provides the base covariates. It then samples the treatment assignment and outcome functions which, in combination with the observed covariate data from the :class:`~maccabee.data_sources.data_sources.DataSource` completely specify the DGP. The two functions are sampled based on the provided sampling parameters to target a location in the :term:`distributional problem space`.

    The :class:`~maccabee.data_generation.data_generating_process_sampler.DataGeneratingProcessSampler` class is designed to work for most users and use cases. However, it is also designed to be customized through inheritance in order to cater to the needs of advanced users. This is achieved by breaking down the DGP sampling process into a series of steps corresponding to class methods which can be overridden individually. Users interested in this should see the linked source code for extensive in-line guiding comments.

    Args:
        parameters (:class:`~maccabee.parameters.parameter_store.ParameterStore`): A :class:`~maccabee.parameters.parameter_store.ParameterStore` instance which contains the parameters that control the sampling process. See the :mod:`maccabee.parameters` module docs for more detail on how to build a :class:`~maccabee.parameters.parameter_store.ParameterStore` instance.
        data_source (:class:`~maccabee.data_sources.data_sources.DataSource`): A :class:`~maccabee.data_sources.data_sources.DataSource` instance which provides observed covariates. See the :mod:`~maccabee.data_sources` module docs for more detail.
        dgp_class (:class:`~maccabee.data_generation.data_generating_process.SampledDataGeneratingProcess`): A class which inherits from :class:`~maccabee.data_generation.data_generating_process.SampledDataGeneratingProcess`. Defaults to :class:`~maccabee.data_generation.data_generating_process.SampledDataGeneratingProcess`. This is only necessary if you would like to customize some aspect of the sampled DGP which is not controllable through the sampling parameters provided in `parameters`.
        dgp_kwargs (dict): A dictionary of keyword arguments which is passed to the sampled DGP at instantiation. Defaults to {}.
    """
    def __init__(self, parameters, data_source,
        dgp_class=SampledDataGeneratingProcess, dgp_kwargs={}):

        self.dgp_class = dgp_class
        self.params = parameters
        self.data_source = data_source
        self.dgp_kwargs = dgp_kwargs

    def sample_dgp(self):
        """This is the primary external method of this class. It is used to sample a new DGP. Internally, a number of steps are executed:

        * A set of observable covariates is sampled from the :class:`~maccabee.data_sources.data_sources.DataSource` supplied at instantiation.
        * A subset of the observable covariates is sampled based on the observation likelihood parameterization.
        * Potential confounders are sampled. These are the covariates which could enter either or both of the treatment and outcome function. Covariates not sampled here are nuisance/non-predictive covariates.
        * The treatment and outcome subfunctions, the transformations which make up the two main functions, are sampled according to the desired parameters for function form and alignment (the degree of confounding).
        * The treatment and outcome functions are assembled and normalized to meet parameters for target propensity score and treatment effect heterogeneity.
        * The DGP instance is assembled using the class and kwargs supplied at instantiation time and the components produced by the steps above.

        Returns:
            :class:`~maccabee.data_generation.data_generating_process.SampledDataGeneratingProcess`: A :class:`~maccabee.data_generation.data_generating_process.SampledDataGeneratingProcess` instance representing a sampled DGP.
        """

        # NOTE: for most customizing cases, this function can and should
        # remain unchanged as it only defines an execution order and passes
        # fairly generic parameters. Rather override the various subroutines
        # below.

        # TODO: Remove
        # np.random.seed()

        source_covariate_data = self.data_source.get_covar_df()
        covariate_symbols = np.array(sp.symbols(self.data_source.get_covar_names()))

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

        # Build the treatment assignment function.
        treatment_assignment_logit_func, treatment_assignment_function = \
            self.sample_treatment_assignment_function(
                treatment_covariate_transforms, observed_covariate_data)

        # Build the outcome and treatment effect functions.
        outcome_function, untreated_outcome_subfunc, treat_effect_subfunc = \
            self.sample_outcome_function(
                outcome_covariate_transforms, observed_covariate_data)

        # Construct DGP
        dgp = self.dgp_class(
            params=self.params,
            observed_covariate_data=observed_covariate_data,
            outcome_covariate_transforms=outcome_covariate_transforms,
            treatment_covariate_transforms=treatment_covariate_transforms,
            treatment_assignment_logit_func=treatment_assignment_logit_func,
            treatment_assignment_function=treatment_assignment_function,
            treatment_effect_subfunction=treat_effect_subfunc,
            untreated_outcome_subfunction=untreated_outcome_subfunc,
            outcome_function=outcome_function,
            data_source=self.data_source,
            **self.dgp_kwargs)

        return dgp

    def sample_observed_covariate_data(self, source_covariate_data):
        # 1. Sample a subset of the observable covariates.
        # For now, we allow simple uniform sampling. In future, we may
        # support more complex sampling procedures to allow for simulation
        # of observation censorship etc.

        observed_covariate_data = source_covariate_data.sample(
            frac=self.params.OBSERVATION_PROBABILITY)

        return observed_covariate_data

    def sample_potential_confounders(self, covariate_symbols):
        # 2. Sample potential confounders. These are the covariates which could
        # enter the treatment and/or outcome functions. All covariates not
        # selected here are non-predictive/nuisance covariates which can
        # make the causal model process harder.
        potential_confounder_symbols = select_objects_given_probability(
            objects_to_sample=covariate_symbols,
            selection_probability=self.params.POTENTIAL_CONFOUNDER_SELECTION_PROBABILITY)

        if len(potential_confounder_symbols) == 0:
            potential_confounder_symbols = [np.random.choice(covariate_symbols)]

        return potential_confounder_symbols

    def sample_covariate_transforms(self, covariate_symbols,
        transform_probabilities, empty_allowed=False):
        # 3A. Sample a set of transforms which will be applied to the covariates
        # supplied in covariate_symbols based on the transform probabilities in
        # transform_probabilities

        # The set of transforms is governed by the specific probabilities for
        # each possible transform type. Each transform is parameterized by
        # a set of constants which are then initialized randomly.

        # These transforms are used in the finalized functional form of the
        # treat/outcome functions. Each of these functions
        # combines a set of random transforms of the covariates (produced by this
        # function). The subfunction form probabilities and the combination form
        # is different for the treat and outcome function.

        # TODO: Post-process the output of this function to group terms
        # based on the same covariates and produce new multiplicative
        # combinations of different covariates as in Dorie et al (2019)

        selected_covariate_transforms = []

        # Loop over the transformation forms in SUBFUNCTION_FORMS.
        for transform_name, transform_spec in SamplingConstants.SUBFUNCTION_FORMS.items():

            # Extract the subfunction form information.
            transform_expression = transform_spec[SamplingConstants.EXPRESSION_KEY]
            transform_covariate_symbols = transform_spec[SamplingConstants.COVARIATE_SYMBOLS_KEY]
            transform_discrete_allowed = transform_spec[SamplingConstants.DISCRETE_ALLOWED_KEY]

            # Filter out covariates which are not allowed in this subfunction.
            usable_covariate_symbols = covariate_symbols
            if not transform_discrete_allowed:
                usable_covariate_symbols = list(filter(
                    lambda sym: str(sym) not in self.data_source.get_discrete_covar_names(),
                    covariate_symbols))

            # TODO: this should be memoized.
            # Generate possible combinations of covariates for the given transform.
            # This is the set of all possible instantiations of this subfunction.
            covariate_combinations = np.array(
                list(combinations(
                    usable_covariate_symbols,
                    len(transform_covariate_symbols))))

            # Sample from the set of all possible combinations.
            selected_covar_combinations = select_objects_given_probability(
                objects_to_sample=covariate_combinations,
                selection_probability=transform_probabilities[transform_name])

            # Instantiate the subfunction with the sampled covariates.
            selected_covariate_transforms.extend([transform_expression.subs(
                                    zip(transform_covariate_symbols, covar_comb))
                                    for covar_comb in selected_covar_combinations
                                ])

        # Add at least one transform if empty_allowed is False
        # and no transforms selected above.
        # TODO: this is an ugly solution to a complex problem. Improve this.
        if len(selected_covariate_transforms) == 0 and not empty_allowed:
            selected_covariate_transforms.append(
                list(SamplingConstants.SUBFUNCTION_CONSTANT_SYMBOLS)[0])

        # TODO: refactor this into a max term count
        # which is then applied after the two sampled functions
        # are fully contructed.

        # The number of combinations grows factorially with the number of
        # covariates. Cap the number of covariate transforms at a multiple
        # of the number of base covariates.
        max_transform_count = \
            SamplingConstants.MAX_MULTIPLE_TRANSFORMED_TO_ORIGINAL_TERMS*len(
                covariate_symbols)

        if len(selected_covariate_transforms) > max_transform_count:
            print("Running term limiter")
            # Randomly sample selected transforms with expected number selected
            # equal to the max.
            selection_p = max_transform_count/len(selected_covariate_transforms)

            selected_covariate_transforms = select_objects_given_probability(
                objects_to_sample=selected_covariate_transforms,
                selection_probability=selection_p)

            selected_covariate_transforms = list(selected_covariate_transforms)

        return selected_covariate_transforms

    def sample_treatment_and_outcome_covariate_transforms(self, potential_confounder_symbols):
        # 3B. Sample covariate transforms for the treatment and outcome function
        # and then modify the sampled transforms to generate desired alignment.
        # IE, adjust so that there is the desired amount of overlap in transformed
        # covariates which is the actual level of confounding.

        outcome_covariate_transforms = self.sample_covariate_transforms(
                potential_confounder_symbols,
                self.params.OUTCOME_MECHANISM_COVARIATE_SELECTION_PROBABILITY)

        treatment_covariate_transforms = self.sample_covariate_transforms(
                potential_confounder_symbols,
                self.params.TREAT_MECHANISM_COVARIATE_SELECTION_PROBABILITY)

        set_outcome_covariate_transforms = set(outcome_covariate_transforms)
        set_treatment_covariate_transforms = set(treatment_covariate_transforms)

        # Unique set of all covariate transforms
        all_transforms = set_outcome_covariate_transforms.union(
            set_treatment_covariate_transforms)

        already_aligned_transforms = set_outcome_covariate_transforms.intersection(
            set_treatment_covariate_transforms)

        # TODO: comment this code.

        current_alignment_proportion = len(already_aligned_transforms)/len(all_transforms)
        alignment_diff = current_alignment_proportion - self.params.ACTUAL_CONFOUNDER_ALIGNMENT
        if alignment_diff > 0:
            print(f"Reducing alignment from {round(current_alignment_proportion, 3)} to {self.params.ACTUAL_CONFOUNDER_ALIGNMENT}")

            expected_num_to_unalign = alignment_diff*len(all_transforms)
            unalign_probability = expected_num_to_unalign/len(already_aligned_transforms)
            transforms_to_unalign = select_objects_given_probability(
                    list(already_aligned_transforms),
                    selection_probability=unalign_probability)

            treatment_relative_size = len(set_treatment_covariate_transforms)/len(all_transforms)
            for transform in transforms_to_unalign:
                already_aligned_transforms.remove(transform)
                if np.random.random() < treatment_relative_size:
                    set_outcome_covariate_transforms.remove(transform)
                else:
                    set_treatment_covariate_transforms.remove(transform)

            aligned_transforms = list(already_aligned_transforms)
        else:
            print("Increasing alignment")
            # Select overlapping covariates transforms (effective confounder space)
            # based on the alignment parameter.
            aligned_transforms = select_objects_given_probability(
                    list(all_transforms - already_aligned_transforms),
                    selection_probability=abs(alignment_diff))

        # Extract treat and outcome exclusive transforms.
        treat_only_transforms = list(set_treatment_covariate_transforms.difference(
            aligned_transforms))
        outcome_only_transforms = list(set_outcome_covariate_transforms.difference(
            aligned_transforms))

        # Initialize the constants in all the transforms.
        aligned_transforms = initialize_expression_constants(
            self.params.sample_subfunction_constants,
            aligned_transforms)

        treat_only_transforms = initialize_expression_constants(
            self.params.sample_subfunction_constants,
            treat_only_transforms)

        outcome_only_transforms = initialize_expression_constants(
            self.params.sample_subfunction_constants,
            outcome_only_transforms)

        # Union the true confounders into the original covariate selections.
        outcome_covariate_transforms = np.hstack(
            [aligned_transforms, outcome_only_transforms])
        treatment_covariate_transforms = np.hstack(
            [aligned_transforms, treat_only_transforms])

        return outcome_covariate_transforms, treatment_covariate_transforms


    def sample_treatment_assignment_function(self,
        treatment_covariate_transforms, observed_covariate_data):
        # 4. Sample a treatment assignment function by combining the sampled covariate
        # transforms and normalizing the function outputs to conform to
        # constraints and targets on the propensity scores.

        # TODO: consider recursively applying the covariate transformation
        # to produce "deep" functions. Probably overkill.
        # TODO: add non-linear activation function
        # TODO: enable overlap adjustment

        # Build base treatment logit function. Additive combination of the true covariates.
        base_treatment_logit_expression = np.sum(
            treatment_covariate_transforms)

        # Normalize if config specifies.
        if SamplingConstants.NORMALIZE_SAMPLED_TREATMENT_FUNCTION:

            # Sample data to evaluate distribution.
            sampled_data = observed_covariate_data.sample(
                frac=SamplingConstants.NORMALIZATION_DATA_SAMPLE_FRACTION)

            # Adjust logit
            logit_values = evaluate_expression(
                base_treatment_logit_expression, sampled_data)

            max_logit = np.max(logit_values)
            min_logit = np.min(logit_values)
            mean_logit = np.mean(logit_values)
            std_logit = np.std(logit_values)

            # Check if logit function is effectively constant. If constant
            # then return the target logit. If not normalize to meet target.
            if np.isclose(max_logit, min_logit):
                # print("Using DGP with equal propensity for all units.")
                treatment_assignment_logit_function = self.params.TARGET_MEAN_LOGIT
            else:
                # This is only an approximate normalization. It will shift the mean to zero
                # but the exact effect on std will depend on the distribution.
                normalized_treatment_logit_expression = \
                    (base_treatment_logit_expression - mean_logit)/std_logit

                targeted_treatment_logit_expression = \
                    normalized_treatment_logit_expression + self.params.TARGET_MEAN_LOGIT

                treatment_assignment_logit_function = targeted_treatment_logit_expression

                # TODO: remove
                # DEPRECATED NORMALIZATION SCHEME.
                # # Rescale to meet min/max constraints and target propensity.
                # # First, construct function to rescale between 0 and 1
                # normalized_logit_expr = (x - min_logit)/range
                #
                # # Second, construct function to rescale to target interval
                # constrained_logit_expr = self.params.TARGET_MIN_LOGIT + \
                #     (x*(self.params.TARGET_MAX_LOGIT - self.params.TARGET_MIN_LOGIT))
                #
                # rescaling_expr = constrained_logit_expr.subs(x, normalized_logit_expr)
                # rescaled_logit_mean = rescaling_expr.evalf(
                #     subs={x: mean_logit})
                #
                # # Third, construct function to apply offset for targeted propensity.
                # # This requires the rescaled mean found above.
                # target_propensity_adjustment = self.params.TARGET_MEAN_LOGIT - rescaled_logit_mean
                # targeted_logit_expr = rescaling_expr + target_propensity_adjustment
                #
                # # Apply max/min truncation to account for adjustment shift.
                # max_min_capped_targeted_logit = sp.functions.Max(
                #         sp.functions.Min(targeted_logit_expr, self.params.TARGET_MAX_LOGIT),
                #         self.params.TARGET_MIN_LOGIT)
                #
                # # Finally, construct the full function expression.
                # treatment_assignment_logit_function = \
                #     max_min_capped_targeted_logit.subs(x, base_treatment_logit_expression)
        else:
            # Shortcircuiting all normalization.
            treatment_assignment_logit_function = base_treatment_logit_expression

        # Build the logistic propensity function.
        exponentiated_neg_logit = sp.functions.exp(-1*treatment_assignment_logit_function)

        treatment_assignment_function = 1/(1 + exponentiated_neg_logit)

        return (treatment_assignment_logit_function,
            treatment_assignment_function)

    def sample_treatment_effect_subfunction(self,
        outcome_covariate_transforms, observed_covariate_data):
        # 5A. Construct the treatment effect subfunction by sampling
        # from the set of transformed covariates which make up the
        # outcome function. The more transformed covariates appear
        # in the treatment function, the more heterogenous the treatment
        # effect.

        # Sample a base effect from the distribution in the parameters.
        base_treatment_effect = self.params.sample_treatment_effect()[0]

        # Sample outcome subfunctions to interact with base treatment effect.
        selected_interaction_terms = select_objects_given_probability(
                objects_to_sample=outcome_covariate_transforms,
                selection_probability=self.params.TREATMENT_EFFECT_HETEROGENEITY)

        # Initialize constants.
        initialized_interaction_terms = initialize_expression_constants(
            self.params.sample_subfunction_constants,
            selected_interaction_terms)

        # Build the covariate multiplier which will interact with the treat effect.
        treatment_effect_multiplier_expr = np.sum(initialized_interaction_terms)

        # Process interaction terms into treatment subfunction.
        if treatment_effect_multiplier_expr != 0:

            # Normalize multiplier size.

            # TODO: vaidate the approach to normalization used here (mean/std)
            # vs treatment assignment function approach.
            sampled_data = observed_covariate_data.sample(
                frac=SamplingConstants.NORMALIZATION_DATA_SAMPLE_FRACTION)

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
        # 5B. construct the outcome function analogously to the way
        # the treatment assignment function was constructed.

        # TODO: consider recursively applying the covariate transformation
        # to produce "deep" functions. Probability overkill.
        # TODO: add non-linear activation function and ensure proper normalization.
        # using or changing the OUTCOME_MECHANISM_EXPONENTIATION param.

        # Build base outcome function. Additive combination of the true covariates.
        base_untreated_outcome_expression = np.sum(outcome_covariate_transforms)

        # Normalize if config set to do so.
        if SamplingConstants.NORMALIZE_SAMPLED_OUTCOME_FUNCTION:
            # Sample data to evaluate distribution.
            sampled_data = observed_covariate_data.sample(
                frac=SamplingConstants.NORMALIZATION_DATA_SAMPLE_FRACTION)

            # Normalized outcome values to have approximate mean=0 and std=1.
            # This prevents situations where large outcome values drown out
            # the treatment effect or the treatment effect dominates small average outcomes.
            outcome_values = evaluate_expression(base_untreated_outcome_expression, sampled_data)
            outcome_mean = np.mean(outcome_values)
            outcome_std = np.std(outcome_values)

            # This is only an approximate normalization. It will shift the mean to zero
            # but the exact effect on std will depend on the distribution.
            normalized_outcome_expression = \
                (base_untreated_outcome_expression - outcome_mean)/outcome_std

            untreated_outcome_subfunction = normalized_outcome_expression
        else:
            untreated_outcome_subfunction = base_untreated_outcome_expression

        # Create the treatment effect subfunction.
        treat_effect_subfunction = self.sample_treatment_effect_subfunction(
            outcome_covariate_transforms, observed_covariate_data)

        outcome_function = untreated_outcome_subfunction + \
            ComponentConstants._OUTCOME_NOISE_SYMBOL + \
            (ComponentConstants._TREATMENT_ASSIGNMENT_SYMBOL *
                treat_effect_subfunction)

        return outcome_function, untreated_outcome_subfunction, treat_effect_subfunction
