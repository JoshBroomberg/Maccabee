from ..constants import Constants
from .data_set import DataSet
from ..utilities import evaluate_expression
import pandas as pd
import numpy as np
from functools import partial


GENERATED_DATA_DICT_NAME = "_generated_data"

class DataGeneratingMethodClass(type):
    def __init__(cls, name, bases, clsdict):
        for method_name, method_obj in clsdict.items():
            if method_name.startswith('_generate'):
                if type(method_obj) != DataGeneratingMethodWrapper:
                    raise Exception(
                        f"{method_obj} is a _generate* method without the data_generating_method decorator.")

        super(DataGeneratingMethodClass, cls).__init__(name, bases, clsdict)

class DataGeneratingMethodWrapper():
    def __init__(self,
        generated_var, required_vars,
        optional, analysis_mode_only, cache_result,
        func):

        self.generated_var = generated_var
        self.required_vars = required_vars
        self.optional = optional
        self.analysis_mode_only = analysis_mode_only
        self.cache_result = cache_result
        self.func = func

    def __call__(self, *args, **kwargs):
        dgp = args[0]
        data_dict = getattr(dgp, GENERATED_DATA_DICT_NAME)

        if self.cache_result and (self.generated_var in data_dict):
            return data_dict[self.generated_var]

        required_var_vals = {
            k:data_dict[k]
            for k in self.required_vars
            if k in data_dict
        }

        # Have all required inputs.
        if len(required_var_vals) == len(self.required_vars):
            # Only run analysis_mode_only methods if dgp in analysis mode.
            if dgp.analysis_mode or not self.analysis_mode_only:
                val = self.func(args[0], required_var_vals, *args[1:], **kwargs)
                data_dict[self.generated_var] = val
                return val

        # Do not have all required inputs.
        elif not self.optional:
            raise Exception(
                f"Missing required value in non-optional method: {self.func}" )

    def __get__(self, instance, owner):
        return partial(self.__call__, instance)

def data_generating_method(
    generated_var, required_vars,
    optional=False, analysis_mode_only=False, cache_result=False):
    return partial(DataGeneratingMethodWrapper,
        generated_var, required_vars,
        optional, analysis_mode_only, cache_result)

#TODO: consider refctoring to splat in the args directly rather
# than via a dict.

# TODO: consider specifying order via a list of method names
# and then replace _generate validation to just use this list of names.

class DataGeneratingProcess(metaclass=DataGeneratingMethodClass):
    def __init__(self, n_observations, analysis_mode=False):
        setattr(self, GENERATED_DATA_DICT_NAME, {})

        self.n_observations = n_observations
        self.analysis_mode = analysis_mode

    # DGP PROCESS
    def generate_dataset(self):
        # Covars
        self._generate_observed_covars()
        self._generate_transformed_covars()

        # Treatment assignment
        self._generate_true_propensity_scores()
        self._generate_true_propensity_score_logits()
        self._generate_treatment_assignments()

        # Outcomes
        self._generate_outcome_noise_samples()
        self._generate_outcomes_without_treatment()
        self._generate_treatment_effects()
        self._generate_outcomes_with_treatment()
        self._generate_observed_outcomes()

        return self._build_dataset()

    # DGP DEFINITION
    @data_generating_method(Constants.COVARIATES_NAME, [])
    def _generate_observed_covars(self, input_vars):
        raise NotImplementedError

    @data_generating_method(
        Constants.TRANSFORMED_COVARIATES_NAME,
        [Constants.COVARIATES_NAME],
        analysis_mode_only=True)
    def _generate_transformed_covars(self, input_vars):
        return None

    @data_generating_method(
        Constants.PROPENSITY_SCORE_NAME,
        [],
        optional=True)
    def _generate_true_propensity_scores(self, input_vars):
        raise NotImplementedError

    @data_generating_method(
        Constants.PROPENSITY_LOGIT_NAME,
        [Constants.PROPENSITY_SCORE_NAME],
        optional=True,
        analysis_mode_only=True)
    def _generate_true_propensity_score_logits(self, input_vars):
        propensity_scores = input_vars[Constants.PROPENSITY_SCORE_NAME]
        return np.log(propensity_scores/(1-propensity_scores))

    @data_generating_method(
        Constants.TREATMENT_ASSIGNMENT_NAME,
        [Constants.PROPENSITY_SCORE_NAME])
    def _generate_treatment_assignments(self, input_vars):
        propensity_scores = input_vars[Constants.PROPENSITY_SCORE_NAME]
        return (np.random.uniform(
            size=len(propensity_scores)) < propensity_scores).astype(int)


    @data_generating_method(Constants.OUTCOME_NOISE_NAME, [])
    def _generate_outcome_noise_samples(self, input_vars):
        return np.zeros(self.n_observations)

    @data_generating_method(
        Constants.POTENTIAL_OUTCOME_WITHOUT_TREATMENT_NAME,
        [Constants.COVARIATES_NAME])
    def _generate_outcomes_without_treatment(self, input_vars):
        raise NotImplementedError

    @data_generating_method(
        Constants.TREATMENT_EFFECT_NAME,
        [Constants.COVARIATES_NAME])
    def _generate_treatment_effects(self, input_vars):
        raise NotImplementedError

    @data_generating_method(
        Constants.POTENTIAL_OUTCOME_WITH_TREATMENT_NAME,
        [Constants.POTENTIAL_OUTCOME_WITHOUT_TREATMENT_NAME, Constants.TREATMENT_EFFECT_NAME])
    def _generate_outcomes_with_treatment(self, input_vars):
        outcome_without_treatment = input_vars[Constants.POTENTIAL_OUTCOME_WITHOUT_TREATMENT_NAME]
        treatment_effect = input_vars[Constants.TREATMENT_EFFECT_NAME]
        return outcome_without_treatment + treatment_effect

    @data_generating_method(
        Constants.OBSERVED_OUTCOME_NAME,
        [
            Constants.POTENTIAL_OUTCOME_WITHOUT_TREATMENT_NAME,
            Constants.TREATMENT_ASSIGNMENT_NAME,
            Constants.POTENTIAL_OUTCOME_WITH_TREATMENT_NAME,
        ])
    def _generate_observed_outcomes(self, input_vars):
        outcome_without_treatment = input_vars[Constants.POTENTIAL_OUTCOME_WITHOUT_TREATMENT_NAME]
        treatment_assignment = input_vars[Constants.TREATMENT_ASSIGNMENT_NAME]
        outcome_with_treatment = input_vars[Constants.POTENTIAL_OUTCOME_WITH_TREATMENT_NAME]
        return (treatment_assignment*outcome_with_treatment) + ((1-treatment_assignment)*outcome_without_treatment)

    # HELPER FUNCTIONS

    def _get_generated_data(self, name):
        data_dict = getattr(self, GENERATED_DATA_DICT_NAME)
        return data_dict.get(name, None)

    def _build_dataframe_for_vars(self, var_names):
        df = pd.DataFrame()
        for name in var_names:
            val = self._get_generated_data(name)
            df[name] = val

        return df

    def _build_dataset(self):

        # Observed data
        observed_covariate_data = self._get_generated_data(
            Constants.COVARIATES_NAME)

        observed_outcome_data = self._build_dataframe_for_vars([
            Constants.TREATMENT_ASSIGNMENT_NAME,
            Constants.OBSERVED_OUTCOME_NAME
        ])

        # Unobserved data
        transformed_covariate_data = self._get_generated_data(
            Constants.TRANSFORMED_COVARIATES_NAME)

        oracle_outcome_data = self._build_dataframe_for_vars([
            Constants.PROPENSITY_SCORE_NAME,
            Constants.PROPENSITY_LOGIT_NAME,
            Constants.OUTCOME_NOISE_NAME,
            Constants.POTENTIAL_OUTCOME_WITHOUT_TREATMENT_NAME,
            Constants.POTENTIAL_OUTCOME_WITH_TREATMENT_NAME,
            Constants.TREATMENT_EFFECT_NAME,
        ])

        # Treatment assignment
        return DataSet(
            observed_covariate_data=observed_covariate_data,
            observed_outcome_data=observed_outcome_data,
            oracle_outcome_data=oracle_outcome_data,
            transformed_covariate_data=transformed_covariate_data)

class ManualDataGeneratingProcess(DataGeneratingProcess):
    pass

class SampledDataGeneratingProcess(DataGeneratingProcess):
    def __init__(self,
        params,
        observed_covariate_data,
        outcome_covariate_transforms,
        treatment_covariate_transforms,
        treatment_assignment_function,
        treatment_effect_subfunction,
        base_outcome_subfunction,
        treatment_assignment_logit_func=None,
        outcome_function=None,
        analysis_mode=True):

        # STANDARD CONFIG
        n_observations = observed_covariate_data.shape[0]
        super().__init__(n_observations, analysis_mode)

        # SAMPLED SGP CONFIG
        self.params = params

        # Sampled covariate transforms for the treat and outcome functions.
        self.outcome_covariate_transforms = outcome_covariate_transforms
        self.treatment_covariate_transforms = treatment_covariate_transforms

        # Treatment assignment function and subfunctions
        self.treatment_assignment_logit_function = treatment_assignment_logit_func
        self.treatment_assignment_function = treatment_assignment_function

        # Outcome function and subfunctions
        self.treatment_effect_subfunction = treatment_effect_subfunction
        self.base_outcome_subfunction = base_outcome_subfunction
        self.outcome_function = outcome_function

        # DATA
        self.observed_covariate_data = observed_covariate_data

    @data_generating_method(Constants.COVARIATES_NAME, [], cache_result=True)
    def _generate_observed_covars(self, input_vars):
        return self.observed_covariate_data

    @data_generating_method(
        Constants.TRANSFORMED_COVARIATES_NAME,
        [Constants.COVARIATES_NAME],
        analysis_mode_only=True,
        cache_result=True)
    def _generate_transformed_covars(self, input_vars):
        # Generate the values of all the transformed covariates by running the
        # original covariate data through the transforms used in the outcome and
        # treatment functions.

        observed_covariate_data = input_vars[Constants.COVARIATES_NAME]

        all_transforms = list(set(self.outcome_covariate_transforms).union(
            self.treatment_covariate_transforms))

        data = {}
        for index, transform in enumerate(all_transforms):
            data[f"{Constants.TRANSFORMED_COVARIATES_NAME}{index}"] = \
                evaluate_expression(transform, observed_covariate_data)

        return pd.DataFrame(data)


    @data_generating_method(
        Constants.PROPENSITY_SCORE_NAME,
        [Constants.COVARIATES_NAME],
        cache_result=True)
    def _generate_true_propensity_scores(self, input_vars):
        observed_covariate_data = input_vars[Constants.COVARIATES_NAME]

        return evaluate_expression(
            self.treatment_assignment_function,
            observed_covariate_data)

    @data_generating_method(
        Constants.TREATMENT_ASSIGNMENT_NAME,
        [Constants.PROPENSITY_SCORE_NAME])
    def _generate_treatment_assignments(self, input_vars):
        propensity_scores = input_vars[Constants.PROPENSITY_SCORE_NAME]

        # Sample treatment assignment given pre-calculated propensity_scores
        T = (np.random.uniform(
            size=self.n_observations) < propensity_scores).astype(int)

        # Only perform balance adjustment if there is some heterogeneity
        # in the propensity scores.
        if not np.all(np.isclose(propensity_scores, propensity_scores[0])):
            # Balance adjustment
            control_p_scores = propensity_scores.where(T == 0)
            treat_p_scores = propensity_scores.where(T == 1)

            num_controls = control_p_scores.count()
            n_to_switch = int(num_controls*self.params.FORCED_IMBALANCE_ADJUSTMENT)

            control_switch_targets = control_p_scores.nlargest(n_to_switch).index.values
            treat_switch_targets = treat_p_scores.nsmallest(n_to_switch).index.values

            T[control_switch_targets] = 1
            T[treat_switch_targets] = 0

        return T

    @data_generating_method(Constants.OUTCOME_NOISE_NAME, [])
    def _generate_outcome_noise_samples(self, input_vars):
        return self.params.sample_outcome_noise(size=self.n_observations)

    @data_generating_method(
        Constants.POTENTIAL_OUTCOME_WITHOUT_TREATMENT_NAME,
        [Constants.COVARIATES_NAME],
        cache_result=True)
    def _generate_outcomes_without_treatment(self, input_vars):
        observed_covariate_data = input_vars[Constants.COVARIATES_NAME]
        return evaluate_expression(
            self.base_outcome_subfunction,
            observed_covariate_data)

    @data_generating_method(
        Constants.TREATMENT_EFFECT_NAME,
        [Constants.COVARIATES_NAME],
        cache_result=True)
    def _generate_treatment_effects(self, input_vars):
        observed_covariate_data = input_vars[Constants.COVARIATES_NAME]
        return evaluate_expression(
            self.treatment_effect_subfunction,
            observed_covariate_data)

    @data_generating_method(
        Constants.POTENTIAL_OUTCOME_WITH_TREATMENT_NAME,
        [
            Constants.POTENTIAL_OUTCOME_WITHOUT_TREATMENT_NAME,
            Constants.TREATMENT_EFFECT_NAME
        ],
        cache_result=True)
    def _generate_outcomes_with_treatment(self, input_vars):
        outcome_without_treatment = input_vars[Constants.POTENTIAL_OUTCOME_WITHOUT_TREATMENT_NAME]
        treatment_effect = input_vars[Constants.TREATMENT_EFFECT_NAME]
        return outcome_without_treatment + treatment_effect

    @data_generating_method(
        Constants.OBSERVED_OUTCOME_NAME,
        [
            Constants.POTENTIAL_OUTCOME_WITHOUT_TREATMENT_NAME,
            Constants.TREATMENT_ASSIGNMENT_NAME,
            Constants.POTENTIAL_OUTCOME_WITH_TREATMENT_NAME,
            Constants.OUTCOME_NOISE_NAME
        ])
    def _generate_observed_outcomes(self, input_vars):
        outcome_without_treatment = input_vars[Constants.POTENTIAL_OUTCOME_WITHOUT_TREATMENT_NAME]
        treatment_assignment = input_vars[Constants.TREATMENT_ASSIGNMENT_NAME]
        outcome_with_treatment = input_vars[Constants.POTENTIAL_OUTCOME_WITH_TREATMENT_NAME]
        outcome_noise_samples = input_vars[Constants.OUTCOME_NOISE_NAME]
        Y = (
            (treatment_assignment*outcome_with_treatment) +
            ((1-treatment_assignment)*outcome_without_treatment) +
            outcome_noise_samples)

        return Y
