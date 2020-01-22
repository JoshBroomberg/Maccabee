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
        optional, analysis_mode_only,
        func):
        self.generated_var = generated_var
        self.required_vars = required_vars
        self.optional = optional
        self.analysis_mode_only = analysis_mode_only
        self.func = func

    def __call__(self, *args, **kwargs):
        dgp = args[0]
        data_dict = getattr(dgp, GENERATED_DATA_DICT_NAME)

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
    optional=False, analysis_mode_only=False):
    return partial(DataGeneratingMethodWrapper,
        generated_var, required_vars,
        optional, analysis_mode_only)

#TODO: consider refctoring to splat in the args directly rather
# than via a dict.

# TODO: consider specifying order via a list of method names
# and then replace _generate validation to just use this list of names.

class DataGeneratingProcess(metaclass=DataGeneratingMethodClass):
    def __init__(self):
        setattr(self, GENERATED_DATA_DICT_NAME, {})

    # DGP submethods
    @data_generating_method(Constants.COVARIATES_NAME, [])
    def _generate_observed_covars(self, input_vars):
        raise NotImplementedError

    @data_generating_method(
        Constants.TRANSFORMED_COVARIATES_NAME,
        [Constants.COVARIATES_NAME],
        optional=True,
        analysis_mode_only=True)
    def _generate_transformed_covars(self, input_vars):
        return None

    @data_generating_method(
        Constants.PROPENSITY_SCORE_NAME,
        [],
        optional=True)
    def _generate_true_propensity_scores(self, input_vars):
        return None

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
        raise NotImplementedError

    @data_generating_method(
        Constants.POTENTIAL_OUTCOME_WITHOUT_TREATMENT_NAME,
        [Constants.COVARIATES_NAME, Constants.OUTCOME_NOISE_NAME])
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

    def _get_generated_data(self, name):
        data_dict = getattr(self, GENERATED_DATA_DICT_NAME)
        return data_dict.get(name, None)

    def _build_dataset(self):

        observed_covars = self._get_generated_data(Constants.COVARIATES_NAME)
        transformed_covars = self._get_generated_data(Constants.TRANSFORMED_COVARIATES_NAME)

        # Treatment assignment
        propensity_scores = self._get_generated_data(Constants.PROPENSITY_SCORE_NAME)
        propensity_logit = self._get_generated_data(Constants.PROPENSITY_LOGIT_NAME)
        T = self._get_generated_data(Constants.TREATMENT_ASSIGNMENT_NAME)

        # Outcome
        outcome_noise = self._get_generated_data(Constants.OUTCOME_NOISE_NAME)
        Y0 = self._get_generated_data(Constants.POTENTIAL_OUTCOME_WITHOUT_TREATMENT_NAME)
        TE = self._get_generated_data(Constants.TREATMENT_EFFECT_NAME)
        Y1 = self._get_generated_data(Constants.POTENTIAL_OUTCOME_WITH_TREATMENT_NAME)
        Y = self._get_generated_data(Constants.OBSERVED_OUTCOME_NAME)

        return DataSet(
            observed_covars=observed_covars,
            transformed_covars=transformed_covars,
            propensity_scores=propensity_scores,
            propensity_logit=propensity_logit,
            T=T,
            outcome_noise=outcome_noise,
            Y0=Y0, TE=TE, Y1=Y1, Y=Y)

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
        self._generate_outcomes_with_treatment()

        return self._build_dataset()



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

        # TODO: can be hidden + automated.
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
            data[f"{Constants.TRANSFORMED_COVARIATES_NAME}{index}"] = \
                evaluate_expression(transform, self.observed_covariate_data)

        return pd.DataFrame(data)

    def generate_dataset(self):
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
        observed_outcome_data = self._build_observed_outcome_df(T, Y)

        # Data not available for causal inference.
        oracle_outcome_data = self._build_oracle_outcome_df(
            propensity_logit=self.logit_values,
            propensity_scores=self.propensity_scores,
            Y0=Y0, Y1=Y1, treatment_effects=self.treatment_effects,
            outcome_noise=noise_samples)


        return self._build_dataset(
            observed_covariate_data=self.observed_covariate_data,
            oracle_covariate_data=self.oracle_covariate_data,
            observed_outcome_data=observed_outcome_data,
            oracle_outcome_data=oracle_outcome_data)
