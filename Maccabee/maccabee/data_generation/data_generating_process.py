"""This module contains the :class:`~maccabee.data_generation.data_generating_process.DataGeneratingProcess` base class that is used to represent sampled and concrete DGPs. Within this class, the DGP is represented as a series of data generating methods which each produce a :term:`DGP variable <dgp variable>` and can depend on other, previously generated, DGP variables. The methods are called in a predetermined order in the main :meth:`~maccabee.data_generation.data_generating_process.DataGeneratingProcess.generate_dataset` method in order to sample a data set.

The base :class:`~maccabee.data_generation.data_generating_process.DataGeneratingProcess` class and its inheriting classes make use of a minimal :term:`DSL <dsl>` which is used to specify the data flow in the DGP. This DSL reduces boilerplate code by automatically managing dgp method dependencies, outputs, and execution. In the future, it will allow for advanced features like dependency resolution and parralelism. The DSL is used by decorating all data generating methods in a :class:`~maccabee.data_generation.data_generating_process.DataGeneratingProcess` class with the :func:`~maccabee.data_generation.data_generating_process.data_generating_method` decorator. This decorator is parameterized
with the DGP variables which the method required, the DGP variable it produces, and other options. See the documentation below for more detail.

The documentation below explains the :func:`~maccabee.data_generation.data_generating_process.data_generating_method` decorator and the :class:`~maccabee.data_generation.data_generating_process.DataGeneratingProcess` class (and its data generating methods) in more detail.
"""

from ..constants import Constants
from ..exceptions import DGPVariableMissingException, DGPInvalidSpecificationException
from .generated_data_set import GeneratedDataSet
from .utils import evaluate_expression, CompiledExpression
import pandas as pd
import numpy as np
from functools import partial, update_wrapper
import types
import sympy as sp

from ..logging import get_logger
logger = get_logger(__name__)

GENERATED_DATA_DICT_NAME = "_generated_data"
DGPVariables = Constants.DGPVariables

class DataGeneratingMethodContainerClass(type):
    # This is a meta-class which is applied to the base DataGeneratingProcess
    # class and ensures that all _generate_* methods are properly decorated
    # in order to conform to the DSL.
    def __init__(cls, name, bases, clsdict):
        for method_name, method_obj in clsdict.items():
            if method_name.startswith('_generate'):
                if type(method_obj) != DataGeneratingMethodWrapper:
                    raise DGPInvalidSpecificationException(method_obj)

        super(DataGeneratingMethodContainerClass, cls).__init__(name, bases, clsdict)

class DataGeneratingMethodWrapper():
    # This class is used internally by the DSL to represent and manage
    # data generating methods. It stores the DGP variables required
    # and produced by the target method as well as meta-data about
    # whether results should be cached, whether the method is optional,
    # and whether it is data-analysis mode only.
    def __init__(self,
        generated_var, required_vars,
        optional, data_analysis_mode_only, cache_result,
        func):

        self.generated_var = generated_var
        self.required_vars = required_vars
        self.optional = optional
        self.data_analysis_mode_only = data_analysis_mode_only
        self.cache_result = cache_result
        self.func = func

        self.wrapped_call = partial(DataGeneratingMethodWrapper.call, self)
        update_wrapper(self.wrapped_call, self.func)

    @staticmethod
    def call(wrapper, dgp, *args, **kwargs):
        # This is the code which is run when a data generating method
        # is executed.
        logger.debug(f"Starting execution of wrapped data generating function: {wrapper.func}")
        # Get the central storage data structure.
        data_dict = getattr(dgp, GENERATED_DATA_DICT_NAME)

        # Check if there is a valid cache hit and return it.
        if wrapper.cache_result and (wrapper.generated_var in data_dict):
            logger.debug("Return cached result for func")
            return data_dict[wrapper.generated_var]

        # Verify that all required variables have been generated.
        required_var_vals = {
            k:data_dict[k]
            for k in wrapper.required_vars
            if k in data_dict
        }

        # Only run if all requirements generated.
        if len(required_var_vals) == len(wrapper.required_vars):

            # Only run data_analysis_mode_only methods if dgp in analysis mode.
            if dgp.data_analysis_mode or not wrapper.data_analysis_mode_only:
                logger.debug("Executing wrapped data generating callable.")
                # Run the stored function.
                val = wrapper.func(dgp, required_var_vals, *args, **kwargs)

                # Store the value in the data dict.
                data_dict[wrapper.generated_var] = val

                return val
            else:
                logger.debug(f"Skipping execution of data analysis func. DGP data analysis mode {dgp.data_analysis_mode}.")

        # Do not have all required inputs.
        elif not wrapper.optional:
            msg = f"Missing required value in non-optional method: {func}"
            raise DGPVariableMissingException(msg)

    def __get__(self, instance, owner):
        # This is the descriptor method which returns the bound/unbound function
        # at the correct times to emulate a DGP instance method.

        # Access at the class level, return the unbound func
        # for documentation etc.
        if instance is None:
            logger.debug("Returning unbound version of data generating method")
            return self.func

        # Access at instance level, return a bound instance of the method.
        else:
            # Bind the DGP instance to the call method to give it
            # access to the instance at execution time.
            logger.debug("Returning bound version of data generating method")
            return types.MethodType(self.wrapped_call, instance)

def data_generating_method(
    generated_var, required_vars,
    optional=False, data_analysis_mode_only=False, cache_result=False):
    """This DGP DSL decorator is applied to all of the ``_generate_*`` methods which comprise the definition of a :class:`~maccabee.data_generation.data_generating_process.DataGeneratingProcess` class. The decorator takes parameters which describe the DGP variables which the generating method requires and generates and a number of other parameters relevant to execution.

    Args:
        generated_var (string): A string DGP variable name from :class:`~maccabee.constants.Constants.DGPVariables`. This is the DGP variable which the decorated method generates.
        required_vars (list): A list of string DGP variable names from :class:`~maccabee.constants.Constants.DGPVariables`. These are DGP variables which the decorated method requires to generate its variable. The values of these variables are passed as a dictionary to the decorated method as the first position argument.
        optional (bool): Indicates whether the decorated method is optional. If ``True``, the method will only run if its requirements are satisfied and there will not be an exception raised if its requirements are missing. If ``False``, there will be an exception if required variables are missing at execution time. Defaults to ``False``.
        data_analysis_mode_only (bool): Indicates if the decorated method should only be run if the DGP is in data analysis mode. IE, the generated DGP variable is required only for data metric calculation. Defaults to False.
        cache_result (bool): Indicates whether the result of the decorated method should be cached so that all samples from the DGP after the first will have the same value of the generated variable. Defaults to False.

    .. warning::
        If ``cache_result=True`` and the decorated method depends on other variables which change (IE, they are not cached), the changes to these variables will not reflect in the variable generated by the decorated method.

    Raises:
        DGPVariableMissingException: if a non-optional decorated method is missing its required variables at execution time.
    """
    return partial(DataGeneratingMethodWrapper,
        generated_var, required_vars,
        optional, data_analysis_mode_only, cache_result)


class DataGeneratingProcess(metaclass=DataGeneratingMethodContainerClass):
    """This class represents a Data Generating Process. A DGP relates the DGP Variables - defined in the constants group :class:`~maccabee.constants.Constants.DGPVariables` - through a series of stochastic/deterministic 'data generating functions'. The nature of these functions defines the location of the resultant data sets in the :term:`distributional problem space`.

    This is the base DGP class. It defines the data generating functions which make up a DGP by generating all of the required DGP variables. These functions are defined without providing concrete implementations (with exceptions made for the data generating functions that are reasonably generic). Parameterized DGP DSL decorators are provided for guidance (see the source code). Inheriting classes are expected to provide the data generating function implementations and to redecorate implemented methods (repeating the generated variable and specifying the correct dependencies and execution options). All data generating functions with concrete implementatins are marked with [CONCRETE].

    This class does define a concrete :meth:`~maccabee.data_generation.data_generating_process.DataGeneratingProcess.generate_dataset` method which specifies the order of execution of the data generating functions and constructs a :class:`~maccabee.data_generation.generated_data_set.GeneratedDataSet` instance from the DGP variables produced by the data generating functions.

    Args:
        n_observations (int): The number of observations which will be present in sampled data sets. This value is used throughout Maccabee to build the correct data structures (and it is useful throughout this class) so it must be specified priori to sampling.
        data_analysis_mode (bool): Indicates whether the DGP should be run in data analysis mode. This will execute all data generating methods marked as `data_analysis_mode_only` in order to generate the dgp variables which are only used in calculating data metrics. Defaults to False.

    Attributes:
        n_observations
        data_analysis_mode

    """
    def __init__(self, n_observations, data_analysis_mode=False):
        setattr(self, GENERATED_DATA_DICT_NAME, {})

        self.n_observations = n_observations
        self.data_analysis_mode = data_analysis_mode

    def set_data_analysis_mode(self, val):
        self.data_analysis_mode = val

    def get_data_analysis_mode(self):
        return self.data_analysis_mode

    # DGP PROCESS
    def generate_dataset(self):
        """This is the primary external API method of this class. It is used to sample a data set (in the form of a :class:`~maccabee.data_generation.generated_data_set.GeneratedDataSet` instance) from the DGP.

        Returns:
            :class:`~maccabee.data_generation.generated_data_set.GeneratedDataSet`: a sampled :class:`~maccabee.data_generation.generated_data_set.GeneratedDataSet` instance.

        Raises:
            DGPVariableMissingException: If the execution order of the data generating methods is in conflict with their specified requirements such that a method's dependencies haven't been generated when it is executed.
        """
        # TODO-FUTURE: consider specifying execution order either via a list of
        # method names or ordering values in method meta-data. This would
        # improve flexibility and extensibility of this base class.
        # If a list is used, replace the _generate* validation and use the list
        # instead.

        # Covars
        logger.debug("Generating Observed Covariates")
        self._generate_observed_covars()
        self._generate_transformed_covars()

        # Treatment assignment
        logger.debug("Generating propensity scores")
        self._generate_true_propensity_scores()
        self._generate_true_propensity_score_logits()

        logger.debug("Generating treatment assignments")
        self._generate_treatment_assignments()

        # Outcomes
        logger.debug("Generating outcome noise")
        self._generate_outcome_noise_samples()

        logger.debug("Generating outcome without treatment")
        self._generate_outcomes_without_treatment()

        logger.debug("Generating treatment effects")
        self._generate_treatment_effects()

        logger.debug("Generating outcome with treatment")
        self._generate_outcomes_with_treatment()

        logger.debug("Generating observed outcomes")
        self._generate_observed_outcomes()

        generated_data_dict = getattr(self, GENERATED_DATA_DICT_NAME)
        return GeneratedDataSet(generated_data_dict)

    # DGP DEFINITION
    @data_generating_method(DGPVariables.COVARIATES_NAME, [])
    def _generate_observed_covars(self, input_vars):
        """_generate_observed_covars(...)

        Generate the observed covariates (``DGPVariables.COVARIATES_NAME``). It is likely that this function can be implemented by using the :meth:`~maccabee.data_sources.data_sources.DataSource.get_covar_df` method of a :class:`~maccabee.data_sources.data_sources.DataSource` instance.

        Returns:
            :class:`pandas.DataFrame`: a :class:`~pandas.DataFrame` containing the covariate observations. It must contain `n_observations` rows.
        """
        raise NotImplementedError

    @data_generating_method(
        DGPVariables.TRANSFORMED_COVARIATES_NAME,
        [DGPVariables.COVARIATES_NAME],
        data_analysis_mode_only=True)
    def _generate_transformed_covars(self, input_vars):
        """_generate_transformed_covars(...)

        Generate the transformed covariates (``DGPVariables.TRANSFORMED_COVARIATES_NAME``). This is only possible if the treatment/outcome functions are additive functions of arbitrary covariate transforms. The method is typically used in data analysis mode only as this DGP variable is not required for causal inference.

        Returns:
            :class:`pandas.DataFrame`: a :class:`~pandas.DataFrame` containing the transformed covariate observations. It must contain `n_observations` rows.
        """
        return None

    @data_generating_method(
        DGPVariables.PROPENSITY_SCORE_NAME,
        [],
        optional=True)
    def _generate_true_propensity_scores(self, input_vars):
        """_generate_true_propensity_scores(...)

        Generate the true propensity scores for each observed unit (``DGPVariables.PROPENSITY_SCORE_NAME``). This implements the treatment assignment function if the function naturally generates probabilities of treatment. It is not necessary as treatment assignments can be generated directly. See :meth:`~maccabee.data_generation.data_generating_process.DataGeneratingProcess._generate_treatment_assignments`.

        Returns:
            :class:`numpy.ndarray`: an array containing the probability of treatment. It must contain `n_observations` entries.
        """
        raise NotImplementedError

    @data_generating_method(
        DGPVariables.PROPENSITY_LOGIT_NAME,
        [DGPVariables.PROPENSITY_SCORE_NAME],
        optional=True,
        data_analysis_mode_only=True)
    def _generate_true_propensity_score_logits(self, input_vars):
        """_generate_true_propensity_score_logits(...)

        [CONCRETE] Generate the true propensity score logits for each observed unit (``DGPVariables.PROPENSITY_LOGIT_NAME``). A concrete implementation is provided which simply calculates the logit of the propensity scores generated by :meth:`~maccabee.data_generation.data_generating_process.DataGeneratingProcess._generate_true_propensity_scores`. In most common use cases, this will be a data analysis only mode function as the logits are not required for causal inference or treatment assignment (if the propensities are known).

        Returns:
            :class:`numpy.ndarray`: an array containing the logit probability of treatment. It must contain `n_observations` entries.
        """
        propensity_scores = input_vars[DGPVariables.PROPENSITY_SCORE_NAME]
        return np.log(propensity_scores/(1-propensity_scores))

    @data_generating_method(
        DGPVariables.TREATMENT_ASSIGNMENT_NAME,
        [DGPVariables.PROPENSITY_SCORE_NAME])
    def _generate_treatment_assignments(self, input_vars):
        """_generate_treatment_assignments(...)

        [CONCRETE] Generate the treatment assignment for each observed unit (``DGPVariables.TREATMENT_ASSIGNMENT_NAME``). A concrete implementation is provided which assigns treatment based on the propensity scores generated by :meth:`~maccabee.data_generation.data_generating_process.DataGeneratingProcess._generate_true_propensity_scores`. This function can be used to assign treatment even if propensity scores are never generated/known.

        Returns:
            :class:`numpy.ndarray`: an array containing the treatment assignment as a integer. 1 for treatment and 0 for control. It must contain `n_observations` entries.
        """
        propensity_scores = input_vars[DGPVariables.PROPENSITY_SCORE_NAME]
        return (np.random.uniform(
            size=len(propensity_scores)) < propensity_scores).astype(int)


    @data_generating_method(DGPVariables.OUTCOME_NOISE_NAME, [])
    def _generate_outcome_noise_samples(self, input_vars):
        """_generate_outcome_noise_samples(...)

        [CONCRETE] Generate the outcome noise for each observed unit (``DGPVariables.OUTCOME_NOISE_NAME``). A concrete implementation is provided which generates a zero noise vector.

        Returns:
            :class:`numpy.ndarray`: an array containing the outcome noise for each observation as a real value. It must contain `n_observations` entries.
        """
        return np.zeros(self.n_observations)

    @data_generating_method(
        DGPVariables.POTENTIAL_OUTCOME_WITHOUT_TREATMENT_NAME,
        [DGPVariables.COVARIATES_NAME])
    def _generate_outcomes_without_treatment(self, input_vars):
        """_generate_outcomes_without_treatment(...)

        Generate the potential outcome without treatment for each observed unit (``DGPVariables.POTENTIAL_OUTCOME_WITHOUT_TREATMENT_NAME``). This implements the base outcome function - without noise or treatment effect.

        Returns:
            :class:`numpy.ndarray`: an array containing the potential outcome without treatment for each observation as a real value. It must contain `n_observations` entries.
        """
        raise NotImplementedError

    @data_generating_method(
        DGPVariables.TREATMENT_EFFECT_NAME,
        [DGPVariables.COVARIATES_NAME])
    def _generate_treatment_effects(self, input_vars):
        """_generate_treatment_effects(...)

        Generate the treatment effect for each observed unit (``DGPVariables.TREATMENT_EFFECT_NAME``). This implements the treatment effect function.

        Returns:
            :class:`numpy.ndarray`: an array containing the treatment effect for each observation as a real value. It must contain `n_observations` entries.
        """
        raise NotImplementedError

    @data_generating_method(
        DGPVariables.POTENTIAL_OUTCOME_WITH_TREATMENT_NAME,
        [DGPVariables.POTENTIAL_OUTCOME_WITHOUT_TREATMENT_NAME,
        DGPVariables.TREATMENT_EFFECT_NAME])
    def _generate_outcomes_with_treatment(self, input_vars):
        """_generate_outcomes_with_treatment(...)

        [CONCRETE] Generate the potential outcome with treatment for each observed unit (``DGPVariables.POTENTIAL_OUTCOME_WITH_TREATMENT_NAME``). A concrete implementation is provided. It sums the potential outcome without treatment generated by :meth:`~maccabee.data_generation.data_generating_process.DataGeneratingProcess._generate_outcomes_without_treatment` and the the treatment effect generated by :meth:`~maccabee.data_generation.data_generating_process.DataGeneratingProcess._generate_treatment_effects`.

        Returns:
            :class:`numpy.ndarray`: an array containing the potential outcome with treatment for each observation as a real value. It must contain `n_observations` entries.
        """
        outcome_without_treatment = input_vars[DGPVariables.POTENTIAL_OUTCOME_WITHOUT_TREATMENT_NAME]
        treatment_effect = input_vars[DGPVariables.TREATMENT_EFFECT_NAME]
        return outcome_without_treatment + treatment_effect

    @data_generating_method(
        DGPVariables.OBSERVED_OUTCOME_NAME,
        [
            DGPVariables.POTENTIAL_OUTCOME_WITHOUT_TREATMENT_NAME,
            DGPVariables.TREATMENT_ASSIGNMENT_NAME,
            DGPVariables.OUTCOME_NOISE_NAME,
            DGPVariables.POTENTIAL_OUTCOME_WITH_TREATMENT_NAME,
        ])
    def _generate_observed_outcomes(self, input_vars):
        """_generate_observed_outcomes(...)

        [CONCRETE] Generate the observed outcome for each observed unit (``DGPVariables.OBSERVED_OUTCOME_NAME``). A concrete implementation is provided. It selects either the the potential outcome with or without treatment generated by :meth:`~maccabee.data_generation.data_generating_process.DataGeneratingProcess._generate_outcomes_with_treatment`/:meth:`~maccabee.data_generation.data_generating_process.DataGeneratingProcess._generate_outcomes_without_treatment` based on the treatment assignment generated by :meth:`~maccabee.data_generation.data_generating_process.DataGeneratingProcess._generate_treatment_assignments` and then adds the outcome noise generated by :meth:`~maccabee.data_generation.data_generating_process.DataGeneratingProcess._generate_outcome_noise_samples`

        Returns:
            :class:`numpy.ndarray`: an array containing the observed outcome for each observation as a real value. It must contain `n_observations` entries.
        """
        # T
        treatment_assignment = input_vars[DGPVariables.TREATMENT_ASSIGNMENT_NAME]

        # Y0
        outcome_without_treatment = input_vars[DGPVariables.POTENTIAL_OUTCOME_WITHOUT_TREATMENT_NAME]

        # Y1
        outcome_with_treatment = input_vars[DGPVariables.POTENTIAL_OUTCOME_WITH_TREATMENT_NAME]

        # Noise
        outcome_noise_samples = input_vars[DGPVariables.OUTCOME_NOISE_NAME]

        # T*Y1 + (1-T)*Y0 + Noise
        return (treatment_assignment*outcome_with_treatment) + ((1-treatment_assignment)*outcome_without_treatment) + outcome_noise_samples

class ConcreteDataGeneratingProcess(DataGeneratingProcess):
    """
    """
    pass

class SampledDataGeneratingProcess(DataGeneratingProcess):
    """
    EDIT THIS:
    In Maccabee, a :class:`~maccabee.data_generation.data_generating_process.DataGeneratingProcess` combines a covariate :class:`~maccabee.data_sources.data_sources.DataSource` and concrete/sampled treatment and outcome functions. These two components provide all the information required to draw sampled data sets.
    """
    def __init__(self,
        params,
        observed_covariate_data,
        outcome_covariate_transforms,
        treatment_covariate_transforms,
        treatment_assignment_function,
        treatment_effect_subfunction,
        untreated_outcome_subfunction,
        treatment_assignment_logit_func=None,
        outcome_function=None,
        data_source=None,
        data_analysis_mode=True,
        compile_functions=False):

        # STANDARD CONFIG
        n_observations = observed_covariate_data.shape[0]
        super().__init__(n_observations, data_analysis_mode)

        if compile_functions:
            symbols = sp.symbols(list(observed_covariate_data.columns))
            treatment_effect_subfunction = \
                CompiledExpression(treatment_effect_subfunction, symbols)
            untreated_outcome_subfunction = \
                CompiledExpression(untreated_outcome_subfunction, symbols)
            treatment_assignment_function = \
                CompiledExpression(treatment_assignment_function, symbols)


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
        self.untreated_outcome_subfunction = untreated_outcome_subfunction
        self.outcome_function = outcome_function

        # DATA
        self.observed_covariate_data = observed_covariate_data

        self.data_source = data_source

    @data_generating_method(DGPVariables.COVARIATES_NAME, [], cache_result=True)
    def _generate_observed_covars(self, input_vars):
        return self.observed_covariate_data

    @data_generating_method(
        DGPVariables.TRANSFORMED_COVARIATES_NAME,
        [DGPVariables.COVARIATES_NAME],
        data_analysis_mode_only=True,
        cache_result=False)
    def _generate_transformed_covars(self, input_vars):
        # Generate the values of all the transformed covariates by running the
        # original covariate data through the transforms used in the outcome and
        # treatment functions.

        observed_covariate_data = input_vars[DGPVariables.COVARIATES_NAME]

        all_transforms = list(set(self.outcome_covariate_transforms).union(
            self.treatment_covariate_transforms))

        data = {}
        for index, transform in enumerate(all_transforms):
            data[f"{DGPVariables.TRANSFORMED_COVARIATES_NAME}{index}"] = \
                evaluate_expression(transform, observed_covariate_data)

        return pd.DataFrame(data)


    @data_generating_method(
        DGPVariables.PROPENSITY_SCORE_NAME,
        [DGPVariables.COVARIATES_NAME],
        cache_result=False)
    def _generate_true_propensity_scores(self, input_vars):
        observed_covariate_data = input_vars[DGPVariables.COVARIATES_NAME]

        return evaluate_expression(
            self.treatment_assignment_function,
            observed_covariate_data)

    @data_generating_method(
        DGPVariables.TREATMENT_ASSIGNMENT_NAME,
        [DGPVariables.PROPENSITY_SCORE_NAME])
    def _generate_treatment_assignments(self, input_vars):
        propensity_scores = input_vars[DGPVariables.PROPENSITY_SCORE_NAME]

        # Sample treatment assignment given pre-calculated propensity_scores
        T = (np.random.uniform(
            size=self.n_observations) < propensity_scores).astype(int)

        # Only perform balance adjustment if there is some heterogeneity
        # in the propensity scores.
        try:
            # Do not run if all propensity scores are very similar.
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
        except IndexError:
            # Catch error thrown by very small/large treatment groups.
            pass

        return T

    @data_generating_method(DGPVariables.OUTCOME_NOISE_NAME, [])
    def _generate_outcome_noise_samples(self, input_vars):
        return self.params.sample_outcome_noise(size=self.n_observations)

    @data_generating_method(
        DGPVariables.POTENTIAL_OUTCOME_WITHOUT_TREATMENT_NAME,
        [DGPVariables.COVARIATES_NAME],
        cache_result=True)
    def _generate_outcomes_without_treatment(self, input_vars):
        observed_covariate_data = input_vars[DGPVariables.COVARIATES_NAME]
        return evaluate_expression(
            self.untreated_outcome_subfunction,
            observed_covariate_data)

    @data_generating_method(
        DGPVariables.TREATMENT_EFFECT_NAME,
        [DGPVariables.COVARIATES_NAME],
        cache_result=True)
    def _generate_treatment_effects(self, input_vars):
        observed_covariate_data = input_vars[DGPVariables.COVARIATES_NAME]
        return evaluate_expression(
            self.treatment_effect_subfunction,
            observed_covariate_data)

    @data_generating_method(
        DGPVariables.POTENTIAL_OUTCOME_WITH_TREATMENT_NAME,
        [
            DGPVariables.POTENTIAL_OUTCOME_WITHOUT_TREATMENT_NAME,
            DGPVariables.TREATMENT_EFFECT_NAME
        ],
        cache_result=True)
    def _generate_outcomes_with_treatment(self, input_vars):
        outcome_without_treatment = input_vars[DGPVariables.POTENTIAL_OUTCOME_WITHOUT_TREATMENT_NAME]
        treatment_effect = input_vars[DGPVariables.TREATMENT_EFFECT_NAME]
        return outcome_without_treatment + treatment_effect

    @data_generating_method(
        DGPVariables.OBSERVED_OUTCOME_NAME,
        [
            DGPVariables.POTENTIAL_OUTCOME_WITHOUT_TREATMENT_NAME,
            DGPVariables.TREATMENT_ASSIGNMENT_NAME,
            DGPVariables.POTENTIAL_OUTCOME_WITH_TREATMENT_NAME,
            DGPVariables.OUTCOME_NOISE_NAME
        ])
    def _generate_observed_outcomes(self, input_vars):
        outcome_without_treatment = input_vars[DGPVariables.POTENTIAL_OUTCOME_WITHOUT_TREATMENT_NAME]
        treatment_assignment = input_vars[DGPVariables.TREATMENT_ASSIGNMENT_NAME]
        outcome_with_treatment = input_vars[DGPVariables.POTENTIAL_OUTCOME_WITH_TREATMENT_NAME]
        outcome_noise_samples = input_vars[DGPVariables.OUTCOME_NOISE_NAME]
        Y = (
            (treatment_assignment*outcome_with_treatment) +
            ((1-treatment_assignment)*outcome_without_treatment) +
            outcome_noise_samples)

        return Y
