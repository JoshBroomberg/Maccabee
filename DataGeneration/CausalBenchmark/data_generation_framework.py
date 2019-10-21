from sympy.abc import a, c, x, y, z
import sympy as sp
import numpy as np
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
from .constants import Constants
from .parameters import Parameters
from .utilities import select_given_probability_distribution, evaluate_expression, initialize_expression_constants

def build_data_frame(covar_data, covar_names):
  n_observations = covar_data.shape[0]

  # Build DF
  data = pd.DataFrame(
      data=covar_data,
      columns=covar_names,
      index=np.arange(n_observations))

  # Sample and add noise column
  noise_samples = Parameters.sample_outcome_noise(size=n_observations)
  data[str(Constants.OUTCOME_NOISE_SYMBOL)] = noise_samples

  # Sample observations to reduce observation count if desired.
  data = data.sample(frac=Parameters.OBSERVATION_PROBABILITY)

  # generate symbols for all covariates
  covar_symbols = np.array(sp.symbols(covar_names))

  return data, covar_symbols


"""### Generate new covariate spaces for Treat/Outcome functions"""

def generate_transformed_covariate_space(covariate_symbols, subfunction_form_probabilities):
  subfunctions = []
  for subfunction_form_name, subfunction_form in Constants.SUBFUNCTION_FORMS.items():
    subfunc_expression = subfunction_form[Constants.EXPRESSION_KEY]
    subfunc_covariate_symbols = subfunction_form[Constants.COVARIATE_SYMBOLS_KEY]

    # All possible combinations of covariates for the given subfunc.
    covariate_combinations = np.array(list(combinations(covariate_symbols,
                                        len(subfunc_covariate_symbols))))

    selected_covar_combinations, _ = select_given_probability_distribution(
      full_list=covariate_combinations,
      selection_probabilities=subfunction_form_probabilities[subfunction_form_name])


    subfunctions.extend([
                          subfunc_expression.subs(
                              zip(subfunc_covariate_symbols, covar_comb))
                          for covar_comb in selected_covar_combinations
                        ])


  return subfunctions

""" Determine allignment/confounding between treat and outcome"""

def generate_aligned_treatment_and_outcome_covariates(covariate_symbols):
  true_outcome_covariates = generate_transformed_covariate_space(
      covariate_symbols,
      Parameters.OUTCOME_MECHANISM_COVARIATE_SELECTION_PROBABILITY)

  true_treatment_covariates = generate_transformed_covariate_space(
      covariate_symbols,
      Parameters.TREAT_MECHANISM_COVARIATE_SELECTION_PROBABILITY)

  # Unique set of all covariates
  true_covariates = np.array(list(set(true_treatment_covariates + true_outcome_covariates)))

  # Select overlapping covariates (confounders) given alignment
  # parameter.
  shared_confounders, _ = select_given_probability_distribution(
      true_covariates,
      selection_probabilities=Parameters.ACTUAL_CONFOUNDER_ALIGNMENT)
  shared_confounders = set(shared_confounders)

  # Union the true confounders into the original covariate selections.
  true_outcome_covariates = list(shared_confounders.union(true_outcome_covariates))
  true_treatment_covariates = list(shared_confounders.union(true_treatment_covariates))

  return true_outcome_covariates, true_treatment_covariates

"""Create treatment function"""

def generate_treatment_function(treatment_subfunctions, data):
  # TODO: consider recursively applying the covariate transformation
  # to produce "deep" functions. Probability overkill.

  # TODO: enable overlap adjustment

  # Randomly initialize subfunction constants
  initialized_subfunctions = initialize_expression_constants(treatment_subfunctions)

  # Build base treatment logit function. Additive combination of the true covariates.
  # TODO: add non-linear activation function
  base_treatment_logit_expression = np.sum(initialized_subfunctions)

  # Sample data to evaluate distribution.
  sampled_data = data.sample(frac=0.25)

  # Adjust logit
  logit_values = evaluate_expression(base_treatment_logit_expression, sampled_data)
  max_logit = np.max(logit_values)
  min_logit = np.min(logit_values)
  mean_logit = np.mean(logit_values)

  # Rescale to meet min/max constraints and target propensity.
  # First, construct function to rescale between 0 and 1
  normalized_logit_expr = (x - min_logit)/(max_logit - min_logit)

  # Second, construct function to rescale to target interval
  constrained_logit_expr = Parameters.TARGET_MIN_LOGIT + \
    (x*(Parameters.TARGET_MAX_LOGIT - Parameters.TARGET_MIN_LOGIT))

  rescaling_expr = constrained_logit_expr.subs(x, normalized_logit_expr)
  rescaled_logit_mean = rescaling_expr.evalf(subs={x: mean_logit})

  # Third, construct function to apply offset for targeted propensity.
  # This requires the rescaled mean found above.
  target_propensity_adjustment = Parameters.TARGET_MEAN_LOGIT - rescaled_logit_mean
  targeted_logit_expr = rescaling_expr + target_propensity_adjustment

  # Apply max/min truncation to account for adjustment shift.
  max_min_capped_targeted_logit = sp.functions.Max(
      sp.functions.Min(targeted_logit_expr, Parameters.TARGET_MAX_LOGIT),
      Parameters.TARGET_MIN_LOGIT)

  # Finally, construct the full function expression.
  treatment_logit_expr = max_min_capped_targeted_logit.subs(
      x, base_treatment_logit_expression)
  exponentiated_logit = sp.functions.exp(treatment_logit_expr)
  logistic_function = exponentiated_logit/(1 + exponentiated_logit)

  return logistic_function

""" Create outcome function"""

def generate_treatment_effect_subfunction(outcome_subfunctions, data):
  base_treatment_effect = Parameters.sample_treatment_effect()[0]

  # Sample outcome subfunctions to interact with Treatment effect.
  selected_interaction_terms, _ = select_given_probability_distribution(
      full_list=outcome_subfunctions,
      selection_probabilities=Parameters.TREATMENT_EFFECT_HETEROGENEITY)

  # Build multiplier
  treatment_effect_multiplier_expr = np.sum(selected_interaction_terms)

  # Normalize multiplier size but not location. This keeps the size
  # of the effect bounded but doesn't center the effect for different units
  # at 0.
  sampled_data = data.sample(frac=0.25)
  treatment_effect_multiplier_values = evaluate_expression(treatment_effect_multiplier_expr, sampled_data)
  multiplier_std = np.std(treatment_effect_multiplier_values)
  multiplier_mean = np.mean(treatment_effect_multiplier_values)
  normalized_treatment_effect_multiplier_expr = (treatment_effect_multiplier_expr - multiplier_mean)/multiplier_std

  treatment_effect_subfunction = base_treatment_effect*(1+normalized_treatment_effect_multiplier_expr)

  return treatment_effect_subfunction

def generate_outcome_function(outcome_subfunctions, data):
  # TODO: consider recursively applying the covariate transformation
  # to produce "deep" functions. Probability overkill.

  # Randomly initialize subfunction constants
  initialized_subfunctions = initialize_expression_constants(outcome_subfunctions)

  # Build base outcome function. Additive combination of the true covariates.
  # TODO: add non-linear activation function
  base_outcome_expression = np.sum(initialized_subfunctions)

  # Sample data to evaluate distribution.
  sampled_data = data.sample(frac=0.25)

  # Normalized outcome values to have approximate mean=0 and std=1.
  # This prevents situations where large outcome values drown out
  # the treatment effect or the treatment effect dominates small average outcomes.
  outcome_values = evaluate_expression(base_outcome_expression, sampled_data)
  outcome_mean = np.mean(outcome_values)
  outcome_std = np.std(outcome_values)

  # This is only an approximate normalization. It will shift the mean to zero
  # but the exact effect on std will depend on the distribution.
  normalized_outcome_expression = (base_outcome_expression - outcome_mean)/outcome_std

  # Create the treatment effect subfunction.
  treatment_effect_expression = generate_treatment_effect_subfunction(
      initialized_subfunctions, data)

  response_surface_expression = normalized_outcome_expression + \
    Constants.TREATMENT_EFFECT_SYMBOL*treatment_effect_expression + \
    Constants.OUTCOME_NOISE_SYMBOL

  return response_surface_expression

"""Perform data generation"""

def simulate_treatment_and_potential_outcomes(treatment_function, outcome_function, data):
  n_observations = data.shape[0]

  simulated_data = data.copy()

  propensity_scores = evaluate_expression(treatment_function, data)
  T = (np.random.uniform(size=n_observations) < propensity_scores).astype(int)

  simulated_data["T"] = T

  Y0 = evaluate_expression(
      outcome_function.subs(Constants.TREATMENT_EFFECT_SYMBOL, 0), data)
  Y1 = evaluate_expression(
      outcome_function.subs(Constants.TREATMENT_EFFECT_SYMBOL, 1), data)
  simulated_data["Y0"] = Y0
  simulated_data["Y1"] = Y1

  Y = (T*Y1) + ((1-T)*Y0)
  simulated_data["Y"] = Y

  treatment_effect = Y1 - Y0
  simulated_data["treatment_effect"] = treatment_effect

  return simulated_data

def created_simulated_data_from_random_covariates():
    N_COVARS = 20
    N_OBSERVATIONS = 1000

    # Generate random covariates and name sequentially
    X = np.random.normal(loc=0, scale=5, size=(N_OBSERVATIONS, N_COVARS))
    COVAR_NAMES = np.array([f"X{i}" for i in range(N_COVARS)])

    # Build data frame
    data, covar_symbols = build_data_frame(X, list(COVAR_NAMES))

    ############################

    potential_confounder_symbols, potential_confounder_selections = select_given_probability_distribution(
        full_list=covar_symbols,
        selection_probabilities=Parameters.POTENTIAL_CONFOUNDER_SELECTION_PROBABILITY)

    potential_confounder_symbols

    ############################

    true_outcome_covariates, true_treatment_covariates = \
      generate_aligned_treatment_and_outcome_covariates(potential_confounder_symbols)

    ############################

    treatment_function = generate_treatment_function(true_treatment_covariates, data)

    ###########################

    outcome_function = generate_outcome_function(true_outcome_covariates, data)

    ###########################

    simulated_data = simulate_treatment_and_potential_outcomes(treatment_function, outcome_function, data)

    return simulated_data
