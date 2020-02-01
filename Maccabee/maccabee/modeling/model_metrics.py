"""This module contains the model metrics used to determine model performance based on a sample of estimand values and true values."""

import numpy as np


def absolute_mean_error(ate_estimate_vals, ate_true_vals):
    non_zeros = np.logical_not(np.isclose(ate_true_vals, 0))
    return 100*np.abs(
        np.mean((ate_estimate_vals[non_zeros] - ate_true_vals[non_zeros]) /
            ate_true_vals[non_zeros]))

def RMSE(ate_estimate_vals, ate_true_vals):
    return np.sqrt(
        np.mean((ate_estimate_vals - ate_true_vals)**2))

ATE_ACCURACY_METRICS = {
    "absolute mean bias %": absolute_mean_error,
    "root mean squared error": RMSE
}

def PEHE(ite_estimated_vals, ite_true_vals):
    return np.mean(
        np.sqrt(
            np.mean((ite_estimated_vals - ite_true_vals)**2, axis=1)))

ITE_ACCURACY_METRICS = {
    "precision in estimation of heterogenous treatment effects": PEHE
}
