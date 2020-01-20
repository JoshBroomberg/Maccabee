from ..constants import Constants
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

class CausalModel():
    def __init__(self, dataset):
        self.dataset = dataset


        self._build_model()
        self._prepare_data()

    def _build_model(self):
        raise NotImplementedError

    def _prepare_data(self):
        pass

    def fit(self):
        raise NotImplementedError

    def estimate(self, estimand, *args, **kwargs):
        if estimand == Constants.Model.ATE_ESTIMAND:
            return self.estimate_ATE(*args, **kwargs)
        elif estimand == Constants.Model.ITE_ESTIMAND:
            return self.estimate_ITE(*args, **kwargs)
        else:
            raise Exception("Unrecognized estimand.")

    def estimate_ITE(self):
        raise NotImplementedError

    def estimate_ATE(self):
        raise NotImplementedError

class LinearRegressionCausalModel(CausalModel):
    def _build_model(self):
        self.model = LinearRegression()

    def _prepare_data(self):
        self.covar_and_treat_data = self.dataset.X.join(self.dataset.T)

    def fit(self):
        self.model.fit(self.covar_and_treat_data, self.dataset.Y)

    def estimate_ITE(self):
        # Generate potential outcomes
        X_under_treatment = self.covar_and_treat_data.copy()
        X_under_treatment["T"] = 1

        X_under_control = self.covar_and_treat_data.copy()
        X_under_control["T"] = 0

        y_1_predicted = self.model.predict(X_under_treatment)
        y_0_predicted = self.model.predict(X_under_control)

        ITE = y_1_predicted - y_0_predicted

        return ITE

    def estimate_ATE(self):
        # The coefficient on the treatment status
        return self.model.coef_[-1]
