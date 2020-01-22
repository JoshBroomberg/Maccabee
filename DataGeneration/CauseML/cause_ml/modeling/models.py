from ..constants import Constants
from sklearn.linear_model import LinearRegression
import numpy as np


class CausalModel():
    def __init__(self, dataset):
        self.dataset = dataset

    def fit(self):
        raise NotImplementedError

    def estimate_ITE(self):
        raise NotImplementedError

    def estimate_ATE(self):
        raise NotImplementedError

    def estimate(self, estimand, *args, **kwargs):
        if estimand == Constants.Model.ATE_ESTIMAND:
            return self.estimate_ATE(*args, **kwargs)
        elif estimand == Constants.Model.ITE_ESTIMAND:
            return self.estimate_ITE(*args, **kwargs)
        else:
            raise Exception("Unrecognized estimand.")



class LinearRegressionCausalModel(CausalModel):
    def __init__(self, dataset):
        self.dataset = dataset
        self.model = LinearRegression()
        self.covar_and_treat_data = np.hstack([self.dataset.X, self.dataset.T])

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
        return self.model.coef_[0, -1]
