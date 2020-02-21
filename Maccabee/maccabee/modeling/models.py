"""This submodule contains the :class:`maccabee.modeling.models.CausalModel` class which is the base class that defines the interface Maccabee expects from all models. IE, all models benchmarked using Maccabee should inherit from this class.

The submodule also contains some example concrete implementations. These serve as an implementation guide for user-defined models and can be used as a baseline for custom model performance.
"""

from ..constants import Constants
from ..exceptions import UnknownEstimandException
from sklearn.linear_model import LinearRegression
import numpy as np


import importlib
rpy2_spec = importlib.util.find_spec("rpy2")
if rpy2_spec is not None:
    # RPY2 is used an interconnect between Python and R. It allows
    # python to run R code in a subprocess.
    import rpy2
    from rpy2.robjects import IntVector, FloatVector, Formula
    from rpy2.robjects.packages import importr
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
    numpy2ri.activate()

class CausalModel():
    """The base :class:`maccabee.modeling.models.CausalModel` class presents a minimal interface. This is important because many models, with diverse operation/characteristics, are expected to conform to this interface. It takes a :class:`~maccabee.data_generation.generated_data_set.GeneratedDataSet` instance which contains the data to be used for estimation. It has an abstract :meth:`~maccabee.modeling.models.CausalModel.fit` method which, when called on inheriting classes, should prepare the model to produce an estimate. This preparation could mean pre-processing data, training a neural network etc. Finally, it has a concrete :meth:`~maccabee.modeling.models.CausalModel.estimate` method which expects to find a defined method with the ``estimate_*`` where \* is an estimand name. It is up to the inheriting class to define the appropriate estimator methods depending on the estimands which will be evaluated.

    Args:
        dataset (:class:`~maccabee.data_generation.generated_data_set.GeneratedDataSet`): A :class:`~maccabee.data_generation.generated_data_set.GeneratedDataSet` instance produced by a :class:`~maccabee.data_generation.data_generating_process.DataGeneratingProcess`.

    Attributes
        dataset: the data set supplied at initialization time.
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def fit(self):
        """The fit method is expected to be called once per model instance. It prepares the model for estimating any of the implemented estimands.

        Returns:
            None: the fit function does not have any meaningful return value.

        Raises:
            NotImplementedError: this is an abstract implementation.
        """
        raise NotImplementedError

    def estimate_ITE(self):
        raise NotImplementedError

    def estimate_ATE(self):
        raise NotImplementedError

    def estimate_ATT(self):
        raise NotImplementedError

    def estimate(self, estimand, *args, **kwargs):
        """The estimate method is called to access an estimand value. It is a convenience method which gives external classes access to any of the (different) estimand functions using a single parameterized function. It does not handle the calculation of the estimand itself but rather delegates to an ``estimate_*`` method on the instance where \* is the name of the estimand.

        Args:
            estimand (string): The name of the estimand to return. This should be one of the estimands in the constants list :data:`~maccabee.constants.Constants.Model.ALL_ESTIMANDS`.
            *args (list): All position args are collected and passed to the estimand function.
            **kwargs (dict): All keyword args are collected and passed to the estimand function.

        Returns:
            float: The value of the estimand.

        Raises:
            UnknownEstimandException: if an unknown estimand is requested.
        """
        if estimand not in Constants.Model.ALL_ESTIMANDS:
            raise UnknownEstimandException()

        estimate_method_name = f"estimate_{estimand}"

        if not hasattr(self, estimate_method_name):
            raise NotImplementedError

        return getattr(self, estimate_method_name)(*args, **kwargs)

class CausalModelR(CausalModel):
    """
    This class inherits from :class:`maccabee.modeling.models.CausalModel` and implements additional tooling using to write causal models with major components in R.
    """

    def __init__(self, dataset):
        super().__init__(dataset)

    def _import_r_package(self, package_name):
        """Helper function to import a package pre-installed in the system's R language.

        Args:
            package_name (str): The string name of the package, as would be used in the R `load` command.

        Returns:
            object: A python object representing the R package with all functions as attribute methods of the object.
        """
        return importr(package_name)

    def _import_r_file_as_package(self, file_path, package_name):
        """Helper function to import an R file as a psuedo-package. The functions from the R file are translated as exported methods of a package called `package_name`.

        Args:
            file_path (str): The path to the R file to import.
            package_name (str): The name to be used for the psuedo-package.

        Returns:
            object: A python object representing the R package with all functions as attribute methods of the object.
        """
        with open(file_path, "r") as prog:
            R_prog = ''.join(prog.readlines())
        return SignatureTranslatedAnonymousPackage(R_prog, package_name)

class LinearRegressionCausalModel(CausalModel):
    """This class inherits from :class:`maccabee.modeling.models.CausalModel` and implements a linear-regression based estimator for the ATE and ITE using SciKit Learn linear regression model. The ITE is a dummy estimand in this case given the linear model assumes a homogenous effect amongst all units.
    """

    def __init__(self, dataset):
        super().__init__(dataset)
        self.model = LinearRegression(fit_intercept=True)
        self.data = dataset.observed_data.drop("Y", axis=1)

    def fit(self):
        """Fit the linear regression model.
        """
        # self.model.fit(np.hstack([self.dataset.X, self.dataset.T.to_numpy().reshape((-1, 1))]), self.dataset.Y)
        self.model.fit(self.data, self.dataset.Y)

    def estimate_ATE(self):
        """
        Return the co-efficient on the treatment status variable as the
        ATE.
        """
        # The coefficient on the treatment status
        return self.model.coef_[-1]

    def estimate_ITE(self):
        """
        Return the difference between the model's predicted value with treatment set
        to 1 and 0 as the ITE. This will be a constant equal to the ATE given
        that this is a linear model.
        """
        # Generate potential outcomes
        X_under_treatment = self.data.copy()
        X_under_treatment["T"] = 1

        X_under_control = self.data.copy()
        X_under_control["T"] = 0

        y_1_predicted = self.model.predict(X_under_treatment)
        y_0_predicted = self.model.predict(X_under_control)

        ITE = y_1_predicted - y_0_predicted

        return ITE
