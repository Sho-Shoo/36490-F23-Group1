import numpy as np
import cvxpy as cp
from numpy import ndarray
from classes.data_loader import DataLoader

try:
    from sklearn import r2_score
except ImportError:
    from sklearn.metrics import r2_score


class LassoQuantileModel(object):

    def __init__(self, data_loader: DataLoader, lambda_value: float):
        self.data_loader = data_loader
        self.lambda_value = lambda_value
        self.beta = np.zeros((len(self.data_loader.cols)))
        self.intercept = 0.0
        self.objective = 0.0

    def _cp_loss_fn(self, X: np.array, Y: np.array, beta: cp.Variable, intercept: cp.Variable):
        raw_predicted_y = X @ beta + intercept
        return (1.0 / X.shape[0]) * (cp.norm2(raw_predicted_y - Y)**2)

    def _objective_fn(self, X, Y, beta, intercept, lambda_value):
        return self._cp_loss_fn(X, Y, beta, intercept) + lambda_value * cp.norm1(beta)

        # train with sklearn
    def fit(self, start: int, end: int) -> None:
        df = self.data_loader.slice(start, end)
        x_train = self.data_loader.get_x(df)
        y_train = self.data_loader.get_y_quantiles(df)

        beta = cp.Variable(len(self.data_loader.cols))
        intercept = cp.Variable(1)
        problem = cp.Problem(cp.Minimize(self._objective_fn(x_train, y_train, beta, intercept, self.lambda_value)))
        problem.solve(solver=cp.SCS)

        self.beta = beta.value
        self.intercept = intercept.value
        self.objective = problem.objective

    @classmethod
    def validate(cls, data_loader: DataLoader,
                 train_start: int,
                 train_end: int,
                 validate_start: int,
                 validate_end: int,
                 lambda_values: list):
        """
        Tune hyperparameters using a validation set
        :param validate_end:
        :param validate_start:
        :param train_end:
        :param train_start:
        :param data_loader: DataLoader object
        :param lambda_values: List of lambda values to conduct grid search
        """
        validate_df = data_loader.slice(validate_start, validate_end)
        y_validate = data_loader.get_y_quantiles(validate_df)

        best_model = None
        best_lambda = None
        best_r2 = -float('inf')

        for lambda_val in lambda_values:
            model = LassoQuantileModel(data_loader, lambda_val)
            model.fit(train_start, train_end)
            y_pred = model.predict(validate_start, validate_end)

            r2 = r2_score(y_validate, y_pred)  # Calculate R-squared

            if r2 > best_r2:
                best_lambda = lambda_val
                best_r2 = r2
                best_model = model

        return best_model, best_r2, best_lambda

    def _to_quantiles(self, raw_y: np.array) -> np.array:
        max = raw_y.max()
        min = raw_y.min()
        rng = max - min
        return (raw_y - min) / rng

    def predict(self, start: int, end: int) -> ndarray:
        # Slice the data for the prediction period
        df = self.data_loader.slice(start, end)
        x_pred = self.data_loader.get_x(df)
        raw_y_pred = x_pred @ self.beta + self.intercept
        return self._to_quantiles(raw_y_pred)

    def evaluate(self, start: int, end: int) -> float:
        """
        Give evaluation metric of a trained/fitted model on a given test/validation period
        :param start: period start year
        :param end: period end year
        :return: an evaluation metric as floating number
        """
        df = self.data_loader.slice(start, end)
        y_actual = self.data_loader.get_y_quantiles(df)
        y_pred = self.predict(start, end)
        # Calculate R-squared
        r2 = r2_score(y_actual, y_pred)
        return r2
