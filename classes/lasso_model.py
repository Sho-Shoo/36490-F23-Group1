import numpy as np
from numpy import ndarray
from sklearn import linear_model
try:
    from sklearn import r2_score
except ImportError:
    from sklearn.metrics import r2_score
from classes.data_loader import DataLoader


class LassoModel(object):

    def __init__(self, data_loader: DataLoader, lambda_value: float):
        self.data_loader = data_loader
        self.lambda_value = lambda_value
        self.cols = ["be_me", "ret_12_1", "market_equity", "ret_1_0", "rvol_252d", "beta_252d", "qmj_safety", "rmax1_21d", "chcsho_12m",
                     "ni_me", "eq_dur", "ret_60_12", "ope_be", "gp_at", "ebit_sale", "at_gr1", "sale_gr1", "at_be", "cash_at", "age", "z_score"]
        self.beta_list = np.zeros((len(self.cols), 3))
        self.intercept_list = np.zeros(3)
        self.objective_list = np.zeros(3)
        self.sklearn_model = None

    # train with sklearn
    def fit(self, start: int, end: int) -> None:
        df = self.data_loader.slice(start, end)
        x_train = self.data_loader.get_x(df)
        y_train = self.data_loader.get_y(df)

        model = linear_model.Lasso(alpha=self.lambda_value/2)
        model.fit(x_train, y_train)
        self.sklearn_model = model
        self.beta_list[:, 0] = model.coef_
        self.intercept_list[0] = model.intercept_
        self.objective_list[0] = np.linalg.norm(model.coef_, ord=1)
    
    @classmethod
    def validate(cls, data_loader: DataLoader,
                 train_start: int,
                 train_end: int,
                 validate_start: int,
                 validate_end: int,
                 lambda_values: list) -> float:
        """
        Tune hyperparameters using a validation set
        :param validate_end:
        :param validate_start:
        :param train_end:
        :param train_start:
        :param data_loader:
        :param lambda_values: List of lambda values to test
        """
        validate_df = data_loader.slice(validate_start, validate_end)
        x_validate = data_loader.get_x(validate_df)
        y_validate = data_loader.get_y(validate_df)

        best_model = None
        best_r2 = -float('inf')

        for lambda_val in lambda_values:
            model = LassoModel(data_loader, lambda_val)
            model.fit(train_start, train_end)
            y_pred = model.predict(x_validate)
            
            r2 = r2_score(y_validate, y_pred)  # Calculate R-squared
        
            if r2 > best_r2:
                best_r2 = r2
                best_model = best_model
                
        return best_model


    def predict(self, start: int, end: int) -> ndarray:
        raise NotImplementedError
        # TODO: Mia

    def evaluate(self, start: int, end: int) -> float:
        """
        Give evaluation metric of a trained/fitted model on a given test/validation period
        :param start: period start year
        :param end: period end year
        :return: an evaluation metric as floating number
        """
        df = self.data_loader.slice(start, end)
        y_actual = self.data_loader.get_y(df)

        y_pred = self.predict(start, end)

        # Calculate R-squared
        r2 = r2_score(y_actual, y_pred)

        return r2
            
