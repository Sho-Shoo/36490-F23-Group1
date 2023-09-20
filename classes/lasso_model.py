import numpy as np
from numpy import ndarray
from sklearn import linear_model
from sklearn import r2_score
from classes.data_loader import DataLoader


class LassoModel(object):

    def __init__(self, data_loader: DataLoader, lambda_values: list):
        self.data_loader = data_loader
        self.lambda_values = lambda_values
        self.cols = ["be_me", "ret_12_1", "market_equity", "ret_1_0", "rvol_252d", "beta_252d", "qmj_safety", "rmax1_21d", "chcsho_12m",
                     "ni_me", "eq_dur", "ret_60_12", "ope_be", "gp_at", "ebit_sale", "at_gr1", "sale_gr1", "at_be", "cash_at", "age", "z_score"]
        self.beta_list = np.zeros((len(self.cols), 3))
        self.intercept_list = np.zeros(3)
        self.objective_list = np.zeros(3)

    # train with sklearn
    def fit(self, start: int, end: int) -> None:
        df = self.data_loader.slice(start, end)
        x_train = self.data_loader.get_x(df)
        y_train = self.data_loader.get_y(df)
        lambda_default = 0.0005

        model = linear_model.Lasso(alpha=lambda_default/2)
        model.fit(x_train, y_train)
        self.beta_list[:, 0] = model.coef_
        self.intercept_list[0] = model.intercept_
        self.objective_list[0] = np.linalg.norm(model.coef_, ord=1)
    
    @classmethod
    def validate(cls, start: int, end: int, lambda_values: list) -> float:
        """
        Tune hyperparameters using a validation set
        :param start: period start year for validation data
        :param end: period end year for validation data
        :param lambda_values: List of lambda values to test
        """
        df = cls.data_loader.slice(start, end)
        x_val = cls.data_loader.get_x(df)
        y_val = cls.data_loader.get_y(df)

        best_lambda = None
        best_r2 = -float('inf')

        for lambda_val in lambda_values:
            model = linear_model.Lasso(alpha=lambda_val/2)
            model.fit(x_val, y_val)
            y_pred = model.predict(x_val)
            
            r2 = r2_score(y_val, y_pred)  # Calculate R-squared
        
            if r2 > best_r2:
                best_r2 = r2
                best_lambda = lambda_val
                
        return best_lambda


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
            
