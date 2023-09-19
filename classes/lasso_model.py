import numpy as np
from numpy import ndarray
from sklearn import linear_model
from classes.data_loader import DataLoader


class LassoModel(object):

    def __init__(self, data_loader: DataLoader, lambda_val: float):
        self.data_loader = data_loader
        self.lambda_val = lambda_val
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

        model = linear_model.Lasso(alpha=self.lambda_val/2)
        model.fit(x_train, y_train)
        self.beta_list[:, 0] = model.coef_
        self.intercept_list[0] = model.intercept_
        self.objective_list[0] = np.linalg.norm(model.coef_, ord=1)

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
        raise NotImplementedError
        # TODO: Flora
