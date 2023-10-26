import numpy as np
from numpy import ndarray
try:
    from sklearn import r2_score
except ImportError:
    from sklearn.metrics import r2_score
from classes.data_loader import DataLoader
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV


class ElasticNet_Model(object):

    def __init__(self, data_loader: DataLoader, alpha: float = 1.0, l1_ratio: float = 0.5):
        self.data_loader = data_loader
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.cols = ["be_me", "ret_12_1", "market_equity", "ret_1_0", "rvol_252d", "beta_252d", "qmj_safety", "rmax1_21d", "chcsho_12m",
                     "ni_me", "eq_dur", "ret_60_12", "ope_be", "gp_at", "ebit_sale", "at_gr1", "sale_gr1", "at_be", "cash_at", "age", "z_score"]
        self.beta_list = np.zeros((len(self.cols), 3))
        self.intercept_list = np.zeros(3)
        self.objective_list = np.zeros(3)
        self.sklearn_model = None

    # # train with sklearn
    # def fit(self, start: int, end: int) -> None:
    #     df = self.data_loader.slice(start, end)
    #     x_train = self.data_loader.get_x(df)
    #     y_train = self.data_loader.get_y(df)

    #     model = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio)
    #     model.fit(x_train, y_train)
    #     self.sklearn_model = model
    #     self.beta_list[:, 0] = model.coef_
    #     self.intercept_list[0] = model.intercept_
    #     self.objective_list[0] = np.linalg.norm(model.coef_, ord=1)
    @classmethod
    def validate(self, data_loader: DataLoader,
                 train_start: int,
                 validate_end: int,
                 alpha_values: list,
                 l1_ratio_values: list):
        """
        Perform Grid search on hyperparameters alpha and l1_ration using 5-fold cross validation 
        on training and validation set, using r2 score as the metric
        :param data_loader: DataLoader object
        :param train_start:
        :param validate_end:
        :param alpha_values: List of alpha values to conduct grid search
        :param l1_ratio_values: List of l1 ratio values to conduct grid search
        """
        validate_df = data_loader.slice(train_start, validate_end)
        x_validate = data_loader.get_x(validate_df)
        y_validate = data_loader.get_y(validate_df)

        param_grid = {'alpha': alpha_values, 'l1_ratio': l1_ratio_values}

        # Perform Grid Search
        grid_search = GridSearchCV(ElasticNet(), param_grid, scoring='r2', cv=5)
        grid_search.fit(x_validate, y_validate)

        best_alpha = grid_search.best_params_['alpha']
        best_l1_ratio = grid_search.best_params_['l1_ratio']
        self.sklearn_model = grid_search.best_estimator_

        return grid_search.best_estimator_, grid_search.best_score_, best_alpha, best_l1_ratio


    # def predict(self, start: int, end: int) -> ndarray:
    #     # Slice the data for the prediction period
    #     df = self.data_loader.slice(start, end)
    #     x_pred = self.data_loader.get_x(df)

    #     return self.sklearn_model.predict(x_pred)

    @staticmethod
    def evaluate(data: DataLoader, best_model, start: int, end: int) -> list[float]:
        """
        Give evaluation metric of a trained/fitted model on a given test/validation period
        :param data: Data loader
        :param start: period start year
        :param end: period end year
        :return: an evaluation metric as a list of floating numbers
        """
        monthly_r2_scores = []
        start_year, end_year = start // 10000, end // 10000

        for year in range(start_year, end_year):
            for month in range(1, 13): 
                start = int(f"{year}{month:02d}01")
                if month == 12:
                    end = int(f"{year + 1}0101") 
                else:
                    end = int(f"{year}{month + 1:02d}01")

                df = data.slice(start, end)
                x_test = data.get_x(df)
                y_actual = data.get_y(df)

                y_pred = best_model.predict(x_test)

                # Calculate R-squared for the month
                r2 = r2_score(y_actual, y_pred)
                monthly_r2_scores.append(r2)

        return monthly_r2_scores
