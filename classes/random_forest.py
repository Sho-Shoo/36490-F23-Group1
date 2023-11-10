import numpy as np
from numpy import ndarray
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from classes.data_loader_dt import DataLoader
try:
    from sklearn import r2_score
except ImportError:
    from sklearn.metrics import r2_score

def r2oos(y, yhat):
    num = np.sum((y- yhat)**2)
    den = np.sum((y)**2)
    return 1 - num/den

class RandomForest(object):

    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.cols = ["be_me", "ret_12_1", "market_equity", "ret_1_0", "rvol_252d", "beta_252d", "qmj_safety", "rmax1_21d", "chcsho_12m",
                     "ni_me", "eq_dur", "ret_60_12", "ope_be", "gp_at", "ebit_sale", "at_gr1", "sale_gr1", "at_be", "cash_at", "age", "z_score"]

        self.beta_list = np.zeros((len(self.cols), 3))
        self.intercept_list = np.zeros(3)
        self.objective_list = np.zeros(3)
        self.sklearn_model = None

    # train with sklearn
    # def fit(self, start: int, end: int) -> None:
    #     print("fit")
    #     df = self.data_loader.slice(start, end)
    #     x_train = self.data_loader.get_x(df)
    #     y_train = self.data_loader.get_y(df)
    #     rf_model = RandomForestClassifier(
    #         n_estimators=100, min_samples_split=20, random_state=42)
    #     rf_model.fit(x_train, y_train)
    #     self.sklearn_model = rf_model

    @classmethod
    def validate(self, data_loader: DataLoader,
                 train_start: int,
                 train_end: int,
                 validate_start: int, 
                 validate_end: int):
        validate_df = data_loader.slice(validate_start, validate_end)
        train_df = data_loader.slice(train_start, train_end)
        x_validate = data_loader.get_x(validate_df)
        y_validate = data_loader.get_y(validate_df)
        x_train = data_loader.get_x(train_df)
        y_train = data_loader.get_y(train_df)

        best_r2, best_model = -1, None

        model = RandomForestRegressor()
        model.fit(x_train, y_train)
        preds = model.predict(x_validate)
        r2 = r2oos(y_validate, preds)
        if r2 > best_r2:
            best_r2 = r2
            best_model = model

        return best_model, best_r2       

    # def predict(self, start: int, end: int) -> ndarray:
    #     # Slice the data for the prediction period
    #     df = self.data_loader.slice(start, end)
    #     x_pred = self.data_loader.get_x(df)
    #     return self.sklearn_model.predict(x_pred)

    def evaluate(self, data: DataLoader, best_model, start: int, end: int) -> tuple:
        """
        Give evaluation metric of a trained/fitted model on a given test/validation period
        :param start: period start year
        :param end: period end year
        :return: an evaluation metric as floating number
        """
        monthly_r2_scores = []
        start_year, end_year = start // 10000, end // 10000
        monthly_predictions = []

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
                monthly_predictions.append(y_pred)
                r2 = r2oos(y_actual, y_pred)
                monthly_r2_scores.append(r2)

        return monthly_r2_scores, monthly_predictions

