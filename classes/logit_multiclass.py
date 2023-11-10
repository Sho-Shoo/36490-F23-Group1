import numpy as np
from numpy import ndarray
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from classes.data_loader_dt import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class Logit(object):

    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.cols = ["be_me", "ret_12_1", "market_equity", "ret_1_0", "rvol_252d", "beta_252d", "qmj_safety", "rmax1_21d", "chcsho_12m",
                     "ni_me", "eq_dur", "ret_60_12", "ope_be", "gp_at", "ebit_sale", "at_gr1", "sale_gr1", "at_be", "cash_at", "age", "z_score"]

        self.sklearn_model = None

    # train with sklearn
    # todo: havn't add any penalty to our model 
    def fit(self, start: int, end: int) -> None:
        print("fit")
        df = self.data_loader.slice(start, end)
        x_train = self.data_loader.get_x(df)
        y_train = self.data_loader.get_y(df)
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        logit_model = LogisticRegression(solver='lbfgs')
        logit_model.fit(x_train,y_train)
        self.sklearn_model = logit_model

    @classmethod
    def validate(self, data_loader: DataLoader,
                 train_start: int, train_end: int,
                 validate_start: int, validate_end: int,
                 alpha_values: list):
        validate_df = data_loader.slice(validate_start, validate_end)
        train_df = data_loader.slice(train_start, train_end)
        x_validate = data_loader.get_x(validate_df)
        y_validate = data_loader.get_y(validate_df)
        x_train = data_loader.get_x(train_df)
        y_train = data_loader.get_y(train_df)

        best_accuracy, best_model, best_alpha = -1, None, None
        for alpha in alpha_values:
            model= LogisticRegression(multi_class='multinomial', 
                                      solver='saga',
                                      C = alpha,
                                      n_jobs=-1,
                                      penalty='l1'
                                      )
            model.fit(x_train, y_train)
            prediction = model.predict(x_validate)
            accuracy = accuracy_score(y_validate, prediction)
            if accuracy >  best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_alpha = alpha
            return best_model, best_accuracy, best_alpha

    def predict(self, start: int, end: int) -> ndarray:
        # Slice the data for the prediction period
        df = self.data_loader.slice(start, end)
        x_pred = self.data_loader.get_x(df)
        sc = StandardScaler()
        x_pred = sc.fit_transform(x_pred)
        return self.sklearn_model.predict(x_pred)

    @classmethod
    def evaluate(self, data_loader: DataLoader, best_model,  start: int, end: int) -> float:
        """
        Give evaluation metric of a trained/fitted model on a given test/validation period
        :param start: period start year
        :param end: period end year
        :return: an evaluation metric as floating number
        """
        df = data_loader.slice(start, end)
        x_test = data_loader.get_x(df)
        y_test  = data_loader.get_y(df)
        y_pred = best_model.predict(x_test)
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        size = data_loader.get_size(df)
        return accuracy,y_test,y_pred ,size
    
    


