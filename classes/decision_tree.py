import numpy as np
from numpy import ndarray
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from classes.data_loader_dt import DataLoader


class DecisionTree(object):

    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.cols = ["be_me", "ret_12_1", "market_equity", "ret_1_0", "rvol_252d", "beta_252d", "qmj_safety", "rmax1_21d", "chcsho_12m",
                     "ni_me", "eq_dur", "ret_60_12", "ope_be", "gp_at", "ebit_sale", "at_gr1", "sale_gr1", "at_be", "cash_at", "age", "z_score"]

        self.sklearn_model = None

    # train with sklearn
    def fit(self, start: int, end: int) -> None:
        print("fit")
        df = self.data_loader.slice(start, end)
        x_train = self.data_loader.get_x(df)
        y_train = self.data_loader.get_y(df)
        dt_model = DecisionTreeClassifier(min_samples_split=20,random_state=42)
        #params = {
        #  'min_samples_leaf':[30,50],
        #  'min_samples_split':[20,30]}
        #GS = GridSearchCV(estimator=dt_model,param_grid=params,cv=3,n_jobs=-1,scoring='accuracy')
        dt_model.fit(x_train, y_train)
        self.sklearn_model = dt_model

    @classmethod
    def validate(cls, data_loader: DataLoader,
                 train_start: int,
                 train_end: int,
                 test_start: int,
                 test_end: int):

        model = DecisionTree(data_loader)
        model.fit(train_start,train_end)
        train_accuracy = model.evaluate(train_start, train_end)
        test_accuracy = model.evaluate(test_start, test_end)
        return train_accuracy, test_accuracy

    def predict(self, start: int, end: int) -> ndarray:
        # Slice the data for the prediction period
        df = self.data_loader.slice(start, end)
        x_pred = self.data_loader.get_x(df)
        return self.sklearn_model.predict(x_pred)

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

        # Calculate accuracy
        accuracy = accuracy_score(y_actual, y_pred)

        return accuracy
    
    


