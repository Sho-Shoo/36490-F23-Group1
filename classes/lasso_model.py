from classes.data_loader import DataLoader
from numpy import ndarray


class LassoModel(object):

    def __init__(self, data: DataLoader, lambda_value: float):
        raise NotImplementedError
        # TODO: Lucy

    def fit(self, start: int, end: int) -> None:
        raise NotImplementedError
        # TODO: Lucy

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
