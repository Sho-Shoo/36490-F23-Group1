from pandas import DataFrame


class DataLoader(object):

    def __init__(self, csv_file_path: str):
        raise NotImplementedError
        # TODO: Steven

    def slice(self, start: int, end: int) -> DataFrame:
        raise NotImplementedError
        # TODO: Steven
