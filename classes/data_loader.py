from pandas import DataFrame
import pandas as pd


class DataLoader(object):

    def __init__(self, csv_file_path: str):
        self.data: DataFrame = pd.read_csv(csv_file_path)

    def slice(self, start: int, end: int) -> DataFrame:
        """
        Slice a dataload to look for data within the certain time period
        :param start: slice start, inclusive. In form of YYYYMMDD (19900000)
        :param end: slice end, exclusive.
        :return: a pandas dataframe after slicing
        """
        data = self.data[(self.data["date"] >= start) & (self.data["date"] < end)]
        data = data.dropna(subset=['me', 'ret_exc_lead1m'])
        # exclude nano caps
        data = data.loc[data['size_grp'] != 'nano']
        # delete observation with more than 5 out of the 21 characteristics missing
        cols = ["be_me", "ret_12_1", "market_equity", "ret_1_0", "rvol_252d", "beta_252d", "qmj_safety", "rmax1_21d",
                "chcsho_12m", "ni_me", "eq_dur", "ret_60_12", "ope_be", "gp_at", "ebit_sale", "at_gr1", "sale_gr1",
                "at_be", "cash_at", "age", "z_score"]

        data["missing_num"] = data[cols].isna().sum(1)
        data = data.loc[data['missing_num'] <= 5]

        # impute the missing characteristics by replacing them with the cross-sectional median
        for i in cols:
            data[i] = data[i].astype(float)
            data[i] = data[i].fillna(data.groupby('date')[i].transform('median'))

        cols1 = ["permno", "date", "ret_exc_lead1m", "be_me", "ret_12_1", "market_equity", "ret_1_0", "rvol_252d",
                 "beta_252d", "qmj_safety", "rmax1_21d", "chcsho_12m", "ni_me", "eq_dur", "ret_60_12", "ope_be",
                 "gp_at", "ebit_sale", "at_gr1", "sale_gr1", "at_be", "cash_at", "age", "z_score"]

        data = data[cols1]
        data = data.dropna()

        # rank transformation
        # each characteristics is transformed into the cross-sectional rank
        for i in cols:
            data[i] = data.groupby("date")[i].rank(pct=True)

        data.sort_values(by=['date', 'permno'], inplace=True)

        return data
