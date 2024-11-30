import re
import numpy as np
import pandas as pd


from tqdm import tqdm


class PreProcess:
    def __init__(self,
                 file_name: str,
                 **kwargs):
        self.file_name = file_name

        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_QW_res_data(self) -> pd.DataFrame:
        data = pd.read_excel(f"{self.file_name}.xlsx",
                             index_col=0,
                             header=7).iloc[6:, :]
        data.index = pd.to_datetime(data.index)
        data.index.name = 'Date'
        return data

    def get_DG_res_data(self) -> pd.DataFrame:
        data = pd.read_excel(f"{self.file_name}.xlsx",
                             index_col=0,
                             header=9).iloc[4:, :]
        data.index = pd.to_datetime(data.index)
        data.index.name = 'Date'
        return data

    def get_QW_bm_data(self) -> pd.DataFrame:
        data = self.get_QW_res_data()
        data.columns = ['KOSPI', 'KOSPI_TR',
                        'KOSPI200', 'KOSPI200_TR',
                        'KOSDAQ', 'KOSDAQ_TR',
                        'KODSAQ150', 'KOSDAQ150_TR']
        return data

    def save_data(self, method=None):
        if method is None:
            method = 'get_QW_res_data'

        try:
            if hasattr(self, method):
                data_method = getattr(self, method)
                data = data_method()
                data.to_pickle(f"{self.file_name}.pkl")
        except Exception as e:
            print(f"Error: {e}")
        return data
