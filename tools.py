import os
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pylab as pylab

from typing import Dict, Type

# NOTE: Plotting format
params = {'figure.figsize': (30, 20),
          'axes.labelsize': 20,
          'axes.titlesize': 25,
          'xtick.labelsize': 15,
          'ytick.labelsize': 15,
          'legend.fontsize': 15
          }
pylab.rcParams.update(params)


class Tools:
    def __init__(self):
        pass

    @staticmethod
    def validation_df_size(*dfs: pd.DataFrame) -> bool:
        if len(dfs) < 2:
            raise ValueError("At least two dfs are required for comparison.")

        ref_df = dfs[0]
        for i, df in enumerate(dfs[1:], start=2):
            for attr, ref_attr, cur_attr in [
                ("shape", ref_df.shape, df.shape),
                ("cols", list(ref_df.columns), list(df.columns)),
                ("idx", list(ref_df.index), list(df.index)),
            ]:
                if ref_attr != cur_attr:
                    raise ValueError(
                        f"{attr} mismatch between df 1 ({ref_attr}) and "
                        f"df {i} ({cur_attr})"
                    )
        return True

    @staticmethod
    def get_data(loader_cls: Type,
                 mkt: str,
                 data_names: list) -> Dict[str, pd.DataFrame]:
        loader = loader_cls(mkt=mkt)
        data = {}
        for name in data_names:
            data[name] = loader(data_name=name)
        return data

    @staticmethod
    def get_data_align(const: pd.DataFrame,
                       prc: pd.DataFrame) -> pd.DataFrame:
        const = const.sort_index()
        prc = prc.sort_index()

        res = const.reindex(prc.index, method='ffill').bfill()
        res = res.reindex(columns=prc.columns)
        if res.isnull().values.any():
            raise ValueError("Unexpected NaN values found after alignment.")
        return res

    @staticmethod
    def get_data_freq(df: pd.DataFrame,
                      freq: str) -> pd.DataFrame:
        groupers = {
            'monthly': [df.index.year, df.index.month],
            'quarterly': [df.index.year, (df.index.month - 1) // 3 + 1],
            'semiannual': [df.index.year, (df.index.month - 1) // 6 + 1],
            'annual': [df.index.year],
        }

        if freq not in groupers:
            raise ValueError(f"Unsupported frequency: {freq}")
        return df.groupby(groupers[freq]).tail(1)

    def get_nan(df: pd.DataFrame,
                val: list) -> pd.DataFrame:
        return df.replace(val, np.nan)

    @staticmethod
    def get_rank(df: pd.DataFrame,
                 ascending: bool = False) -> pd.DataFrame:
        return df.rank(axis=1, method='first', na_option='keep', ascending=ascending)

    @staticmethod
    def get_quantile(row,
                     q: int = 10) -> pd.Series:
        if row.isna().all():
            return pd.Series([np.nan] * len(row), index=row.index)
        if row.nunique() <= 1:
            return pd.Series([np.nan] * len(row), index=row.index)

        try:
            quantiles = pd.qcut(row, q=q, labels=False, duplicates='drop')
            return pd.Series(quantiles, index=row.index).add(1)
        except ValueError as e:
            return pd.Series([np.nan] * len(row), index=row.index)

    @staticmethod
    def get_quantile_weights(row: pd.Series,
                             nums: list) -> pd.Series:
        return row.where(row.isin(nums), other=np.nan).apply(lambda x: 1 if not pd.isna(x) else np.nan)

    def get_dt_to_str(dt: dt.datetime, fmt: str = "%Y-%m-%d") -> str:
        return dt.strftime(fmt)

    @staticmethod
    def get_drawdown(cumret: pd.DataFrame) -> pd.DataFrame:
        peak = cumret.cummax()
        drawdown = (cumret - peak) / peak
        drawdown.replace([np.inf, -np.inf], np.nan, inplace=True)
        drawdown.fillna(0, inplace=True)
        return drawdown
