import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
from pathlib import Path

from bt.tools import Tools
from bt.loader import DataLoader
from bt.methodology import Methodology


class MethodologyPriceTrendsAbs(Methodology):
    def __init__(self,
                 mkt: str = 'KOSPI',
                 start_date: str = '20150101',
                 end_date: str = '20241031',
                 **kwargs):
        freq = kwargs.pop('freq', 'monthly')
        quantile = kwargs.pop('quantile', 10)
        quantile_position = kwargs.pop('quantile_position', [1])
        weight_type = kwargs.pop('weight_type', 'ew')
        score_threshold = kwargs.pop('score_threshold', 0.65)
        inverse_threshold = kwargs.pop('inverse_threshold', False)
        keep_empty_periods = kwargs.pop('keep_empty_periods', True)
        file_names = kwargs.pop('file_name', 'price_trends_avg_test_20.parquet')
        if isinstance(file_names, str):
            file_names = [file_names]
        
        super().__init__(mkt, start_date, end_date, **kwargs)

        self.freq = freq
        self.quantile = quantile
        self.quantile_position = quantile_position
        self.weight_type = weight_type
        self.score_threshold = score_threshold
        self.inverse_threshold = inverse_threshold
        self.keep_empty_periods = keep_empty_periods
        self.file_names = file_names
        
        self.load_data()
        self.load_const()

    def load_data(self) -> Dict[str, pd.DataFrame]:
        data_names = ['price_adj',
                      'mktcap_float']
        raw_data = Tools.get_data(mkt=self.mkt,
                                  data_names=data_names,
                                  loader_cls=DataLoader)
        self.data = {name: df[self.start_date:self.end_date]
                     for name, df in raw_data.items()}

    def load_const(self):
        const = DataLoader(
            mkt=self.mkt).data_constituents[self.start_date: self.end_date]
        
        raw_factor_for_align = self.get_raw_factor()
        if isinstance(raw_factor_for_align, list):
            raw_factor_for_align = raw_factor_for_align[0]

        self.const = Tools.get_data_align(const=const,
                                          prc=raw_factor_for_align)

    def get_raw_factor(self) -> List[pd.DataFrame]:
        try:
            raw_factors = []
            for file_name in self.file_names:
                file_path = Path(__file__).parent / file_name
                raw_factor = pd.read_parquet(file_path)
                orig_idx = self.data['price_adj'].index

                raw_factor = raw_factor.reindex(orig_idx, method='bfill')
                raw_factor.index.name = None
                raw_factor.columns.name = None

                # For KOSPI, exclude ids not in idx
                # Will automatically be excluded in KOSPI200
                exclude_ids = ['A900030', 'A900050',
                               'A900140', 'A950010', 'A950100', 'A950210']
                raw_factor = raw_factor.drop(columns=exclude_ids, errors='ignore')
                raw_factors.append(raw_factor)
            return raw_factors
        except (ValueError, RuntimeError) as e:
            raise ValueError(f"Failed to create raw weight: {e}")

    def get_pp_data(self) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
        const_df = Tools.get_nan(df=self.const, val=[0])
        raw_factors_list = self.get_raw_factor()

        all_dfs = [const_df] + raw_factors_list
        common_cols = all_dfs[0].columns
        for df in all_dfs[1:]:
            common_cols = common_cols.intersection(df.columns)

        const_aligned = const_df[common_cols]
        raw_factors_aligned = [rf[common_cols] for rf in raw_factors_list]
        
        const = Tools.get_data_freq(df=const_aligned, freq=self.freq)
        raw_factors = [Tools.get_data_freq(df=rf, freq=self.freq) 
                       for rf in raw_factors_aligned]

        try:
            for raw_factor in raw_factors:
                Tools.validation_df_size(const, raw_factor)
            return const, raw_factors
        except ValueError as e:
            raise ValueError(f"Failed to match frequency: {e}")

    def get_quantile(self):
        const, raw_factors = self.get_pp_data()
        
        if not raw_factors:
            return pd.DataFrame(index=const.index, columns=const.columns).fillna(0)

        is_in_universe = (const > 0)
        quantiles = []
        for raw_factor in raw_factors:
            if self.inverse_threshold:
                selection = (raw_factor <= self.score_threshold)
            else:
                selection = (raw_factor >= self.score_threshold)
            
            quantile = selection & is_in_universe
            quantiles.append(quantile)
            
        combined_quantile = quantiles[0]
        for i in range(1, len(quantiles)):
            aligned_q1, aligned_q2 = combined_quantile.align(
                quantiles[i], join='outer', fill_value=False)
            combined_quantile = aligned_q1 & aligned_q2
        return combined_quantile.astype(int)

    def get_quantile_weights(self):
        quantile = self.get_quantile()
        quantile_weights = quantile.apply(lambda row: Tools.get_quantile_weights(row=row,
                                                                                 nums=self.quantile_position),
                                          axis=1)
        return quantile_weights

    @property
    def weights(self):
        mktcap_float = Tools.get_data_freq(df=self.data['mktcap_float'],
                                           freq=self.freq)
        quantile_weights = self.get_quantile_weights()
        w = quantile_weights * \
            mktcap_float if self.weight_type == 'mktcap_float' else quantile_weights

        if self.keep_empty_periods:
            empty_periods = w.sum(axis=1) == 0
            w.loc[empty_periods, :] = np.nan
        else:
            w.dropna(axis=0, how='all', inplace=True)
            
        w = w.div(w.sum(axis=1), axis=0)
        return w


if __name__ == "__main__":
    file_name = ['price_trends_avg_test_5.parquet',
                 'price_trends_avg_test_20.parquet',
                 'price_trends_avg_test_60.parquet']
        
    method = MethodologyPriceTrendsAbs(mkt="KOSPI200",
                                       start_date="20200101",
                                       end_date="20250627",
                                       weight_type="ew",
                                       score_threshold=0.40,
                                       inverse_threshold=True,
                                       keep_empty_periods=True,
                                       file_name=file_name)    