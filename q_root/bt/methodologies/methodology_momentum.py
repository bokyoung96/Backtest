import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict

from bt.tools import Tools
from bt.loader import DataLoader
from bt.methodology import Methodology


class MethodologyMomentum(Methodology):
    def __init__(self,
                 mkt: str = 'KOSPI200',
                 start_date: str = '20110101',
                 end_date: str = '20241031',
                 **kwargs):
        freq = kwargs.pop('freq', 'monthly')
        quantile = kwargs.pop('quantile', 10)
        quantile_position = kwargs.pop('quantile_position', [1])
        weight_type = kwargs.pop('weight_type', 'mktcap_float')
        lookback = kwargs.pop('lookback', 3)
        super().__init__(mkt, start_date, end_date, **kwargs)

        self.freq = freq
        self.quantile = quantile
        self.quantile_position = quantile_position
        self.weight_type = weight_type
        self.lookback = lookback

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
        self.const = Tools.get_data_align(const=const,
                                          prc=self.data['price_adj'],
                                          check_nan=True,
                                          fill_method='ffill_bfill')

    def get_raw_factor(self):
        try:
            price_adj = self.data['price_adj'].copy()
            price_adj = price_adj.where(price_adj.notna(), np.nan).infer_objects(copy=False)

            orig_idx = price_adj.index.copy()
            month_periods = orig_idx.to_period('M')
            
            month_end_df = price_adj.groupby(month_periods).last()
            
            raw_factor_m = month_end_df.pct_change(periods=self.lookback, 
                                                   fill_method=None)
            raw_factor_m = raw_factor_m.where(np.isfinite(raw_factor_m), 
                                              np.nan).infer_objects(copy=False)
            
            raw_factor = raw_factor_m.reindex(month_periods).set_index(orig_idx)
            return raw_factor
            
        except ValueError as e:
            raise ValueError(f"Failed to create factor: {e}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise ValueError(f"Error in get_raw_factor: {e}")

    def get_pp_data(self):
        const = Tools.get_data_freq(df=Tools.get_nan(df=self.const,
                                                     val=[0]),
                                    freq=self.freq)
        raw_factor = Tools.get_data_freq(df=self.get_raw_factor(),
                                         freq=self.freq)
        
        aligned_factor = Tools.get_data_align(
            const=raw_factor,
            prc=const,
            check_nan=False,
            fill_method=None
        )
        
        try:
            Tools.validation_df_size(const, aligned_factor)
            return const, aligned_factor
        except ValueError as e:
            raise ValueError(f"Failed to match frequency: {e}")
    
    def get_quantile(self):
        const, raw_factor = self.get_pp_data()
        factor = const.mul(raw_factor)
        ranks = Tools.get_rank(df=factor, ascending=False)
        quantile = ranks.apply(lambda row: Tools.get_quantile(row=row,
                                                              q=self.quantile),
                               axis=1)
        return quantile

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

        w.dropna(axis=0, how='all', inplace=True)
        w = w.div(w.sum(axis=1), axis=0)
        return w


if __name__ == "__main__":
    m = MethodologyMomentum(mkt='KOSPI200',
                            start_date='20110101',
                            end_date='20250101',
                            quantile=5,
                            quantile_position=[1],
                            lookback=3)

