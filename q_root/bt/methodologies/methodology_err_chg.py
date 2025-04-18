import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict

from bt.tools import Tools
from bt.loader import DataLoader
from bt.methodology import Methodology


class MethodologyERRChg(Methodology):
    def __init__(self,
                 mkt: str = 'KOSPI200',
                 start_date: str = '20110101',
                 end_date: str = '20241031',
                 **kwargs):
        freq = kwargs.pop('freq', 'monthly')
        quantile = kwargs.pop('quantile', 10)
        quantile_position = kwargs.pop('quantile_position', [1])
        weight_type = kwargs.pop('weight_type', 'mktcap_float')
        super().__init__(mkt, start_date, end_date, **kwargs)

        self.freq = freq
        self.quantile = quantile
        self.quantile_position = quantile_position
        self.weight_type = weight_type

        self.load_data()
        self.load_const()
        self.load_sector()

    def load_data(self) -> Dict[str, pd.DataFrame]:
        data_names = ['price_adj',
                      'mktcap_float',
                      'err_1m',
                      'err_2m',
                      'wics_sector_big']
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
        
    def load_sector(self):
        self.sector = Tools.get_data_align(const=self.data['wics_sector_big'],
                                          prc=self.data['price_adj'],
                                          check_nan=False,
                                          fill_method='ffill_bfill')

    def get_raw_factor(self):
        try:
            Tools.validation_df_size(self.data['err_1m'],
                                     self.data['err_2m'])
            
            err_1m = self.data['err_1m'].copy()
            err_2m = self.data['err_2m'].copy()
            
            raw_factor = pd.DataFrame(index=err_1m.index, columns=err_1m.columns)
            raw_factor = (err_1m + err_2m) / 2
            
            mask_nan = err_1m.isna() | err_2m.isna()
            raw_factor[mask_nan] = np.nan
            
            orig_idx = raw_factor.index.copy()            
            m_periods = pd.PeriodIndex(orig_idx, freq='M')
            m_data = {}
            for month in set(m_periods):
                m_dates = orig_idx[m_periods == month]
                last_date = m_dates[-1]
                m_data[month] = raw_factor.loc[last_date].copy()
            
            shifted_data = {}
            for month in sorted(m_data.keys()):
                prev_month = month - 1
                if prev_month in m_data:
                    shifted_data[month] = m_data[prev_month].copy()
            
            lagged_factor = pd.DataFrame(index=orig_idx, columns=raw_factor.columns)            
            for date, period in zip(orig_idx, m_periods):
                if period in shifted_data:
                    lagged_factor.loc[date] = shifted_data[period]
            return lagged_factor
            
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
        
        percentile_ranks = pd.DataFrame(index=factor.index, columns=factor.columns)
        
        for date in factor.index:
            row_data = factor.loc[date]
            valid_data = row_data.dropna()
            
            if not valid_data.empty:
                values_array = valid_data.values
                
                for col in valid_data.index:
                    try:
                        value = valid_data[col]
                        if len(values_array) <= 1:
                            percentile_ranks.loc[date, col] = 0.5
                        else:
                            percentile = stats.percentileofscore(values_array, value, kind='weak') / 100
                            percentile_ranks.loc[date, col] = percentile
                    except Exception:
                        percentile_ranks.loc[date, col] = np.nan
        
        quantile = percentile_ranks.apply(lambda row: Tools.get_quantile(row=row, 
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
    m = MethodologyERRChg(mkt='KOSPI200',
                          start_date='20110101',
                          end_date='20250101',
                          quantile=4)

