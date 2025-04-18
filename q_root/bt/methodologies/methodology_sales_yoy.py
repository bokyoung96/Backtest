import pandas as pd
from typing import Dict

from bt.tools import Tools
from bt.loader import DataLoader
from bt.methodology import Methodology


class MethodologySalesYoy(Methodology):
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

    def load_data(self) -> Dict[str, pd.DataFrame]:
        data_names = ['price_adj',
                      'mktcap_float',
                      'sales_ttm_nfq1']
        raw_data = Tools.get_data(mkt=self.mkt,
                                  data_names=data_names,
                                  loader_cls=DataLoader)
        self.data = {name: df[self.start_date:self.end_date]
                     for name, df in raw_data.items()}

    def load_const(self):
        const = DataLoader(
            mkt=self.mkt).data_constituents[self.start_date: self.end_date]
        self.const = Tools.get_data_align(const=const,
                                          prc=self.data['price_adj'])

    @property
    def sales_yoy_qtrly(self):
        try:
            sales_data = self.data['sales_ttm_nfq1']
            sales_qtrly = sales_data.resample('QE').last()
            sales_yoy_qtrly = sales_qtrly.pct_change(periods=4,
                                                     fill_method=None)
            return sales_yoy_qtrly
        except KeyError:
            raise ValueError("Required data 'sales_ttm_nfq1' not found in self.data.")
        except Exception as e:
            raise RuntimeError(f"Error calculating sales YoY (Quarterly): {e}")

    @property
    def sales_yoy_3yr_avg(self):
        try:
            sales_yoy_qtrly = self.sales_yoy_qtrly
            if sales_yoy_qtrly is None or sales_yoy_qtrly.empty:
                return pd.DataFrame(index=sales_yoy_qtrly.index, 
                                    columns=sales_yoy_qtrly.columns)

            res = pd.DataFrame(index=sales_yoy_qtrly.index, 
                               columns=sales_yoy_qtrly.columns, 
                               dtype=float)
            for q in range(1, 5):
                qtr_mask = sales_yoy_qtrly.index.quarter == q
                qtr_data = sales_yoy_qtrly.loc[qtr_mask]

                if not qtr_data.empty:
                    qtr_avg = qtr_data.rolling(window=3, min_periods=3).mean()
                    res.loc[qtr_mask] = qtr_avg
            return res
        except Exception as e:
            raise RuntimeError(f"Error calculating quarter-specific 3yr average sales YoY: {e}")

    def get_raw_factor(self):
        try:
            factor_qtrly = self.sales_yoy_3yr_avg
            orig_idx = self.data['price_adj'].index
            
            factor = factor_qtrly.reindex(orig_idx, method='bfill')
            return factor
        except (ValueError, RuntimeError) as e:
            raise ValueError(f"Failed to create factor: {e}")

    def get_pp_data(self):
        const = Tools.get_data_freq(df=Tools.get_nan(df=self.const,
                                                     val=[0]),
                                    freq=self.freq)
        raw_factor = Tools.get_data_freq(df=self.get_raw_factor(),
                                         freq=self.freq)

        try:
            Tools.validation_df_size(const,
                                     raw_factor)
            return const, raw_factor
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
    method = MethodologySalesYoy(mkt="KOSPI200",
                                 start_date="20110101",
                                 end_date="20241031")
