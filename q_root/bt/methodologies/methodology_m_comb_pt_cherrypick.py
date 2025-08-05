import numpy as np
import pandas as pd
from typing import Dict
from functools import cached_property

from bt.tools import Tools
from bt.loader import DataLoader
from bt.methodology import Methodology
from bt.methodologies.methodology_momentum_comb import MethodologyMomentumComb
from bt.methodologies.methodology_price_trends import MethodologyPriceTrends


class MethodologyMCombPTCherrypick(Methodology):
    def __init__(self,
                 mkt: str = 'KOSPI200',
                 start_date: str = '20110101',
                 end_date: str = '20241031',
                 **kwargs):
        freq = kwargs.pop('freq', 'monthly')
        quantile = kwargs.pop('quantile', 10)
        quantile_position = kwargs.pop('quantile_position', [1])
        weight_type = kwargs.pop('weight_type', 'mktcap_float')
        lookback = kwargs.pop('lookback', [3, 6, 1])
        lookback_weights = kwargs.pop('lookback_weights', [2, 1, -1])
        top_pct = kwargs.pop('top_pct', 0.1)
        super().__init__(mkt, start_date, end_date, **kwargs)

        self.freq = freq
        self.quantile = quantile
        self.quantile_position = quantile_position
        self.weight_type = weight_type
        self.lookback = lookback
        self.lookback_weights = lookback_weights
        self.top_pct = top_pct

        self.load_data()
        self.load_const()

        self._momentum_methodology = None
        self._price_trends_methodology = None

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
    
    def get_momentum_methodology(self) -> MethodologyMomentumComb:
        if self._momentum_methodology is None:
            self._momentum_methodology = MethodologyMomentumComb(
                mkt=self.mkt,
                start_date=self.start_date,
                end_date=self.end_date,
                freq=self.freq,
                quantile=self.quantile,
                quantile_position=self.quantile_position,
                weight_type=self.weight_type,
                lookback=self.lookback,
                lookback_weights=self.lookback_weights
            )
        return self._momentum_methodology

    def get_price_trends_methodology(self) -> MethodologyPriceTrends:
        if self._price_trends_methodology is None:
            self._price_trends_methodology = MethodologyPriceTrends(
                mkt=self.mkt,
                start_date=self.start_date,
                end_date=self.end_date,
                freq=self.freq,
                quantile=self.quantile,
                quantile_position=self.quantile_position,
                weight_type=self.weight_type
            )
        return self._price_trends_methodology

    def get_momentum_quantile(self) -> pd.DataFrame:
        momentum_method = self.get_momentum_methodology()
        return momentum_method.get_quantile_weights()

    def get_price_trends_factor(self) -> pd.DataFrame:
        price_trends_method = self.get_price_trends_methodology()
        return price_trends_method.get_pp_data()[1]

    @cached_property
    def get_raw_factors(self) -> Dict[str, pd.DataFrame]:
        def _combine(df1: pd.DataFrame, df2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
            idx = df1.index.intersection(df2.index)
            cols = df1.columns.intersection(df2.columns)
            return (df1.loc[idx, cols].sort_index(axis=0).sort_index(axis=1),
                    df2.loc[idx, cols].sort_index(axis=0).sort_index(axis=1))
        
        mom, price_trends = _combine(self.get_momentum_quantile(), self.get_price_trends_factor())
        return {'momentum': mom, 'price_trends': price_trends}
    
    def get_factors(self) -> pd.DataFrame:
        # NOTE: Get stocks with top momentum, cherrypick top price trends
        raw_factors = self.get_raw_factors
        factors = raw_factors['momentum'].mul(raw_factors['price_trends'], axis=0)
        
        def select_top_pct(row):
            n_select = int(np.ceil(row.notna().sum() * self.top_pct))
            if n_select == 0:
                return row * np.nan
            ranks = row.rank(method='min', ascending=False)
            return row.where(ranks <= n_select)
        return factors.apply(select_top_pct, axis=1)
    
    def get_quantile_weights(self) -> pd.DataFrame:
        factors = self.get_factors()
        return factors.where(factors.isna(), 1)

    @property
    def weights(self):
        mktcap_float = Tools.get_data_freq(df=self.data['mktcap_float'],
                                           freq=self.freq)
        if self.weight_type == 'score':
            quantile_weights = self.get_factors()
        else:
            quantile_weights = self.get_quantile_weights()
        w = quantile_weights * \
            mktcap_float if self.weight_type == 'mktcap_float' else quantile_weights

        w.dropna(axis=0, how='all', inplace=True)
        w = w.div(w.sum(axis=1), axis=0)
        return w


if __name__ == "__main__":
    m = MethodologyMCombPTCherrypick(mkt='KOSPI200',
                                 start_date='20110101',
                                 end_date='20250627',
                                 freq='monthly',
                                 quantile=4,
                                 quantile_position=[1],
                                 weight_type='ew',
                                 lookback=[3, 6, 1],
                                 lookback_weights=[2, 1, -1],
                                 top_pct=0.1)