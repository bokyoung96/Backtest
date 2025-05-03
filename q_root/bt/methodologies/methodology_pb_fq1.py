import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict

from bt.tools import Tools
from bt.loader import DataLoader
from bt.methodology import Methodology


class MethodologyPBFQ1SectorNeutral(Methodology):
    def __init__(self,
                 mkt: str = 'KOSPI200',
                 start_date: str = '20110101',
                 end_date: str = '20250331',
                 **kwargs):
        freq = kwargs.pop('freq', 'monthly')
        quantile = kwargs.pop('quantile', 10)
        quantile_position = kwargs.pop('quantile_position', [1])
        weight_type = kwargs.pop('weight_type', 'ew')
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
                      'mktcap',
                      'equity_nfq1',
                      'wics_sector_26']
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
            equity_nfq1 = self.data['equity_nfq1'].copy()
            mktcap = self.data['mktcap'].copy()
            wics_sector = self.data['wics_sector_26'].copy()

            # NOTE: PB = Equity NQ1 / Mktcap. Considering 5Q, reverse the order.
            # NOTE: 상사, 자본재 sector will be removed. Can be modified.
            factor = equity_nfq1.div(mktcap, axis=0)
            factor[wics_sector == '상사,자본재'] = np.nan
            return factor

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

    def get_z_score(self):
        const, raw_factor = self.get_pp_data()
        factor = const.mul(raw_factor)
        wics_sector = self.data['wics_sector_26'].copy()

        wics_sector = Tools.get_data_freq(df=wics_sector,
                                          freq=self.freq)
        wics_sector = Tools.get_data_align(const=wics_sector,
                                           prc=const,
                                           check_nan=False,
                                           fill_method=None)

        z_score = pd.DataFrame(index=factor.index, columns=factor.columns)

        for date in factor.index:
            date_factor = factor.loc[date]
            date_sector = wics_sector.loc[date]

            valid_mask = ~date_factor.isna() & ~date_sector.isna()
            date_factor = date_factor[valid_mask]
            date_sector = date_sector[valid_mask]

            for sector in date_sector.unique():
                if pd.isna(sector):
                    continue

                sector_stocks = date_sector[date_sector == sector].index
                sector_factors = date_factor[sector_stocks]

                if len(sector_factors) <= 2:
                    z_score.loc[date, sector_stocks] = 0
                else:
                    mean = sector_factors.mean()
                    std = sector_factors.std(ddof=0)
                    if std != 0:
                        z_score.loc[date, sector_stocks] = (
                            sector_factors - mean) / std
                    else:
                        z_score.loc[date, sector_stocks] = 0
        return z_score

    def get_quantile(self):
        factor = self.get_z_score()

        percentile_ranks = pd.DataFrame(
            index=factor.index, columns=factor.columns)

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
                            percentile = stats.percentileofscore(
                                values_array, value, kind='weak') / 100
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
    m = MethodologyPBFQ1SectorNeutral(mkt='KOSPI200',
                                      start_date='20110101',
                                      end_date='20250331',
                                      quantile=5,
                                      quantile_position=[5])
