import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
from pathlib import Path

from bt.tools import Tools
from bt.loader import DataLoader
from bt.methodology import Methodology


class MethodologyPriceTrendsAbs1(Methodology):
    def __init__(self,
                 mkt: str = 'KOSPI',
                 start_date: str = '20150101',
                 end_date: str = '20241031',
                 **kwargs):
        freq = kwargs.pop('freq', 'monthly')
        quantile = kwargs.pop('quantile', 10)
        quantile_position = kwargs.pop('quantile_position', [1])
        weight_type = kwargs.pop('weight_type', 'ew')
        select_lowest = kwargs.pop('select_lowest', True)
        keep_empty_periods = kwargs.pop('keep_empty_periods', True)
        file_name = kwargs.pop('file_name', 'price_trends_avg_test_avg.parquet')
        super().__init__(mkt, start_date, end_date, **kwargs)

        self.freq = freq
        self.quantile = quantile
        self.quantile_position = quantile_position
        self.weight_type = weight_type
        self.select_lowest = select_lowest
        self.keep_empty_periods = keep_empty_periods
        self.file_name = file_name
        
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
                                          prc=self.get_raw_factor())

    def get_raw_factor(self):
        try:
            # file_path = Path(__file__).parent / 'price_trends_avg.parquet'
            file_path = Path(__file__).parent / self.file_name
            
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
            return raw_factor
        except (ValueError, RuntimeError) as e:
            raise ValueError(f"Failed to create raw weight: {e}")

    def analyze_raw_factor(self):
        raw_factor = self.get_raw_factor()
        data = raw_factor.values.flatten()
        clean_data = data[~np.isnan(data)]
        
        print(f"Shape: {raw_factor.shape}")
        print(f"Missing: {np.isnan(data).sum():,} ({np.isnan(data).mean()*100:.1f}%)")
        print(f"Min: {clean_data.min():.3f}, Max: {clean_data.max():.3f}")
        print(f"Mean: {clean_data.mean():.3f}, Std: {clean_data.std():.3f}")
        print(f"Quantiles: P5={np.percentile(clean_data, 5):.3f}, P50={np.percentile(clean_data, 50):.3f}, P95={np.percentile(clean_data, 95):.3f}")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.hist(clean_data, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.set_title('Raw Factor Distribution')
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        time_means = raw_factor.mean(axis=1)
        ax2.plot(time_means.index, time_means.values, color='red', linewidth=1)
        ax2.set_title('Time Series Mean')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Mean Value')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()

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
        
        # Select stocks based on ranking
        if self.select_lowest:
            ranks = factor.rank(axis=1, method='first', ascending=True)  # Lower is better
        else:
            ranks = factor.rank(axis=1, method='first', ascending=False)  # Higher is better
        
        quantile = (ranks == 1).astype(int)
        return quantile

    def get_quantile_weights(self):
        quantile = self.get_quantile()
        quantile_weights = quantile.apply(lambda row: Tools.get_quantile_weights(row=row,
                                                                                 nums=self.quantile_position),
                                          axis=1)
        return quantile_weights

    def get_num_invested(self):
        num_invested = self.get_quantile().sum(axis=1)

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(num_invested.index, num_invested.values,
                color='blue', linewidth=2)
        ax.set_title('Number of Invested Stocks Per Period')
        ax.set_ylabel('Number of Stocks')
        ax.set_xlabel('Date')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

        print(
            f"Average stocks invested per period: {num_invested.mean():.1f}")
        return num_invested

    def calculate_forward_returns(self) -> pd.DataFrame:
        weights = self.weights
        prices = self.data['price_adj']

        selected_tickers = weights.idxmax(axis=1)
        selected_tickers.dropna(inplace=True)

        results = []
        forward_periods = [1, 2, 3, 4, 5, 20, 60]
        price_dates = prices.index

        for date, ticker in selected_tickers.items():
            result_row = {'ticker': ticker}

            try:
                start_idx_loc = price_dates.get_loc(date)
            except KeyError:
                continue

            p_t = prices.loc[date, ticker]
            if pd.isna(p_t) or p_t == 0:
                continue

            for n in forward_periods:
                col_name = f'{n}D_fwd_ret'
                fwd_date_loc = start_idx_loc + n

                if fwd_date_loc < len(price_dates):
                    fwd_price = prices.iloc[fwd_date_loc][ticker]
                    if pd.notna(fwd_price):
                        result_row[col_name] = (fwd_price / p_t) - 1
                    else:
                        result_row[col_name] = np.nan
                else:
                    result_row[col_name] = np.nan

            results.append((date, result_row))

        if not results:
            return pd.DataFrame()

        dates, data = zip(*results)
        result_df = pd.DataFrame(data, index=pd.Index(dates, name='date'))

        cols = ['ticker'] + [f'{n}D_fwd_ret' for n in forward_periods]
        return result_df[cols]

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
    method = MethodologyPriceTrendsAbs1(mkt="KOSPI200",
                                        start_date="20150101",
                                        end_date="20250627",
                                        weight_type="ew",
                                        select_lowest=False,
                                        keep_empty_periods=True)
    
    forward_returns = method.calculate_forward_returns()
    print(forward_returns)
    