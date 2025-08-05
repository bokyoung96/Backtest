import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
from pathlib import Path

from bt.tools import Tools
from bt.loader import DataLoader
from bt.methodology import Methodology


class MethodologyPriceTrends(Methodology):
    def __init__(self,
                 mkt: str = 'KOSPI',
                 start_date: str = '20150101',
                 end_date: str = '20241031',
                 factor_filename: str = 'price_trends_avg_prod.parquet',
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
        self.factor_filename = factor_filename
        
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
            file_path = Path(__file__).parent / self.factor_filename
            
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


class TopPriceTrends:
    def __init__(self, methodology: MethodologyPriceTrends, 
                 periods: List[int] = [5, 20, 60]):
        self.methodology = methodology
        self.periods = periods
        self.price_data = methodology.data['price_adj']
        
        self._top_stocks = None
        self._returns_data = None
        
    @property
    def top_stocks(self) -> pd.Series:
        if self._top_stocks is None:
            const, raw_factor = self.methodology.get_pp_data()
            scores = const.mul(raw_factor)
            scores_valid = scores.dropna(how='all')
            self._top_stocks = scores_valid.idxmax(axis=1)
        return self._top_stocks
    
    @property 
    def returns_data(self) -> pd.DataFrame:
        if self._returns_data is None:
            self._returns_data = self._calculate_returns()
        return self._returns_data
    
    def _calculate_returns(self) -> pd.DataFrame:
        results = []
        all_periods = list(range(1, 61))
        
        for date, stock_id in self.top_stocks.items():
            if stock_id not in self.price_data.columns or date not in self.price_data.index:
                continue
                
            stock_prices = self.price_data[stock_id]
            current_price = stock_prices.loc[date]
            
            if pd.isna(current_price) or current_price <= 0:
                continue
                
            period_returns = {'stock_id': stock_id, 'date': date}
            
            for period in all_periods:
                try:
                    future_date_idx = stock_prices.index.get_loc(date) + period
                    if future_date_idx < len(stock_prices):
                        future_price = stock_prices.iloc[future_date_idx]
                        if not pd.isna(future_price) and future_price > 0:
                            period_returns[f'return_{period}d'] = (future_price / current_price) - 1
                except (KeyError, IndexError):
                    continue
                
            if len(period_returns) > 10:
                results.append(period_returns)
        
        return pd.DataFrame(results).set_index('date')
    
    def get_summary_stats(self) -> Dict[str, pd.Series]:
        key_returns = self.returns_data[[f'return_{p}d' for p in self.periods]]
        
        return {
            'mean': key_returns.mean(),
            'std': key_returns.std(),
            'sharpe': key_returns.mean() / key_returns.std(),
            'win_rate': (key_returns > 0).mean(),
            'max': key_returns.max(),
            'min': key_returns.min()
        }
    
    def get_best_worst_cases(self, n: int = 3) -> Dict[str, pd.DataFrame]:
        key_returns = self.returns_data[[f'return_{p}d' for p in self.periods]]
        total_return = key_returns.sum(axis=1)
        
        best_indices = total_return.nlargest(n).index
        worst_indices = total_return.nsmallest(n).index
        
        return {
            'best': self.returns_data.loc[best_indices],
            'worst': self.returns_data.loc[worst_indices]
        }
    
    def plot_analysis(self, figsize: Tuple[int, int] = (24, 20)):
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(4, 5, height_ratios=[1, 0.7, 0.6, 0.6], width_ratios=[1, 1, 1, 0.1, 1],
                             hspace=0.4, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, :4])
        ax2 = fig.add_subplot(gs[1, :4])
        ax3_1 = fig.add_subplot(gs[2, 0])
        ax3_2 = fig.add_subplot(gs[2, 1]) 
        ax3_3 = fig.add_subplot(gs[2, 2])
        ax3_text = fig.add_subplot(gs[2, 4])
        ax4_1 = fig.add_subplot(gs[3, 0])
        ax4_2 = fig.add_subplot(gs[3, 1])
        ax4_3 = fig.add_subplot(gs[3, 2])
        ax4_text = fig.add_subplot(gs[3, 4])
        
        self._plot_cumulative_trajectory(ax1)
        self._plot_return_distributions(ax2)
        self._plot_individual_best_cases([ax3_1, ax3_2, ax3_3])
        self._plot_individual_worst_cases([ax4_1, ax4_2, ax4_3])
        self._add_top_text_box(ax3_text)
        self._add_bottom_text_box(ax4_text)
        
        plt.tight_layout()
        
        return fig
    
    def _plot_cumulative_trajectory(self, ax):
        days = range(0, 61)
        
        for _, row in self.returns_data.iterrows():
            trajectory = [0]
            for day in range(1, 61):
                col = f'return_{day}d'
                if col in row and not pd.isna(row[col]):
                    trajectory.append(row[col])
                else:
                    trajectory.append(np.nan)
            
            valid_points = sum(1 for x in trajectory if not pd.isna(x))
            if valid_points >= 10:
                ax.plot(days, trajectory, '-', alpha=0.1, linewidth=0.6, color='lightgray')
        
        ax2 = ax.twinx()
        avg_trajectory = [0]
        for day in range(1, 61):
            col = f'return_{day}d'
            if col in self.returns_data.columns:
                avg_trajectory.append(self.returns_data[col].mean())
            else:
                avg_trajectory.append(np.nan)
        
        ax2.plot(days, avg_trajectory, '-', color='steelblue', linewidth=5, 
                label='Average Return', zorder=10)
        
        for period in self.periods:
            if period <= 60:
                ax.axvline(x=period, color='red', linestyle='--', alpha=0.8, linewidth=2)
                if period < len(avg_trajectory) and not pd.isna(avg_trajectory[period]):
                    ret_val = avg_trajectory[period]
                    ax2.annotate(f'{period}d\n{ret_val:.1%}', 
                               xy=(period, ret_val), 
                               xytext=(10, 20), 
                               textcoords='offset points',
                               fontsize=12, 
                               bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.9),
                               ha='center')
        
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.4)
        ax2.axhline(y=0, color='steelblue', linestyle='-', alpha=0.8, linewidth=2)
        
        ax.set_title('Average Returns (0-60 Days)', fontsize=16)
        ax.set_xlabel('Days', fontsize=14)
        ax.set_ylabel('Individual Stock Returns', fontsize=14, color='gray')
        ax2.set_ylabel('Average Return', fontsize=14, color='steelblue')
        
        ax2.legend(fontsize=12, loc='upper left')
        ax.grid(True, alpha=0.3)
        
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        ax.tick_params(axis='y', labelcolor='gray')
        ax2.tick_params(axis='y', labelcolor='steelblue')
    
    def _plot_return_distributions(self, ax):
        key_returns = self.returns_data[[f'return_{p}d' for p in self.periods]]
        colors = ['skyblue', 'orange', 'lightgreen']
        
        for i, period in enumerate(self.periods):
            col = f'return_{period}d'
            data = key_returns[col].dropna()
            
            ax.hist(data, bins=30, alpha=0.6, density=True, 
                   color=colors[i % len(colors)], 
                   label=f'{period}d (μ={data.mean():.1%})',
                   edgecolor='black', linewidth=0.5)
        
        ax.axvline(0, color='red', linestyle='--', alpha=0.8, linewidth=2)
        ax.set_title('Return Distributions', fontsize=16)
        ax.set_xlabel('Return', fontsize=14)
        ax.set_ylabel('Density', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    
    def _plot_individual_best_cases(self, axs):
        cases = self.get_best_worst_cases()
        best_data = cases['best']
        days = range(0, 61)
        colors = ['black', 'gray', 'lightgray']
        
        for i, (date, row) in enumerate(best_data.iterrows()):
            if i >= len(axs):
                break
                
            ax = axs[i]
            trajectory = [0]
            for day in range(1, 61):
                col = f'return_{day}d'
                if col in row and not pd.isna(row[col]):
                    trajectory.append(row[col])
                else:
                    trajectory.append(np.nan)
            
            ax.plot(days, trajectory, '-', linewidth=3, color=colors[i])
            
            for period in self.periods:
                if period < len(trajectory) and not pd.isna(trajectory[period]):
                    ax.plot(period, trajectory[period], 'o', markersize=8, color=colors[i])
                ax.axvline(x=period, color='red', linestyle=':', alpha=0.6, linewidth=1)
            
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.set_title(f'Top {i+1}', fontsize=14)
            ax.set_xlabel('Days', fontsize=12)
            ax.set_ylabel('Return', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    def _plot_individual_worst_cases(self, axs):
        cases = self.get_best_worst_cases()
        worst_data = cases['worst']
        days = range(0, 61)
        colors = ['black', 'gray', 'lightgray']
        
        for i, (date, row) in enumerate(worst_data.iterrows()):
            if i >= len(axs):
                break
                
            ax = axs[i]
            trajectory = [0]
            for day in range(1, 61):
                col = f'return_{day}d'
                if col in row and not pd.isna(row[col]):
                    trajectory.append(row[col])
                else:
                    trajectory.append(np.nan)
            
            ax.plot(days, trajectory, '-', linewidth=3, color=colors[i])
            
            for period in self.periods:
                if period < len(trajectory) and not pd.isna(trajectory[period]):
                    ax.plot(period, trajectory[period], 'o', markersize=8, color=colors[i])
                ax.axvline(x=period, color='red', linestyle=':', alpha=0.6, linewidth=1)
            
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.set_title(f'Bottom {i+1}', fontsize=14)
            ax.set_xlabel('Days', fontsize=12)
            ax.set_ylabel('Return', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    def _add_top_text_box(self, ax):
        cases = self.get_best_worst_cases()
        best_data = cases['best']
        
        text_lines = ["TOP 3 CASES\n"]
        
        for i, (date, row) in enumerate(best_data.iterrows()):
            if i >= 3:
                break
            stock_id = row['stock_id']
            
            text_lines.append(f"Top {i+1}: {date.strftime('%Y/%m/%d')} {stock_id}")
            for period in self.periods:
                col = f'return_{period}d'
                if col in row and not pd.isna(row[col]):
                    text_lines.append(f"  {period}d: {row[col]:.2f}")
            text_lines.append("")
        
        text = "\n".join(text_lines)
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=12, 
               va='top', ha='left', fontfamily='monospace')
        ax.axis('off')
    
    def _add_bottom_text_box(self, ax):
        cases = self.get_best_worst_cases()
        worst_data = cases['worst']
        
        text_lines = ["BOTTOM 3 CASES\n"]
        
        for i, (date, row) in enumerate(worst_data.iterrows()):
            if i >= 3:
                break
            stock_id = row['stock_id']
            
            text_lines.append(f"Bottom {i+1}: {date.strftime('%Y/%m/%d')} {stock_id}")
            for period in self.periods:
                col = f'return_{period}d'
                if col in row and not pd.isna(row[col]):
                    text_lines.append(f"  {period}d: {row[col]:.2f}")
            text_lines.append("")
        
        text = "\n".join(text_lines)
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=12, 
               va='top', ha='left', fontfamily='monospace')
        ax.axis('off')
    
    def analyze(self) -> Dict:
        stats = self.get_summary_stats()
        cases = self.get_best_worst_cases()
        
        print("=== Top Price Trends Analysis ===")
        print(f"Total signals: {len(self.top_stocks)}")
        print(f"Valid calculations: {len(self.returns_data)}")
        print("\n--- Summary Statistics ---")
        
        for period in self.periods:
            col = f'return_{period}d'
            mean_ret = stats['mean'][col]
            win_rate = stats['win_rate'][col]
            sharpe = stats['sharpe'][col]
            print(f"{period:2d}d: {mean_ret:6.1%} (WR: {win_rate:5.1%}, Sharpe: {sharpe:5.2f})")
        
        return {
            'summary_stats': stats,
            'best_cases': cases['best'],
            'worst_cases': cases['worst'],
            'total_signals': len(self.top_stocks),
            'valid_returns': len(self.returns_data)
        }


if __name__ == "__main__":
    method = MethodologyPriceTrends(mkt="KOSPI200",
                                    start_date="20150101",
                                    end_date="20250627",
                                    weight_type="ew",
                                    quantile=10,
                                    quantile_position=[1],
                                    freq='weekly',
                                    factor_filename='price_trends_avg_prod.parquet')
    
    # For additional analysis
    # analyzer = TopPriceTrends(method)
    # results = analyzer.analyze()
    
    # analyzer.plot_analysis()
    # plt.show()