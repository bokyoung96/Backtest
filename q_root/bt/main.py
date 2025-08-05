import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from threading import Lock
from typing import Dict, Optional

from bt.methodology_type import MethodologyType, methodology_pool
from bt.performance import PortfolioPerformance
from bt.backtest import PortfolioBacktester


class PortfolioAnalysis:
    def __init__(self,
                 init_invest: float,
                 mkt: str,
                 start_date: str,
                 end_date: str,
                 methodology_type: MethodologyType,
                 multiplier: str,
                 buy_commission: float = 0.0,
                 sell_commission: float = 0.0,
                 slippage: float = 0.0,
                 sell_tax: float = 0.0,
                 cash_rate: float = 0.0,
                 rebal_timing: str = 'next',
                 **kwargs) -> None:
        self.init_invest = init_invest
        self.mkt = mkt
        self.start_date = start_date
        self.end_date = end_date
        self.selected_methodology = methodology_type
        self.multiplier = multiplier

        self.methodology = None
        self.portfolio_constructor = None
        self.perf_msre = None

        self.buy_commission = buy_commission
        self.sell_commission = sell_commission
        self.slippage = slippage
        self.sell_tax = sell_tax
        self.cash_rate = cash_rate
        self.rebal_timing = rebal_timing

        self.methodology_kwargs = kwargs

    def calculate_weights(self, **kwargs) -> None:
        self.methodology = methodology_pool(methodology_type=self.selected_methodology,
                                            mkt=self.mkt,
                                            start_date=self.start_date,
                                            end_date=self.end_date,
                                            **self.methodology_kwargs)

    def construct_portfolio(self) -> None:
        if self.methodology is None:
            self.calculate_weights()
        weights = self.methodology.weights
        self.portfolio_constructor = PortfolioBacktester(
            mkt=self.mkt,
            weights=weights,
            init_invest=self.init_invest,
            buy_commission=self.buy_commission,
            sell_commission=self.sell_commission,
            slippage=self.slippage,
            sell_tax=self.sell_tax,
            cash_rate=self.cash_rate,
            rebal_timing=self.rebal_timing
        )

    def calculate_performance(self) -> None:
        print("\nRetrieving performance data...")
        pf_ret = self.portfolio_constructor.portfolio_returns
        bm_ret = self.portfolio_constructor.benchmark_returns
        transactions = self.portfolio_constructor.transaction_costs_summary
        cash_summary = self.portfolio_constructor.cash_balance_summary
        holdings = self.portfolio_constructor.get_holdings_snapshot()

        if pf_ret is None or pf_ret.empty:
            logging.error(
                "Portfolio returns are empty, cannot calculate performance.")
            return

        self.perf_msre = PortfolioPerformance(pf_ret=pf_ret,
                                              bm_ret=bm_ret,
                                              multiplier=self.multiplier)

    def display_results(self) -> None:
        print(f"\nSelected methodology: {self.selected_methodology.name}")
        print(f"\nConditions:\n{self.portfolio_constructor}")
        print("\nRetrieving performance data...\n")
        self.perf_msre.performance_plot()
        print(self.perf_msre)
        print("\nTransaction Costs Summary:")
        print(self.transaction_costs_summary)
        print("\nCash Balance Summary:")
        print(self.cash_balance_summary)
        print("\nHoldings Snapshot (Sample - First & Last Rebalancing Date):")
        if self.holdings_snapshot:
            dates = list(self.holdings_snapshot.keys())
            if dates:
                first_date = dates[0]
                last_date = dates[-1]
                print(f"--- Holdings on {first_date} (Top 5) ---")
                print(self.holdings_snapshot[first_date].head(
                    5).to_string(float_format="{:,.4f}".format))
                if first_date != last_date:
                    print(f"\n--- Holdings on {last_date} (Top 5) ---")
                    print(self.holdings_snapshot[last_date].head(
                        5).to_string(float_format="{:,.4f}".format))
            else:
                print("Holdings snapshot data is empty.")
        else:
            print("Holdings snapshot not available.")

    def run_analysis(self) -> None:
        self.calculate_weights()
        self.construct_portfolio()
        _ = self.portfolio_constructor.run_backtest()
        self.calculate_performance()
        if self.perf_msre:
            print(self.perf_msre)
            # self.perf_msre.performance_plot()

    @classmethod
    def run(cls,
            init_invest: float,
            mkt: str,
            start_date: str,
            end_date: str,
            methodology_type: MethodologyType,
            multiplier: str,
            buy_commission: float,
            sell_commission: float,
            slippage: float,
            sell_tax: float,
            cash_rate: float,
            **kwargs) -> 'PortfolioAnalysis':
        inst = cls(init_invest,
                   mkt,
                   start_date,
                   end_date,
                   methodology_type,
                   multiplier,
                   buy_commission=buy_commission,
                   sell_commission=sell_commission,
                   slippage=slippage,
                   sell_tax=sell_tax,
                   cash_rate=cash_rate,
                   **kwargs)
        inst.run_analysis()
        return inst

    @property
    def transaction_costs_summary(self) -> pd.DataFrame:
        if self.portfolio_constructor is None:
            raise ValueError("PortfolioConstructor not initialized.")
        return self.portfolio_constructor.transaction_costs_summary

    @property
    def cash_balance_summary(self) -> pd.DataFrame:
        if self.portfolio_constructor is None:
            raise ValueError("PortfolioConstructor not initialized.")
        return self.portfolio_constructor.cash_balance_summary

    @property
    def holdings_snapshot(self) -> Optional[Dict[str, pd.DataFrame]]:
        if self.portfolio_constructor:
            try:
                return self.portfolio_constructor.get_holdings_snapshot()
            except Exception as e:
                print(f"Error accessing holdings_snapshot: {e}")
                return None
        return None

    @property
    def sector_snapshot(self) -> Optional[Dict[str, Dict[str, float]]]:
        if self.portfolio_constructor:
            try:
                return self.portfolio_constructor.get_sector_snapshot()
            except Exception as e:
                logging.error(f"Error accessing sector_snapshot: {e}")
                return None
        return None

    def get_holding_period_returns(self) -> Optional[pd.DataFrame]:
        try:
            rebalancing_dates = pd.to_datetime(
                self.portfolio_constructor.date_manager.rebalancing_dates)
            price_data = self.portfolio_constructor.price
        except AttributeError:
            logging.error("Constructor attributes not initialized.")
            return None

        if rebalancing_dates.empty or price_data is None or price_data.empty:
            logging.warning("Rebalancing dates or price data unavailable.")
            return None

        if not self.holdings_snapshot:
            logging.warning("Holdings snapshot is empty.")
            return None

        price_data.index = pd.to_datetime(price_data.index)
        snapshot_dates = sorted(
            [pd.to_datetime(d) for d in self.holdings_snapshot.keys()])

        records = []
        for holding_start_date in snapshot_dates:
            holdings = self.holdings_snapshot[holding_start_date.strftime('%Y%m%d')]
            if holdings.empty:
                continue

            next_rebal_idx = rebalancing_dates.searchsorted(
                holding_start_date, side='right')

            if next_rebal_idx >= len(rebalancing_dates):
                holding_end_date = pd.to_datetime(self.end_date)
            else:
                holding_end_date = rebalancing_dates[next_rebal_idx]

            for ticker in holdings.index:
                try:
                    start_price = price_data.loc[price_data.index.asof(
                        holding_start_date), ticker]
                    end_price = price_data.loc[price_data.index.asof(
                        holding_end_date), ticker]

                    if pd.notna(start_price) and pd.notna(end_price) and start_price > 0:
                        period_return = (end_price / start_price) - 1
                        records.append({
                            'rebal_date': holding_start_date.strftime('%Y-%m-%d'),
                            'ticker': ticker,
                            'start_price': start_price,
                            'end_date': holding_end_date.strftime('%Y-%m-%d'),
                            'end_price': end_price,
                            'return': period_return
                        })
                except (KeyError, IndexError):
                    continue

        if not records:
            return pd.DataFrame()

        return pd.DataFrame(records)

    def plot_holding_period_analysis(self, hpr_df: pd.DataFrame) -> None:
        if hpr_df is None or hpr_df.empty:
            print("Holding period returns data is not available for plotting.")
            return

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axes = plt.subplots(2, 1, figsize=(15, 12), constrained_layout=True)
        fig.suptitle('Holding Period Return Analysis', fontsize=16)

        axes[0].hist(hpr_df['return'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].set_title('Distribution of Monthly Holding Returns')
        axes[0].set_xlabel('Return')
        axes[0].set_ylabel('Frequency')
        axes[0].axvline(hpr_df['return'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {hpr_df['return'].mean():.2%}")
        axes[0].legend()

        top_3 = hpr_df.nlargest(3, 'return')
        bottom_3 = hpr_df.nsmallest(3, 'return')
        extreme_performers = pd.concat([top_3, bottom_3])

        price_data = self.portfolio_constructor.price
        price_data.index = pd.to_datetime(price_data.index)

        axes[1].set_title(
            'Cumulative Returns of Top/Bottom 3 Holdings')

        for _, row in extreme_performers.iterrows():
            start_date = pd.to_datetime(row['rebal_date'])
            end_date = pd.to_datetime(row['end_date'])
            ticker = row['ticker']

            period_prices = price_data.loc[start_date:end_date, ticker].dropna()
            if period_prices.empty or len(period_prices) < 2:
                continue

            cumulative_returns = (period_prices / period_prices.iloc[0])
            days_from_start = range(len(cumulative_returns))

            label = f"{ticker} ({start_date.strftime('%Y-%m-%d')}, Ret: {row['return']:.2%})"
            axes[1].plot(days_from_start, cumulative_returns.values,
                         label=label, linestyle='--', alpha=0.7)

        axes[1].axhline(1.0, color='black', linestyle='-', linewidth=1)
        axes[1].set_ylabel('Cumulative Return (Normalized to 1.0 at Start)')
        axes[1].set_xlabel('Trading Days from Investment Start')
        axes[1].legend(loc='best', fontsize='small')
        plt.show()


class AnalysisManager:
    def __init__(self, config: Dict, methodology_types: list) -> None:
        self.config = config
        self.methodology_types = methodology_types
        self.analyses: Dict[MethodologyType, PortfolioAnalysis] = {}
        self.lock = Lock()

    def run(self, method_type: MethodologyType) -> PortfolioAnalysis:
        analysis = PortfolioAnalysis.run(
            methodology_type=method_type,
            **self.config
        )
        with self.lock:
            self.analyses[method_type] = analysis
        return analysis

    def __repr__(self) -> str:
        return f"AnalysisManager(methodology_types={self.methodology_types})"

    def __str__(self) -> str:
        return "Available Methodologies:\n" + "\n".join(
            str(method) for method in self.methodology_types
        )


if __name__ == "__main__":
    common_params = {
        'init_invest': 1000000,
        'mkt': 'KOSPI200',
        'start_date': '20200101',
        'end_date': '20250627',
        'methodology_type': MethodologyType.MethodologyPriceTrendsAbs,
        'multiplier': 'Y',
        'buy_commission': 0.02/100,
        'sell_commission': 0.02/100,
        'slippage': 0.01/100,
        'sell_tax': 0.15/100,
        'cash_rate': 0.02,
        'rebal_timing': 'same',
        'weight_type': 'ew'
    }

    analysis1 = PortfolioAnalysis.run(
        file_name='price_trends_avg_test_20.parquet',
        score_threshold=0.35,
        inverse_threshold=True,
        keep_empty_periods=True,
        freq='monthly',
        **common_params
    )

    hpr_df = analysis1.get_holding_period_returns()
    if hpr_df is not None and not hpr_df.empty:
        print("\n--- Holding Period Returns ---")
        with pd.option_context('display.max_rows', None, 'display.width', 1000):
            print(hpr_df)
    
    analysis1.plot_holding_period_analysis(hpr_df)

