import logging
import pandas as pd
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
            logging.error("Portfolio returns are empty, cannot calculate performance.")
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
