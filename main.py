from threading import Lock

from cost import *
from performance import *


class PortfolioAnalysis:
    def __init__(self,
                 init_invest,
                 mkt,
                 start_date,
                 end_date,
                 methodology_type,
                 multiplier,
                 buy_commission=0.0,
                 sell_commission=0.0,
                 slippage=0.0,
                 sell_tax=0.0,
                 cash_rate=0.0,
                 **kwargs):
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

        self.methodology_kwargs = kwargs

    def calculate_weights(self, **kwargs):
        self.methodology = methodology_pool(methodology_type=self.selected_methodology,
                                            mkt=self.mkt,
                                            start_date=self.start_date,
                                            end_date=self.end_date,
                                            **self.methodology_kwargs)

    def construct_portfolio(self):
        if self.methodology is None:
            self.calculate_weights()
        weights = self.methodology.weights
        self.portfolio_constructor = PortfolioConstructor(
            mkt=self.mkt,
            weights=weights,
            init_invest=self.init_invest,
            buy_commission=self.buy_commission,
            sell_commission=self.sell_commission,
            slippage=self.slippage,
            sell_tax=self.sell_tax,
            cash_rate=self.cash_rate
        )

    def calculate_performance(self):
        if self.portfolio_constructor is None:
            self.construct_portfolio()
        pf_ret = self.portfolio_constructor.pf_ret
        bm_ret = self.portfolio_constructor.bm_ret
        self.perf_msre = PortfolioPerformance(pf_ret=pf_ret,
                                              bm_ret=bm_ret,
                                              multiplier=self.multiplier)

    def display_results(self):
        print(f"\nSelected methodology: {self.selected_methodology.name}")
        print(f"\nConditions:\n{self.portfolio_constructor}")
        print("\nRetrieving performance data...\n")
        self.perf_msre.performance_plot()
        print(self.perf_msre)
        print("\nTransaction Costs Summary:")
        print(self.transaction_costs_summary)
        print("\nCash Balance Summary:")
        print(self.cash_balance_summary)

    def run_analysis(self):
        self.calculate_weights()
        self.construct_portfolio()
        self.calculate_performance()
        self.display_results()

    @classmethod
    def run(cls,
            init_invest,
            mkt,
            start_date,
            end_date,
            methodology_type,
            multiplier,
            buy_commission,
            sell_commission,
            slippage,
            sell_tax,
            cash_rate,
            **kwargs):
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


class AnalysisManager:
    def __init__(self, config, methodology_types):
        self.config = config
        self.methodology_types = methodology_types
        self.analyses = {}
        self.lock = Lock()

    def run(self, method_type):
        analysis = PortfolioAnalysis.run(
            methodology_type=method_type,
            **self.config
        )
        with self.lock:
            self.analyses[method_type] = analysis
        return analysis

    def __repr__(self):
        return f"AnalysisManager(methodology_types={self.methodology_types})"

    def __str__(self):
        return "Available Methodologies:\n" + "\n".join(
            str(method) for method in self.methodology_types
        )


if __name__ == "__main__":
    cost = KoreaTransactionCost()

    config = {
        'init_invest': 1e8,
        'mkt': 'KOSPI200',
        'start_date': '20130101',
        'end_date': '20230701',
        'multiplier': 'Y',
        'buy_commission': cost.buy_commission,
        'sell_commission': cost.sell_commission,
        'slippage': cost.slippage,
        'sell_tax': cost.sell_tax,
        'cash_rate': cost.cash_rate,
        'freq': 'monthly',
        'quantile': 5,
        'quantile_position': [1],
        'weight_type': 'mktcap_float'
    }

    methodology_types = [
        MethodologyType.DataValidation,
        MethodologyType.GPAlfq0,
        MethodologyType.EBITDAEVttmlfq0,
        MethodologyType.FCFEVttmlfq0,
        MethodologyType.Momentum3612_1,
        MethodologyType.Payoutttmlfq0
    ]

    mgr = AnalysisManager(config, methodology_types)

    selected_methodology = MethodologyType.DataValidation
    analysis = mgr.run(selected_methodology)
