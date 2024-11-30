import os
from functools import cached_property
from concurrent.futures import ThreadPoolExecutor, as_completed

from main import *


class Factors:
    def __init__(self,
                 analysis_manager: AnalysisManager):
        self.analysis_manager = analysis_manager

        self.max_workers = min(32, os.cpu_count() + 4)

        self._factor_returns_cache = None
        self._factor_weights_cache = None

    @property
    def methodology_names(self):
        return [member.name for member in self.analysis_manager.methodology_types]

    def _generate_factors(self):
        if self._factor_returns_cache is not None and self._factor_weights_cache is not None:
            return

        factor_returns = {}
        factor_weights = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.analysis_manager.run, method_type): method_type
                       for method_type in self.analysis_manager.methodology_types}
            for future in as_completed(futures):
                method_type = futures[future]
                try:
                    analysis = future.result()
                    factor_returns[method_type.name] = analysis.perf_msre.pf_ret
                    factor_weights[method_type.name] = analysis.portfolio_constructor.weights
                except Exception as exc:
                    print(f"Error occurred in {method_type}: {exc}")
        self._factor_returns_cache = factor_returns
        self._factor_weights_cache = factor_weights

    @cached_property
    def factor_returns(self):
        try:
            return pd.read_excel('./factors.xlsx', index_col=0, header=0)
        except FileNotFoundError:
            print("factors.xlsx not found. Calculating factor returns dynamically...")
            if self._factor_returns_cache is None:
                self._generate_factors()
            res = self._factor_returns_cache
            df = pd.concat(res.values(), axis=1)
            df.columns = list(res.keys())
            return df

    @cached_property
    def factor_weights(self):
        try:
            df = pd.read_excel('./factors_w.xlsx', index_col=0, header=0)
            df.index = df.index.to_series().ffill()
            df = df.reset_index(names="Factors").set_index(["Factors", "Date"])
            return df
        except FileNotFoundError:
            print("factors_w.xlsx not found. Calculating factor weights dynamically...")
            if self._factor_weights_cache is None:
                self._generate_factors()
            res = self._factor_weights_cache
            df = pd.concat(res)
            df.index.names = ['Factors', 'Date']
            return df


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
        MethodologyType.GPAlfq0,
        MethodologyType.EBITDAEVttmlfq0,
        MethodologyType.FCFEVttmlfq0,
        MethodologyType.Momentum3612_1,
        MethodologyType.Payoutttmlfq0
    ]

    mgr = AnalysisManager(config, methodology_types)
    factors = Factors(analysis_manager=mgr)
