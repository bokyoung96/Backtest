import gc
import functools
import pandas as pd
from bt.methodology_type import MethodologyType
from bt.cost import NoTransactionCost, KoreaTransactionCost
from bt.main import AnalysisManager


def memory_mgr(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        gc.collect()
        try:
            res = func(*args, **kwargs)
            return res
        finally:
            gc.collect()
    return wrapper


class AnalysisInteractive:
    def __init__(self, analysis):
        self._analysis = analysis
        
    @memory_mgr
    def get_performance(self):
        return self._analysis.perf_msre.performance_table()
    
    @memory_mgr
    def get_analysis_attr(self, attr_name):
        if hasattr(self._analysis, attr_name):
            return getattr(self._analysis, attr_name)
        return None
    
    def clear_cache(self):
        for attr in dir(self._analysis):
            if attr.startswith('_cache_'):
                delattr(self._analysis, attr)
        gc.collect()
    
    def __getattr__(self, name):
        return getattr(self._analysis, name)


cost = NoTransactionCost()

config = {
    'init_invest': 1e8,
    'mkt': 'KOSPI200',
    'start_date': '20200101',
    'end_date': '20231231',
    'multiplier': 'Y',
    'buy_commission': cost.buy_commission,
    'sell_commission': cost.sell_commission,
    'slippage': cost.slippage,
    'sell_tax': cost.sell_tax,
    'cash_rate': cost.cash_rate,
    'rebal_timing': 'now',
    'freq': 'monthly',
    'quantile': 5,
    'quantile_position': [1],
    'weight_type': 'ew'
}


if __name__ == "__main__":
    methodology_types = [
        MethodologyType.DataValidation,
        MethodologyType.ERRChg
    ]

    mgr = AnalysisManager(config, methodology_types)
    
    selected_methodology = MethodologyType.DataValidation
    raw_analysis = mgr.run(selected_methodology)
    
    analysis = AnalysisInteractive(raw_analysis)