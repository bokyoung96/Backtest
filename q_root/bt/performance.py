import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

from bt.tools import Tools


class PerformanceMeasure(ABC):
    def __init__(self):
        pass

    @property
    @abstractmethod
    def pf_cumret(self):
        pass

    @property
    @abstractmethod
    def bm_cumret(self):
        pass

    @property
    @abstractmethod
    def performance_mean(self):
        pass

    @property
    @abstractmethod
    def performance_std(self):
        pass

    @property
    @abstractmethod
    def performance_cagr(self):
        pass

    @property
    @abstractmethod
    def performance_sharpe(self):
        pass

    @property
    @abstractmethod
    def performance_mdd(self):
        pass

    @property
    @abstractmethod
    def performance_hit(self):
        pass

    @property
    @abstractmethod
    def performance_cumret(self):
        pass

    @abstractmethod
    def performance_table(self):
        pass

    @abstractmethod
    def performance_plot(self):
        pass


class PortfolioPerformance(PerformanceMeasure):
    def __init__(self,
                 pf_ret: pd.DataFrame,
                 bm_ret: pd.DataFrame,
                 multiplier: str = 'D'):
        super().__init__()
        multiplier_dict = {'Y': 252,
                           'M': 20,
                           'D': 1}
        self.multiplier_ = multiplier
        self.multiplier = multiplier_dict.get(multiplier, 1)

        self.pf_ret = pf_ret.astype(float)
        self.bm_ret = bm_ret.astype(float)

        self.pf_dd = Tools.get_drawdown(cumret=self.pf_cumret)
        self.bm_dd = Tools.get_drawdown(cumret=self.bm_cumret)

    def __repr__(self):
        params = (
            f"multiplier: {self.multiplier_}, {self.multiplier}\n"
            f"Performance Summary: \n{self.performance_table()}\n"
            f"Performance Specific: \n{self.performance_specific()}"
        )
        return params

    @property
    def pf_cumret(self) -> pd.DataFrame:
        return (1 + self.pf_ret.fillna(0)).cumprod()

    @property
    def bm_cumret(self) -> pd.DataFrame:
        return (1 + self.bm_ret.fillna(0)).cumprod()

    @property
    def performance_mean(self) -> list:
        return [np.mean(self.pf_ret, axis=0).iloc[0] * self.multiplier,
                np.mean(self.bm_ret, axis=0).iloc[0] * self.multiplier]

    @property
    def performance_std(self) -> list:
        return [np.std(self.pf_ret, ddof=1, axis=0).iloc[0] * np.sqrt(self.multiplier),
                np.std(self.bm_ret, ddof=1, axis=0).iloc[0] * np.sqrt(self.multiplier)]

    @property
    def performance_cagr(self) -> list:
        pf_val_1 = self.pf_cumret.iloc[-1] / self.pf_cumret.iloc[0]
        pf_val_2 = 252 / len(self.pf_cumret)

        bm_val_1 = self.bm_cumret.iloc[-1] / self.bm_cumret.iloc[0]
        bm_val_2 = 252 / len(self.bm_cumret)
        return [((pf_val_1 ** pf_val_2) - 1).iloc[0],
                ((bm_val_1 ** bm_val_2) - 1).iloc[0]]

    @property
    def performance_sharpe(self) -> list:
        perf_mean = self.performance_mean
        perf_std = self.performance_std
        return [mean / std for mean, std in zip(perf_mean, perf_std)]

    @property
    def performance_mdd(self) -> list:
        return [np.min(self.pf_dd, axis=0).iloc[0], np.min(self.bm_dd, axis=0).iloc[0]]

    @property
    def performance_hit(self) -> list:
        pf_ret = self.pf_ret.copy()[1:]
        bm_ret = self.bm_ret.copy()[1:]

        # NOTE: ASSUME 0 AS NOT INVESTED, CASH-IN POSITION
        pf_invested = pf_ret[pf_ret != 0].dropna()
        bm_invested = bm_ret[bm_ret != 0].dropna()

        pf_hit = (pf_invested > 0).sum().iloc[0] / len(pf_invested) if len(pf_invested) > 0 else 0
        bm_hit = (bm_invested > 0).sum().iloc[0] / len(bm_invested) if len(bm_invested) > 0 else 0
        return [pf_hit, bm_hit]

    @property
    def performance_cumret(self) -> list:
        pf_cumret = self.pf_cumret.iloc[-1].values[0]
        bm_cumret = self.bm_cumret.iloc[-1].values[0]
        return [pf_cumret - 1, bm_cumret - 1]

    def performance_table(self) -> pd.DataFrame:
        def performance_pct_chg(perfs: list) -> list:
            res = np.round([perf * 100 for perf in perfs], 4)
            return res

        investing = self.pf_ret.index

        msres = [performance_pct_chg(self.performance_mean),
                 performance_pct_chg(self.performance_std),
                 performance_pct_chg(self.performance_cagr),
                 np.round(self.performance_sharpe, 4),
                 performance_pct_chg(self.performance_mdd),
                 performance_pct_chg(self.performance_hit),
                 performance_pct_chg(self.performance_cumret)]

        res = pd.DataFrame(msres,
                           columns=[f"Performance ({self.multiplier_}, Portfolio)",
                                    f"Performance ({self.multiplier_}, BM)"],
                           index=['Mean (%)',
                                  'Standard Deviation (%)',
                                  'CAGR (%)',
                                  'Sharpe Ratio',
                                  'MDD (%)',
                                  'Hit Ratio (%)',
                                  'CumRet (%)'])

        res.index.name = f"{Tools.get_dt_to_str(investing[0])} ~ {Tools.get_dt_to_str(investing[-1])}"
        return res

    def performance_specific(self) -> pd.DataFrame:
        pf_ret = (1 + self.pf_ret.iloc[:, 0]
                  ).groupby(self.pf_ret.index.year).cumprod()
        bm_ret = (1 + self.bm_ret.iloc[:, 0]
                  ).groupby(self.bm_ret.index.year).cumprod()

        pf_res = pf_ret.groupby(pf_ret.index.year).last()
        bm_res = bm_ret.groupby(bm_ret.index.year).last()

        res = pd.concat([pf_res, bm_res], axis=1)
        res.index.name = 'Year'
        res.columns = ['Portfolio', 'BM']

        res['ExcessRet'] = res['Portfolio'] - res['BM']
        res = np.round(res, 4)
        return res

    def performance_plot(self) -> plt.plot:
        fig, axs = plt.subplots(2, 1,
                                sharex=True,
                                gridspec_kw={'height_ratios': [2, 1]})

        axs[0].plot(self.pf_cumret,
                    label='Portfolio',
                    color='red',
                    linewidth=2,
                    linestyle='-')
        axs[0].plot(self.bm_cumret,
                    label='BM',
                    color='black',
                    linewidth=1.5,
                    linestyle='--')
        axs[0].set_ylabel('Cumulative return')
        axs[0].set_title('Portfolio versus BM cumulative return')
        axs[0].legend(loc='best')

        axs[1].plot(self.pf_dd,
                    label='Portfolio drawdown',
                    color='red',
                    linewidth=2,
                    linestyle='-')
        axs[1].plot(self.bm_dd,
                    label='BM drawdown',
                    color='black',
                    linewidth=1.5,
                    linestyle='--')
        axs[1].fill_between(self.pf_dd.index.to_pydatetime(),
                            self.pf_dd.values.flatten(),
                            color='red',
                            alpha=0.1)
        axs[1].fill_between(self.bm_dd.index.to_pydatetime(),
                            self.bm_dd.values.flatten(),
                            color='black',
                            alpha=0.1)
        axs[1].set_xlabel('Date')
        axs[1].set_ylabel('Drawdown')

        plt.tight_layout()
        plt.show()