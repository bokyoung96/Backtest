import datetime as dt
from tqdm import tqdm
import pandas as pd
import numpy as np
from tools import *
from loader import *


class PortfolioConstructor:
    def __init__(self,
                 mkt: str,
                 weights: pd.DataFrame,
                 init_invest: float = 1e8,
                 buy_commission: float = 0.0,
                 sell_commission: float = 0.0,
                 slippage: float = 0.0,
                 sell_tax: float = 0.0,
                 cash_rate: float = 0.0
                 ):
        self.mkt = mkt
        self.weights = weights
        self.init_invest = init_invest

        self.buy_commission = buy_commission
        self.sell_commission = sell_commission
        self.slippage = slippage
        self.sell_tax = sell_tax
        self.cash_rate = cash_rate

        data_loader = DataLoader(mkt=mkt)
        self.price = data_loader(data_name='price_adj').fillna(0)
        self.bm = data_loader(data_name='bm')
        self.trans_ban = data_loader(data_name='trans_ban')

        self._res_quantity = None
        self._transaction_summaries = []
        self._cash_balances = []

    def __repr__(self) -> str:
        params = (
            f"init_invest: {self.init_invest}\n"
            f"mkt: {self.mkt}\n"
            f"init_invest_date: {self.init_invest_date}\n"
            f"last_invest_date: {self.last_invest_date}"
        )
        return params

    @property
    def bday_dates(self) -> list:
        return list(self.price.index)

    @property
    def formation_dates(self) -> list:
        return list(self.weights.index)

    @property
    def rebalancing_dates(self) -> list:
        bday_dates = np.array(self.bday_dates)
        res = []
        for formation_date in self.formation_dates:
            idx = np.searchsorted(bday_dates, formation_date, side='right')
            if idx < len(bday_dates):
                nearest = bday_dates[idx]
            else:
                nearest = bday_dates[-1]
                print(f"No {formation_date} bday: Substituted to {nearest}.")
            res.append(nearest)
        return res

    @property
    def init_invest_date(self) -> str:
        return dt.datetime.strftime(self.rebalancing_dates[0], format='%Y%m%d')

    @property
    def last_invest_date(self) -> str:
        return dt.datetime.strftime(self.rebalancing_dates[-1], format='%Y%m%d')

    def get_quantity(self,
                     formation_date: str,
                     rebalancing_date: str,
                     total_assets: float) -> pd.Series:
        weights_ = self.weights.loc[formation_date].dropna()
        init_price_ = self.price.loc[rebalancing_date]
        init_trans_ban = self.trans_ban.loc[rebalancing_date]

        invested = weights_[(init_price_ > 0) & (init_trans_ban == 0)].index
        init_price = init_price_.loc[invested]
        weights = weights_.loc[invested]

        weights /= weights.sum()
        init_invest = weights * total_assets
        q = init_invest / init_price
        return q

    def pf_cashflow(self) -> pd.DataFrame:
        print("\n***** Portfolio under backtest *****\n")
        rebalancing_periods = len(self.rebalancing_dates)

        temp = []
        res_quantity = []
        previous_quantity = None
        self._transaction_summaries = []
        self._cash_balances = []

        cash_balance = self.init_invest

        for period in tqdm(range(1, rebalancing_periods + 1)):
            start_date = self.rebalancing_dates[period - 1]
            formation_date = self.formation_dates[period - 1]

            if period != rebalancing_periods:
                end_date = self.rebalancing_dates[period]
                days_in_period = (end_date - start_date).days
            else:
                end_date = self.price.index[-1]
                days_in_period = (end_date - start_date).days

            interest = cash_balance * (self.cash_rate / 365) * days_in_period
            cash_balance += interest

            if previous_quantity is None:
                portfolio_value = 0.0
            else:
                portfolio_value = np.dot(
                    previous_quantity, self.price.loc[start_date, previous_quantity.index])

            total_assets = portfolio_value + cash_balance
            total_cost_rate = self.buy_commission + \
                self.slippage + self.sell_commission + self.sell_tax
            investable_assets = total_assets / (1 + total_cost_rate)

            quantity = self.get_quantity(formation_date=formation_date,
                                         rebalancing_date=start_date,
                                         total_assets=investable_assets)
            res_quantity.append(quantity)

            total_transaction_cost, transaction_summary = self._calculate_transaction_costs(
                quantity, previous_quantity, start_date, total_assets)
            self._transaction_summaries.append(transaction_summary)

            cash_balance = total_assets - investable_assets - total_transaction_cost
            if cash_balance < 0:
                raise ValueError(
                    f"Negative cash balance encountered on {start_date}. Cash Balance: {cash_balance}")
            self._cash_balances.append(
                {'Date': start_date, 'Cash Balance': cash_balance})

            if end_date is not None:
                price_temp = self.price[quantity.index][start_date:end_date].iloc[:-1]
            else:
                price_temp = self.price[quantity.index][start_date:]
            dates = price_temp.index

            portfolio_values = np.dot(quantity, price_temp.to_numpy().T)
            cf = pd.DataFrame(portfolio_values, index=dates,
                              columns=['Portfolio'])

            previous_quantity = quantity

            temp.append(cf)
            print(
                f"Backtest in progress: {start_date} completed. Moving on...")

        res = pd.concat(temp)
        res.index = pd.to_datetime(res.index)
        res.index.name = 'Date'

        cash_df = pd.DataFrame(self._cash_balances).set_index('Date')
        res['Cash'] = cash_df['Cash Balance'].reindex(
            res.index, method='ffill').fillna(0)
        res['Total'] = res['Portfolio'] + res['Cash']

        self._res_quantity = res_quantity
        return res

    def _calculate_transaction_costs(self, current_quantity, previous_quantity, date, total_assets):
        price_at_date = self.price.loc[date, current_quantity.index]

        if previous_quantity is None:
            net_trades = current_quantity
            previous_quantity_reindexed = pd.Series(
                0, index=current_quantity.index)
        else:
            previous_quantity_reindexed = previous_quantity.reindex(
                current_quantity.index).fillna(0).astype(float)
            net_trades = current_quantity - previous_quantity_reindexed

        buys = net_trades[net_trades > 0]
        sells = -net_trades[net_trades < 0]

        buy_values = (buys * price_at_date[buys.index]).sum()
        sell_values = (sells * price_at_date[sells.index]).sum()

        buy_cost = (self.buy_commission + self.slippage) * buy_values
        sell_cost = (self.sell_commission + self.slippage +
                     self.sell_tax) * sell_values

        total_transaction_cost = buy_cost + sell_cost

        shares_bought = buys.sum()
        shares_sold = sells.sum()

        transaction_cost_bp = (
            total_transaction_cost / total_assets) * 10000 if total_assets != 0 else np.nan

        transaction_summary = {
            'Date': date,
            'Total Buy Value': buy_values,
            'Total Sell Value': sell_values,
            'Shares Bought': shares_bought,
            'Shares Sold': shares_sold,
            'Total Buy Cost': buy_cost,
            'Total Sell Cost': sell_cost,
            'Total Transaction Cost': total_transaction_cost,
            'Total NAV': total_assets,
            'Transaction Cost (bp)': transaction_cost_bp
        }

        return total_transaction_cost, transaction_summary

    @property
    def transaction_costs_summary(self) -> pd.DataFrame:
        if not self._transaction_summaries:
            raise ValueError(
                "Transaction costs have not been calculated. Execute pf_cashflow() or pf_ret.")
        return pd.DataFrame(self._transaction_summaries)

    @property
    def cash_balance_summary(self) -> pd.DataFrame:
        if not self._cash_balances:
            raise ValueError(
                "Cash balances have not been calculated. Execute pf_cashflow() or pf_ret.")
        return pd.DataFrame(self._cash_balances).set_index('Date')

    @property
    def pf_ret(self) -> pd.DataFrame:
        cashflow = self.pf_cashflow()
        pf_ret = pd.DataFrame(cashflow['Total'].pct_change())
        pf_ret.index = pd.to_datetime(cashflow.index)
        return pf_ret

    @property
    def bm_ret(self) -> pd.DataFrame:
        bm_ret = self.bm.loc[self.init_invest_date:].pct_change()
        bm_ret.index = pd.to_datetime(bm_ret.index)
        return bm_ret

    @property
    def pf_quantity(self) -> pd.DataFrame:
        if self._res_quantity is None:
            raise ValueError(
                "Portfolio quantities have not been calculated. Execute pf_cashflow() or pf_ret.")
        return pd.DataFrame(self._res_quantity).T
