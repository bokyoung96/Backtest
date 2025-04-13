import datetime as dt
from tqdm import tqdm
import pandas as pd
import numpy as np
import logging
from typing import Literal, List, Dict, Tuple, Optional
from tools import *
from loader import *

log_filename = 'backtest.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, mode='w'),
        logging.StreamHandler()
    ]
)


class PortfolioConstructor:
    """
    Constructs and backtests a portfolio based on given weights and market data.

    Handles rebalancing, transaction costs, slippage, taxes, and cash management.
    """

    def __init__(self,
                 mkt: str,
                 weights: pd.DataFrame,
                 init_invest: float = 1e8,
                 buy_commission: float = 0.0,
                 sell_commission: float = 0.0,
                 slippage: float = 0.0,
                 sell_tax: float = 0.0,
                 cash_rate: float = 0.0,
                 rebal_timing: Literal['next', 'now'] = 'next'
                 ):
        """
        Initializes the PortfolioConstructor.

        Args:
            mkt: Market identifier(e.g., 'KOSPI').
            weights: DataFrame with dates as index and stock codes as columns,
                     containing target portfolio weights.
            init_invest: Initial investment amount.
            buy_commission: Commission rate for buying assets.
            sell_commission: Commission rate for selling assets.
            slippage: Slippage rate applied to transactions.
            sell_tax: Tax rate applied to profits from selling assets.
            cash_rate: Annual interest rate earned on cash balance.
            rebal_timing: How to determine the rebalancing date from the formation date.
                          'next': Use the next business day strictly after the formation date.
                          'now': Use the formation date if it's a business day, otherwise the next business day.
        """
        self.mkt = mkt
        self.weights = weights
        self.init_invest = init_invest

        self.buy_commission = buy_commission
        self.sell_commission = sell_commission
        self.slippage = slippage
        self.sell_tax = sell_tax
        self.cash_rate = cash_rate
        self.rebal_timing = rebal_timing

        data_loader = DataLoader(mkt=mkt)
        self.price = data_loader(data_name='price_adj').fillna(0)
        self.bm = data_loader(data_name='bm')
        self.trans_ban = data_loader(data_name='trans_ban')

        self._res_quantity: Optional[List[pd.Series]] = None
        self._transaction_summaries: List[Dict] = []
        self._cash_balances: List[Dict] = []
        self._rebalancing_dates_cache: Optional[List[dt.datetime]] = None
        # Store actual final holdings after scaling
        self._final_holdings_data: List[Dict] = []

    def __repr__(self) -> str:
        params = (
            f"init_invest: {self.init_invest}\n"
            f"mkt: {self.mkt}\n"
            f"init_invest_date: {self.init_invest_date}\n"
            f"last_invest_date: {self.last_invest_date}\n"
            f"rebal_timing: {self.rebal_timing}"
        )
        return params

    @property
    def bday_dates(self) -> List[dt.datetime]:
        """Returns a list of business days based on the price data index."""
        return list(pd.to_datetime(self.price.index))

    @property
    def formation_dates(self) -> List[dt.datetime]:
        """Returns a list of formation dates based on the weights data index."""
        return list(pd.to_datetime(self.weights.index))

    @property
    def rebalancing_dates(self) -> List[dt.datetime]:
        """
        Calculates the rebalancing dates based on formation dates and the chosen strategy.

        'next': Finds the first business day strictly after the formation date.
        'now': Finds the first business day on or after the formation date.

        Returns:
            List of rebalancing dates(as datetime objects).
        """
        if self._rebalancing_dates_cache is not None:
            return self._rebalancing_dates_cache

        bday_dates_np = np.array(self.bday_dates)
        res = []
        search_side: Literal['left',
                             'right'] = 'right' if self.rebal_timing == 'next' else 'left'

        for formation_date in self.formation_dates:
            idx = np.searchsorted(
                bday_dates_np, formation_date, side=search_side)

            if idx < len(bday_dates_np):
                rebalancing_date = bday_dates_np[idx]
            else:
                rebalancing_date = bday_dates_np[-1]
                logging.warning(f"Formation date {formation_date.strftime('%Y-%m-%d')} is after the last available price date "
                                f"{bday_dates_np[-1].strftime('%Y-%m-%d')}. Using the last price date as the rebalancing date.")

            res.append(pd.to_datetime(rebalancing_date))

        self._rebalancing_dates_cache = sorted(list(set(res)))
        logging.info(f"Calculated {len(self._rebalancing_dates_cache)} unique rebalancing dates using strategy '{self.rebal_timing}'. "
                     f"First 5: {[d.strftime('%Y-%m-%d') for d in self._rebalancing_dates_cache[:5]]}")
        return self._rebalancing_dates_cache

    @property
    def init_invest_date(self) -> str:
        """Returns the first rebalancing date formatted as YYYYMMDD."""
        return dt.datetime.strftime(self.rebalancing_dates[0], format='%Y%m%d')

    @property
    def last_invest_date(self) -> str:
        """Returns the last rebalancing date formatted as YYYYMMDD."""
        return dt.datetime.strftime(self.rebalancing_dates[-1], format='%Y%m%d')

    def get_quantity(self,
                     formation_date: dt.datetime,
                     rebalancing_date: dt.datetime,
                     total_assets: float) -> pd.Series:
        """
        Calculates the target quantity of shares for each stock based on weights and total assets.

        Filters out stocks with zero price or trading bans on the rebalancing date.

        Args:
            formation_date: The date the weights were determined.
            rebalancing_date: The date the portfolio is rebalanced and trades occur.
            total_assets: The total value of assets available for investment at rebalancing time.

        Returns:
            pd.Series: Target quantity for each stock, index = stock code, values = number of shares. Returns empty Series if no stocks qualify.
        """
        try:
            weights_ = self.weights.loc[formation_date].dropna()
            init_price_ = self.price.loc[rebalancing_date]
            init_trans_ban = self.trans_ban.loc[rebalancing_date]
        except KeyError as e:
            logging.error(
                f"Data missing for date {rebalancing_date} or {formation_date}. Error: {e}")
            return pd.Series(dtype=float)

        investable = weights_.index.intersection(
            init_price_[init_price_ > 0].index)
        investable = investable.intersection(
            init_trans_ban[init_trans_ban == 0].index)

        if investable.empty:
            logging.warning(
                f"{rebalancing_date}: No investable stocks found for formation date {formation_date} (positive price, no ban, non-NA weight).")
            return pd.Series(dtype=float)

        init_price = init_price_.loc[investable]
        weights = weights_.loc[investable]

        weights_sum = weights.sum()
        if abs(weights_sum) < 1e-9:
            logging.warning(
                f"{rebalancing_date}: Sum of weights for investable stocks is near zero ({weights_sum}). Cannot calculate quantities.")
            return pd.Series(dtype=float)
        weights /= weights_sum

        init_invest = weights * total_assets
        q = init_invest / init_price

        # FutureWarning 해결: replace 후 명시적으로 infer_objects() 호출
        q = q.replace([np.inf, -np.inf],
                      np.nan).dropna().infer_objects(copy=False)

        return q

    def pf_cashflow(self) -> pd.DataFrame:
        """
        Performs the core backtest simulation, calculating portfolio and cash values over time.

        Iterates through rebalancing periods, simulates trades considering costs and cash constraints,
        and calculates the daily portfolio value.

        Returns:
            pd.DataFrame: Time series of Portfolio value, Cash balance, and Total NAV. Index is Date.
        """
        print("\n***** Portfolio backtest simulation starting *****\n")
        rebalancing_dates = self.rebalancing_dates
        formation_dates = self.formation_dates

        date_map: Dict[dt.datetime, dt.datetime] = {}
        bday_dates_np = np.array(self.bday_dates)
        search_side: Literal['left',
                             'right'] = 'right' if self.rebal_timing == 'next' else 'left'

        temp_formation_dates = sorted(self.formation_dates)
        temp_rebalancing_dates = sorted(rebalancing_dates)

        current_formation_idx = 0
        for reb_date in temp_rebalancing_dates:
            associated_formation_date = None
            for form_idx in range(current_formation_idx, len(temp_formation_dates)):
                form_date = temp_formation_dates[form_idx]
                idx = np.searchsorted(
                    bday_dates_np, form_date, side=search_side)
                calculated_reb_date = bday_dates_np[idx] if idx < len(
                    bday_dates_np) else bday_dates_np[-1]

                if pd.to_datetime(calculated_reb_date) == reb_date:
                    associated_formation_date = form_date
                elif pd.to_datetime(calculated_reb_date) > reb_date:
                    break
            if associated_formation_date:
                date_map[reb_date] = associated_formation_date
            else:
                logging.error(
                    f"Could not find a formation date mapping to rebalancing date: {reb_date.strftime('%Y-%m-%d')}")
                possible_form_dates = [
                    fd for fd in temp_formation_dates if fd <= reb_date]
                if possible_form_dates:
                    fallback_form_date = max(possible_form_dates)
                    date_map[reb_date] = fallback_form_date
                    logging.warning(
                        f"Using fallback formation date {fallback_form_date.strftime('%Y-%m-%d')} for rebalancing date {reb_date.strftime('%Y-%m-%d')}")
                else:
                    raise ValueError(
                        f"Cannot determine formation date for rebalancing date {reb_date.strftime('%Y-%m%d')}")

        rebalancing_periods = len(rebalancing_dates)
        if rebalancing_periods == 0:
            logging.warning(
                "No rebalancing dates found. Returning empty DataFrame.")
            return pd.DataFrame(columns=['Portfolio', 'Cash', 'Total'], index=pd.to_datetime([]), dtype=float)

        portfolio_values_over_time = []
        self._transaction_summaries = []
        self._cash_balances = []
        self._final_holdings_data = []

        cash_balance = self.init_invest
        current_quantity = pd.Series(dtype=float)

        for period in tqdm(range(rebalancing_periods), desc="Backtesting Periods"):
            rebalancing_date = rebalancing_dates[period]
            formation_date = date_map.get(rebalancing_date)
            if formation_date is None:
                logging.error(
                    f"Critical error: Missing formation date for rebalancing date {rebalancing_date} in date_map.")
                continue

            if period < rebalancing_periods - 1:
                next_rebalancing_date = rebalancing_dates[period + 1]
            else:
                next_rebalancing_date = self.price.index[-1]
                if next_rebalancing_date < rebalancing_date:
                    logging.warning(
                        f"Last price date {next_rebalancing_date.strftime('%Y%m%d')} is before the last rebalancing date {rebalancing_date.strftime('%Y%m%d')}. Final period will have zero duration.")
                    next_rebalancing_date = rebalancing_date

            days_in_period = (
                next_rebalancing_date - rebalancing_date).days if next_rebalancing_date > rebalancing_date else 0
            interest = cash_balance * \
                (self.cash_rate / 365.0) * \
                days_in_period if days_in_period > 0 else 0.0
            cash_balance += interest
            logging.info(
                f"{rebalancing_date.strftime('%Y%m%d')}: Interest earned ({days_in_period} days): {interest:.2f}. Cash before rebalance: {cash_balance:.2f}")

            portfolio_value_before = 0.0
            if not current_quantity.empty:
                try:
                    current_prices = self.price.loc[rebalancing_date,
                                                    current_quantity.index]
                    current_prices = current_prices.fillna(0)
                    portfolio_value_before = np.dot(
                        current_quantity, current_prices)
                except KeyError as e:
                    logging.warning(
                        f"Price data missing for some held stocks {list(e.args[0])} on {rebalancing_date.strftime('%Y%m%d')}. Value calculated based on available prices.")
                    valid_index = current_quantity.index.intersection(
                        self.price.columns)
                    if not valid_index.empty:
                        current_prices = self.price.loc[rebalancing_date, valid_index].fillna(
                            0)
                        portfolio_value_before = np.dot(
                            current_quantity.loc[valid_index], current_prices)
                    else:
                        portfolio_value_before = 0.0

            total_assets_before_rebalance = portfolio_value_before + cash_balance
            logging.info(f"{rebalancing_date.strftime('%Y%m%d')}: Total assets BEFORE rebalance: {total_assets_before_rebalance:.2f} (Portfolio: {portfolio_value_before:.2f}, Cash: {cash_balance:.2f})")

            ideal_target_quantity = self.get_quantity(formation_date=formation_date,
                                                      rebalancing_date=rebalancing_date,
                                                      total_assets=total_assets_before_rebalance)

            if ideal_target_quantity.empty:
                logging.warning(
                    f"{rebalancing_date.strftime('%Y%m%d')}: Ideal target quantity is empty (no investable stocks or zero weights). Portfolio will be liquidated if holdings exist.")
                ideal_target_quantity = pd.Series(dtype=float)

            buy_value_ideal, sell_value_ideal, buy_cost_ideal, sell_cost_ideal, _, _ = self._calculate_transaction_costs(
                ideal_target_quantity, current_quantity, rebalancing_date, total_assets_before_rebalance, calculate_summary=False)

            cash_from_sells_ideal_net = sell_value_ideal - sell_cost_ideal
            cash_needed_for_buys_gross = buy_value_ideal + buy_cost_ideal
            cash_available_after_ideal_sells = cash_balance + cash_from_sells_ideal_net

            logging.info(f"{rebalancing_date.strftime('%Y%m%d')}: Ideal Trades - Sell Value: {sell_value_ideal:.2f}, Sell Cost: {sell_cost_ideal:.2f}, Net Cash from Sells: {cash_from_sells_ideal_net:.2f}")
            logging.info(f"{rebalancing_date.strftime('%Y%m%d')}: Ideal Trades - Buy Value: {buy_value_ideal:.2f}, Buy Cost: {buy_cost_ideal:.2f}, Gross Cash for Buys: {cash_needed_for_buys_gross:.2f}")
            logging.info(
                f"{rebalancing_date.strftime('%Y%m%d')}: Cash Available Post-Ideal Sells: {cash_available_after_ideal_sells:.2f}")

            final_target_quantity = ideal_target_quantity.copy()
            scale_factor = 1.0

            if cash_needed_for_buys_gross > cash_available_after_ideal_sells + 1e-6:
                logging.warning(
                    f"{rebalancing_date.strftime('%Y%m%d')}: Insufficient cash for ideal buys. Available: {cash_available_after_ideal_sells:.2f}, Needed: {cash_needed_for_buys_gross:.2f}. Scaling down buys.")

                if cash_needed_for_buys_gross <= 1e-6:
                    scale_factor = 0.0
                    logging.warning(
                        f"{rebalancing_date.strftime('%Y%m%d')}: Ideal buy value is zero or negative, cannot scale. Setting buy scale factor to 0.")
                elif cash_available_after_ideal_sells < 0:
                    scale_factor = 0.0
                    logging.warning(
                        f"{rebalancing_date.strftime('%Y%m%d')}: Cash available after ideal sells is negative ({cash_available_after_ideal_sells:.2f}). Setting buy scale factor to 0.")
                else:
                    scale_factor = cash_available_after_ideal_sells / cash_needed_for_buys_gross
                    scale_factor = max(0.0, min(1.0, scale_factor))
                    logging.info(
                        f"{rebalancing_date.strftime('%Y%m%d')}: Buy scaling factor calculated: {scale_factor:.6f}")

                union_index = ideal_target_quantity.index.union(
                    current_quantity.index)
                ideal_target_aligned = ideal_target_quantity.reindex(
                    union_index, fill_value=0.0)
                current_aligned = current_quantity.reindex(
                    union_index, fill_value=0.0)

                net_trades_ideal = ideal_target_aligned - current_aligned

                final_target_quantity = (
                    current_aligned + net_trades_ideal * scale_factor).round(8)

            else:
                logging.info(
                    f"{rebalancing_date.strftime('%Y%m%d')}: Ideal trades are affordable.")
                final_target_quantity = ideal_target_quantity

            buy_value, sell_value, buy_cost, sell_cost, total_transaction_cost, transaction_summary = self._calculate_transaction_costs(
                final_target_quantity, current_quantity, rebalancing_date, total_assets_before_rebalance, calculate_summary=True)

            if transaction_summary:
                self._transaction_summaries.append(transaction_summary)
                logging.info(f"{rebalancing_date.strftime('%Y%m%d')}: Final Trades - Sell Value: {sell_value:.2f}, Sell Cost: {sell_cost:.2f}, Buy Value: {buy_value:.2f}, Buy Cost: {buy_cost:.2f}, Total Cost: {total_transaction_cost:.2f}")
            else:
                logging.warning(
                    f"{rebalancing_date.strftime('%Y%m%d')}: Transaction summary not generated due to calculation errors.")
                buy_value, sell_value, buy_cost, sell_cost, total_transaction_cost = 0.0, 0.0, 0.0, 0.0, 0.0

            net_cash_flow_from_trades = (
                sell_value - sell_cost) - (buy_value + buy_cost)
            cash_balance += net_cash_flow_from_trades

            logging.info(
                f"{rebalancing_date.strftime('%Y%m%d')}: Cash Balance Updated. Net flow from trades: {net_cash_flow_from_trades:.2f}. Final Cash Balance: {cash_balance:.2f}")

            if cash_balance < 0:
                if cash_balance < -1.0:
                    logging.error(
                        f"{rebalancing_date.strftime('%Y%m%d')}: Significant negative cash balance detected ({cash_balance:,.2f}) after trades! Check scaling logic or costs.")
                    logging.warning(
                        "Clamping significantly negative cash balance to 0.0 to continue simulation, but results might be inaccurate.")
                else:
                    logging.warning(
                        f"{rebalancing_date.strftime('%Y%m%d')}: Near-zero negative cash balance ({cash_balance:.2f}) detected after trades. Clamping to 0.0")
                cash_balance = 0.0

            self._cash_balances.append(
                {'Date': rebalancing_date, 'Cash Balance': cash_balance, 'State': 'Post-Trade'})

            holding_period_dates = self.price.index[
                (self.price.index > rebalancing_date) & (
                    self.price.index <= next_rebalancing_date)
            ]

            if not final_target_quantity.empty and not holding_period_dates.empty:
                valid_cols = final_target_quantity.index.intersection(
                    self.price.columns)
                qty_filtered = final_target_quantity.reindex(
                    valid_cols).fillna(0)

                if not qty_filtered.empty:
                    price_period = self.price.loc[holding_period_dates, qty_filtered.index].fillna(
                        0)
                    daily_portfolio_values = price_period @ qty_filtered
                    period_cf = daily_portfolio_values.to_frame('Portfolio')
                else:
                    period_cf = pd.DataFrame(
                        0.0, index=holding_period_dates, columns=['Portfolio'])

                missing_cols = final_target_quantity.index.difference(
                    valid_cols)
                if not missing_cols.empty:
                    logging.warning(
                        f"Stocks {missing_cols.tolist()} in target quantity not found in price data columns during holding period {rebalancing_date.strftime('%Y%m%d')} to {next_rebalancing_date.strftime('%Y%m%d')}. Excluded from value calculation.")

            else:
                period_cf = pd.DataFrame(
                    0.0, index=holding_period_dates, columns=['Portfolio'])

            portfolio_values_over_time.append(period_cf)

            current_quantity = final_target_quantity.copy()

            self._final_holdings_data.append({
                'Date': rebalancing_date,
                'Quantity': final_target_quantity.copy()
            })

            logging.info(
                f"--- Period End: {rebalancing_date.strftime('%Y%m%d')} (Held until {next_rebalancing_date.strftime('%Y%m%d')}) ---")

        if not portfolio_values_over_time:
            logging.warning(
                "No portfolio values were generated during the backtest.")
            if rebalancing_dates:
                start_date = rebalancing_dates[0]
                initial_cash = self.init_invest
                if self._cash_balances:
                    cash_at_start = [
                        cb['Cash Balance'] for cb in self._cash_balances if cb['Date'] == start_date]
                    if cash_at_start:
                        initial_cash = cash_at_start[0]

                initial_df = pd.DataFrame({'Portfolio': [0.0], 'Cash': [initial_cash], 'Total': [
                                          initial_cash]}, index=[start_date])
                initial_df.index.name = 'Date'
                return initial_df
            else:
                return pd.DataFrame(columns=['Portfolio', 'Cash', 'Total'], index=pd.to_datetime([]), dtype=float)

        results_df = pd.concat(portfolio_values_over_time)
        results_df.index.name = 'Date'

        cash_df = pd.DataFrame(self._cash_balances).set_index('Date')
        results_df['Cash'] = cash_df['Cash Balance'].reindex(
            results_df.index, method='ffill')

        first_recorded_cash_date = cash_df.index.min() if not cash_df.empty else None
        if first_recorded_cash_date is not None:
            results_df.loc[results_df.index <
                           first_recorded_cash_date, 'Cash'] = self.init_invest
        else:
            results_df['Cash'] = self.init_invest

        results_df['Cash'].fillna(method='ffill', inplace=True)
        results_df['Cash'].fillna(method='bfill', inplace=True)
        results_df['Cash'].fillna(self.init_invest, inplace=True)

        results_df['Total'] = results_df['Portfolio'] + results_df['Cash']

        self._res_quantity = self._final_holdings_data

        print("***** Portfolio backtest simulation finished *****\n")
        return results_df.sort_index()

    def _calculate_transaction_costs(self,
                                     target_quantity: pd.Series,
                                     previous_quantity: pd.Series,
                                     date: dt.datetime,
                                     total_assets_before_rebalance: float,
                                     calculate_summary: bool = True) -> Tuple[float, float, float, float, float, Optional[Dict]]:
        """
        Calculates buy/sell values and associated costs for transitioning from previous_quantity to target_quantity.

        Args:
            target_quantity: The desired quantity of shares after rebalancing.
            previous_quantity: The quantity of shares held before rebalancing.
            date: The date of the transaction.
            total_assets_before_rebalance: NAV just before these trades. Used for cost % calculation.
            calculate_summary: Whether to generate a dictionary summarizing the transaction details.

        Returns:
            Tuple containing:
            - Total value of assets bought(excluding costs).
            - Total value of assets sold(excluding costs).
            - Total cost associated with buying(commission + slippage).
            - Total cost associated with selling(commission + slippage + tax).
            - Total transaction cost(buy_cost + sell_cost).
            - Dictionary with transaction summary if calculate_summary is True, else None.
        """
        target_qty = target_quantity if isinstance(
            target_quantity, pd.Series) else pd.Series(dtype=float)
        prev_qty = previous_quantity if isinstance(
            previous_quantity, pd.Series) else pd.Series(dtype=float)

        all_involved_stocks = target_qty.index.union(prev_qty.index)

        if all_involved_stocks.empty:
            summary = self._generate_empty_summary(
                date, total_assets_before_rebalance) if calculate_summary else None
            return 0.0, 0.0, 0.0, 0.0, 0.0, summary

        try:
            if date not in self.price.index:
                raise KeyError(
                    f"Date {date.strftime('%Y-%m-%d')} not found in price index.")
            missing_cols = all_involved_stocks.difference(self.price.columns)
            if not missing_cols.empty:
                logging.warning(
                    f"Price data missing for stocks {missing_cols.tolist()} on {date.strftime('%Y-%m-%d')}. They will be excluded from cost calculation.")
                all_involved_stocks = all_involved_stocks.intersection(
                    self.price.columns)
                if all_involved_stocks.empty:
                    raise ValueError(
                        f"No price data available for any involved stock on {date.strftime('%Y-%m-%d')}.")

            price_at_date = self.price.loc[date, all_involved_stocks].fillna(0)
            if (price_at_date <= 0).any():
                zero_price_stocks = price_at_date[price_at_date <= 0].index.tolist(
                )
                logging.warning(
                    f"Stocks {zero_price_stocks} have zero or negative price on {date.strftime('%Y-%m-%d')}. Excluding from trade calculations.")
                all_involved_stocks = all_involved_stocks.difference(
                    zero_price_stocks)
                price_at_date = price_at_date.loc[all_involved_stocks]
                if all_involved_stocks.empty:
                    raise ValueError(
                        f"All involved stocks have zero/negative price on {date.strftime('%Y-%m-%d')}.")

        except (KeyError, ValueError) as e:
            logging.error(
                f"Error fetching price data for transaction cost calculation on {date.strftime('%Y-%m-%d')}: {e}")
            summary = self._generate_error_summary(
                date, total_assets_before_rebalance, f"Price data error: {e}") if calculate_summary else None
            return 0.0, 0.0, 0.0, 0.0, 0.0, summary

        target_aligned = target_qty.reindex(
            all_involved_stocks, fill_value=0.0)
        prev_aligned = prev_qty.reindex(all_involved_stocks, fill_value=0.0)

        net_trades = target_aligned - prev_aligned
        net_trades = net_trades[net_trades.abs() > 1e-8]

        if net_trades.empty:
            summary = self._generate_empty_summary(
                date, total_assets_before_rebalance) if calculate_summary else None
            return 0.0, 0.0, 0.0, 0.0, 0.0, summary

        price_trades = price_at_date.reindex(net_trades.index)

        buys_qty = net_trades[net_trades > 0]
        sells_qty = -net_trades[net_trades < 0]

        buy_value = (buys_qty * price_trades.loc[buys_qty.index]).sum()
        sell_value = (sells_qty * price_trades.loc[sells_qty.index]).sum()

        buy_cost = (self.buy_commission + self.slippage) * buy_value
        sell_cost = (self.sell_commission + self.slippage +
                     self.sell_tax) * sell_value
        total_cost = buy_cost + sell_cost

        shares_bought = buys_qty.sum()
        shares_sold = sells_qty.sum()

        cost_bp = (total_cost / total_assets_before_rebalance) * \
            10000 if total_assets_before_rebalance > 1e-6 else 0.0

        summary = None
        if calculate_summary:
            summary = {
                'Date': date,
                'Total Buy Value': buy_value,
                'Total Sell Value': sell_value,
                'Shares Bought': shares_bought,
                'Shares Sold': shares_sold,
                'Total Buy Cost': buy_cost,
                'Total Sell Cost': sell_cost,
                'Total Transaction Cost': total_cost,
                'Total NAV': total_assets_before_rebalance,
                'Transaction Cost (bp)': cost_bp,
                'Error': None
            }

        return buy_value, sell_value, buy_cost, sell_cost, total_cost, summary

    def _generate_empty_summary(self, date: dt.datetime, nav: float) -> Dict:
        """Generates a summary dictionary for a day with no trades."""
        return {
            'Date': date, 'Total Buy Value': 0.0, 'Total Sell Value': 0.0,
            'Shares Bought': 0.0, 'Shares Sold': 0.0, 'Total Buy Cost': 0.0,
            'Total Sell Cost': 0.0, 'Total Transaction Cost': 0.0,
            'Total NAV': nav, 'Transaction Cost (bp)': 0.0, 'Error': 'No trades'
        }

    def _generate_error_summary(self, date: dt.datetime, nav: float, error_msg: str) -> Dict:
        """Generates a summary dictionary when an error occurs during calculation."""
        return {
            'Date': date, 'Total Buy Value': 0.0, 'Total Sell Value': 0.0,
            'Shares Bought': 0.0, 'Shares Sold': 0.0, 'Total Buy Cost': 0.0,
            'Total Sell Cost': 0.0, 'Total Transaction Cost': 0.0,
            'Total NAV': nav, 'Transaction Cost (bp)': 0.0, 'Error': error_msg
        }

    @property
    def transaction_costs_summary(self) -> pd.DataFrame:
        """Returns a DataFrame summarizing transaction details for each rebalancing date.
        Raises:
            ValueError: If pf_cashflow() has not been run yet.

        Returns:
            pd.DataFrame: Summary of transactions(values, costs, shares, etc.).
        """
        if not self._transaction_summaries:
            if self._cash_balances is not None:
                logging.warning(
                    "pf_cashflow() was called, but no transaction summaries were generated (possibly due to errors or no trades). Returning empty summary.")
                return pd.DataFrame(columns=['Date', 'Total Buy Value', 'Total Sell Value', 'Shares Bought', 'Shares Sold', 'Total Buy Cost', 'Total Sell Cost', 'Total Transaction Cost', 'Total NAV', 'Transaction Cost (bp)', 'Error']).set_index('Date')
            else:
                raise ValueError(
                    "Transaction costs have not been calculated. Execute pf_cashflow() first.")
        valid_summaries = [
            s for s in self._transaction_summaries if s is not None]
        if not valid_summaries:
            logging.warning(
                "No valid transaction summaries found. Returning empty DataFrame.")
            return pd.DataFrame(columns=['Date', 'Total Buy Value', 'Total Sell Value', 'Shares Bought', 'Shares Sold', 'Total Buy Cost', 'Total Sell Cost', 'Total Transaction Cost', 'Total NAV', 'Transaction Cost (bp)', 'Error']).set_index('Date')

        return pd.DataFrame(valid_summaries).set_index('Date')

    @property
    def cash_balance_summary(self) -> pd.DataFrame:
        """
        Returns a DataFrame showing the cash balance after each rebalancing.

        Raises:
            ValueError: If pf_cashflow() has not been run yet.

        Returns:
            pd.DataFrame: Time series of cash balance, indexed by Date.
        """
        if not self._cash_balances:
            if self._transaction_summaries is not None:
                logging.warning(
                    "pf_cashflow() was called, but no cash balances were recorded. Returning empty summary.")
                return pd.DataFrame(columns=['Cash Balance'], index=pd.to_datetime([]))
            else:
                raise ValueError(
                    "Cash balances have not been calculated. Execute pf_cashflow() first.")
        return pd.DataFrame(self._cash_balances).set_index('Date')

    @property
    def pf_ret(self) -> pd.DataFrame:
        """
        Calculates the daily percentage returns of the total portfolio NAV.

        Calls pf_cashflow() if results are not already computed.

        Returns:
            pd.DataFrame: Daily returns ('Total' column), indexed by Date.
        """
        if not self._cash_balances:
            logging.info(
                "Cash balances not found, running pf_cashflow() to calculate returns.")
            cashflow_df = self.pf_cashflow()
        else:
            cashflow_df = self.pf_cashflow()

        if cashflow_df.empty:
            logging.warning(
                "Cashflow calculation resulted in an empty DataFrame. Cannot calculate returns.")
            return pd.DataFrame(columns=['Total'], index=pd.to_datetime([]), dtype=float)

        pf_ret = pd.DataFrame(cashflow_df['Total'].pct_change())
        pf_ret.index = pd.to_datetime(cashflow_df.index)
        pf_ret.index.name = 'Date'
        return pf_ret

    @property
    def bm_ret(self) -> pd.DataFrame:
        """
        Calculates the daily percentage returns of the benchmark index.

        Returns:
            pd.DataFrame: Daily benchmark returns, indexed by Date.
        """
        try:
            start_date = self.rebalancing_dates[0]
            if start_date not in self.bm.index:
                available_bm_dates = self.bm.index[self.bm.index >= start_date]
                if available_bm_dates.empty:
                    logging.error(
                        f"Benchmark data does not cover the backtest start date {start_date}. Cannot calculate benchmark returns.")
                    return pd.DataFrame(columns=[self.bm.columns[0] if not self.bm.empty else 'Benchmark'], index=pd.to_datetime([]), dtype=float)
                start_date = available_bm_dates[0]
                logging.warning(
                    f"Benchmark data starts at {start_date}, using this as the start for benchmark returns.")

            bm_relevant = self.bm.loc[start_date:]
            if bm_relevant.empty or len(bm_relevant) < 2:
                logging.warning(
                    "Not enough benchmark data points to calculate returns.")
                return pd.DataFrame(columns=[self.bm.columns[0] if not self.bm.empty else 'Benchmark'], index=pd.to_datetime([]), dtype=float)

            bm_ret = bm_relevant.pct_change()
            bm_ret.index = pd.to_datetime(bm_relevant.index)
            bm_ret.index.name = 'Date'
            return bm_ret

        except Exception as e:
            logging.error(f"Error calculating benchmark returns: {e}")
            return pd.DataFrame(columns=[self.bm.columns[0] if not self.bm.empty else 'Benchmark'], index=pd.to_datetime([]), dtype=float)

    @property
    def pf_quantity(self) -> pd.DataFrame:
        """Returns a DataFrame containing the actual portfolio quantities over time."""
        if not self._final_holdings_data:
            logging.warning(
                "No holdings data available. Run pf_cashflow() first.")
            return pd.DataFrame()

        quantities_list = []
        for item in self._final_holdings_data:
            date = item['Date']
            qty = item['Quantity']
            qty_series = pd.Series(qty, name=date)
            quantities_list.append(qty_series)

        if not quantities_list:
            return pd.DataFrame()

        return pd.concat(quantities_list, axis=1).T

    def get_holdings_snapshot(self) -> Dict[str, pd.DataFrame]:
        """
        Returns a dictionary mapping rebalancing dates to DataFrames containing portfolio holdings information.
        Each DataFrame includes columns for quantity, price, market value, and weight for each stock.

        Uses the actual final quantities held after accounting for cash constraints and scaling.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping dates (YYYYMMDD format) to holdings DataFrames
        """
        if not self._final_holdings_data:
            logging.warning(
                "No holdings data available. Run pf_cashflow() first.")
            return {}

        holdings = {}
        for item in self._final_holdings_data:
            date = item['Date']
            quantities = item['Quantity']

            if quantities.empty:
                continue

            date_str = dt.datetime.strftime(date, format='%Y%m%d')
            prices = self.price.loc[date]

            valid_stocks = quantities.index.intersection(prices.index)
            if valid_stocks.empty:
                logging.warning(
                    f"No valid stocks with price data found for date {date_str}")
                continue

            holdings_df = pd.DataFrame({
                'quantity': quantities.loc[valid_stocks],
                'price': prices.loc[valid_stocks],
            })

            holdings_df['market_value'] = holdings_df['quantity'] * \
                holdings_df['price']
            holdings_df = holdings_df[holdings_df['market_value'] > 0]

            if holdings_df.empty:
                continue

            total_market_value = holdings_df['market_value'].sum()
            holdings_df['weight'] = holdings_df['market_value'] / \
                total_market_value if total_market_value > 0 else 0

            holdings_df = holdings_df.sort_values(
                'market_value', ascending=False)
            holdings[date_str] = holdings_df
        return holdings
