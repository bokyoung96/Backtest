import logging
import numpy as np
import pandas as pd
import datetime as dt
from tqdm import tqdm
from typing import Literal, List, Dict, Tuple, Optional

from bt.loader import DataLoader

log_filename = 'backtest.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, mode='w'),
        logging.StreamHandler()
    ]
)


class DateManager:
    """
    Manages calculation and retrieval of dates relevant for backtesting.
    """

    def __init__(self,
                 formation_dates_index: pd.DatetimeIndex,
                 bday_dates_index: pd.DatetimeIndex,
                 rebal_timing: Literal['next', 'now']):
        """
        Initializes the DateManager.

        Args:
            formation_dates_index: Index of the weights DataFrame.
            bday_dates_index: Index of the price DataFrame (business days).
            rebal_timing: Rebalancing timing strategy ('next' or 'now').
        """
        self._formation_dates_pd = pd.to_datetime(formation_dates_index)
        self._bday_dates_pd = pd.to_datetime(bday_dates_index)
        self._rebal_timing = rebal_timing

        self._bday_dates_np = np.array(
            self._bday_dates_pd, dtype='datetime64[ns]')
        self._rebalancing_dates_cache: Optional[List[dt.datetime]] = None
        self._date_map_cache: Optional[Dict[dt.datetime, dt.datetime]] = None

        self.rebalancing_dates
        self.date_map

    @property
    def formation_dates(self) -> List[dt.datetime]:
        """Returns a sorted list of formation dates."""
        return sorted(list(self._formation_dates_pd))

    @property
    def bday_dates(self) -> List[dt.datetime]:
        """Returns a sorted list of business days."""
        return sorted(list(self._bday_dates_pd))

    @property
    def rebalancing_dates(self) -> List[dt.datetime]:
        """
        Calculates and caches the unique, sorted rebalancing dates.
        """
        if self._rebalancing_dates_cache is not None:
            return self._rebalancing_dates_cache

        res = []
        search_side: Literal['left',
                             'right'] = 'right' if self._rebal_timing == 'next' else 'left'

        for formation_date in self.formation_dates:
            formation_date_np = np.datetime64(formation_date)
            idx = np.searchsorted(self._bday_dates_np,
                                  formation_date_np, side=search_side)

            if idx < len(self._bday_dates_np):
                rebalancing_date = self._bday_dates_np[idx]
            else:
                rebalancing_date = self._bday_dates_np[-1]
                logging.warning(f"[DateManager] Formation date {formation_date.strftime('%Y-%m-%d')} is after the last available price date "
                                f"{self._bday_dates_np[-1].strftime('%Y-%m-%d')}. Using the last price date as the rebalancing date.")

            res.append(pd.to_datetime(rebalancing_date))

        self._rebalancing_dates_cache = sorted(list(set(res)))
        logging.info(f"[DateManager] Calculated {len(self._rebalancing_dates_cache)} unique rebalancing dates using strategy '{self._rebal_timing}'. "
                     f"First 5: {[d.strftime('%Y-%m-%d') for d in self._rebalancing_dates_cache[:5]]}")
        return self._rebalancing_dates_cache

    @property
    def date_map(self) -> Dict[dt.datetime, dt.datetime]:
        """
        Calculates and caches the mapping from rebalancing dates to their corresponding formation dates.
        Mirrors the logic originaly in pf_cashflow.
        """
        if self._date_map_cache is not None:
            return self._date_map_cache

        date_map: Dict[dt.datetime, dt.datetime] = {}
        search_side: Literal['left',
                             'right'] = 'right' if self._rebal_timing == 'next' else 'left'
        temp_formation_dates = self.formation_dates
        temp_rebalancing_dates = self.rebalancing_dates

        formation_to_rebal_map = {}
        for form_date in temp_formation_dates:
            form_date_np = np.datetime64(form_date)
            idx = np.searchsorted(self._bday_dates_np,
                                  form_date_np, side=search_side)
            calculated_reb_date = self._bday_dates_np[idx] if idx < len(
                self._bday_dates_np) else self._bday_dates_np[-1]
            formation_to_rebal_map[form_date] = pd.to_datetime(
                calculated_reb_date)

        current_formation_idx = 0
        for reb_date in temp_rebalancing_dates:
            associated_formation_date = None
            possible_formation_dates_for_reb = []

            for form_date, calculated_reb_date in formation_to_rebal_map.items():
                if calculated_reb_date == reb_date:
                    possible_formation_dates_for_reb.append(form_date)

            if possible_formation_dates_for_reb:
                associated_formation_date = max(
                    possible_formation_dates_for_reb)
                date_map[reb_date] = associated_formation_date
            else:
                fallback_candidates = [
                    fd for fd in temp_formation_dates
                    if formation_to_rebal_map.get(fd, pd.Timestamp.max) <= reb_date
                ]
                if fallback_candidates:
                    fallback_form_date = max(fallback_candidates)
                    date_map[reb_date] = fallback_form_date
                    logging.warning(f"[DateManager] Could not find a direct formation date mapping to rebalancing date: {reb_date.strftime('%Y-%m-%d')}. "
                                    f"Using fallback formation date: {fallback_form_date.strftime('%Y-%m-%d')}")
                else:
                    logging.error(
                        f"[DateManager] Cannot determine formation date for rebalancing date {reb_date.strftime('%Y-%m-%d')}. This might indicate an issue.")
                    date_map[reb_date] = None

        self._date_map_cache = date_map
        return self._date_map_cache

    @property
    def init_invest_date_str(self) -> Optional[str]:
        """Returns the first rebalancing date formatted as YYYYMMDD, or None if no dates."""
        if not self.rebalancing_dates:
            return None
        return dt.datetime.strftime(self.rebalancing_dates[0], format='%Y%m%d')

    @property
    def last_invest_date_str(self) -> Optional[str]:
        """Returns the last rebalancing date formatted as YYYYMMDD, or None if no dates."""
        if not self.rebalancing_dates:
            return None
        return dt.datetime.strftime(self.rebalancing_dates[-1], format='%Y%m%d')


class TransactionCostCalculator:
    """
    Calculates transaction costs based on portfolio changes.
    """

    def __init__(self,
                 buy_commission: float = 0.0,
                 sell_commission: float = 0.0,
                 slippage: float = 0.0,
                 sell_tax: float = 0.0):
        """
        Initializes the calculator with cost parameters.
        """
        self.buy_commission = buy_commission
        self.sell_commission = sell_commission
        self.slippage = slippage
        self.sell_tax = sell_tax

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

    def calculate_costs(self,
                        target_quantity: pd.Series,
                        previous_quantity: pd.Series,
                        date: dt.datetime,
                        price_data_full: pd.DataFrame,
                        total_assets_before_rebalance: float,
                        calculate_summary: bool = True) -> Tuple[float, float, float, float, float, Optional[Dict]]:
        """
        Calculates buy/sell values and associated costs for transitioning.

        Args:
            target_quantity: The desired quantity of shares after rebalancing.
            previous_quantity: The quantity of shares held before rebalancing.
            date: The date of the transaction.
            price_data_full: The full price DataFrame to extract prices for the given date.
            total_assets_before_rebalance: NAV just before these trades. Used for cost % calculation.
            calculate_summary: Whether to generate a dictionary summarizing the transaction details.

        Returns:
            Tuple containing:
            - Total value of assets bought (excluding costs).
            - Total value of assets sold (excluding costs).
            - Total cost associated with buying (commission + slippage).
            - Total cost associated with selling (commission + slippage + tax).
            - Total transaction cost (buy_cost + sell_cost).
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
            if date not in price_data_full.index:
                raise KeyError(
                    f"Date {date.strftime('%Y-%m-%d')} not found in price index.")

            missing_cols = all_involved_stocks.difference(
                price_data_full.columns)
            if not missing_cols.empty:
                logging.warning(
                    f"[CostCalc] Price data missing for stocks {missing_cols.tolist()} on {date.strftime('%Y-%m-%d')}. They will be excluded from cost calculation.")
                all_involved_stocks = all_involved_stocks.intersection(
                    price_data_full.columns)
                if all_involved_stocks.empty:
                    raise ValueError(
                        f"No price data available for any involved stock on {date.strftime('%Y-%m-%d')}.")

            price_at_date = price_data_full.loc[date, all_involved_stocks].fillna(
                0)

            if (price_at_date <= 0).any():
                zero_price_stocks = price_at_date[price_at_date <= 0].index.tolist(
                )
                logging.warning(
                    f"[CostCalc] Stocks {zero_price_stocks} have zero or negative price on {date.strftime('%Y-%m-%d')}. Excluding from trade calculations.")
                all_involved_stocks = all_involved_stocks.difference(
                    zero_price_stocks)
                price_at_date = price_at_date.loc[all_involved_stocks]
                if all_involved_stocks.empty:
                    raise ValueError(
                        f"All involved stocks have zero/negative price on {date.strftime('%Y-%m-%d')}.")

        except (KeyError, ValueError) as e:
            logging.error(
                f"[CostCalc] Error fetching price data for transaction cost calculation on {date.strftime('%Y-%m-%d')}: {e}")
            summary = self._generate_error_summary(
                date, total_assets_before_rebalance, f"Price data error: {e}") if calculate_summary else None
            return 0.0, 0.0, 0.0, 0.0, 0.0, summary

        target_aligned = target_qty.reindex(
            all_involved_stocks, fill_value=0.0)
        prev_aligned = prev_qty.reindex(all_involved_stocks, fill_value=0.0)

        net_trades = target_aligned - prev_aligned
        net_trades = net_trades[net_trades.abs() > 1e-9]

        if net_trades.empty:
            summary = self._generate_empty_summary(
                date, total_assets_before_rebalance) if calculate_summary else None
            return 0.0, 0.0, 0.0, 0.0, 0.0, summary

        price_trades = price_at_date.loc[net_trades.index]

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


class PortfolioBacktester:
    """
    Constructs and backtests a portfolio based on given weights and market data.
    Uses DateManager and TransactionCostCalculator.
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
                 rebal_timing: Literal['next', 'now'] = 'next'):
        """
        Initializes the PortfolioBacktester.
        """
        self.mkt = mkt
        self.weights = weights
        self.init_invest = init_invest
        self.cash_rate = cash_rate
        self.rebal_timing = rebal_timing

        data_loader = DataLoader(mkt=mkt)
        self.price = data_loader(data_name='price_adj').fillna(0)
        self.bm = data_loader(data_name='bm')
        self.trans_ban = data_loader(data_name='trans_ban')
        self.sector = data_loader(data_name='wics_sector_big')

        self.date_manager = DateManager(formation_dates_index=self.weights.index,
                                        bday_dates_index=self.price.index,
                                        rebal_timing=self.rebal_timing)

        self.cost_calculator = TransactionCostCalculator(buy_commission=buy_commission,
                                                         sell_commission=sell_commission,
                                                         slippage=slippage,
                                                         sell_tax=sell_tax)

        self._results_df: Optional[pd.DataFrame] = None
        self._transaction_summaries: List[Dict] = []
        self._cash_balances: List[Dict] = []
        self._final_holdings_data: List[Dict] = []

    def __repr__(self) -> str:
        init_date = self.date_manager.init_invest_date_str or "N/A"
        last_date = self.date_manager.last_invest_date_str or "N/A"
        params = (
            f"init_invest: {self.init_invest}\\n"
            f"mkt: {self.mkt}\\n"
            f"init_invest_date: {init_date}\\n"
            f"last_invest_date: {last_date}\\n"
            f"rebal_timing: {self.rebal_timing}"
        )
        return params

    def get_target_quantity(self,
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
            pd.Series: Target quantity for each stock. Returns empty Series if no stocks qualify or data is missing.
        """
        if formation_date not in self.weights.index:
            logging.error(
                f"[QtyCalc] Formation date {formation_date} not found in weights index.")
            return pd.Series(dtype=float)
        if rebalancing_date not in self.price.index:
            logging.error(
                f"[QtyCalc] Rebalancing date {rebalancing_date} not found in price index.")
            return pd.Series(dtype=float)
        if rebalancing_date not in self.trans_ban.index:
            logging.error(
                f"[QtyCalc] Rebalancing date {rebalancing_date} not found in trans_ban index.")
            return pd.Series(dtype=float)

        try:
            weights_ = self.weights.loc[formation_date].dropna()
            init_price_ = self.price.loc[rebalancing_date]
            init_trans_ban = self.trans_ban.loc[rebalancing_date]

            investable_price = init_price_[init_price_ > 0].index
            investable_ban = init_trans_ban[init_trans_ban == 0].index
            investable = weights_.index.intersection(
                investable_price).intersection(investable_ban)

            if investable.empty:
                logging.warning(f"[QtyCalc] {rebalancing_date}: No investable stocks found for formation date {formation_date} "
                                "(positive price, no ban, non-NA weight).")
                return pd.Series(dtype=float)

            init_price = init_price_.loc[investable]
            weights = weights_.loc[investable]

            weights_sum = weights.sum()
            if abs(weights_sum) < 1e-9:
                logging.warning(
                    f"[QtyCalc] {rebalancing_date}: Sum of weights for investable stocks is near zero ({weights_sum}). Cannot calculate quantities.")
                return pd.Series(dtype=float)

            target_investment = weights * total_assets
            q = target_investment / init_price.replace(0, np.nan)
            q = q.replace([np.inf, -np.inf], np.nan).dropna()

            return q.astype(float)

        except KeyError as e:
            logging.error(
                f"[QtyCalc] Unexpected KeyError accessing data for date {rebalancing_date} or {formation_date}. Error: {e}")
            return pd.Series(dtype=float)
        except Exception as e:
            logging.error(
                f"[QtyCalc] Error calculating quantity for {rebalancing_date} (Formation: {formation_date}): {e}")
            return pd.Series(dtype=float)

    def run_backtest(self) -> pd.DataFrame:
        """
        Performs the core backtest simulation, calculating portfolio and cash values over time.
        """
        if self._results_df is not None:
            logging.info("Backtest already run. Returning cached results.")
            return self._results_df

        print("\n***** Portfolio backtest simulation starting *****\n")
        rebalancing_dates = self.date_manager.rebalancing_dates
        date_map = self.date_manager.date_map

        rebalancing_periods = len(rebalancing_dates)
        if rebalancing_periods == 0:
            logging.warning(
                "[Backtester] No rebalancing dates found. Returning empty DataFrame.")
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

            # NOTE: In progress if formation_date is None or formation_date not in self.weights.index
            if formation_date is None or formation_date not in self.weights.index:
                logging.error(
                    f"[Backtester] Critical error: Invalid or missing formation date ({formation_date}) for rebalancing date {rebalancing_date} in date_map. Skipping period.")

                if period < rebalancing_periods - 1:
                    next_rebalancing_date_for_interest = rebalancing_dates[period+1]
                else:
                    next_rebalancing_date_for_interest = self.price.index[-1]
                    if next_rebalancing_date_for_interest < rebalancing_date:
                        next_rebalancing_date_for_interest = rebalancing_date

                days_in_skipped_period = (next_rebalancing_date_for_interest -
                                          rebalancing_date).days if next_rebalancing_date_for_interest > rebalancing_date else 0
                interest = cash_balance * \
                    (self.cash_rate / 365.0) * \
                    days_in_skipped_period if days_in_skipped_period > 0 else 0.0
                cash_balance += interest
                logging.info(
                    f"[Backtester] {rebalancing_date.strftime('%Y%m%d')} (SKIPPED): Interest earned ({days_in_skipped_period} days): {interest:.2f}. Cash: {cash_balance:.2f}")

                self._cash_balances.append(
                    {'Date': rebalancing_date, 'Cash Balance': cash_balance, 'State': 'Post-Interest (Skipped)'})
                holding_period_dates_skipped = self.price.index[
                    (self.price.index > rebalancing_date) & (
                        self.price.index <= next_rebalancing_date_for_interest)
                ]
                portfolio_values_over_time.append(pd.DataFrame(
                    0.0, index=holding_period_dates_skipped, columns=['Portfolio']))
                self._final_holdings_data.append(
                    {'Date': rebalancing_date, 'Quantity': current_quantity.copy()})
                continue

            # NOTE: In progress if required datas are aligned
            if period < rebalancing_periods - 1:
                next_rebalancing_date = rebalancing_dates[period + 1]
            else:
                next_rebalancing_date = self.price.index[-1]
                if next_rebalancing_date < rebalancing_date:
                    logging.warning(
                        f"[Backtester] Last price date {next_rebalancing_date.strftime('%Y%m%d')} is before the last rebalancing date {rebalancing_date.strftime('%Y%m%d')}. Final period will have zero duration.")
                    next_rebalancing_date = rebalancing_date

            days_in_period = (
                next_rebalancing_date - rebalancing_date).days if next_rebalancing_date > rebalancing_date else 0
            interest = cash_balance * \
                (self.cash_rate / 365.0) * \
                days_in_period if days_in_period > 0 else 0.0
            cash_balance += interest
            logging.info(
                f"[Backtester] {rebalancing_date.strftime('%Y%m%d')}: Interest earned ({days_in_period} days): {interest:.2f}. Cash before rebalance: {cash_balance:.2f}")
            self._cash_balances.append(
                {'Date': rebalancing_date, 'Cash Balance': cash_balance, 'State': 'Post-Interest'})

            portfolio_value_before = 0.0
            if not current_quantity.empty:
                try:
                    if rebalancing_date not in self.price.index:
                        raise KeyError(
                            f"Price data missing for rebalancing date {rebalancing_date.strftime('%Y%m%d')}")
                    valid_held_stocks = current_quantity.index.intersection(
                        self.price.columns)
                    if not valid_held_stocks.empty:
                        current_prices = self.price.loc[rebalancing_date, valid_held_stocks].fillna(
                            0)
                        portfolio_value_before = np.dot(
                            current_quantity.loc[valid_held_stocks], current_prices)
                    else:
                        portfolio_value_before = 0.0

                    missing_price_for_held = current_quantity.index.difference(
                        valid_held_stocks)
                    if not missing_price_for_held.empty:
                        logging.warning(
                            f"[Backtester] Price data missing for some currently held stocks {missing_price_for_held.tolist()} on {rebalancing_date.strftime('%Y%m%d')}. Value calculated based on available prices.")

                except KeyError as e:
                    logging.error(
                        f"[Backtester] Error getting prices for current holdings on {rebalancing_date.strftime('%Y%m%d')}: {e}. Portfolio value assumed 0 for this calculation.")
                    portfolio_value_before = 0.0
                except Exception as e:
                    logging.error(
                        f"[Backtester] Unexpected error calculating portfolio value before rebalance on {rebalancing_date.strftime('%Y%m%d')}: {e}")
                    portfolio_value_before = 0.0

            total_assets_before_rebalance = portfolio_value_before + cash_balance
            logging.info(
                f"[Backtester] {rebalancing_date.strftime('%Y%m%d')}: Total assets BEFORE rebalance: {total_assets_before_rebalance:.2f} (Portfolio: {portfolio_value_before:.2f}, Cash: {cash_balance:.2f})")

            ideal_target_quantity = self.get_target_quantity(formation_date=formation_date,
                                                             rebalancing_date=rebalancing_date,
                                                             total_assets=total_assets_before_rebalance)

            # NOTE: For banned stocks, if in portfolio, maintain current position
            banned_stocks = []
            if not current_quantity.empty and rebalancing_date in self.trans_ban.index:
                banned_stocks = self.trans_ban.loc[rebalancing_date][self.trans_ban.loc[rebalancing_date] > 0].index
                banned_holdings = current_quantity.index.intersection(banned_stocks)
                
                if not banned_holdings.empty:
                    for stock in banned_holdings:
                        ideal_target_quantity[stock] = current_quantity[stock]
                    
                    logging.warning(
                        f"[Backtester] {rebalancing_date.strftime('%Y%m%d')}: Found {len(banned_holdings)} banned stocks in current holdings. Maintaining current positions for: {banned_holdings.tolist()}")

            if ideal_target_quantity.empty and len(banned_stocks) == 0:
                logging.warning(
                    f"[Backtester] {rebalancing_date.strftime('%Y%m%d')}: Ideal target quantity is empty. Portfolio will be liquidated if holdings exist.")
                ideal_target_quantity = pd.Series(dtype=float)

            buy_val_ideal, sell_val_ideal, buy_cost_ideal, sell_cost_ideal, _, _ = self.cost_calculator.calculate_costs(
                ideal_target_quantity, current_quantity, rebalancing_date, self.price, total_assets_before_rebalance, calculate_summary=False)

            cash_from_sells_ideal_net = sell_val_ideal - sell_cost_ideal
            cash_needed_for_buys_gross = buy_val_ideal + buy_cost_ideal
            cash_available_after_ideal_sells = cash_balance + cash_from_sells_ideal_net

            logging.info(
                f"[Backtester] {rebalancing_date.strftime('%Y%m%d')}: Ideal Trades - Sell Value: {sell_val_ideal:.2f}, Sell Cost: {sell_cost_ideal:.2f}, Net Cash from Sells: {cash_from_sells_ideal_net:.2f}")
            logging.info(
                f"[Backtester] {rebalancing_date.strftime('%Y%m%d')}: Ideal Trades - Buy Value: {buy_val_ideal:.2f}, Buy Cost: {buy_cost_ideal:.2f}, Gross Cash for Buys: {cash_needed_for_buys_gross:.2f}")
            logging.info(
                f"[Backtester] {rebalancing_date.strftime('%Y%m%d')}: Cash Available Post-Ideal Sells: {cash_available_after_ideal_sells:.2f}")

            # NOTE: Scale factor explanation:
            # 1. Scale factor is used to adjust buy trades when there is insufficient cash
            # 2. Scale factor = Available Cash / Required Cash for Buys (capped between 0 and 1)
            # 3. Scale factor of 1.0 means all ideal buys can be executed fully
            # 4. Scale factor of 0.0 means no buys can be executed (in cases of zero/negative cash)
            # 5. Scale factor is only applied to buy trades, sell trades remain unchanged
            # 6. Final target quantity = Current holdings + (Scaled buys) + (Original sells)
            final_target_quantity = ideal_target_quantity.copy()
            scale_factor = 1.0

            if cash_needed_for_buys_gross > cash_available_after_ideal_sells + 1e-6:
                logging.warning(
                    f"[Backtester] {rebalancing_date.strftime('%Y%m%d')}: Insufficient cash for ideal buys. Available: {cash_available_after_ideal_sells:.2f}, Needed: {cash_needed_for_buys_gross:.2f}. Scaling down buys.")

                if cash_needed_for_buys_gross <= 1e-6:
                    scale_factor = 0.0
                    logging.warning(
                        f"[Backtester] {rebalancing_date.strftime('%Y%m%d')}: Ideal buy value is zero or negative, cannot scale. Setting buy scale factor to 0.")
                elif cash_available_after_ideal_sells < 0:
                    scale_factor = 0.0
                    logging.warning(
                        f"[Backtester] {rebalancing_date.strftime('%Y%m%d')}: Cash available after ideal sells is negative ({cash_available_after_ideal_sells:.2f}). Setting buy scale factor to 0.")
                else:
                    scale_factor = cash_available_after_ideal_sells / cash_needed_for_buys_gross
                    scale_factor = max(0.0, min(1.0, scale_factor))
                    logging.info(
                        f"[Backtester] {rebalancing_date.strftime('%Y%m%d')}: Buy scaling factor calculated: {scale_factor:.6f}")

                union_index = ideal_target_quantity.index.union(
                    current_quantity.index)
                ideal_target_aligned = ideal_target_quantity.reindex(
                    union_index, fill_value=0.0)
                current_aligned = current_quantity.reindex(
                    union_index, fill_value=0.0)

                net_trades_ideal = ideal_target_aligned - current_aligned
                scaled_buys = net_trades_ideal[net_trades_ideal >
                                               0] * scale_factor
                sells = net_trades_ideal[net_trades_ideal <= 0]

                final_target_quantity = current_aligned.copy()
                final_target_quantity.update(
                    final_target_quantity.add(scaled_buys, fill_value=0.0))
                final_target_quantity.update(
                    final_target_quantity.add(sells, fill_value=0.0))
                final_target_quantity = final_target_quantity[final_target_quantity.abs(
                ) > 1e-9].round(8)

            else:
                logging.info(
                    f"[Backtester] {rebalancing_date.strftime('%Y%m%d')}: Ideal trades are affordable.")
                final_target_quantity = final_target_quantity[final_target_quantity.abs(
                ) > 1e-9].round(8)

            # NOTE: Calculate costs under final target quantity (Scale considered)
            buy_value, sell_value, buy_cost, sell_cost, total_transaction_cost, transaction_summary = self.cost_calculator.calculate_costs(
                final_target_quantity, current_quantity, rebalancing_date, self.price, total_assets_before_rebalance, calculate_summary=True)

            if transaction_summary:
                self._transaction_summaries.append(transaction_summary)
                logging.info(
                    f"[Backtester] {rebalancing_date.strftime('%Y%m%d')}: Final Trades - Sell Value: {sell_value:.2f}, Sell Cost: {sell_cost:.2f}, Buy Value: {buy_value:.2f}, Buy Cost: {buy_cost:.2f}, Total Cost: {total_transaction_cost:.2f}")
            else:
                logging.warning(
                    f"[Backtester] {rebalancing_date.strftime('%Y%m%d')}: Transaction summary not generated (likely due to calculation errors). Assuming zero values/costs.")
                buy_value, sell_value, buy_cost, sell_cost, total_transaction_cost = 0.0, 0.0, 0.0, 0.0, 0.0

            net_cash_flow_from_trades = (
                sell_value - sell_cost) - (buy_value + buy_cost)
            cash_balance += net_cash_flow_from_trades

            logging.info(
                f"[Backtester] {rebalancing_date.strftime('%Y%m%d')}: Cash Balance Updated. Net flow from trades: {net_cash_flow_from_trades:.2f}. Final Cash Balance: {cash_balance:.2f}")

            # NOTE: Assume long-only, non-leveraged portfolio, so cash balance cannot be negative
            # If cash balance is negative due to floating point errors, clamp to 0.0
            if cash_balance < 0:
                if abs(cash_balance) > 1.0: 
                    logging.error(
                        f"[Backtester] {rebalancing_date.strftime('%Y%m%d')}: Significant negative cash balance detected ({cash_balance:,.2f}) after trades! Check scaling logic or costs.")
                    logging.warning(
                        "[Backtester] Clamping significantly negative cash balance to 0.0. Results might be inaccurate.")
                    cash_balance = 0.0
                else:
                    logging.warning(
                        f"[Backtester] {rebalancing_date.strftime('%Y%m%d')}: Near-zero negative cash balance ({cash_balance:.2f}) detected after trades. Clamping to 0.0")
                    cash_balance = 0.0

            self._cash_balances.append(
                {'Date': rebalancing_date, 'Cash Balance': cash_balance, 'State': 'Post-Trade'})

            holding_period_dates = self.price.index[
                (self.price.index > rebalancing_date) & (
                    self.price.index <= next_rebalancing_date)
            ]

            # NOTE: Period portfolio values and cash values under holding period
            # Period cash values are updated after trades
            # Get valid stocks present in both target quantity and price columns
            period_portfolio_values = pd.Series(
                dtype=float, index=holding_period_dates)

            period_cash_values = pd.Series(
                cash_balance, index=holding_period_dates, dtype=float)

            if not final_target_quantity.empty and not holding_period_dates.empty:
                valid_cols = final_target_quantity.index.intersection(
                    self.price.columns)
                qty_filtered = final_target_quantity.reindex(
                    valid_cols).fillna(0)

                if not qty_filtered.empty:
                    try:
                        price_period = self.price.loc[holding_period_dates, qty_filtered.index].fillna(
                            0)
                        daily_values = price_period @ qty_filtered
                        period_portfolio_values = daily_values
                    except Exception as e:
                        logging.error(
                            f"[Backtester] Error calculating daily portfolio values for period starting {rebalancing_date}: {e}")

                missing_cols = final_target_quantity.index.difference(
                    valid_cols)
                if not missing_cols.empty:
                    logging.warning(f"[Backtester] Stocks {missing_cols.tolist()} in target quantity not found in price data columns "
                                    f"during holding period {rebalancing_date.strftime('%Y%m%d')} to {next_rebalancing_date.strftime('%Y%m%d')}. "
                                    "Excluded from daily value calculation.")

            period_df = pd.DataFrame({
                'Portfolio': period_portfolio_values,
                'Cash': period_cash_values,
                'Total': period_portfolio_values + period_cash_values
            })
            portfolio_values_over_time.append(period_df)

            # NOTE: Update for next period
            current_quantity = final_target_quantity.copy()
            self._final_holdings_data.append(
                {'Date': rebalancing_date, 'Quantity': current_quantity.copy()})

            logging.info(
                f"--- Period End: {rebalancing_date.strftime('%Y%m%d')} (Held until {next_rebalancing_date.strftime('%Y%m%d')}) ---")

        # NOTE: Handle case with rebalancing dates but no values generated (e.g., all periods skipped)
        if not portfolio_values_over_time:
            logging.warning(
                "[Backtester] No portfolio values were generated during the backtest loop.")
            if rebalancing_dates:
                start_date = rebalancing_dates[0]
                initial_cash_state = next(
                    (item for item in self._cash_balances if item['Date'] == start_date), None)
                initial_cash = initial_cash_state['Cash Balance'] if initial_cash_state else self.init_invest

                initial_df = pd.DataFrame({'Portfolio': [0.0], 'Cash': [initial_cash], 'Total': [
                                          initial_cash]}, index=[start_date])
                initial_df.index.name = 'Date'
                self._results_df = initial_df
                logging.warning(
                    "[Backtester] Returning DataFrame with initial cash state only.")
                return self._results_df
            else:
                self._results_df = pd.DataFrame(
                    columns=['Portfolio', 'Cash', 'Total'], index=pd.to_datetime([]), dtype=float)
                return self._results_df

        # NOTE: Concatenate daily portfolio values
        results_df = pd.concat(portfolio_values_over_time)
        results_df.index.name = 'Date'

        self._results_df = results_df.sort_index()
        print("***** Portfolio backtest simulation finished *****\n")
        return self._results_df

    @property
    def results(self) -> Optional[pd.DataFrame]:
        """Returns the main results DataFrame (Portfolio, Cash, Total). Runs backtest if needed."""
        if self._results_df is None:
            logging.info(
                "[Backtester] Results not available. Running run_backtest() first.")
            self.run_backtest()
        return self._results_df

    @property
    def transaction_costs_summary(self) -> pd.DataFrame:
        """Returns a DataFrame summarizing transaction details for each rebalancing date."""
        if not self._transaction_summaries:
            if self._results_df is not None:
                logging.warning(
                    "[Backtester] Backtest was run, but no transaction summaries were generated (possibly no trades or errors during cost calculation). Returning empty summary.")
            else:
                logging.warning(
                    "[Backtester] Backtest has not been run. Execute run_backtest() first to get transaction summaries.")
            cols = ['Date', 'Total Buy Value', 'Total Sell Value', 'Shares Bought', 'Shares Sold',
                    'Total Buy Cost', 'Total Sell Cost', 'Total Transaction Cost', 'Total NAV',
                    'Transaction Cost (bp)', 'Error']
            return pd.DataFrame(columns=cols).set_index('Date')

        valid_summaries = [
            s for s in self._transaction_summaries if s is not None]
        if not valid_summaries:
            logging.warning(
                "[Backtester] No valid transaction summaries found. Returning empty DataFrame.")
            cols = ['Date', 'Total Buy Value', 'Total Sell Value', 'Shares Bought', 'Shares Sold',
                    'Total Buy Cost', 'Total Sell Cost', 'Total Transaction Cost', 'Total NAV',
                    'Transaction Cost (bp)', 'Error']
            return pd.DataFrame(columns=cols).set_index('Date')

        return pd.DataFrame(valid_summaries).set_index('Date')

    @property
    def cash_balance_summary(self) -> pd.DataFrame:
        """Returns a DataFrame showing the cash balance at different stages for each rebalancing."""
        if not self._cash_balances:
            if self._results_df is not None:
                logging.warning(
                    "[Backtester] Backtest was run, but no cash balances were recorded internally. Returning empty summary.")
            else:
                logging.warning(
                    "[Backtester] Backtest has not been run. Execute run_backtest() first to get cash balance details.")
            return pd.DataFrame(columns=['Date', 'Cash Balance', 'State']).set_index('Date')
        return pd.DataFrame(self._cash_balances).set_index('Date')

    @property
    def portfolio_returns(self) -> pd.DataFrame:
        """Calculates and returns the daily percentage returns of the total portfolio NAV."""
        results_df = self.results 
        if results_df is None or results_df.empty or 'Total' not in results_df.columns:
            logging.warning(
                "[Backtester] Cannot calculate portfolio returns: Results DataFrame is missing, empty, or lacks 'Total' column.")
            return pd.DataFrame(columns=['Portfolio Return'], index=pd.to_datetime([]), dtype=float)
        if len(results_df['Total']) < 2:
            logging.warning(
                "[Backtester] Cannot calculate portfolio returns: Not enough data points.")
            return pd.DataFrame(columns=['Portfolio Return'], index=results_df.index, dtype=float)

        pf_ret = results_df['Total'].pct_change().to_frame('Portfolio Return')
        pf_ret.index = pd.to_datetime(results_df.index)
        pf_ret.index.name = 'Date'
        return pf_ret

    @property
    def benchmark_returns(self) -> pd.DataFrame:
        """Calculates and returns the daily percentage returns of the benchmark index over the backtest period."""
        results_df = self.results
        if results_df is None or results_df.empty:
            logging.warning(
                "[Backtester] Cannot calculate benchmark returns: Backtest results unavailable for date range.")
            return pd.DataFrame(columns=[self.bm.columns[0] if not self.bm.empty else 'Benchmark'], index=pd.to_datetime([]), dtype=float)

        start_date = results_df.index.min()
        end_date = results_df.index.max()

        try:
            bm_relevant = self.bm.loc[start_date:end_date]

            if bm_relevant.empty:
                logging.warning(
                    f"[Backtester] No benchmark data available for the backtest period [{start_date} to {end_date}].")
                return pd.DataFrame(columns=[self.bm.columns[0] if not self.bm.empty else 'Benchmark'], index=pd.to_datetime([]), dtype=float)
            if len(bm_relevant) < 2:
                logging.warning(
                    "[Backtester] Not enough benchmark data points within the backtest period to calculate returns.")
                return pd.DataFrame(columns=[self.bm.columns[0] if not self.bm.empty else 'Benchmark'], index=bm_relevant.index, dtype=float)

            bm_ret = bm_relevant.pct_change()
            bm_ret.index = pd.to_datetime(
                bm_relevant.index)
            bm_ret.index.name = 'Date'
            return bm_ret.reindex(results_df.index)

        except Exception as e:
            logging.error(
                f"[Backtester] Error calculating benchmark returns: {e}")
            return pd.DataFrame(columns=[self.bm.columns[0] if not self.bm.empty else 'Benchmark'], index=results_df.index, dtype=float)

    @property
    def portfolio_quantity(self) -> pd.DataFrame:
        """Returns a DataFrame containing the actual portfolio quantities at each rebalancing date."""
        if not self._final_holdings_data:
            if self._results_df is not None:
                logging.warning(
                    "[Backtester] Backtest run, but no final holdings data recorded.")
            else:
                logging.warning(
                    "[Backtester] Backtest has not been run. Run run_backtest() first.")
            return pd.DataFrame()

        quantities_list = []
        dates_list = []
        for item in self._final_holdings_data:
            if not item['Quantity'].empty:
                dates_list.append(item['Date'])
                quantities_list.append(item['Quantity'])

        if not quantities_list:
            logging.warning(
                "[Backtester] No non-empty portfolio quantities found.")
            return pd.DataFrame()

        df_quantities = pd.DataFrame(
            {date: qty for date, qty in zip(dates_list, quantities_list)}).T
        df_quantities.index.name = 'Date'
        return df_quantities.sort_index()

    def get_holdings_snapshot(self, top_n: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """
        Returns a dictionary mapping rebalancing dates to DataFrames containing portfolio holdings information
        (quantity, price, market value, weight) for that date.

        Args:
            top_n: If provided, only return the top N holdings by market value for each date.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping dates (YYYYMMDD format) to holdings DataFrames.
        """
        if not self._final_holdings_data:
            if self._results_df is not None:
                logging.warning(
                    "[Backtester] Backtest run, but no final holdings data recorded for snapshots.")
            else:
                logging.warning(
                    "[Backtester] Backtest has not been run. Run run_backtest() first to get holdings snapshots.")

            return {}

        holdings_snapshots = {}
        for item in self._final_holdings_data:
            date = item['Date']
            quantities = item['Quantity']

            if quantities.empty:
                logging.debug(
                    f"[Holdings] No holdings recorded for date {date.strftime('%Y%m%d')}.")
                continue

            date_str = dt.datetime.strftime(date, format='%Y%m%d')

            try:
                if date not in self.price.index:
                    logging.warning(
                        f"[Holdings] Price data missing for snapshot date {date_str}. Cannot create snapshot.")
                    continue

                prices_at_date = self.price.loc[date]
                valid_stocks = quantities.index.intersection(
                    prices_at_date.index)

                if valid_stocks.empty:
                    logging.warning(
                        f"[Holdings] No valid stocks with price data found for holdings snapshot on date {date_str}")
                    continue

                holdings_df = pd.DataFrame({
                    'quantity': quantities.loc[valid_stocks],
                    'price': prices_at_date.loc[valid_stocks],
                })

                holdings_df['market_value'] = holdings_df['quantity'] * \
                    holdings_df['price']
                holdings_df = holdings_df[holdings_df['market_value'] > 1e-9]

                if holdings_df.empty:
                    logging.debug(
                        f"[Holdings] Holdings for date {date_str} have zero market value after calculation.")
                    continue

                total_market_value = holdings_df['market_value'].sum()
                holdings_df['weight'] = holdings_df['market_value'] / \
                    total_market_value if total_market_value > 0 else 0

                holdings_df = holdings_df.sort_values(
                    'market_value', ascending=False)

                if top_n is not None and top_n > 0:
                    holdings_df = holdings_df.head(top_n)

                holdings_snapshots[date_str] = holdings_df

            except KeyError as e:
                logging.error(
                    f"[Holdings] Error accessing price data for holdings snapshot on {date_str}: {e}")
                continue
            except Exception as e:
                logging.error(
                    f"[Holdings] Unexpected error creating holdings snapshot for {date_str}: {e}")
                continue
        return holdings_snapshots

    def get_sector_snapshot(self) -> Dict[str, Dict[str, float]]:
        """
        Calculates the sector allocation (based on 'wics_sector_big') for the portfolio
        at each rebalancing date by leveraging the pre-calculated holdings snapshot.

        Returns:
            Dict[str, Dict[str, float]]: Dictionary mapping dates (YYYYMMDD format)
                                         to inner dictionaries mapping sector names to weights (%).
        """
        holdings_snapshots = self.get_holdings_snapshot()
        if not holdings_snapshots:
            logging.warning(
                "[SectorSnapshot] No holdings snapshots available to calculate sector data.")
            return {}

        if self.sector is None or self.sector.empty:
            logging.error(
                "[SectorSnapshot] Sector data (self.sector) is not loaded or empty.")
            return {}
        logging.info(
            f"[SectorSnapshot] Using pre-loaded sector data for market {self.mkt}")

        sector_snapshots = {}
        for date_str, holdings_df in holdings_snapshots.items():
            if holdings_df.empty or 'weight' not in holdings_df.columns:
                logging.debug(
                    f"[SectorSnapshot] Holdings snapshot for {date_str} is empty or missing 'weight' column. Skipping.")
                continue

            try:
                try:
                    date = dt.datetime.strptime(date_str, '%Y%m%d')
                except ValueError:
                    logging.error(
                        f"[SectorSnapshot] Invalid date format in holdings snapshot key: {date_str}")
                    continue

                if date not in self.sector.index:
                    logging.warning(
                        f"[SectorSnapshot] Date {date_str} not found in sector data index. Skipping sector calculation for this date.")
                    continue

                tickers = holdings_df.index
                sector_data_for_date = self.sector.loc[[
                    date], self.sector.columns.intersection(tickers)]

                date_sectors: Dict[str, float] = {}
                for ticker in tickers:
                    if ticker not in sector_data_for_date.columns:
                        continue

                    sector = sector_data_for_date.loc[date, ticker]

                    if pd.notna(sector) and sector != 'None' and isinstance(sector, str):
                        weight = holdings_df.loc[ticker, 'weight']
                        date_sectors[sector] = date_sectors.get(
                            sector, 0.0) + weight
                    else:
                        logging.debug(
                            f"[SectorSnapshot] Ticker {ticker} has missing/invalid sector ('{sector}') on {date_str}.")

                if date_sectors:
                    current_total_weight = sum(date_sectors.values())
                    if current_total_weight > 1e-9:
                        normalized_sectors = {
                            sector: (weight / current_total_weight) * 100.0
                            for sector, weight in date_sectors.items()
                        }
                        sector_snapshots[date_str] = normalized_sectors
                    else:
                        logging.debug(
                            f"[SectorSnapshot] Total sector weight is zero for date {date_str} after aggregation.")
                else:
                    logging.debug(
                        f"[SectorSnapshot] No valid sector data found for any holdings on {date_str}.")

            except KeyError as e:
                logging.error(
                    f"[SectorSnapshot] KeyError accessing data for sector snapshot on {date_str}: {e}")
                continue
            except Exception as e:
                logging.error(
                    f"[SectorSnapshot] Unexpected error creating sector snapshot for {date_str}: {e}")
                import traceback
                traceback.print_exc()
                continue
        return sector_snapshots