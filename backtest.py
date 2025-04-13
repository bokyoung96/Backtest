import datetime as dt
from tqdm import tqdm
import pandas as pd
import numpy as np
import logging
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
        previous_portfolio_value = 0.0

        for period in tqdm(range(1, rebalancing_periods + 1)):
            start_date = self.rebalancing_dates[period - 1]
            formation_date = self.formation_dates[period - 1]

            if period != rebalancing_periods:
                end_date = self.rebalancing_dates[period]
                days_in_period = (end_date - start_date).days
            else:
                end_date = self.price.index[-1]
                days_in_period = (end_date - start_date).days
                if end_date < start_date:
                    end_date = start_date
                    days_in_period = 0
                    print(
                        f"Warning: Last price date {self.price.index[-1]} is before last rebalancing date {start_date}. Using start date as end date for final period.")

            interest = cash_balance * \
                (self.cash_rate / 365) * \
                days_in_period if days_in_period > 0 else 0
            cash_balance += interest

            if previous_quantity is not None and not previous_quantity.empty:
                current_prices = self.price.loc[start_date,
                                                previous_quantity.index]
                portfolio_value = np.dot(previous_quantity, current_prices)
            else:
                portfolio_value = 0.0

            total_assets_before_rebalance = portfolio_value + cash_balance
            logging.info(
                f"{start_date}: Total assets before rebalance: {total_assets_before_rebalance:.2f} (Portfolio: {portfolio_value:.2f}, Cash: {cash_balance:.2f})")

            # 4. Determine IDEAL target quantity based on total assets BEFORE costs
            ideal_target_quantity = self.get_quantity(formation_date=formation_date,
                                                      rebalancing_date=start_date,
                                                      total_assets=total_assets_before_rebalance)
            if ideal_target_quantity.empty:
                logging.warning(
                    f"{start_date}: Ideal target quantity is empty.")

            # 5. Calculate the cash impact of achieving the IDEAL target quantity
            # Returns: buy_val, sell_val, buy_cost, sell_cost, total_cost, summary (None)
            buy_value_ideal, sell_value_ideal, buy_cost_ideal, sell_cost_ideal, total_cost_ideal, _ = self._calculate_transaction_costs(
                ideal_target_quantity, previous_quantity, start_date, total_assets_before_rebalance, calculate_summary=False)

            # Cash generated from selling assets (net of sell costs)
            cash_from_sells = sell_value_ideal - sell_cost_ideal
            # Cash needed to buy assets (including buy costs)
            cash_needed_for_buys = buy_value_ideal + buy_cost_ideal

            # Total cash available assuming ideal sells are executed
            cash_available_after_sells = cash_balance + cash_from_sells

            logging.info(
                f"{start_date}: Ideal Trades - Sell Value: {sell_value_ideal:.2f}, Sell Cost: {sell_cost_ideal:.2f}, Buy Value: {buy_value_ideal:.2f}, Buy Cost: {buy_cost_ideal:.2f}")
            logging.info(
                f"{start_date}: Cash from Sells: {cash_from_sells:.2f}, Cash Needed for Buys: {cash_needed_for_buys:.2f}, Cash Available After Sells: {cash_available_after_sells:.2f}")

            # 6. Check if ideal buy trades are affordable
            scale_factor = 1.0  # Default: no scaling
            if cash_needed_for_buys > cash_available_after_sells + 1e-6:  # Add tolerance
                logging.warning(
                    f"{start_date}: Insufficient cash. Available for buys: {cash_available_after_sells:.2f}, Needed: {cash_needed_for_buys:.2f}. Scaling down buys.")

                if cash_needed_for_buys <= 1e-6:  # Avoid division by zero if ideal buys are zero
                    scale_factor = 0.0
                    logging.warning(
                        f"{start_date}: Ideal buy value is zero, cannot scale. Setting buy scale factor to 0.")
                elif cash_available_after_sells < 0:
                    # If even after selling we don't have cash, we cannot buy anything
                    scale_factor = 0.0
                    logging.warning(
                        f"{start_date}: Cash available after sells is negative ({cash_available_after_sells:.2f}). Setting buy scale factor to 0.")
                else:
                    # Calculate the scaling factor based on affordable buy spending
                    # We need: buy_value_new * (1 + buy_commission + slippage) = cash_available_after_sells
                    # scale_factor = buy_value_new / buy_value_ideal
                    # Combined: scale_factor = cash_available_after_sells / (buy_value_ideal * (1 + buy_cost_rate))
                    # Simplified using cash_needed_for_buys = buy_value_ideal * (1 + buy_cost_rate)
                    buy_cost_rate = self.buy_commission + self.slippage
                    # affordable_buy_value = cash_available_after_sells / (1 + buy_cost_rate)
                    # scale_factor = affordable_buy_value / buy_value_ideal
                    scale_factor = cash_available_after_sells / cash_needed_for_buys
                    # Ensure factor is between 0 and 1
                    scale_factor = max(0.0, min(1.0, scale_factor))
                    logging.info(
                        f"{start_date}: Buy scaling factor calculated: {scale_factor:.6f}")

                # Align indices for scaling calculation
                union_index = ideal_target_quantity.index
                if previous_quantity is not None:
                    union_index = union_index.union(previous_quantity.index)

                ideal_target_aligned = ideal_target_quantity.reindex(
                    union_index, fill_value=0.0)
                previous_aligned = previous_quantity.reindex(
                    union_index, fill_value=0.0) if previous_quantity is not None else pd.Series(0.0, index=union_index)

                net_trades_ideal = ideal_target_aligned - previous_aligned
                buys_ideal_qty = net_trades_ideal[net_trades_ideal > 0]
                # Negative values representing sells
                sells_ideal_qty = net_trades_ideal[net_trades_ideal < 0]

                # Scale down only the buy quantities
                scaled_buys_qty = buys_ideal_qty * scale_factor

                # Combine sells (which are unchanged) and scaled buys
                final_net_trades = sells_ideal_qty.combine_first(
                    scaled_buys_qty)

                # Recalculate final target quantity based on scaled net trades
                # Increased rounding precision slightly
                final_target_quantity = (
                    previous_aligned + final_net_trades).round(8)
                # Remove dust with higher precision
                final_target_quantity = final_target_quantity[final_target_quantity > 1e-8]
                logging.info(
                    f"{start_date}: Buy trades scaled. Final target quantity calculated.")

            else:
                # Affordable, use the ideal target quantity
                final_target_quantity = ideal_target_quantity
                logging.info(
                    f"{start_date}: Ideal trades are affordable. Using ideal target quantity.")

            # 7. Calculate final transaction details and costs based on the FINAL (potentially scaled) target quantity
            buy_value, sell_value, buy_cost, sell_cost, total_transaction_cost, transaction_summary = self._calculate_transaction_costs(
                final_target_quantity, previous_quantity, start_date, total_assets_before_rebalance, calculate_summary=True)

            # Check if summary exists (it might not if there were errors in calculation)
            if transaction_summary:
                self._transaction_summaries.append(transaction_summary)
                logging.info(
                    f"{start_date}: Final Trades - Sell Value: {sell_value:.2f}, Sell Cost: {sell_cost:.2f}, Buy Value: {buy_value:.2f}, Buy Cost: {buy_cost:.2f}, Total Cost: {total_transaction_cost:.2f}")
            else:
                logging.warning(
                    f"{start_date}: Transaction summary was not generated, likely due to errors in _calculate_transaction_costs.")
                # Still need placeholder values for cash calculation
                buy_value, sell_value, buy_cost, sell_cost, total_transaction_cost = 0.0, 0.0, 0.0, 0.0, 0.0

            # 8. Update cash balance: Start Cash + Net Cash from Sells - Net Cash for Buys
            # Start Cash = cash_balance (before adding interest for *this* period)
            # We added interest already. So: cash_balance (with interest) + (sell_value - sell_cost) - (buy_value + buy_cost)
            net_cash_flow_from_trades = (
                sell_value - sell_cost) - (buy_value + buy_cost)
            cash_balance += net_cash_flow_from_trades

            logging.info(
                f"{start_date}: Cash Balance Updated. Net flow from trades: {net_cash_flow_from_trades:.2f}. Final Cash: {cash_balance:.2f}")

            # 9. Final check for negative cash balance (should be extremely unlikely now)
            if cash_balance < -1e-6:  # Allow for small floating point inaccuracies
                logging.error(
                    f"{start_date}: Negative cash balance encountered. Balance: {cash_balance:.2f}")
                raise ValueError(
                    f"Negative cash balance encountered on {start_date}. Cash Balance: {cash_balance:.2f}. Total Assets Before Rebalance: {total_assets_before_rebalance:.2f}, Net Trade Flow: {net_cash_flow_from_trades:.2f}, Total Transaction Cost: {total_transaction_cost:.2f}")
            elif cash_balance < 0:
                logging.warning(
                    f"{start_date}: Near-zero negative cash balance. Setting to 0.")
                cash_balance = 0.0

            self._cash_balances.append(
                {'Date': start_date, 'Cash Balance': cash_balance})

            # --- Portfolio Value Calculation for the Period ---\
            # Use the FINAL target_quantity for calculating portfolio value during the holding period
            if not final_target_quantity.empty:
                # Ensure we only select valid dates between start_date (exclusive) and end_date (inclusive)
                period_dates = self.price.index[(self.price.index > start_date) & (
                    self.price.index <= end_date)]

                if not period_dates.empty:
                    # Ensure final_target_quantity index exists in price columns
                    valid_cols = final_target_quantity.index.intersection(
                        self.price.columns)
                    if not valid_cols.equals(final_target_quantity.index):
                        missing_cols = final_target_quantity.index.difference(
                            valid_cols)
                        logging.warning(
                            f"Stocks {missing_cols.tolist()} in final quantity not found in price data for period starting after {start_date}. Setting their value to 0 for this period.")
                        final_target_quantity_filtered = final_target_quantity.reindex(
                            valid_cols).fillna(0)
                    else:
                        final_target_quantity_filtered = final_target_quantity

                    if not final_target_quantity_filtered.empty and not period_dates.empty:
                        price_temp = self.price.loc[period_dates,
                                                    final_target_quantity_filtered.index]
                        portfolio_values = np.dot(
                            final_target_quantity_filtered, price_temp.to_numpy().T)
                        cf = pd.DataFrame(
                            portfolio_values, index=period_dates, columns=['Portfolio'])
                    else:
                        cf = pd.DataFrame(
                            0.0, index=period_dates, columns=['Portfolio'])

                else:
                    # If no dates in the period, create an empty DataFrame with correct index type
                    cf = pd.DataFrame(
                        columns=['Portfolio'], index=pd.to_datetime([]))

            # Handle case with empty final_target_quantity (e.g., all cash or scaled to zero)
            else:
                period_dates = self.price.index[(self.price.index > start_date) & (
                    self.price.index <= end_date)]
                cf = pd.DataFrame(0.0, index=period_dates,
                                  columns=['Portfolio'])

            # Store the calculated final quantity for this period
            res_quantity.append(final_target_quantity.copy())  # Store a copy

            # Update previous quantity for the next iteration
            previous_quantity = final_target_quantity.copy()  # Use a copy

            temp.append(cf)
            # Minor logging change for clarity
            logging.info(
                f"Processed period ending {end_date.strftime('%Y%m%d')} (Rebalance Date: {start_date.strftime('%Y%m%d')}).")

        # Concatenate results and add cash component
        if temp:
            res = pd.concat(temp)
            res.index = pd.to_datetime(res.index)
            res.index.name = 'Date'

            cash_df = pd.DataFrame(self._cash_balances).set_index('Date')
            res['Cash'] = cash_df['Cash Balance'].reindex(
                res.index, method='ffill')
            first_valid_cash_index = cash_df.index.min()
            if res.index.min() < first_valid_cash_index:
                res.loc[res.index < first_valid_cash_index,
                        'Cash'] = self.init_invest

            res['Cash'].fillna(method='ffill', inplace=True)
            res['Cash'].fillna(0, inplace=True)

            res['Total'] = res['Portfolio'] + res['Cash']
        else:
            res = pd.DataFrame(
                columns=['Portfolio', 'Cash', 'Total'], index=pd.to_datetime([]))
            res.index.name = 'Date'

        self._res_quantity = res_quantity
        return res

    def _calculate_transaction_costs(self, target_quantity, previous_quantity, date, total_assets_before_rebalance, calculate_summary=True):
        # (Existing code for quantity alignment and price fetching...)
        # ... [rest of the function remains largely the same, but ensure it uses the potentially scaled target_quantity] ...

        # Ensure target_quantity is a Series, handle potential empty DataFrame/Series
        if isinstance(target_quantity, pd.DataFrame):
            target_quantity = target_quantity.iloc[:, 0] if not target_quantity.empty else pd.Series(
                dtype=float)
        if target_quantity is None:
            target_quantity = pd.Series(dtype=float)

        # Get prices for relevant stocks on the transaction date
        # Combine indices to ensure all necessary prices are fetched
        all_involved_stocks = target_quantity.index
        if previous_quantity is not None and not previous_quantity.empty:
            all_involved_stocks = all_involved_stocks.union(
                previous_quantity.index)

        # Handle case where calculation might be done with no stocks (e.g. initial state or after full liquidation)
        if all_involved_stocks.empty:
            transaction_summary = None
            if calculate_summary:
                transaction_summary = {
                    'Date': date, 'Total Buy Value': 0.0, 'Total Sell Value': 0.0,
                    'Shares Bought': 0.0, 'Shares Sold': 0.0, 'Total Buy Cost': 0.0,
                    'Total Sell Cost': 0.0, 'Total Transaction Cost': 0.0,
                    'Total NAV': total_assets_before_rebalance, 'Transaction Cost (bp)': 0.0
                }
            # buy_value, sell_value, buy_cost, sell_cost, total_transaction_cost, summary
            return 0.0, 0.0, 0.0, 0.0, 0.0, transaction_summary

        # Fetch prices only for stocks involved in potential trades
        # Add error handling for missing price data on the specific date
        try:
            price_at_date = self.price.loc[date, all_involved_stocks]
        except KeyError:
            logging.error(
                f"Price data missing for one or more stocks {all_involved_stocks.tolist()} on date {date}. Cannot calculate costs.")
            # Decide handling: raise error, return zeros, or try to fill? Returning zeros for now.
            transaction_summary = None
            if calculate_summary:
                transaction_summary = {
                    'Date': date, 'Total Buy Value': 0.0, 'Total Sell Value': 0.0,
                    'Shares Bought': 0.0, 'Shares Sold': 0.0, 'Total Buy Cost': 0.0,
                    'Total Sell Cost': 0.0, 'Total Transaction Cost': 0.0,
                    'Total NAV': total_assets_before_rebalance, 'Transaction Cost (bp)': 0.0,
                    'Error': 'Missing price data'
                }
            return 0.0, 0.0, 0.0, 0.0, 0.0, transaction_summary

        # Align previous_quantity to target_quantity's index, filling missing with 0
        if previous_quantity is None or previous_quantity.empty:
            union_index = target_quantity.index
            target_quantity_aligned = target_quantity
            previous_quantity_aligned = pd.Series(0.0, index=union_index)
        else:
            # Align both series to the union of their indices
            union_index = target_quantity.index.union(previous_quantity.index)
            target_quantity_aligned = target_quantity.reindex(
                union_index, fill_value=0.0)
            previous_quantity_aligned = previous_quantity.reindex(
                union_index, fill_value=0.0)

        # Calculate net trades based on aligned quantities
        net_trades = target_quantity_aligned - previous_quantity_aligned

        # Filter out zero trades (using tolerance)
        # Use a slightly higher precision for filtering near-zero trades
        net_trades = net_trades[net_trades.abs() > 1e-8]

        if net_trades.empty:
            # No effective trades needed
            transaction_summary = None
            if calculate_summary:
                transaction_summary = {
                    'Date': date, 'Total Buy Value': 0.0, 'Total Sell Value': 0.0,
                    'Shares Bought': 0.0, 'Shares Sold': 0.0, 'Total Buy Cost': 0.0,
                    'Total Sell Cost': 0.0, 'Total Transaction Cost': 0.0,
                    'Total NAV': total_assets_before_rebalance, 'Transaction Cost (bp)': 0.0
                }
            return 0.0, 0.0, 0.0, 0.0, 0.0, transaction_summary

        # Use aligned prices for calculations, ensure prices exist for all trades
        try:
            price_at_date_aligned = price_at_date.reindex(net_trades.index)
            # Check for NaNs in prices which would indicate missing data for a stock involved in a trade
            if price_at_date_aligned.isnull().any():
                missing_stocks = price_at_date_aligned[price_at_date_aligned.isnull(
                )].index.tolist()
                logging.error(
                    f"Price data NaN for stocks involved in trades {missing_stocks} on date {date}.")
                # Handle NaN prices - filter them out for calculation.
                valid_trades_idx = net_trades.index[~price_at_date_aligned.isnull(
                )]
                if valid_trades_idx.empty:
                    logging.error(
                        f"All trades on {date} involved stocks with NaN prices. Skipping cost calculation for this date.")
                    transaction_summary = None
                    if calculate_summary:
                        transaction_summary = {
                            'Date': date, 'Total Buy Value': 0.0, 'Total Sell Value': 0.0,
                            'Shares Bought': 0.0, 'Shares Sold': 0.0, 'Total Buy Cost': 0.0,
                            'Total Sell Cost': 0.0, 'Total Transaction Cost': 0.0,
                            'Total NAV': total_assets_before_rebalance, 'Transaction Cost (bp)': 0.0,
                            'Error': 'NaN prices for all trade stocks'
                        }
                    return 0.0, 0.0, 0.0, 0.0, 0.0, transaction_summary
                else:
                    logging.warning(
                        f"Calculating costs only for stocks with valid prices on {date}. Missing: {missing_stocks}")
                    net_trades = net_trades.loc[valid_trades_idx]
                    price_at_date_aligned = price_at_date_aligned.loc[valid_trades_idx]

        except KeyError as e:
            logging.error(
                f"Price data missing (KeyError: {e}) for trades on {date}. Stocks: {net_trades.index.tolist()}")
            transaction_summary = None
            if calculate_summary:
                transaction_summary = {
                    'Date': date, 'Total Buy Value': 0.0, 'Total Sell Value': 0.0,
                    'Shares Bought': 0.0, 'Shares Sold': 0.0, 'Total Buy Cost': 0.0,
                    'Total Sell Cost': 0.0, 'Total Transaction Cost': 0.0,
                    'Total NAV': total_assets_before_rebalance, 'Transaction Cost (bp)': 0.0,
                    'Error': f'Missing price data (KeyError: {e})'
                }
            return 0.0, 0.0, 0.0, 0.0, 0.0, transaction_summary

        buys = net_trades[net_trades > 0]
        sells = -net_trades[net_trades < 0]  # Sell quantities are positive

        buy_values = (buys * price_at_date_aligned.loc[buys.index]).sum()
        sell_values = (sells * price_at_date_aligned.loc[sells.index]).sum()

        buy_cost = (self.buy_commission + self.slippage) * buy_values
        sell_cost = (self.sell_commission + self.slippage +
                     self.sell_tax) * sell_values

        total_transaction_cost = buy_cost + sell_cost

        shares_bought = buys.sum()
        shares_sold = sells.sum()

        # Calculate transaction cost in basis points relative to NAV *before* rebalancing
        transaction_cost_bp = (total_transaction_cost / total_assets_before_rebalance) * \
            10000 if total_assets_before_rebalance > 1e-6 else 0.0

        transaction_summary = None
        if calculate_summary:
            transaction_summary = {
                'Date': date,
                'Total Buy Value': buy_values,
                'Total Sell Value': sell_values,
                'Shares Bought': shares_bought,
                'Shares Sold': shares_sold,
                'Total Buy Cost': buy_cost,
                'Total Sell Cost': sell_cost,
                'Total Transaction Cost': total_transaction_cost,
                'Total NAV': total_assets_before_rebalance,  # NAV before rebalancing
                'Transaction Cost (bp)': transaction_cost_bp
            }

        # Return values needed for cash flow calculation in pf_cashflow
        return buy_values, sell_values, buy_cost, sell_cost, total_transaction_cost, transaction_summary

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
