# Backtesting Framework

A powerful and flexible backtesting framework for portfolio strategy testing and analysis.

## Installation

### Prerequisites

- Python >= 3.8
- pip package manager

### Installation Steps

1. Clone the repository:

```bash
git clone [your-repository-url]
cd bt
```

2. Install the package:

```bash
# Option 1: Using pip
pip install -e . --use-pep517

# Option 2: Using setup batch file (Windows)
setup.bat
```

## Framework Overview

<details>
<summary>📊 Backtesting Framework Architecture</summary>

The framework consists of several key components:

### 1. Portfolio Analysis & Backtesting

- `PortfolioAnalysis`: Main class for running backtests
  - Handles methodology selection and initialization
  - Manages portfolio construction
  - Calculates performance metrics
- `PortfolioBacktester`: Core backtesting engine
  - Executes backtesting simulation
  - Manages portfolio rebalancing
  - Tracks portfolio values and cash balances

### 2. Methodology Implementation

- `Methodology` (Abstract Base Class)
  - Define custom investment strategies
  - Implement weight calculation logic
  - Supports long/short positions (1, -1, NaN)
- Pre-implemented methodologies available via `MethodologyType`

### 3. Transaction Cost Handling

- `TransactionCostCalculator`: Comprehensive cost modeling
  - Buy/sell commissions
  - Slippage
  - Sell tax
  - Transaction cost analysis

### 4. Performance Analysis

- `PortfolioPerformance`: Performance metrics calculation
  - Returns analysis
  - Risk metrics
  - Benchmark comparison
  </details>

<details>
<summary>💡 Key Features</summary>

- **Flexible Strategy Implementation**: Create custom trading strategies by extending the base Strategy class
- **Comprehensive Cost Modeling**: Including commissions, slippage, and taxes
- **Portfolio Management**: Sophisticated portfolio rebalancing and position sizing
- **Performance Analytics**: Detailed performance metrics and analysis tools
- **Risk Management**: Built-in risk assessment capabilities
</details>

## Usage Examples

### Basic Strategy Implementation

```python
from bt import Methodology
import pandas as pd

class MyMethodology(Methodology):
    def __init__(self, mkt: str, start_date: str, end_date: str, **kwargs):
        super().__init__(mkt, start_date, end_date, **kwargs)
        # Initialize your strategy parameters

    @property
    def weights(self) -> pd.DataFrame:
        # Implement your strategy logic here
        # Return a DataFrame with weights for each asset
        # Use 1 for long, -1 for short, NaN for no position
        return weights_df

```

### Running a Backtest

```python
from bt import PortfolioAnalysis
from bt.methodology_type import MethodologyType

# Initialize the analysis
analysis = PortfolioAnalysis(
    mkt="KOSPI",
    start_date="20200101",
    end_date="20231231",
    methodology_type=MethodologyType.MyStrategy,  # Your registered strategy
    init_invest=100000000,
    buy_commission=0.0025,
    sell_commission=0.0025,
    slippage=0.001,
    sell_tax=0.003,
    cash_rate=0.03,
    rebal_timing='next',  # 'next' or 'close'
    multiplier="1"  # For performance calculation
)

# Run the backtest
analysis.calculate_weights()  # Initialize strategy
analysis.construct_portfolio()  # Build portfolio
analysis.calculate_performance()  # Calculate performance metrics

# Access results
portfolio_returns = analysis.portfolio_constructor.portfolio_returns
performance_metrics = analysis.perf_msre
transaction_summary = analysis.portfolio_constructor.transaction_costs_summary
holdings = analysis.portfolio_constructor.get_holdings_snapshot()
```

## Advanced Usage

<details>
<summary>🔧 Transaction Cost Analysis</summary>

The framework includes detailed transaction cost tracking:

```python
# Access transaction cost details
transaction_costs = analysis.portfolio_constructor.transaction_costs_summary

# Transaction cost components for each trade:
# - Commission (buy/sell)
# - Slippage
# - Sell tax
# - Total transaction cost in basis points
```

</details>

<details>
<summary>📈 Performance Analysis</summary>

Comprehensive performance metrics available through `PortfolioPerformance`:

```python
performance = analysis.perf_msre

# Available metrics
returns = performance.pf_ret  # Portfolio returns
benchmark_comparison = performance.bm_ret  # Benchmark returns
risk_metrics = performance.risk_metrics  # Risk analysis
```

</details>

<details>
<summary>🔄 Multiple Strategy Analysis</summary>

Run multiple strategies using `AnalysisManager`:

```python
from bt import AnalysisManager
from bt.methodology_type import MethodologyType

config = {
    'mkt': 'KOSPI',
    'start_date': '20200101',
    'end_date': '20231231',
    'init_invest': 100000000,
    # ... other parameters
}

methodology_types = [
    MethodologyType.Strategy1,
    MethodologyType.Strategy2
]

manager = AnalysisManager(config, methodology_types)
results = {method: manager.run(method) for method in methodology_types}
```

</details>

<details>
<summary>🔍 Backtest Logic Details</summary>

## Overview

The `backtest.py` file defines the `PortfolioConstructor` class for simulating portfolio performance (backtesting). It calculates portfolio value changes, returns, and transaction history based on user-defined investment strategies (weights), market data, and transaction costs.

### Core Components

#### Initialization Parameters

```python
portfolio_constructor = PortfolioConstructor(
    mkt="KOSPI200",                # Market type
    weights=strategy_weights_df,    # Portfolio weights (pd.DataFrame)
    init_invest=100_000_000,       # Initial investment (KRW)
    buy_commission=0.0025,         # Buy commission rate (0.25%)
    sell_commission=0.0025,        # Sell commission rate (0.25%)
    slippage=0.001,               # Slippage rate (0.1%)
    sell_tax=0.003,               # Sell tax rate (0.3%)
    cash_rate=0.03,               # Annual cash interest rate (3%)
    rebal_timing='next'           # Rebalancing timing ('next' or 'now')
)
```

#### Key Properties and Methods

```python
# Access key properties
trading_days = portfolio_constructor.bday_dates          # All trading days
rebal_dates = portfolio_constructor.rebalancing_dates    # Rebalancing dates
costs = portfolio_constructor.transaction_costs_summary   # Transaction costs
cash = portfolio_constructor.cash_balance_summary        # Cash balances
returns = portfolio_constructor.pf_ret                   # Portfolio returns
holdings = portfolio_constructor.pf_quantity             # Holdings quantity
```

### Rebalancing Timing Logic

<details>
<summary>Click to expand rebalancing timing details</summary>

The `rebal_timing` parameter determines how rebalancing dates are set from formation dates:

#### 1. **`next` option** (Default and Recommended)

```python
# Example timeline with 'next' option
formation_date = '2023-01-15'    # Sunday (non-trading day)
next_trading_day = '2023-01-16'  # Monday
rebalancing_date = '2023-01-17'  # Tuesday (rebalancing occurs here)

# Python implementation
idx = np.searchsorted(trading_days, formation_date, side='right')
rebalancing_date = trading_days[idx]
```

- Always uses the trading day **after** the formation date
- Ensures clean separation between signal generation and execution
- Helps avoid look-ahead bias
- Recommended for most backtesting scenarios

#### 2. **`now` option**

```python
# Example timeline with 'now' option
formation_date = '2023-01-16'    # Monday (trading day)
rebalancing_date = '2023-01-16'  # Same day rebalancing

# Another example
formation_date = '2023-01-15'    # Sunday (non-trading day)
rebalancing_date = '2023-01-16'  # Next trading day

# Python implementation
idx = np.searchsorted(trading_days, formation_date, side='left')
rebalancing_date = trading_days[idx]
```

- Rebalances on the formation date if it's a trading day
- Use when immediate execution is required
- Caution: May introduce look-ahead bias if not carefully implemented

#### Example Usage:

```python
# Configure rebalancing timing
config = {
    'rebal_timing': 'next',  # or 'now'
    # ... other parameters
}

# Check rebalancing dates
print("Formation dates:", portfolio_constructor.formation_dates)
print("Rebalancing dates:", portfolio_constructor.rebalancing_dates)

# Example output:
# Formation dates: ['2023-01-15', '2023-02-15', '2023-03-15']
# Rebalancing dates: ['2023-01-17', '2023-02-16', '2023-03-16']
```

</details>

### Cash Constraints and Scaling Factor Logic

<details>
<summary>Click to expand cash handling details</summary>

The framework uses a scaling factor approach to handle cash constraints:

#### 1. Ideal Portfolio Calculation

```python
# Example of ideal portfolio calculation
ideal_weights = {
    'Stock1': 0.3,  # 30% allocation
    'Stock2': 0.4,  # 40% allocation
    'Stock3': 0.3   # 30% allocation
}

total_assets = 100_000_000  # KRW 100M
prices = {
    'Stock1': 50000,
    'Stock2': 30000,
    'Stock3': 40000
}

# Calculate ideal quantities
ideal_quantities = {
    stock: (weight * total_assets) // price
    for stock, (weight, price) in
    zip(ideal_weights.keys(), zip(ideal_weights.values(), prices.values()))
}

# Example output:
# {
#     'Stock1': 600,   # 30M / 50000
#     'Stock2': 1333,  # 40M / 30000
#     'Stock3': 750    # 30M / 40000
# }
```

#### 2. Cash Requirement Calculation

```python
# Example cash flow calculation
def calculate_cash_needs(ideal_quantities, current_quantities, prices, costs):
    buy_value = sum(
        prices[stock] * (ideal_quantities[stock] - current_quantities[stock])
        for stock in ideal_quantities
        if ideal_quantities[stock] > current_quantities[stock]
    )

    sell_value = sum(
        prices[stock] * (current_quantities[stock] - ideal_quantities[stock])
        for stock in ideal_quantities
        if current_quantities[stock] > ideal_quantities[stock]
    )

    # Calculate costs
    buy_costs = buy_value * (costs['commission'] + costs['slippage'])
    sell_costs = sell_value * (costs['commission'] + costs['slippage'] + costs['tax'])

    return buy_value, sell_value, buy_costs, sell_costs

# Example usage:
cash_needs = calculate_cash_needs(
    ideal_quantities,
    current_quantities,
    prices,
    {'commission': 0.0025, 'slippage': 0.001, 'tax': 0.003}
)
```

#### 3. Scaling Factor Application

```python
def calculate_scaling_factor(cash_available, buy_value, buy_costs):
    total_cash_needed = buy_value + buy_costs

    if total_cash_needed <= cash_available:
        return 1.0  # Can execute full trade
    elif total_cash_needed <= 0:
        return 0.0  # No buys needed
    else:
        # Scale down proportionally
        return max(0.0, min(1.0, cash_available / total_cash_needed))

# Example:
cash_available = 50_000_000  # KRW 50M available
buy_value = 80_000_000      # KRW 80M needed for buys
buy_costs = 2_000_000       # KRW 2M in costs

scale_factor = calculate_scaling_factor(cash_available, buy_value, buy_costs)
# Output: 0.61 (can execute ~61% of desired trades)
```

#### 4. Final Trade Execution

```python
def apply_scaling_factor(ideal_quantities, current_quantities, scale_factor):
    """Apply scaling factor while maintaining portfolio ratios"""
    scaled_trades = {}
    for stock in ideal_quantities:
        trade_size = ideal_quantities[stock] - current_quantities[stock]
        scaled_trade = int(trade_size * scale_factor)
        scaled_quantities[stock] = current_quantities[stock] + scaled_trade
    return scaled_quantities

# Example output with scale_factor = 0.61:
# Original ideal trades: Buy 600 Stock1, Sell 200 Stock2, Buy 300 Stock3
# Scaled trades: Buy 366 Stock1, Sell 200 Stock2, Buy 183 Stock3
```

#### Real-world Example:

```python
# Complete rebalancing example
class PortfolioRebalancer:
    def __init__(self, cash, holdings, prices, costs):
        self.cash = cash
        self.holdings = holdings
        self.prices = prices
        self.costs = costs

    def rebalance(self, target_weights):
        # 1. Calculate ideal quantities
        total_assets = self.cash + sum(
            self.prices[stock] * qty
            for stock, qty in self.holdings.items()
        )
        ideal_quantities = self._calculate_ideal_quantities(
            target_weights, total_assets
        )

        # 2. Calculate cash needs
        buy_value, sell_value, buy_costs, sell_costs = \
            self._calculate_cash_needs(ideal_quantities)

        # 3. Apply scaling if needed
        cash_after_sells = self.cash + sell_value - sell_costs
        scale_factor = self._calculate_scaling_factor(
            cash_after_sells, buy_value, buy_costs
        )

        # 4. Execute scaled trades
        final_quantities = self._apply_scaling_factor(
            ideal_quantities, scale_factor
        )

        return final_quantities

# Usage example:
rebalancer = PortfolioRebalancer(
    cash=50_000_000,
    holdings={'Stock1': 500, 'Stock2': 800, 'Stock3': 600},
    prices={'Stock1': 50000, 'Stock2': 30000, 'Stock3': 40000},
    costs={'commission': 0.0025, 'slippage': 0.001, 'tax': 0.003}
)

new_portfolio = rebalancer.rebalance({
    'Stock1': 0.3,
    'Stock2': 0.4,
    'Stock3': 0.3
})
```

</details>

### Transaction Cost Calculation

<details>
<summary>Click to expand transaction cost details</summary>

The `_calculate_transaction_costs` method provides detailed cost analysis:

#### Cost Components

```python
class TransactionCostCalculator:
    def __init__(self, buy_commission, sell_commission, slippage, sell_tax):
        self.buy_commission = buy_commission   # e.g., 0.0025 (0.25%)
        self.sell_commission = sell_commission # e.g., 0.0025 (0.25%)
        self.slippage = slippage              # e.g., 0.001 (0.1%)
        self.sell_tax = sell_tax              # e.g., 0.003 (0.3%)

    def calculate_costs(self, trades, prices):
        """
        Calculate transaction costs for a set of trades

        Parameters:
        - trades: Dict[str, int] - Number of shares to trade (+ for buy, - for sell)
        - prices: Dict[str, float] - Current prices for each stock

        Returns:
        - Dict containing cost breakdown and total costs
        """
        costs = {
            'buy_value': 0,
            'sell_value': 0,
            'buy_commission': 0,
            'sell_commission': 0,
            'slippage': 0,
            'sell_tax': 0,
            'total_costs': 0
        }

        for stock, quantity in trades.items():
            value = abs(quantity * prices[stock])

            if quantity > 0:  # Buy
                costs['buy_value'] += value
                costs['buy_commission'] += value * self.buy_commission
                costs['slippage'] += value * self.slippage
            else:  # Sell
                costs['sell_value'] += value
                costs['sell_commission'] += value * self.sell_commission
                costs['slippage'] += value * self.slippage
                costs['sell_tax'] += value * self.sell_tax

        costs['total_costs'] = sum([
            costs['buy_commission'],
            costs['sell_commission'],
            costs['slippage'],
            costs['sell_tax']
        ])

        return costs

# Usage example:
calculator = TransactionCostCalculator(
    buy_commission=0.0025,
    sell_commission=0.0025,
    slippage=0.001,
    sell_tax=0.003
)

trades = {
    'Stock1': 100,   # Buy 100 shares
    'Stock2': -50,   # Sell 50 shares
    'Stock3': 200    # Buy 200 shares
}

prices = {
    'Stock1': 50000,
    'Stock2': 30000,
    'Stock3': 40000
}

costs = calculator.calculate_costs(trades, prices)
```

#### Example Cost Analysis Output:

```python
# Example transaction cost summary
{
    'buy_value': 13_000_000,      # KRW 13M in buys
    'sell_value': 1_500_000,      # KRW 1.5M in sells
    'buy_commission': 32_500,      # KRW 32.5K (0.25% of buy value)
    'sell_commission': 3_750,      # KRW 3.75K (0.25% of sell value)
    'slippage': 14_500,           # KRW 14.5K (0.1% of total value)
    'sell_tax': 4_500,            # KRW 4.5K (0.3% of sell value)
    'total_costs': 55_250         # KRW 55.25K total costs
}
```

#### Cost Impact Analysis:

```python
def analyze_cost_impact(costs, total_portfolio_value):
    """Analyze the impact of transaction costs"""

    cost_basis_points = {
        category: (value / total_portfolio_value) * 10000
        for category, value in costs.items()
    }

    return {
        'absolute_costs': costs,
        'basis_points': cost_basis_points,
        'total_impact_bps': cost_basis_points['total_costs']
    }

# Example usage:
portfolio_value = 100_000_000  # KRW 100M
impact = analyze_cost_impact(costs, portfolio_value)

# Output example:
# {
#     'absolute_costs': {...},
#     'basis_points': {
#         'buy_commission': 0.325,  # 0.325 bps
#         'sell_commission': 0.0375,
#         'slippage': 0.145,
#         'sell_tax': 0.045,
#         'total_costs': 0.5525     # 0.5525 bps total impact
#     }
# }
```

</details>

</details>

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

[Your License] - See LICENSE file for details
