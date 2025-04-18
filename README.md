# Backtest

A Python-based quantitative investment strategy backtesting framework with web visualization.

## Overview

This framework provides a comprehensive solution for backtesting quantitative investment strategies based on financial statement analysis. It includes performance analysis, transaction cost calculation, and web-based visualization of results.

## Key Features

- Financial statement-based strategy implementations
- Portfolio performance analysis with key metrics
- Transaction cost analysis including commissions, taxes, and slippage
- Web-based visualization dashboard
- Excel report generation
- Automated development environment setup

## Project Structure

```
Backtest/
├── q_root/                  # Core backtesting package
│   ├── bt/                 # Main backtesting module
│   │   ├── __init__.py
│   │   ├── __version__.py  # Version information
│   │   ├── main.py        # Core functionality
│   │   ├── methodologies/ # Strategy implementations
│   │   │   ├── methodology_err_chg.py    # Earnings change analysis
│   │   │   ├── methodology_opr_chg.py    # Operating profit analysis
│   │   │   ├── methodology_sales_yoy.py  # Sales growth analysis
│   │   │   └── methodology_data_validation.py
│   │   ├── utils/        # Utility functions
│   │   └── data/         # Data handling
│   ├── setup.sh           # Development environment setup
│   ├── setup.py          # Package setup configuration
│   ├── requirements.txt   # Python dependencies
│   └── .env              # Environment variables
│
├── bt_web/                 # Web interface
│   ├── app.py             # Flask web application
│   ├── templates/         # HTML templates
│   │   ├── index.html
│   │   ├── performance.html
│   │   └── ...
│   └── static/            # Static assets
│       ├── css/
│       ├── js/
│       └── images/
│
├── .gitignore             # Git ignore rules
├── .vscode/               # VS Code/Cursor settings
│   └── settings.json     # Editor configuration
└── README.md              # This file
```

## Hidden Files and Their Purpose

- `.gitignore`: Specifies which files Git should ignore

  ```
  __pycache__/
  *.pyc
  *.pyo
  *.pyd
  .Python
  env/
  venv/
  .env
  .venv
  .idea/
  .vscode/
  ```

- `.env`: Environment configuration

  ```
  PYTHONPATH=/path/to/project/q_root
  ```

- `.vscode/settings.json`: Editor settings
  ```json
  {
    "python.defaultInterpreterPath": "/path/to/python",
    "python.analysis.extraPaths": ["/path/to/project/q_root"]
  }
  ```

## Prerequisites

- Python 3.9 or higher
- Conda (recommended for environment management)
- Git

## Installation

1. Clone the repository

```bash
git clone https://github.com/yourusername/Backtest.git
cd Backtest
```

2. Create and activate Python virtual environment (recommended)

```bash
conda create -n myquants python=3.9
conda activate myquants
```

3. Install the package

```bash
cd q_root
./setup.sh
```

The `setup.sh` script automatically:

- Installs the bt package in development mode
- Configures VS Code/Cursor settings
- Sets up Python interpreter paths
- Creates necessary environment variables

## Web Interface

1. Start the Flask application

```bash
cd bt_web
python app.py
```

2. Access the dashboard at `http://localhost:5001`

## Available Backtesting Methodologies

### Financial Statement Analysis Strategies

- **Earnings Change Analysis (`methodology_err_chg.py`)**

  - Net income change analysis
  - Earnings trend detection
  - Profitability change signals

- **Operating Profit Analysis (`methodology_opr_chg.py`)**

  - Operating profit change analysis
  - Operating margin trends
  - Operational efficiency metrics

- **Sales Growth Analysis (`methodology_sales_yoy.py`)**
  - Year-over-Year (YoY) sales growth
  - Revenue trend analysis
  - Growth momentum indicators

### Data Validation (`methodology_data_validation.py`)

The framework includes robust data validation mechanisms to ensure:

- Financial statement data integrity
- Proper handling of missing values
- Outlier detection and treatment
- Data quality and consistency checks

### Custom Strategies

Implement your own strategy by extending the base Strategy class:

```python
from bt.methodology_type import BaseMethodology

class CustomStrategy(BaseMethodology):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self):
        # Implement your strategy logic here
        pass
```

## Backtesting Parameters

### Basic Parameters

- `init_invest`: Initial investment amount
- `mkt`: Target market (e.g., 'KOSPI200')
- `start_date`: Backtest start date (YYYYMMDD)
- `end_date`: Backtest end date (YYYYMMDD)

### Trading Parameters

- `freq`: Rebalancing frequency ('daily', 'weekly', 'monthly')
- `buy_commission`: Buy commission rate (default: 0.0002)
- `sell_commission`: Sell commission rate (default: 0.0002)
- `slippage`: Slippage rate (default: 0.0001)
- `sell_tax`: Sell tax rate (default: 0.003)
- `cash_rate`: Risk-free rate (default: 0.02)

### Strategy Parameters

- `quantile`: Number of quantiles for factor sorting (default: 5)
- `quantile_position`: List of quantiles to invest in (e.g., [1] for top quantile)
- `weight_type`: Weighting scheme ('ew': equal weight, 'vw': value weight)

## Analysis Results

### Performance Metrics

- Cumulative Return
- Annualized Return (CAGR)
- Sharpe Ratio
- Maximum Drawdown (MDD)
- Win Rate
- Information Ratio
- Sortino Ratio
- Beta
- Alpha

### Transaction Analysis

- Trading volume
- Turnover ratio
- Transaction costs breakdown
- Tax implications

### Position Information

- Current holdings
- Position weights
- Sector allocation
- Risk exposure

## Data Export

Export all analysis results to Excel with the following sheets:

1. **Summary**

   - Key performance indicators
   - Risk metrics
   - Strategy parameters

2. **Returns_and_MDD**

   - Daily returns
   - Cumulative returns
   - Drawdown series

3. **Transaction_Costs**

   - Trading volume
   - Commission costs
   - Tax amounts
   - Slippage impact

4. **Cash_Balance**

   - Daily cash positions
   - Interest earned

5. **Holdings**
   - Daily position snapshots
   - Weight distribution
   - Sector allocation

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
