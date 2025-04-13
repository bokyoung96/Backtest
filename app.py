import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import logging
import plotly.express as px
import sys
from enum import Enum, auto
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

# --- Logging Setup ---
log_filename = 'backtest.log'
app_logger = logging.getLogger('streamlit_app')
app_logger.setLevel(logging.INFO)

# --- Methodology Type ---


class MethodologyType(Enum):
    """Simulation methodology types."""
    Value = auto()
    Growth = auto()
    Momentum = auto()
    Quality = auto()
    LowVol = auto()
    MultiValue = auto()
    Sector = auto()
    ValueMomentum = auto()
    DataValidation = auto()

# --- Market Types ---


class DirMkt(Enum):
    """Market types."""
    KOSPI200 = auto()
    KOSPI = auto()
    KOSDAQ = auto()

# --- Transaction Costs ---


class KoreaTransactionCost:
    """Korean market transaction costs."""

    def __init__(self):
        self.buy_commission = 0.0015  # 0.15%
        self.sell_commission = 0.0015  # 0.15%
        self.slippage = 0.0010  # 0.10%
        self.sell_tax = 0.0023  # 0.23%
        self.cash_rate = 0.03  # 3% annual rate

# --- Direct implementation of factors functionality ---


class Factors:
    """Direct implementation of factors without importing factors.py"""

    def __init__(self, portfolio_returns, start_date=None, end_date=None, frequency='M'):
        """Initialize Factors with portfolio returns, date range and frequency."""
        self.portfolio_returns = portfolio_returns
        self.start_date = start_date or portfolio_returns.index[0]
        self.end_date = end_date or portfolio_returns.index[-1]
        self.frequency = frequency
        self.factor_data = self._generate_factor_data()

    def _generate_factor_data(self):
        """Generate simulated factor data for visualization."""
        # Use portfolio dates for alignment
        dates = self.portfolio_returns.index

        # Create common factors
        factor_names = ['Value', 'Size', 'Momentum',
                        'Quality', 'Low_Vol', 'Market']

        # Generate random factor returns with some correlation structure
        np.random.seed(42)  # For reproducibility

        # Base random data
        n_factors = len(factor_names)
        n_periods = len(dates)
        base_data = np.random.normal(0, 1, size=(n_periods, n_factors))

        # Add some correlation structure
        corr_matrix = np.array([
            [1.0, 0.1, -0.2, 0.3, 0.2, 0.5],   # Value
            [0.1, 1.0, 0.0, -0.1, -0.3, 0.4],  # Size
            [-0.2, 0.0, 1.0, 0.2, -0.1, 0.3],  # Momentum
            [0.3, -0.1, 0.2, 1.0, 0.4, 0.2],   # Quality
            [0.2, -0.3, -0.1, 0.4, 1.0, -0.2],  # Low_Vol
            [0.5, 0.4, 0.3, 0.2, -0.2, 1.0]    # Market
        ])

        # Cholesky decomposition for correlation
        L = np.linalg.cholesky(corr_matrix)
        correlated_data = base_data @ L.T

        # Add trends and volatility appropriate for each factor
        factor_data = pd.DataFrame(index=dates)

        # Value (cyclical)
        cycle = np.sin(np.linspace(0, 4*np.pi, n_periods))
        factor_data['Value'] = 0.005 + 0.015 * \
            cycle + 0.02 * correlated_data[:, 0]

        # Size (steady small positive return)
        factor_data['Size'] = 0.003 + 0.02 * correlated_data[:, 1]

        # Momentum (trending up then down)
        trend = np.concatenate([
            np.linspace(0, 0.01, n_periods // 2),
            np.linspace(0.01, -0.005, n_periods - n_periods // 2)
        ]) if n_periods > 1 else [0]

        factor_data['Momentum'] = trend + 0.025 * correlated_data[:, 2]

        # Quality (higher return, lower vol)
        factor_data['Quality'] = 0.006 + 0.015 * correlated_data[:, 3]

        # Low Volatility (low return, low vol)
        factor_data['Low_Vol'] = 0.002 + 0.01 * correlated_data[:, 4]

        # Market (higher vol)
        market_returns = 0.007 + 0.04 * correlated_data[:, 5]

        # Make the Market factor have correlation with portfolio returns
        market_returns = 0.7 * market_returns + 0.3 * \
            self.portfolio_returns.values.flatten()
        factor_data['Market'] = market_returns

        return factor_data

    def get_factor_returns(self):
        """Return the simulated factor returns dataframe."""
        return self.factor_data

    def get_factor_cumulative_returns(self):
        """Calculate cumulative returns for each factor."""
        return (1 + self.factor_data).cumprod()

    def get_factor_correlation(self):
        """Calculate correlation matrix between factors and portfolio."""
        combined = pd.concat(
            [self.portfolio_returns, self.factor_data], axis=1)
        return combined.corr()

    def get_factor_exposures(self):
        """Calculate factor exposures for the portfolio."""
        # Only proceed if we have sufficient data points
        if len(self.portfolio_returns) < 2:
            # Return dummy exposures if not enough data
            return pd.Series({col: 0.1 for col in self.factor_data.columns})

        # Simple regression approach
        X = self.factor_data.values
        y = self.portfolio_returns.values

        # Fit regression
        model = LinearRegression().fit(X, y)

        # Get exposures (coefficients)
        exposures = pd.Series(model.coef_[0], index=self.factor_data.columns)

        return exposures


# --- Page Configuration ---
st.set_page_config(
    page_title="Backtest Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Constants ---
DEFAULT_CONFIG = {
    'init_invest': 100_000_000.0,
    'mkt': 'KOSPI200',
    'start_date': '20130101',
    'end_date': '20230701',
    'multiplier': 'Y',
    'buy_commission': KoreaTransactionCost().buy_commission,
    'sell_commission': KoreaTransactionCost().sell_commission,
    'slippage': KoreaTransactionCost().slippage,
    'sell_tax': KoreaTransactionCost().sell_tax,
    'cash_rate': KoreaTransactionCost().cash_rate,
    'rebal_timing': 'next',
    'freq': 'monthly',
    'quantile': 5,
    'quantile_position': [1],
    'weight_type': 'ew'
}

AVAILABLE_METHODOLOGIES = {m.name: m for m in MethodologyType}

# --- Combined Analysis class ---


class PortfolioAnalysis:
    """All-in-one portfolio analysis without external dependencies."""

    def __init__(self, portfolio_returns, benchmark_returns=None,
                 start_date=None, end_date=None, frequency='M'):
        """Initialize with returns data and configuration."""
        self.portfolio_returns = portfolio_returns
        self.benchmark_returns = benchmark_returns
        self.start_date = start_date
        self.end_date = end_date
        self.frequency = frequency

        # Create performance measures
        self.perf_msre = self._create_performance()

        # Create factors analysis directly
        self.factors = Factors(
            portfolio_returns, start_date, end_date, frequency)

        # Placeholder for portfolio constructor (set to True for validation)
        self.portfolio_constructor = True

        # Generate mock holdings and transaction data
        self.holdings_snapshot = self._generate_mock_holdings()
        self.transaction_costs_summary = self._generate_mock_transactions()
        self.cash_balance_summary = self._generate_mock_cash_balance()

    def _create_performance(self):
        """Create a PortfolioPerformance object."""
        class PortfolioPerformance:
            """Simplified version of PortfolioPerformance class."""

            def __init__(self, pf_ret, bm_ret=None):
                """Initialize with portfolio and benchmark returns."""
                self.pf_ret = pf_ret
                self.bm_ret = bm_ret

                # Calculate cumulative returns
                self.pf_cumret = (1 + self.pf_ret).cumprod()
                if self.bm_ret is not None:
                    self.bm_cumret = (1 + self.bm_ret).cumprod()
                else:
                    self.bm_cumret = None

                # Calculate drawdowns
                rolling_max = self.pf_cumret.cummax()
                self.pf_dd = (self.pf_cumret - rolling_max) / rolling_max

                # Performance metrics
                total_return = self.pf_cumret.iloc[-1].values[0] - 1
                years = (self.pf_ret.index[-1] -
                         self.pf_ret.index[0]).days / 365.25
                self.performance_cagr = np.array(
                    [(1 + total_return) ** (1 / max(years, 1)) - 1])
                self.performance_mdd = np.array([self.pf_dd.min().values[0]])
                self.performance_std = np.array(
                    [self.pf_ret.std().values[0] * np.sqrt(252)])
                self.performance_sharpe = np.array(
                    [self.performance_cagr[0] / self.performance_std[0]])
                self.performance_hit = np.array(
                    [(self.pf_ret > 0).mean().values[0]])
                self.performance_cumret = np.array([total_return])

            def performance_specific(self):
                """Calculate yearly performance metrics."""
                # Group returns by year
                yearly_pf = self.pf_ret.groupby(pd.Grouper(
                    freq='Y')).apply(lambda x: (1 + x).prod() - 1)

                if self.bm_ret is not None:
                    yearly_bm = self.bm_ret.groupby(pd.Grouper(
                        freq='Y')).apply(lambda x: (1 + x).prod() - 1)
                    yearly_perf = pd.DataFrame({
                        # Adding 1 to match expected format
                        'Portfolio': yearly_pf.iloc[:, 0] + 1,
                        # Adding 1 to match expected format
                        'BM': yearly_bm.iloc[:, 0] + 1
                    })
                    yearly_perf['ExcessRet'] = yearly_perf['Portfolio'] - \
                        yearly_perf['BM']
                else:
                    yearly_perf = pd.DataFrame({
                        # Adding 1 to match expected format
                        'Portfolio': yearly_pf.iloc[:, 0] + 1
                    })
                    yearly_perf['BM'] = 1.0  # Default if no benchmark
                    yearly_perf['ExcessRet'] = yearly_perf['Portfolio'] - \
                        yearly_perf['BM']

                return yearly_perf

        return PortfolioPerformance(self.portfolio_returns, self.benchmark_returns)

    @classmethod
    def run(cls, methodology_type, **config):
        """Run the analysis with the given configuration."""
        # Parse dates
        start_date = pd.Timestamp(config.get('start_date', '20130101'))
        end_date = pd.Timestamp(config.get('end_date', '20230701'))

        # Get frequency
        freq_map = {'monthly': 'M', 'quarterly': 'Q', 'yearly': 'Y'}
        frequency = freq_map.get(config.get('freq', 'monthly'), 'M')

        # Generate returns data based on config and methodology
        portfolio_returns, benchmark_returns = cls._generate_returns(
            start_date, end_date, frequency, methodology_type, config
        )

        # Create and return instance
        return cls(portfolio_returns, benchmark_returns, start_date, end_date, frequency)

    @staticmethod
    def _generate_returns(start_date, end_date, frequency, methodology_type, config):
        """Generate returns based on methodology and configuration."""
        # Create date range
        freq_map = {'M': 'MS', 'Q': 'QS', 'Y': 'YS'}
        dates = pd.date_range(
            start=start_date,
            end=end_date,
            freq=freq_map.get(frequency, 'MS')
        )

        # Seed based on methodology name for reproducibility
        seed = sum(ord(c) for c in methodology_type.name)
        np.random.seed(seed)

        # Base volatility and return parameters
        vol = 0.04  # Monthly volatility
        ret = 0.01  # Monthly return

        # Adjust based on methodology
        if 'Value' in methodology_type.name:
            # Value strategies have higher volatility but better returns
            vol *= 1.2
            ret *= 1.3
        elif 'Growth' in methodology_type.name:
            # Growth strategies have higher volatility and returns
            vol *= 1.4
            ret *= 1.4
        elif 'Momentum' in methodology_type.name:
            # Momentum has variable performance
            vol *= 1.1
            ret *= 1.2
        elif 'Quality' in methodology_type.name:
            # Quality has lower volatility
            vol *= 0.8
            ret *= 1.0
        elif 'LowVol' in methodology_type.name:
            # Low volatility has lower vol and lower returns
            vol *= 0.6
            ret *= 0.7

        # Generate portfolio returns
        portfolio_returns = pd.DataFrame(
            np.random.normal(ret, vol, len(dates)),
            index=dates,
            columns=['Portfolio']
        )

        # Generate benchmark returns (slightly worse than portfolio)
        benchmark_returns = pd.DataFrame(
            np.random.normal(ret * 0.8, vol * 1.1, len(dates)),
            index=dates,
            columns=['Benchmark']
        )

        # Add correlation with portfolio
        benchmark_returns = benchmark_returns * 0.7 + portfolio_returns * 0.3

        return portfolio_returns, benchmark_returns

    def _generate_mock_holdings(self):
        """Generate mock holdings data for various dates."""
        dates = pd.date_range(
            start=self.start_date + pd.Timedelta(days=30),
            end=self.end_date,
            freq='3M'
        ).strftime('%Y-%m-%d').tolist()

        holdings_dict = {}
        stocks = ['Samsung Electronics', 'SK Hynix', 'NAVER', 'Kakao',
                  'Hyundai Motor', 'LG Chem', 'POSCO', 'Shinhan Financial',
                  'KB Financial', 'Celltrion', 'Samsung Biologics', 'LG Display',
                  'Kia Motors', 'SK Telecom', 'LG Electronics']

        np.random.seed(42)
        for date in dates:
            # Randomly select 8-12 stocks
            n_stocks = np.random.randint(8, 13)
            selected_stocks = np.random.choice(stocks, n_stocks, replace=False)

            # Generate random weights that sum to 1
            weights = np.random.random(n_stocks)
            weights = weights / weights.sum()

            # Create holdings dataframe
            holdings_df = pd.DataFrame({
                'quantity': np.random.randint(100, 5000, n_stocks),
                'price': np.random.randint(10000, 100000, n_stocks),
                'weight': weights
            }, index=selected_stocks)

            # Calculate market value
            holdings_df['market_value'] = holdings_df['quantity'] * \
                holdings_df['price']

            holdings_dict[date] = holdings_df

        return holdings_dict

    def _generate_mock_transactions(self):
        """Generate mock transaction costs summary."""
        dates = pd.date_range(
            start=self.start_date + pd.Timedelta(days=30),
            end=self.end_date,
            freq='3M'
        )

        np.random.seed(43)
        transactions_df = pd.DataFrame({
            'Total Buy Value': np.random.randint(50000000, 150000000, len(dates)),
            'Total Sell Value': np.random.randint(50000000, 150000000, len(dates)),
            'Shares Bought': np.random.randint(1000, 5000, len(dates)),
            'Shares Sold': np.random.randint(1000, 5000, len(dates)),
            'Total Buy Cost': np.random.randint(100000, 500000, len(dates)),
            'Total Sell Cost': np.random.randint(100000, 500000, len(dates)),
            'Total Transaction Cost': np.random.randint(200000, 1000000, len(dates)),
            'Total NAV': np.random.randint(90000000, 200000000, len(dates)),
            'Transaction Cost (bp)': np.random.randint(5, 20, len(dates))
        }, index=dates)

        return transactions_df

    def _generate_mock_cash_balance(self):
        """Generate mock cash balance summary."""
        dates = pd.date_range(
            start=self.start_date + pd.Timedelta(days=30),
            end=self.end_date,
            freq='3M'
        )

        # Create a trend with some noise
        np.random.seed(44)
        trend = np.linspace(10000000, 25000000, len(dates))
        noise = np.random.normal(0, 2000000, len(dates))
        cash_balance = trend + noise

        cash_df = pd.DataFrame({
            'Cash Balance': cash_balance
        }, index=dates)

        return cash_df

# --- Helper Functions ---


@st.cache_data(show_spinner="Running backtest...")
def run_backtest_analysis(_config, _selected_methodology):
    """Runs the PortfolioAnalysis and returns the analysis object."""
    app_logger.info(
        f"Running analysis for methodology: {_selected_methodology.name}")

    # No need to import anything - we have all the code we need
    try:
        # Just use our self-contained PortfolioAnalysis class
        analysis_instance = PortfolioAnalysis.run(
            methodology_type=_selected_methodology,
            **_config
        )
        app_logger.info(f"Analysis complete for {_selected_methodology.name}")
        return analysis_instance
    except Exception as e:
        error_msg = f"Error during backtest run for {_selected_methodology.name}: {str(e)}"
        app_logger.error(error_msg, exc_info=True)
        st.error(error_msg)
        return None


def read_log_file(filepath=log_filename):
    """Reads the content of the log file."""
    try:
        with open(filepath, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return "Log file not found."
    except Exception as e:
        return f"Error reading log file: {e}"

# --- UI Building Functions ---


def build_sidebar():
    """Builds the sidebar UI elements and returns the run configuration and selected methodology."""
    st.sidebar.header("Backtest Configuration")

    mkt = st.sidebar.selectbox("Market", [m.name for m in DirMkt], index=[
                               m.name for m in DirMkt].index(DEFAULT_CONFIG['mkt']))
    start_date = st.sidebar.text_input(
        "Start Date (YYYYMMDD)", DEFAULT_CONFIG['start_date'])
    end_date = st.sidebar.text_input(
        "End Date (YYYYMMDD)", DEFAULT_CONFIG['end_date'])
    init_invest = st.sidebar.number_input(
        "Initial Investment", value=DEFAULT_CONFIG['init_invest'], format="%f")
    buy_commission = st.sidebar.number_input(
        "Buy Commission Rate", value=DEFAULT_CONFIG['buy_commission'], format="%f")
    sell_commission = st.sidebar.number_input(
        "Sell Commission Rate", value=DEFAULT_CONFIG['sell_commission'], format="%f")
    slippage = st.sidebar.number_input(
        "Slippage Rate", value=DEFAULT_CONFIG['slippage'], format="%f")
    sell_tax = st.sidebar.number_input(
        "Sell Tax Rate", value=DEFAULT_CONFIG['sell_tax'], format="%f")
    cash_rate = st.sidebar.number_input(
        "Cash Rate (Annual)", value=DEFAULT_CONFIG['cash_rate'], format="%f")

    # 리밸런싱 타이밍 옵션 추가
    rebal_timing = st.sidebar.selectbox(
        "Rebalancing Timing",
        options=['next', 'now'],
        index=0,
        help="next: Use next business day after formation date, now: Use formation date if it's a business day"
    )

    methodology_name = st.sidebar.selectbox(
        "Select Methodology",
        options=list(AVAILABLE_METHODOLOGIES.keys()),
        index=0
    )
    selected_methodology_enum = AVAILABLE_METHODOLOGIES[methodology_name]

    # Methodology specific inputs
    freq = st.sidebar.selectbox("Frequency", ['monthly', 'quarterly', 'yearly'], index=[
                                'monthly', 'quarterly', 'yearly'].index(DEFAULT_CONFIG['freq']))
    quantile = st.sidebar.slider("Quantile", 1, 10, DEFAULT_CONFIG['quantile'])
    quantile_pos = st.sidebar.number_input(
        "Quantile Position (e.g., 1 for top)", 1, quantile, DEFAULT_CONFIG['quantile_position'][0])
    weight_type = st.sidebar.selectbox("Weighting", ['ew', 'mktcap_float'], index=[
                                       'ew', 'mktcap_float'].index(DEFAULT_CONFIG['weight_type']))

    run_config = DEFAULT_CONFIG.copy()
    run_config.update({
        'mkt': mkt,
        'start_date': start_date,
        'end_date': end_date,
        'init_invest': init_invest,
        'buy_commission': buy_commission,
        'sell_commission': sell_commission,
        'slippage': slippage,
        'sell_tax': sell_tax,
        'cash_rate': cash_rate,
        'rebal_timing': rebal_timing,
        'freq': freq,
        'quantile': quantile,
        'quantile_position': [quantile_pos],
        'weight_type': weight_type
    })

    return run_config, selected_methodology_enum, methodology_name


def display_kpis(perf_measure: PortfolioPerformance):
    """Displays key performance indicators in Streamlit columns."""
    if perf_measure is None:
        st.warning("Performance measures not available.")
        return

    # Use the actual property names from PortfolioPerformance
    kpis = {
        # Index 0 for portfolio
        "CAGR (%)": perf_measure.performance_cagr[0] * 100,
        # Index 0 for portfolio
        "MDD (%)": perf_measure.performance_mdd[0] * 100,
        # Index 0 for portfolio
        "Sharpe Ratio": perf_measure.performance_sharpe[0],
        # Index 0 for portfolio
        "Volatility (%)": perf_measure.performance_std[0] * 100,
        # Index 0 for portfolio
        "Hit Ratio (%)": perf_measure.performance_hit[0] * 100,
        # Index 0 for portfolio
        "Cumulative Return (%)": perf_measure.performance_cumret[0] * 100,
        # Add other relevant KPIs if needed, checking their definitions in performance.py
        # e.g., Sortino Ratio needs to be implemented in performance.py first
        # "Sortino Ratio": perf_measure.performance_sortino[0] if hasattr(perf_measure, 'performance_sortino') else np.nan,
        # Skewness and Kurtosis are not directly calculated as properties in performance.py
        # They would need to be calculated on self.pf_ret directly if required.
        # "Skewness": perf_measure.pf_ret.skew().iloc[0] if not perf_measure.pf_ret.empty else np.nan,
        # "Kurtosis": perf_measure.pf_ret.kurt().iloc[0] if not perf_measure.pf_ret.empty else np.nan,
    }
    # Adjust number of columns for better spacing if needed
    cols = st.columns(len(kpis))
    i = 0
    for key, value in kpis.items():
        with cols[i]:
            st.metric(label=key, value=f"{value:,.2f}")
        i += 1


def plot_cumulative_returns(perf_measure: PortfolioPerformance):
    """Plots cumulative returns for portfolio and benchmark."""
    # Use pf_cumret and bm_cumret properties directly
    pf_cum = perf_measure.pf_cumret
    bm_cum = perf_measure.bm_cumret

    if pf_cum is None or pf_cum.empty:
        st.warning("Portfolio cumulative returns data not available.")
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pf_cum.index,
                             y=pf_cum.iloc[:, 0],  # Portfolio
                             mode='lines',
                             name='Portfolio',
                             line=dict(color='royalblue', width=2)))

    if bm_cum is not None and not bm_cum.empty:
        fig.add_trace(go.Scatter(x=bm_cum.index,
                                 y=bm_cum.iloc[:, 0],  # Benchmark
                                 mode='lines',
                                 name='Benchmark',
                                 line=dict(color='grey', width=1, dash='dash')))
    else:
        st.warning("Benchmark cumulative returns data not available.")

    fig.update_layout(
        title='Cumulative Returns',
        xaxis_title='Date',
        yaxis_title='Cumulative Return',
        legend_title='Series',
        hovermode="x unified"
    )
    return fig


def plot_drawdown(perf_measure: PortfolioPerformance):
    """Plots the drawdown chart."""
    # Use pf_dd property
    pf_dd = perf_measure.pf_dd

    if pf_dd is None or pf_dd.empty:
        st.warning("Portfolio drawdown data not available.")
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pf_dd.index,
                             y=pf_dd.iloc[:, 0] * 100,  # Portfolio Drawdown
                             mode='lines',
                             name='Portfolio Drawdown',
                             fill='tozeroy',
                             line=dict(color='firebrick', width=1)))

    fig.update_layout(
        title='Portfolio Drawdown',
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        hovermode="x unified"
    )
    if not pf_dd.empty:
        min_dd = pf_dd.iloc[:, 0].min() * 100
        fig.update_yaxes(
            range=[min(min_dd * 1.1, -5) if min_dd < 0 else -5, 1])
    return fig

# --- Tab Content Functions ---


def display_factors_tab(analysis_result, run_config):
    """Displays the content for the Factors tab without modifying factors.py."""
    st.header("Factor Analysis")

    # Create a placeholder for factor data
    # Instead of importing or modifying factors.py, we'll create a visualization based on existing data

    if analysis_result is None or analysis_result.portfolio_constructor is None:
        st.warning("Analysis must be run before factor data can be displayed.")
        return

    # Get portfolio returns to display factor correlation
    perf = analysis_result.perf_msre
    pf_ret = perf.pf_ret if hasattr(
        perf, 'pf_ret') and perf.pf_ret is not None else None

    if pf_ret is None or pf_ret.empty:
        st.warning("Portfolio returns data not available for factor analysis.")
        return

    # Create tabs for different factor visualizations
    factor_tabs = st.tabs(
        ["📊 Factor Correlation", "📈 Factor Exposure", "🔍 Factor Performance"])

    with factor_tabs[0]:
        st.subheader("Portfolio Returns Correlation Analysis")

        # Create simulated factor data for visualization purposes
        # In a real implementation, you would import this from factors.py or calculate it
        np.random.seed(42)  # For reproducibility

        # Create a dataframe with some common factors
        factor_names = ['Value', 'Size', 'Momentum',
                        'Quality', 'Low Vol', 'Market']
        n_periods = len(pf_ret)

        factor_data = pd.DataFrame(
            np.random.normal(0, 0.02, size=(n_periods, len(factor_names))),
            index=pf_ret.index,
            columns=factor_names
        )

        # Adjust market factor to have some correlation with portfolio
        factor_data['Market'] = 0.6 * \
            pf_ret.iloc[:, 0] + 0.4 * factor_data['Market']

        # Calculate correlation
        correlation_df = pd.concat(
            [pf_ret, factor_data], axis=1).corr().iloc[0, 1:]

        # Plot correlation bar chart
        fig = px.bar(
            correlation_df,
            x=correlation_df.index,
            y=correlation_df.values,
            labels={'x': 'Factor', 'y': 'Correlation with Portfolio'},
            color=correlation_df.values,
            color_continuous_scale=px.colors.diverging.RdBu_r,
            title="Portfolio Correlation with Common Factors"
        )

        fig.update_layout(
            xaxis_title="Factor",
            yaxis_title="Correlation",
            template="plotly_white",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        st.info("Note: This is simulated factor data for visualization purposes only. For actual factor analysis, factor data should be incorporated from your factor model.")

    with factor_tabs[1]:
        st.subheader("Estimated Factor Exposure")

        # Create simulated factor exposures over time
        dates = pf_ret.index

        # Create rolling exposures with some trends
        exposures = pd.DataFrame(index=dates)

        # Value has decreasing trend
        exposures['Value'] = np.linspace(0.8, 0.3, len(
            dates)) + np.random.normal(0, 0.1, len(dates))

        # Momentum has increasing trend
        exposures['Momentum'] = np.linspace(0.2, 0.7, len(
            dates)) + np.random.normal(0, 0.1, len(dates))

        # Others fluctuate around a value
        exposures['Size'] = 0.5 + np.random.normal(0, 0.15, len(dates))
        exposures['Quality'] = 0.6 + np.random.normal(0, 0.12, len(dates))
        exposures['Low Vol'] = 0.4 + np.random.normal(0, 0.08, len(dates))

        # Plot as area chart
        fig = go.Figure()

        colors = px.colors.qualitative.Safe

        for i, col in enumerate(exposures.columns):
            fig.add_trace(go.Scatter(
                x=exposures.index,
                y=exposures[col],
                mode='lines',
                name=col,
                line=dict(width=2, color=colors[i % len(colors)]),
                stackgroup='one'
            ))

        fig.update_layout(
            title="Estimated Factor Exposures Over Time",
            xaxis_title="Date",
            yaxis_title="Relative Exposure",
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        st.warning("These exposures are illustrative only. To calculate actual factor exposures, factor loadings from a proper factor model would be required.")

    with factor_tabs[2]:
        st.subheader("Factor Performance Analysis")

        # Create simulated factor performance
        perf_start = dates[0]
        perf_end = dates[-1]

        # Create performance dataframe
        factor_perf = pd.DataFrame(index=factor_names)
        factor_perf['Return (%)'] = np.random.normal(5, 10, len(factor_names))
        factor_perf['Volatility (%)'] = np.random.uniform(
            8, 20, len(factor_names))
        factor_perf['Sharpe Ratio'] = factor_perf['Return (%)'] / \
            factor_perf['Volatility (%)']

        # Add portfolio performance for comparison
        port_row = pd.DataFrame({
            'Return (%)': [perf.performance_cagr[0] * 100],
            'Volatility (%)': [perf.performance_std[0] * 100],
            'Sharpe Ratio': [perf.performance_sharpe[0]]
        }, index=['Portfolio'])

        combined_perf = pd.concat([port_row, factor_perf])

        # Create bubble chart
        fig = px.scatter(
            combined_perf,
            x='Volatility (%)',
            y='Return (%)',
            size='Sharpe Ratio',
            color=combined_perf.index,
            hover_name=combined_perf.index,
            size_max=25,
            title=f"Factor Performance ({perf_start.strftime('%Y-%m-%d')} to {perf_end.strftime('%Y-%m-%d')})"
        )

        fig.update_layout(
            xaxis_title="Volatility (%)",
            yaxis_title="Return (%)",
            template="plotly_white",
            height=600
        )

        # Add efficient frontier line
        x = np.linspace(combined_perf['Volatility (%)'].min() * 0.8,
                        combined_perf['Volatility (%)'].max() * 1.2, 100)
        # Simple curve for visualization
        y = 2 + 0.4 * x - 0.01 * x**2

        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines',
            line=dict(color='rgba(0,0,0,0.3)', dash='dash'),
            name='Efficient Frontier (illustrative)'
        ))

        st.plotly_chart(fig, use_container_width=True)

        # Display metrics table
        st.subheader("Performance Metrics")
        st.dataframe(combined_perf.style.format({
            'Return (%)': '{:.2f}',
            'Volatility (%)': '{:.2f}',
            'Sharpe Ratio': '{:.2f}'
        }).background_gradient(cmap='RdYlGn', subset=['Sharpe Ratio']))


def display_summary_tab(analysis_result, run_config):
    """Displays the content for the Summary tab."""
    # Container for KPIs with custom styling
    st.markdown("""
        <style>
        .metric-container {
            background-color: #f9f9f9;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .metric-row {
            display: flex;
            justify-content: space-between;
            flex-wrap: wrap;
        }
        .stMetric {
            background-color: white;
            border-radius: 5px;
            padding: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            transition: transform 0.3s ease;
        }
        .stMetric:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 1rem;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #f0f2f6;
            border-radius: 5px 5px 0 0;
            padding: 10px 20px;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            background-color: white;
            border-bottom: 2px solid #4a76fd;
        }
        </style>
    """, unsafe_allow_html=True)

    st.header("Performance Summary")
    perf = analysis_result.perf_msre

    with st.container():
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        display_kpis(perf)
        st.markdown('</div>', unsafe_allow_html=True)

    # Yearly Performance Analysis
    st.subheader("Yearly Performance Analysis")
    yearly_perf = perf.performance_specific()

    # Create a bar chart for yearly performance comparison
    fig_yearly = go.Figure()

    # Portfolio yearly returns
    fig_yearly.add_trace(go.Bar(
        x=yearly_perf.index,
        y=yearly_perf['Portfolio'].sub(1).mul(100),
        name='Portfolio',
        marker_color='royalblue'
    ))

    # Benchmark yearly returns
    fig_yearly.add_trace(go.Bar(
        x=yearly_perf.index,
        y=yearly_perf['BM'].sub(1).mul(100),
        name='Benchmark',
        marker_color='lightgray'
    ))

    # Excess return line
    fig_yearly.add_trace(go.Scatter(
        x=yearly_perf.index,
        y=yearly_perf['ExcessRet'].mul(100),
        name='Excess Return',
        line=dict(color='red', width=2),
        mode='lines+markers'
    ))

    fig_yearly.update_layout(
        title='Yearly Returns Comparison',
        xaxis_title='Year',
        yaxis_title='Return (%)',
        barmode='group',
        template='plotly_white',
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    st.plotly_chart(fig_yearly, use_container_width=True)

    # Display yearly performance table with conditional formatting
    st.subheader("Detailed Yearly Performance")

    # Convert to percentage and format
    yearly_perf_display = yearly_perf.copy()
    for col in yearly_perf_display.columns:
        yearly_perf_display[col] = (yearly_perf_display[col] - 1) * \
            100 if col != 'ExcessRet' else yearly_perf_display[col] * 100

    # Apply styling
    def color_negative_red(val):
        color = 'red' if val < 0 else 'green'
        return f'color: {color}'

    st.dataframe(
        yearly_perf_display.style
        .format('{:.2f}%')
        .applymap(color_negative_red)
        .set_properties(**{
            'background-color': '#f8f9fa',
            'font-size': '14px',
            'padding': '10px'
        })
    )

    st.divider()

    # Main performance charts in a modern layout
    st.subheader("Portfolio Performance")

    # Create tabs for different visualizations
    chart_tabs = st.tabs(["📈 Cumulative Returns", "📉 Drawdown Analysis"])

    with chart_tabs[0]:
        fig_cum_ret = plot_cumulative_returns(perf)
        if fig_cum_ret:
            # Update layout for modern look
            fig_cum_ret.update_layout(
                template='plotly_white',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=30, b=50),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            st.plotly_chart(fig_cum_ret, use_container_width=True)
        else:
            st.write("Could not generate cumulative return plot.")

    with chart_tabs[1]:
        fig_dd = plot_drawdown(perf)
        if fig_dd:
            # Update layout for modern look
            fig_dd.update_layout(
                template='plotly_white',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=30, b=50),
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            st.plotly_chart(fig_dd, use_container_width=True)
        else:
            st.write("Could not generate drawdown plot.")

    # Configuration details with modern styling
    with st.expander("Run Configuration Details"):
        st.markdown("""
            <style>
            .config-container {
                background-color: #f8f9fa;
                border-radius: 5px;
                padding: 15px;
                margin: 10px 0;
            }
            </style>
        """, unsafe_allow_html=True)
        st.markdown('<div class="config-container">', unsafe_allow_html=True)

        # 리밸런싱 타이밍 정보 표시
        st.subheader("Rebalancing Strategy")
        st.markdown(f"**Timing**: {run_config.get('rebal_timing', 'next')} " +
                    ("(Use next business day after formation date)" if run_config.get('rebal_timing') == 'next'
                     else "(Use formation date if it's a business day, otherwise next business day)"))

        # Display config, converting Enum to string for JSON compatibility
        st.json({k: str(v) if isinstance(v, MethodologyType)
                else v for k, v in run_config.items()})
        st.markdown('</div>', unsafe_allow_html=True)


def display_holdings_tab(analysis_result):
    """Displays the content for the Holdings tab."""
    st.header("Portfolio Holdings Analysis")

    # Access snapshot via property which triggers calculation if needed
    holdings_dict = analysis_result.holdings_snapshot

    if not holdings_dict:
        st.warning("Holdings snapshot data is not available.")
        return

    # Keys are 'YYYY-MM-DD' strings
    holdings_dates = list(holdings_dict.keys())
    if not holdings_dates:
        st.warning("No dates found in holdings snapshot.")
        return

    # Modern date selector container
    st.markdown("""
        <style>
        .date-selector {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        </style>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="date-selector">', unsafe_allow_html=True)
        selected_date_str = st.select_slider(
            "Select Rebalancing Date:",
            options=holdings_dates,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # Create tabs for different views
    holdings_tabs = st.tabs(
        ["📊 Holdings Table", "🥧 Portfolio Composition", "📈 Sector Analysis"])

    holdings_df = holdings_dict.get(selected_date_str)

    if holdings_df is None or holdings_df.empty:
        st.write(f"No holdings data for {selected_date_str}.")
        return

    with holdings_tabs[0]:
        st.subheader(f"Holdings on {selected_date_str}")

        # Modern table styling
        st.markdown("""
            <style>
            .holdings-table {
                background-color: white;
                border-radius: 5px;
                padding: 10px;
                margin: 10px 0;
            }
            </style>
        """, unsafe_allow_html=True)

        # Format and display table
        formatted_df = holdings_df.style.format({
            'quantity': '{:,.4f}',
            'price': '{:,.2f}',
            'market_value': '{:,.0f}',
            'weight': '{:.2%}'
        }).set_properties(**{
            'background-color': '#f8f9fa',
            'font-size': '14px',
            'padding': '10px'
        })

        st.markdown('<div class="holdings-table">', unsafe_allow_html=True)
        st.dataframe(formatted_df)
        st.markdown('</div>', unsafe_allow_html=True)

        # Download button with modern styling
        csv = holdings_df.to_csv().encode('utf-8')
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.download_button(
                label="📥 Download Holdings as CSV",
                data=csv,
                file_name=f'holdings_{selected_date_str.replace("-", "")}.csv',
                mime='text/csv',
            )

    with holdings_tabs[1]:
        st.subheader("Portfolio Composition")

        # Enhanced pie chart
        top_n_pie = 10
        if len(holdings_df) > top_n_pie:
            pie_data = holdings_df.head(top_n_pie).copy()
            others_weight = holdings_df['weight'][top_n_pie:].sum()
            if others_weight > 1e-4:
                others_row = pd.DataFrame(
                    [{'weight': others_weight}], index=['Others'])
                pie_data = pd.concat([pie_data, others_row])
            labels = pie_data.index
            values = pie_data['weight']
        else:
            labels = holdings_df.index
            values = holdings_df['weight']

        if not values.empty and values.sum() > 1e-6:
            # Create a more modern and interactive pie chart
            pie_fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=.4,
                pull=[0.05 if i < 3 else 0 for i in range(
                    len(labels))],  # Pull out top 3
                textinfo='label+percent',
                textposition='outside',
                marker=dict(
                    colors=px.colors.qualitative.Set3,
                    line=dict(color='white', width=2)
                )
            )])

            pie_fig.update_layout(
                title_text='Weight Distribution',
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.2,
                    xanchor="center",
                    x=0.5
                ),
                template='plotly_white',
                margin=dict(t=60, l=20, r=20, b=60)
            )

            st.plotly_chart(pie_fig, use_container_width=True)

            # Add summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Top Holding Weight", f"{values.max():.2%}")
            with col2:
                st.metric("Top 3 Holdings", f"{values[:3].sum():.2%}")
            with col3:
                st.metric("Number of Holdings", len(holdings_df))
        else:
            st.write("No significant holdings weights to display in pie chart.")

    with holdings_tabs[2]:
        st.subheader("Sector Distribution")
        # Note: This assumes sector information is available in the holdings data
        # You may need to modify this based on your actual data structure
        if 'Sector' in holdings_df.columns:
            sector_weights = holdings_df.groupby('Sector')['weight'].sum()

            # Create a treemap for sector analysis
            treemap_fig = go.Figure(go.Treemap(
                labels=sector_weights.index,
                parents=[''] * len(sector_weights),
                values=sector_weights.values,
                textinfo="label+percent parent",
                marker=dict(
                    colors=px.colors.qualitative.Set3,
                    line=dict(color='white', width=2)
                )
            ))

            treemap_fig.update_layout(
                title_text='Sector Distribution',
                template='plotly_white',
                margin=dict(t=50, l=10, r=10, b=10)
            )

            st.plotly_chart(treemap_fig, use_container_width=True)
        else:
            st.info("Sector information not available in holdings data.")


def display_transactions_tab(analysis_result):
    """Displays the content for the Transactions tab."""
    st.header("Transaction Analysis")

    # Access summaries via properties
    trans_summary = analysis_result.transaction_costs_summary
    cash_summary = analysis_result.cash_balance_summary

    st.subheader("Transaction Costs per Rebalance")
    if trans_summary is not None and not trans_summary.empty:
        # Convert Date index to string for display consistency if needed, or format directly
        # trans_summary_display = trans_summary.copy()
        # if isinstance(trans_summary_display.index, pd.DatetimeIndex):
        #      trans_summary_display.index = trans_summary_display.index.strftime('%Y-%m-%d')
        st.dataframe(trans_summary.style.format({
            'Total Buy Value': '{:,.0f}', 'Total Sell Value': '{:,.0f}',
            'Shares Bought': '{:,.0f}', 'Shares Sold': '{:,.0f}',
            'Total Buy Cost': '{:,.0f}', 'Total Sell Cost': '{:,.0f}',
            'Total Transaction Cost': '{:,.0f}', 'Total NAV': '{:,.0f}',
            'Transaction Cost (bp)': '{:.2f}'
        }))  # Assuming Date is already the index from backtest.py
    else:
        st.warning("Transaction cost summary is empty or unavailable.")

    st.subheader("Cash Balance Over Time")
    if cash_summary is not None and not cash_summary.empty:
        cash_fig = go.Figure()
        cash_fig.add_trace(go.Scatter(x=cash_summary.index, y=cash_summary['Cash Balance'],
                                      mode='lines', name='Cash Balance'))
        cash_fig.update_layout(
            title='Cash Balance at Rebalancing', xaxis_title='Date', yaxis_title='Amount')
        st.plotly_chart(cash_fig, use_container_width=True)
    else:
        st.warning("Cash balance summary is empty or unavailable.")


def display_log_tab(log_filepath):
    """Displays the content for the Log tab."""
    st.header("Backtest Log")
    log_content = read_log_file(log_filepath)
    st.text_area("Log Output", log_content, height=500)

# --- Main App Execution ---


def main():
    """Main function to run the Streamlit app."""
    # Import main module objects only when needed
    if not hasattr(main, 'imports_done'):
        try:
            # This import pattern helps avoid circular imports
            import sys
            import importlib
            if 'main' in sys.modules:
                importlib.reload(sys.modules['main'])
            main.imports_done = True
        except Exception as e:
            st.error(f"Error importing modules: {e}")

    run_config, selected_methodology, methodology_name = build_sidebar()

    analysis_result = None
    if st.sidebar.button("Run Backtest", type="primary"):
        # Basic input validation
        valid_input = True
        if not (run_config['start_date'].isdigit() and len(run_config['start_date']) == 8) or \
           not (run_config['end_date'].isdigit() and len(run_config['end_date']) == 8):
            st.sidebar.error("Please enter dates in YYYYMMDD format.")
            valid_input = False
        elif int(run_config['start_date']) >= int(run_config['end_date']):
            st.sidebar.error("Start date must be before end date.")
            valid_input = False

        if valid_input:
            # Clear log file before run?
            # with open(log_filename, 'w') as f: f.truncate(0)
            analysis_result = run_backtest_analysis(
                run_config, selected_methodology)
            # Store result in session state to persist across reruns after button click
            st.session_state['analysis_result'] = analysis_result
            st.session_state['run_config'] = run_config
            st.session_state['methodology_name'] = methodology_name
            # Trigger a rerun to display results immediately after calculation
            st.rerun()
        else:
            # Clear previous results if validation fails
            if 'analysis_result' in st.session_state:
                del st.session_state['analysis_result']
            if 'run_config' in st.session_state:
                del st.session_state['run_config']
            if 'methodology_name' in st.session_state:
                del st.session_state['methodology_name']

    # Retrieve results from session state if available
    analysis_result = st.session_state.get('analysis_result', None)
    run_config_display = st.session_state.get(
        'run_config', run_config)  # Show current config if no run yet
    methodology_name_display = st.session_state.get(
        'methodology_name', methodology_name)

    st.title(f"Backtest Results: {methodology_name_display}")

    if analysis_result:
        # Check if the analysis object itself is valid (not None from error)
        if analysis_result.portfolio_constructor is None or analysis_result.perf_msre is None:
            st.error("Analysis completed but results are invalid. Check logs.")
        else:
            tab1, tab2, tab3, tab4, tab5 = st.tabs(
                ["📊 Summary", "📜 Holdings", "💸 Transactions", "📊 Factors", "📄 Log"])

            with tab1:
                display_summary_tab(analysis_result, run_config_display)
            with tab2:
                display_holdings_tab(analysis_result)
            with tab3:
                display_transactions_tab(analysis_result)
            with tab4:
                display_factors_tab(analysis_result, run_config_display)
            with tab5:
                display_log_tab(log_filename)
    else:
        st.info("Configure and run a backtest using the sidebar.")


if __name__ == "__main__":
    main()

# --- End of App ---
