from bt.methodology_type import MethodologyType

# --- Analysis Mode ---
# 'FACTOR': Compare different factor files (original behavior).
# 'FREQUENCY': Compare different frequencies for a single factor file (new behavior).
ANALYSIS_MODE = 'FREQUENCY'  # Change to 'FREQUENCY' to run the new analysis

common_params = {
    'init_invest': 1e8,
    'mkt': 'KOSPI200',
    'start_date': '20200101',
    'end_date': '20250627',
    'methodology_type': MethodologyType.MethodologyPriceTrendsAbs,
    'multiplier': 'Y',
    'buy_commission': 0.0002,
    'sell_commission': 0.0002,
    'slippage': 0.0001,
    'sell_tax': 0.0015,
    'cash_rate': 0.02,
    'rebal_timing': 'same',
    'freq': 'monthly',  # Default frequency for FACTOR mode
    'weight_type': 'ew',
    'quantile': 5,
    'score_threshold': 0.35,
    'inverse_threshold': True,
    'select_lowest': True,
    'keep_empty_periods': True
}

# --- Config for 'FACTOR' mode ---
factor_files = {
    # "i5": "price_trends_avg_test_5.parquet",
    "i20": "price_trends_avg_test_20.parquet",
    # "i60": "price_trends_avg_test_60.parquet",
    # "avg": "price_trends_avg_test_avg.parquet",
    # "[5,20,60]": ["price_trends_avg_test_5.parquet",
    #               "price_trends_avg_test_20.parquet",
    #               "price_trends_avg_test_60.parquet"],
    # "[20,60]": ["price_trends_avg_test_20.parquet",
    #             "price_trends_avg_test_60.parquet"]
}

# --- Config for 'FREQUENCY' mode ---
frequency_analysis_config = {
    "factor_name": "i60",
    "factor_file": "price_trends_avg_test_60.parquet",
    "frequencies": ['weekly', 'monthly', 'quarterly']
}


quintile_positions = {
    "Q1": [1],
    "Q2": [2],
    "Q3": [3],
    "Q4": [4],
    "Q5": [5]
} 