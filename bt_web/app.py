from __version__ import __title__, __version__
from bt.methodology_type import MethodologyType, methodology_clses
from bt.main import PortfolioAnalysis
from bt.methodologies import *
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from flask import Flask, render_template, request, send_file
from io import BytesIO
import json
from datetime import datetime

matplotlib.use('Agg')
plt.show = lambda: None


app = Flask(__name__)

analysis_results = {}


def get_performance_metrics(performance_table):
    portfolio_column = performance_table.columns[0]

    perf_dict = {}
    for idx, value in performance_table[portfolio_column].items():
        metric_name = idx.replace('(%)', '').strip()
        perf_dict[metric_name] = value

        if 'CumRet' in idx:
            perf_dict['Cumulative Return (%)'] = value
        elif 'CAGR' in idx:
            perf_dict['Annualized Return (%)'] = value
        elif 'MDD' in idx:
            perf_dict['Maximum Drawdown (%)'] = value
        elif 'Hit Ratio' in idx:
            perf_dict['Win Rate (%)'] = value
        elif 'Standard Deviation' in idx:
            perf_dict['Standard Deviation (%)'] = value
    return perf_dict


def get_template(df, rename_map=None, date_column='date'):
    """Process DataFrame for template rendering with common operations."""
    df.reset_index(inplace=True)
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]

    if 'index' in df.columns:
        df.rename(columns={'index': date_column}, inplace=True)

    if rename_map:
        for old_col, new_col in rename_map.items():
            if old_col in df.columns:
                df.rename(columns={old_col: new_col}, inplace=True)

    if date_column in df.columns and pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df[date_column] = df[date_column].dt.strftime('%Y-%m-%d')
    return df


def get_holdings_template(df, top_n=10):
    """Process holdings DataFrame with common operations."""
    if isinstance(df.index, pd.Index) and not df.index.name:
        df.reset_index(inplace=True)

    rename_map = {
        'shares': 'quantity',
        'value': 'market_value',
        'ticker': 'ticker',
        'index': 'ticker'
    }
    for old_col, new_col in rename_map.items():
        if old_col in df.columns:
            df.rename(columns={old_col: new_col}, inplace=True)

    df = df.nlargest(top_n, 'weight')
    return df


def get_date_format(date_key):
    """Format date key consistently."""
    if isinstance(date_key, str):
        return date_key
    elif hasattr(date_key, 'strftime'):
        return date_key.strftime('%Y-%m-%d')
    return str(date_key)


@app.route('/')
def index():
    """Displays the list of available methodologies."""
    try:
        methodologies = [
            method_type.name for method_type in methodology_clses.keys()]
    except Exception as e:
        print(f"Error fetching methodologies: {e}")
        methodologies = []
    return render_template('index.html', methodologies=methodologies)


@app.route('/about')
def about():
    """Displays the About page."""
    return render_template('about.html')


@app.route('/help')
def help():
    """Displays the Help page."""
    return render_template('help.html')


@app.route('/terms')
def terms():
    """Displays the Terms and Conditions page."""
    return render_template('terms.html')


@app.route('/privacy')
def privacy():
    """Displays the Privacy Policy page."""
    return render_template('privacy.html')


@app.route('/contact')
def contact():
    """Displays the Contact page."""
    return render_template('contact.html')


@app.route('/methodology/<string:name>')
def methodology_performance(name):
    """Fetches and displays performance for a specific methodology."""
    try:
        available_methodologies = [
            method_type.name for method_type in methodology_clses.keys()]
        if name not in available_methodologies:
            return render_template('error.html', message=f"Methodology '{name}' not found or not implemented.")

        method_type = getattr(MethodologyType, name)

        params = {
            'init_invest': float(request.args.get('initial_investment', 100000000)),
            'mkt': request.args.get('market', 'KOSPI200'),
            'start_date': request.args.get('start_date', '20200101'),
            'end_date': request.args.get('end_date', '20250331'),
            'methodology_type': method_type,
            'multiplier': request.args.get('multiplier', 'Y'),
            'buy_commission': float(request.args.get('buy_commission', 0.0002)),
            'sell_commission': float(request.args.get('sell_commission', 0.0002)),
            'slippage': float(request.args.get('slippage', 0.0001)),
            'sell_tax': float(request.args.get('sell_tax', 0.003)),
            'cash_rate': float(request.args.get('cash_rate', 0.02)),
            'rebal_timing': request.args.get('rebal_timing', 'next'),
            'freq': request.args.get('freq', 'monthly'),
            'quantile': int(request.args.get('quantile', 5)),
            'quantile_position': json.loads(request.args.get('quantile_position', '[1]')),
            'weight_type': request.args.get('weight_type', 'ew')
        }

        try:
            analysis = PortfolioAnalysis.run(**params)

            cache_key = f"{name}_{request.query_string.decode('utf-8')}"
            analysis_results[cache_key] = analysis

            perf_metrics = {}
            if analysis.perf_msre:
                perf_table = analysis.perf_msre.performance_table()
                perf_data = get_performance_metrics(perf_table)

                perf_metrics = {
                    "name": name,
                    "total_return": f"{perf_data.get('Cumulative Return (%)', 0):.2f}%",
                    "annualized_return": f"{perf_data.get('Annualized Return (%)', 0):.2f}%",
                    "sharpe_ratio": round(perf_data.get('Sharpe Ratio', 0), 2),
                    "volatility": f"{perf_data.get('Standard Deviation (%)', 0):.2f}%",
                    "max_drawdown": f"{perf_data.get('Maximum Drawdown (%)', 0):.2f}%",
                    "win_rate": f"{perf_data.get('Win Rate (%)', 0):.2f}%"
                }

                if analysis.portfolio_constructor and analysis.portfolio_constructor.results is not None:
                    res = analysis.portfolio_constructor.results
                    if not res.empty:
                        pf_cumret = (1 + analysis.perf_msre.pf_ret).cumprod()
                        bm_cumret = (1 + analysis.perf_msre.bm_ret).cumprod()

                        def calculate_drawdown_series(returns):
                            cumulative = (1 + returns).cumprod()
                            rolling_max = cumulative.expanding().max()
                            drawdown = (cumulative - rolling_max) / rolling_max
                            return drawdown

                        pf_mdd = calculate_drawdown_series(
                            analysis.perf_msre.pf_ret)
                        bm_mdd = calculate_drawdown_series(
                            analysis.perf_msre.bm_ret)

                        chart_data = pd.DataFrame({
                            'date': pf_cumret.index,
                            'portfolio': pf_cumret.iloc[:, 0],
                            'benchmark': bm_cumret.iloc[:, 0],
                            'portfolio_mdd': pf_mdd.iloc[:, 0],
                            'benchmark_mdd': bm_mdd.iloc[:, 0]
                        }).reset_index(drop=True)

                        chart_data['date'] = chart_data['date'].dt.strftime(
                            '%Y-%m-%d')
                        perf_metrics['chart_data'] = chart_data.to_dict(
                            'records')

            transaction_costs = None
            cash_balance = None
            holdings_snapshot = None

            if analysis.portfolio_constructor:
                transaction_costs = analysis.transaction_costs_summary
                cash_balance = analysis.cash_balance_summary
                holdings_snapshot = analysis.holdings_snapshot

            transactions_data = None
            if transaction_costs is not None and not transaction_costs.empty:
                try:
                    renamed_cols = {
                        'total_buy_value': 'buy_value',
                        'total_sell_value': 'sell_value',
                        'total_buy_cost': 'buy_cost',
                        'total_sell_cost': 'sell_cost',
                        'total_transaction_cost': 'total_cost',
                        'total_nav': 'nav',
                        'transaction_cost_(bp)': 'cost_bp'
                    }
                    transactions_df = get_template(transaction_costs,
                                                   rename_map=renamed_cols)
                    transactions_data = transactions_df.to_dict('records')
                except Exception as e:
                    print(f"Error processing transaction costs: {e}")
                    import traceback
                    traceback.print_exc()
                    transactions_data = None

            cash_data = None
            if cash_balance is not None and not cash_balance.empty:
                try:
                    cash_df = get_template(cash_balance)
                    cash_data = cash_df.to_dict('records')
                except Exception as e:
                    print(f"Error processing cash balance: {e}")
                    import traceback
                    traceback.print_exc()
                    cash_data = None

            holdings_data = {}
            if holdings_snapshot:
                for date_key, holdings in holdings_snapshot.items():
                    try:
                        if not holdings.empty:
                            date_str = get_date_format(date_key)
                            holdings_df = get_holdings_template(holdings)
                            holdings_data[date_str] = holdings_df.to_dict(
                                'records')
                    except Exception as e:
                        print(
                            f"Error processing holdings for date {date_key}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue

            performance_data = {
                **perf_metrics,
                'transactions': transactions_data,
                'cash_balance': cash_data,
                'holdings': holdings_data,
                'params': params
            }

        except Exception as e:
            print(f"Error running backtest for {name}: {e}")
            import traceback
            traceback.print_exc()
            return render_template('error.html', message=f"Error running backtest for {name}: {e}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        return render_template('error.html', message=f"An unexpected error occurred: {e}")

    return render_template('performance.html', data=performance_data)


@app.route('/download_data/<string:name>')
def download_data(name):
    """Downloads all analysis data as an Excel file."""
    try:
        cache_key = f"{name}_{request.query_string.decode('utf-8')}"
        analysis = analysis_results.get(cache_key)

        if not analysis:
            for key, stored_analysis in analysis_results.items():
                if key.startswith(f"{name}_"):
                    analysis = stored_analysis
                    break

        if not analysis:
            return {"error": "No analysis data found. Please run the backtest first."}, 404

        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            has_data = False

            if analysis.perf_msre:
                perf_table = analysis.perf_msre.performance_table()
                perf_data = get_performance_metrics(perf_table)

                summary_data = {
                    'Metric': [
                        'Total Return',
                        'Annualized Return',
                        'Sharpe Ratio',
                        'Volatility',
                        'Maximum Drawdown',
                        'Win Rate'
                    ],
                    'Value': [
                        f"{perf_data.get('Cumulative Return (%)', 0):.2f}%",
                        f"{perf_data.get('Annualized Return (%)', 0):.2f}%",
                        round(perf_data.get('Sharpe Ratio', 0), 2),
                        f"{perf_data.get('Standard Deviation (%)', 0):.2f}%",
                        f"{perf_data.get('Maximum Drawdown (%)', 0):.2f}%",
                        f"{perf_data.get('Win Rate (%)', 0):.2f}%"
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                has_data = True

                if hasattr(analysis.perf_msre, 'pf_ret') and hasattr(analysis.perf_msre, 'bm_ret'):
                    pf_ret = analysis.perf_msre.pf_ret
                    bm_ret = analysis.perf_msre.bm_ret

                    pf_cumret = (1 + pf_ret).cumprod()
                    bm_cumret = (1 + bm_ret).cumprod()

                    def calculate_drawdown_series(returns):
                        cumulative = (1 + returns).cumprod()
                        rolling_max = cumulative.expanding().max()
                        drawdown = (cumulative - rolling_max) / rolling_max
                        return drawdown

                    pf_mdd = calculate_drawdown_series(pf_ret)
                    bm_mdd = calculate_drawdown_series(bm_ret)

                    chart_data = pd.DataFrame({
                        'date': pf_ret.index,
                        'portfolio_returns': pf_ret.iloc[:, 0],
                        'benchmark_returns': bm_ret.iloc[:, 0],
                        'portfolio_cumulative': pf_cumret.iloc[:, 0],
                        'benchmark_cumulative': bm_cumret.iloc[:, 0],
                        'portfolio_drawdown': pf_mdd.iloc[:, 0],
                        'benchmark_drawdown': bm_mdd.iloc[:, 0]
                    })

                    chart_data.to_excel(
                        writer, sheet_name='Returns_and_MDD', index=False)
                    has_data = True

            if hasattr(analysis, 'transaction_costs_summary') and analysis.transaction_costs_summary is not None:
                transaction_costs = analysis.transaction_costs_summary
                if not transaction_costs.empty:
                    transaction_costs.to_excel(
                        writer, sheet_name='Transaction_Costs', index=True)
                    has_data = True

            if hasattr(analysis, 'cash_balance_summary') and analysis.cash_balance_summary is not None:
                cash_balance = analysis.cash_balance_summary
                if not cash_balance.empty:
                    cash_balance.to_excel(
                        writer, sheet_name='Cash_Balance', index=True)
                    has_data = True

            if hasattr(analysis, 'holdings_snapshot') and analysis.holdings_snapshot:
                holdings_snapshot = analysis.holdings_snapshot
                for date_key, holdings in holdings_snapshot.items():
                    if not holdings.empty:
                        date_str = get_date_format(date_key)
                        sheet_name = f"h_{date_str}"
                        if len(sheet_name) > 31:
                            sheet_name = sheet_name[:31]

                        holdings_copy = holdings.copy()
                        holdings_copy['date'] = date_str

                        cols = holdings_copy.columns.tolist()
                        cols = ['date'] + \
                            [col for col in cols if col != 'date']

                        holdings_copy = holdings_copy[cols]
                        holdings_copy.to_excel(
                            writer, sheet_name=sheet_name, index=True)
                        has_data = True

            params = getattr(analysis, 'params', None)
            if params:
                params_dict = {}
                for k, v in params.items():
                    if k == 'methodology_type' and hasattr(v, 'name'):
                        params_dict[k] = v.name
                    else:
                        params_dict[k] = v

                params_df = pd.DataFrame([params_dict]).T
                params_df.columns = ['Value']
                params_df.index.name = 'Parameter'
                params_df.to_excel(writer, sheet_name='Parameters')
                has_data = True

            if not has_data:
                empty_df = pd.DataFrame({'Message': ['No data available']})
                empty_df.to_excel(writer, sheet_name='Info', index=False)

        output.seek(0)
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=f'backtest_data_{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
        )

    except Exception as e:
        print(f"Error generating Excel file: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}, 500


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500


@app.context_processor
def inject_metadata():
    """Inject metadata into templates."""
    return {
        'app_name': __title__,
        'app_version': __version__
    }


if __name__ == '__main__':
    print("Available Methodologies:", [
          m.name for m in methodology_clses.keys()])
    app.run(debug=True, port=5001, host='0.0.0.0')
