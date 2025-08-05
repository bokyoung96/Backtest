import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Union
import numpy as np

from bt.main import PortfolioAnalysis
from bt.pt_config import (
    common_params, factor_files, quintile_positions, ANALYSIS_MODE,
    frequency_analysis_config
)
from bt.methodology_type import MethodologyType
from bt.loader import DataLoader

THRESHOLD_METHODOLOGIES = {
    MethodologyType.MethodologyPriceTrendsAbs,
    MethodologyType.MethodologyPriceTrendsAbs1
}


class EventStudy:
    def __init__(self, analysis: PortfolioAnalysis, event_window: tuple = (-30, 60)):
        self.analysis = analysis
        self.start_offset, self.end_offset = event_window
        self.data_loader = DataLoader(mkt=analysis.mkt)
        self.price_data = self.data_loader(data_name='price_adj')
        self.event_results = {}
        
    def run_event_study(self) -> Dict[str, pd.DataFrame]:
        """Run event study on selected stocks from threshold-based methodology"""
        if self.analysis.selected_methodology not in THRESHOLD_METHODOLOGIES:
            print("Event study only available for threshold-based methodologies")
            return {}
            
        # Get selected stocks and dates from methodology
        methodology = self.analysis.methodology
        weights = methodology.weights
        
        # Find dates where stocks were selected (non-zero weights)
        selected_dates = []
        selected_stocks = []
        
        for date in weights.index:
            date_weights = weights.loc[date]
            selected = date_weights[date_weights > 0]
            if not selected.empty:
                selected_dates.append(date)
                selected_stocks.extend(selected.index.tolist())
        
        print(f"Found {len(selected_dates)} selection events with {len(set(selected_stocks))} unique stocks")
        
        # Calculate event returns for each selection
        all_event_returns = {}
        event_count = 0
        
        for date, stock_list in zip(selected_dates, [weights.loc[date][weights.loc[date] > 0].index for date in selected_dates]):
            for stock in stock_list:
                if stock not in self.price_data.columns:
                    continue
                    
                stock_prices = self.price_data[stock]
                if date not in stock_prices.index:
                    continue
                    
                entry_price = stock_prices.loc[date]
                if pd.isna(entry_price) or entry_price <= 0:
                    continue
                
                try:
                    date_idx = stock_prices.index.get_loc(date)
                    start_idx = max(0, date_idx + self.start_offset)
                    end_idx = min(len(stock_prices) - 1, date_idx + self.end_offset)
                    
                    event_series = stock_prices.iloc[start_idx:end_idx + 1]
                    normalized_returns = (event_series / entry_price) - 1
                    relative_days = range(self.start_offset, self.end_offset + 1)
                    
                    event_key = f"{stock}_{date.strftime('%Y-%m-%d')}"
                    all_event_returns[event_key] = pd.Series(normalized_returns.values, index=relative_days)
                    event_count += 1
                    
                except (KeyError, IndexError):
                    continue
        
        if not all_event_returns:
            print("No valid events found for analysis")
            return {}
        
        # Create wide format DataFrame
        all_returns_df = pd.DataFrame(all_event_returns)
        
        # Calculate average returns
        average_returns = pd.DataFrame(index=all_returns_df.index)
        average_returns['avg_return_from_p0'] = all_returns_df.mean(axis=1)
        average_returns['std_return'] = all_returns_df.std(axis=1)
        average_returns['event_count'] = all_returns_df.count(axis=1)
        
        self.event_results = {
            'average_returns': average_returns,
            'all_event_returns': all_returns_df,
            'event_count': event_count
        }
        
        return self.event_results
    
    def get_summary_statistics(self) -> pd.Series:
        if not self.event_results:
            return pd.Series(dtype=object)

        summary = self.event_results['average_returns']
        post_event_summary = summary.loc[0:]
        
        max_ret_post = post_event_summary['avg_return_from_p0'].max()
        max_ret_post_day = post_event_summary['avg_return_from_p0'].idxmax()
        
        min_ret_post = post_event_summary['avg_return_from_p0'].min()
        min_ret_post_day = post_event_summary['avg_return_from_p0'].idxmin()

        stats = {
            'Total Events': self.event_results['event_count'],
            'Event Window': f"{self.start_offset} to {self.end_offset} days",
            'Pre-event avg return (day -1)': f"{summary.loc[-1, 'avg_return_from_p0']:.4f}" if -1 in summary.index else 'N/A',
            'Event day return (day 0)': f"{summary.loc[0, 'avg_return_from_p0']:.4f}" if 0 in summary.index else 'N/A',
            'Post-event avg return (day 60)': f"{summary.loc[60, 'avg_return_from_p0']:.4f}" if 60 in summary.index else 'N/A',
            'Maximum return (post-event)': f"{max_ret_post:.4f} (day {max_ret_post_day})",
            'Minimum return (post-event)': f"{min_ret_post:.4f} (day {min_ret_post_day})"
        }
        return pd.Series(stats)

    def plot_event_study(self, ax: plt.Axes) -> None:
        """Plot event study results on a given axes object"""
        if not self.event_results:
            print("No event study results available. Run run_event_study() first.")
            return
            
        stats = self.get_summary_statistics()
        print(stats.to_string())
        
        summary = self.event_results['average_returns']
        
        # Plot: Average Cumulative Return from P0
        ax.plot(summary.index, summary['avg_return_from_p0'] * 100, 
                label='Avg. Cumulative Return (from P0)', color='green', linewidth=2)
        
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Selection Event (Day 0)')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        ax.set_title('Event Study: Avg. Cumulative Return (P0)', fontsize=14)
        ax.set_xlabel('Days Relative to Selection Event')
        ax.set_ylabel('Avg. Cumulative Return (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Print detailed statistics
        print("\n--- Detailed Statistics by Period (P0-based Cumulative Return) ---")
        periods = [
            ('Pre-event (-30d)', -30),
            ('Event day (0d)', 0),
            ('Short-term (5d)', 5),
            ('Medium-term (20d)', 20),
            ('Long-term (60d)', 60)
        ]
        
        for period_name, day in periods:
            if day in summary.index:
                avg_return = summary.loc[day, 'avg_return_from_p0']
                print(f"{period_name}: {avg_return:.4f}")


class QuintileAnalysis:
    def __init__(self, params: Dict[str, Any], factor_name: str, factor_file: str):
        self.params = params
        self.factor_name = factor_name
        self.factor_file = factor_file
        self.results: Dict[str, PortfolioAnalysis] = {}

    def run(self) -> None:
        print(f"--- Running Analysis for {self.factor_name} ---")
        
        # Check if methodology is threshold-based (PriceTrendsAbs)
        if self.params.get('methodology_type') in THRESHOLD_METHODOLOGIES:
            print(f"  - Threshold-based methodology detected, running single analysis")
            analysis_params = {
                **self.params,
                'file_name': self.factor_file,
            }
            self.results['Threshold'] = PortfolioAnalysis.run(**analysis_params)
        else:
            # Original quintile-based analysis
            for name, pos in quintile_positions.items():
                print(f"  - Quintile: {name}")
                analysis_params = {
                    **self.params,
                    'factor_filename': self.factor_file,
                    'quantile_position': pos,
                }
                self.results[name] = PortfolioAnalysis.run(**analysis_params)

    def run_event_study(self, event_window: tuple = (-30, 60)) -> Dict[str, Any]:
        """Run event study for threshold-based methodology"""
        if self.params.get('methodology_type') not in THRESHOLD_METHODOLOGIES:
            print("Event study only available for threshold-based methodologies")
            return {}
            
        if 'Threshold' not in self.results:
            print("No threshold analysis results available")
            return {}
            
        event_study = EventStudy(self.results['Threshold'], event_window)
        return event_study.run_event_study()

    @property
    def cumulative_returns(self) -> pd.DataFrame:
        if self.params.get('methodology_type') in THRESHOLD_METHODOLOGIES:
            # For threshold-based methodology, return single result
            if 'Threshold' in self.results:
                df = pd.DataFrame({
                    'Threshold': self.results['Threshold'].perf_msre.pf_cumret.iloc[:, 0],
                    'Benchmark': self.results['Threshold'].perf_msre.bm_cumret.iloc[:, 0]
                })
                return df
            return pd.DataFrame()
        else:
            # Original quintile-based logic
            df = pd.DataFrame({
                name: analysis.perf_msre.pf_cumret.iloc[:, 0]
                for name, analysis in self.results.items()
            })
            if self.results:
                first_quintile = next(iter(quintile_positions))
                df['Benchmark'] = self.results[first_quintile].perf_msre.bm_cumret.iloc[:, 0]
            return df

    @property
    def full_summary_table(self) -> pd.DataFrame:
        if self.params.get('methodology_type') in THRESHOLD_METHODOLOGIES:
            # For threshold-based methodology, return single result
            if 'Threshold' in self.results:
                summary_data = {
                    'Threshold': self._get_performance_metrics(self.results['Threshold'].perf_msre, is_benchmark=False),
                    'Benchmark': self._get_performance_metrics(self.results['Threshold'].perf_msre, is_benchmark=True)
                }
                df = pd.DataFrame(summary_data)
                if not df.empty:
                    df = df.T
                    df.index.name = 'Scenario'
                return df
            return pd.DataFrame()
        else:
            # Original quintile-based logic
            summary_data = {}
            for name, analysis in self.results.items():
                summary_data[name] = self._get_performance_metrics(analysis.perf_msre, is_benchmark=False)
            
            if self.results:
                first_quintile = next(iter(quintile_positions))
                bm_perf = self.results[first_quintile].perf_msre
                summary_data['Benchmark'] = self._get_performance_metrics(bm_perf, is_benchmark=True)

            df = pd.DataFrame(summary_data)
            if not df.empty:
                df = df.T
                df.index.name = 'Scenario'
            return df

    @property
    def q1q5_summary_table(self) -> pd.DataFrame:
        if self.params.get('methodology_type') in THRESHOLD_METHODOLOGIES:
            # For threshold-based methodology, same as full summary
            return self.full_summary_table
        else:
            # Original quintile-based logic
            summary_data = {}
            if not self.results:
                return pd.DataFrame()

            quintile_keys = list(quintile_positions.keys())
            q1_name = quintile_keys[0]
            q5_name = quintile_keys[-1]

            if q1_name in self.results:
                summary_data[q1_name] = self._get_performance_metrics(self.results[q1_name].perf_msre, is_benchmark=False)
            
            if len(quintile_keys) > 1 and q5_name in self.results:
                summary_data[q5_name] = self._get_performance_metrics(self.results[q5_name].perf_msre, is_benchmark=False)

            if self.results:
                first_quintile = next(iter(quintile_positions))
                bm_perf = self.results[first_quintile].perf_msre
                summary_data['Benchmark'] = self._get_performance_metrics(bm_perf, is_benchmark=True)

            df = pd.DataFrame(summary_data)
            if not df.empty:
                df = df.T
                df.index.name = 'Scenario'
            return df

    def _get_performance_metrics(self, perf_measure, is_benchmark: bool) -> Dict[str, str]:
        idx = 1 if is_benchmark else 0
        
        metrics = {
            'CAGR (%)': perf_measure.performance_cagr[idx] * 100,
            'MDD (%)': perf_measure.performance_mdd[idx] * 100,
            'Sharpe Ratio': perf_measure.performance_sharpe[idx],
            'Hit Ratio (%)': perf_measure.performance_hit[idx] * 100,
            'Mean Return (%)': perf_measure.performance_mean[idx] * 100,
            'Std Dev (%)': perf_measure.performance_std[idx] * 100,
            'Cumulative Return (%)': perf_measure.performance_cumret[idx] * 100
        }
        return {k: f"{v:.2f}" for k, v in metrics.items()}


class FrequencyAnalysis:
    def __init__(self, params: Dict[str, Any], factor_name: str, factor_file: str, freq: str):
        self.params = {**params, 'freq': freq}
        self.factor_name = factor_name
        self.factor_file = factor_file
        self.freq = freq
        self.analysis: PortfolioAnalysis = None

    def run(self) -> None:
        print(f"--- Running Analysis for {self.factor_name} with frequency: {self.freq} ---")
        analysis_params = {**self.params, 'file_name': self.factor_file}
        self.analysis = PortfolioAnalysis.run(**analysis_params)

    def run_event_study(self, event_window: tuple = (-30, 60)) -> Dict[str, Any]:
        if self.params.get('methodology_type') not in THRESHOLD_METHODOLOGIES:
            return {}
        if not self.analysis:
            return {}
        event_study = EventStudy(self.analysis, event_window)
        return event_study.run_event_study()

    def get_summary_metrics(self) -> pd.Series:
        if not self.analysis:
            return pd.Series(dtype=object)
        return QuintileAnalysis._get_performance_metrics(None, self.analysis.perf_msre, is_benchmark=False)

    @staticmethod
    def get_benchmark_summary(analysis: PortfolioAnalysis) -> pd.Series:
        return QuintileAnalysis._get_performance_metrics(None, analysis.perf_msre, is_benchmark=True)


class ComparisonPlotter:
    def __init__(self, analyses: Dict[str, Union[QuintileAnalysis, FrequencyAnalysis]], event_studies: Dict[str, Any]):
        self.analyses = analyses
        self.event_studies = event_studies
        self.mode = ANALYSIS_MODE

    def plot(self) -> None:
        if self.mode == 'FREQUENCY':
            self._plot_frequency_comparison()
        else:
            self._plot_by_scenario()
            self._plot_by_quintile()
            self._plot_yearly_performance()
        
        self._plot_combined_event_study()
        plt.show()

    def _plot_frequency_comparison(self):
        num_scenarios = len(self.analyses)
        if num_scenarios == 0:
            return
            
        fig, axs = plt.subplots(num_scenarios, 4, figsize=(36, 6 * num_scenarios), squeeze=False)
        fig.suptitle('Performance Comparison by Frequency', fontsize=16, y=1.02)

        scenarios = list(self.analyses.keys())
        colors = plt.cm.get_cmap('viridis', len(scenarios))
        scenario_color_map = {scenario: colors(i) for i, scenario in enumerate(scenarios)}

        for i, (scenario_name, freq_analysis) in enumerate(self.analyses.items()):
            analysis = freq_analysis.analysis
            if not analysis:
                for ax in axs[i, :]:
                    ax.set_visible(False)
                continue

            ax_cumret, ax_dd, ax_event, ax_num_stocks = axs[i, 0], axs[i, 1], axs[i, 2], axs[i, 3]
            
            color = scenario_color_map.get(scenario_name, 'blue')
            
            # Cumulative Returns
            returns = analysis.perf_msre.pf_cumret
            ax_cumret.plot(returns.index, returns.iloc[:, 0], label=scenario_name, linewidth=2, color=color)
            
            # Drawdown
            dd_data = analysis.perf_msre.pf_dd
            ax_dd.plot(dd_data.index, dd_data.values, label=scenario_name, linewidth=1, color=color)
            ax_dd.fill_between(dd_data.index, dd_data.values.flatten(), color=color, alpha=0.15)

            # Add benchmark to Cumulative and Drawdown plots
            bm_returns = analysis.perf_msre.bm_cumret
            ax_cumret.plot(bm_returns.index, bm_returns.iloc[:, 0], label='Benchmark', 
                          color='black', linestyle='--', linewidth=1.5)
            
            bm_dd = analysis.perf_msre.bm_dd
            ax_dd.plot(bm_dd.index, bm_dd.values, label='Benchmark', 
                      color='black', linestyle='--', linewidth=1)

            # --- Formatting ---
            ax_cumret.set_title(f'{scenario_name} - Cumulative Returns')
            ax_cumret.set_ylabel('Cumulative Return')
            ax_cumret.legend()
            ax_cumret.grid(True, alpha=0.3)

            ax_dd.set_title(f'{scenario_name} - Drawdown')
            ax_dd.set_ylabel('Drawdown')
            ax_dd.legend()
            ax_dd.grid(True, alpha=0.3)
            
            # Plot Event Study
            if scenario_name in self.event_studies and self.event_studies[scenario_name]:
                event_study = EventStudy(analysis)
                event_study.event_results = self.event_studies[scenario_name]
                event_study.plot_event_study(ax=ax_event)
            else:
                ax_event.set_visible(False)
            
            # Plot Number of Stocks
            weights = analysis.methodology.weights
            num_stocks = (weights > 0).sum(axis=1)
            
            ax_num_stocks.plot(num_stocks.index, num_stocks.values, label='Num. of Stocks', color='purple', linewidth=1)
            ax_num_stocks.set_title(f'{scenario_name} - Number of Holdings')
            ax_num_stocks.set_ylabel('Count')
            ax_num_stocks.legend()
            ax_num_stocks.grid(True, alpha=0.3)

        fig.tight_layout(pad=3.0)

    def _plot_combined_event_study(self):
        """Plot combined event study for all threshold-based methodologies"""
        if not self.event_studies:
            return
            
        fig, ax = plt.subplots(1, 1, figsize=(15, 8))
        
        colors = plt.cm.get_cmap('viridis', len(self.event_studies))
        color_map = {scenario: colors(i) for i, scenario in enumerate(self.event_studies.keys())}
        
        for scenario_name, event_data in self.event_studies.items():
            if event_data and 'average_returns' in event_data:
                summary = event_data['average_returns']
                # Filter to show only 0 to 60 days
                post_event_data = summary.loc[0:60]
                
                ax.plot(post_event_data.index, post_event_data['avg_return_from_p0'] * 100, 
                       label=scenario_name, color=color_map.get(scenario_name, 'gray'), linewidth=2)
        
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Selection Event (Day 0)')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        ax.set_title('Combined Event Study: Average Cumulative Return (P0) - Post Event Period', fontsize=14)
        ax.set_xlabel('Days After Selection Event')
        ax.set_ylabel('Average Cumulative Return (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()

    def _plot_by_scenario(self):
        num_scenarios = len(self.analyses)
        fig, axs = plt.subplots(num_scenarios, 1, figsize=(12, 6 * num_scenarios), squeeze=False)
        fig.suptitle('Performance by Scenario', fontsize=16, y=1.02)

        for i, (name, analysis) in enumerate(self.analyses.items()):
            ax = axs[i, 0]
            data = analysis.cumulative_returns
            
            # Check if this is a threshold-based methodology
            is_threshold = analysis.params.get('methodology_type') in THRESHOLD_METHODOLOGIES
            
            if is_threshold:
                # For threshold-based, plot Threshold vs Benchmark
                if 'Threshold' in data:
                    ax.plot(data.index, data['Threshold'], label='Threshold', color='blue', linewidth=2)
                if 'Benchmark' in data:
                    ax.plot(data.index, data['Benchmark'], label='Benchmark', color='black', linestyle='--', linewidth=1.5)
            else:
                # Original quintile-based plotting
                colors = plt.cm.get_cmap('RdYlGn', len(quintile_positions))
                for j, quintile in enumerate(quintile_positions.keys()):
                    if quintile in data:
                        ax.plot(data.index, data[quintile], label=quintile, color=colors(j), linewidth=2)
                
                if 'Benchmark' in data:
                    ax.plot(data.index, data['Benchmark'], label='Benchmark', color='black', linestyle='--', linewidth=1.5)
            
            ax.set_title(f'Performance - {name}', fontsize=14)
            ax.set_ylabel('Cumulative Return')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        fig.tight_layout(pad=3.0)

    def _plot_by_quintile(self):
        # Check if any analysis uses threshold-based methodology
        has_threshold = any(analysis.params.get('methodology_type') in THRESHOLD_METHODOLOGIES 
                           for analysis in self.analyses.values())
        
        if has_threshold:
            # For threshold-based methodologies, create a simpler plot
            self._plot_threshold_comparison()
        else:
            # Original quintile-based plotting
            self._plot_quintile_comparison()

    def _plot_threshold_comparison(self):
        num_scenarios = len(self.analyses)
        fig, axs = plt.subplots(num_scenarios, 4, figsize=(36, 6 * num_scenarios), squeeze=False)
        fig.suptitle('Threshold-based Performance Comparison', fontsize=16, y=1.02)

        scenarios = list(self.analyses.keys())
        colors = plt.cm.get_cmap('viridis', len(scenarios))
        scenario_color_map = {scenario: colors(i) for i, scenario in enumerate(scenarios)}

        for i, (scenario_name, analysis) in enumerate(self.analyses.items()):
            if analysis.params.get('methodology_type') in THRESHOLD_METHODOLOGIES:
                ax_cumret, ax_dd, ax_event, ax_num_stocks = axs[i, 0], axs[i, 1], axs[i, 2], axs[i, 3]

                if 'Threshold' in analysis.results:
                    color = scenario_color_map.get(scenario_name, 'blue')
                    returns = analysis.results['Threshold'].perf_msre.pf_cumret
                    ax_cumret.plot(returns.index, returns.iloc[:, 0], label=scenario_name, linewidth=2, color=color)
                    
                    dd_data = analysis.results['Threshold'].perf_msre.pf_dd
                    ax_dd.plot(dd_data.index, dd_data.values, label=scenario_name, linewidth=1, color=color)
                    ax_dd.fill_between(dd_data.index, dd_data.values.flatten(), color=color, alpha=0.15)

                # Add benchmark
                if 'Threshold' in analysis.results:
                    bm_returns = analysis.results['Threshold'].perf_msre.bm_cumret
                    bm_dd = analysis.results['Threshold'].perf_msre.bm_dd
                    
                    ax_cumret.plot(bm_returns.index, bm_returns.iloc[:, 0], label='Benchmark', 
                                 color='black', linestyle='--', linewidth=1.5)
                    ax_dd.plot(bm_dd.index, bm_dd.values, label='Benchmark', 
                              color='black', linestyle='--', linewidth=1)

                ax_cumret.set_title(f'{scenario_name} - Cumulative Returns')
                ax_cumret.set_ylabel('Cumulative Return')
                ax_cumret.legend()
                ax_cumret.grid(True, alpha=0.3)

                ax_dd.set_title(f'{scenario_name} - Drawdown')
                ax_dd.set_ylabel('Drawdown')
                ax_dd.set_xlabel('Date')
                ax_dd.legend()
                ax_dd.grid(True, alpha=0.3)
                
                # Plot Event Study
                if scenario_name in self.event_studies and self.event_studies[scenario_name]:
                    event_study = EventStudy(analysis.results['Threshold'])
                    event_study.event_results = self.event_studies[scenario_name]
                    event_study.plot_event_study(ax=ax_event)
                else:
                    ax_event.set_visible(False)
                
                # Plot Number of Stocks
                if 'Threshold' in analysis.results:
                    portfolio_analysis = analysis.results['Threshold']
                    weights = portfolio_analysis.methodology.weights
                    num_stocks = (weights > 0).sum(axis=1)
                    
                    ax_num_stocks.plot(num_stocks.index, num_stocks.values, label='Num. of Stocks', color='purple', linewidth=1)
                    ax_num_stocks.set_title(f'{scenario_name} - Number of Holdings')
                    ax_num_stocks.set_ylabel('Count')
                    ax_num_stocks.legend()
                    ax_num_stocks.grid(True, alpha=0.3)
                else:
                    ax_num_stocks.set_visible(False)
            else:
                 for ax in axs[i, :]:
                    ax.set_visible(False)


        fig.tight_layout(pad=3.0)

    def _plot_quintile_comparison(self):
        num_quintiles = len(quintile_positions)
        fig, axs = plt.subplots(num_quintiles, 2, figsize=(20, 6 * num_quintiles), squeeze=False)
        fig.suptitle('Performance by Quintile (Return & Drawdown)', fontsize=16, y=1.02)

        scenarios = list(self.analyses.keys())
        colors = plt.cm.get_cmap('viridis', len(scenarios))
        scenario_color_map = {scenario: colors(i) for i, scenario in enumerate(scenarios)}

        for i, (quintile_name, _) in enumerate(quintile_positions.items()):
            ax_cumret = axs[i, 0]
            ax_dd = axs[i, 1]

            for scenario_name, analysis in self.analyses.items():
                if quintile_name in analysis.results:
                    color = scenario_color_map.get(scenario_name, 'gray')
                    returns = analysis.results[quintile_name].perf_msre.pf_cumret
                    ax_cumret.plot(returns.index, returns.iloc[:, 0], label=scenario_name, linewidth=2, color=color)
                    
                    dd_data = analysis.results[quintile_name].perf_msre.pf_dd
                    ax_dd.plot(dd_data.index, dd_data.values, label=scenario_name, linewidth=1, color=color)
                    ax_dd.fill_between(dd_data.index, dd_data.values.flatten(), color=color, alpha=0.15)

            ax_cumret.set_title(f'{quintile_name} - Cumulative Returns')
            ax_cumret.set_ylabel('Cumulative Return')
            ax_cumret.legend()
            ax_cumret.grid(True, alpha=0.3)

            ax_dd.set_title(f'{quintile_name} - Drawdown')
            ax_dd.set_ylabel('Drawdown')
            ax_dd.set_xlabel('Date')
            ax_dd.legend()
            ax_dd.grid(True, alpha=0.3)

        fig.tight_layout(pad=3.0)

    def _plot_yearly_performance(self):
        # Check if any analysis uses threshold-based methodology
        has_threshold = any(analysis.params.get('methodology_type') in THRESHOLD_METHODOLOGIES 
                           for analysis in self.analyses.values())
        
        if has_threshold:
            # For threshold-based methodologies, create a simpler yearly plot
            self._plot_threshold_yearly_performance()
        else:
            # Original quintile-based yearly plotting
            self._plot_quintile_yearly_performance()

    def _plot_threshold_yearly_performance(self):
        num_scenarios = len(self.analyses)
        fig, axs = plt.subplots(num_scenarios, 1, figsize=(12, 3.6 * num_scenarios), squeeze=False, sharex=True)
        fig.suptitle('Yearly Performance Comparison - Threshold-based', fontsize=16, y=1.02)

        scenarios = list(self.analyses.keys())
        colors = plt.cm.get_cmap('viridis', len(scenarios))
        scenario_color_map = {scenario: colors(i) for i, scenario in enumerate(scenarios)}
        scenario_color_map['Benchmark'] = 'black'

        for i, (scenario_name, analysis) in enumerate(self.analyses.items()):
            if analysis.params.get('methodology_type') in THRESHOLD_METHODOLOGIES:
                ax = axs[i, 0]
                
                yearly_data = {}
                if 'Threshold' in analysis.results:
                    yearly_returns = (analysis.results['Threshold'].perf_msre.performance_specific()['Portfolio'] - 1) * 100
                    yearly_data[scenario_name] = yearly_returns
                    
                    bm_yearly_returns = (analysis.results['Threshold'].perf_msre.performance_specific()['BM'] - 1) * 100
                    yearly_data['Benchmark'] = bm_yearly_returns

                df = pd.DataFrame(yearly_data)
                
                ordered_cols = [scenario_name, 'Benchmark']
                plot_df = df[[col for col in ordered_cols if col in df.columns]]
                plot_colors = [scenario_color_map[col] for col in plot_df.columns]

                plot_df.plot(kind='bar', ax=ax, width=0.8, color=plot_colors)
                ax.set_title(f'{scenario_name} - Yearly Performance')
                ax.set_ylabel('Return (%)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='x', rotation=45)

        fig.tight_layout(pad=3.0)

    def _plot_quintile_yearly_performance(self):
        num_quintiles = len(quintile_positions)
        fig, axs = plt.subplots(num_quintiles, 1, figsize=(12, 3.6 * num_quintiles), squeeze=False, sharex=True)
        fig.suptitle('Yearly Performance Comparison by Quintile', fontsize=16, y=1.02)

        scenarios = list(self.analyses.keys())
        colors = plt.cm.get_cmap('viridis', len(scenarios))
        scenario_color_map = {scenario: colors(i) for i, scenario in enumerate(scenarios)}
        scenario_color_map['Benchmark'] = 'black'

        for i, quintile_name in enumerate(quintile_positions.keys()):
            ax = axs[i, 0]
            
            yearly_data = {}
            for scenario_name, analysis in self.analyses.items():
                if quintile_name in analysis.results:
                    yearly_returns = (analysis.results[quintile_name].perf_msre.performance_specific()['Portfolio'] - 1) * 100
                    yearly_data[scenario_name] = yearly_returns
            
            if self.analyses and quintile_name in next(iter(self.analyses.values())).results:
                bm_yearly_returns = (next(iter(self.analyses.values())).results[quintile_name].perf_msre.performance_specific()['BM'] - 1) * 100
                yearly_data['Benchmark'] = bm_yearly_returns

            df = pd.DataFrame(yearly_data)
            
            ordered_cols = scenarios + ['Benchmark']
            plot_df = df[[col for col in ordered_cols if col in df.columns]]
            plot_colors = [scenario_color_map[col] for col in plot_df.columns]

            plot_df.plot(kind='bar', ax=ax, width=0.8, color=plot_colors)
            ax.set_title(f'{quintile_name} - Yearly Performance')
            ax.set_ylabel('Return (%)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)

        fig.tight_layout(pad=3.0)


def run_factor_comparison() -> Dict[str, Any]:
    all_analyses: Dict[str, QuintileAnalysis] = {}
    for factor_name, factor_file in factor_files.items():
        quintile_analysis = QuintileAnalysis(common_params, factor_name, factor_file)
        quintile_analysis.run()
        all_analyses[factor_name] = quintile_analysis

    # Event study, summary tables, etc. (original logic)
    event_study_results = {}
    full_event_study_summary_data = {}
    for factor_name, analysis in all_analyses.items():
        if analysis.params.get('methodology_type') in THRESHOLD_METHODOLOGIES:
            event_results = analysis.run_event_study(event_window=(-30, 60))
            if event_results:
                event_study_results[factor_name] = event_results
                
                if 'Threshold' in analysis.results:
                    event_study_instance = EventStudy(analysis.results['Threshold'])
                    event_study_instance.event_results = event_results
                    full_event_study_summary_data[factor_name] = event_study_instance.get_summary_statistics()

    full_event_study_table = pd.DataFrame(full_event_study_summary_data)
    if not full_event_study_table.empty:
        full_event_study_table.index.name = "항목"
                
    full_summaries, q1q5_summaries = [], []
    for factor_name, analysis in all_analyses.items():
        full_summary = analysis.full_summary_table.copy()
        if 'Benchmark' in full_summary.index:
            full_summary = full_summary.drop('Benchmark')
        full_summary.index = [f"{factor_name} - {idx}" for idx in full_summary.index]
        full_summaries.append(full_summary)

        q1q5_summary = analysis.q1q5_summary_table.copy()
        if 'Benchmark' in q1q5_summary.index:
            q1q5_summary = q1q5_summary.drop('Benchmark')
        q1q5_summary.index = [f"{factor_name} - {idx}" for idx in q1q5_summary.index]
        q1q5_summaries.append(q1q5_summary)

    full_summary_table = pd.concat(full_summaries) if full_summaries else pd.DataFrame()
    q1q5_summary_table = pd.concat(q1q5_summaries) if q1q5_summaries else pd.DataFrame()

    if all_analyses:
        first_analysis = next(iter(all_analyses.values()))
        benchmark_row = first_analysis.full_summary_table.loc[['Benchmark']]

        if not full_summary_table.empty:
            full_summary_table = pd.concat([full_summary_table, benchmark_row])
        if not q1q5_summary_table.empty:
            q1q5_summary_table = pd.concat([q1q5_summary_table, benchmark_row])

    # Print summaries
    if not full_summary_table.empty:
        print("\n--- Aggregated Full Performance Summary ---")
        print(full_summary_table.to_string())
    if not q1q5_summary_table.empty:
        print(f"\n--- Aggregated Q1 vs Q5 Performance Summary ---")
        print(q1q5_summary_table.to_string())
    print("\n" + "="*80 + "\n")

    plotter = ComparisonPlotter(all_analyses, event_study_results)
    plotter.plot()
    
    return {
        **all_analyses,
        'Full': full_summary_table,
        'Q1Q5': q1q5_summary_table,
        'EventStudy': event_study_results,
        'FullEventStudy': full_event_study_table
    }

def run_frequency_comparison() -> Dict[str, Any]:
    config = frequency_analysis_config
    factor_name = config['factor_name']
    factor_file = config['factor_file']
    
    all_analyses: Dict[str, FrequencyAnalysis] = {}
    for freq in config['frequencies']:
        scenario_name = f"{factor_name}_{freq}"
        freq_analysis = FrequencyAnalysis(common_params, factor_name, factor_file, freq)
        freq_analysis.run()
        all_analyses[scenario_name] = freq_analysis
        
    # Event study and summary table
    event_study_results = {}
    full_event_study_summary_data = {}
    summary_data = {}
    
    for scenario_name, analysis in all_analyses.items():
        event_results = analysis.run_event_study(event_window=(-30, 60))
        if event_results:
            event_study_results[scenario_name] = event_results
            event_study_instance = EventStudy(analysis.analysis)
            event_study_instance.event_results = event_results
            full_event_study_summary_data[scenario_name] = event_study_instance.get_summary_statistics()

        summary_data[scenario_name] = analysis.get_summary_metrics()

    if all_analyses:
        first_analysis = next(iter(all_analyses.values())).analysis
        if first_analysis:
            summary_data['Benchmark'] = FrequencyAnalysis.get_benchmark_summary(first_analysis)
            
    summary_table = pd.DataFrame(summary_data).T
    summary_table.index.name = 'Scenario'

    full_event_study_table = pd.DataFrame(full_event_study_summary_data)
    if not full_event_study_table.empty:
        full_event_study_table.index.name = "항목"
        
    print("\n--- Performance Summary by Frequency ---")
    if not summary_table.empty:
        print(summary_table.to_string())
    print("\n" + "="*80 + "\n")
    
    plotter = ComparisonPlotter(all_analyses, event_study_results)
    plotter.plot()

    return {
        **all_analyses,
        'Summary': summary_table,
        'EventStudy': event_study_results,
        'FullEventStudy': full_event_study_table
    }

def main() -> Dict[str, Any]:
    if ANALYSIS_MODE == 'FREQUENCY':
        return run_frequency_comparison()
    else:
        return run_factor_comparison()


if __name__ == "__main__":
    analyses_results = main() 