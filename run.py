# run.py

import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from main import AnalysisManager, MethodologyType, KoreaTransactionCost
from tools import Tools  # For plotting parameters if needed

# --- 결과 저장 디렉토리 생성 ---
RESULTS_DIR = "./results"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# --- 백테스팅 설정 (main.py와 유사하게) ---
cost = KoreaTransactionCost()
DEFAULT_CONFIG = {
    'init_invest': 1e8,
    'mkt': 'KOSPI200',
    'start_date': '20130101',
    'end_date': '20230701',
    'multiplier': 'Y',  # Affects annualized metrics in performance table
    'buy_commission': cost.buy_commission,
    'sell_commission': cost.sell_commission,
    'slippage': cost.slippage,
    'sell_tax': cost.sell_tax,
    'cash_rate': cost.cash_rate,
    # Default methodology kwargs (can be overridden in UI)
    'freq': 'monthly',
    'quantile': 5,
    'quantile_position': [1],
    'weight_type': 'mktcap_float'
}

METHODOLOGY_TYPES = [
    MethodologyType.GPAlfq0,
    MethodologyType.EBITDAEVttmlfq0,
    MethodologyType.FCFEVttmlfq0,
    MethodologyType.Momentum3612_1,
    MethodologyType.Payoutttmlfq0,
    MethodologyType.DataValidation  # Include validation if desired
]

# --- Streamlit 앱 ---
st.set_page_config(layout="wide")
st.title("백테스팅 결과 분석 대시보드")

# --- 사이드바: 설정 및 실행 ---
st.sidebar.header("백테스팅 설정")

# 기본 설정값 표시 및 수정 기능 (선택 사항)
# config_override = {}
# for key, value in DEFAULT_CONFIG.items():
#     # Simplified example: allow editing start/end dates
#     if key in ['start_date', 'end_date']:
#         config_override[key] = st.sidebar.text_input(f"{key}", value)
#     elif key == 'mkt':
#          config_override[key] = st.sidebar.selectbox(f"{key}", ['KOSPI200', 'KOSDAQ', 'ALL'], index=0) # Add more if needed
#     # Add more inputs for other config options if needed
#     else:
#         config_override[key] = value # Keep default for others

# config = {**DEFAULT_CONFIG, **config_override} # Merge overrides
config = DEFAULT_CONFIG  # Use default config for simplicity for now

selected_methodology = st.sidebar.selectbox(
    "전략 선택",
    METHODOLOGY_TYPES,
    format_func=lambda x: x.name  # Show enum name
)

# Add methodology specific kwargs if needed (e.g., quantile for GPA)
# Example:
# if selected_methodology == MethodologyType.GPAlfq0:
#     config['quantile'] = st.sidebar.number_input("Quantile", min_value=1, max_value=10, value=config.get('quantile', 5))
#     config['quantile_position'] = # Add logic for multiple selections if needed

run_button = st.sidebar.button("백테스팅 실행 및 결과 저장")

if run_button:
    st.sidebar.write(f"{selected_methodology.name} 전략 백테스팅 실행 중...")
    try:
        # AnalysisManager를 사용하여 백테스팅 실행
        # Note: AnalysisManager runs only one analysis in the current main.py logic when called directly
        # If you need multiple runs managed by AnalysisManager, adjust main.py or call PortfolioAnalysis directly
        # Pass only the selected one
        mgr = AnalysisManager(config, [selected_methodology])
        analysis_instance = mgr.run(selected_methodology)

        st.sidebar.success("백테스팅 완료!")

        # --- 결과 저장 ---
        result_filename_base = f"{selected_methodology.name}_{config['start_date']}_{config['end_date']}"
        result_filepath = os.path.join(
            RESULTS_DIR, f"{result_filename_base}.xlsx")

        with pd.ExcelWriter(result_filepath) as writer:
            # 성과 요약 테이블 저장
            perf_table = analysis_instance.perf_msre.performance_table()
            perf_table.to_excel(writer, sheet_name="Performance Summary")

            # 연도별 성과 저장
            perf_specific = analysis_instance.perf_msre.performance_specific()
            perf_specific.to_excel(writer, sheet_name="Yearly Performance")

            # 거래 비용 요약 저장
            cost_summary = analysis_instance.transaction_costs_summary
            # index=False if Date is not the primary index needed
            cost_summary.to_excel(
                writer, sheet_name="Transaction Costs", index=False)

            # 현금 잔고 요약 저장
            cash_summary = analysis_instance.cash_balance_summary
            # Assuming Date index is desired
            cash_summary.to_excel(writer, sheet_name="Cash Balance")

            # 누적 수익률 데이터 저장 (그래프용)
            cumret_df = pd.concat([analysis_instance.perf_msre.pf_cumret,
                                   analysis_instance.perf_msre.bm_cumret], axis=1)
            cumret_df.columns = ['Portfolio', 'Benchmark']
            cumret_df.to_excel(writer, sheet_name="Cumulative Return Data")

            # 낙폭 데이터 저장 (그래프용)
            drawdown_df = pd.concat([analysis_instance.perf_msre.pf_dd,
                                     analysis_instance.perf_msre.bm_dd], axis=1)
            drawdown_df.columns = ['Portfolio DD', 'Benchmark DD']
            drawdown_df.to_excel(writer, sheet_name="Drawdown Data")

        st.sidebar.success(f"결과 저장 완료: {result_filepath}")

    except Exception as e:
        st.sidebar.error(f"백테스팅 중 오류 발생: {e}")
        analysis_instance = None  # Ensure it's None on error

# --- 메인 화면: 결과 표시 ---
st.header("저장된 결과 조회")

result_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('.xlsx')]
selected_file = st.selectbox("결과 파일 선택", result_files)

if selected_file:
    filepath = os.path.join(RESULTS_DIR, selected_file)
    try:
        xls = pd.ExcelFile(filepath)

        st.subheader("성과 요약 (Performance Summary)")
        perf_summary_df = pd.read_excel(
            xls, sheet_name="Performance Summary", index_col=0)
        st.dataframe(perf_summary_df)

        st.subheader("누적 수익률 및 낙폭 그래프")
        try:
            cumret_df = pd.read_excel(
                xls, sheet_name="Cumulative Return Data", index_col=0, parse_dates=True)
            drawdown_df = pd.read_excel(
                xls, sheet_name="Drawdown Data", index_col=0, parse_dates=True)

            fig, axs = plt.subplots(2, 1, figsize=(
                15, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

            # 누적 수익률 그래프
            axs[0].plot(cumret_df.index, cumret_df['Portfolio'],
                        label='Portfolio', color='red', linewidth=2)
            axs[0].plot(cumret_df.index, cumret_df['Benchmark'],
                        label='Benchmark', color='black', linewidth=1.5, linestyle='--')
            axs[0].set_ylabel('Cumulative Return')
            axs[0].set_title('Portfolio vs Benchmark Cumulative Return')
            axs[0].legend(loc='best')
            axs[0].grid(True)

            # 낙폭 그래프
            axs[1].plot(drawdown_df.index, drawdown_df['Portfolio DD'],
                        label='Portfolio Drawdown', color='red', linewidth=2)
            axs[1].plot(drawdown_df.index, drawdown_df['Benchmark DD'],
                        label='Benchmark Drawdown', color='black', linewidth=1.5, linestyle='--')
            axs[1].fill_between(
                drawdown_df.index, drawdown_df['Portfolio DD'].values.flatten(), color='red', alpha=0.1)
            axs[1].fill_between(drawdown_df.index, drawdown_df['Benchmark DD'].values.flatten(
            ), color='black', alpha=0.1)
            axs[1].set_xlabel('Date')
            axs[1].set_ylabel('Drawdown')
            axs[1].grid(True)

            st.pyplot(fig)
            plt.close(fig)  # Close the plot to free memory

        except Exception as plot_e:
            st.error(f"그래프 생성 중 오류 발생: {plot_e}")

        st.subheader("연도별 성과 (Yearly Performance)")
        yearly_perf_df = pd.read_excel(
            xls, sheet_name="Yearly Performance", index_col=0)
        st.dataframe(yearly_perf_df)

        st.subheader("거래 비용 요약 (Transaction Costs)")
        cost_summary_df = pd.read_excel(xls, sheet_name="Transaction Costs")
        st.dataframe(cost_summary_df)

        st.subheader("현금 잔고 요약 (Cash Balance)")
        cash_summary_df = pd.read_excel(
            xls, sheet_name="Cash Balance", index_col=0, parse_dates=True)
        st.dataframe(cash_summary_df)

    except Exception as e:
        st.error(f"결과 파일 로딩 중 오류 발생: {e}")

else:
    st.info("분석할 결과 파일을 선택해주세요.")

# --- 추가 정보 ---
st.sidebar.info("""
**사용 방법:**
1. 사이드바에서 원하는 전략을 선택합니다.
2. (선택사항) 필요한 경우 설정을 수정합니다.
3. '백테스팅 실행 및 결과 저장' 버튼을 클릭합니다.
4. 실행 완료 후 메인 화면에서 결과 파일을 선택하여 내용을 확인합니다.
""")
