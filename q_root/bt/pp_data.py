import pandas as pd
import os


class PreProcess:
    def __init__(self,
                 file_name: str,
                 data_dir: str = None,
                 **kwargs):
        self.file_name = file_name
        self.data_dir = data_dir if data_dir else os.path.join(os.path.dirname(__file__), 'DATA')

        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_QW_res_data(self) -> pd.DataFrame:
        file_path = os.path.join(self.data_dir, f"{self.file_name}.xlsx")
        data = pd.read_excel(file_path,
                             index_col=0,
                             header=7).iloc[6:, :]
        data.index = pd.to_datetime(data.index)
        data.index.name = 'Date'
        return data

    def get_DG_res_data(self) -> pd.DataFrame:
        file_path = os.path.join(self.data_dir, f"{self.file_name}.xlsx")
        data = pd.read_excel(file_path,
                             index_col=0,
                             header=9).iloc[4:, :]
        data.index = pd.to_datetime(data.index)
        data.index.name = 'Date'
        return data

    def get_QW_bm_data(self) -> pd.DataFrame:
        data = self.get_QW_res_data()
        data.columns = ['KOSPI', 'KOSPI_TR',
                        'KOSPI200', 'KOSPI200_TR',
                        'KOSDAQ', 'KOSDAQ_TR',
                        'KODSAQ150', 'KOSDAQ150_TR']
        return data

    def remove_str_data(self, data: pd.DataFrame) -> pd.DataFrame:
        pp_data = data.copy()
        
        for col in pp_data.columns:
            if pp_data[col].dtype == 'object':
                pp_data[col] = pd.to_numeric(pp_data[col], errors='coerce')        
        return pp_data

    def lag_data(self, 
                 data: pd.DataFrame, 
                 lag_months: list = [3,3,3,4]) -> pd.DataFrame:
        if len(lag_months) != 4:
            raise ValueError("lag_months must be a list of 4 integers")

        orig_idx = data.index
        if not orig_idx.is_monotonic_increasing:
            orig_idx = orig_idx.sort_values()
            data = data.loc[orig_idx]

        q_data = data.resample('QE').last()

        availability_info = []
        for q_end_date in q_data.index:
            if q_data.loc[q_end_date].isnull().all():
                continue

            quarter_idx = (q_end_date.month - 1) // 3
            lag = lag_months[quarter_idx]
            q_end_month = q_end_date.month

            # NOTE: Q1 (month 3) + lag 3 months -> available month 7 (July)
            target_month_num = q_end_month + lag + 1
            target_year = q_end_date.year + (target_month_num - 1) // 12
            target_month = ((target_month_num - 1) % 12) + 1

            available_from_date = pd.Timestamp(target_year, target_month, 1)

            availability_info.append({
                'available_date': available_from_date,
                'data_row': q_data.loc[q_end_date]
            })

        if not availability_info:
            print("Warning: No quarterly data found to lag.")
            return pd.DataFrame(index=orig_idx, columns=data.columns)

        df = pd.DataFrame([item['data_row'] for item in availability_info])
        df.index = pd.DatetimeIndex([item['available_date'] for item in availability_info])
        df = df.sort_index()

        combined_idx = orig_idx.union(df.index).sort_values()

        lagged_data = df.reindex(combined_idx, method='ffill').reindex(orig_idx)
        return lagged_data

    def calculate_ttm(self, data: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data index must be DatetimeIndex")

        q_data = data.resample('QE').last()
        
        ttm_data = []
        
        for i in range(len(q_data)):
            if i < 3:
                continue
                
            current_q = q_data.iloc[i]
            prev_q1 = q_data.iloc[i-1]
            prev_q2 = q_data.iloc[i-2]
            prev_q3 = q_data.iloc[i-3]
            
            ttm_sum = current_q + prev_q1 + prev_q2 + prev_q3
            
            ttm_data.append({
                'date': q_data.index[i],
                'data': ttm_sum
            })
        
        if not ttm_data:
            print("Warning: Not enough quarters to calculate TTM data")
            return pd.DataFrame(index=data.index, columns=data.columns)
            
        ttm_df = pd.DataFrame([item['data'] for item in ttm_data])
        ttm_df.index = pd.DatetimeIndex([item['date'] for item in ttm_data])
        ttm_df = ttm_df.reindex(data.index, method='ffill')
        return ttm_df

    def save_data(self, method=None, lag_months=None, remove_strings=False, calculate_ttm=False):
        if method is None:
            method = 'get_QW_res_data'

        data = None
        try:
            if hasattr(self, method):
                data_method = getattr(self, method)
                data = data_method()
                print(f"Original data loaded. Shape: {data.shape}")

                if remove_strings:
                    print("Removing string values from data...")
                    data = self.remove_str_data(data)
                    print(f"Data after string removal. Shape: {data.shape}")

                if calculate_ttm:
                    print("Calculating TTM data...")
                    data = self.calculate_ttm(data)
                    print(f"TTM data generated. Shape: {data.shape}")

                if lag_months is not None:
                    print(f"Applying lag months: {lag_months}")
                    data = self.lag_data(data, lag_months)
                    print(f"Lagged data generated. Shape: {data.shape}")

                if data is not None and not data.empty:
                    output_path = os.path.join(self.data_dir, f"{self.file_name}.parquet")
                    data.to_parquet(output_path)
                    print(f"Data saved to {output_path}")
                else:
                    print("Warning: No data to save.")

        except Exception as e:
            print(f"Error in save_data: {e}")
            import traceback
            traceback.print_exc()
        return data


if __name__ == "__main__":
    file_name = "data_base_bm"
    
    pp = PreProcess(file_name=file_name)
    df = pp.save_data(method="get_QW_bm_data")
    # pp.save_data(method="get_QW_res_data", lag_months=[3,3,3,4], remove_strings=True, calculate_ttm=False)
