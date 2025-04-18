import pandas as pd
from bt.loader import DirMkt


CONST = "data_const_%s"


class KorConstHelper:
    def __init__(self,
                 mkt: str = 'KOSPI200'):
        try:
            self.data_verifier(mkt=mkt)
        except ValueError as error:
            print(error)

        self.mkt_value = getattr(DirMkt, mkt).name
        self.file_name = f"{CONST % self.mkt_value}"

    @staticmethod
    def data_verifier(mkt):
        mkt_keys = list(DirMkt.__members__)

        if mkt not in mkt_keys:
            raise ValueError(
                f"Invalid mkt value. Valid options are: {','.join(mkt_keys)}")

    def __repr__(self) -> str:
        return f"mkt_value: {self.mkt_value}"

    def get_DG_raw_data(self) -> pd.DataFrame:
        data = pd.read_excel(f"./DATA/{self.file_name}.xlsx", header=6)
        data.columns = ['Date', 'Code', 'Name']
        return data

    def get_DG_res_data(self) -> pd.DataFrame:
        data = self.get_DG_raw_data()

        data = data.drop(['Name'], axis=1)
        data = data.pivot(index='Date',
                          columns='Code',
                          values='Code')

        res = data.notna().astype(int)
        res.index = pd.to_datetime(res.index)
        return res

    def save_data(self):
        return self.get_DG_res_data().to_parquet(f'./DATA/{self.file_name}.parquet')


def update_all():
    mkt_keys = list(DirMkt.__members__)
    
    if 'ALL' in mkt_keys:
        mkt_keys.remove('ALL')
        print(f"Excluded 'ALL' from processing. Processing {len(mkt_keys)} markets...")
    else:
        print(f"Starting to update constituent data for all {len(mkt_keys)} markets...")
    
    res = {}
    
    for mkt in mkt_keys:
        try:
            print(f"Processing {mkt}...")
            helper = KorConstHelper(mkt=mkt)
            helper.save_data()
            res[mkt] = True
            print(f"Successfully updated {mkt}")
        except Exception as e:
            print(f"Error updating {mkt}: {e}")
            res[mkt] = False
    
    success_count = sum(res.values())
    print(f"Completed updating {success_count} out of {len(mkt_keys)} markets")
    return res


if __name__ == "__main__":
    # Individual market update
    # kor_const = KorConstHelper(mkt='KOSPI200')
    
    # Update all constituents
    update_all()
