import pandas as pd
from pathlib import Path

from bt.methodology import Methodology


class MethodologyEMP008(Methodology):
    def __init__(self,
                 mkt: str = 'KOSPI200',
                 start_date: str = '20241230',
                 end_date: str = '20250812',
                 factor_filename: str = 'emp008_w_cnn.xlsx',
                 **kwargs):
        freq = kwargs.pop('freq', 'monthly')
        quantile = kwargs.pop('quantile', 10)
        quantile_position = kwargs.pop('quantile_position', [1])
        weight_type = kwargs.pop('weight_type', 'mktcap_float')
        super().__init__(mkt, start_date, end_date, **kwargs)

        self.freq = freq
        self.quantile = quantile
        self.quantile_position = quantile_position
        self.weight_type = weight_type
        self.factor_filename = factor_filename

    @property
    def weights(self) -> pd.DataFrame:
        path = Path(__file__).parent / 'emp008' / self.factor_filename
        if not path.exists():
            raise FileNotFoundError(str(path))

        df = pd.read_excel(path)
        df.set_index('Unnamed: 0', inplace=True)
        df.index = pd.to_datetime(df.index)
        df.index.name = None
        return df


if __name__ == "__main__":
    method = MethodologyEMP008(mkt="KOSPI200",
                               start_date="20241230",
                               end_date="20250812",
                               factor_filename="emp008_w_cnn.xlsx")
