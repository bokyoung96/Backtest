import time
import datetime as dt

from tools import *
from loader import *
from methodology import *

# NOTE: Used for data validation by comparing KOSPI 200.


class MethodologyDataValidation(Methodology):
    def __init__(self,
                 mkt: str = 'KOSPI200',
                 start_date: str = '20121228',
                 end_date: str = '20241031',
                 **kwargs):
        freq = kwargs.pop('freq', 'monthly')
        super().__init__(mkt, start_date, end_date, **kwargs)

        self.freq = freq

        self.load_data()

    def load_data(self) -> Dict[str, pd.DataFrame]:
        data_names = ['price_adj',
                      'mktcap_float']
        raw_data = Tools.get_data(mkt=self.mkt,
                                  data_names=data_names,
                                  loader_cls=DataLoader)
        self.data = {name: df[self.start_date:self.end_date]
                     for name, df in raw_data.items()}
        self.const = DataLoader(
            mkt=self.mkt).data_constituents[self.start_date: self.end_date]

    @property
    def weights(self) -> pd.DataFrame:
        df1 = Tools.get_data_freq(self.data['mktcap_float'],
                                  freq=self.freq)

        df2 = self.const.copy().replace(0, np.nan)
        df2.index = df1.index

        w = df1 * df2
        w = w.div(w.sum(axis=1), axis=0)
        return w
