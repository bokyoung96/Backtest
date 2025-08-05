import pandas as pd
import numpy as np
from typing import Dict
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter

from bt.tools import Tools
from bt.loader import DataLoader
from bt.methodology import Methodology


class MethodologyLifecycle(Methodology):
    def __init__(self,
                 mkt: str = 'KOSPI200',
                 start_date: str = '20110101',
                 end_date: str = '20241031',
                 **kwargs):
        freq = kwargs.pop('freq', 'monthly')
        quantile = kwargs.pop('quantile', 10)
        quantile_position = kwargs.pop('quantile_position', [1])
        weight_type = kwargs.pop('weight_type', 'mktcap_float')
        lifecycle_stage = kwargs.pop('lifecycle_stage', 'maturity')
        super().__init__(mkt, start_date, end_date, **kwargs)

        self.freq = freq
        self.quantile = quantile
        self.quantile_position = quantile_position
        self.weight_type = weight_type
        self.lifecycle_stage = lifecycle_stage

        self.load_data()
        self.load_const()

    def load_data(self) -> Dict[str, pd.DataFrame]:
        data_names = ['price_adj',
                      'mktcap_float',
                      'ocf_nrfq1',
                      'facf_nrfq1',
                      'icf_nrfq1']
        raw_data = Tools.get_data(mkt=self.mkt,
                                  data_names=data_names,
                                  loader_cls=DataLoader)
        self.data = {name: df[self.start_date:self.end_date]
                     for name, df in raw_data.items()}

    def load_const(self):
        const = DataLoader(
            mkt=self.mkt).data_constituents[self.start_date: self.end_date]
        self.const = Tools.get_data_align(const=const,
                                          prc=self.data['price_adj'])
                                          
    def get_lifecycle(self):
        """
        Classifier
        - Introduction: OCF(-), ICF(-), FCF(+)
        - Growth: OCF(+), ICF(-), FCF(+)
        - Maturity: OCF(+), ICF(-), FCF(-)
        - Shake-out: OCF(+), ICF(+), FCF(-)
        - Decline: OCF(-), ICF(+), FCF(-)
        """
        ocf = self.data['ocf_nrfq1']
        icf = self.data['icf_nrfq1']
        fcf = self.data['facf_nrfq1']

        ocf_sign = np.sign(ocf)
        icf_sign = np.sign(icf)
        fcf_sign = np.sign(fcf)
        
        introduction = (ocf_sign < 0) & (icf_sign < 0) & (fcf_sign > 0)
        growth = (ocf_sign > 0) & (icf_sign < 0) & (fcf_sign > 0)
        maturity = (ocf_sign > 0) & (icf_sign < 0) & (fcf_sign < 0)
        shake_out = (ocf_sign > 0) & (icf_sign > 0) & (fcf_sign < 0)
        decline = (ocf_sign < 0) & (icf_sign > 0) & (fcf_sign < 0)
        
        classifier = {
            'introduction': introduction,
            'growth': growth,
            'maturity': maturity,
            'shake_out': shake_out,
            'decline': decline
        }
        return classifier
    
    def get_raw_factor(self):
        if self.lifecycle_stage:
            lifecycle_classifications = self.get_lifecycle()
            if self.lifecycle_stage in lifecycle_classifications:
                raw_factor = lifecycle_classifications[self.lifecycle_stage]
                return raw_factor
            else:
                raise ValueError(f"Invalid lifecycle stage: {self.lifecycle_stage}. " 
                                 f"Valid stages are: {list(lifecycle_classifications.keys())}")
        else:
            raise ValueError("Lifecycle stage is not specified.")

    def get_pp_data(self):
        const = Tools.get_data_freq(df=Tools.get_nan(df=self.const,
                                                     val=[0]),
                                    freq=self.freq)
        raw_factor = Tools.get_data_freq(df=self.get_raw_factor(),
                                         freq=self.freq)
        
        aligned_factor = Tools.get_data_align(
            const=raw_factor,
            prc=const,
            check_nan=False,
            fill_method=None
        )

        try:
            Tools.validation_df_size(const, aligned_factor)
            aligned_factor = aligned_factor.replace({True: 1, False: np.nan}).infer_objects(copy=False)
            return const, aligned_factor
        except ValueError as e:
            raise ValueError(f"Failed to match frequency: {e}")

    def get_raw_weights(self):
        const, aligned_factor = self.get_pp_data()
        factor = const.mul(aligned_factor)
        return factor

    @property
    def weights(self):
        mktcap_float = Tools.get_data_freq(df=self.data['mktcap_float'],
                                           freq=self.freq)
        quantile_weights = self.get_raw_weights()
        w = quantile_weights * \
            mktcap_float if self.weight_type == 'mktcap_float' else quantile_weights

        w.dropna(axis=0, how='all', inplace=True)
        w = w.div(w.sum(axis=1), axis=0)
        return w


if __name__ == "__main__":
    method = MethodologyLifecycle(mkt="KOSPI200",
                                 start_date="20110101",
                                 end_date="20241031",
                                 lifecycle_stage="growth")
