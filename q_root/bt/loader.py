import os
import pandas as pd

from enum import Enum, unique


# NOTE: If updating price data, benchmark data & trans_ban data should be updated in addition.
# NOTE: If updating price data, update mktcap_float data, ensuring its data size equals.
# NOTE: For price, trans_ban data, KOSPI200 data is used as substitute for index futures data.

BS = "data_bs_%s"
IS = "data_is_%s"
CF = "data_cf_%s"
FN = "data_fn_%s"
BASE = "data_base_%s"


@unique
class DirMkt(Enum):
    KOSPI = "KOSPI"
    KOSPI200 = "KOSPI200"
    KOSDAQ = "KOSDAQ"
    KOSDAQ150 = "KOSDAQ150"
    ALL = "ALL"


@unique
class BSItem(Enum):
    asset_lfq0 = "asset_lfq0"
    equity_nfq1 = "equity_nfq1"

    def as_bs(self):
        return BS % self.value


@unique
class ISItem(Enum):
    gp_lfq0 = "gp_lfq0"
    ebitda_ttm_lfq0 = "ebitda_ttm_lfq0"
    ni_ttm_lfq0 = "ni_ttm_lfq0"
    sales_ttm_nfq1 = "sales_ttm_nfq1"
    err_1m = "err_1m"
    err_2m = "err_2m"
    opr_nfq1 = "opr_nfq1"
    opr_nfq1_e = "opr_nfq1_e"

    def as_is(self):
        return IS % self.value


@unique
class CFItem(Enum):
    fcf_ttm_lfq0 = "fcf_ttm_lfq0"
    dividends_ttm_lfq0 = "dividends_ttm_lfq0"
    ocf_nrfq1 = "ocf_nrfq1"
    facf_nrfq1 = "facf_nrfq1"
    icf_nrfq1 = "icf_nrfq1"

    def as_cf(self):
        return CF % self.value


@unique
class FNItem(Enum):
    pbr_ttm_lfq0 = "pbr_ttm_lfq0"
    mtob_ttm_lfq0 = "mtob_ttm_lfq0"
    eps_nfq1_e = "eps_nfq1_e"
    eps_nfy1_e = "eps_nfy1_e"
    eps_nfy2_e = "eps_nfy2_e"
    eps_nfy0 = "eps_nfy0"
    
    def as_fn(self):
        return FN % self.value


@unique
class BaseItem(Enum):
    bm = "bm"
    price_adj = "price_adj"
    volume = "volume"
    shares_outstanding = "shares_outstanding"
    mktcap_float = "mktcap_float"
    mktcap = "mktcap"
    trans_ban = "trans_ban"
    ev_ttm_lfq0 = "ev_ttm_lfq0"
    wics_sector_big = "wics_sector_big"
    wics_sector_26 = "wics_sector_26"
    donchian_ohlc = "donchian_ohlc"

    def as_base(self):
        return BASE % self.value


@unique
class DataPool(Enum):
    # NOTE: Balance Sheet Items
    data_bs_asset_lfq0 = BSItem.asset_lfq0.as_bs()
    data_bs_equity_nfq1 = BSItem.equity_nfq1.as_bs()

    # NOTE: Income Statement Items
    data_is_gp_lfq0 = ISItem.gp_lfq0.as_is()
    data_ebitda_ttm_lfq0 = ISItem.ebitda_ttm_lfq0.as_is()
    data_ni_ttm_lfq0 = ISItem.ni_ttm_lfq0.as_is()
    data_sales_ttm_nfq1 = ISItem.sales_ttm_nfq1.as_is()
    data_err_1m = ISItem.err_1m.as_is()
    data_err_2m = ISItem.err_2m.as_is()
    data_opr_nfq1 = ISItem.opr_nfq1.as_is()
    data_opr_nfq1_e = ISItem.opr_nfq1_e.as_is()

    # NOTE: CF Items
    data_cf_fcf_ttm_lfq0 = CFItem.fcf_ttm_lfq0.as_cf()
    data_dividends_ttm_lfq0 = CFItem.dividends_ttm_lfq0.as_cf()
    data_ocf_nrfq1 = CFItem.ocf_nrfq1.as_cf()
    data_facf_nrfq1 = CFItem.facf_nrfq1.as_cf()
    data_icf_nrfq1 = CFItem.icf_nrfq1.as_cf()

    # NOTE: Financial Items
    data_fn_pbr_ttm_lfq0 = FNItem.pbr_ttm_lfq0.as_fn()
    data_fn_mtob_ttm_lfq0 = FNItem.mtob_ttm_lfq0.as_fn()
    data_fn_eps_nfq1_e = FNItem.eps_nfq1_e.as_fn()
    data_fn_eps_nfy1_e = FNItem.eps_nfy1_e.as_fn()
    data_fn_eps_nfy2_e = FNItem.eps_nfy2_e.as_fn()
    data_fn_eps_nfy0 = FNItem.eps_nfy0.as_fn()
    
    # NOTE: Base Items
    data_base_bm = BaseItem.bm.as_base()
    data_base_price_adj = BaseItem.price_adj.as_base()
    data_base_volume = BaseItem.volume.as_base()
    data_base_shares_outstanding = BaseItem.shares_outstanding.as_base()
    data_base_mktcap_float = BaseItem.mktcap_float.as_base()
    data_base_mktcap = BaseItem.mktcap.as_base()
    data_base_trans_ban = BaseItem.trans_ban.as_base()
    data_base_ev_ttm_lfq0 = BaseItem.ev_ttm_lfq0.as_base()
    data_base_wics_sector_big = BaseItem.wics_sector_big.as_base()
    data_base_donchian_ohlc = BaseItem.donchian_ohlc.as_base()
    data_base_wics_sector_26 = BaseItem.wics_sector_26.as_base()

class DataLoader:
    def __init__(self,
                 mkt: str = 'KOSPI200',
                 data_dir: str = None):
        self.data_verifier(mkt=mkt)
        self.mkt_value = DirMkt[mkt].value
        self.data_dir = data_dir if data_dir else os.path.join(
            os.path.dirname(__file__), 'DATA')

    @staticmethod
    def data_verifier(mkt):
        if mkt not in DirMkt.__members__:
            valid_options = ', '.join(DirMkt.__members__)
            raise ValueError(
                f"Invalid mkt value. Valid options are: {valid_options}")

    @property
    def data_constituents(self) -> pd.DataFrame:
        data_path = os.path.join(
            self.data_dir, f'data_const_{self.mkt_value}.parquet')
        return pd.read_parquet(data_path)

    def __repr__(self) -> str:
        return f"mkt_value: {self.mkt_value}, data_dir: {self.data_dir}"

    def __call__(self,
                 data_name: str,
                 tr_yn: bool = False) -> pd.DataFrame:
        res = None
        for member_name, member in DataPool.__members__.items():
            if data_name == member.value:
                res = member_name
                break
        
        if res is None:
            for member_name, member in DataPool.__members__.items():
                short_name = member.value.split('_', 2)[-1]
                if data_name == short_name:
                    res = member_name
                    break
        
        if res is None:
            available_data = []
            for member in DataPool.__members__.values():
                full_name = member.value
                short_name = full_name.split('_', 2)[-1]
                available_data.append(f"{short_name} (full: {full_name})")
            raise ValueError(f"Data name '{data_name}' not found. Available data names are: {available_data}")

        data_path = os.path.join(
            self.data_dir, f'{DataPool[res].value}.parquet')
        if data_name == 'bm' or data_name == 'data_base_bm':
            data = self.data_loader_bm(data_path=data_path,
                                       tr_yn=tr_yn)
        else:
            data = self.data_loader_others(data_path=data_path)
        return data

    def data_loader_bm(self,
                       data_path: str,
                       tr_yn: bool) -> pd.DataFrame:
        df = pd.read_parquet(data_path)
        key = f"{self.mkt_value}_TR" if tr_yn else self.mkt_value
        if key in df.columns:
            return df[[key]]
        else:
            raise ValueError(f"Benchmark data for {key} not found.")

    def data_loader_others(self,
                           data_path: str) -> pd.DataFrame:
        df = pd.read_parquet(data_path)
        if self.mkt_value == 'ALL':
            return df
        else:
            valid_items = [
                item for item in self.data_constituents.columns if item in df.columns]
            return df[valid_items]

    @staticmethod
    def get_data_size(directory: str = None) -> pd.DataFrame:
        if directory is None:
            directory = os.path.join(os.path.dirname(__file__), 'DATA')
        files = [f for f in os.listdir(directory) if f.endswith('.parquet')]

        file_info = []
        for file in files:
            file_path = os.path.join(directory, file)
            try:
                df = pd.read_parquet(file_path)
                file_info.append(
                    {'File_name': file, 'Idx': df.shape[0], 'Col': df.shape[1]})
            except Exception as e:
                print(f"Error loading {file}: {e}")
                file_info.append({'File_name': file, 'Idx': None, 'Col': None})

        df = pd.DataFrame(file_info)
        return df
