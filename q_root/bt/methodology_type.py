from typing import Dict, Type
from enum import Enum, unique

from bt.methodologies.methodology_data_validation import *
from bt.methodologies.methodology_err_chg import *
from bt.methodologies.methodology_opr_chg import *
from bt.methodologies.methodology_eps_chg import *
from bt.methodologies.methodology_eps_chg_fy1 import *
from bt.methodologies.methodology_eps_chg_fy2_vol_adj import *
from bt.methodologies.methodology_pb_fq1 import *
from bt.methodologies.methodology_price_trends import *
from bt.methodologies.methodology_momentum import *
from bt.methodologies.methodology_m_pt_cherrypick import *
from bt.methodologies.methodology_momentum_comb import *
from bt.methodologies.methodology_m_comb_pt_cherrypick import *
from bt.methodologies.methodology_price_trends_abs import *
from bt.methodologies.methodology_price_trends_abs1 import *

@unique
class MethodologyType(Enum):
    DataValidation = "DataValidation"
    ERRChg = "ERRChg"
    OPRChg = "OPRChg"
    EPSChg = "EPSChg"
    EPSChgFY1 = "EPSChgFY1"
    DonchianChannel = "DonchianChannel"
    EPSChgFY2VolAdj = "EPSChgFY2VolAdj"
    MethodologyPBFQ1SectorNeutral = "MethodologyPBFQ1SectorNeutral"
    
    # PRICE TRENDS BY CNN CHART IMAGE
    MethodologyPriceTrends = "MethodologyPriceTrends"
    MethodologyMomentum = "MethodologyMomentum"
    MethodologyMPTCherrypick = "MethodologyMPTCherrypick"
    MethodologyMomentumComb = "MethodologyMomentumComb"
    MethodologyMCombPTCherrypick = "MethodologyMCombPTCherrypick"
    MethodologyPriceTrendsAbs = "MethodologyPriceTrendsAbs"
    MethodologyPriceTrendsAbs1 = "MethodologyPriceTrendsAbs1"

methodology_clses: Dict[MethodologyType, Type] = {
    MethodologyType.DataValidation: MethodologyDataValidation,
    MethodologyType.ERRChg: MethodologyERRChg,
    MethodologyType.OPRChg: MethodologyOPRChg,
    MethodologyType.EPSChg: MethodologyEPSChg,
    MethodologyType.EPSChgFY1: MethodologyEPSChgFY1,
    MethodologyType.EPSChgFY2VolAdj: MethodologyEPSChgFY2VolAdj,
    MethodologyType.MethodologyPBFQ1SectorNeutral: MethodologyPBFQ1SectorNeutral,
    MethodologyType.MethodologyPriceTrends: MethodologyPriceTrends,
    MethodologyType.MethodologyMomentum: MethodologyMomentum,
    MethodologyType.MethodologyMPTCherrypick: MethodologyMPTCherrypick,
    MethodologyType.MethodologyMomentumComb: MethodologyMomentumComb,
    MethodologyType.MethodologyMCombPTCherrypick: MethodologyMCombPTCherrypick,
    MethodologyType.MethodologyPriceTrendsAbs: MethodologyPriceTrendsAbs,
    MethodologyType.MethodologyPriceTrendsAbs1: MethodologyPriceTrendsAbs1
}


def methodology_pool(methodology_type: MethodologyType,
                     mkt: str,
                     start_date: str,
                     end_date: str,
                     **kwargs):
    methodology_cls = methodology_clses.get(methodology_type)
    if methodology_cls:
        return methodology_cls(mkt, start_date, end_date, **kwargs)
    else:
        raise ValueError(f"Unknown methodology: {methodology_type}")