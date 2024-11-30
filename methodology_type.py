from typing import Dict, Type
from enum import Enum, unique

from methodology_data_validation import *
from methodology_gpa_lfq0 import *
from methodology_ebitdaev_ttm_lfq0 import *
from methodology_fcfev_ttm_lfq0 import *
from methodology_momentum3612_1 import *
from methodology_payout_ttm_lfq0 import *


@unique
class MethodologyType(Enum):
    DataValidation = "DataValidation"
    GPAlfq0 = "GPAlfq0"
    EBITDAEVttmlfq0 = "EBITDAEVttmlfq0"
    FCFEVttmlfq0 = "FCFEVttmlfq0"
    Momentum3612_1 = "Momentum3612_1"
    Payoutttmlfq0 = "Payoutttmlfq0"


methodology_clses: Dict[MethodologyType, Type] = {
    MethodologyType.DataValidation: MethodologyDataValidation,
    MethodologyType.GPAlfq0: MethodologyGPAlfq0,
    MethodologyType.EBITDAEVttmlfq0: MethodologyEBITDAEVttmlfq0,
    MethodologyType.FCFEVttmlfq0: MethodologyFCFEVttmlfq0,
    MethodologyType.Momentum3612_1: MethodologyMomentum3612_1,
    MethodologyType.Payoutttmlfq0: MethodologyPayoutttmlfq0
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
