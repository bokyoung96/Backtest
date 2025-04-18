from typing import Dict, Type
from enum import Enum, unique

from bt.methodologies.methodology_data_validation import *
from bt.methodologies.methodology_err_chg import *
from bt.methodologies.methodology_opr_chg import *

@unique
class MethodologyType(Enum):
    DataValidation = "DataValidation"
    ERRChg = "ERRChg"
    OPRChg = "OPRChg"

methodology_clses: Dict[MethodologyType, Type] = {
    MethodologyType.DataValidation: MethodologyDataValidation,
    MethodologyType.ERRChg: MethodologyERRChg,
    MethodologyType.OPRChg: MethodologyOPRChg
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
