# -*- coding: utf-8 -*-

from enum import Enum
from enum import unique


class StrEnum(str, Enum):
    pass


@unique
class ColName(StrEnum):
    Code = 'code'
    Date = 'date'
    Return = 'ret'


@unique
class DBTable(StrEnum):
    ForecastTable = 'CON_FORECAST_STK'
    FinancialStatement = 'FinancialStatement'
    AlphaFactors = 'StockAlphaFactors'
