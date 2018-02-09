# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 18:49:22 2018

@author: admin
"""
from enum import IntEnum
from enum import unique

@unique
class Descriptors(IntEnum):
    BETA = 1
    HSIGMA = 2
    RSTR = 3
    LNCAP = 4
    EPIBS = 5
    CETOP = 6
    ETOP = 7
    DASTD = 8
    CMRA = 9
    EGRO = 10
    SGRO = 11
    EGRLF = 12
    EGRSF = 13
    BTOP = 14
    MLEV = 15
    DTOA = 16
    BLEV = 17
    STOM = 18
    STOQ = 19
    STOA = 20
    NLSIZE = 21
                      
@unique
class StyleFactors(IntEnum):
    Beta = 1
    Momentum = 2
    Size = 3
    Earnings_Yield = 4
    Residual_Volatility = 5
    Growth = 6
    Book_to_Price = 7
    Leverage = 8
    Liquidity = 9
    Non_linear_Size = 10