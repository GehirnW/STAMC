# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 09:49:17 2018

@author: admin
"""
import pandas as pd
import numpy as np

import utils
from imp import reload
reload(utils)


def get_tradeday_before(date, tdays, timelong=0):
    '''
    Get the tradeday before timelong days
    
    Params:
        date:
            string, like '%Y%m%d'
        tdays:
            trade day series
        timelong:
            int, how many days before 
    Returns:
        startdate:
            string, like '%Y%m%d'
    '''
    return tdays[tdays[tdays == date].index[0] - timelong]


def get_alpha2(date, tdays, stock_market_data, timelong=1):
    '''
    (-1 * DELTA((((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW)), 1))
    
    Params:
        date:
            string, like '%Y%m%d'
        tdays:
            trade day series
        stock_market_data:
            pd.DataFrame, multiindex: trade_date, stock_id 
                          columns: Open, High, Low, Close, PctChg, Vol, Amount 
        timelong:
            int, how many days before 
    Return:
        pd.Series, index: stock_id, value: alpha    
    '''
    startdate = get_tradeday_before(date, tdays, timelong)   
  
    res = (stock_market_data.loc[startdate: date]
           .assign(temp = lambda df: ( (df['Close'] - df['Low']) - (df['High'] - df['Close']) )
                                     / (df['High'] - df['Low']))
           .groupby(level='stock_id')['temp'].diff(timelong)
           .loc[date]
           .dropna()
           .reset_index(level='trade_date', drop=True)
           .rename('alpha2'))
           
    return -1 * res

    
def get_alpha6(date, tdays, stock_market_data, timelong=4):
    '''
    (RANK(SIGN(  DELTA( ( (OPEN * 0.85) + (HIGH * 0.15) ), 4 )  ))* -1)
    
    Params:
        date:
            string, like '%Y%m%d'
        tdays:
            trade day series
        stock_market_data:
            pd.DataFrame, multiindex: trade_date, stock_id 
                          columns: Open, High, Low, Close, PctChg, Vol, Amount 
        timelong:
            int, how many days before 
    Return:
        pd.Series, index: stock_id, value: alpha
    '''
    startdate = get_tradeday_before(date, tdays, timelong)
    
    res = (stock_market_data.loc[startdate:date]
           .assign(temp = lambda df: df['Open'] * 0.85 + df['High'] * 0.15)
           .assign(delta = lambda df: df.groupby(level='stock_id')['temp'].diff(timelong))
           .assign(sign = lambda df: utils.Sign(df['delta']))
           .loc[date]
           .dropna() #need to drop before rank
           .assign(rank = lambda df: df['sign'].rank(ascending=True, pct=True))
           .reset_index(level='trade_date', drop=True)
           .loc[:,'rank']
           .rename('alpha6'))
    
    return -1 * res
    

    
def get_alpha14(date, tdays, stock_market_data, timelong=5):
    '''
    CLOSE-DELAY(CLOSE,5)
    
    Params:
        date:
            string, like '%Y%m%d'
        tdays:
            trade day series
        stock_market_data:
            pd.DataFrame, multiindex: trade_date, stock_id 
                          columns: Open, High, Low, Close, PctChg, Vol, Amount 
        timelong:
            int, how many days before 
    Return:
        pd.Series, index: stock_id, value: alpha
    '''
    startdate = get_tradeday_before(date, tdays, timelong)
    
    res = (stock_market_data.loc[startdate: date]
           .groupby(level='stock_id')['Close'].diff(timelong)
           .loc[date]
           .dropna()
           .reset_index(level='trade_date', drop=True)
           .rename('alpha14'))
    
    return res

    
def get_alpha15(date, tdays, stock_market_data, timelong=1):
    '''
    OPEN/DELAY(CLOSE,1)-1
    
    Params:
        date:
            string, like '%Y%m%d'
        tdays:
            trade day series
        stock_market_data:
            pd.DataFrame, multiindex: trade_date, stock_id 
                          columns: Open, High, Low, Close, PctChg, Vol, Amount 
        timelong:
            int, how many days before 
    Return:
        pd.Series, index: stock_id, value: alpha
    '''
    startdate = get_tradeday_before(date, tdays, timelong)
    
    data = (stock_market_data.loc[startdate: date]
           .assign(delay_close = lambda df: df.groupby(level='stock_id')['Close'].shift(timelong))
           .loc[date]
           .dropna()
           .reset_index(level='trade_date', drop=True))
    
    res = data['Open'] / data['delay_close'] -1
    res.name = 'alpha15'
    return res
    

def get_alpha18(date, tdays, stock_market_data, timelong=5):
    '''
    CLOSE/DELAY(CLOSE,5)
    
    Params:
        date:
            string, like '%Y%m%d'
        tdays:
            trade day series
        stock_market_data:
            pd.DataFrame, multiindex: trade_date, stock_id 
                          columns: Open, High, Low, Close, PctChg, Vol, Amount 
        timelong:
            int, how many days before 
    Return:
        pd.Series, index: stock_id, value: alpha
    '''
    startdate = get_tradeday_before(date, tdays, timelong)
    
    res = (stock_market_data.loc[startdate: date]
          .groupby(level='stock_id')['Close'].pct_change(timelong)
          .loc[date]
          .dropna()
          .reset_index(level='trade_date', drop=True)
          .rename('alpha18'))
    
    return res + 1

    
def get_alpha20(date, tdays, stock_market_data, timelong=6):
    '''
    (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100
    
    Params:
        date:
            string, like '%Y%m%d'
        tdays:
            trade day series
        stock_market_data:
            pd.DataFrame, multiindex: trade_date, stock_id 
                          columns: Open, High, Low, Close, PctChg, Vol, Amount 
        timelong:
            int, how many days before 
    Return:
        pd.Series, index: stock_id, value: alpha
    '''
    startdate = get_tradeday_before(date, tdays, timelong)
    
    res = (stock_market_data.loc[startdate: date]
          .groupby(level='stock_id')['Close'].pct_change(timelong)
          .loc[date]
          .dropna()
          .reset_index(level='trade_date', drop=True)
          .rename('alpha20'))
    
    return 100 * res
    

def get_alpha31(date, tdays, stock_market_data, timelong=12):
    '''
    (CLOSE-MEAN(CLOSE,12))/MEAN(CLOSE,12)*100
    
    Params:
        date:
            string, like '%Y%m%d'
        tdays:
            trade day series
        stock_market_data:
            pd.DataFrame, multiindex: trade_date, stock_id 
                          columns: Open, High, Low, Close, PctChg, Vol, Amount 
        timelong:
            int, how many days before 
    Return:
        pd.Series, index: stock_id, value: alpha
    '''
    startdate = get_tradeday_before(date, tdays, timelong)
    
    data = (stock_market_data.loc[startdate: date]
           .assign(rolling_mean = lambda df: df['Close']
                   .groupby(level='stock_id', group_keys=False).rolling(timelong).mean())
           .loc[date]
           .dropna()
           .reset_index(level='trade_date', drop=True))
           
    res = 100 * (data['Close'] / data['rolling_mean'] -1)
    res.name = 'alpha31'
    return res


def get_alpha34(date, tdays, stock_market_data, timelong=12):
    '''
    MEAN(CLOSE,12)/CLOSE
    
    Params:
        date:
            string, like '%Y%m%d'
        tdays:
            trade day series
        stock_market_data:
            pd.DataFrame, multiindex: trade_date, stock_id 
                          columns: Open, High, Low, Close, PctChg, Vol, Amount 
        timelong:
            int, how many days before 
    Return:
        pd.Series, index: stock_id, value: alpha
    '''
    startdate = get_tradeday_before(date, tdays, timelong)
    
    data = (stock_market_data.loc[startdate: date]
           .assign(rolling_mean = lambda df: df['Close']
                   .groupby(level='stock_id', group_keys=False).rolling(timelong).mean())
           .loc[date]
           .dropna()
           .reset_index(level='trade_date', drop=True))
    res = data['rolling_mean'] / data['Close']

    if timelong == 12:
        res.name = 'alpha34'
    elif timelong == 6:
        res.name = 'alpha65'
    return res  


def get_alpha53(date, tdays, stock_market_data, timelong=12):
    '''
    COUNT(CLOSE>DELAY(CLOSE,1),12) / 12 * 100, 12 is the timelong
    
    Params:
        date:
            string, like '%Y%m%d'
        tdays:
            trade day series
        stock_market_data:
            pd.DataFrame, multiindex: trade_date, stock_id 
                          columns: Open, High, Low, Close, PctChg, Vol, Amount 
        timelong:
            int, how many days before 
    Return:
        pd.Series, index: stock_id, value: alpha
    '''
    startdate = get_tradeday_before(date, tdays, timelong)
    
    data = (stock_market_data.loc[startdate: date]
           .assign(delay_close = lambda df: df.groupby(level='stock_id')['Close'].shift(1))
           .assign(temp = lambda df: np.where(df['Close']-df['delay_close']>0, 1, 0))
           .assign(count = lambda df: df['temp'].groupby(level='stock_id', group_keys=False)
                                               .rolling(timelong).sum())
           .loc[date]
           .dropna()
           .reset_index(level='trade_date', drop=True))
    res = 100 * data['count'] / timelong

    if timelong == 12:
        res.name = 'alpha53'
    elif timelong == 20:
        res.name = 'alpha58'
    return res
    
"""
def get_alpha137():
    '''
    16*(CLOSE-DELAY(CLOSE,1)+(CLOSE-OPEN)/2+DELAY(CLOSE,1)-DELAY(OPEN,1))/((ABS(HIGH-DELAY(CLOSE,
    1))>ABS(LOW-DELAY(CLOSE,1)) &
    ABS(HIGH-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1))?ABS(HIGH-DELAY(CLOSE,1))+ABS(LOW-DELAY(CLOS
    E,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:(ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1)) &
    ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(CLOSE,1))?ABS(LOW-DELAY(CLOSE,1))+ABS(HIGH-DELAY(CLO
    SE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:ABS(HIGH-DELAY(LOW,1))+ABS(DELAY(CLOSE,1)-DELAY(OP
    EN,1))/4)))*MAX(ABS(HIGH-DELAY(CLOSE,1)),ABS(LOW-DELAY(CLOSE,1)))
    '''
    return res
"""

@utils.timer
def main(date, tdays, stock_market_data, indus_factor, cap):
    '''
    Get all the factor values
    
    Params:
        date:
            string, like '%Y%m%d'
        tdays:
            trade day series
        stock_market_data:
            pd.DataFrame, multiindex: trade_date, stock_id 
                     columns: Open, High, Low, Close, PctChg, Vol, Amount
        indus_factor:
            pd.DataFrame, index: stock_id, columns: industry name
        cap:
            pd.Seris, index: stock_id
    Return:
        pd.DataFrame:
            index: stock_id, columns: alpha names
    '''
    alpha2 = get_alpha2(date, tdays, stock_market_data)
    # alpha6 = get_alphas.get_alpha6(date, tdays, stock_market_data)
    alpha14 = get_alpha14(date, tdays, stock_market_data)
    alpha15 = get_alpha15(date, tdays, stock_market_data)
    alpha18 = get_alpha18(date, tdays, stock_market_data)
    alpha20 = get_alpha20(date, tdays, stock_market_data)
    alpha31 = get_alpha31(date, tdays, stock_market_data)
    alpha34 = get_alpha34(date, tdays, stock_market_data)
    alpha53 = get_alpha53(date, tdays, stock_market_data)
    alpha58 = get_alpha53(date, tdays, stock_market_data, timelong=20)
    alpha65 = get_alpha34(date, tdays, stock_market_data, timelong=6)
    
    df_alpha = pd.concat([alpha2, alpha14, alpha15,
                          alpha18, alpha20, alpha31, alpha34,
                          alpha53, alpha58, alpha65], axis=1)
    df_alpha = utils.data_replacement(df_alpha, indus_factor, cap)
    return df_alpha

