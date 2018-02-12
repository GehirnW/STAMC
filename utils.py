# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 10:04:46 2018

@author: admin
"""

import pandas as pd
import numpy as np
from pypika import Query, Table, Tables, Field
from functools import wraps
import time
from datetime import timedelta
from enums import FactorNum
from scipy import stats
from cvxpy import Variable, Maximize, sum_entries, Problem, abs


def timer(function):
  @wraps(function)
  def function_timer(*args, **kwargs):
      t0 = time.time()
      result = function(*args, **kwargs)
      t1 = time.time()
      print ("Total time running %s: %s seconds" %(function.__name__, str(round((t1-t0), 2))))
      return result
  return function_timer
  
  
def get_tdays(engine):
    '''
    Params:
        engine:
            database engine
    Returns:
        pd.Series, values: str
    
    '''
    a = Table('trade_calendar')
    q = Query.from_(a).select(
        a.calendar_date
    ).where(
        a.is_trade_day == 1
    ).orderby(
        a.calendar_date)
    
    tdays = (pd.read_sql(q.get_sql(), engine)
           .calendar_date
           .apply(lambda dt: dt.strftime('%Y%m%d')))
    return tdays

def get_startdate(date, timelong=0):
    '''
    Params:
        date:
            string, like '%Y%m%d'
        timelong:
            float, how many years date before 
    Returns:
        startdate:
            string, like '%Y%m%d'
    '''
    return (pd.to_datetime(date) - timedelta(days = int(365 * timelong))).strftime('%Y%m%d')    

    
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
    
    
def get_tradeday_after(date, tdays, timelong=0):
    '''
    Get the tradeday after timelong days
    
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
    return tdays[tdays[tdays == date].index[0] + timelong]    


def winsorize(ts, method = 'mad', alpha = 0.05, nsigma = 3):
    '''
    Remove abnormal value from pd.Series data.
    
    see: 东方金工: 选股因子数据的异常值处理和正态转换——《金工磨刀石系列之二》
    
    Params:
        ts:
            pd.Series
        method:
            'quantiles': 
                set data under alpha/2 and above 1 - alpha/2 to the quantile value
            'mv': 
                set data beyond nsigma(3) sigma to the nsigma(3) sigma value
            'mad': 
                use Median Absolute Deviation(MAD) instead of mean,
                md = median(dataset)
                MAD = 1.483*median(|dataset - md|), this MAD is similar to sigma
                set data beyond nsigma(3) MAD to the nsigma(3) MAD value
            'boxplot_adj':
                Turkey, MedCouple. see: https://en.wikipedia.org/wiki/Medcouple
        [option]:
            alpha:  valid for method = 'quantiles'
            nsigma: valid for method = 'mv' and 'mad'
    Return:
        winsorized pd.Series
    '''
    ts = ts.copy().dropna()
    if method == 'quantiles':
        p_d = ts.quantile(alpha / 2.)
        p_u = ts.quantile(1 - alpha / 2.)
        ts[ts > p_u] = p_u
        ts[ts < p_d] = p_d
    elif method == 'mv':
        sigma = ts.std()
        ts[ts > ts.mean() + sigma * nsigma] = ts.mean() + sigma * nsigma
        ts[ts < ts.mean() - sigma * nsigma] = ts.mean() - sigma * nsigma
    elif method == 'mad':
        md = ts.median()
        MAD = 1.483 * (ts - md).abs().median()
        ts[ts > md + MAD * nsigma] = md + MAD * nsigma
        ts[ts < md - MAD * nsigma] = md - MAD * nsigma
    elif method == 'boxplot_adj':
        ts = ts.sort_values(ascending = False)
        md = ts.median()
        x_u, x_d = ts[ts >= md], ts[ts <= md]
        def h(i, j):
            a, b = x_u.iloc[i], x_d.iloc[j]
            if a == b:
                return np.sign(len(x_u) - 1 - i - j)
            else:
                return (a + b - 2 * md) / (a - b)
        # mc = pd.Series([h(i, i) for i in range(len(x_u))]).median()
        h = [h(i, j) for i in range(len(x_u)) for j in range(len(x_d))]
        mc = pd.Series(h).median()
        Q1 = ts.quantile(0.25)
        Q3 = ts.quantile(0.75)
        IQR = Q3 - Q1
        if mc >= 0:
            L = Q1 - 1.5 * np.exp(-3.5 * mc) * IQR
            U = Q1 + 1.5 * np.exp(4 * mc) * IQR
        else:
            L = Q1 - 1.5 * np.exp(-4 * mc) * IQR
            U = Q1 + 1.5 * np.exp(3.5 * mc) * IQR
        ts[(ts > U)] = U
        ts[(ts < L)] = L
    else:
        raise ValueError('No method called: ', method)
    return ts
    

def standardize(ts, method='equal_weighted', cap=None):
    '''
    Standardize data to mean 0 and std 1.
    
    Params:
        ts:
            pd.Series
        method:
            weighting style when calculating mean, 'equal_weighted' or 'cap_weighted'
        cap:
            None or pd.Series, cap, index:code, value:cap
    Returns:
        standardized pd.Series
    '''
    ts = ts.copy().dropna()
    if method == 'equal_weighted':
        return (ts - ts.mean()) / ts.std()
        
    elif method == 'cap_weighted':
        if ts.index.nlevels == 2:
            df = pd.concat([ts.reset_index(level='date',drop=True),cap],axis=1).dropna()
        else:
            df = pd.concat([ts,cap],axis=1).dropna()
        avg = np.average(df.iloc[:,0],weights=df.iloc[:,1])
        return (ts - avg) / ts.std()    

        
def get_stocklist(date, engine, dategap = 90):
    '''
    Get stocklist that :
        1) listed before dategap from date, and has not been delisted till date 
            b.Delisting == '19000101' means still on trade
        2) on date, Turnover > 0 
    
    Params:
        date: 
            string, like '%Y%m%d'
        engine:
            database engine
        dategap:
            days from startdate to date
    Returns:
        list, list of stockcode like ['000001.SZ', '000002.SZ', ...]
    '''
    startdate = (pd.to_datetime(date) - timedelta(days = dategap)).strftime('%Y%m%d')
    
    a, b = Tables('stock_alpha_factors', 'stock_listing_info')       
    q = Query.from_(a).join(
        b
    ).on(
        a.stock_id == b.stock_id
    ).select(
        a.stock_id
    ).where(
        (a.trade_date == date) & (a.Turnover>0) & 
        (b.Listing <= startdate) & ((b.Delisting > date) | (b.Delisting == '19000101'))
    ).orderby(
        a.stock_id)
    
    stocklist = (pd.read_sql(q.get_sql(),engine)
                   .stock_id
                   .drop_duplicates()
                   .tolist())
    return stocklist

    
def get_stocklist_st_limit(date, engine):
    '''
    Get the stocks that are st or limit-up or limit-down
    '''
    a = Table('stock_market_data')
    q = Query.from_(a).select(
        a.stock_id
    ).where(
        (a.trade_date==date) & ((a.st==1) | (a.PctChg > 9.5) | (a.PctChg < -9.5))
    ).orderby(
        a.stock_id)
    stocklist = (pd.read_sql(q.get_sql(),engine)
                   .stock_id
                   .tolist())
    return stocklist
  
    
def get_stock_market_data(date, tdays, engine, stocklist, timelong=22):
    '''
    Get the last 1 month stock market data
    
    Params:
        date:
            string, like '%Y%m%d'
        engine:
            database engine
        stocklist:
            list, list of stockcode like ['000001.SZ', '000002.SZ', ...]
        timelong:
            int, how many days before 
    Return:
        pd.DataFrame, multiindex: trade_date, stock_id 
                     columns: Open, High, Low, Close, PctChg, Vol, Amount  
    '''
    startdate = get_tradeday_before(date, tdays, timelong)   
 
    a = Table('stock_market_data')
    q = Query.from_(a).select(
        a.trade_date, a.stock_id, a.Open, a.High, a.Low, a.Close, a.PctChg, a.Vol, a.Amount
    ).where(
        (a.trade_date[startdate:date]) & (a.stock_id.isin(stocklist)) 
    ).orderby(
        a.trade_date, a.stock_id
    )
    
    data = (pd.read_sql(q.get_sql(), engine)
              .set_index(['trade_date', 'stock_id']))
    
    return data

    
@timer    
def get_close(start, end, engine):
    '''
    Get the close price of stocks between start and end 
    Params:
        start:
            str, like '%Y%m%d'
        end:
            str, like '%Y%m%d'
    Return:
        pd.Series,multiindex, trade_date and stock_id, value: close price
    '''
    a = Table('stock_market_data')
    q = Query.from_(a).select(
        a.trade_date, a.stock_id, a.Close
    ).where(
        a.trade_date[start: end] 
    ).orderby(
        a.trade_date, a.stock_id
    )
    
    data = (pd.read_sql(q.get_sql(), engine)
              .set_index(['trade_date', 'stock_id']))
    return data['Close']
    
    
def get_indus(date, engine, stocklist, timelong=0):
    '''
    get industry factor 
    Better select single date or multidate data
    
    Params:
        date:
            str, like '%Y%m%d'
        engine:
            database engine
        stocklist:
            list, list of stockcode like ['000001.SZ', '000002.SZ', ...]
        timelong:
            float, how many years data to select, 
            default 0, means only selcct one day data
    Returns:
        if timelong=0:
            pd.Series, index:code, value: indus
        if timelong>0:
            pd.Series  multiindex: date and code, value: indus
    '''
    startdate = get_startdate(date, timelong)
    
    a = Table('stock_industry_citic')
    q = Query.from_(a).select(
        a.trade_date, a.stock_id, a.industry_citic
    ).where(
        (a.trade_date[startdate:date]) & (a.stock_id.isin(stocklist)) 
    ).orderby(
        a.trade_date, a.stock_id
    )
    
    data = (pd.read_sql(q.get_sql(), engine)
              .dropna()
           #   .rename(columns = {'SecID':'code', 'Date':'date', 'SectorID':'indus'})
              .assign(trade_date = lambda df: pd.to_datetime(df['trade_date'])))
    
    if timelong == 0:
        return data.set_index('stock_id')['industry_citic']
    else:
        return data.set_index(['trade_date','stock_id'])['industry_citic']
    
    
def get_cap(date, engine, stocklist, timelong=0):
    '''
    Get total market cap of stocks in stocklist (specific date or a series of dates)

    Params:
        date: 
            string, like '%Y%m%d'
        engine:
            database engine
        stocklist:
            list, list of stockcode like ['000001.SZ', '000002.SZ', ...]
        timelong:
            float, how many years data to select, 
            default 0, means only selcct one day data
    Returns:
        if timelong=0:
            pd.Series, index:code, value: cap
        if timelong>0:
            pd.Series  multiindex: code and date, value: cap
    ''' 
    startdate = get_startdate(date, timelong)
    
    a = Table('stock_alpha_factors')      
    q = Query.from_(a).select(     
        a.trade_date, a.stock_id, a.TotalValue
    ).where(
        (a.trade_date[startdate:date]) & (a.stock_id.isin(stocklist))
    ).orderby(
        a.trade_date, a.stock_id
    )
    
    data = (pd.read_sql(q.get_sql(), engine)
              .dropna()
              .rename(columns = {'TotalValue':'cap'})
    #          .rename(columns = {'SecID':'code', 'Date':'date', 'TotalValue':'cap'})
              .assign(trade_date = lambda df: pd.to_datetime(df['trade_date'])))
    
    if timelong == 0:
        return data.set_index('stock_id')['cap']
    else:
        return data.set_index(['trade_date','stock_id'])['cap'] 

                              
@timer
def get_style_factor(date, engine, stocklist, cap, timelong=0, name=None):
    '''
    get style factor exposure after standarlize('cap_weighted')
    Attention: Not to winsorize, because winsorize when calculate descriptors,  
               Beacause it is better winsorize when calculate descriptors
    
    Params:
        date:
            str, like '%Y%m%d'
        engine:
            database engine
        stocklist:
            list, list of stockcode like ['000001.SZ', '000002.SZ', ...]
        cap:
            market value, use to calcualte weighted mean when standarlize
        timelong:
            float, how many years data to select, 
            default 0, means only selcct one day data
        name:
            None or list, the style factor names, default None, all 10 style factors
            ['Beta','Momentum','Size', 'Earnings_Yield','Residual_Volatility',
             'Growth','Book_to_Price','Leverage','Liquidity','Non_linear_Size']
    Returns:
        pd.DataFrame, 
        if timelong=0:
            one day data, index:code, columns: factor names, value: factorvalue
        if timelong>0:
            index: multiindex,'code','date'   columns: factor names, value: factorvalue
    '''
    startdate = get_startdate(date, timelong)
    
    a = Table('BarraStyleFactors_new')      
    q = Query.from_(a).select(
        a.stockcode, a.tradedate, a.factorid, a.factorvalue
    ).where(
        (a.tradedate[startdate:date]) & (a.stockcode.isin(stocklist)) 
    ).orderby(
        a.stockcode, a.tradedate, a.factorid
    )
    
    if name:
        name_id = [int(FactorNum.StyleFactors[i]) for i in name]
        q = q.where(a.factorid.isin(name_id))
    
    data = (pd.read_sql(q.get_sql(), engine)
              .rename(columns = {'stockcode':'stock_id', 'tradedate': 'date'})
              .assign(date = lambda df: pd.to_datetime(df['date'])))    
            
    if timelong == 0:
        data = (pd.pivot_table(data, values='factorvalue',
                               index = 'stock_id',
                               columns = 'factorid')
                  .dropna()
                  .apply(standardize, method='cap_weighted', cap=cap))
    else:
        data = (pd.pivot_table(data, values='factorvalue',
                               index = ['date','stock_id'], 
                               columns='factorid')
                  .dropna()
                  .groupby(level='date')
                  .apply(lambda obj: pd.DataFrame(obj).apply(standardize, method='equal_weighted')))

    data.columns = pd.Series({factorid: name for name, factorid 
                              in FactorNum.StyleFactors.__members__.items()}).loc[data.columns]
    return data

    
def Sign(ts):
    '''
    If value>0 in ts, the new value is 1
    If value=0 in ts, the new value is 0
    If value<0 in ts, the new value is -1
    
    Params:
        ts: pd.Series
    Return:
        pd.Series, with the same index and name as ts
    '''
    res = pd.Series(np.where(ts>0, 1, np.where(ts<0,-1,0)) ,index=ts.index, name=ts.name)
    return res

    
def data_replacement(df, indus_factor, cap):
    df = df.copy()
    len_indus = indus_factor.shape[1]
    LNCAP = np.log(cap).rename('LNCAP')
    df = pd.concat([df, indus_factor, LNCAP], axis=1, join='inner')
    df_reg = df.dropna().copy()
    Y = np.mat(df_reg.iloc[:, :-(len_indus+1)])
    X = np.mat(df_reg.iloc[:, -(len_indus+1):-1])
    weight = np.mat(np.diag(df_reg.LNCAP / df_reg.LNCAP.sum()))
    beta = (X.T * weight * X).I * X.T * weight * Y
    factor_est = pd.DataFrame(np.mat(df.iloc[:, -(len_indus+1): -1]) * beta,
                              index=df.index, columns=df.columns[:-(len_indus+1)])
    df = df.iloc[:, :-(len_indus+1)].copy()
    df[df.isnull()] = factor_est
    return df

    
def get_forward_rtn(date, stock_market_data_after, periods):
    '''
    Get forward return, cross-sectional
    Params:
        stock_market_data_after:
            the several days data after date
            pd.DataFrame  multiindex: trade_date, stock_id 
                         columns: Open, High, Low, Close, PctChg, Vol, Amount
        periods:
            list, default [1, 2, 3, 4, 5]
    Return:
        forward_rtn: 
            pd.DataFrame, index: stock_id, columns: period
    '''
    close = stock_market_data_after['Close'].unstack()
    forward_rtn = pd.DataFrame(index=close.columns)
    for period in periods:
        delta = close.pct_change(period).shift(-period)
        forward_rtn[period] = delta.loc[date]
    return forward_rtn    


def get_stk_rtn(date, stock_market_data, periods):
        '''
        Get forward return, cross-sectional
        Params:
            stock_market_data:
                pd.DataFrame  multiindex: trade_date, stock_id 
                             columns: Open, High, Low, Close, PctChg, Vol, Amount
            periods:
                list, default [1, 2, 3, 4, 5]
        Return:
            stk_rtn: 
                pd.DataFrame, index: stock_id, columns: period
        '''
        close = stock_market_data['Close'].unstack()
        stk_rtn = pd.DataFrame(index=close.columns)
        for period in periods:
            delta = close.pct_change(period)
            stk_rtn[period] = delta.loc[date]  
        return stk_rtn
        
    
def get_factor_return(date, tdays, engine, 
                      table='factor_return_single', 
                      timelong=0, 
                      period=None, 
                      factor=None):
    '''
    Get factor return that calculated by factor analysis, single or multi
    
    Params:
        date:
            string, like '%Y%m%d'
        engine:
            database engine
        table:
            'factor_return_single' or factor_return_multi
        timelong:
            int, how many days data to select, 
            default 0, means only selcct one day data\
        period:
            None or list, the predicted period, such as [1, 3]
        factor:
            None or list, factor names, such as ['alpha1', 'alpha3']
    Returns:
        pd.DataFrame
        multiindex, trade_date and period, columns: factor_names, value: factor_return
    '''
    startdate = get_tradeday_before(date, tdays, timelong)
    
    a = Table(table)
    q = Query.from_(a).select(
        a.trade_date, a.period
    ).where(
        a.trade_date[startdate: date] 
    ).orderby(
        a.trade_date, a.period
    )
    
    if period:
        q = q.where(a.period.isin(period))
        
    if factor:
        q = q.select(*[Field(f) for f in factor])
    else:
        q = q.select(a.star)
        
    data = (pd.read_sql(q.get_sql(), engine)
              .assign(trade_date = lambda df: pd.to_datetime(df['trade_date']))
              .set_index(['trade_date', 'period']))
                
    if 'id' in data.columns:
        del data['id']
            
    return data    
    

def get_ic(date, tdays, engine, timelong=0, period=None):
    '''
    Params:
        date:
            string, like '%Y%m%d'
        engine:
            database engine
        timelong:
            int, how many days data to select, 
            default 0, means only selcct one day data\
        period:
            None or list, the predicted period, such as [1, 3]

    Returns:
        pd.Series
        multiindex, trade_date and period, value: ic
    '''
    startdate = get_tradeday_before(date, tdays, timelong)
    
    a = Table('ic')
    q = Query.from_(a).select(
        a.trade_date, a.period, a.ic
    ).where(
        a.trade_date[startdate: date] 
    ).orderby(
        a.trade_date, a.period
    )
    
    if period:
        q = q.where(a.period.isin(period))

    data = (pd.read_sql(q.get_sql(), engine)
              .assign(trade_date = lambda df: pd.to_datetime(df['trade_date']))
              .set_index(['trade_date', 'period']))
    return data['ic']


def t_test(ts, type='tstas'):
    '''
    type:
        'tstas' or 'p_value'
    '''
    if type == 'tstas':
        res = stats.ttest_1samp(ts, 0)[0]
    elif type == 'p_value':
        res = stats.ttest_1samp(ts, 0)[1]
    return res
    

@timer    
def get_weight(date, tdays, engine, timelong=0):
    '''
    get stock weight
    
    Params:
        date:
            string, like '%Y%m%d'
        engine:
            database engine
        timelong:
            int, how many days data to select, 
            default 0, means only selcct one day data
    Returns:
        if timelong=0:
            pd.Series, index:stock_id, value: weight
        if timelong>0:
            pd.Series  multiindex: trade_date and stock_id, value: weight 
    '''
    startdate = get_tradeday_before(date, tdays, timelong)
    
    a = Table('weight')
    q = Query.from_(a).select(
        a.trade_date, a.stock_id, a.weight
    ).where(
        a.trade_date[startdate: date] 
    ).orderby(
        a.trade_date, a.stock_id
    )
    
    data = (pd.read_sql(q.get_sql(), engine)
              .assign(trade_date = lambda df: pd.to_datetime(df['trade_date'])))
    if timelong == 0:
        return data.set_index('stock_id')['weight']
    else:
        return data.set_index(['trade_date', 'stock_id'])['weight']


@timer    
def get_weight2(start, end, engine):
    '''
    get stock weight
    
    Params:
        start:
            string, like '%Y%m%d' 
        end:
            string, like '%Y%m%d'
        engine:
            database engine
        
    Returns:
        pd.Series  multiindex: trade_date and stock_id, value: weight 
    '''
    a = Table('weight')
    q = Query.from_(a).select(
        a.trade_date, a.stock_id, a.weight
    ).where(
        a.trade_date[start: end] 
    ).orderby(
        a.trade_date, a.stock_id
    )
    
    data = (pd.read_sql(q.get_sql(), engine)
              .assign(trade_date = lambda df: pd.to_datetime(df['trade_date'])))

    return data.set_index(['trade_date', 'stock_id'])['weight']


@timer                             
def get_index_weight(date, tdays, benchmark, engine, timelong=0):
    '''
    Get index constitution and the weight
    
    Params:
        date:
            string, like '%Y%m%d'
        engine:
            database engine
        benchmark:
            str, '000985.SH'(中证全指), '881001.WI'(万得全A，无), '000001.SH'(上证综指),
            '000300.SH'(沪深300), '000905.SH'(中证500), '000906.SH'(中证800)，'000016.SH'(上证50), 
            '399006.SZ'(创业板指),'399102.SZ'（创业板综） 等股指
        timelong:
            float, how many years data to select, 
            default 0, means only selcct one day data
    Returns:
        if timelong=0:
            pd.Series, index:stock_id, value: index weight
        if timelong>0:
            pd.Series  multiindex: trade_date and stock_id, value: index weight
    '''
    startdate = get_tradeday_before(date, tdays, timelong)
    
    a = Table('index_constitution')
    q = Query.from_(a).select(
        a.trade_date, a.stock_id, a.weight
    ).where(
        (a.trade_date[startdate: date]) & (a.index_id == benchmark) 
    ).orderby(
        a.trade_date, a.stock_id
    )
    
    data = (pd.read_sql(q.get_sql(), engine)
              .assign(trade_date = lambda df: pd.to_datetime(df['trade_date'])))
    if timelong == 0:
        return data.set_index('stock_id')['weight']
    else:
        return data.set_index(['trade_date', 'stock_id'])['weight']  


@timer                              
def get_index_weight_indus(date, tdays, benchmark, engine, timelong=0):
    '''
    Get index constitution and the weight and the corresponding citic industry
    
    Params:
        date:
            string, like '%Y%m%d'
        engine:
            database engine
        benchmark:
            str, '000985.SH'(中证全指), '881001.WI'(万得全A，无), '000001.SH'(上证综指),
            '000300.SH'(沪深300), '000905.SH'(中证500), '000906.SH'(中证800)，'000016.SH'(上证50), 
            '399006.SZ'(创业板指),'399102.SZ'（创业板综） 等股指
        timelong:
            float, how many years data to select, 
            default 0, means only selcct one day data
    Returns:
        if timelong=0:
            pd.DataFrame, index:stock_id, value: index weight and industry_citic
        if timelong>0:
            pd.DataFrame  multiindex: trade_date and stock_id, value: index weight and industry_citic
    '''
    startdate = get_tradeday_before(date, tdays, timelong)
    
    a,b = Tables('index_constitution', 'stock_industry_citic')
    q = Query.from_(a).join(
        b
    ).on(
        (a.trade_date == b.trade_date) & (a.stock_id == b.stock_id)
    ).select(
        a.trade_date, a.stock_id, a.weight, b.industry_citic
    ).where(
        (a.trade_date[startdate: date]) & (a.index_id == benchmark)
    ).orderby(
        a.trade_date, a.stock_id
    )

    data = (pd.read_sql(q.get_sql(), engine)
              .assign(trade_date = lambda df: pd.to_datetime(df['trade_date'])))
    if timelong == 0:
        return data.set_index('stock_id')[['weight', 'industry_citic']]
    else:
        return data.set_index(['trade_date', 'stock_id'])[['weight', 'industry_citic']]      


def get_benchmark(start, end, engine, benchmark):
    '''
    Params:
        start:
            string, like '%Y%m%d'
        end:
            string, like '%Y%m%d'
        engine:
            database engine
        benchmark:
            str, '000985.SH'(中证全指), '881001.WI'(万得全A，无), '000001.SH'(上证综指),
            '000300.SH'(沪深300), '000905.SH'(中证500), '000906.SH'(中证800)，'000016.SH'(上证50), 
            '399006.SZ'(创业板指),'399102.SZ'（创业板综） 等股指
    Returns:
        pd.Series  index: trade_date, value: idx_rtn
    '''
    
    a = Table('index_market_data')
    q = Query.from_(a).select(
        a.trade_date, a.pctchg
    ).where(
        (a.trade_date[start: end]) & (a.stock_id == benchmark)
    ).orderby(
        a.trade_date
    )
    
    data = (pd.read_sql(q.get_sql(), engine)
              .rename(columns = {'pctchg': 'idx_rtn'})
              .assign(idx_rtn = lambda df: df['idx_rtn'] / 100,
                      trade_date = lambda df: pd.to_datetime(df['trade_date']))
              .set_index('trade_date'))
    return data['idx_rtn']

             
def Optimization(alpha_prediction, date, tdays, engine_quant, engine_price_vol, 
                 stocklist, df_benchmark_weight_indus, style_factor, indus_factor,
                 t = 2,
                 Tc = 0.003,
                 style_max = 0.1,
                 style_min = -0.1,
                 indus_pct = 0.05):
    '''
    Optimizatin to get the optimal stock weights
    
    Calculate the weight everyday, but the transaction cost is according to period t's turnover
    
    从stocklist股票池（剔除停牌，上市没多久的股票）中筛选股票，
    并添加条件：剔除ST、涨跌停（大于9.5%或小于-9.5%）
    
    Params:
        alpha_prediction:
            pd.DataFrame, idnex: stock_id, columns: periods, 1,2,3,4,5
        date:
            string, like '%Y%m%d'
        stocklist:
            list, 1) listed before dategap from date, and has not been delisted till date 
            b.Delisting == '19000101' means still on trade
                 2) on date, Turnover > 0 
        df_benchmark_weight_indus:
            pd.DataFrame, multiindex, trade_date and stock_id,  columns: weight and industry_citic
        style_factor:
            pd.DataFrame, index: stock_id, columns: 9 style factor name
        indus_factor:
            pd.DataFrame, index: stock_id, columns: industry name
        t:
            int, Prediction Period
        Tc:
            transaction cost, double cost
        style_max:
            float, the maximum value of style factor exposure
        style_min:
            float, the minimum value of style factor exposure
        indus_pct:
            relative percent fo industry weight
    '''
    stocklist_st_limit = get_stocklist_st_limit(date, engine_quant)
    stk_pool = sorted(list(set(stocklist) - set(stocklist_st_limit)))
    stk_pool = sorted(list(set(alpha_prediction.index.tolist()) & set(stk_pool)))
       
    w = Variable(len(stk_pool))
    date_last = get_tradeday_before(date, tdays, timelong=t)
    w_last = get_weight(date_last, tdays, engine_price_vol)
    if len(w_last) == 0:
        w_last = pd.Series(0, index=stk_pool)
    w_last = w_last.reindex(stk_pool).fillna(0)
        
    ret = np.mat(alpha_prediction[t].reindex(stk_pool)) * w
    cost = Tc * sum_entries(abs(w - w_last.values)) / 2
    
    objective = Maximize(ret - cost)
    
    w_benchmark = df_benchmark_weight_indus.loc[df_benchmark_weight_indus.index.levels[0].asof(date)]['weight']
    w_benchmark = w_benchmark.reindex(stk_pool).fillna(0) #这种做法有待商榷
    w_benchmark = w_benchmark / w_benchmark.sum()         
    X_style = style_factor.reindex(stk_pool)
    style_constraint = (w - w_benchmark.values).T * np.mat(X_style)
    
    w_benchmark_industry = df_benchmark_weight_indus.loc[df_benchmark_weight_indus.index.levels[0].asof(date)]
    w_benchmark_industry = w_benchmark_industry.groupby('industry_citic')['weight'].sum()   
    w_benchmark_industry = w_benchmark_industry.reindex(indus_factor.columns).fillna(0)      
    w_benchmark_industry = w_benchmark_industry / w_benchmark_industry.sum()                                                 
    X_indus = indus_factor.reindex(stk_pool)
    indus_constraint = w.T * np.mat(X_indus)
    
    constraints = [sum_entries(w) == 1,
                   w >= 0,
                   style_constraint <= np.mat([style_max for i in range(X_style.shape[1])]),
                   style_constraint >= np.mat([style_min for i in range(X_style.shape[1])]),
                   indus_constraint <= np.mat(w_benchmark_industry * (1 + indus_pct)),
                   indus_constraint >= np.mat(w_benchmark_industry * (1 - indus_pct))]
    prob = Problem(objective, constraints)
    prob.solve()
    weight = pd.Series(w.value.A1, index=stk_pool, name='weight')
    return weight
                         