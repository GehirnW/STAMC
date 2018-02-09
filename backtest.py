# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 09:26:55 2018

@author: admin
"""

import pandas as pd
import numpy as np
import utils
from sqlalchemy import create_engine


def backtest(start, end, engine_quant, engine_price_vol, benchmark_name='000905.SH'):
    '''
    Get backtest indictors according to the stock weights
    
    Params:
        start:
            str, like '%Y%m%d'
        end:
            str, like '%Y%m%d'
        engine_quant:
            database engine
        engine_price_vol:
            database engine
        benchmark: 
            benchmark name, default '000905.SH' 中证500
    Return:
        backtest indictors, 
    '''
    initial_capital = 10000
    price = utils.get_close(start, end, engine_quant)
    benchmark_rtn = utils.get_benchmark(start, end, engine_quant, benchmark_name)
    stk_weight = utils.get_weight2(start, end, engine_price_vol)
    
    portfolio_rtn = 
    
    stat = get_stat(portfolio_rtn, benchmark_rtn)
    return stat


def get_turnover(stk_weight,lags=1):
    '''
    根据权重获取资金序列
    ## 只适用于等权
    stk_weight: 
        pd.Series Multi-Index date, asset    weight 换仓期的权重
    lags: 
        表示计算当前股票相对于lags期前股票的差异
    返回换手率时间序列:换手率 = 当期相对于lags期前新加入的股票 / 当期股票总数
    '''
    sets = stk_weight.groupby(level=['date']).apply(lambda x: set(x.index.get_level_values('code')))
    news = (sets - sets.shift(lags)).dropna()
    turnover = news.apply(len) / sets.apply(len)
    turnover.fillna(0.5,inplace=True)
    return turnover    
    
def get_rtn_s(stk_weight, price, td, fee=0.003):
    '''
    根据持仓权重，获取投资组合资金序列
    先获得各日期的持仓数量（换仓期为(资金-换仓成本)*股票权重/股票价格），非换仓期为上一天的
    考虑换手率手续费
    判断持仓的价格是否存在nan值啥的， 不然影响很大
    params:
        stk_weight:
            pd.Series, multiindex: trade_date and stock_id,  value:weight
        price: 
            dataframe, index: date  colulmns: code
        td: 
            开始回测日期到结束日期的交易日序列
        fee: 
            float, 双边
    return:
        rtn_s: 组合收益率序列，Series, index: date  
    '''
    initial_capital=10000              

    price = price.loc[td[0]:td[-1]].stack().copy()
    price.index.names = ['date','code']
    turnover_s = get_turnover(stk_weight)
    
    holding = {}
    holding_pro = initial_capital * stk_weight.loc[td[0]] / price.loc[td[0]].reindex(stk_weight.loc[td[0]].index)
    holding[td[0]] = holding_pro
    for dt in td[1:]:
        if dt in stk_weight.index:
            capital = (price.loc[dt].reindex(holding_pro.index) * holding_pro).sum()
            holding_pro = (capital - capital * turnover_s.loc[dt] * fee) * \
                           stk_weight.loc[dt] / price.loc[dt].reindex(stk_weight.loc[dt].index)
        holding[dt] = holding_pro
    holding = pd.concat(holding,names=['holding'])        
    holding.index.names = ['date','code']
    if price.reindex(holding.index).isnull().values.any():
        # 比如2016-08-12号买到 300372，不知道为啥数据就没有价格了（退市，但之后其实还是有价格的），所以默认价格数据等于前一天价格
        # print ('############  price has error #############')
        price = price.reindex(holding.index).groupby(level='code').fillna(method='ffill')
    else:
        price = price.reindex(holding.index)
        
    rtn_s = (holding * price).groupby(level='date').sum().pct_change().fillna(0)
    return rtn_s    
    

def get_stat(rtn, benchmark_rtn, rf=0):
    '''
    获取回测指标，rtn:可以为单列(Series)，也可以为多列（DataFrame）
    params: 
        rtn: 
            pd.Series or pd.DataFrame, index: trade_date, columns: 方法名称， values: return
        benchmark: 
            benchmark return, pd.Series, index: date, values: return
        rf: 
            risk_free rate
        ps: 1）rtn和benchmark的index需完全一致； 2）考虑了费率就不用考虑换手率
    '''
    stat = {}
    max_drawdown = {}
    max_dd_date = {}
    cumulative_return = {}
    annual_return = {}
    annual_std = {}
    sharpe = {}
    hit_ratio = {}
    annual_alpha = {}
    annual_std_excess = {}
    IR = {}
    max_drawdown_excess = {}
    max_dd_date_excess = {}

    rtn = pd.DataFrame(rtn)
    for col in rtn.columns:
        ts = rtn[col]
        try:
            nv = (ts + 1).cumprod() #净值
            dd = nv / (nv.cummax()) - 1 #回撤
            max_drawdown[col] = dd.min() #最大回撤
            max_dd_date[col] = dd.idxmin().strftime('%Y-%m-%d') #最大回撤日期
            cumulative_return[col] = nv[-1] - 1 #累计收益
        except:
            raise ValueError('inf in cumulative_return') 
            
        annual_return[col] = nv[-1]**(250.0 / len(ts)) - 1 #年化收益
        annual_std[col] = ts.std() * np.sqrt(250.0)  #年化波动率
        try:
            sharpe[col] = (annual_return[col] - rf) / annual_std[col] #夏普比率
        except:
            sharpe[col] = 0
        
        if benchmark_rtn is not None:
            ts_excess = ts - benchmark_rtn
            hit_ratio[col] = (ts_excess>0).sum() / len(ts_excess)
            
            nv_excess = (ts_excess + 1).cumprod()
            dd_excess = nv_excess / (nv_excess.cummax()) - 1 #回撤
            max_drawdown_excess[col] = dd_excess.min() #最大回撤
            max_dd_date_excess[col] = dd_excess.idxmin().strftime('%Y-%m-%d') #最大回撤日期
            
            annual_alpha[col] = nv_excess[-1]**(250.0 / len(ts_excess)) - 1 #年化超额收益
            annual_std_excess[col] = ts_excess.std() * np.sqrt(250)  #年化超额波动率
            IR[col] = annual_alpha[col] / annual_std_excess[col]  # IR  
        else:
            hit_ratio[col], max_drawdown_excess[col], max_dd_date_excess[col], \
            annual_alpha[col], IR[col] = 0, 0, 0, 0, 0
            
    stat['max_drawdown'] = max_drawdown
    stat['max_dd_date'] = max_dd_date
    stat['cumulative_return'] = cumulative_return
    stat['annual_return'] = annual_return
    stat['annual_std'] = annual_std
    stat['sharpe'] = sharpe
    stat['hit_ratio'] = hit_ratio
    stat['annual_alpha'] = annual_alpha
    stat['annual_std_excess'] = annual_std_excess
    stat['IR'] = IR
    stat['max_drawdown_excess'] = max_drawdown_excess
    stat['max_dd_date_excess'] = max_dd_date_excess
    
    stat = pd.DataFrame(stat).loc[:,['annual_return', 'annual_std', 'cumulative_return', 
                                     'sharpe', 'max_drawdown', 'max_dd_date', 'annual_alpha', 'annual_std_excess',
                                     'hit_ratio', 'IR', 'max_drawdown_excess', 'max_dd_date_excess']].T
    stat.index.name = 'stat'
    return stat    

if __name__ == "__main__":
    engine_quant = create_engine(r'mysql+pyodbc://quant')
    engine_price_vol = create_engine(r'mysql+pyodbc://price_vol')
    
    start = '20100101'
    end = '20101231'
    
    stat = backtest(start, end, engine_quant, engine_price_vol)
    

    
    

