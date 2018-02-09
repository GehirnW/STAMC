# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 09:26:55 2018

@author: admin
"""

import pandas as pd
import numpy as np
import utils
from sqlalchemy import create_engine


def backtest(start, end, engine_quant, engine_price_vol, 
             benchmark_name='000905.SH',
             t = 2,
             Tc = 0.003
             ):
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
        t:
            int, Prediction Period, default 2, can't be changed beacause the weights are calculated 
                 when the period is 2
        Tc:
            transaction cost, double cost
    Return:
        backtest indictors, 
    '''
    initial_capital = 10000
    price = utils.get_close(start, end, engine_quant)
    benchmark_rtn = utils.get_benchmark(start, end, engine_quant, benchmark_name)
    stk_weight = utils.get_weight2(start, end, engine_price_vol)
    stk_weight = stk_weight.groupby(level='trade_date').apply(truncate_weights)
    stk_weight = stk_weight[stk_weight != 0]

#   price 和 stk_weight的index要完全匹配
    
    # Real Trade Date in the period of backtesting
    td = sorted(list(
                      set(price.index.levels[0].tolist())
                    & set(benchmark_rtn.index.tolist())
                    & set(stk_weight.index.levels[0].tolist())
                    ))
    turnover_td = td[0: : t] # the date series of turnover 
    
    holding = {}
    weight_pro = stk_weight.loc[td[0]]
    temp = initial_capital * weight_pro / price.loc[td[0]].reindex(weight_pro.index)
    holding[td[0]] = temp

    for dt in td[1:]:
        if dt in turnover_td:
            capital = (price.loc[dt].reindex(temp.index) * temp).sum()
            # cost 需要修改，先用并集合
            cost = Tc * np.abs(stk_weight.loc[dt] 
                             - weight_pro.reindex(stk_weight.loc[dt].index).fillna(0)).sum() / 2
            weight_pro = stk_weight.loc[dt]
            temp = (capital - capital * cost) * weight_pro / price.loc[dt].reindex(weight_pro.index)
        holding[dt] = temp
    holding = pd.concat(holding,names=['trade_date', 'stock_id'])        
    
    '''
    if price.reindex(holding.index).isnull().values.any():
        # print ('############  price has error #############')
        price = price.reindex(holding.index).groupby(level='code').fillna(method='ffill')
    else:
        price = price.reindex(holding.index)
    '''
    portfolio_rtn = (holding * price).groupby(level='date').sum().pct_change().fillna(0)
    
    stat = get_stat(portfolio_rtn, benchmark_rtn)
    return stat


def truncate_weights(weights, min_weight=0.001, rescale=True):
    """
    Truncates small weight vectors, i.e. sets weights below a treshold to zero.
    This can be helpful to remove portfolio weights, which are negligibly small.
    
    Parameters
    ----------
    weights: pandas.Series
        Optimal asset weights.
    min_weight: float, optional
        All weights, for which the absolute value is smaller
        than this parameter will be set to zero.
    rescale: boolean, optional
        If 'True', rescale weights so that weights.sum() == 1.
        If 'False', do not rescale.

    Returns
    -------
    adj_weights: pandas.Series
        Adjusted weights.
    """
    if not isinstance(weights, pd.Series):
        raise ValueError("Weight vector is not a Series")

    adj_weights = weights[:]
    adj_weights[adj_weights.abs() < min_weight] = 0.0

    if rescale:
        if not adj_weights.sum():
            raise ValueError("Cannot rescale weight vector as sum is not finite")
        
        adj_weights /= adj_weights.sum()

    return adj_weights        

    
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
    

    
    

