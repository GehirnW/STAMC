# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 20:22:40 2018

@author: chenxbo
"""
import pandas as pd
import numpy as np
from logbook import FileHandler, Logger
from sqlalchemy import create_engine
from statsmodels.regression.linear_model import OLS
from imp import reload

import mysql_table
import get_alphas
import utils
reload(mysql_table)
reload(get_alphas)
reload(utils)

log_file_name = r'price_vol.log'
log_handler = FileHandler(log_file_name).push_application()

def check_log(date, tdays):
    '''
    if the date next day successfully wrote to database, then ignore the date
    because there four py file need to be executed
    
    Params:
        date:
            str, like '%Y%m%d'
        tdays:
            trade day series
    '''
    with open(log_file_name) as f:
        log_text = f.read()
    if tdays[tdays>date].iloc[0] + ': 1' in log_text:
        return True
    else:
        return False


@utils.timer
def main(date, tdays, engine_price_vol, engine_quant, engine_lhtz, df_benchmark_weight_indus,
         periods= [1, 2, 3, 4, 5]):
    '''
    Params:
        date:
            str, like '%Y%m%d'
        periods:
            list, the prediciton of factor return
    '''
    print("***************** %s *****************" % date)
    
    ##################################################################
    # Basic Data
    ####################################################################
    stocklist = utils.get_stocklist(date, engine_quant, 90)
    cap = utils.get_cap(date, engine_quant, stocklist)
    indus_factor = pd.get_dummies(utils.get_indus(date, engine_quant, stocklist))
    style_factor = utils.get_style_factor(date, engine_lhtz, stocklist, cap)
    if 'Non_linear_Size' in style_factor.columns:
        del style_factor['Non_linear_Size']
    len_indus_style = indus_factor.shape[1] + style_factor.shape[1]

    date_after = utils.get_tradeday_after(date, tdays, timelong=max(periods))
    stock_market_data =  utils.get_stock_market_data(date, tdays, engine_quant, stocklist, timelong=22) 
    
    #####################################################################
    # Calculate and Save alpha factors
    #####################################################################
    df_alpha = get_alphas.main(date, tdays, stock_market_data, indus_factor, cap)
    df_alpha.index.name = 'stock_id'
    mysql_table.update_factor_exposure_original(date, df_alpha, engine_price_vol)
    
    #############################################################################
    # Single Factor Analysis to judge which factors can be used to be alpha factors
    ###############################################################################
    # step1  winsorize、standardize and orthogonalization and Save alpha_resid
    df_alpha_standard = df_alpha.apply(lambda ts: utils.standardize( utils.winsorize(ts) ))
    df = pd.concat([df_alpha_standard, indus_factor, style_factor], axis=1, join='inner')
    est = OLS(df.iloc[:, :-len_indus_style], df.iloc[:, -len_indus_style:]).fit()
    alpha_resid = est.resid
    mysql_table.update_factor_exposure_resid(date, alpha_resid, engine_price_vol)
    
    # step2 Calculate factor return, use stk_rtn in date t and factor exposure in date t-d
    
    stock_market_data_after = utils.get_stock_market_data(
                            date_after, tdays, engine_quant, stocklist, timelong=max(periods)) 
    forward_rtn = utils.get_forward_rtn(date, stock_market_data_after, periods)
    
    alpha_rtn = pd.DataFrame()
    for name in alpha_resid.columns:
        df2 = pd.concat([forward_rtn, df.iloc[:, -len_indus_style:], alpha_resid[name]], axis=1, join='inner')
        est = OLS(df2.iloc[:, :len(periods)], df2.iloc[:, len(periods):]).fit()
        temp = est.params.iloc[-1]
        alpha_rtn = pd.concat([alpha_rtn, temp], axis=1)
    alpha_rtn.index = periods
    alpha_rtn.index.name = 'period'
    mysql_table.update_factor_return_single(date, alpha_rtn, engine_price_vol)    
    del df2
    

    # step3 Annual return and Information Ratio
    factor_return_single = utils.get_factor_return(date, tdays, engine_price_vol, 
                                                   table='factor_return_single',
                                                   timelong=500)        
    
    def Annual_rtn_IR(date, data, min_period=252):
        '''
        Calculate the annual return and Information Ratio,
        Save them to mysql
        
        Params:
            data:
                single factor return, pd.DataFrame, 
                mulitiindex, trade_date and period  columns: factor names
            min_period:
                the minminum period when calculating
        '''
        data = data.copy()
        
        if len(data.index.get_level_values(level='trade_date').unique()) < min_period:
            return
            
        data = data.div(data.index.get_level_values(level='period'), axis='index')
        annual_rtn = 252 * data.groupby(level='period').mean()
        IR = np.sqrt(252) * data.groupby(level='period').mean() / data.groupby(level='period').std()
        
        try:  
            mysql_table.update_annual_return_single(date, annual_rtn, engine_price_vol)
        except:
            raise ValueError('Fail to update_annual_return_single:' + date)
    
        try:
            mysql_table.update_ir_single(date, IR, engine_price_vol)
        except:
            raise ValueError('Fail to update_ir_single:' + date)
        
    Annual_rtn_IR(date, factor_return_single)
    
    
    ###############################################################
    # Multiple Factor Analysis（将所有alpha放在一起分析）
    ###############################################################
    df3 = pd.concat([forward_rtn, df.iloc[:, -len_indus_style:], alpha_resid], axis=1, join='inner')
    est = OLS(df3.iloc[:, :len(periods)], df3.iloc[:, len(periods):]).fit()
    alpha_rtn_multi = est.params.iloc[len_indus_style:].T
    alpha_rtn_multi.index = periods
    alpha_rtn_multi.index.name = 'period'
    mysql_table.update_factor_return_multi(date, alpha_rtn_multi, engine_price_vol)
    
    # alpha return prediction vector, use the mean of last one year factor return to be the prediction
    factor_return_multi = utils.get_factor_return(date, tdays, engine_price_vol, 
                                                  table='factor_return_multi',
                                                  timelong=500)   
    
    def IC_TStas(date, data, alpha_resid, min_period=260):
        '''
        Calculate the IC and T stast of IC, then Save them to mysql
        Params:
            alpha_resid:
                alpha resid in date t,
                pd.DataFrame, index:code, columns:factors anme
        '''
        data = data.copy()
        
        if len(data.index.get_level_values(level='trade_date').unique()) < min_period:
            return
        
        # 1、预测的阿尔法收益(处理后没有用到未来信息)
        alpha_ = pd.DataFrame(index=data.columns)
        for period in periods:
            date_before = utils.get_tradeday_before(date, tdays, timelong=period)
            alpha_[period] = data.loc[:date_before].groupby(level='period').mean().loc[period]            
        alpha_prediction = alpha_resid.dot(alpha_)
            
        # 2、真实的阿尔法收益（与预测的阿尔法收益应该相差period天，而不只是相差一天）            
        alpha_real = alpha_resid.dot(data.loc[date].reset_index(level='trade_date', drop=True).T)
        
        ic = alpha_prediction.corrwith(alpha_real)
        ic.name = 'ic'
        mysql_table.update_ic(date, ic, engine_price_vol)
        
        def TStas_pvalue(date, timelong=250, min_period=125):
            ic_s = utils.get_ic(date, tdays, engine_price_vol, timelong=timelong)
            if len(ic_s.index.get_level_values(level='trade_date').unique()) < min_period:
                return
            
            TStas = ic_s.groupby(level='period').apply(utils.t_test, 'tstas')
            TStas.name = 'tstas'
            p_value = ic_s.groupby(level='period').apply(utils.t_test, 'p_value')
            p_value.name = 'p_value'
            temp = pd.concat([TStas, p_value], axis=1)
            mysql_table.update_tstas(date, temp, engine_price_vol)
            return
            
        TStas_pvalue(date)
        return alpha_prediction
        
    alpha_prediction = IC_TStas(date, factor_return_multi, alpha_resid)
    
    ################################################################
    # Optimizaiton
    ###############################################################
    if alpha_prediction is not None:
        
        weight = utils.Optimization(alpha_prediction, date, tdays, engine_quant, engine_price_vol, 
                                    stocklist, df_benchmark_weight_indus, style_factor, indus_factor)
        weight.index.name = 'stock_id'
        mysql_table.update_weight(date, weight, engine_price_vol)
        
    return 
    
   

if __name__ == "__main__":
    engine_price_vol = create_engine(r'mysql+pyodbc://price_vol')
    engine_quant = create_engine(r'mysql+pyodbc://quant')
    engine_lhtz = create_engine(r'mysql+pyodbc://jh_barra')
    
    mysql_table.create_table(engine_price_vol)
    
    tdays = utils.get_tdays(engine_quant)
    
    benchmark = '000905.SH'
    df_benchmark_weight_indus = utils.get_index_weight_indus(tdays.iloc[-1], tdays, benchmark, 
                                                             engine_quant, timelong=15*250)

    '''                                            
    date_s = ['20170405', '20170406', '20170407']
    for date in date_s:
        main(date, tdays, engine_price_vol, engine_quant, engine_lhtz, df_benchmark_weight_indus)
    ''' 
       
    for tday in tdays[tdays > '20081231']:
        if not check_log(tday, tdays):
            try:
                main(tday, tdays, engine_price_vol, engine_quant, engine_lhtz, df_benchmark_weight_indus)
                Logger('price_vol').info(tday + ': 1')
            except ValueError as e:
                Logger('price_vol').info(tday + ': ' + str(e))
            except:
                Logger('price_vol').info(tday + ': Unknow Error') 


    

    '''
    后续：
    1）将函数的参数完善
    2）加入日志，参考barra的文档设置。 程序格式要不要设计成barra的那样，分模块运行？
    3)将那么多因子，看如何统一起来？存量管理，就不用每个地方都输入191个因子，还包括创建表的地方
    '''
    