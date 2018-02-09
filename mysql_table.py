# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 17:07:55 2018

@author: chenxbo
"""
from imp import reload
import utils
reload(utils)


def create_table(engine):
    sql = '''
             create table if not exists factor_exposure_original
             (
                id           int             not null   primary key auto_increment,
                trade_date   varchar(8)      not null,
                stock_id     varchar(10)     not null,
                alpha2       float(20,5)     not null,
                alpha14       float(20,5)     not null,
                alpha15       float(20,5)     not null,
                alpha18       float(20,5)     not null,
                alpha20       float(20,5)     not null,
                alpha31       float(20,5)     not null,
                alpha34       float(20,5)     not null,
                alpha53       float(20,5)     not null,
                alpha58       float(20,5)     not null,
                alpha65       float(20,5)     not null,
                unique key index_1(trade_date, stock_id) 
             );
          '''
    engine.execute(sql)
    
    sql = '''
             create table if not exists factor_exposure_resid
             (
                id           int             not null   primary key auto_increment,
                trade_date   varchar(8)      not null,
                stock_id     varchar(10)     not null,
                alpha2       float(20,5)     not null,
                alpha14       float(20,5)     not null,
                alpha15       float(20,5)     not null,
                alpha18       float(20,5)     not null,
                alpha20       float(20,5)     not null,
                alpha31       float(20,5)     not null,
                alpha34       float(20,5)     not null,
                alpha53       float(20,5)     not null,
                alpha58       float(20,5)     not null,
                alpha65       float(20,5)     not null,
                unique key index_1(trade_date, stock_id) 
             );
          '''
    engine.execute(sql)
    
    
    sql = '''
             create table if not exists factor_return_single
             (
                id           int             not null   primary key auto_increment,
                trade_date   varchar(8)      not null,
                period       int             not null,
                alpha2       float(20,5)     not null,
                alpha14       float(20,5)     not null,
                alpha15       float(20,5)     not null,
                alpha18       float(20,5)     not null,
                alpha20       float(20,5)     not null,
                alpha31       float(20,5)     not null,
                alpha34       float(20,5)     not null,
                alpha53       float(20,5)     not null,
                alpha58       float(20,5)     not null,
                alpha65       float(20,5)     not null,
                unique key index_1(trade_date, period) 
             );
          '''
    engine.execute(sql)
    
    
    sql = '''
             create table if not exists annual_return_single
             (
                id           int             not null   primary key auto_increment,
                trade_date   varchar(8)      not null,
                period       int             not null,
                alpha2       float(20,5)     not null,
                alpha14       float(20,5)     not null,
                alpha15       float(20,5)     not null,
                alpha18       float(20,5)     not null,
                alpha20       float(20,5)     not null,
                alpha31       float(20,5)     not null,
                alpha34       float(20,5)     not null,
                alpha53       float(20,5)     not null,
                alpha58       float(20,5)     not null,
                alpha65       float(20,5)     not null,
                unique key index_1(trade_date, period) 
             );
          '''
    engine.execute(sql)
    

    sql = '''
             create table if not exists ir_single
             (
                id           int             not null   primary key auto_increment,
                trade_date   varchar(8)      not null,
                period       int             not null,
                alpha2       float(20,5)     not null,
                alpha14       float(20,5)     not null,
                alpha15       float(20,5)     not null,
                alpha18       float(20,5)     not null,
                alpha20       float(20,5)     not null,
                alpha31       float(20,5)     not null,
                alpha34       float(20,5)     not null,
                alpha53       float(20,5)     not null,
                alpha58       float(20,5)     not null,
                alpha65       float(20,5)     not null,
                unique key index_1(trade_date, period) 
             );
          '''
    engine.execute(sql)

    
    """
    sql = '''
             create table if not exists factor_return_multi
             (
                id           int             not null   primary key auto_increment,
                trade_date   varchar(8)      not null,
                period       int             not null,
                alpha2       float(20,5)      ,
                alpha14       float(20,5)     ,
                alpha15       float(20,5)     ,
                alpha18       float(20,5)     ,
                alpha20       float(20,5)     ,
                alpha31       float(20,5)     ,
                alpha34       float(20,5)     ,
                alpha53       float(20,5)     ,
                alpha58       float(20,5)     ,
                alpha65       float(20,5)     ,
                unique key index_1(trade_date, period) 
             );
          '''
    engine.execute(sql)
    """
     
    
    sql = '''
             create table if not exists factor_return_multi
             (
                id           int             not null   primary key auto_increment,
                trade_date   varchar(8)      not null,
                period       int             not null,
                alpha2       float(20,5)     not null,
                alpha14       float(20,5)     not null,
                alpha15       float(20,5)     not null,
                alpha18       float(20,5)     not null,
                alpha20       float(20,5)     not null,
                alpha31       float(20,5)     not null,
                alpha34       float(20,5)     not null,
                alpha53       float(20,5)     not null,
                alpha58       float(20,5)     not null,
                alpha65       float(20,5)     not null,
                unique key index_1(trade_date, period) 
             );
          '''
    engine.execute(sql)
    
    
    sql = '''
             create table if not exists ic
             (
                id           int             not null   primary key auto_increment,
                trade_date   varchar(8)      not null,
                period       int             not null,
                ic          float(20,5)     not null,
                unique key index_1(trade_date, period) 
             );
          '''
    engine.execute(sql)
    
    
    sql = '''
             create table if not exists tstas
             (
                id           int             not null   primary key auto_increment,
                trade_date   varchar(8)      not null,
                period       int             not null,
                tstas        float(20,5)     not null,
                p_value      float(20,5)     not null,
                unique key index_1(trade_date, period) 
             );
          '''
    engine.execute(sql)
    
    
    sql = '''
             create table if not exists weight
             (
                id           int             not null   primary key auto_increment,
                trade_date   varchar(8)      not null,
                stock_id     varchar(10)     not null,
                weight       float(20,5)     not null,
                unique key index_1(trade_date, stock_id) 
             );
          '''
    engine.execute(sql)
    
    
@utils.timer    
def update_factor_exposure_original(date, data, engine):
    '''
    Params:
        date: 
            string, like '%Y%m%d'
        data:
            pd.DataFrame with index: stock_id, columns: multiple alpha names
        engine:
            target database engine to save data
    '''
    try:
        data = (data.round(5)
                    .copy()
                    .reset_index()
                    .assign(trade_date=date))
        try:
            data.to_sql(name='factor_exposure_original', con=engine, if_exists='append', index=False, chunksize=1000)
            print('Successfully fininshed updating factor_exposure_original:', date)
            return
        except:
            print(date, 'factor_exposure_original already exists')
            return
    except:
        raise ValueError('Fail to update factor_exposure_original : ', date)
        
        
@utils.timer    
def update_factor_exposure_resid(date, data, engine):
    '''
    Params:
        date: 
            string, like '%Y%m%d'
        data:
            pd.DataFrame with index: stock_id, columns: multiple alpha names
        engine:
            target database engine to save data
    '''
    try:
        data = (data.round(5)
                    .copy()
                    .reset_index()
                    .assign(trade_date=date))
        try:
            data.to_sql(name='factor_exposure_resid', con=engine, if_exists='append', index=False, chunksize=1000)
            print('Successfully fininshed updating factor_exposure_resid:', date)
            return
        except:
            print(date, 'factor_exposure_resid already exists')
            return
    except:
        raise ValueError('Fail to update factor_exposure_resid : ', date)   
        

@utils.timer    
def update_factor_return_single(date, data, engine):
    '''
    Params:
        date: 
            string, like '%Y%m%d'
        data:
            pd.DataFrame with index: period, columns: multiple alpha names
        engine:
            target database engine to save data
    '''
    try:
        data = (data.round(5)
                    .copy()
                    .reset_index()
                    .assign(trade_date=date))
        try:
            data.to_sql(name='factor_return_single', con=engine, if_exists='append', index=False, chunksize=1000)
            print('Successfully fininshed updating factor_return_single:', date)
            return
        except:
            print(date, 'factor_return_single already exists')
            return
    except:
        raise ValueError('Fail to update factor_return_single : ', date)     
    

@utils.timer    
def update_annual_return_single(date, data, engine):
    '''
    Params:
        date: 
            string, like '%Y%m%d'
        data:
            pd.DataFrame with index: period, columns: multiple alpha names
        engine:
            target database engine to save data
    '''
    try:
        data = (data.round(5)
                    .copy()
                    .reset_index()
                    .assign(trade_date=date))
        try:
            data.to_sql(name='annual_return_single', con=engine, if_exists='append', index=False, chunksize=1000)
            print('Successfully fininshed updating annual_return_single:', date)
            return
        except:
            print(date, 'annual_return_single already exists')
            return
    except:
        raise ValueError('Fail to update annual_return_single : ', date)   


@utils.timer    
def update_ir_single(date, data, engine):
    '''
    Params:
        date: 
            string, like '%Y%m%d'
        data:
            pd.DataFrame with index: period, columns: multiple alpha names
        engine:
            target database engine to save data
    '''
    try:
        data = (data.round(5)
                    .copy()
                    .reset_index()
                    .assign(trade_date=date))
        try:
            data.to_sql(name='ir_single', con=engine, if_exists='append', index=False, chunksize=1000)
            print('Successfully fininshed updating ir_single:', date)
            return
        except:
            print(date, 'ir_single already exists')
            return
    except:
        raise ValueError('Fail to update ir_single : ', date)           
        
        
       
@utils.timer    
def update_factor_return_multi(date, data, engine):
    '''
    Params:
        date: 
            string, like '%Y%m%d'
        data:
            pd.DataFrame with index: period, columns: multiple alpha names
        engine:
            target database engine to save data
    '''
    try:
        data = (data.round(5)
                    .copy()
                    .reset_index()
                    .assign(trade_date=date))
        try:
            data.to_sql(name='factor_return_multi', con=engine, if_exists='append', index=False, chunksize=1000)
            print('Successfully fininshed updating factor_return_multi:', date)
            return
        except:
            print(date, 'factor_return_multi already exists')
            return
    except:
        raise ValueError('Fail to update factor_return_multi : ', date) 
        

@utils.timer    
def update_ic(date, data, engine):
    '''
    Params:
        date: 
            string, like '%Y%m%d'
        data:
            pd.Series with index: period, value: ic
        engine:
            target database engine to save data
    '''
    try:
        data = (data.round(5)
                    .copy()
                    .reset_index()
                    .assign(trade_date=date))
        try:
            data.to_sql(name='ic', con=engine, if_exists='append', index=False, chunksize=1000)
            print('Successfully fininshed updating ic:', date)
            return
        except:
            print(date, 'ic already exists')
            return
    except:
        raise ValueError('Fail to update ic : ', date) 
        
        
@utils.timer    
def update_tstas(date, data, engine):
    '''
    Params:
        date: 
            string, like '%Y%m%d'
        data:
            pd.DataFrame with index: period, columns: tstas, p_value
        engine:
            target database engine to save data
    '''
    try:
        data = (data.round(5)
                    .copy()
                    .reset_index()
                    .assign(trade_date=date))
        try:
            data.to_sql(name='tstas', con=engine, if_exists='append', index=False, chunksize=1000)
            print('Successfully fininshed updating tstas:', date)
            return
        except:
            print(date, 'tstas already exists')
            return
    except:
        raise ValueError('Fail to update tstas : ', date) 
        

@utils.timer    
def update_weight(date, data, engine):
    '''
    Params:
        date: 
            string, like '%Y%m%d'
        data:
            pd.Series with index: stock_id, value: weight
        engine:
            target database engine to save data
    '''
    try:
        data = (data.round(5)
                    .copy()
                    .reset_index()
                    .assign(trade_date=date))
        try:
            data.to_sql(name='weight', con=engine, if_exists='append', index=False, chunksize=1000)
            print('Successfully fininshed updating weight:', date)
            return
        except:
            print(date, 'weight already exists')
            return
    except:
        raise ValueError('Fail to update weight : ', date) 
