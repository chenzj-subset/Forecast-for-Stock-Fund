import os
import pandas as pd
import tushare as ts

ts.set_token('cd386d3761611059bea60abb186636554725860c63da71637b466d44')
# my token here
# you can get your own token from https://tushare.pro/register?reg=387255
pro = ts.pro_api()
# initialize the api
listall = pro.stock_basic(exchange='', list_status='L',
                          fields='ts_code,symbol,name,area,industry,list_date')
# stock list
# print(listall.columns)
# Index(['ts_code', 'symbol', 'name', 'area', 'industry', 'list_date'], dtype='object')
ts_code_list = list(set(listall.ts_code[0:99]))
for ts_code in ts_code_list:
    k = pro.daily(ts_code='000001.SZ',
                  start_date='20200101', end_date='20200817')
    print(k)
    # K lines
    # lstm to predict
    os.system("lstm.py")
    break
