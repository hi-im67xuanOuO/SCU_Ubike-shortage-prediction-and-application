# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 20:51:19 2020

@author: s9631
"""


import pandas as pd
import os
os.chdir("D:/project/專題/資料庫備份")

def CleanData_AddTime(file):
    data = pd.read_csv(file+'.csv')
    data = data[data['act'] == 1]
    print(len(data))
    data.drop_duplicates(keep="first", inplace=True)
    print(len(data))
    # 刪除英文欄位
    # data = data[['sno','sna','tot','sbi','sarea','mday','lat','lng','ar','bemp','act']]
    # # 轉時間格式
    # data.reset_index(drop = True, inplace = True)
    # data.astype({'mday': 'object'})#轉字串
    # data['mday'] = pd.to_datetime(data['mday'], format="%Y%m%d%H%M%S")
    # # 星期幾
    # # 0~6，0週一6週日
    # data['weekday'] = data['mday'].dt.dayofweek
    # data = AddHour(data)
    # data = AddMinute(data)
    # print(data.tail()['mday'])
    # return data

def AddHour(data):
    x = []
    for i in range(len(data)):
        x.append(data['sna'][i]+str(data['mday'][i]).split(':')[0])
    data['hour'] = x
    return data

def AddMinute(data):
    x = []
    for i in range(len(data)):
        x.append(data['sna'][i]+str(data['mday'][i]).split(':')[0]+str(data['mday'][i]).split(':')[1])
    # data_8_m['min'] = x
    data['min'] = x
    return data

data = CleanData_AddTime('bike_8_ok')
data.to_csv('bike_8_all.csv', index = False)

data = CleanData_AddTime('bike_9_ok')
data.to_csv('bike_9_all.csv', index = False)
