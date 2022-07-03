# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 09:50:18 2020

@author: s9631
"""


import pandas as pd
import os 
os.chdir("D:/project/專題/資料庫備份")
data = pd.read_csv("bike_ok.csv")
# import datetime
# 抓出八月資料
x = []
for i in data['mday']:
    x.append(str(i)[0:6])
data['ym'] = x
data_8 = data[data['ym'] == '202008']
del data
# 去除重複資料
data_8.drop_duplicates(keep="first", inplace=True)
# data_8.to_csv('bike_done_8.csv', index = False)

# 刪除英文欄位
data_8 = data_8[['sno','sna','tot','sbi','sarea','mday','lat','lng','ar','bemp','act']]
# 轉時間格式
data_8.reset_index(drop = True, inplace = True)
data_8.astype({'mday': 'object'})#轉字串
# data_8.dtypes#確認格式
data_8['mday'] = pd.to_datetime(data_8['mday'], format="%Y%m%d%H%M%S")
# data_8.dtypes
# data_8.to_csv('bike_done_8_TimeVersion.csv', index = False)

# 星期幾
# 0~6，0週一6週日
data_8['weekday'] = data_8['mday'].dt.dayofweek

# 時
data_8 = pd.read_csv('缺車率計算.csv')
# data_8_h = data_8.copy()
x = []
for i in range(len(data_8)):
    x.append(data_8['sna'][i]+str(data_8['mday'][i]).split(':')[0])
# data_8_h['hour'] = x
data_8['hour'] = x
# data_8_h.drop_duplicates(subset=['hour'], keep='first', inplace = True)
# 分
# data_8_m = data_8.copy()
x = []
for i in range(len(data_8)):
    x.append(data_8['sna'][i]+str(data_8['mday'][i]).split(':')[0]+str(data_8['mday'][i]).split(':')[1])
# data_8_m['min'] = x
data_8['min'] = x
# data_8_m.drop_duplicates(subset=['min'], keep='first', inplace = True)

data_8.to_csv('bike_WeekdayHourMin_v2.csv', index = False)

# join
# data_8_h = data_8_h[['hour']]
# data_8_m = data_8_m[['min']]
# data_8_all = data_8.join(data_8_h)
# data_8_all = data_8_all.join(data_8_m)

# data_8_all.to_csv('bike_WeekdayHourMin.csv', index = False)