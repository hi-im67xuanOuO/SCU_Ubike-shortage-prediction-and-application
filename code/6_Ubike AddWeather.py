# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 00:06:05 2020

@author: s9631
"""


import pandas as pd
import os
os.chdir('D:/project/專題/資料庫備份')

data = pd.read_csv('8月9月標註sbi增減.csv')
weather = pd.read_csv('天氣測站處理.csv')
date_hour = [i.split(':')[0] for i in data['mday']]
weather_hour = [i.split(':')[0] for i in weather['new_Date']]
data['Hour'] = date_hour
weather['Hour'] = weather_hour
del date_hour,weather_hour
type1=[]
for i in data['sarea']:
    if i == '北投區' or i == '士林區':
        type1.append('s')
    if i == '中正區' or i == '萬華區' or i == '大同區' or i == '中山區' or i == '大安區':
        type1.append('t')
    if i == '信義區' or i == '松山區' or i == '文山區':
        type1.append('y')
    if i == '南港區' or i == '內湖區':
        type1.append('n')
data['type'] = type1
del type1
d = pd.merge(data, weather, on=['type', 'Hour'])
d = d[['sno','sna','sbi','sarea','mday','lat','lng','bemp','weekday','sbi_I/D','Precp','RH','Station_no','Temperature']]
d.to_csv('8月9月sbi增減合併天氣.csv', index=False)
