# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 19:33:50 2020

@author: s9631
"""


import pandas as pd
import os
os.chdir("D:/project/專題/資料庫備份")

d_8 = pd.read_csv('缺車率計算_hour_9/缺車率計算_hour_8.csv')
d_9 = pd.read_csv('缺車率計算_hour_9/缺車率計算_hour_9.csv')

d = pd.concat([d_8, d_9])
d = d[['sno','sna','tot','sbi','sarea','mday','bemp','weekday','sbi_tot','sbi_mean','sbi_tot_mean']]
hour = [i.split(' ')[1].split(':')[0] for i in d['mday']]
d['hour'] = hour

group = d.groupby('sno')
sno_list = list(group.size().index)

mydict={}
for s in sno_list:
    g = group.get_group(s)
    g = g.groupby('hour')
    hour_list = list(g.size().index)
    hour = []
    for h in hour_list:
        x = g.get_group(h)['sbi_tot_mean'].mean()
        hour.append(x)
    mydict[str(s)] = hour
    
df = pd.DataFrame(mydict)

mydict={}
for s in sno_list:
    Q1 = df[str(s)].quantile([0.25])[0.25]
    Q3 = df[str(s)].quantile([0.75])[0.75]
    q = []
    for i in df[str(s)]:
        if i < Q1:
            q.append(-1)
        elif i > Q3:
            q.append(1)
        else:
            q.append(0)
    mydict[str(s)] = q
        
df2 = pd.DataFrame(mydict)

df.to_csv('尖峰離峰hour_sbitotmean.csv', index=False)
df2.to_csv('尖峰離峰hour_01.csv', index=False)

corr_pd = df2.corr()
corr_pd.to_csv('相關性矩陣_v3.csv', index=False)