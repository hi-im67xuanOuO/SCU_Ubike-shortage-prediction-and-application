# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 10:57:49 2020

@author: s9631
"""


import pandas as pd
import os
from datetime import datetime
os.chdir('D:/project/專題/資料庫備份/合併天氣資料_min')

data = pd.read_csv('合併天氣資料_min.csv', encoding='big5')

date = [i.split(' ')[0] for i in data['mday']]
data['date'] = date

date = data.groupby('date')

group = data.groupby('sno')
sno_list = list(group.size().index)
d_list=[]
for s in sno_list:
    d = group.get_group(s)
    d_list.append(d.reset_index(drop=True)['date'][0])
x = pd.DataFrame({'sno':sno_list,'date':d_list})
d = x.groupby('date')
d_list = list(d.size().index)

sno = {}
for i in d_list:
    sno[i] = list(d.get_group(i)['sno'].values)


time = {}
for i in d_list:
    d1 = date.get_group(i)
    sno_group = d1.groupby('sno')
    for j in sno[i]:
        s = sno_group.get_group(j)
        s = s[s['sbi_I/D'] == 0]
        time[j]=list(s['mday'].values)
        

# def StoDT(string):
#     return datetime.strptime(string, '%Y-%m-%d %H:%M:%S')

import numpy as np

def meanANDmode(sno_list, time):
    time_diff = {}
    for s in sno_list:
        next_t = [np.nan]+time[s]
        next_t = next_t[:len(time[s])]
        y = pd.DataFrame({'t':time[s], 't+1':next_t})
        y['t'] = pd.to_datetime(y['t'], format="%Y-%m-%d %H:%M:%S")
        y['t+1'] = pd.to_datetime(y['t+1'], format="%Y-%m-%d %H:%M:%S")
        y['diff'] = y['t']-y['t+1']
        y['diff'] = y['diff']/np.timedelta64(1, 's')
        time_diff[s]=list(y['diff'].values)
    
    time_mean=[]
    time_mode=[]
    time_mode2=[]
    error=[]
    for s in sno_list:
        # print(s)
        try:
            mean = round(sum(time_diff[s][1:])/(len(time_diff[s])-1))
            next_mode = time_diff[s][1:]
            while 60.0 in next_mode:
                next_mode.remove(60.0)
            mode = max(set(time_diff[s][1:]), key=time_diff[s][1:].count)
            mode2 = max(set(next_mode), key=next_mode.count)
            time_mean.append(mean)
            time_mode.append(mode)
            time_mode2.append(mode2)
        except:
            time_mean.append(np.nan)
            time_mode.append(np.nan)
            time_mode2.append(np.nan)
            error.append(s)
    return time_mean,time_mode,time_mode2,error

time_mean,time_mode,time_mode2,error = meanANDmode(sno_list, time)
final = pd.DataFrame({'sno': sno_list,
                      'mean_time': time_mean,
                      'mode_time': time_mode,
                      'mode_time2': time_mode2})
# final.to_csv('變動率.csv', index=False)

# 補資料
time1 = {}
d1 = date.get_group('2020-09-30')
sno_group = d1.groupby('sno')
for j in error:
    s = sno_group.get_group(j)
    s = s[s['sbi_I/D'] == 0]
    time1[j]=list(s['mday'].values)
    
time_mean1,time_mode1,time_mode1_2,error1 = meanANDmode(error, time1)
i=0
for j in error:
    y = final[final['sno'] == j].index[0]
    final['mean_time'][y] = time_mean1[i]
    final['mode_time'][y] = time_mode1[i]
    final['mode_time2'][y] = time_mode1_2[i]
    i+=1

final.to_csv('變動率完整版_v2.csv', index=False)
