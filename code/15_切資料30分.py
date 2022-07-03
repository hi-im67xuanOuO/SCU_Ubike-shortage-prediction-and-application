# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 18:47:22 2020

@author: s9631
"""

import os
os.chdir('D:/project/專題/資料庫備份/合併天氣資料_min')
import pandas as pd
from tqdm import tqdm

df = pd.read_csv('sample_final_v2.csv')
m = []
for i in df['mday']:
  mm = i.split(':')
  m.append(mm[0]+':'+mm[1])
df['m'] = m
df.head()

df['m'] = pd.to_datetime(df['m'], format="%Y-%m-%d %H:%M")

group = df.groupby('sno')

def count_diff(df):
    diff = []
    diff_time = []
    for i in range(len(df)):
      if i == 0:
        diff.append(1)
        diff_time.append(0)
        x = i
      else:
        a = df['m'][i]-df['m'][x]
        if a.seconds//60 >= 30:
          diff.append(1)
          diff_time.append(a.seconds//60)
          x = i
        else:
          diff.append(0)
          diff_time.append(a.seconds//60)
    return diff,diff_time
    
sno_list = list(group.size().index)

for s in tqdm(sno_list):
    data = group.get_group(s)
    data.reset_index(drop=True, inplace=True)
    diff, diff_time = count_diff(data)
    data['diff'], data['diff_time'] = diff, diff_time
    if s == 1:
        final = data.copy()
    else:
        final = pd.concat([final, data])
    
f = final[final['diff'] == 1]
f.reset_index(drop=True, inplace=True)
f.to_csv('間隔30分鐘.csv', index=False)
