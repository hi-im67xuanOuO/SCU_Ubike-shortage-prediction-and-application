# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 11:02:42 2020

@author: s9631
"""

import pandas as pd
import os
os.chdir('D:/project/專題/資料庫備份/合併天氣資料_min')
df = pd.read_csv('sample_final_v2.csv')

m = []
for i in df['mday']:
    s = i.split('-')
    m.append(s[0]+'-'+s[1])

df['m'] = m

group = df.groupby('m')
m8 = group.get_group('2020-08')
m8.reset_index(drop=True, inplace=True)
m8.to_csv('8月.csv', index=False)

group = m8.groupby('sno')
sno = list(group.size().index)

for s in sno:
    s8 = group.get_group(s)
    s8.reset_index(drop=True, inplace=True)
    l = []
    for i in range(len(s8)):
        if i % 2 == 1:
            l.append(i)
    s8 = s8.drop(l)
    # print(s8)
    if s != 1:
        s_all = pd.concat([s_all,s8])
    else:
        s_all = s8.copy()
        
s_all.to_csv('8月刪減.csv', index=False)
