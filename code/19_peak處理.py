# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 00:51:09 2020

@author: s9631
"""

import pandas as pd
import os
os.chdir('D:/project/專題/資料庫備份/合併天氣資料_min')
from tqdm import tqdm
data = pd.read_csv('sample_final_v2.csv')

group = data.groupby('sno')
sno_list = list(group.size().index)

q1 = []
q3 = []
for s in tqdm(sno_list):
    g = group.get_group(s)
    g_detail = g.describe()
    Min = g_detail['sbi_tot_mean']['min']
    Max = g_detail['sbi_tot_mean']['max']
    Q1 = g_detail['sbi_tot_mean']['25%']
    Q3 = g_detail['sbi_tot_mean']['75%']
    q1.append(Q1)
    q3.append(Q3)

d = pd.DataFrame({'sno':sno_list, 'Q1':q1, 'Q3':q3, 'min':Min, 'max':Max})    

d.to_csv('Q1Q3.csv', index=False)

df = pd.read_csv('outlier移除後.csv')
for i in range(len(d)):
    df['peak'][(df['sbi_tot_mean']<d['Q1'][i]) & (df['sno']==d['sno'][i])] = -1
    df['peak'][(df['sbi_tot_mean']>d['Q3'][i]) & (df['sno']==d['sno'][i])] = 1
    df['peak'][(df['sbi_tot_mean']<d['Q1'][i]) & (df['sbi_tot_mean']>d['Q3'][i]) & (df['sno']==d['sno'][i])] = 0
    
print(df['peak'].value_counts())
df.to_csv('peak更新.csv', index=False)


p = pd.read_csv('D:/project/專題/資料庫備份/分群的/分三類後的data.csv', encoding='big5')
for i in range(len(d)):
    p['peak'][(p['sbi_tot_mean']<d['Q1'][i]) & (p['sno']==d['sno'][i])] = -1
    p['peak'][(p['sbi_tot_mean']>0.9) & (p['sno']==d['sno'][i])] = 1
    p['peak'][(p['sbi_tot_mean']<d['Q1'][i]) & (p['sbi_tot_mean']>0.9) & (p['sno']==d['sno'][i])] = 0
    
p.to_csv('D:/project/專題/資料庫備份/分群的/peak更新_new.csv')
p['peak'].value_counts()
