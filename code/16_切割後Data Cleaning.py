# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 17:34:22 2020

@author: s9631
"""

import os
os.chdir('D:/project/專題/資料庫備份/合併天氣資料_min')
import pandas as pd
from tqdm import tqdm

df = pd.read_csv('間隔30分鐘.csv')
m = [i.split(' ')[1].split(':')[0] for i in df['m']]
df['hour'] = m
# df['weekday_s'] = df['weekday'].astype('str')
# df['week_h'] = df['weekday_s']+'-'+df['h']

df.to_csv('盒鬚.csv', index=False)

import seaborn as sns
import matplotlib.pyplot as plt

def boxplot(group, date):
    w = group.get_group(date)
    day=INTtoDAY(date)
    plt.figure(figsize=(20,5))
    plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1]) 
    sns.set(style='whitegrid')
    sns.boxplot(x='hour', y='sbi_tot_mean', data=w, width=0.5, palette='Set3')
    plt.title(day)
    plt.tight_layout()
    plt.savefig(day+'_boxplot2.png', dpi=300)
    plt.show()

def INTtoDAY(date):
    if date==0:
        return "Monday"
    if date==1:
        return "Tuesday"
    if date==2:
        return "Wednesday"
    if date==3:
        return "Thursday"
    if date==4:
        return "Friday"
    if date==5:
        return "Saturday"
    if date==6:
        return "Sunday"

os.chdir('boxplot')
group = df.groupby('weekday')
for d in range(7):
    boxplot(group, d)
    
outlier = {0:['07','12','13','14','17','20','21'],
           1:['07','12','13','16','17','18','20','21'],
           2:['07','11','12','13','16','17','18','19','20','21'],
           3:['07','11','12','13','16','17','18','20','21'],
           4:['07','11','12','13','14','15','16','17','18','19','20','21'],
           5:['08','09','10','11','15','16','17','18','19','20','21','22'],
           6:['08','09','10','11','12','13','14','15','16','17','18','19','20','21','22']}

def outlier_value(group,d,outlier):
    w = group.get_group(d)
    h = w.groupby('hour')
    top_list = {}
    for o in outlier[d]:
        out = h.get_group(o)
        q = out['sbi_tot_mean'].describe()
        IQR = q['75%']-q['25%']
        top = q['75%']+1.5*IQR
        top_list[o]=top
    return top_list

top_all = {d:outlier_value(group,d,outlier) for d in range(7)}
top_all[0]

for w in tqdm(range(7)):
    for t in top_all[w]:
        df['sbi_tot_mean'][(df['sbi_tot_mean']>top_all[w][t]) & (df['weekday']==w) & (df['hour']==t)] = 10
        # df_mean = df['sbi_tot_mean'][(df['sbi_tot_mean']!='outlier') &  (df['weekday']==w) & (df['hour']==t)].mean()
        # df['sbi_tot_mean'][(df['sbi_tot_mean']=='outlier') &  (df['weekday']==w) & (df['hour']==t)] = df_mean
        
# df.to_csv('outlier移除後.csv', index=False)
