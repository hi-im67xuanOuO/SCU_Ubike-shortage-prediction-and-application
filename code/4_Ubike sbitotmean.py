# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 18:50:18 2020

@author: s9631
"""


import pandas as pd
import os
os.chdir("D:/project/專題/資料庫備份")

data = pd.read_csv('9月原始資料.csv')
data['sbi_tot'] = data['sbi']/data['tot']

data_new = data.drop_duplicates(subset = 'min')
data_new.set_index(['min'], inplace=True)
min_mean = data.groupby('min').mean()
sbi_mean = min_mean['sbi']
sbi_mean = pd.DataFrame(sbi_mean)
sbi_mean.columns = ['sbi_mean']
df_minnew = pd.concat([data_new, sbi_mean],axis =1)
sbi_tot_mean = min_mean['sbi_tot']
sbi_tot_mean = pd.DataFrame(sbi_tot_mean)
sbi_tot_mean.columns = ['sbi_tot_mean']
df_minnew = pd.concat([df_minnew, sbi_tot_mean],axis =1)
df_minnew.to_csv("缺車率計算_min_9.csv")

data_new = data.drop_duplicates(subset = 'hour')
data_new.set_index(['hour'], inplace=True)
min_mean = data.groupby('hour').mean()
sbi_mean = min_mean['sbi']
sbi_mean = pd.DataFrame(sbi_mean)
sbi_mean.columns = ['sbi_mean']
df_minnew = pd.concat([data_new, sbi_mean],axis =1)
sbi_tot_mean = min_mean['sbi_tot']
sbi_tot_mean = pd.DataFrame(sbi_tot_mean)
sbi_tot_mean.columns = ['sbi_tot_mean']
df_minnew = pd.concat([df_minnew, sbi_tot_mean],axis =1)
df_minnew.to_csv("缺車率計算_hour_9.csv")