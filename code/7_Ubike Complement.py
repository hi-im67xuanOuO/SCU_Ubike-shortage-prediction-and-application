# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 00:02:10 2020

@author: s9631
"""


import pandas as pd
import os
os.chdir("D:/project/專題/資料庫備份")

data = pd.read_csv('9月原始資料/bike_9_all.csv')

sat = data[data['weekday']==5]
sat.reset_index(drop=True, inplace=True)

date = [i.split(' ')[0] for i in sat['mday']]
hour = [i.split(' ')[1].split(':')[0] for i in sat['mday']]
sat['hour_1'] = hour
sat['date_1'] = date

sat = sat[sat['hour_1'] == '09']
sat_error = sat[sat['date_1']=='2020-09-12']
sat = sat[sat['date_1'] != '2020-09-12']
sat.reset_index(drop=True, inplace=True)
sat_error.reset_index(drop=True, inplace=True)

minute = [i.split(' ')[1].split(':')[1] for i in sat['mday']]
sat['minute_1'] = minute

min_list = sorted([i for i in list(sat['minute_1'].value_counts().index)])
sno_list = [i for i in list(sat['sno'].value_counts().index)]
# 按車站補值    
sbi_mean = [round(sat[sat['sno']==i]['sbi'].mean()) for i in sno_list]
bemp_mean = [round(sat[sat['sno']==i]['bemp'].mean()) for i in sno_list]

com_df = pd.DataFrame({'sno':sno_list,'sbi':sbi_mean,'bemp':bemp_mean})
final1 = []
for i in min_list:
    for j in range(391):
        final1.append(i)
final2 = sno_list*60
final_df = pd.DataFrame({'sno':final2,'minute_1':final1})
final_df = pd.merge(final_df, com_df, on='sno')
station =  pd.read_excel('ss.xlsx', encoding='big5')
final_df = pd.merge(final_df, station, on='sno')
final_df['mday'] = '2020-09-12 09:'
final_df['mday'] = final_df['mday']+final_df['minute_1']+':00'
final_df['hour'] = final_df['sna']+'2020-09-12 09'
final_df['min'] = final_df['hour']+final_df['minute_1']
final_df['act'] = 1
final_df['weekday'] = 5
sat_error = sat_error[['sno','sna','tot','sbi','sarea','mday','lat','lng','ar','bemp','act','hour','min', 'weekday', 'hour_1']]

df = pd.concat([sat_error, final_df])
df.drop_duplicates(keep='first', subset=['min'], inplace=True)

final = pd.concat([data, df])
final.sort_values(by=['mday','sno'], inplace = True)
final = final[['sno','sna','tot','sbi','sarea','mday','lat','lng','ar','bemp','act','hour','min', 'weekday', 'hour_1']]
final.reset_index(drop=True, inplace=True)

final.to_csv('9月原始資料.csv', index=False)
