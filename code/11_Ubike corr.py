# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 23:13:14 2020

@author: s9631
"""


import pandas as pd
import os
os.chdir("D:/project/專題/資料庫備份")

data = pd.read_csv('尖峰離峰hour_01.csv')

hour = [i.split(' ')[1].split(':')[0] for i in data['mday']]
data['hour'] = hour
group = data.groupby('sno')
sno_list = list(group.size().index)
hour = group.get_group(58)
hour = hour.groupby('hour')
hour_list = list(hour.size().index)

# mydict = {}
# for i in sno_list:
#     hour = group.get_group(i)
#     hour = hour.groupby('hour')
#     hour_list = list(hour.size().index)
#     mylist = []
#     for j in hour_list:
#         h = hour.get_group(j)['peak'].value_counts().index[0]
#         mylist.append(h)
#     mydict[str(i)] = mylist
    # print(i)
    
# peak = pd.DataFrame(mydict)
corr_pd = data.corr()
corr_pd.to_csv('相關性矩陣_v2.csv', index=False)
# import seaborn as sns
# import matplotlib.pyplot as plt
# # 繪製熱力圖
# sns.heatmap(corr_pd, cmap='YlGnBu')
# plt.savefig('scatterplot1006.png', dpi=300)